"""
LUNAR (Activation Redirection) Unlearning — FINAL BUGFIXED VERSION

- Correct non-constant objective (base-model target fitting)
- Correct hook semantics (block output)
- Fix hook-handle shadowing bug
- Only updates W_down (mlp.c_proj.weight)
- Interface identical to previous do_lunar usage

Author: patched for memorization experiments
Date: 2026-01-05
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from typing import Optional, Dict
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utilities
# ============================================================

def _get_mlp(model: nn.Module, layer_idx: int):
    return model.transformer.h[layer_idx].mlp


def _block_output_hook():
    """
    Hook that captures block output hidden states [B,S,D]
    """
    storage = {"act": None}

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            out = out[0]
        storage["act"] = out

    return hook, storage


# ============================================================
# Step 1: collect mean activations
# ============================================================

@torch.no_grad()
def collect_mean_activations(
    model: nn.Module,
    dataloader: DataLoader,
    layer_idx: int,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    model.to(device)

    block = model.transformer.h[layer_idx]
    hook_fn, store = _block_output_hook()
    handle = block.register_forward_hook(hook_fn)

    all_means = []

    try:
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            x = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
            store["act"] = None
            _ = model(x)

            h = store["act"]
            if h is None:
                continue

            # [B,S,D] -> [B,D]
            h_mean = h.mean(dim=1)
            all_means.append(h_mean.cpu())
    finally:
        handle.remove()

    if len(all_means) == 0:
        raise RuntimeError("No activations collected.")

    all_means = torch.cat(all_means, dim=0)
    return all_means.mean(dim=0)  # [D]


def compute_unlearning_vector(a_f: torch.Tensor, a_ref: torch.Tensor) -> torch.Tensor:
    """
    r_uv = a_ref - a_f
    """
    return (a_ref - a_f).detach()


# ============================================================
# Step 2: optimize W_down
# ============================================================

def optimize_downproj(
    model: nn.Module,
    unlearn_loader: DataLoader,
    r_uv: torch.Tensor,
    layer_idx: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
):
    model = model.to(device)
    model.train()

    # ---------- frozen base model ----------
    base_model = copy.deepcopy(model).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # ---------- hooks ----------
    block_cur = model.transformer.h[layer_idx]
    block_base = base_model.transformer.h[layer_idx]

    hook_cur, store_cur = _block_output_hook()
    hook_base, store_base = _block_output_hook()

    handle_cur = block_cur.register_forward_hook(hook_cur)
    handle_base = block_base.register_forward_hook(hook_base)

    # ---------- optimize only W_down ----------
    mlp = _get_mlp(model, layer_idx)
    W = mlp.c_proj.weight  # W_down

    for p in model.parameters():
        p.requires_grad = False
    W.requires_grad = True

    optimizer = optim.AdamW([W], lr=lr, weight_decay=weight_decay)

    r_uv = r_uv.to(device)

    stats = {"loss": [], "delta_w": []}

    try:
        for ep in range(num_epochs):
            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(unlearn_loader, leave=False, desc=f"LUNAR {ep+1}/{num_epochs}")
            for batch in pbar:
                x = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)

                optimizer.zero_grad(set_to_none=True)

                # ---- base forward (no grad) ----
                store_base["act"] = None
                with torch.no_grad():
                    _ = base_model(x)
                    act_base = store_base["act"]
                    if act_base is None:
                        continue
                    act_base = act_base.mean(dim=1)  # [B,D]

                # ---- current forward (with grad) ----
                store_cur["act"] = None
                _ = model(x)
                act_cur = store_cur["act"]
                if act_cur is None:
                    continue
                act_cur = act_cur.mean(dim=1)  # [B,D]

                target = (act_base + r_uv).detach()
                loss = ((act_cur - target) ** 2).mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            stats["loss"].append(avg_loss)

            with torch.no_grad():
                W0 = base_model.transformer.h[layer_idx].mlp.c_proj.weight
                delta = (W - W0).norm().item()
                stats["delta_w"].append(delta)

            if (ep + 1) % max(1, num_epochs // 5) == 0:
                print(f"  epoch {ep+1}: loss={avg_loss:.3e}, ||ΔW||={delta:.3e}")

    finally:
        handle_cur.remove()
        handle_base.remove()

    model.eval()
    return model, stats


# ============================================================
# Main entry (interface unchanged)
# ============================================================

def do_lunar(
    model: nn.Module,
    unlearn_set: torch.Tensor,
    reference_set: torch.Tensor,
    layer_idx: Optional[int] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 0.0,
    max_batches_collection: Optional[int] = None,
):
    model = model.to(device)

    if layer_idx is None:
        layer_idx = len(model.transformer.h) // 2
        print(f"[LUNAR] Using middle layer {layer_idx}")

    print(f"[LUNAR] Unlearn set: {tuple(unlearn_set.shape)}")
    print(f"[LUNAR] Reference set: {tuple(reference_set.shape)}")

    forget_loader = DataLoader(unlearn_set, batch_size=batch_size, shuffle=False)
    ref_loader = DataLoader(reference_set, batch_size=batch_size, shuffle=False)

    # ---- collect means ----
    print("[LUNAR] Collecting mean activations...")
    a_f = collect_mean_activations(model, forget_loader, layer_idx, max_batches_collection)
    a_ref = collect_mean_activations(model, ref_loader, layer_idx, max_batches_collection)

    r_uv = compute_unlearning_vector(a_f, a_ref)
    print(f"[LUNAR] ||r_uv|| = {r_uv.norm().item():.4f}")

    # ---- sanity: save pre-edit W ----
    W_before = model.transformer.h[layer_idx].mlp.c_proj.weight.detach().cpu().clone()

    # ---- optimize ----
    model, stats = optimize_downproj(
        model,
        forget_loader,
        r_uv,
        layer_idx,
        num_epochs,
        lr,
        weight_decay,
    )

    # ---- final sanity ----
    W_after = model.transformer.h[layer_idx].mlp.c_proj.weight.detach().cpu().clone()
    print("[LUNAR] FINAL ||W_after - W_before|| =",
          (W_after - W_before).norm().item())
    print("[LUNAR] FINAL max|ΔW| =",
          (W_after - W_before).abs().max().item())

    return model
