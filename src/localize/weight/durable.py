# src/localize/weight/durable.py
from torch.utils.data import DataLoader
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def _streaming_global_topk_mlp_grads(model, ratio: float):
    """
    Return a set of global indices (CPU LongTensor) for the top-K gradient magnitudes
    across *all* MLP parameters, without materializing the full concatenated vector.
    """
    # 1) compute total length across selected params
    total = 0
    grads_meta = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and ("mlp" in name):
            n = p.grad.numel()
            grads_meta.append((name, p, total, n))
            total += n

    if total == 0:
        return torch.empty(0, dtype=torch.long)

    K = max(1, int(total * ratio))

    # 2) streaming maintain global topK (values on CPU, indices on CPU)
    best_vals = None  # CPU float32 [<=K]
    best_idx  = None  # CPU int64  [<=K]

    # heuristic: per-tensor local topk size
    # keep it at least K/num_tensors, but also cap to something reasonable
    num_tensors = max(1, len(grads_meta))
    k_local = max(1, min(K, max(K // num_tensors, 1024)))

    for _, p, offset, n in grads_meta:
        g = p.grad.detach()
        # use fp32 magnitude for stable ranking
        flat = g.float().abs().view(-1)

        # local topk on device (GPU) then move only small results to CPU
        kk = min(k_local, flat.numel())
        vals, idx = torch.topk(flat, kk, largest=True, sorted=False)   # device
        vals = vals.to("cpu")
        idx = (idx.to("cpu") + offset).long()

        if best_vals is None:
            best_vals, best_idx = vals, idx
        else:
            best_vals = torch.cat([best_vals, vals], dim=0)
            best_idx  = torch.cat([best_idx,  idx], dim=0)

        # prune back to K on CPU
        if best_vals.numel() > K:
            v, j = torch.topk(best_vals, K, largest=True, sorted=False)
            best_idx = best_idx[j]
            best_vals = v

    # final prune exactly K
    if best_vals.numel() > K:
        v, j = torch.topk(best_vals, K, largest=True, sorted=False)
        best_idx = best_idx[j]

    return best_idx  # CPU LongTensor of global indices in [0, total)


def get_grad_mask_list(model, noise_data, ratio=0.05, aggregate_all_layer=0, batch_size=1, use_amp=True):
    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if isinstance(noise_data, torch.Tensor) and noise_data.is_cuda:
        noise_data = noise_data.detach().cpu()

    loader = DataLoader(noise_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.zero_grad(set_to_none=True)
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)

        if use_amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast(True):
                out = model(batch, labels=batch)
                loss = out.loss
        else:
            out = model(batch, labels=batch)
            loss = out.loss
        loss.backward()

    mask_grad_list = []

    if aggregate_all_layer == 1:
        # ===== NEW SAFE GLOBAL TOPK =====
        topk_global = _streaming_global_topk_mlp_grads(model, ratio=ratio)
        topk_global = topk_global.to("cpu")
        topk_set = set(topk_global.tolist())  # size K, manageable when ratio small

        # build masks by streaming ranges; DO NOT allocate huge intermediate arrays
        count = 0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and ("mlp" in name):
                n = p.grad.numel()
                # mask: 1 keep, 0 zero-out selected
                mask = torch.ones(n, dtype=p.dtype, device=p.device)

                # find which global indices fall into [count, count+n)
                # iterate topk_set (size K) is ok when K is small
                # (ratio=1e-4 on mlp-only typically still manageable)
                local = [i - count for i in topk_set if count <= i < count + n]
                if len(local) > 0:
                    mask_idx = torch.tensor(local, device=p.device, dtype=torch.long)
                    mask[mask_idx] = 0.0

                mask_grad_list.append(mask.view_as(p.grad))
                count += n
    else:
        # layer-wise (your existing logic, but avoid .cuda() hardcode)
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and ("mlp" in name):
                g = p.grad.detach().float().abs().view(-1)
                L = g.numel()
                k = max(1, int(L * ratio))
                _, idx = torch.topk(g, k)
                mask = torch.ones(L, dtype=p.dtype, device=p.device)
                mask[idx.to(p.device)] = 0.0
                mask_grad_list.append(mask.view_as(p.grad))

    model.zero_grad(set_to_none=True)
    return mask_grad_list


def apply_grad_mask_to_params(model, mask_grad_list):
    """
    Multiply parameters by mask (0 out selected weights).
    Note: this is in-place-ish via state_dict replace.
    """
    mask_iter = iter(mask_grad_list)
    sd = model.state_dict()

    for name, parms in model.named_parameters():
        if parms.requires_grad and ("mlp" in name):
            m = next(mask_iter).to(parms.device)
            sd[name] = sd[name] * m

    model.load_state_dict(sd)
    return model


def do_durable(model, noise_data, ratio, aggregate_all_layer):
    # optimizer is unused here; you don't step.
    # keep interface unchanged.
    grad_mask_list = get_grad_mask_list(
        model,
        noise_data,
        ratio=ratio,
        aggregate_all_layer=int(aggregate_all_layer),
        batch_size=1,     # ⭐ 你可以改成 1/2/8
        use_amp=True,     # ⭐ 显存紧就 True
    )
    model = apply_grad_mask_to_params(model, grad_mask_list)
    return model
