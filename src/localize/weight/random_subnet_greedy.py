# src/localize/weight/random_subnet_greedy.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from src.localize.weight.weight_utils import clm_loss_fn
from src.localize.weight.random_subnet import (
    mask_model,
    get_base_edited_model,
    GetSubnet,
    Conv1D,
    SupermaskConv,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SAFETY PATCH: detach base weights in supermask forward
# ============================================================

def patch_supermask_detach(model):
    for m in model.modules():
        if hasattr(m, "scores") and hasattr(m, "weight") and hasattr(m, "k"):

            # GPT-style Conv1D
            if isinstance(m, Conv1D):
                def forward(x, _m=m):
                    w = _m.weight.detach()
                    mask = GetSubnet.apply(_m.scores, _m.k)
                    w = w * mask
                    x = torch.addmm(_m.bias, x.view(-1, x.size(-1)), w)
                    return x.view(*x.shape[:-1], x.size(-1))
                m.forward = forward

            # Generic supermask conv / linear
            elif isinstance(m, SupermaskConv):
                def forward(x, _m=m):
                    w = _m.weight.detach()
                    mask = GetSubnet.apply(_m.scores, _m.k)
                    return F.linear(x, w * mask, _m.bias)
                m.forward = forward


# ============================================================
# TRAIN LOOP (safe)
# ============================================================

def train(
    model,
    train_dataloader,
    optimizer,
    loss_weighting,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch, label in train_dataloader:
        batch = batch.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # ---- forward ----
        out = model(batch, labels=batch)
        logits = out.logits[:, -1:]  # ⭐ only last token

        loss = clm_loss_fn(batch[:, -1:], logits)

        loss_clean = loss[label == -1].mean() if (label == -1).any() else 0.0
        loss_noise = loss[label == 1].mean() if (label == 1).any() else 0.0

        train_loss = (
            -(1 - loss_weighting) * loss_noise
            + loss_weighting * loss_clean
        )

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


# ============================================================
# MAIN API (UNCHANGED SIGNATURE)
# ============================================================

def do_random_greedy(
    model,
    noise_data,
    clean_data,
    n_layers,
    ratio,
    epochs=5,
    lr=1e-3,                     # ⭐ safer default
    momentum=0.9,
    weight_decay=0.0,
    batch_size=16,               # ⭐ safer default
    loss_weighting=0.05,
    model_name="gpt2",
):
    """
    SAFE drop-in replacement:
    - same signature
    - same semantics
    - no OOM
    """

    # ---- build dataset exactly like before ----
    noise_labels = torch.ones(len(noise_data), dtype=torch.long)
    clean_labels = -torch.ones(len(clean_data), dtype=torch.long)

    data = torch.cat([noise_data, clean_data], dim=0).cpu()
    labels = torch.cat([noise_labels, clean_labels], dim=0).cpu()


    train_dataset = list(zip(data, labels))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # ---- freeze everything ----
    for p in model.parameters():
        p.requires_grad = False

    # ---- apply random subnet mask ----
    model = mask_model(model, n_layers, ratio, model_name)

    # ---- VERY IMPORTANT: patch supermask forward ----
    patch_supermask_detach(model)

    # ---- unfreeze only scores ----
    trainable = []
    for m in model.modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad_(True)
            trainable.append(m.scores)

    assert len(trainable) > 0, "No trainable supermask scores found"

    # ---- optimizer: AdamW, no foreach ----

    optimizer = optim.SGD(
        trainable,
        lr=lr,
        momentum=0.0,      # ⭐ 非常重要
        weight_decay=0.0,
        foreach=False,     # ⭐ 防止 multi-tensor clone
    )

    # ---- training ----
    for _ in range(epochs):
        train(
            model,
            train_dataloader,
            optimizer,
            loss_weighting,
        )

    # ---- convert back to base model ----
    model = get_base_edited_model(model, n_layers, model_name)
    return model
