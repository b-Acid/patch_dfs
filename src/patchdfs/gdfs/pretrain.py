"""Port of the authors' MaskingPretrainer.fit / BaseModel.fit
(third_party/dynamic-selection/dynamic_selection/models.py).

Adam, ReduceLROnPlateau on the validation loss (factor 0.2), early stopping once the validation
loss has not improved for more than patience + 1 epochs, best weights restored at the end.
forward_fn(x, m) maps raw inputs and a (B, mask_size) mask to class logits; module is the
nn.Module switched between train/eval. Masks are fresh uniform-random masks every batch (the
authors' generate_uniform_mask); pass a different mask_sampler for static masks or full inputs.
"""
from __future__ import annotations

import copy
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from patchdfs.common import generate_uniform_mask
from patchdfs.common.io import RunLogger


def pretrain_predictor(
    module: nn.Module,
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask_size: int,
    train_loader,
    val_loader,
    *,
    lr: float = 1e-3,
    nepochs: int = 100,
    patience: int = 2,
    min_lr: float = 1e-6,
    factor: float = 0.2,
    loss_fn: nn.Module | None = None,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    log: RunLogger | None = None,
    tag: str = "pretrain",
    mask_sampler: Callable[[int], torch.Tensor] | None = None,
    max_steps_per_epoch: int | None = None,
    restore_best: bool = True,
) -> list[dict]:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    opt = optim.Adam(module.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=factor, patience=patience, min_lr=min_lr)
    early_stop_after = patience + 1
    if mask_sampler is None:
        def mask_sampler(n: int) -> torch.Tensor:  # noqa: F811
            return generate_uniform_mask(n, mask_size, device=device)

    def masked(x: torch.Tensor) -> torch.Tensor:
        return forward_fn(x, mask_sampler(len(x)))

    best_sd, num_bad, history = None, 0, []
    for epoch in range(nepochs):
        t0 = time.time()
        module.train()
        tot = n = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
                loss = loss_fn(masked(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
            if max_steps_per_epoch and step + 1 >= max_steps_per_epoch:
                break
        module.eval()
        vtot = vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
                    loss = loss_fn(masked(x), y)
                vtot += loss.item() * len(x)
                vn += len(x)
        val_loss = vtot / max(vn, 1)
        sch.step(val_loss)
        if val_loss == sch.best:
            best_sd = copy.deepcopy(module.state_dict())
            num_bad = 0
        else:
            num_bad += 1
        rec = {
            "epoch": epoch + 1,
            "train_loss": tot / max(n, 1),
            "val_loss": val_loss,
            "lr": opt.param_groups[0]["lr"],
            "sec": round(time.time() - t0, 1),
        }
        history.append(rec)
        if log:
            log.log(tag, **rec)
        if num_bad > early_stop_after:
            break
    # The GDFS code restores the best epoch; the DIME Lightning pretrainer has no checkpoint
    # callback and therefore keeps the last epoch (restore_best=False).
    if restore_best and best_sd is not None:
        module.load_state_dict(best_sd)
    return history