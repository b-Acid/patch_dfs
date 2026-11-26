"""CIFAR-10 static baselines from the GDFS paper: random mask, centre crop, and the supervised
Concrete Autoencoder (CAE). Ports of third_party/dynamic-selection/experiments/cifar/run_random.py,
run_center.py, run_cae.py and baselines/{differentiable.py, utils.py}.
Each baseline trains a fresh predictor on the statically masked inputs for every budget.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import RelaxedOneHotCategorical

from patchdfs.common.io import RunLogger
from patchdfs.gdfs.pretrain import pretrain_predictor
from patchdfs.models import CifarClassifierHead, CifarResNetBackbone

CENTER_CROPS = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5), (5, 6), (6, 6), (6, 7), (7, 7)]


class StaticMaskLayer2d(nn.Module):
    """Zero all patches except those in a fixed (width x width) mask grid."""

    def __init__(self, mask: torch.Tensor, patch_size: int):
        super().__init__()
        mask = torch.as_tensor(mask, dtype=torch.float32).reshape(1, 1, mask.shape[0], mask.shape[1])
        self.upsample = nn.Upsample(scale_factor=patch_size) if patch_size > 1 else nn.Identity()
        self.register_buffer("mask", mask)
        self.mask_size = int(mask.shape[2] * mask.shape[3])
        self.width = int(mask.shape[2])

    def forward(self, x, m=None):
        return x * self.upsample(self.mask).to(x.dtype)

    def flat_mask(self, n: int, device) -> torch.Tensor:
        return self.mask.reshape(1, -1).expand(n, -1).to(device)


def center_mask(width_grid: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((width_grid, width_grid))
    x0, y0 = width_grid // 2 - w // 2, width_grid // 2 - h // 2
    mask[x0 : x0 + w, y0 : y0 + h] = 1
    return mask


def random_mask(width_grid: int, num: int, rng: np.random.RandomState) -> np.ndarray:
    inds = rng.choice(width_grid**2, size=num, replace=False)
    return np.isin(np.arange(width_grid**2), inds).reshape(width_grid, width_grid).astype(np.float32)


def mask_to_inds(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask.reshape(-1))


def inds_to_mask(inds, width_grid: int = 8) -> np.ndarray:
    return np.isin(np.arange(width_grid**2), np.asarray(inds)).reshape(width_grid, width_grid).astype(np.float32)


def build_static_predictor(num_classes: int = 10) -> nn.Sequential:
    return nn.Sequential(CifarResNetBackbone(), CifarClassifierHead(num_classes))


def train_static(mask_layer: StaticMaskLayer2d, train_loader, val_loader, *, lr=1e-3, nepochs=100, patience=2, device, amp_dtype=torch.bfloat16, log=None, tag="static"):
    mask_layer.to(device)
    model = build_static_predictor().to(device)
    hist = pretrain_predictor(
        model, lambda x, m: model(mask_layer(x)), mask_layer.mask_size, train_loader, val_loader, lr=lr, nepochs=nepochs, patience=patience,
        device=device, amp_dtype=amp_dtype, log=log, tag=tag,
        mask_sampler=lambda n: mask_layer.flat_mask(n, device),
    )
    return model, hist


@torch.no_grad()
def evaluate_model(model, loader, device, amp_dtype=torch.bfloat16) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            pred = model(x)
        correct += (pred.argmax(1) == y).sum().item()
        total += len(x)
    return correct / total


class ConcreteMask2d(nn.Module):
    """Concrete selector over width**2 patches choosing num_select of them (authors' code)."""

    def __init__(self, width: int, patch_size: int, num_select: int, gamma: float = 0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_select, width**2, dtype=torch.float32))
        self.upsample = nn.Upsample(scale_factor=patch_size)
        self.width = width
        self.patch_size = patch_size
        self.gamma = gamma

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        m = self.upsample(m.reshape(-1, 1, self.width, self.width).to(x.dtype))
        return x * m


def fit_cae_selector(
    model: nn.Module,
    selector_layer: ConcreteMask2d,
    train_loader,
    val_loader,
    *,
    lr=1e-3,
    nepochs=250,
    patience=5,
    min_lr=1e-5,
    start_temp=10.0,
    end_temp=0.01,
    temp_steps=10,
    device,
    amp_dtype=torch.bfloat16,
    log: RunLogger | None = None,
) -> dict:
    """Port of DifferentiableSelector.fit (val loss = CE with soft samples, zero-temp = hard)."""
    loss_fn = nn.CrossEntropyLoss()
    early_after = patience + 1
    best_val, best_sd = None, None
    history = []
    ac = lambda: torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None)  # noqa: E731
    for temp in np.geomspace(start_temp, end_temp, temp_steps):
        opt = optim.Adam(list(model.parameters()) + list(selector_layer.parameters()), lr=lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.2, patience=patience, min_lr=min_lr)
        best_soft, num_bad = None, 0
        for epoch in range(nepochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with ac():
                    loss = loss_fn(model(selector_layer(x, temp)), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            # Re-initialise logits whose argmax collides with an earlier row (authors' code).
            logits = selector_layer.logits
            argmax = logits.argmax(dim=1).cpu().numpy()
            selected: list[int] = []
            for i, ind in enumerate(argmax):
                if ind in selected:
                    logits.data[i] = 0
                else:
                    selected.append(ind)
            model.eval()
            soft_preds, hard_preds, ys = [], [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    with ac():
                        soft_preds.append(model(selector_layer(x, temp)).float().cpu())
                        hard_preds.append(model(selector_layer(x, 1e-6)).float().cpu())
                    ys.append(y)
            val_loss = loss_fn(torch.cat(soft_preds), torch.cat(ys)).item()
            val_hard = loss_fn(torch.cat(hard_preds), torch.cat(ys)).item()
            sch.step(val_loss)
            if val_loss == sch.best:
                best_soft = (copy.deepcopy(model.state_dict()), copy.deepcopy(selector_layer.state_dict()))
                num_bad = 0
            else:
                num_bad += 1
            if best_val is None or val_hard < best_val:
                best_val = val_hard
                best_sd = (copy.deepcopy(model.state_dict()), copy.deepcopy(selector_layer.state_dict()))
            history.append({"temp": float(temp), "epoch": epoch + 1, "val_loss": val_loss, "val_hard_loss": val_hard})
            if log:
                log.log("cae", temp=float(temp), epoch=epoch + 1, val_loss=val_loss, val_hard_loss=val_hard)
            if num_bad > early_after:
                break
        model.load_state_dict(best_soft[0])
        selector_layer.load_state_dict(best_soft[1])
    model.load_state_dict(best_sd[0])
    selector_layer.load_state_dict(best_sd[1])
    return {"best_val_hard_loss": best_val, "history": history}


def cae_selected_inds(selector_layer: ConcreteMask2d, num: int, width: int = 8) -> np.ndarray:
    logits = selector_layer.logits.detach().cpu().numpy()
    inds = np.sort(logits.argmax(axis=1))
    uniq = np.unique(inds)
    if len(uniq) != num:
        extras = np.setdiff1d(np.arange(width**2), inds)[: num - len(uniq)]
        inds = np.sort(np.concatenate([uniq, extras]))
    return inds