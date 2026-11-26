"""Port of the authors' GDFS training loops:
GreedyDynamicSelection.fit (experiments/mnist/run.py) and GreedyDynamicSelectionPTL
(experiments/cifar/run_greedy.py). Both train with a sequence of Concrete temperatures; within a
temperature, Adam + ReduceLROnPlateau(mode max) on the soft-sample validation accuracy, early
stopping on the same quantity; after each temperature continue from the best soft checkpoint; the
final model is the best zero-temperature (hard argmax) checkpoint over all temperatures.
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from patchdfs.common import ConcreteSelector, make_onehot
from patchdfs.common.io import RunLogger


def _autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def train_epoch(net, loader, opt, temp: float, max_features: int, loss_fn, device, amp: bool, max_steps=None):
    net.train()
    selector_layer = ConcreteSelector()
    tot = n = 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        m = torch.zeros(len(x), net.mask_size, device=device)
        opt.zero_grad(set_to_none=True)
        for _ in range(max_features):
            with _autocast(amp):
                logits = net.select_logits(x, m)
                soft = selector_layer(logits, temp)
                m_soft = torch.max(m, soft)
                pred = net.predict(x, m_soft)
                loss = loss_fn(pred, y)
            (loss / max_features).backward()
            with torch.no_grad():
                m = torch.max(m, make_onehot(logits - 1e6 * m))
        opt.step()
        tot += loss.item() * len(x)
        n += len(x)
        if max_steps and step + 1 >= max_steps:
            break
    return tot / max(n, 1)


@torch.no_grad()
def validate(net, loader, temp: float, max_features: int, device, amp: bool):
    """Accuracy over all steps of (a) soft-sample predictions and (b) hard argmax predictions."""
    net.eval()
    selector_layer = ConcreteSelector()
    soft_correct = hard_correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        m = torch.zeros(len(x), net.mask_size, device=device)
        for _ in range(max_features):
            with _autocast(amp):
                logits = net.select_logits(x, m) - 1e6 * m
                soft = selector_layer(logits, temp)
                m_soft = torch.max(m, soft)
                m = torch.max(m, make_onehot(soft))
                pred = net.predict(x, m_soft)
                hard_pred = net.predict(x, m)
            soft_correct += (pred.argmax(1) == y).sum().item()
            hard_correct += (hard_pred.argmax(1) == y).sum().item()
            total += len(x)
    return soft_correct / total, hard_correct / total


def fit_gdfs(
    net,
    train_loader,
    val_loader,
    *,
    max_features: int,
    temps,
    lr: float,
    nepochs: int,
    scheduler_patience: int,
    early_stop_patience: int,
    min_lr: float,
    device,
    loss_fn=None,
    amp: bool = False,
    log: RunLogger | None = None,
    max_steps_per_epoch: int | None = None,
    graphed: bool = False,
    ckpt_path: str | None = None,
) -> dict:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    best_global = {"val_hard": -1.0, "sd": None, "temp": None, "epoch": None}
    history = []
    if graphed:
        from .graphed import train_epoch_graphed

        run_epoch = train_epoch_graphed
    else:
        run_epoch = train_epoch
    for temp in temps:
        opt = optim.Adam(net.parameters(), lr=lr, capturable=graphed)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.2, patience=scheduler_patience, min_lr=min_lr
        )
        best_soft = {"val": -1.0, "sd": None, "epoch": None}
        num_bad = 0
        for epoch in range(nepochs):
            t0 = time.time()
            train_loss = run_epoch(net, train_loader, opt, temp, max_features, loss_fn, device, amp, max_steps_per_epoch) if not graphed else run_epoch(net, train_loader, opt, temp, max_features, loss_fn, device, max_steps_per_epoch)
            val_soft, val_hard = validate(net, val_loader, temp, max_features, device, amp)
            sch.step(val_soft)
            if val_soft > best_soft["val"]:
                best_soft = {"val": val_soft, "sd": copy.deepcopy(net.state_dict()), "epoch": epoch + 1}
                num_bad = 0
            else:
                num_bad += 1
            if val_hard > best_global["val_hard"]:
                best_global = {"val_hard": val_hard, "sd": copy.deepcopy(net.state_dict()), "temp": float(temp), "epoch": epoch + 1}
                if ckpt_path:
                    torch.save(best_global["sd"], ckpt_path)
            rec = {
                "temp": float(temp),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_acc_soft": val_soft,
                "val_acc_hard": val_hard,
                "lr": opt.param_groups[0]["lr"],
                "sec": round(time.time() - t0, 1),
            }
            history.append(rec)
            if log:
                log.log("fit", **rec)
            if num_bad >= early_stop_patience:
                break
        net.load_state_dict(best_soft["sd"])
    net.load_state_dict(best_global["sd"])
    return {"best_val_hard_acc": best_global["val_hard"], "best_temp": best_global["temp"], "best_epoch": best_global["epoch"], "history": history}