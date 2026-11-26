"""Port of the authors' CMIEstimator (third_party/DIME/dime/cmi_estimator.py) in plain PyTorch:
training step, greedy validation, exploration-decay early stopping, and inference with the
three stopping criteria (budget / confidence / penalised).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from patchdfs.common.io import RunLogger


def _autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def entropy(pred: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=pred).entropy()


def confidence(pred: torch.Tensor) -> torch.Tensor:
    return torch.softmax(pred, dim=1).max(dim=1).values


def ind_to_onehot(inds: torch.Tensor, n: int) -> torch.Tensor:
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot.scatter_(1, inds.unsqueeze(1), 1)  # scatter (no host sync; CUDA-graph safe)
    return onehot


def scaled_cmi(net, val_logits: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """cmi_scaling='bounded': sigmoid(value) * entropy of the current prediction."""
    return torch.sigmoid(val_logits) * entropy(pred).unsqueeze(1)


def train_epoch(net, loader, opt, *, eps: float, max_features: int, device, amp: bool, max_steps=None, fused: bool) -> dict:
    net.train()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    mask_size = net.mask_size
    costs = torch.ones(mask_size, device=device)
    tot = {"pred_loss": 0.0, "value_loss": 0.0, "n": 0}
    K = max_features
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        mask = torch.zeros(len(x), mask_size, device=device)
        opt.zero_grad(set_to_none=True)

        # Step 0: prediction with no features.
        with _autocast(amp):
            pred0, val0 = net.both(x, mask)
            loss0 = loss_fn(pred0, y)
        (loss0.mean() / (K + 1)).backward(retain_graph=fused and K > 0)
        loss_prev, pred_prev = loss0.detach(), pred0.detach()
        val_prev = val0 if fused else None

        for t in range(K):
            with _autocast(amp):
                val_logits = val_prev if fused else net.value_logits(x, mask)
                pred_cmi = scaled_cmi(net, val_logits, pred_prev)
            best = torch.argmax(pred_cmi / costs, dim=1)
            random = torch.randint(mask_size, (len(x),), device=device)
            exploit = (torch.rand(len(x), device=device) > eps).int()
            actions = exploit * best + (1 - exploit) * random
            mask = torch.max(mask, ind_to_onehot(actions, mask_size))
            with _autocast(amp):
                pred_next, val_next = net.both(x, mask)
                loss_next = loss_fn(pred_next, y)
            delta = loss_prev - loss_next.detach()
            value_loss = nn.functional.mse_loss(pred_cmi[torch.arange(len(x)), actions], delta)
            total = value_loss.mean() + loss_next.mean()
            (total / (K + 1)).backward(retain_graph=fused and t < K - 1)
            loss_prev, pred_prev = loss_next.detach(), pred_next.detach()
            if fused:
                val_prev = val_next
        opt.step()
        tot["pred_loss"] += loss_next.detach().mean().item() * len(x)
        tot["value_loss"] += value_loss.detach().mean().item() * len(x)
        tot["n"] += len(x)
        if max_steps and step + 1 >= max_steps:
            break
    if tot["n"] == 0:
        raise RuntimeError("train loader yielded no batches (drop_last with a subset smaller than batch_size?)")
    return {"pred_loss": tot["pred_loss"] / tot["n"], "value_loss": tot["value_loss"] / tot["n"]}


def make_dime_step(net, opt, eps: float, max_features: int):
    """Whole DIME training step (fp32 MLP path, two passes per step) for CUDA-graph capture."""
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    K = max_features
    costs = torch.ones(net.mask_size, device=next(net.parameters()).device)

    def step(x: torch.Tensor, y: torch.Tensor):
        net.train()
        opt.zero_grad(set_to_none=False)
        mask = torch.zeros(len(x), net.mask_size, device=x.device)
        pred0, val0 = net.both(x, mask)
        loss0 = loss_fn(pred0, y)
        (loss0.mean() / (K + 1)).backward()
        loss_prev, pred_prev = loss0.detach(), pred0.detach()
        val_prev = val0
        pred_sum = torch.zeros((), device=x.device)
        val_sum = torch.zeros((), device=x.device)
        for t in range(K):
            pred_cmi = torch.sigmoid(val_prev) * entropy_manual(pred_prev).unsqueeze(1)
            best = torch.argmax(pred_cmi / costs, dim=1)
            random = torch.randint(net.mask_size, (len(x),), device=x.device)
            exploit = (torch.rand(len(x), device=x.device) > eps).int()
            actions = exploit * best + (1 - exploit) * random
            mask = torch.max(mask, ind_to_onehot(actions, net.mask_size))
            pred_next, val_next = net.both(x, mask)
            loss_next = loss_fn(pred_next, y)
            delta = loss_prev - loss_next.detach()
            value_loss = nn.functional.mse_loss(pred_cmi[torch.arange(len(x), device=x.device), actions], delta)
            total = value_loss.mean() + loss_next.mean()
            (total / (K + 1)).backward()
            loss_prev, pred_prev, val_prev = loss_next.detach(), pred_next.detach(), val_next
            pred_sum = pred_sum + loss_next.detach().mean() / K
            val_sum = val_sum + value_loss.detach() / K
        opt.step()
        return pred_sum, val_sum

    return step


def entropy_manual(pred: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(pred, dim=1)
    return -(p * torch.log(p + 1e-10)).sum(dim=1)


def train_epoch_graphed(net, loader, opt, *, eps: float, max_features: int, device, max_steps=None) -> dict:
    from patchdfs.common.cuda_graphs import GraphedStep

    torch.distributions.Distribution.set_default_validate_args(False)
    x0, y0 = next(iter(loader))
    graphed = GraphedStep(make_dime_step(net, opt, eps, max_features), x0.to(device), y0.to(device))
    tot_pred = tot_val = n = 0
    for i, (x, y) in enumerate(loader):
        pl, vl = graphed(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
        tot_pred += pl.item() * len(x)
        tot_val += vl.item() * len(x)
        n += len(x)
        if max_steps and i + 1 >= max_steps:
            break
    return {"pred_loss": tot_pred / n, "value_loss": tot_val / n}


@torch.no_grad()
def greedy_rollout(net, loader, max_features: int, device, amp: bool, fused: bool):
    """Greedy (eps=0, no repeats) rollout; returns per-step loss/accuracy (mean over steps and at
    the final step) and the per-step accuracies. Calls net.both() max_features + 1 times."""
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    K = max_features
    loss_sum = torch.zeros(K + 1)
    correct = torch.zeros(K + 1, dtype=torch.long)
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        mask = torch.zeros(len(x), net.mask_size, device=device)
        with _autocast(amp):
            pred, val = net.both(x, mask)
        for t in range(K + 1):
            loss_sum[t] += loss_fn(pred.float(), y).sum().cpu()
            correct[t] += (pred.argmax(1) == y).sum().cpu()
            if t == K:
                break
            with _autocast(amp):
                pred_cmi = scaled_cmi(net, val, pred) - 1e6 * mask
                mask = torch.max(mask, ind_to_onehot(pred_cmi.argmax(dim=1), net.mask_size))
                pred, val = net.both(x, mask)
        n += len(x)
    per_step_loss = (loss_sum / n).tolist()
    per_step_acc = (correct / n).tolist()
    return {
        "loss_mean": float(np.mean(per_step_loss)),
        "acc_mean": float(np.mean(per_step_acc)),
        "loss_final": per_step_loss[-1],
        "acc_final": per_step_acc[-1],
        "per_step_acc": per_step_acc,
    }


def fit_dime(
    net,
    train_loader,
    val_loader,
    *,
    max_features: int,
    lr: float,
    min_lr: float,
    eps0: float,
    eps_decay: float,
    eps_steps: int,
    patience: int,
    max_epochs: int,
    device,
    amp: bool = False,
    fused: bool = False,
    log: RunLogger | None = None,
    max_steps_per_epoch: int | None = None,
    graphed: bool = False,
    ckpt_path: str | None = None,
) -> dict:
    opt = optim.Adam(net.parameters(), lr=lr, capturable=graphed)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.2, patience=patience, min_lr=min_lr)
    early_after = patience + 1
    eps, n_eps, num_bad = eps0, 0, 0
    best = {"acc_final": -1.0, "sd": None, "epoch": None}
    history = []
    for epoch in range(max_epochs):
        t0 = time.time()
        if graphed:
            tr = train_epoch_graphed(net, train_loader, opt, eps=eps, max_features=max_features, device=device, max_steps=max_steps_per_epoch)
        else:
            tr = train_epoch(net, train_loader, opt, eps=eps, max_features=max_features, device=device, amp=amp, max_steps=max_steps_per_epoch, fused=fused)
        va = greedy_rollout(net, val_loader, max_features, device, amp, fused)
        sch.step(va["loss_mean"])
        if va["loss_mean"] == sch.best:
            num_bad = 0
        else:
            num_bad += 1
        if va["acc_final"] > best["acc_final"]:
            best = {"acc_final": va["acc_final"], "sd": copy.deepcopy(net.state_dict()), "epoch": epoch + 1}
            if ckpt_path:
                torch.save(best["sd"], ckpt_path)
        rec = {"epoch": epoch + 1, "eps": eps, "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1), **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items() if k != "per_step_acc"}}
        history.append(rec)
        if log:
            log.log("fit", **rec)
        if num_bad > early_after:
            eps *= eps_decay
            num_bad = 0
            n_eps += 1
            if log:
                log.log("eps_decay", eps=eps, step=n_eps)
            if n_eps >= eps_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr
    net.load_state_dict(best["sd"])
    return {"best_val_acc_final": best["acc_final"], "best_epoch": best["epoch"], "epochs": len(history), "history": history}


@torch.no_grad()
def infer(net, loader, device, *, amp: bool = False, fused: bool = False, budget=None, lam=None, confidence_level=None, max_steps=None, feature_costs=None, multi: int = 1):
    """Authors' predict_step: per-sample stopping with budget / penalised (lam) / confidence criteria.
    The prediction recorded for a sample is the one computed for its final mask (identical to the
    authors' extra final predictor pass)."""
    assert sum(v is not None for v in (budget, lam, confidence_level)) == 1
    net.eval()
    d = net.mask_size
    costs = torch.ones(d, device=device) if feature_costs is None else torch.as_tensor(feature_costs, dtype=torch.float32, device=device)
    max_steps = max_steps or d
    if multi > 1:
        max_steps = (d + multi - 1) // multi
    masks, preds, ys, passes = [], [], [], 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        mask = torch.zeros(len(x), d, device=device)
        with _autocast(amp):
            pred, val = net.both(x, mask)
        passes += 1
        final_pred = pred.float().clone()
        for _ in range(max_steps):
            with _autocast(amp):
                pred_cmi = scaled_cmi(net, val, pred)
                check_pos = pred_cmi.max(dim=1).values >= 0
                pred_cmi = pred_cmi - 1e6 * mask
                if multi > 1 and budget is not None:
                    # pick up to 'multi' patches per step (fewer sequential passes)
                    m = int(min(multi, max(int(budget), 1)))
                    top = torch.topk(pred_cmi / costs, m, dim=1).indices
                    selection = torch.zeros_like(mask).scatter_(1, top, 1)
                else:
                    best = torch.argmax(pred_cmi / costs, dim=1)
                    selection = ind_to_onehot(best, d)
                if lam is not None:
                    accept = torch.max(pred_cmi / costs, dim=1).values > lam
                elif budget is not None:
                    accept = torch.sum(costs * torch.max(mask, selection), dim=1) <= budget
                else:
                    accept = confidence(pred) < confidence_level
                accept = accept & check_pos
            if accept.sum().item() == 0:
                # Nobody accepts another feature: pred was computed for everyone's final mask.
                final_pred = pred.float()
                break
            final_pred[~accept] = pred.float()[~accept]
            mask[accept] = torch.max(mask[accept], selection[accept])
            with _autocast(amp):
                pred, val = net.both(x, mask)
            passes += 1
        else:
            # max_steps reached with samples still accepting: last pred is their final prediction.
            final_pred = pred.float()
        masks.append(mask.cpu())
        preds.append(final_pred.cpu())
        ys.append(y)
    mask = torch.cat(masks)
    pred = torch.cat(preds)
    y = torch.cat(ys)
    return {"mask": mask.numpy(), "pred": pred.numpy(), "y": y.numpy(), "n_features": mask.sum(1).numpy(), "passes": passes}