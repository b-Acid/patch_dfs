"""Greedy (argmax) selection at test time, recording predictions after every step so that every
budget is read off a single pass (identical to the authors' per-budget evaluate, which repeats the
same deterministic selection)."""
from __future__ import annotations

import torch

from patchdfs.common import make_onehot


@torch.no_grad()
def greedy_select(net, x: torch.Tensor, k_max: int, amp: bool = False):
    """Returns selections (N, k_max) of flat feature indices and predictions (N, k_max, C)."""
    net.eval()
    device = x.device
    m = torch.zeros(len(x), net.mask_size, device=device)
    sels, preds = [], []
    for _ in range(k_max):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = net.select_logits(x, m) - 1e6 * m
        idx = logits.argmax(dim=1)
        m = torch.max(m, make_onehot(logits))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = net.predict(x, m)
        sels.append(idx.cpu())
        preds.append(pred.float().cpu())
    return torch.stack(sels, dim=1), torch.stack(preds, dim=1)


def evaluate_budgets(net, loader, budgets, amp: bool = False, device=None):
    device = device or next(net.parameters()).device
    k_max = max(budgets)
    all_sel, all_pred, all_y = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        sel, pred = greedy_select(net, x, k_max, amp)
        all_sel.append(sel)
        all_pred.append(pred)
        all_y.append(y)
    import numpy as np

    sel = torch.cat(all_sel).numpy()
    pred = torch.cat(all_pred).numpy()
    y = torch.cat(all_y).numpy()
    acc = {int(b): float((pred[:, b - 1].argmax(1) == y).mean()) for b in budgets}
    return {"acc": acc, "selections": sel, "preds": pred, "y": y}
