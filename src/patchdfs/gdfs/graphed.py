"""Graphed version of the GDFS training step (see common/cuda_graphs.py)."""
from __future__ import annotations

import torch

from patchdfs.common import ConcreteSelector, make_onehot
from patchdfs.common.cuda_graphs import GraphedStep

# torch.distributions validates args with a device->host sync, which is illegal during capture.
torch.distributions.Distribution.set_default_validate_args(False)


def make_gdfs_step(net, opt, loss_fn, temp: float, max_features: int):
    selector_layer = ConcreteSelector()

    def step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        net.train()
        opt.zero_grad(set_to_none=False)
        m = torch.zeros(len(x), net.mask_size, device=x.device)
        total = x.new_zeros(())
        for _ in range(max_features):
            logits = net.select_logits(x, m)
            soft = selector_layer(logits, temp)
            m_soft = torch.max(m, soft)
            pred = net.predict(x, m_soft)
            loss = loss_fn(pred, y)
            (loss / max_features).backward()
            with torch.no_grad():
                m = torch.max(m, make_onehot(logits - 1e6 * m))
            total = total + loss.detach() / max_features
        opt.step()
        return total

    return step


def train_epoch_graphed(net, loader, opt, temp: float, max_features: int, loss_fn, device, max_steps=None) -> float:
    """One epoch with the whole 50-step iteration captured in a CUDA graph. Only for fp32 MLP runs
    (no autocast) and Adam(capturable=True)."""
    x0, y0 = next(iter(loader))
    x0, y0 = x0.to(device), y0.to(device)
    graphed = GraphedStep(make_gdfs_step(net, opt, loss_fn, temp, max_features), x0, y0)
    tot = n = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = graphed(x, y)
        tot += loss.item() * len(x)
        n += len(x)
        if max_steps and i + 1 >= max_steps:
            break
    return tot / max(n, 1)
