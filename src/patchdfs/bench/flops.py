"""FLOPs counting with torch.utils.flop_counter (FLOPs = 2 x MACs; conv, matmul, sdpa, ...)."""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_flops(fn: Callable[[], object], *, warn_unsupported: bool = False) -> tuple[float, dict]:
    """Run fn() under FlopCounterMode; returns (total FLOPs, per-op FLOPs)."""
    mode = FlopCounterMode(display=False)
    with mode:
        fn()
    total = mode.get_total_flops()
    per_op = {str(k).replace("aten::", ""): float(v) for k, v in mode.get_flop_counts()["Global"].items()}
    return float(total), per_op


def conv_linear_macs(module: torch.nn.Module, x: torch.Tensor) -> float:
    """Analytic MACs for Conv2d / Linear / ConvTranspose2d / BatchNorm-free feed-forward modules
    (used as a cross-check of the counter)."""
    macs = [0.0]
    hooks = []

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs[0] += out.numel() * k

    def linear_hook(m, inp, out):
        macs[0] += out.numel() * m.in_features

    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.ConvTranspose2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    with torch.no_grad():
        module(x)
    for h in hooks:
        h.remove()
    return macs[0]
