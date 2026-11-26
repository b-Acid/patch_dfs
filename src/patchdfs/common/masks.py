"""Masking utilities ported from the authors' code.

Sources: dynamic-selection/dynamic_selection/utils.py (GDFS, ICML 2023) and
DIME/dime/utils.py (ICLR 2024), vendored under third_party/. Behaviour is identical;
only the random-mask generator gained an explicit torch.Generator.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical


def make_onehot(x: torch.Tensor) -> torch.Tensor:
    """Turn a (B, d) score matrix into one-hot rows via argmax."""
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot.scatter_(1, argmax.unsqueeze(1), 1)  # scatter (no host sync; CUDA-graph safe)
    return onehot


def generate_uniform_mask(
    batch_size: int,
    num_features: int,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Binary masks whose number of kept features is uniform on {0, ..., d}.

    Each feature is kept iff U_i > R, with U_i iid Uniform[0, 1) and one R per row.
    """
    unif = torch.rand(batch_size, num_features, device=device, generator=generator, dtype=dtype)
    ref = torch.rand(batch_size, 1, device=device, generator=generator, dtype=dtype)
    return (unif > ref).to(dtype)


class MaskLayer(nn.Module):
    """Mask 1d features: x * m, optionally appending m as extra inputs."""

    def __init__(self, append: bool, mask_size: int | None = None):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


MaskLayer1d = MaskLayer


class MaskLayer2d(nn.Module):
    """Zero out image patches.

    m is (B, mask_width**2) or (B, 1, g, g) with g = mask_width. It is upsampled by
    patch_size (nearest) and multiplied into x of shape (B, C, H, W).
    """

    def __init__(self, append: bool, mask_width: int, patch_size: int):
        super().__init__()
        self.append = append
        self.mask_width = mask_width
        self.mask_size = mask_width**2
        if patch_size == 1:
            self.upsample: nn.Module = nn.Identity()
        elif patch_size > 1:
            self.upsample = nn.Upsample(scale_factor=patch_size)
        else:
            raise ValueError("patch_size should be int >= 1")
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if len(m.shape) == 2:
            m = m.reshape(-1, 1, self.mask_width, self.mask_width)
        elif len(m.shape) != 4:
            raise ValueError(f"cannot determine how to reshape mask with shape = {m.shape}")
        m = self.upsample(m.to(x.dtype))
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


class ConcreteSelector(nn.Module):
    """Sample a near-one-hot selection from the Concrete (Gumbel-softmax) distribution."""

    def __init__(self, gamma: float = 0.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, temp: float, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
        return dist.rsample()
