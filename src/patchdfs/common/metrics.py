"""Metrics."""
from __future__ import annotations

import torch


@torch.no_grad()
def top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def confidence(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1).max(dim=1).values


def entropy(logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Per-sample entropy of the softmax distribution, natural log units (as in DIME)."""
    p = torch.softmax(logits, dim=1)
    return -(p * torch.log(p + eps)).sum(dim=1)
