"""Two-hidden-layer MLP as in both papers (ReLU + dropout 0.3)."""
from __future__ import annotations

import torch.nn as nn


def MLP(d_in: int, hidden: int, d_out: int, dropout: float = 0.3, num_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(num_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)
