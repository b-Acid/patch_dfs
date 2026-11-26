"""ViT wrappers ported from DIME/dime/vit.py. The backbone is shared between the
predictor and the value network; each has its own head."""
from __future__ import annotations

import os

import torch
import torch.nn as nn

DEFAULT_VIT = "vit_small_patch16_224"


def create_vit(name: str = DEFAULT_VIT, pretrained: bool = True):
    """timm backbone. Weights come from the HF mirror when huggingface.co is unreachable."""
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    import timm

    return timm.create_model(name, pretrained=pretrained)


class PredictorViT(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(backbone.embed_dim, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)
        x = self.backbone.forward_head(x, pre_logits=True)
        return self.fc(x)


class ValueNetworkViT(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.hidden = nn.Linear(backbone.embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)[:, 1:]
        return self.hidden(x).squeeze(-1)
