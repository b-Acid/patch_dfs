"""GDFS networks (Covert et al., ICML 2023).

MNIST: predictor and selector are two independent MLPs on the masked input (mask appended).
CIFAR-10: selector and predictor share a ResNet backbone and have their own heads.
Source: third_party/dynamic-selection/experiments/{mnist/run.py, cifar/run_greedy.py}.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from patchdfs.common import MaskLayer, MaskLayer2d
from patchdfs.models import MLP, CifarClassifierHead, CifarResNetBackbone, CifarSelectorHead


class GDFSNet(nn.Module):
    def __init__(
        self,
        mask_layer: nn.Module,
        *,
        backbone: nn.Module | None = None,
        selector: nn.Module | None = None,
        predictor: nn.Module | None = None,
        selector_head: nn.Module | None = None,
        predictor_head: nn.Module | None = None,
    ):
        super().__init__()
        self.mask_layer = mask_layer
        self.backbone = backbone
        self.selector = selector
        self.predictor = predictor
        self.selector_head = selector_head
        self.predictor_head = predictor_head

    @property
    def mask_size(self) -> int:
        return self.mask_layer.mask_size

    def select_logits(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        xm = self.mask_layer(x, m)
        if self.backbone is not None:
            return self.selector_head(self.backbone(xm)).flatten(1)
        return self.selector(xm)

    def predict(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        xm = self.mask_layer(x, m)
        if self.backbone is not None:
            return self.predictor_head(self.backbone(xm))
        return self.predictor(xm)

    def predictor_module(self) -> nn.Module:
        """Module mapping masked inputs to class logits (shares parameters with self)."""
        if self.backbone is not None:
            return nn.Sequential(self.backbone, self.predictor_head)
        return self.predictor


def build_gdfs(model_cfg: dict, d_in: int, d_out: int) -> GDFSNet:
    kind = model_cfg["kind"]
    if kind == "mlp":
        append = bool(model_cfg.get("append_mask", True))
        mask_layer = MaskLayer(append=append, mask_size=d_in)
        m_in = d_in * (2 if append else 1)
        hidden, dropout = model_cfg["hidden"], model_cfg["dropout"]
        return GDFSNet(
            mask_layer,
            selector=MLP(m_in, hidden, d_in, dropout),
            predictor=MLP(m_in, hidden, d_out, dropout),
        )
    if kind == "cifar_resnet":
        mask_layer = MaskLayer2d(
            append=False, mask_width=int(model_cfg["mask_width"]), patch_size=int(model_cfg["patch_size"])
        )
        return GDFSNet(
            mask_layer,
            backbone=CifarResNetBackbone(),
            selector_head=CifarSelectorHead(),
            predictor_head=CifarClassifierHead(d_out),
        )
    raise ValueError(kind)
