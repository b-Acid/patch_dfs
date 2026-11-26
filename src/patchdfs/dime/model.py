"""DIME networks (Gadgil et al., ICLR 2024): a predictor f(x_S) and a value network v(x_S).

MNIST: two MLPs with the same architecture; the first layer is tied (authors' MNIST script).
Imagenette: a shared timm ViT-S/16 backbone; the predictor head is a linear layer on the CLS
token, the value head a linear layer on the patch tokens. For the shared backbone, both()
computes both outputs from ONE backbone pass, which is numerically identical to the authors'
two separate passes (see tests/test_dime_core.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from patchdfs.common import MaskLayer, MaskLayer2d
from patchdfs.models import MLP


class PredictorViT(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(backbone.embed_dim, num_classes)
        self.backbone = backbone

    def head_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone.forward_head(feats, pre_logits=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_from_features(self.backbone.forward_features(x))


class ValueNetworkViT(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.hidden = nn.Linear(backbone.embed_dim, 1)

    def head_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.hidden(feats[:, 1:]).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_from_features(self.backbone.forward_features(x))


class DIMENet(nn.Module):
    def __init__(self, mask_layer, predictor, value_network, backbone: nn.Module | None = None):
        super().__init__()
        self.mask_layer = mask_layer
        self.predictor = predictor
        self.value_network = value_network
        self.backbone = backbone

    @property
    def mask_size(self) -> int:
        return self.mask_layer.mask_size

    def predict(self, x, m):
        return self.predictor(self.mask_layer(x, m))

    def value_logits(self, x, m):
        return self.value_network(self.mask_layer(x, m))

    def both(self, x, m):
        """(class logits, value logits) for the same masked input."""
        xm = self.mask_layer(x, m)
        if self.backbone is None:
            return self.predictor(xm), self.value_network(xm)
        feats = self.backbone.forward_features(xm)
        return self.predictor.head_from_features(feats), self.value_network.head_from_features(feats)

    def predictor_module(self) -> nn.Module:
        return self.predictor


def build_dime(model_cfg: dict, d_in: int, d_out: int) -> DIMENet:
    kind = model_cfg["kind"]
    if kind == "mlp":
        append = bool(model_cfg.get("append_mask", True))
        mask_layer = MaskLayer(append=append, mask_size=d_in)
        m_in = d_in * (2 if append else 1)
        hidden, dropout = model_cfg["hidden"], model_cfg["dropout"]
        predictor = MLP(m_in, hidden, d_out, dropout)
        value = MLP(m_in, hidden, d_in, dropout)
        value[0] = predictor[0]  # tied first layer, as in the authors' MNIST script
        return DIMENet(mask_layer, predictor, value)
    if kind == "vit":
        from patchdfs.models import create_vit

        mask_layer = MaskLayer2d(
            append=False, mask_width=int(model_cfg["mask_width"]), patch_size=int(model_cfg["patch_size"])
        )
        backbone = create_vit(model_cfg.get("name", "vit_small_patch16_224"), pretrained=bool(model_cfg.get("pretrained", True)))
        return DIMENet(mask_layer, PredictorViT(backbone, d_out), ValueNetworkViT(backbone), backbone=backbone)
    if kind == "vit_token_dropped":
        from patchdfs.models import create_vit
        from patchdfs.sparse.token_vit import TokenDroppedViT

        backbone = create_vit(model_cfg.get("name", "vit_small_patch16_224"), pretrained=bool(model_cfg.get("pretrained", True)))
        return TokenDroppedViT(backbone, d_out)
    raise ValueError(kind)
