"""Single-pass dynamic token selection for ViT (our variant).

The first 'glance_layers' blocks see every token. The CLS attention of the next block then ranks
the patches; the top-k are kept, the dropped patches are fused into one attention-pooled token,
and the remaining blocks run on CLS + fused + k tokens. One forward pass per image whatever k is,
with static shapes for a given k (so the pass can be compiled or captured in a CUDA graph).

This is the limiting case of sequential selection (GDFS, DIME): everything is chosen at once after
a glance instead of one patch per pass, trading adaptivity for k fewer passes. The mechanism
follows EViT (Liang et al., CVPR 2022); here it is the vehicle for measuring how much of the
compute and latency of sequential selection is structural.

The glance is not extra work relative to the dense model: it is the first layers of the same
backbone, and the kept tokens continue from their hidden states (nothing is recomputed).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .token_vit import block_forward


def cls_attention(blk, tokens: torch.Tensor) -> torch.Tensor:
    """Attention from the CLS token to each patch token in block 'blk', averaged over heads.
    'tokens' is the input of the block, shape (B, 1 + N, D); returns (B, N)."""
    a = blk.attn
    B, T, _ = tokens.shape
    qkv = a.qkv(blk.norm1(tokens)).reshape(B, T, 3, a.num_heads, a.head_dim)
    q = qkv[:, 0, 0]  # (B, heads, head_dim)
    k = qkv[:, :, 1]  # (B, T, heads, head_dim)
    attn = (torch.einsum("bhd,bthd->bht", q, k) * a.scale).softmax(-1)
    return attn[:, :, 1:].mean(1)


class OnePassViT(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10, glance_layers: int = 2, fuse_dropped: bool = True):
        super().__init__()
        n_blocks = len(backbone.blocks)
        if not 1 <= glance_layers < n_blocks:
            raise ValueError(f"glance_layers must be in [1, {n_blocks - 1}], got {glance_layers}")
        if getattr(backbone, "num_prefix_tokens", 1) != 1:
            raise ValueError("expected exactly one prefix (CLS) token")
        self.backbone = backbone
        self.glance_layers = glance_layers
        self.fuse_dropped = fuse_dropped
        self.grid_tokens = backbone.pos_embed.shape[1] - 1
        self.mask_size = self.grid_tokens
        self.fc = nn.Linear(backbone.embed_dim, num_classes)
        self.fused_token = nn.Parameter(torch.zeros(1, 1, backbone.embed_dim))  # EViT's extra token

    def token_count(self, k: int) -> int:
        """Tokens through the remaining blocks: CLS + k kept (+ 1 fused)."""
        return 1 + min(int(k), self.grid_tokens) + int(self.fuse_dropped)

    def forward(self, x: torch.Tensor, k: int):
        """Returns (logits (B, num_classes), kept indices (B, k), CLS-attention scores (B, N))."""
        bb = self.backbone
        B = x.shape[0]
        N = self.grid_tokens
        k = min(int(k), N)
        tokens = bb.patch_embed(x)
        tokens = torch.cat([bb.cls_token.expand(B, -1, -1).to(tokens.dtype), tokens], dim=1)
        tokens = bb.norm_pre(tokens + bb.pos_embed.to(tokens.dtype))
        for blk in bb.blocks[: self.glance_layers]:
            tokens = block_forward(blk, tokens, None)
        score = cls_attention(bb.blocks[self.glance_layers], tokens)  # (B, N)
        idx = score.topk(k, dim=1).indices
        D = tokens.shape[-1]
        patches = tokens[:, 1:]
        keep = patches.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        drop_w = score * (1.0 - torch.zeros_like(score).scatter(1, idx, 1.0))
        drop_w = drop_w / drop_w.sum(1, keepdim=True).clamp_min(1e-6)
        fused = torch.einsum("bn,bnd->bd", drop_w, patches.to(drop_w.dtype)).unsqueeze(1)
        fused = fused + self.fused_token.to(fused.dtype)
        tokens = torch.cat([tokens[:, :1], fused, keep] if self.fuse_dropped else [tokens[:, :1], keep], dim=1)
        for blk in bb.blocks[self.glance_layers :]:
            tokens = block_forward(blk, tokens, None)
        tokens = bb.fc_norm(bb.norm(tokens))
        return self.fc(tokens[:, 0]), idx, score
