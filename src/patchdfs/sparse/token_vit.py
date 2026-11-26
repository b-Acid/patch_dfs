"""Token-dropped ViT for DIME: only the CLS token and the observed patch tokens are processed.

This is the ViT analogue of skipping convolutions on zeroed blocks. The dense DIME model runs the
full 197-token ViT on the masked image at every step; here an unobserved patch is removed from the
token sequence, so the cost of a pass is proportional to the number of observed patches.

- Patch embedding: each observed 16x16 patch is embedded with the conv weights as a linear map
  (numerically identical to the dense patch embedding at that position).
- Position embeddings are gathered for the observed positions; within a batch the token sets are
  padded to the longest and masked out in attention.
- Predictor: linear head on the CLS token (as DIME).
- Value head: the dense version reads a token at each candidate position; with dropped tokens that
  token does not exist, so all 196 candidate positions are scored by learned per-position queries
  (initialised from the patch position embeddings) that cross-attend to the observed tokens.

The function computed is NOT the same as the dense masked ViT (empty tokens are absent from
attention), so the model must be trained with this forward; that is the point of the experiment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block_forward(blk, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    """timm Block computation (no drop-path) with an optional key padding mask (True = padded
    token to ignore). Same weights as blk; see tests for exactness."""
    B, N, C = x.shape
    a = blk.attn
    qkv = a.qkv(blk.norm1(x)).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attn_mask = None
    if key_padding_mask is not None:
        attn_mask = ~key_padding_mask[:, None, None, :]  # bool: True = attend
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=a.scale)
    y = y.transpose(1, 2).reshape(B, N, C)
    y = a.proj_drop(a.proj(y))
    x = x + blk.drop_path1(blk.ls1(y))
    x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return x


class TokenDroppedViT(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10, num_heads: int | None = None):
        super().__init__()
        self.backbone = backbone
        d = backbone.embed_dim
        self.num_prefix = getattr(backbone, "num_prefix_tokens", 1)
        self.grid_tokens = backbone.pos_embed.shape[1] - self.num_prefix
        self.mask_size = self.grid_tokens
        self.fc = nn.Linear(d, num_classes)
        # Value head: per-position queries cross-attending to the observed tokens.
        self.queries = nn.Parameter(backbone.pos_embed[0, self.num_prefix :].clone())
        self.q_norm = nn.LayerNorm(d)
        self.kv_norm = nn.LayerNorm(d)
        heads = num_heads or backbone.blocks[0].attn.num_heads
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.value_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    # --- token sequence -------------------------------------------------------------
    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) embeddings of all patches, using the conv weights as a linear map."""
        pe = self.backbone.patch_embed.proj
        w = pe.weight.reshape(pe.out_channels, -1).to(x.dtype)  # (D, C*P*P)
        patches = F.unfold(x, kernel_size=pe.kernel_size, stride=pe.stride).transpose(1, 2)  # (B, N, CPP)
        return F.linear(patches, w, pe.bias.to(x.dtype))

    def encode(self, x: torch.Tensor, m: torch.Tensor, max_tokens: int | None = None):
        """Run the transformer on CLS + observed tokens. Returns (tokens (B, 1+T, D) after the
        final norm, pad mask (B, 1+T) with True = padded). With max_tokens given, no host sync
        happens and the shapes are static, so the pass can be captured in a CUDA graph."""
        B = x.shape[0]
        N = self.grid_tokens
        obs = m.reshape(B, N) > 0.5
        if max_tokens is None:
            T = max(int(obs.sum(dim=1).max().item()), 1)
            static = False
        else:
            T = max_tokens
            static = True
        idx = torch.argsort(obs.to(torch.int8), dim=1, descending=True, stable=True)[:, :T]  # observed first
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, self.backbone.embed_dim)
        patches = self._patch_tokens(x)
        sel = patches.gather(1, gather_idx) * obs.gather(1, idx).unsqueeze(-1).to(patches.dtype)
        cls = self.backbone.cls_token.expand(B, -1, -1).to(patches.dtype)
        tokens = torch.cat([cls, sel], dim=1)
        pos_cols = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), idx + self.num_prefix], dim=1)
        pos = self.backbone.pos_embed.expand(B, -1, -1).gather(1, pos_cols.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        tokens = tokens + pos.to(tokens.dtype)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~obs.gather(1, idx)], dim=1)
        tokens = self.backbone.norm_pre(tokens)
        kpm = None if B == 1 else (pad if (static or pad.any()) else None)  # a single image never needs padding
        for blk in self.backbone.blocks:
            tokens = block_forward(blk, tokens, kpm)
        return self.backbone.norm(tokens), pad

    def predict_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone.fc_norm(tokens[:, 0]))

    def value_from_tokens(self, tokens: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.q_norm(self.queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.kv_norm(tokens)
        out, _ = self.cross(q, kv, kv, key_padding_mask=pad, need_weights=False)
        return self.value_mlp(out).squeeze(-1)

    # --- DIMENet-compatible interface ------------------------------------------------
    def predict(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.encode(x, m)
        return self.predict_from_tokens(tokens)

    def value_logits(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        tokens, pad = self.encode(x, m)
        return self.value_from_tokens(tokens, pad)

    def both(self, x: torch.Tensor, m: torch.Tensor, max_tokens: int | None = None):
        tokens, pad = self.encode(x, m, max_tokens)
        return self.predict_from_tokens(tokens), self.value_from_tokens(tokens, pad)

    def predictor_module(self) -> nn.Module:
        return self