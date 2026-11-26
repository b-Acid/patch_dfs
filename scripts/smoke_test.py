"""Smoke test on the selected GPU: forward+backward for the CIFAR ResNet and ViT-S/16.
Usage: PATCHDFS_GPU=0 scripts/smoke_test.sh"""
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from patchdfs.common import MaskLayer2d  # noqa: E402
from patchdfs.models import (  # noqa: E402
    CifarClassifierHead,
    CifarResNetBackbone,
    CifarSelectorHead,
    create_vit,
)

dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(dev), torch.cuda.get_device_capability(dev))


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3


# CIFAR ResNet (GDFS): selector and predictor share the backbone.
bb = CifarResNetBackbone().to(dev)
sel = CifarSelectorHead().to(dev)
cls = CifarClassifierHead().to(dev)
mask = MaskLayer2d(append=False, mask_width=8, patch_size=4)
opt = torch.optim.Adam(list(bb.parameters()) + list(sel.parameters()) + list(cls.parameters()), lr=1e-3)
x = torch.randn(128, 3, 32, 32, device=dev)
m = torch.zeros(128, 64, device=dev)


def cifar_step():
    logits = sel(bb(mask(x, m))).flatten(1)
    pred = cls(bb(mask(x, m)))
    loss = pred.square().mean() + logits.square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


with torch.autocast("cuda", dtype=torch.bfloat16):
    ms = bench(cifar_step)
print(f"CIFAR resnet fwd+bwd (selector+predictor, B=128): {ms:.2f} ms/iter")

# ViT-S/16 (DIME).
vit = create_vit().to(dev)
n_params = sum(p.numel() for p in vit.parameters())
opt = torch.optim.Adam(vit.parameters(), lr=1e-5)
x = torch.randn(32, 3, 224, 224, device=dev)
mask14 = MaskLayer2d(append=False, mask_width=14, patch_size=16)
m = torch.zeros(32, 196, device=dev)


def vit_step():
    feats = vit.forward_features(mask14(x, m))
    loss = feats.square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


with torch.autocast("cuda", dtype=torch.bfloat16):
    ms = bench(vit_step, iters=10)
print(f"ViT-S/16 ({n_params/1e6:.1f}M params) fwd+bwd (B=32): {ms:.2f} ms/iter")
print("peak GPU memory: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("SMOKE OK")
