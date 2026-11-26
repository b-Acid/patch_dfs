"""Measure the compute ledger on an idle GPU.

  python -m patchdfs.bench.run_ledger --run-dir artifacts/ledger --gdfs-mnist artifacts/gdfs/mnist/seed0/final.pt \
      --gdfs-cifar artifacts/gdfs/cifar10/seed0/final.pt --dime-vit artifacts/dime/imagenette/seed0/final.pt \
      --dime-token artifacts/dime/imagenette_token/seed0/final.pt
Writes ledger.json; copy to results/compute_ledger.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from patchdfs.bench.ledger import dime_ledger, gdfs_ledger, token_dropped_ledger
from patchdfs.common import MaskLayer
from patchdfs.common.io import save_json
from patchdfs.dime.model import build_dime
from patchdfs.gdfs.model import build_gdfs
from patchdfs.models import create_vit
from patchdfs.sparse.token_vit import TokenDroppedViT

BUDGETS_GDFS = [1, 5, 10, 15, 20, 30, 50]
BUDGETS_DIME = [1, 5, 10, 15, 20, 30, 50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gdfs-mnist")
    ap.add_argument("--gdfs-cifar")
    ap.add_argument("--dime-mnist")
    ap.add_argument("--dime-vit")
    ap.add_argument("--dime-token")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    device = torch.device("cuda")
    out = {"device": torch.cuda.get_device_name(device), "batch": args.batch, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu_mem_used_mib": torch.cuda.memory_allocated() / 2**20}
    B = args.batch
    iters = args.iters

    if args.gdfs_mnist:
        net = build_gdfs({"kind": "mlp", "hidden": 512, "dropout": 0.3}, 784, 10).to(device).eval()
        net.load_state_dict(torch.load(args.gdfs_mnist, map_location=device))
        out["gdfs_mnist"] = gdfs_ledger(net, torch.randn(B, 784, device=device), BUDGETS_GDFS, amp=False, iters=iters)
        print("gdfs_mnist done")
    if args.gdfs_cifar:
        net = build_gdfs({"kind": "cifar_resnet", "mask_width": 8, "patch_size": 4}, 64, 10).to(device).eval()
        net.load_state_dict(torch.load(args.gdfs_cifar, map_location=device))
        out["gdfs_cifar10"] = gdfs_ledger(net, torch.randn(B, 3, 32, 32, device=device), [k for k in BUDGETS_GDFS if k <= 30], amp=True, iters=iters)
        print("gdfs_cifar10 done")
    if args.dime_mnist:
        net = build_dime({"kind": "mlp", "hidden": 512, "dropout": 0.3}, 784, 10).to(device).eval()
        net.load_state_dict(torch.load(args.dime_mnist, map_location=device))
        out["dime_mnist"] = dime_ledger(net, torch.randn(B, 784, device=device), BUDGETS_DIME, amp=False, iters=iters, fused=False)
        print("dime_mnist done")
    if args.dime_vit:
        net = build_dime({"kind": "vit", "mask_width": 14, "patch_size": 16}, 196, 10).to(device).eval()
        net.load_state_dict(torch.load(args.dime_vit, map_location=device))
        out["dime_imagenette_vit"] = dime_ledger(net, torch.randn(B, 3, 224, 224, device=device), BUDGETS_DIME, amp=True, iters=iters, fused=True)
        print("dime_imagenette_vit done")
    if args.dime_token:
        bb = create_vit("vit_small_patch16_224", pretrained=True)
        net = TokenDroppedViT(bb, 10).to(device).eval()
        net.load_state_dict(torch.load(args.dime_token, map_location=device))
        out["dime_imagenette_token"] = token_dropped_ledger(net, torch.randn(B, 3, 224, 224, device=device), BUDGETS_DIME, amp=True, iters=iters)
        print("dime_imagenette_token done")
    os.makedirs(args.run_dir, exist_ok=True)
    path = os.path.join(args.run_dir, f"ledger_b{B}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(path)


if __name__ == "__main__":
    main()
