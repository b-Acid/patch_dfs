"""Train the CIFAR-10 static baselines (random mask / centre crop / CAE) of the GDFS paper.

  python -m patchdfs.gdfs.train_baselines --method random --run-dir artifacts/gdfs/cifar10_baselines/random
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import yaml

from patchdfs.common.io import RunLogger
from patchdfs.common.seeding import seed_everything, seed_worker
from patchdfs.data import get_cifar10
from patchdfs.gdfs.baselines import (
    CENTER_CROPS,
    ConcreteMask2d,
    StaticMaskLayer2d,
    cae_selected_inds,
    center_mask,
    evaluate_model,
    fit_cae_selector,
    inds_to_mask,
    random_mask,
    train_static,
)
from patchdfs.gdfs.baselines import build_static_predictor
from patchdfs.gdfs.train import get_loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["random", "center", "cae"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="*", default=None, help="patch counts (random/cae)")
    ap.add_argument("--amp", default="bfloat16")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)
    device = torch.device("cuda")
    amp_dtype = {"bfloat16": torch.bfloat16, "float32": None, "none": None}[args.amp]
    cfg = yaml.safe_load(open("configs/gdfs/cifar10.yaml"))
    if args.data_root:
        cfg["data_root"] = args.data_root
    train_loader, val_loader, test_loader, _, d_out = get_loaders(cfg, smoke=False)
    log = RunLogger(args.run_dir, {"method": args.method, "seed": args.seed, "amp": args.amp, "budgets": args.budgets})
    rng = np.random.RandomState(args.seed)
    results = {}
    t_all = time.time()
    if args.method == "center":
        jobs = [(h * w, center_mask(8, w, h)) for h, w in CENTER_CROPS]
    else:
        budgets = args.budgets or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
        jobs = [(n, None) for n in budgets]
    for num, mask in jobs:
        t0 = time.time()
        if args.method == "random":
            mask = random_mask(8, num, rng)
        elif args.method == "cae":
            selector = ConcreteMask2d(width=8, patch_size=4, num_select=num).to(device)
            model = build_static_predictor().to(device)
            info = fit_cae_selector(model, selector, train_loader, val_loader, device=device, amp_dtype=amp_dtype, log=log)
            mask = inds_to_mask(cae_selected_inds(selector, num))
            log.log("cae_done", num=num, best_val_hard_loss=info["best_val_hard_loss"], epochs=len(info["history"]))
            del model, selector
        mask_layer = StaticMaskLayer2d(torch.from_numpy(mask), patch_size=4)
        model, hist = train_static(mask_layer, train_loader, val_loader, device=device, amp_dtype=amp_dtype, log=log, tag=f"static{num}")
        acc = evaluate_model(model, test_loader, device, amp_dtype)
        results[int(num)] = {"test_acc": acc, "epochs": len(hist), "sec": round(time.time() - t0, 1)}
        print(f"[{args.method}] num={num} test_acc={acc:.4f} ({results[int(num)]['sec']}s)", flush=True)
        log.log("budget_done", method=args.method, num=int(num), test_acc=acc, sec=results[int(num)]["sec"])
        del model
        torch.cuda.empty_cache()
    print(log.summary({"method": args.method, "seed": args.seed, "results": results, "total_hours": round((time.time() - t_all) / 3600, 3)}))


if __name__ == "__main__":
    main()