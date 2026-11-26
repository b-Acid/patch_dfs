"""Evaluate a trained DIME model with the three stopping criteria (used to re-evaluate checkpoints).

  python -m patchdfs.dime.eval --config configs/dime/mnist.yaml --ckpt artifacts/dime/mnist/seed0/final.pt --run-dir artifacts/dime/mnist/seed0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml

from patchdfs.common.io import RunLogger
from patchdfs.common.seeding import seed_everything
from patchdfs.dime.cmi import infer
from patchdfs.dime.model import build_dime
from patchdfs.dime.train import acc_of, evaluate_all, get_loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--lams", default=None, help="comma-separated override of the penalised lambda grid")
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--multi", type=int, default=1, help="patches selected per step (budget mode)")
    ap.add_argument("--confidences", default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.lams:
        cfg["eval"]["lams"] = [float(v) for v in args.lams.split(",")]
        cfg["eval"]["budgets"] = []
        cfg["eval"]["confidences"] = []
    if args.budgets:
        cfg["eval"]["budgets"] = [int(v) for v in args.budgets.split(",")]
    seed_everything(args.seed)
    device = torch.device("cuda")
    amp = cfg.get("amp") == "bfloat16"
    fused = cfg["model"]["kind"] in ("vit", "vit_token_dropped")
    train_loader, val_loader, test_loader, d_in, d_out = get_loaders(cfg, smoke=False)
    net = build_dime(cfg["model"], d_in, d_out).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    if args.multi > 1:
        results = {"budget": [], "penalized": [], "confidence": []}
        for b in cfg["eval"]["budgets"]:
            r = infer(net, test_loader, device, amp=amp, fused=fused, budget=b, multi=args.multi)
            results["budget"].append({"param": b, "avg_features": float(r["n_features"].mean()), "acc": acc_of(r["pred"], r["y"]), "passes": r["passes"], "multi": args.multi})
            print(f"multi={args.multi} budget={b}: acc={results['budget'][-1]['acc']:.4f}", flush=True)
    else:
        results = evaluate_all(net, test_loader, cfg["eval"], d_in, device, amp, fused, args.run_dir)
    out = os.path.join(args.run_dir, f"summary_{args.tag}.json")
    import json

    json.dump({"config": cfg, "ckpt": args.ckpt, "results": results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out, "w"), indent=1)
    for mode, rows in results.items():
        print(f"== {mode}")
        for r in rows:
            print(f"  param={r['param']}: avg_features={r['avg_features']:.2f} acc={r['acc']:.4f}")


if __name__ == "__main__":
    main()