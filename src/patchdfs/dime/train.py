"""Train DIME on MNIST or Imagenette with the authors' settings, then evaluate the three stopping
criteria on the test set.

  python -m patchdfs.dime.train --config configs/dime/mnist.yaml --seed 0 --run-dir artifacts/dime/mnist/seed0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from patchdfs.common.io import RunLogger
from patchdfs.common.seeding import seed_everything, seed_worker
from patchdfs.data import get_imagenette, get_mnist
from patchdfs.gdfs.pretrain import pretrain_predictor
from .cmi import fit_dime, infer
from .model import build_dime


def get_loaders(cfg, smoke: bool):
    root = cfg["data_root"]
    name = cfg["dataset"]
    if name == "mnist":
        train, val, test = get_mnist(root)
        d_in, d_out = 784, 10
    elif name == "imagenette":
        train, val, test = get_imagenette(root)
        d_in, d_out = 196, 10
    else:
        raise ValueError(name)
    if smoke:
        train = Subset(train, range(64))
        val = Subset(val, range(64))
        test = Subset(test, range(64))
        cfg["batch_size"] = 16
        cfg["eval_batch_size"] = 16
    kw = dict(
        num_workers=cfg["num_workers"], pin_memory=True, worker_init_fn=seed_worker,
        persistent_workers=cfg["num_workers"] > 0,
    )
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val, batch_size=cfg["eval_batch_size"], shuffle=False, **kw)
    test_loader = DataLoader(test, batch_size=cfg["eval_batch_size"], shuffle=False, **kw)
    return train_loader, val_loader, test_loader, d_in, d_out


def acc_of(pred, y):
    return float((pred.argmax(1) == y).mean())


def evaluate_all(net, test_loader, ev: dict, d_in: int, device, amp: bool, fused: bool, run_dir: str) -> dict:
    """Run budget / penalised / confidence inference on the test set; save masks; return rows."""
    results = {"budget": [], "penalized": [], "confidence": []}
    saved: dict = {}
    for b in ev["budgets"]:
        r = infer(net, test_loader, device, amp=amp, fused=fused, budget=b)
        results["budget"].append({"param": b, "avg_features": float(r["n_features"].mean()), "acc": acc_of(r["pred"], r["y"]), "passes": r["passes"]})
        saved[f"mask_budget{b}"] = r["mask"].astype(np.uint8)
        saved["y"] = r["y"]
    for lam in ev["lams"]:
        r = infer(net, test_loader, device, amp=amp, fused=fused, lam=float(lam))
        results["penalized"].append({"param": float(lam), "avg_features": float(r["n_features"].mean()), "acc": acc_of(r["pred"], r["y"]), "passes": r["passes"], "n_features": np.bincount(r["n_features"].astype(int), minlength=d_in + 1).tolist() if len(r["n_features"]) else []})
        saved[f"mask_lam{lam}"] = r["mask"].astype(np.uint8)
    for c in ev["confidences"]:
        r = infer(net, test_loader, device, amp=amp, fused=fused, confidence_level=float(c))
        results["confidence"].append({"param": float(c), "avg_features": float(r["n_features"].mean()), "acc": acc_of(r["pred"], r["y"]), "passes": r["passes"]})
    np.savez_compressed(os.path.join(run_dir, "test_masks.npz"), **saved)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--no-fuse", action="store_true", help="use the authors' two separate backbone passes")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.smoke:
        cfg["num_workers"] = 0
        cfg["max_features"] = 4
        cfg["pretrain"].update(nepochs=1, patience=1, max_steps=2)
        cfg["fit"].update(max_epochs=1, patience=1, max_steps=2)
        cfg["eval"] = {"budgets": [1, 2], "lams": [0.01, 0.1], "confidences": [0.5]}
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)
    device = torch.device("cuda")
    log = RunLogger(args.run_dir, {"config": cfg, "seed": args.seed, "smoke": args.smoke, "fuse": not args.no_fuse})
    amp = cfg.get("amp") == "bfloat16"
    fused = (cfg["model"]["kind"] in ("vit", "vit_token_dropped")) and not args.no_fuse

    train_loader, val_loader, test_loader, d_in, d_out = get_loaders(cfg, args.smoke)
    net = build_dime(cfg["model"], d_in, d_out).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log.log("start", d_in=d_in, n_params=n_params, device=torch.cuda.get_device_name(device), fused=fused)

    # 1) Masking pretraining of the predictor. The authors' Lightning pretrainer has no checkpoint
    # callback, so the last epoch's weights are kept (restore_best=False).
    t0 = time.time()
    pre_hist = pretrain_predictor(
        net.predictor_module(), lambda x, m: net.predict(x, m), net.mask_size, train_loader, val_loader,
        lr=cfg["pretrain"]["lr"], nepochs=cfg["pretrain"]["nepochs"], patience=cfg["pretrain"]["patience"],
        min_lr=cfg["pretrain"].get("min_lr", 1e-6), device=device, amp_dtype=torch.bfloat16 if amp else None,
        log=log, tag="pretrain", max_steps_per_epoch=cfg["pretrain"].get("max_steps"), restore_best=False,
    )
    log.log("pretrain_done", best_val_loss=min(h["val_loss"] for h in pre_hist), epochs=len(pre_hist), sec=round(time.time() - t0, 1))
    torch.save(net.state_dict(), os.path.join(args.run_dir, "pretrained.pt"))

    # 2) Joint training of the value network and the predictor.
    t0 = time.time()
    f = cfg["fit"]
    info = fit_dime(
        net, train_loader, val_loader, max_features=cfg["max_features"], lr=f["lr"], min_lr=f["min_lr"],
        eps0=f["eps"], eps_decay=f["eps_decay"], eps_steps=f["eps_steps"], patience=f["patience"],
        max_epochs=f["max_epochs"], device=device, amp=amp, fused=fused, log=log,
        max_steps_per_epoch=f.get("max_steps"), graphed=bool(f.get("cuda_graph", False)),
        ckpt_path=os.path.join(args.run_dir, "best.pt"),
    )
    fit_sec = time.time() - t0
    torch.save(net.state_dict(), os.path.join(args.run_dir, "final.pt"))

    # 3) Inference with the three stopping criteria.
    results = evaluate_all(net, test_loader, cfg["eval"], d_in, device, amp, fused, args.run_dir)
    summary = {
        "config": cfg, "seed": args.seed, "n_params": n_params, "fused": fused,
        "pretrain_epochs": len(pre_hist), "fit_epochs": info["epochs"], "best_val_acc_final": info["best_val_acc_final"],
        "best_epoch": info["best_epoch"], "fit_seconds": round(fit_sec, 1),
        "total_gpu_hours": round((time.time() - log.t0) / 3600, 3), "results": results, "fit_history": info["history"],
    }
    print(log.summary(summary))
    for mode, rows in results.items():
        print(f"== {mode}")
        for r in rows:
            print(f"  param={r['param']}: avg_features={r['avg_features']:.2f} acc={r['acc']:.4f}")


if __name__ == "__main__":
    main()