"""Train GDFS on MNIST or CIFAR-10 with the authors' settings.

  python -m patchdfs.gdfs.train --config configs/gdfs/mnist.yaml --seed 0 --run-dir artifacts/gdfs/mnist/seed0
Add --smoke for a tiny end-to-end pipeline check.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from patchdfs.common.io import RunLogger, save_json
from patchdfs.common.seeding import seed_everything, seed_worker
from patchdfs.data import get_cifar10, get_mnist
from .evaluate import evaluate_budgets
from .fit import fit_gdfs
from .model import build_gdfs
from .pretrain import pretrain_predictor


def get_loaders(cfg, smoke: bool):
    root = cfg["data_root"]
    name = cfg["dataset"]
    if name == "mnist":
        train, val, test = get_mnist(root)
        d_in, d_out = 784, 10
    elif name == "cifar10":
        train, val, test = get_cifar10(root)
        d_in, d_out = 64, 10
    else:
        raise ValueError(name)
    if smoke:
        train = Subset(train, range(256))
        val = Subset(val, range(256))
        test = Subset(test, range(256))
    kw = dict(
        num_workers=cfg["num_workers"], pin_memory=True, worker_init_fn=seed_worker,
        persistent_workers=cfg["num_workers"] > 0,
    )
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val, batch_size=cfg["eval_batch_size"], shuffle=False, **kw)
    test_loader = DataLoader(test, batch_size=cfg["eval_batch_size"], shuffle=False, **kw)
    return train_loader, val_loader, test_loader, d_in, d_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.smoke:
        cfg["num_workers"] = 0
        cfg["max_features"] = 4
        cfg["pretrain"].update(nepochs=1, patience=1)
        cfg["fit"].update(nepochs=1, early_stop_patience=1, max_steps=2)
        cfg["eval_budgets"] = [1, 2, 4]
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)
    device = torch.device("cuda")
    log = RunLogger(args.run_dir, {"config": cfg, "seed": args.seed, "smoke": args.smoke})

    train_loader, val_loader, test_loader, d_in, d_out = get_loaders(cfg, args.smoke)
    net = build_gdfs(cfg["model"], d_in, d_out).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log.log("start", d_in=d_in, d_out=d_out, n_params=n_params, device=torch.cuda.get_device_name(device))

    # 1) Pretrain the predictor with random masks (authors run this in fp32).
    t0 = time.time()
    pre_hist = pretrain_predictor(
        net.predictor_module(), lambda x, m: net.predict(x, m), net.mask_size, train_loader, val_loader,
        lr=cfg["pretrain"]["lr"], nepochs=cfg["pretrain"]["nepochs"], patience=cfg["pretrain"]["patience"],
        min_lr=cfg["pretrain"]["min_lr"], device=device, amp_dtype=None, log=log, tag="pretrain",
        max_steps_per_epoch=cfg["pretrain"].get("max_steps"),
    )
    torch.save(net.state_dict(), os.path.join(args.run_dir, "pretrained.pt"))
    log.log("pretrain_done", best_val_loss=min(h["val_loss"] for h in pre_hist), epochs=len(pre_hist), sec=round(time.time() - t0, 1))

    # 2) Joint training over the temperature schedule.
    t0 = time.time()
    fcfg = cfg["fit"]
    temps = np.geomspace(fcfg["temps"]["start"], fcfg["temps"]["end"], int(fcfg["temps"]["steps"]))
    fit_info = fit_gdfs(
        net, train_loader, val_loader, max_features=cfg["max_features"], temps=temps,
        lr=fcfg["lr"], nepochs=fcfg["nepochs"], scheduler_patience=fcfg["scheduler_patience"],
        early_stop_patience=fcfg["early_stop_patience"], min_lr=fcfg["min_lr"], device=device,
        amp=cfg.get("amp") == "bfloat16", log=log, max_steps_per_epoch=fcfg.get("max_steps"),
        graphed=bool(fcfg.get("cuda_graph", False)), ckpt_path=os.path.join(args.run_dir, "best.pt"),
    )
    fit_sec = time.time() - t0
    torch.save(net.state_dict(), os.path.join(args.run_dir, "final.pt"))

    # 3) Evaluate on validation and test at every budget with greedy (argmax) selection.
    amp = cfg.get("amp") == "bfloat16"
    budgets = list(cfg["eval_budgets"])
    val_res = evaluate_budgets(net, val_loader, budgets, amp=amp, device=device)
    test_res = evaluate_budgets(net, test_loader, budgets, amp=amp, device=device)
    np.savez_compressed(
        os.path.join(args.run_dir, "test_selections.npz"),
        selections=test_res["selections"], preds=test_res["preds"], y=test_res["y"],
    )
    summary = {
        "config": cfg, "seed": args.seed, "n_params": n_params,
        "pretrain_epochs": len(pre_hist), "fit_epochs": len(fit_info["history"]),
        "best_val_hard_acc": fit_info["best_val_hard_acc"], "best_temp": fit_info["best_temp"],
        "fit_seconds": round(fit_sec, 1), "total_gpu_hours": round((time.time() - log.t0) / 3600, 3),
        "val_acc": {str(k): v for k, v in val_res["acc"].items()},
        "test_acc": {str(k): v for k, v in test_res["acc"].items()},
        "fit_history": fit_info["history"],
    }
    print(log.summary(summary))
    for b in budgets:
        print(f"budget {b:3d}: val {val_res['acc'][b]:.4f}  test {test_res['acc'][b]:.4f}")


if __name__ == "__main__":
    main()