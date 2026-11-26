"""Train and evaluate the single-pass selection ViT (OnePassViT) on Imagenette.

  python -m patchdfs.sparse.onepass_train --config configs/sparse/imagenette_onepass.yaml --seed 0 --run-dir artifacts/sparse/imagenette_onepass/seed0 --glance-layers 2

Training: cross-entropy on the labels + DeiT-style hard distillation from the dense DIME predictor
of configs/dime/imagenette.yaml evaluated on the full image. k is drawn per batch from a range so
that ONE model serves every budget (a single value trains a budget-specific model). Evaluation:
one forward per image per budget on the test set; the kept indices are saved for figures.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from patchdfs.common.io import RunLogger
from patchdfs.common.seeding import seed_everything
from patchdfs.dime.model import build_dime
from patchdfs.dime.train import get_loaders
from patchdfs.models import create_vit
from .onepass import OnePassViT


def _autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


class FullImageTeacher(torch.nn.Module):
    """The dense DIME predictor on the unmasked image, with the OnePassViT calling convention."""

    def __init__(self, net: torch.nn.Module, n: int):
        super().__init__()
        self.net = net
        self.n = n

    def forward(self, x, k):  # noqa: ARG002 (same signature as OnePassViT.forward)
        return self.net.predict(x, torch.ones(x.shape[0], self.n, device=x.device)), None, None


@torch.no_grad()
def evaluate(net, loader, budgets, device, amp: bool, collect: bool = False):
    """Accuracy per budget (each budget = one forward pass per image)."""
    if isinstance(net, torch.nn.Module):
        net.eval()
    correct = {b: 0 for b in budgets}
    kept = {b: [] for b in budgets}
    mass = {b: 0.0 for b in budgets}
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        total += y.numel()
        with _autocast(amp):
            for b in budgets:
                logits, idx, score = net(x, b)
                correct[b] += int((logits.float().argmax(1) == y).sum())
                if score is not None:
                    mass[b] += float(score.gather(1, idx).sum(1).sum())
                    if collect:
                        kept[b].append(idx.cpu())
    acc = {int(b): correct[b] / total for b in budgets}
    out = {"acc": acc}
    if kept[budgets[0]]:
        out["attn_mass"] = {int(b): mass[b] / total for b in budgets}
        if collect:
            out["kept"] = {int(b): torch.cat(kept[b]).numpy() for b in budgets}
    return out


def parse_ints(s: str) -> list[int]:
    return [int(v) for v in str(s).split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--glance-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--k-train", default="1,30", help="inclusive range for the per-batch k; one value = fixed k")
    ap.add_argument("--val-budgets", default="5,10,20")
    ap.add_argument("--eval-budgets", default="1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100,150,196")
    ap.add_argument("--no-fuse-dropped", action="store_true")
    ap.add_argument("--teacher", default=None, help="dense DIME checkpoint; default artifacts/dime/<dataset>/seed<seed>/final.pt; 'none' disables distillation")
    ap.add_argument("--eval-only", default=None, help="checkpoint path: skip training and only run the budget evaluation")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.smoke:
        cfg["num_workers"] = 0
        args.epochs, args.patience = 1, 1
        args.val_budgets, args.eval_budgets = "2,4", "1,2,4"
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)
    device = torch.device("cuda")
    amp = cfg.get("amp") == "bfloat16"
    run_cfg = {"config": cfg, "seed": args.seed, "smoke": args.smoke, **{k: v for k, v in vars(args).items() if k not in ("config", "seed", "smoke")}}
    log = RunLogger(args.run_dir, run_cfg)

    train_loader, val_loader, test_loader, d_in, d_out = get_loaders(cfg, args.smoke)
    N = d_in
    ks = parse_ints(args.k_train)
    kmin, kmax = ks[0], ks[-1]
    val_budgets = [b for b in parse_ints(args.val_budgets) if b <= N]
    eval_budgets = [b for b in parse_ints(args.eval_budgets) if b <= N]

    # Teacher: the dense DIME predictor on the full image (same backbone family and preprocessing).
    art_root = os.environ.get("PATCHDFS_ARTIFACTS", "artifacts")
    tpath = args.teacher or os.path.join(art_root, "dime", cfg["dataset"], f"seed{args.seed}", "final.pt")
    teacher = None
    if args.teacher != "none" and os.path.exists(tpath):
        tnet = build_dime({**cfg["model"], "kind": "vit", "pretrained": False}, d_in, d_out).to(device)
        tnet.load_state_dict(torch.load(tpath, map_location=device))
        tnet.eval()
        for p in tnet.parameters():
            p.requires_grad_(False)
        teacher = FullImageTeacher(tnet, N)
    log.log("start", teacher=tpath if teacher else None, k_train=[kmin, kmax], n=N, device=torch.cuda.get_device_name(device))

    net = OnePassViT(
        create_vit(cfg["model"]["name"], pretrained=bool(cfg["model"].get("pretrained", True))), d_out,
        glance_layers=args.glance_layers, fuse_dropped=not args.no_fuse_dropped,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    if args.eval_only:
        net.load_state_dict(torch.load(args.eval_only, map_location=device))
        args.epochs = 0
        log.log("eval_only", ckpt=args.eval_only)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    best = {"crit": -1.0, "sd": None, "epoch": None}
    bad = 0
    history = []
    t_fit = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        net.train()
        ep = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "correct": 0, "n": 0, "mass": 0.0, "k_sum": 0}
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            k = random.randint(kmin, kmax) if kmin != kmax else kmin
            with torch.no_grad(), _autocast(amp):
                t_logits = teacher(x, N)[0].float() if teacher else None
            with _autocast(amp):
                logits, idx, score = net(x, k)
            logits = logits.float()
            ce = F.cross_entropy(logits, y)
            kd = F.cross_entropy(logits, t_logits.argmax(1)) if t_logits is not None else torch.zeros((), device=device)
            loss = ce if t_logits is None else 0.5 * ce + 0.5 * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            bs = y.numel()
            ep["loss"] += float(loss) * bs
            ep["ce"] += float(ce) * bs
            ep["kd"] += float(kd) * bs
            ep["correct"] += int((logits.argmax(1) == y).sum())
            ep["mass"] += float(score.gather(1, idx).sum(1).sum())
            ep["k_sum"] += k * bs
            ep["n"] += bs
        va = evaluate(net, val_loader, val_budgets, device, amp)
        crit = float(np.mean([va["acc"][b] for b in val_budgets]))
        if crit > best["crit"]:
            best = {"crit": crit, "sd": {k: v.detach().clone() for k, v in net.state_dict().items()}, "epoch": epoch + 1}
            torch.save(best["sd"], os.path.join(args.run_dir, "best.pt"))
            bad = 0
        else:
            bad += 1
        n = ep["n"]
        rec = {
            "epoch": epoch + 1, "sec": round(time.time() - t0, 1), "lr": opt.param_groups[0]["lr"],
            "train_loss": ep["loss"] / n, "train_ce": ep["ce"] / n, "train_kd": ep["kd"] / n,
            "train_acc": ep["correct"] / n, "train_attn_mass": ep["mass"] / n, "train_k_mean": ep["k_sum"] / n,
            **{f"val_acc@{b}": va["acc"][b] for b in val_budgets}, "val_crit": crit, "best_epoch": best["epoch"],
        }
        history.append(rec)
        log.log("fit", **rec)
        if bad > args.patience:
            break
    fit_sec = time.time() - t_fit
    if args.eval_only:
        pass  # keep the evaluated weights
    else:
        torch.save(net.state_dict(), os.path.join(args.run_dir, "final.pt"))
        net.load_state_dict(best["sd"])

    te = evaluate(net, test_loader, eval_budgets, device, amp, collect=True)
    np.savez_compressed(os.path.join(args.run_dir, "test_kept.npz"), **{f"kept_{b}": te["kept"][b] for b in eval_budgets})
    teacher_full = evaluate(teacher, test_loader, [N], device, amp)["acc"][N] if teacher else None
    summary = {
        "run": run_cfg, "n_params": n_params, "best_epoch": best["epoch"], "best_val_crit": best["crit"],
        "epochs_run": len(history), "fit_seconds": round(fit_sec, 1),
        "total_gpu_hours": round((time.time() - log.t0) / 3600, 3),
        "teacher": {"path": tpath, "test_acc_full": teacher_full},
        "results": {"budget": [
            {"budget": b, "acc": te["acc"][b], "tokens_focus": net.token_count(b), "attn_mass": te["attn_mass"][b]} for b in eval_budgets
        ]},
        "fit_history": history,
    }
    print(log.summary(summary))


if __name__ == "__main__":
    main()
