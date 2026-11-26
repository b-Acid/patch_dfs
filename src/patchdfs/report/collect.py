"""Build tables and figures from the summaries pulled into results/.

  python -m patchdfs.report.collect --results results --out results
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import paper_refs as P  # noqa: E402

C_OURS, C_PAPER, C_BASE = "#d62728", "#1f77b4", "#7f7f7f"


def _load(results, rel):
    path = os.path.join(results, rel, "summary.json")
    if os.path.exists(path):
        d = json.load(open(path))
    else:
        return None
    # a re-evaluation (fixed inference) supersedes the training-time numbers for DIME runs
    alt = os.path.join(results, rel, "summary_eval.json")
    if os.path.exists(alt):
        d["results"] = json.load(open(alt))["results"]
    return d


def _curve(d):
    xs = sorted(int(k) for k in d)
    return xs, [d[str(x)] for x in xs]


def _dime_curve(rows):
    xs = [r["avg_features"] for r in rows]
    ys = [r["acc"] for r in rows]
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


def table_rows(name, ours: dict | None, paper: dict | None, budgets) -> list[str]:
    rows = []
    for b in budgets:
        o = f"{ours[str(b)]:.3f}" if ours and str(b) in ours else "-"
        p = f"{paper[b]:.3f}" if paper and b in paper else "-"
        d = f"{ours[str(b)] - paper[b]:+.3f}" if (ours and paper and str(b) in ours and b in paper) else "-"
        rows.append(f"| {b} | {o} | {p} | {d} |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    R, O = args.results, args.out
    os.makedirs(O, exist_ok=True)
    md = ["# 结果表（由 collect.py 自动生成）", ""]
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # ---------------- GDFS MNIST ----------------
    g_m = _load(R, "gdfs/mnist/seed0")
    if g_m:
        md += ["## GDFS · MNIST（top-1，单种子）", "", "| 像素预算 | 复现 | 论文 Table 3 | 差值 |", "| --- | --- | --- | --- |"]
        md += table_rows("gdfs_mnist", g_m["test_acc"], P.GDFS_MNIST_GREEDY, [5, 10, 15, 20, 25, 30, 40, 50]) + [""]
        xs, ys = _curve(g_m["test_acc"])
        plt.figure(figsize=(4.2, 3))
        plt.plot(xs, ys, "-o", ms=3, color=C_OURS, label="GDFS (reproduction)")
        px, py = _curve({str(k): v for k, v in P.GDFS_MNIST_GREEDY.items()})
        plt.plot(px, py, "s--", ms=4, color=C_PAPER, label="GDFS (paper, Table 3)")
        plt.xscale("log")
        plt.xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
        plt.xlabel("# selected pixels")
        plt.ylabel("top-1 accuracy")
        plt.title("GDFS on MNIST")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(O, "fig_gdfs_mnist.png"))
        plt.close()

    # ---------------- GDFS CIFAR-10 ----------------
    g_c = _load(R, "gdfs/cifar10/seed0")
    if g_c:
        md += ["## GDFS · CIFAR-10（top-1，单种子）", "", "| 块预算 | 复现 greedy | 论文 greedy | 差值 | CAE 复现 | CAE 论文 | 随机 复现 | 随机 论文 | 中心 复现 | 中心 论文 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        rnd = _load(R, "gdfs/cifar10_baselines/random")
        cen = _load(R, "gdfs/cifar10_baselines/center")
        cae = _load(R, "gdfs/cifar10_baselines/cae")
        def acc(d, b):
            if not d:
                return "-"
            if "test_acc" in d:
                return f"{d['test_acc'][str(b)]:.3f}" if str(b) in d["test_acc"] else "-"
            return f"{d['results'][str(b)]['test_acc']:.3f}" if str(b) in d["results"] else "-"
        for b in [1, 3, 5, 7, 10, 15, 20, 30]:
            md.append(f"| {b} | {acc(g_c, b)} | {P.GDFS_CIFAR['greedy'].get(b, '-')} | - | {acc(cae, b)} | {P.GDFS_CIFAR['cae'].get(b, '-')} | {acc(rnd, b)} | {P.GDFS_CIFAR['random'].get(b, '-')} | {acc(cen, b)} | {P.GDFS_CIFAR['center'].get(b, '-')} |")
        md.append("")
        plt.figure(figsize=(4.6, 3.2))
        xs, ys = _curve(g_c["test_acc"])
        plt.plot(xs, ys, "-o", ms=3, color=C_OURS, label="GDFS (reproduction)")
        for name, d, c in [("cae", cae, "#2ca02c"), ("random", rnd, "#9467bd"), ("center", cen, "#ff7f0e")]:
            if d:
                xs, ys = _curve({k: v["test_acc"] for k, v in d["results"].items()})
                plt.plot(xs, ys, "-^", ms=3, color=c, label=f"{name} (reproduction)")
        for name, c in [("greedy", C_PAPER), ("cae", "#2ca02c"), ("random", "#9467bd"), ("center", "#ff7f0e")]:
            xs, ys = _curve({str(k): v for k, v in P.GDFS_CIFAR[name].items()})
            plt.plot(xs, ys, "s--", ms=3, color=c, alpha=0.6, label=f"{name} (paper)")
        plt.xlabel("# selected patches (of 64)")
        plt.ylabel("top-1 accuracy")
        plt.title("GDFS on CIFAR-10")
        plt.legend(fontsize=6, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(O, "fig_gdfs_cifar10.png"))
        plt.close()

    # ---------------- DIME ----------------
    for name, title, ref in [("dime/mnist/seed0", "DIME · MNIST", None), ("dime/imagenette/seed0", "DIME · Imagenette (dense ViT-S/16)", P.DIME_IMAGENETTE_VIT)]:
        d = _load(R, name)
        if not d:
            continue
        md += [f"## {title}（单种子）", "", "| 准则 | 参数 | 平均块数 | top-1 |", "| --- | --- | --- | --- |"]
        for mode, rows in d["results"].items():
            for r in rows:
                md.append(f"| {mode} | {r['param']} | {r['avg_features']:.2f} | {r['acc']:.4f} |")
        md.append("")
        plt.figure(figsize=(4.6, 3.2))
        for mode, c in [("budget", "#1f77b4"), ("penalized", "#d62728"), ("confidence", "#2ca02c")]:
            xs, ys = _dime_curve(d["results"][mode])
            plt.plot(xs, ys, "-o", ms=3, color=c, label=f"DIME {mode} (reproduction)")
        if ref:
            for mode, c in [("budget", "#1f77b4"), ("penalized", "#d62728")]:
                xs, ys = _curve({str(k): v for k, v in ref[mode].items()})
                plt.plot(xs, ys, "s--", ms=4, color=c, alpha=0.6, label=f"DIME {mode} (paper)")
        plt.xlabel("avg # selected patches" if "imagenette" in name else "avg # selected pixels")
        plt.xscale("log" if "mnist" in name else "linear")
        plt.ylabel("top-1 accuracy")
        plt.title(title)
        plt.legend(fontsize=7)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(O, f"fig_{name.replace('/', '_')}.png"))
        plt.close()

    # ---------------- DIME token-dropped vs dense ----------------
    d_dense = _load(R, "dime/imagenette/seed0")
    d_tok = _load(R, "dime/imagenette_token/seed0")
    if d_dense and d_tok:
        md += ["## DIME · Imagenette：稠密 vs 丢 token（同一预算下的 top-1）", "", "| 预算 | 稠密 | 丢 token | 差值 |", "| --- | --- | --- | --- |"]
        bd = {r["param"]: r["acc"] for r in d_dense["results"]["budget"]}
        bt = {r["param"]: r["acc"] for r in d_tok["results"]["budget"]}
        for b in sorted(set(bd) & set(bt)):
            md.append(f"| {b} | {bd[b]:.4f} | {bt[b]:.4f} | {bt[b] - bd[b]:+.4f} |")
        md.append("")
        plt.figure(figsize=(4.6, 3.2))
        for mode, c in [("budget", "#1f77b4"), ("penalized", "#d62728")]:
            xs, ys = _dime_curve(d_dense["results"][mode])
            plt.plot(xs, ys, "-o", ms=3, color=c, label=f"dense {mode}")
            xs, ys = _dime_curve(d_tok["results"][mode])
            plt.plot(xs, ys, "--^", ms=3, color=c, label=f"token-dropped {mode}")
        plt.xlabel("avg # selected patches")
        plt.ylabel("top-1 accuracy")
        plt.title("DIME on Imagenette: dense ViT vs token-dropped ViT")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(O, "fig_dime_token_vs_dense.png"))
        plt.close()

    # ---------------- compute ledger ----------------
    import glob as _glob

    for led_path in sorted(_glob.glob(os.path.join(R, "compute_ledger", "ledger_b*.json"))):
        b = os.path.basename(led_path)[len("ledger_b") : -len(".json")]
        L = json.load(open(led_path))
        md += [f"## 算力账本（相对整图一次前向；batch {b}，{L['device']}）", "", "| 模型 | 预算 k | 作者实现 FLOPs 倍数 | 作者实现 延迟倍数 | 融合 FLOPs 倍数 | 丢 token FLOPs 倍数 |", "| --- | --- | --- | --- | --- | --- |"]
        g = L.get("gdfs_cifar10") or L.get("gdfs_mnist")
        dv = L.get("dime_imagenette_vit")
        dt = L.get("dime_imagenette_token")
        ref_f = g["dense_reference"]["gflops"] if g else None
        ref_ms = g["dense_reference"]["ms"] if g else None
        for k in [1, 5, 10, 15, 20, 30]:
            row = [f"gdfs", str(k)]
            if g and str(k) in g["per_budget"]:
                a = g["per_budget"][str(k)]["authors"]
                row += [f"{a['gflops'] / ref_f:.1f}x", f"{a['ms'] / ref_ms:.1f}x"]
            else:
                row += ["-", "-"]
            if dv and str(k) in dv["per_budget"]:
                row += [f"{dv['per_budget'][str(k)]['fused']['gflops'] / dv['dense_reference']['gflops']:.1f}x"]
            else:
                row += ["-"]
            if dt and str(k) in dt["per_budget"]:
                row += [f"{dt['per_budget'][str(k)]['gflops'] / dt['dense_reference']['gflops']:.1f}x"]
            else:
                row += ["-"]
            md.append("| " + " | ".join(row) + " |")
        md.append("")
        plt.figure(figsize=(4.6, 3.2))
        ks = [int(k) for k in g["per_budget"]] if g else []
        if g:
            plt.plot(ks, [g["per_budget"][str(k)]["authors"]["gflops"] / ref_f for k in ks], "-o", ms=3, color="#d62728", label="GDFS CIFAR-10 (authors' code)")
        if dv:
            ks2 = [int(k) for k in dv["per_budget"]]
            plt.plot(ks2, [dv["per_budget"][str(k)]["authors_unfused"]["gflops"] / dv["dense_reference"]["gflops"] for k in ks2], "-o", ms=3, color="#1f77b4", label="DIME ViT (authors' code, 2 passes/step)")
            plt.plot(ks2, [dv["per_budget"][str(k)]["fused"]["gflops"] / dv["dense_reference"]["gflops"] for k in ks2], "--", color="#1f77b4", label="DIME ViT (fused heads)")
        if dt:
            ks3 = [int(k) for k in dt["per_budget"]]
            plt.plot(ks3, [dt["per_budget"][str(k)]["gflops"] / dt["dense_reference"]["gflops"] for k in ks3], "-^", ms=3, color="#2ca02c", label="DIME ViT (token-dropped)")
        plt.axhline(1.0, color="k", lw=0.8, ls=":")
        plt.text(1, 1.05, "one dense forward pass", fontsize=7)
        plt.yscale("log")
        plt.xlabel("budget k (selected patches)")
        plt.ylabel("FLOPs relative to one dense forward")
        plt.title(f"Compute multiplier at inference (batch {b})")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(O, f"fig_compute_multiplier_b{b}.png"))
        plt.close()

    # ---------------- sparse-conv bound ----------------
    sc_path = os.path.join(R, "sparse_conv_bound.json")
    if os.path.exists(sc_path):
        S = json.load(open(sc_path))
        md += ["## 稀疏卷积上界（GDFS CIFAR-10，真实选择，零开销理想核）", "", "| 块预算 | 选择器前向可省 | 预测器前向可省 | GDFS 推理整体可省 |", "| --- | --- | --- | --- |"]
        for k, v in S["per_budget"].items():
            md.append(f"| {k} | {v['selector_pass_saving']:.1%} | {v['predictor_pass_saving']:.1%} | {v['gdfs_inference_saving']:.1%} |")
        md.append("")
        plt.figure(figsize=(4.6, 3))
        ks = [int(k) for k in S["per_budget"]]
        plt.plot(ks, [S["per_budget"][str(k)]["predictor_pass_saving"] for k in ks], "-o", ms=3, label="predictor pass (backbone only)")
        plt.plot(ks, [S["per_budget"][str(k)]["selector_pass_saving"] for k in ks], "-^", ms=3, label="selector pass (backbone + upsampling head)")
        plt.xscale("log")
        plt.xlabel("observed patches k (of 64)")
        plt.ylabel("max fraction of MACs skippable")
        plt.title("Sparse-convolution upper bound on CIFAR-10 (4x4 patches)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(O, "fig_sparse_conv_bound.png"))
        plt.close()

    open(os.path.join(O, "tables.md"), "w", encoding="utf-8").write("\n".join(md))
    print("wrote", os.path.join(O, "tables.md"))


if __name__ == "__main__":
    main()
