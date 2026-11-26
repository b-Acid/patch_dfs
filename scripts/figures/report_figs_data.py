"""Redraw every data figure for the report in one consistent style.

    python3 scripts/figures/report_figs_data.py

Reads only local JSON under results/ (plus the paper readings in
src/patchdfs/report/paper_refs.py) and writes results/report_fig*.png.
CPU only: no GPU, no torch. No figure carries a suptitle: each one's finding
belongs in the caption.
"""
from __future__ import annotations

import importlib.util
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")

_spec = importlib.util.spec_from_file_location(
    "paper_refs", os.path.join(ROOT, "src", "patchdfs", "report", "paper_refs.py")
)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

# DIME MNIST budget-mode readings off the paper's Fig. 2 (not in paper_refs.py).
DIME_MNIST_BUDGET = {5: 0.60, 10: 0.82, 20: 0.92}

# ---------------------------------------------------------------- palette
C_SERIAL = "#4C72B0"    # authors' method / serial pipeline
C_TOKEN = "#2CA02C"     # serial, token dropped
C_GLANCE2 = "#D62728"   # one-pass, 2 glance layers
C_GLANCE4 = "#9467BD"   # one-pass, 4 glance layers
C_FULL = "#111111"      # dense ViT, full image
C_BASE = "#7F7F7F"      # static baselines
C_ALT = "#DD8452"       # second series where the palette has no semantic slot
C_PENAL = "#E07B39"     # DIME penalised control

LW, MS = 1.7, 4.5
WRITTEN = []

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.88,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "grid.alpha": 0.25,
    }
)


# ---------------------------------------------------------------- helpers
def _load(rel):
    """Same convention as patchdfs.report.collect._load."""
    path = os.path.join(RES, rel, "summary.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    alt = os.path.join(RES, rel, "summary_eval.json")
    if os.path.exists(alt):
        d["results"] = json.load(open(alt))["results"]
    return d


def _json(rel):
    return json.load(open(os.path.join(RES, rel)))


def _curve(d):
    xs = sorted(int(k) for k in d)
    return np.array(xs, float), np.array([d[str(x)] for x in xs], float)


def _dime(rows):
    xs = np.array([r["avg_features"] for r in rows], float)
    ys = np.array([r["acc"] for r in rows], float)
    order = np.argsort(xs)
    return xs[order], ys[order]


def style(ax, xlabel="", ylabel="", title="", logx=False, logy=False, ylim=None, xlim=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", alpha=0.25)
    ax.set_axisbelow(True)
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=6)


def save(fig, name):
    fig.savefig(os.path.join(RES, name), dpi=200)
    plt.close(fig)
    WRITTEN.append(name)


# ---------------------------------------------------------------- fig 05
def fig05():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))

    # (a) GDFS on MNIST -------------------------------------------------
    ax = axes[0, 0]
    xs, ys = _curve(_load("gdfs/mnist/seed0")["test_acc"])
    ax.plot(xs, ys, "-o", color=C_SERIAL, lw=LW, ms=MS, label="GDFS (ours)", zorder=4)
    px, py = _curve({str(k): v for k, v in P.GDFS_MNIST_GREEDY.items()})
    ax.plot(px, py, "--s", color=C_SERIAL, lw=LW, ms=MS, mfc="white",
            label="GDFS (paper)", zorder=5)
    style(ax, "selected pixels", "top-1 accuracy", "(a) GDFS, MNIST",
          logx=True, ylim=(0.25, 1.0))
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.legend(loc="lower right", framealpha=0.88)

    # (b) GDFS on CIFAR-10 with the three static baselines ---------------
    ax = axes[0, 1]
    # one colour per baseline; every budget is an independent single run for all three baselines
    base = {
        "center": ("Center crop", "D", "#8C564B", "-"),
        "cae": ("CAE", "s", "#17BECF", "-"),
        "random": ("Random mask", "^", "#7F7F7F", "-"),
    }
    for key, (name, mk, col, ls) in base.items():
        bx, by = _curve({k: v["test_acc"] for k, v in
                         _load("gdfs/cifar10_baselines/" + key)["results"].items() if int(k) <= 30})
        ax.plot(bx, by, linestyle=ls, marker=mk, color=col, lw=1.2, ms=3.5, alpha=0.85,
                label=name, zorder=3)
    xs, ys = _curve(_load("gdfs/cifar10/seed0")["test_acc"])
    ax.plot(xs, ys, "-o", color=C_SERIAL, lw=LW, ms=MS, label="GDFS (ours)", zorder=4)
    px, py = _curve({str(k): v for k, v in P.GDFS_CIFAR["greedy"].items()})
    ax.plot(px, py, "--s", color=C_SERIAL, lw=LW, ms=MS, mfc="white",
            label="GDFS (paper)", zorder=5)
    style(ax, "selected patches (of 64)", "top-1 accuracy", "(b) GDFS, CIFAR-10",
          logx=True, ylim=(0.1, 1.0), xlim=(0.85, 36))
    ax.set_xticks([1, 2, 5, 10, 20, 30])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "30"])
    ax.minorticks_off()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", ncol=1, framealpha=0.92,
              fontsize=8, handlelength=1.5, borderpad=0.35)

    # (c) DIME on MNIST, budget mode ------------------------------------
    ax = axes[1, 0]
    xs, ys = _dime(_load("dime/mnist/seed0")["results"]["budget"])
    ax.plot(xs, ys, "-o", color=C_SERIAL, lw=LW, ms=MS, label="DIME (ours, zeroed)", zorder=4)
    px = np.array(sorted(DIME_MNIST_BUDGET), float)
    py = np.array([DIME_MNIST_BUDGET[int(k)] for k in px])
    ax.plot(px, py, "--s", color=C_SERIAL, lw=LW, ms=MS, mfc="white",
            label="DIME (paper)", zorder=5)
    style(ax, "selected pixels", "top-1 accuracy", "(c) DIME, MNIST",
          logx=True, ylim=(0.25, 1.0))
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.legend(loc="lower right", framealpha=0.88)

    # (d) DIME on Imagenette, budget and penalised ----------------------
    ax = axes[1, 1]
    d = _load("dime/imagenette/seed0")
    ref = P.DIME_IMAGENETTE_VIT
    xs, ys = _dime(d["results"]["budget"])
    ax.plot(xs, ys, "-o", color=C_SERIAL, lw=LW, ms=MS, label="DIME (ours, zeroed), budget", zorder=4)
    px, py = _curve({str(k): v for k, v in ref["budget"].items()})
    ax.plot(px, py, "--s", color=C_SERIAL, lw=LW, ms=MS, mfc="white",
            label="DIME (paper), budget", zorder=5)
    xs, ys = _dime(d["results"]["penalized"])
    ax.plot(xs, ys, "-^", color=C_PENAL, lw=LW, ms=MS, label="DIME (ours, zeroed), penalised", zorder=4)
    px, py = _curve({str(k): v for k, v in ref["penalized"].items()})
    ax.plot(px, py, "--^", color=C_PENAL, lw=LW, ms=MS, mfc="white",
            label="DIME (paper), penalised", zorder=5)
    style(ax, "avg. selected patches", "top-1 accuracy", "(d) DIME, Imagenette",
          ylim=(0.25, 1.0), xlim=(0.5, 16.5))
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.legend(loc="lower right", framealpha=0.88)

    fig.tight_layout()
    save(fig, "report_fig05_reproduction.png")


# ---------------------------------------------------------------- fig 06
def _token_grid(pbt):
    d = {int(v["label"].split("_")[1]): v["ms"] for v in pbt.values()}
    toks = sorted(d)
    return np.array(toks, float), np.array([d[t] for t in toks])


def fig06():
    D = _json("latency_logs/master_5060.json")["token"]
    t = np.array([1.0, 16.0, 196.0])
    m1 = np.array([D["1"]["eager"][str(n)]["ms"] for n in (1, 16, 196)])
    m2 = np.array([D["256"]["eager"][str(n)]["ms"] for n in (1, 16, 196)])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, ms_, batch, top in [(axes[0], m1, 1, 9.0), (axes[1], m2, 256, 170.0)]:
        ax.plot(t, ms_, "-o", color=C_SERIAL, lw=LW, ms=MS, zorder=3)
        style(ax, "tokens fed to one forward pass", "latency of one forward pass (ms)",
              f"batch {batch}", logx=True, ylim=(0.0, top))
        ax.set_xticks([1, 16, 196])
        ax.set_xticklabels(["1", "16", "196"])
        ax.set_xlim(0.85, 240)

    fig.tight_layout()
    save(fig, "report_fig06_fixed_cost.png")

def fig06b():
    D = _json("latency_logs/master_5060.json")["token"]
    ct = np.array([1.0, 16.0, 196.0])
    m1 = np.array([D["1"]["eager"][str(n)]["ms"] for n in (1, 16, 196)])
    m2 = np.array([D["256"]["eager"][str(n)]["ms"] for n in (1, 16, 196)])
    c1 = np.array([D["1"]["reduce-overhead"][str(n)]["ms"] for n in (1, 16, 196)])
    c2 = np.array([D["256"]["reduce-overhead"][str(n)]["ms"] for n in (1, 16, 196)])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, eager, comp, batch, top in [
        (axes[0], m1, c1, 1, 9.0),
        (axes[1], m2, c2, 256, 170.0),
    ]:
        ax.plot(ct, eager, "-o", color=C_SERIAL, lw=LW, ms=MS, label="eager", zorder=3)
        ax.plot(ct, comp, "--s", color=C_ALT, lw=LW, ms=MS, mfc="white",
                label="compiled (reduce-overhead)", zorder=3)
        for x, y in zip(ct, comp):
            ax.annotate(f"{y:g}", (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7.5, color=C_ALT)
        style(ax, "tokens fed to one forward pass", "latency of one forward pass (ms)",
              f"batch {batch}", logx=True, ylim=(0.0, top))
        ax.set_xticks([1, 16, 196])
        ax.set_xticklabels(["1", "16", "196"])
        ax.set_xlim(0.85, 240)
        ax.legend(loc="upper left")

    fig.tight_layout()
    save(fig, "report_fig06b_fixed_cost_compiled.png")

def _ledger_lines():
    L1 = _json("compute_ledger/ledger_b1.json")
    dv, dt = L1["dime_imagenette_vit"], L1["dime_imagenette_token"]
    ref_v, ref_t = dv["dense_reference"]["gflops"], dt["dense_reference"]["gflops"]
    ks = np.array(sorted(int(k) for k in dv["per_budget"] if int(k) <= 30), float)
    out = {
        "ks": ks,
        "authors": np.array([dv["per_budget"][str(int(k))]["authors_unfused"]["gflops"] / ref_v for k in ks]),
        "fused": np.array([dv["per_budget"][str(int(k))]["fused"]["gflops"] / ref_v for k in ks]),
        "token": np.array([dt["per_budget"][str(int(k))]["gflops"] / ref_t for k in ks]),
    }
    f2 = _json("latency_logs/onepass_imagenette_onepass_g2_b1.json")["flops_ratio"]
    f4 = _json("latency_logs/onepass_imagenette_onepass_g4_b1.json")["flops_ratio"]
    out["g2"] = np.array([f2[str(int(k))] for k in ks])
    out["g4"] = np.array([f4[str(int(k))] for k in ks])
    return out


def _plot_flops(ax, L, title, methods):
    for key, color, ls, mk, label in methods:
        ax.plot(L["ks"], L[key], ls=ls, marker=mk, color=color, lw=LW, ms=MS,
                mfc="white" if ls == "--" else color, label=label)
    ax.axhline(1.0, color="k", lw=1.0, ls=":", zorder=1)
    ax.annotate("one dense forward", (1.0, 1.0), textcoords="offset points",
                xytext=(4, 5), fontsize=8, color="k")
    style(ax, "budget k (selected patches)", "FLOPs / one dense forward", title,
          logx=True, logy=True)
    ax.set_xticks([1, 5, 10, 20, 30])
    ax.set_xticklabels(["1", "5", "10", "20", "30"])


def _bars(ax, rows, bottom, top, xlabel, ylabel, title, label_size=8.5, rotate=0):
    x = np.arange(len(rows))
    b1 = np.array([r[1] for r in rows], float)
    b256 = np.array([r[2] for r in rows], float)
    ax.bar(x - 0.2, b1 - bottom, 0.38, bottom=bottom, color=C_SERIAL, label="batch 1")
    ax.bar(x + 0.2, b256 - bottom, 0.38, bottom=bottom, color=C_ALT, label="batch 256")
    for xi, v in zip(x - 0.2, b1):
        ax.annotate(f"{v:g}", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color=C_SERIAL)
    for xi, v in zip(x + 0.2, b256):
        ax.annotate(f"{v:g}", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color=C_ALT)
    style(ax, xlabel, ylabel, title, logy=True, ylim=(bottom, top))
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=rotate,
                       ha="right" if rotate else "center", fontsize=label_size)
    ax.legend(loc="upper right")


def fig07_paradox():
    L = _ledger_lines()
    D = _json("latency_logs/master_5060.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    _plot_flops(ax, L, "(a) FLOPs multiplier during inference", [
        ("authors", C_SERIAL, "-", "o", "DIME (authors' code)"),
        ("fused", C_SERIAL, "--", "s", "DIME (fused heads)"),
    ])
    ax.set_ylim(0.5, 90.0)
    ax.legend(loc="upper left")

    ax = axes[1]
    rows = [
        ("DIME (authors' code)", D["dense"]["1"]["authors_k10"], D["dense"]["256"]["authors_k10"]),
        ("DIME (fused heads)", D["dense"]["1"]["fused_k10"], D["dense"]["256"]["fused_k10"]),
        ("Dense ViT\n(one forward, eager)", D["token"]["1"]["eager"]["196"]["ms"], D["token"]["256"]["eager"]["196"]["ms"]),
    ]
    _bars(ax, rows, 1.0, 6000.0, "method", "end-to-end latency at k = 10 (ms)",
          "(b) Latency of one 10-patch prediction", label_size=9)

    fig.tight_layout()
    save(fig, "report_fig07_paradox.png")

def fig14_ledger_full():
    L = _ledger_lines()
    D = _json("latency_logs/master_5060.json")
    T, O = D["token"], D["onepass"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    _plot_flops(ax, L, "(a) FLOPs multiplier, all variants", [
        ("authors", C_SERIAL, "-", "o", "DIME (authors' code)"),
        ("fused", C_SERIAL, "--", "s", "DIME (fused heads)"),
        ("token", C_TOKEN, "-", "^", "DIME (token-dropped)"),
        ("g2", C_GLANCE2, "-", "D", "One-pass ViT (2 glance layers)"),
        ("g4", C_GLANCE4, "-", "v", "One-pass ViT (4 glance layers)"),
    ])
    ax.set_ylim(0.09, 200.0)
    ax.legend(loc="upper left", ncol=2, fontsize=8, handlelength=1.7,
              columnspacing=0.9, borderpad=0.4)

    ax = axes[1]
    rows = [
        ("DIME (authors' code)", D["dense"]["1"]["authors_k10"], D["dense"]["256"]["authors_k10"]),
        ("DIME (fused heads)", D["dense"]["1"]["fused_k10"], D["dense"]["256"]["fused_k10"]),
        ("DIME (token-dropped)", T["1"]["eager"]["total_k10"], T["256"]["eager"]["total_k10"]),
        ("DIME (token-dropped),\n+ torch.compile", T["1"]["reduce-overhead"]["total_k10"], T["256"]["reduce-overhead"]["total_k10"]),
        ("Dense ViT (full image)", T["1"]["reduce-overhead"]["196"]["ms"], T["256"]["reduce-overhead"]["196"]["ms"]),
        ("One-pass ViT\n(4 glance layers)", O["4"]["1"]["reduce-overhead"]["10"]["ms"], O["4"]["256"]["reduce-overhead"]["10"]["ms"]),
        ("One-pass ViT\n(2 glance layers)", O["2"]["1"]["reduce-overhead"]["10"]["ms"], O["2"]["256"]["reduce-overhead"]["10"]["ms"]),
    ]
    _bars(ax, rows, 0.3, 6000.0, "method (slowest to fastest)",
          "end-to-end latency at k = 10 (ms)",
          "(b) Latency for one 10-patch prediction", label_size=7.5, rotate=28)

    fig.tight_layout()
    save(fig, "report_fig14_ledger_full.png")

def fig08():
    S = _json("sparse_conv_bound.json")["per_budget"]
    ks = np.array(sorted(int(k) for k in S), float)
    sel = np.array([S[str(int(k))]["selector_pass_saving"] for k in ks])
    pred = np.array([S[str(int(k))]["predictor_pass_saving"] for k in ks])
    tot = np.array([S[str(int(k))]["gdfs_inference_saving"] for k in ks])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ks, pred, "-^", color=C_ALT, lw=LW, ms=MS, label="predictor pass (backbone only)", zorder=3)
    ax.plot(ks, sel, "--s", color=C_SERIAL, lw=LW, ms=MS,
            label="selector pass (backbone + head)", zorder=3)
    ax.plot(ks, tot, "-o", color=C_GLANCE4, lw=2.0, ms=MS, label="GDFS inference, end to end", zorder=4)
    i10 = int(np.where(ks == 10)[0][0])
    ax.annotate(
        f"{tot[i10]:.1%} saved at k = 10", (9.2, tot[i10]), xytext=(1.2, 0.135),
        textcoords="data", fontsize=9, color=C_GLANCE4, va="center",
        arrowprops=dict(arrowstyle="->", color=C_GLANCE4, lw=0.9),
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    style(ax, "observed patches k (of 64, log scale)", "max fraction of MACs skippable",
          "", logx=True, ylim=(0.0, 0.65))
    ax.set_xticks([1, 2, 5, 10, 20, 30])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "30"])
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "report_fig08_sparse_bound.png")


# ---------------------------------------------------------------- fig 10
def fig10():
    dense = _load("dime/imagenette/seed0")
    tok = _load("dime/imagenette_token/seed0")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, mode, xlabel, title in [
        (axes[0], "budget", "avg. selected patches (budget)", "(a) budget control"),
        (axes[1], "penalized", "avg. selected patches (penalised)", "(b) penalised control"),
    ]:
        xs, ys = _dime(dense["results"][mode])
        ax.plot(xs, ys, "-o", color=C_SERIAL, lw=LW, ms=MS, label="DIME (zeroed, dense ViT)", zorder=3)
        xs, ys = _dime(tok["results"][mode])
        ax.plot(xs, ys, "-^", color=C_TOKEN, lw=LW, ms=MS, label="DIME (token-dropped)", zorder=3)
        style(ax, xlabel, "top-1 accuracy", title, ylim=(0.3, 1.0))
        ax.legend(loc="lower right")
    axes[0].set_xticks([1, 5, 10, 15, 20, 25, 30])
    fig.tight_layout()
    save(fig, "report_fig10_token_vs_dense.png")


# ---------------------------------------------------------------- fig 11
def _onepass(glance):
    seeds = []
    for s in range(3):
        d = json.load(open(os.path.join(RES, "sparse", f"imagenette_onepass_g{glance}",
                                        f"seed{s}", "summary.json")))
        seeds.append({r["budget"]: r["acc"] for r in d["results"]["budget"]})
    budgets = sorted(set.intersection(*[set(s) for s in seeds]))
    mean = np.array([np.mean([s[b] for s in seeds]) for b in budgets])
    std = np.array([np.std([s[b] for s in seeds]) for b in budgets])
    return np.array(budgets, float), mean, std


def fig11():
    dense = _load("dime/imagenette/seed0")
    tok = _load("dime/imagenette_token/seed0")
    m2 = _json("dime_eval/imagenette_token_seed0_multi2.json")["results"]["budget"]
    m4 = _json("dime_eval/imagenette_token_seed0_multi4.json")["results"]["budget"]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    xs, ys = _dime(dense["results"]["budget"])
    ax.plot(xs, ys, "-o", color=C_SERIAL, lw=LW, ms=MS, label="DIME (zeroed, dense ViT)")
    xs, ys = _dime(tok["results"]["budget"])
    ax.plot(xs, ys, "-^", color=C_TOKEN, lw=LW, ms=MS, label="DIME (token-dropped)")

    xs, ys = _dime(m2)
    ax.plot(xs, ys, "--v", color=C_TOKEN, lw=LW, ms=MS, mfc="white",
            label="DIME (token-dropped, 2 per step)")
    xs, ys = _dime(m4)
    ax.plot(xs, ys, ":D", color=C_TOKEN, lw=LW, ms=MS,
            label="DIME (token-dropped, 4 per step)")

    for glance, color, mk, label in [
        (2, C_GLANCE2, "o", "One-pass ViT (2 glance layers)"),
        (4, C_GLANCE4, "s", "One-pass ViT (4 glance layers)"),
    ]:
        b, mean, std = _onepass(glance)
        ax.plot(b, mean, ls="-", marker=mk, color=color, lw=LW, ms=MS, label=label, zorder=3)
        ax.fill_between(b, mean - std, mean + std, color=color, alpha=0.20, lw=0, zorder=2)

    ax.axhline(0.9944, color="k", lw=1.1, ls=":", zorder=1)
    ax.annotate("Dense ViT (full image), 0.994", (1.0, 0.9944), textcoords="offset points",
                xytext=(2, -8), ha="left", va="top", fontsize=8.5, color="k")
    style(ax, "budget k (patches attended)", "test accuracy (top-1)", "",
          ylim=(0.30, 1.0), xlim=(0.5, 31.0))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="lower right", ncol=2, fontsize=8, columnspacing=0.9, borderpad=0.4)
    fig.tight_layout()
    save(fig, "report_fig11_accuracy_budget.png")


# ---------------------------------------------------------------- fig 12
_M = _json("latency_logs/master_5060.json")
_T, _O, _D = _M["token"], _M["onepass"], _M["dense"]
PTS = [
    # key, accuracy, batch-1 ms, batch-256 ms, colour, marker
    ("DIME (authors' code)", 0.892, _D["1"]["authors_k10"], _D["256"]["authors_k10"], C_SERIAL, "o"),
    ("DIME (fused heads)", 0.892, _D["1"]["fused_k10"], _D["256"]["fused_k10"], C_SERIAL, "s"),
    ("DIME (token-dropped)", 0.852, _T["1"]["eager"]["total_k10"], _T["256"]["eager"]["total_k10"], C_TOKEN, "o"),
    ("DIME (token-dropped), compiled", 0.852, _T["1"]["reduce-overhead"]["total_k10"], _T["256"]["reduce-overhead"]["total_k10"], C_TOKEN, "s"),
    ("Dense ViT (full image)", 0.994, _T["1"]["reduce-overhead"]["196"]["ms"], _T["256"]["reduce-overhead"]["196"]["ms"], C_FULL, "D"),
    ("One-pass ViT (4 glance layers)", 0.951, _O["4"]["1"]["reduce-overhead"]["10"]["ms"], _O["4"]["256"]["reduce-overhead"]["10"]["ms"], C_GLANCE4, "o"),
    ("One-pass ViT (2 glance layers)", 0.915, _O["2"]["1"]["reduce-overhead"]["10"]["ms"], _O["2"]["256"]["reduce-overhead"]["10"]["ms"], C_GLANCE2, "o"),
]

# hand-placed label anchors in data coordinates (panel, method) -> (x, y)
ANCHOR = {
    1: {
        "DIME (authors' code)": (250, 0.918),
        "DIME (fused heads)": (120, 0.866),
        "DIME (token-dropped)": (160, 0.833),
        "DIME (token-dropped), compiled": (18, 0.883),
        "Dense ViT (full image)": (3.2, 0.974),
        "One-pass ViT (4 glance layers)": (3.2, 0.946),
        "One-pass ViT (2 glance layers)": (3.2, 0.898),
    },
    256: {
        "DIME (authors' code)": (2600, 0.928),
        "DIME (fused heads)": (1450, 0.866),
        "DIME (token-dropped)": (500, 0.834),
        "DIME (token-dropped), compiled": (180, 0.878),
        "Dense ViT (full image)": (120, 0.972),
        "One-pass ViT (4 glance layers)": (70, 0.946),
        "One-pass ViT (2 glance layers)": (60, 0.888),
    },
}


def fig12():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax, batch in [(axes[0], 1), (axes[1], 256)]:
        for key, acc, b1, b256, color, mk in PTS:
            ms = b1 if batch == 1 else b256
            ax.plot(ms, acc, marker=mk, color=color, ms=5.5, ls="none", mfc=color,
                    mec="white", mew=0.6, zorder=4)
            anchor = ANCHOR[batch][key]
            ha = "left" if ms <= anchor[0] else "right"
            ax.annotate(
                key, (ms, acc), xytext=anchor, textcoords="data",
                ha=ha, va="center", fontsize=7.5, color=color, zorder=5,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.85,
                                shrinkA=2, shrinkB=4),
            )
        style(
            ax, "end-to-end latency at k = 10 (ms, log scale)", "top-1 accuracy at k = 10",
            f"batch {batch}", logx=True, ylim=(0.80, 1.0), xlim=(0.3, 5000.0),
        )
    fig.tight_layout()
    save(fig, "report_fig12_pareto.png")


def main():
    fig05()
    fig06()
    fig06b()
    fig07_paradox()
    fig14_ledger_full()
    fig08()
    fig10()
    fig11()
    fig12()
    print(f"wrote {len(WRITTEN)} figures into results/:")
    for name in WRITTEN:
        print("  " + os.path.join("results", name))


if __name__ == "__main__":
    main()
