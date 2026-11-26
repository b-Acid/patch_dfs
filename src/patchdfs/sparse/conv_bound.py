"""Upper bound on what block-sparse convolution could save for GDFS on CIFAR-10.

With a zero-preserving network, the output of a conv layer at a site is non-zero only if the site's
receptive field intersects an observed patch. So the exact sparse computation evaluates exactly
those sites; everything else is provably zero and can be skipped. We compute, per layer, the
fraction of output sites whose receptive field hits any observed patch, using the real selections
of the trained selector at budget k, and multiply by the layer's dense MACs.

Assumptions (stated in the report): zero-preserving backbone (no bias / no mean-centred norm),
exact gather-scatter sparse convolution with no overhead, selection patterns as observed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from patchdfs.bench.flops import conv_linear_macs
from patchdfs.gdfs.model import build_gdfs

IMG = 32
PATCH = 4


def _layers_of(net) -> list[dict]:
    """Walk the GDFS CIFAR net in execution order, tracking (resolution, receptive-field radius).
    Each entry: name, module, res (output resolution), r (output RF radius in input pixels), macs."""
    layers: list[dict] = []

    def rf_after(r_in: int, k: int, stride: int, site_in: float) -> float:
        # radius grows by (k-1)/2 input sites; an input site is site_in input pixels
        return r_in + (k - 1) / 2 * site_in

    def visit_block(prefix, blk: nn.Module, res: int, r_in: float):
        site = IMG / res
        k1, s1 = blk.conv1.kernel_size[0], blk.conv1.stride[0]
        res1 = res // s1
        r1 = rf_after(r_in, k1, s1, site)
        layers.append(dict(name=f"{prefix}.conv1", module=blk.conv1, res=res1, r=r1))
        site1 = IMG / res1
        r2 = rf_after(r1, blk.conv2.kernel_size[0], 1, site1)
        layers.append(dict(name=f"{prefix}.conv2", module=blk.conv2, res=res1, r=r2))
        sc = getattr(blk, "shortcut", None)
        if sc is not None and len(sc) > 0:
            # 1x1 stride-s2 conv: radius unchanged (kernel 1), output resolution res1
            layers.append(dict(name=f"{prefix}.shortcut", module=sc, res=res1, r=rf_after(r_in, 1, s1, site)))
        return res1, r2

    bb = net.backbone
    stem, block1, block2, block3, block4 = list(bb.children())[0], *[list(bb.children())[i][0] for i in (3, 4, 5, 6)]
    layers.append(dict(name="stem", module=stem, res=IMG, r=rf_after(0, 3, 1, 1.0)))
    res, r = IMG, 1.0
    for i, blk in enumerate((block1, block2, block3, block4), 1):
        res, r = visit_block(f"stage{i}", blk, res, r)
    # selector head: upsampling block from res x res, then 1x1 conv
    up = net.selector_head[0]
    site = IMG / res
    # ConvTranspose2d(k=4, s=2, p=1): output site depends on ~2 input sites -> radius += 1 input site
    res_u = res * 2
    layers.append(dict(name="sel.up.convT", module=up.conv1, res=res_u, r=r + site))
    site_u = IMG / res_u
    layers.append(dict(name="sel.up.conv", module=up.conv2, res=res_u, r=r + site + (3 - 1) / 2 * site_u))
    layers.append(dict(name="sel.up.shortcut", module=up.shortcut, res=res_u, r=r))  # ConvTranspose2d(k=2, s=2)
    layers.append(dict(name="sel.out", module=net.selector_head[1], res=res_u, r=r + site + site_u))
    for L in layers:
        m = L["module"]
        if isinstance(m, nn.Sequential):
            m = [x for x in m.modules() if isinstance(x, (nn.Conv2d, nn.ConvTranspose2d))][0]
        L["module"] = m
        in_res = L["res"] // m.stride[0] if isinstance(m, nn.ConvTranspose2d) else L["res"]
        L["macs"] = float(conv_linear_macs(m, torch.zeros(1, m.in_channels, in_res, in_res)))
    return layers


def active_fraction(sel: np.ndarray, res: int, r: float) -> np.ndarray:
    """sel: (N, k) observed patch indices (row-major 8x8 grid). Returns (N,) fraction of the
    res x res output sites whose (2r+1)^2 receptive field (in input pixels) intersects any patch."""
    site = IMG / res
    centers = (np.arange(res) + 0.5) * site - 0.5  # input-pixel centre of each site
    lo = centers - r - 0.5  # RF interval in input pixels (pixel-centre convention)
    hi = centers + r + 0.5
    px = (sel % 8) * PATCH
    py = (sel // 8) * PATCH
    out = np.empty(len(sel))
    for n in range(len(sel)):
        xs, ys = px[n], py[n]
        # patch interval [p, p+PATCH); intersects RF site interval if p < hi and p+PATCH > lo
        act_x = ((xs[:, None] < hi[None, :]) & (xs[:, None] + PATCH > lo[None, :])).any(axis=0)
        act_y = ((ys[:, None] < hi[None, :]) & (ys[:, None] + PATCH > lo[None, :])).any(axis=0)
        out[n] = act_x.sum() * act_y.sum() / (res * res)
    return out


def sparse_bound(selections: np.ndarray, budgets, n_max: int = 2000, seed: int = 0) -> dict:
    """selections: (N, k_max) selection order. Returns per-layer active fractions and MAC savings."""
    net = build_gdfs({"kind": "cifar_resnet", "mask_width": 8, "patch_size": 4}, 64, 10).eval()
    layers = _layers_of(net)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(selections), size=min(n_max, len(selections)), replace=False)
    out = {"layers": [{k: v for k, v in L.items() if k != "module"} for L in layers], "per_budget": {}}
    sel_macs = sum(L["macs"] for L in layers if not L["name"].startswith("sel."))
    head_macs = sum(L["macs"] for L in layers if L["name"].startswith("sel."))
    for k in budgets:
        sel = selections[idx, :k]
        fracs = np.array([active_fraction(sel, L["res"], L["r"]).mean() for L in layers])
        macs = np.array([L["macs"] for L in layers])
        is_sel = np.array([L["name"].startswith("sel.") for L in layers])
        dense_sel_pass = sel_macs + head_macs
        sparse_sel_pass = float((macs * fracs).sum())
        dense_pred_pass = sel_macs
        sparse_pred_pass = float((macs[~is_sel] * fracs[~is_sel]).sum())
        out["per_budget"][int(k)] = {
            "layer_active_fraction": {L["name"]: float(f) for L, f in zip(layers, fracs)},
            "dense_selector_pass_macs": dense_sel_pass,
            "sparse_selector_pass_macs": sparse_sel_pass,
            "selector_pass_saving": 1 - sparse_sel_pass / dense_sel_pass,
            "dense_predictor_pass_macs": dense_pred_pass,
            "sparse_predictor_pass_macs": sparse_pred_pass,
            "predictor_pass_saving": 1 - sparse_pred_pass / dense_pred_pass,
            "gdfs_inference_saving": 1 - (k * sparse_sel_pass + sparse_pred_pass) / (k * dense_sel_pass + dense_pred_pass),
        }
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", required=True, help="test_selections.npz from a GDFS CIFAR run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--random", action="store_true", help="use random selections instead (sanity)")
    args = ap.parse_args()
    if args.random:
        rng = np.random.RandomState(0)
        selections = np.stack([rng.permutation(64) for _ in range(args.n)])
    else:
        selections = np.load(args.selections)["selections"]
    budgets = [1, 2, 3, 5, 7, 10, 15, 20, 30, 64]
    budgets = [k for k in budgets if k <= selections.shape[1]] if not args.random else budgets
    res = sparse_bound(selections, budgets, n_max=args.n)
    json.dump(res, open(args.out, "w"), indent=1)
    for k, v in res["per_budget"].items():
        print(f"k={k:3d}: selector pass saving {v['selector_pass_saving']:.3f}, predictor pass saving {v['predictor_pass_saving']:.3f}, GDFS inference saving {v['gdfs_inference_saving']:.3f}")