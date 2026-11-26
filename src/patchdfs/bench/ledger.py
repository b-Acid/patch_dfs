"""Compute ledger: backbone passes, FLOPs and latency for GDFS / DIME inference at budget k,
relative to one dense forward pass of the same backbone.

Inference accounting (one sample, budget k):
  GDFS (authors' forward):      k selector passes + 1 predictor pass.
  DIME (authors' predict_step): (k + 1) iterations x (1 predictor + 1 value pass) + 1 final predictor pass.
  DIME (fused, ours):           (k + 1) fused passes + 1 final predictor pass (same outputs).
  DIME (token-dropped, ours):   sum_{t=0..k} fused pass with t + 1 tokens (+ final).
"""
from __future__ import annotations

import time

import torch

from .flops import count_flops


@torch.no_grad()
def _time(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # median ms


def measure(label: str, fn, iters: int = 50) -> dict:
    flops, per_op = count_flops(fn)
    return {"label": label, "gflops": flops / 1e9, "ms": _time(fn, iters=iters), "per_op_gflops": {k: v / 1e9 for k, v in per_op.items()}}


def gdfs_ledger(net, x: torch.Tensor, budgets, *, amp: bool, iters: int = 50, passes_only: bool = False) -> dict:
    """net: GDFSNet. Returns per-budget FLOPs/latency relative to one dense predictor pass."""
    device = x.device
    m0 = torch.zeros(len(x), net.mask_size, device=device)
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
    sel = lambda: (ac.__enter__(), net.select_logits(x, m0), ac.__exit__(None, None, None))[1]  # noqa: E731
    pred = lambda: (ac.__enter__(), net.predict(x, m0), ac.__exit__(None, None, None))[1]  # noqa: E731
    s = measure("selector_pass", sel, iters)
    p = measure("predictor_pass", pred, iters)
    out = {"selector_pass": s, "predictor_pass": p, "dense_reference": p, "per_budget": {}}
    for k in budgets:
        out["per_budget"][int(k)] = {
            "authors": {"passes": k + 1, "gflops": k * s["gflops"] + p["gflops"], "ms": k * s["ms"] + p["ms"]},
        }
    return out


def dime_ledger(net, x: torch.Tensor, budgets, *, amp: bool, iters: int = 50, fused: bool = True) -> dict:
    device = x.device
    m0 = torch.zeros(len(x), net.mask_size, device=device)
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
    run = lambda fn: (ac.__enter__(), fn(), ac.__exit__(None, None, None))[1]  # noqa: E731
    both = measure("fused_pass", lambda: run(lambda: net.both(x, m0)), iters)
    if fused:
        pred = both  # same backbone pass; the two heads share it
        val = both
    else:
        pred = measure("predictor_pass", lambda: run(lambda: net.predict(x, m0)), iters)
        val = measure("value_pass", lambda: run(lambda: net.value_logits(x, m0)), iters)
    out = {"fused_pass": both, "predictor_pass": pred, "value_pass": val, "dense_reference": pred, "per_budget": {}}
    for k in budgets:
        auth = {"passes": 2 * (k + 1) + 1, "gflops": (k + 1) * (pred["gflops"] + val["gflops"]) + pred["gflops"], "ms": (k + 1) * (pred["ms"] + val["ms"]) + pred["ms"]}
        fus = {"passes": k + 2, "gflops": (k + 1) * both["gflops"] + pred["gflops"], "ms": (k + 1) * both["ms"] + pred["ms"]}
        out["per_budget"][int(k)] = {"authors_unfused": auth, "fused": fus}
    return out


def token_dropped_ledger(net, x: torch.Tensor, budgets, *, amp: bool, iters: int = 30) -> dict:
    """Measured FLOPs/latency of fused passes with n = 1, 2, 4, 8, 16, 32, 64, 128, 196 observed
    tokens; per-budget totals are summed over steps t = 0..k (t + 1 tokens) plus a final pass."""
    device = x.device
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
    run = lambda m: (ac.__enter__(), net.both(x, m), ac.__exit__(None, None, None))[1]  # noqa: E731
    grid = {n: measure(f"tokens_{n + 1}", lambda n=n: run(torch.cat([torch.ones(len(x), n, device=device), torch.zeros(len(x), net.mask_size - n, device=device)], 1)), iters) for n in (0, 1, 3, 7, 15, 31, 63, 127, 195)}
    dense = measure("tokens_197", lambda: run(torch.ones(len(x), net.mask_size, device=device)), iters)

    def interp(n: int) -> dict:
        keys = sorted(grid)
        if n in keys:
            return grid[n]
        lo = max([k for k in keys if k < n], default=keys[0])
        hi = min([k for k in keys if k > n], default=keys[-1])
        if hi == lo:
            return grid[lo]
        w = (n - lo) / (hi - lo)
        return {"gflops": grid[lo]["gflops"] * (1 - w) + grid[hi]["gflops"] * w, "ms": grid[lo]["ms"] * (1 - w) + grid[hi]["ms"] * w}

    out = {"pass_by_tokens": grid, "dense_reference": dense, "per_budget": {}}
    for k in budgets:
        passes = [interp(t) for t in range(k + 1)] + [interp(k)]
        out["per_budget"][int(k)] = {"passes": k + 2, "gflops": sum(p["gflops"] for p in passes), "ms": sum(p["ms"] for p in passes)}
    return out
