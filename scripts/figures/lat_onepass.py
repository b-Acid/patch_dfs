"""One-pass selection ViT: FLOPs and latency against its own full-budget (dense) pass (bf16).
usage: python scripts/figures/lat_onepass.py <ckpt> <glance_layers> <batch> [modes...]   (idle GPU only)"""
import json
import sys
import time

sys.path.insert(0, "src")
import torch

from patchdfs.bench.flops import count_flops
from patchdfs.models import create_vit
from patchdfs.sparse.onepass import OnePassViT

ckpt, glance, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
modes = sys.argv[4:] or ["eager", "default", "reduce-overhead"]
dev = torch.device("cuda")
net = OnePassViT(create_vit("vit_small_patch16_224", pretrained=False), 10, glance_layers=glance).to(dev).eval()
net.load_state_dict(torch.load(ckpt, map_location=dev))


def fwd(x, k):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return net(x, k)[0]


def timeit(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


x1 = torch.randn(1, 3, 224, 224, device=dev)
flops = {}
with torch.no_grad():
    for k in (1, 5, 10, 15, 20, 30, 50, 196):
        flops[k] = count_flops(lambda k=k: fwd(x1, k))[0]
print("flops", {k: f"{v / 1e9:.3f}G" for k, v in flops.items()}, "ratio", {k: round(v / flops[196], 3) for k, v in flops.items()}, flush=True)

x = torch.randn(B, 3, 224, 224, device=dev)
rows = []
with torch.no_grad():
    for mode in modes:
        row = {"mode": mode}
        for k in (10, 20, 196):
            fn = lambda k=k: fwd(x, k)
            if mode != "eager":
                t0 = time.time()
                cfn = torch.compile(fn, mode=mode, dynamic=False)
                out_c = cfn()
                row["compile_s"] = round(time.time() - t0, 1)
                out_e = fn()
                row[f"err{k}"] = f"{(out_c.float() - out_e.float()).abs().max().item():.1e}"
                fn = cfn
            row[f"k{k}"] = round(timeit(fn), 2)
        row["ratio_k10"] = round(row["k10"] / row["k196"], 2)
        row["ratio_k20"] = round(row["k20"] / row["k196"], 2)
        rows.append(row)
        print(f"batch {B}", row, flush=True)
tag = ckpt.split("/")[-3] if ckpt.endswith("best.pt") else "onepass"
json.dump({"ckpt": ckpt, "glance": glance, "batch": B, "flops": flops, "flops_ratio": {k: v / flops[196] for k, v in flops.items()}, "latency_ms": rows},
          open(f"results/latency_logs/onepass_{tag}_b{B}.json", "w"), indent=1)
