"""Operator fusion for the token-dropped ViT: eager vs torch.compile modes (bf16).
Measures one fused pass at batch B with n observed tokens, checks outputs, and prints k=10 totals."""
import sys, time
sys.path.insert(0, "src")
import torch
from patchdfs.models import create_vit
from patchdfs.sparse.token_vit import TokenDroppedViT

dev = torch.device("cuda")
net = TokenDroppedViT(create_vit("vit_small_patch16_224", pretrained=True), 10).to(dev).eval()

def make_fn(T):
    def fn(x, m):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net.both(x, m, max_tokens=T)
    return fn

def timeit(fn, x, m, iters=50):
    for _ in range(10): fn(x, m)
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(x, m); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts)//2] * 1e3

modes = [a for a in sys.argv[2:]] or ["eager", "default", "reduce-overhead", "max-autotune-no-cudagraphs"]
for B in [int(sys.argv[1])]:
    print(f"===== batch {B} =====", flush=True)
    for mode in modes:
        row = {}
        for n in (1, 16, 196):
            x = torch.randn(B, 3, 224, 224, device=dev)
            m = torch.zeros(B, 196, device=dev); m[:, :n] = 1
            fn = make_fn(n)
            if mode != "eager":
                t0 = time.time()
                cfn = torch.compile(fn, mode=mode, dynamic=False)
                with torch.no_grad(): out_c = cfn(x, m)
                row[f"compile_s"] = round(time.time() - t0, 1)
                with torch.no_grad(): out_e = fn(x, m)
                err = max((a.float() - b.float()).abs().max().item() for a, b in zip(out_c, out_e))
                row[f"err{n}"] = f"{err:.2e}"
                fn = lambda x, m, cfn=cfn: cfn(x, m)
            else:
                with torch.no_grad(): fn(x, m)
            with torch.no_grad():
                row[n] = round(timeit(fn, x, m), 2)
        print(mode, row, flush=True)
        # k=10 total: 12 passes with t+1 tokens (interpolate between measured token counts)
        import numpy as np
        ns = np.array([1, 16, 196]); ms = np.array([row[1], row[16], row[196]])
        per_pass = lambda t: float(np.interp(t + 1, ns, ms))
        total = sum(per_pass(t) for t in range(10)) + per_pass(10)
        print(f"   k=10 total ~{total:.1f} ms = {total / row[196]:.1f}x one dense pass", flush=True)
