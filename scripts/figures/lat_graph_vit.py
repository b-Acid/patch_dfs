"""Does a CUDA graph turn the FLOPs saving into latency at batch 1? Time eager vs graphed
token-dropped ViT passes with 1, 16, 64, 197 tokens (bf16)."""
import sys, time
sys.path.insert(0, "src")
import torch
from patchdfs.models import create_vit
from patchdfs.sparse.token_vit import TokenDroppedViT

dev = torch.device("cuda")
net = TokenDroppedViT(create_vit("vit_small_patch16_224", pretrained=True), 10).to(dev).eval()
def run(x, m, n=None):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return net.both(x, m, max_tokens=n)

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1

def mask(n, B=B):
    m = torch.zeros(B, 196, device=dev); m[:, :n] = 1; return m

def timeit(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts)//2] * 1e3

print(f"batch {B}: {'tokens':>6} {'eager ms':>9} {'graph ms':>9}")
for n in (1, 16, 64, 196):
    x = torch.randn(B, 3, 224, 224, device=dev); m = mask(n)
    eager = timeit(lambda: run(x, m))
    with torch.no_grad():
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run(x, m)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = run(x, m, n)
        graphed = timeit(g.replay)
    print(f"{n:>6} {eager:9.2f} {graphed:9.2f}")