"""Download all datasets and the ViT-S/16 weights into PATCHDFS_DATA (default: ./data)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from patchdfs.data import get_cifar10, get_imagenette, get_mnist  # noqa: E402

DATA = os.path.abspath(os.environ.get("PATCHDFS_DATA", "data"))
os.makedirs(DATA, exist_ok=True)
t0 = time.time()
mnist = get_mnist(DATA)
print(f"MNIST       train/val/test = {len(mnist[0])}/{len(mnist[1])}/{len(mnist[2])}  ({time.time()-t0:.0f}s)", flush=True)
cifar = get_cifar10(DATA)
print(f"CIFAR-10    train/val/test = {len(cifar[0])}/{len(cifar[1])}/{len(cifar[2])}  ({time.time()-t0:.0f}s)", flush=True)
inet = get_imagenette(DATA)
print(f"Imagenette train/val/test = {len(inet[0])}/{len(inet[1])}/{len(inet[2])}  ({time.time()-t0:.0f}s)", flush=True)

from patchdfs.models import create_vit  # noqa: E402

m = create_vit()
n_params = sum(p.numel() for p in m.parameters())
print(f"ViT-S/16 pretrained ok: {n_params/1e6:.1f}M params  ({time.time()-t0:.0f}s)", flush=True)
