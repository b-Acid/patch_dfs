"""Export a few Imagenette test images (test transform, 224x224, un-normalised) as PNG for the report figures.
usage: python scripts/figures/export_assets.py  -> results/report_assets/img_<idx>.png"""
import os
import sys

sys.path.insert(0, "src")
import torch  # noqa: E402
from torchvision import utils as vutils  # noqa: E402

from patchdfs.data.cifar10 import MEAN, STD  # noqa: E402
from patchdfs.data.imagenette import get_imagenette  # noqa: E402

_, _, test = get_imagenette(os.environ.get("PATCHDFS_DATA", "data"))
os.makedirs("results/report_assets", exist_ok=True)
for idx in (2600, 3300, 1900, 803):
    img, lab = test[idx]
    img = (img * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)
    vutils.save_image(img, f"results/report_assets/img_{idx}_{test.classes[lab]}.png")
print("ok", test.classes)
