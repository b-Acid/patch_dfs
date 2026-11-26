"""Qualitative figure for the report: which patches each method selects at k = 10 (Imagenette test).
usage: python scripts/figures/fig_qualitative.py <indices> <dense_masks.npz> <token_masks.npz>
-> results/report_fig13_selected_patches.png"""
import os
import sys

sys.path.insert(0, "src")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from patchdfs.data.cifar10 import MEAN, STD  # noqa: E402
from patchdfs.data.imagenette import get_imagenette  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
RED, EDGE = "#D62728", "#FFFFFF"
ZH = {"n01440764": "金鱼", "n02102040": " Cassinian 鹟".strip(), "n02979186": "收音机", "n03000684": "链锯", "n03028079": "教堂",
      "n03394916": "法式号角", "n03417042": "卡车", "n03425413": "加油站", "n03445777": "高尔夫球", "n03888257": "降落伞"}
ART = os.environ.get("PATCHDFS_ARTIFACTS", "artifacts")
_, _, test = get_imagenette(os.environ.get("PATCHDFS_DATA", "data"))
g2 = np.load(os.path.join(ART, "sparse/imagenette_onepass_g2/seed0/test_kept.npz"))["kept_10"]
g4 = np.load(os.path.join(ART, "sparse/imagenette_onepass_g4/seed0/test_kept.npz"))["kept_10"]
m_sd = np.load(sys.argv[2])["mask_budget10"]
m_st = np.load(sys.argv[3])["mask_budget10"]
rows = [int(a) for a in sys.argv[1].split(",")]
cols = ["原图", "串行 DIME（置零）", "串行 DIME（丢 token）", "单趟 ViT，看 2 层", "单趟 ViT，看 4 层"]
n_r, n_c = len(rows), len(cols)
fig, axes = plt.subplots(n_r, n_c, figsize=(2.1 * n_c + 1.1, 2.1 * n_r + 0.55))
fig.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.005, wspace=0.03, hspace=0.08)
mean, std = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]


def draw(ax, mask):
    for p in np.flatnonzero(mask.reshape(14, 14)):
        r, c = divmod(int(p), 14)
        ax.add_patch(plt.Rectangle((c * 16 - 0.5, r * 16 - 0.5), 16, 16, fc=RED, alpha=0.55, ec=EDGE, lw=1.1))


for i, idx in enumerate(rows):
    img, lab = test[idx]
    wnid = test.classes[lab]
    img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    masks = [None, m_sd[idx], m_st[idx], np.bincount(g2[idx], minlength=196), np.bincount(g4[idx], minlength=196)]
    for j in range(n_c):
        ax = axes[i, j]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if masks[j] is not None:
            draw(ax, masks[j])
        if i == 0:
            ax.set_title(cols[j], fontsize=10.5, pad=6)
    axes[i, 0].set_ylabel(ZH.get(wnid, wnid), fontsize=11, labelpad=10)
plt.savefig("results/report_fig13_selected_patches.png", dpi=200)
print("ok")
