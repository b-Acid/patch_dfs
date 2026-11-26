"""Concept / architecture figures for the report (Chinese annotations).

usage: python3 scripts/figures/report_figs_concept.py
outputs:
  results/report_fig01_concept.png       "only a few patches are enough"
  results/report_fig02_sequential.png    how sequential selection works and what it costs
  results/report_fig03_kvcache.png       KV cache vs bidirectional ViT vs token-dropped ViT
  results/report_fig09_onepass_arch.png  one-pass selection architecture
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402

plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

BLUE, GREEN, ORANGE, RED = "#4C72B0", "#2CA02C", "#FF7F0E", "#D62728"
PURPLE, GREY, DARK = "#9467BD", "#7F7F7F", "#333333"

KEPT = np.load("results/sparse/imagenette_onepass_g4/seed0/test_kept.npz")["kept_10"]


class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.axis("off")

    def dx(self, inches):
        return inches / self.w * 100.0

    def dy(self, inches):
        return inches / self.h * 100.0

    def text(self, x, y, s, fs=9, color="black", ha="center", va="center",
             bold=False, zorder=12, ls=1.35, **kw):
        self.ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
                     fontweight="bold" if bold else "normal", zorder=zorder,
                     linespacing=ls, **kw)

    def box(self, x, y, w, h, s="", color=GREY, fs=9, fill=0.12, lw=1.7,
            bold=False, tc="black", r=1.0, zorder=4, ls="-" , alpha_edge=1.0, **kw):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            fc=color, ec="none", alpha=fill, zorder=zorder))
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            fc="none", ec=color, lw=lw, ls=ls, alpha=alpha_edge, zorder=zorder + 0.1))
        if s:
            self.ax.text(x + w / 2, y + h / 2, s, fontsize=fs, color=tc,
                         ha=kw.get("ha", "center"), va="center", zorder=zorder + 0.2,
                         fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(self, x1, y1, x2, y2, color="black", lw=1.7, ls="-", ms=13,
              style="-|>", zorder=5, rad=0.0):
        self.ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, color=color,
            lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=zorder,
            connectionstyle=f"arc3,rad={rad}"))

    def poly(self, pts, color="black", lw=1.7, ls="-", zorder=5):
        for a, b in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                         solid_capstyle="butt", zorder=zorder)
        self.arrow(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color,
                   lw=lw, ls=ls, zorder=zorder)

    def token_grid(self, x, y, w, h, kept=(), n=14, base=GREY, keep=RED,
                   zeros=False, fs=8, base_alpha=0.16, keep_alpha=0.92):
        cw, ch = w / n, h / n
        for r in range(n):
            for cc in range(n):
                idx = r * n + cc
                on = idx in kept
                self.ax.add_patch(Rectangle(
                    (x + cc * cw, y + h - (r + 1) * ch), cw * 0.94, ch * 0.94,
                    fc=keep if on else base, ec="none",
                    alpha=keep_alpha if on else base_alpha, zorder=2))
                if zeros and not on:
                    self.ax.text(x + cc * cw + cw / 2, y + h - (r + 1) * ch + ch / 2,
                                 "0", fontsize=fs, color="#9C9C9C", ha="center",
                                 va="center", zorder=3)

    def img_panel(self, x_in, y_in, size_in, path, kept):
        ax = self.fig.add_axes([x_in / self.w, y_in / self.h,
                                size_in / self.w, size_in / self.h])
        img = plt.imread(path)
        ax.imshow(img, extent=(0, 224, 224, 0), interpolation="nearest")
        for i in range(15):
            ax.plot([i * 16, i * 16], [0, 224], color="white", lw=0.45, alpha=0.35)
            ax.plot([0, 224], [i * 16, i * 16], color="white", lw=0.45, alpha=0.35)
        for idx in kept:
            r, cc = divmod(int(idx), 14)
            ax.add_patch(Rectangle((cc * 16, r * 16), 16, 16, fc=RED, ec=RED,
                                   lw=1.1, alpha=0.42, zorder=5))
        ax.add_patch(Rectangle((0, 0), 224, 224, fill=False, ec="#666666", lw=1.0, zorder=6))
        ax.set_xlim(0, 224)
        ax.set_ylim(224, 0)
        ax.axis("off")
        return ax


# --------------------------------------------------------------------------
# figure 1: few patches are enough
# --------------------------------------------------------------------------
def fig01():
    c = Canvas(11, 4.5)
    c.text(2.5, 96.5, "判断一辆卡车只需约 10 个块；读懂一句话也只需关键那一段",
           fs=13, bold=True, ha="left")

    size = 2.12
    c.img_panel(0.35, 1.72, size, "results/report_assets/img_2600_n03417042.png", KEPT[2600])
    c.img_panel(2.72, 1.72, size, "results/report_assets/img_3300_n03445777.png", KEPT[3300])
    c.text(12.8, 35.0, "196 块里约 10 块\n就能判断是卡车", fs=8.8, va="top")
    c.text(34.4, 35.0, "高尔夫球同样：\n10 块就够", fs=8.8, va="top")

    # right panel: sentence analogy
    x0, x1 = 48.0, 97.5
    c.text(x0, 83.5, "一句话：也只要关键的那一段", fs=10.5, bold=True, ha="left")
    widths = [7.0, 5.0, 9.0, 6.6, 6.0, 9.0]
    keys = [False, False, False, True, False, False]
    gap = 0.9
    bx = x0 + 1.0
    by, bh = 65.0, 8.0
    centres = []
    for wd, key in zip(widths, keys):
        c.ax.add_patch(FancyBboxPatch(
            (bx, by), wd, bh, boxstyle="round,pad=0,rounding_size=0.8",
            fc=RED if key else GREY, ec="none", alpha=0.85 if key else 0.30, zorder=3))
        c.ax.add_patch(FancyBboxPatch(
            (bx, by), wd, bh, boxstyle="round,pad=0,rounding_size=0.8",
            fc="none", ec=RED if key else GREY, lw=1.6 if key else 1.0, zorder=3.1))
        centres.append(bx + wd / 2)
        bx += wd + gap
    c.arrow(centres[3], 75.5, centres[3], 73.6, color=RED, lw=1.7)
    c.text(centres[3], 77.2, "关键那一段", fs=8.8, color=RED, bold=True)
    c.text(x0 + 1.0, 60.5, "灰色 = 不关键的段，可以跳过不算", fs=8.2, color="#555", ha="left")
    c.text(x0 + 1.0, 50.0, "理解一句话也往往只需要关键那一段：\n判别信息集中在少数位置，其余段对答案\n几乎没有贡献。", fs=8.6, ha="left", va="center")

    # bottom flow: dense reference row, then the on-demand reading chain
    yA, yB, hh = 19.0, 5.5, 7.0
    c.box(1.5, yA - hh / 2, 24.5, hh, "全部特征：196 个块", GREY, fs=8.8, fill=0.10)
    c.box(30.5, yA - hh / 2, 29.0, hh, "预测器：同一套 12 层 ViT", GREY, fs=8.8, fill=0.10)
    c.box(63.5, yA - hh / 2, 28.0, hh, "答案：卡车", GREY, fs=8.8, fill=0.10)
    c.arrow(26.4, yA, 30.1, yA, color=GREY, lw=1.8)
    c.arrow(59.9, yA, 63.1, yA, color=GREY, lw=1.8)

    c.box(1.5, yB - hh / 2, 17.0, hh, "已读特征：10 块", GREY, fs=8.5, fill=0.12)
    c.box(20.5, yB - hh / 2, 17.0, hh, "推断还差哪块", ORANGE, fs=8.5)
    c.box(39.5, yB - hh / 2, 15.0, hh, "按需读入", ORANGE, fs=8.5)
    c.box(56.5, yB - hh / 2, 15.0, hh, "更少的总阅读", GREEN, fs=8.5)
    c.box(73.5, yB - hh / 2, 18.0, hh, "同样的答案：卡车", RED, fs=8.5, fill=0.14)
    c.arrow(18.8, yB, 20.1, yB, color=DARK, lw=1.6, ms=10)
    c.arrow(37.8, yB, 39.1, yB, color=DARK, lw=1.6, ms=10)
    c.arrow(54.8, yB, 56.1, yB, color=DARK, lw=1.6, ms=10)
    c.arrow(71.8, yB, 73.1, yB, color=DARK, lw=1.6, ms=10)
    c.text(47.5, 12.6, "按需取样", fs=8.5, color=ORANGE, bold=True)
    c.fig.savefig("results/report_fig01_concept.png", dpi=200)
    plt.close(c.fig)


# --------------------------------------------------------------------------
# figure 2: sequential selection loop and its cost
# --------------------------------------------------------------------------
def fig02():
    c = Canvas(11, 6)
    c.text(1.5, 96.5, "串行选块每加一块都要重跑整张图：k=10 就是 11 次整图前向",
           fs=13, bold=True, ha="left")

    yb, hb = 70.0, 12.0
    c.box(2.0, yb, 15.0, hb, "已读集合 S\n从空集开始", BLUE, fs=8.8)
    c.box(21.5, yb, 25.0, hb, "未读块置零，整图过 12 层 ViT\n197 个 token 全部参与计算", BLUE, fs=8.5)
    c.box(51.0, yb, 17.0, hb, "价值头给\n196 个块打分", BLUE, fs=8.8)
    c.box(72.5, yb, 16.5, hb, "选最高分的一块\n加入已读集合", BLUE, fs=8.8)
    for xa, xbb in [(17.4, 21.1), (46.9, 50.6), (68.4, 72.1)]:
        c.arrow(xa, yb + hb / 2, xbb, yb + hb / 2, color=BLUE, lw=1.8)

    # loop back over the top
    c.poly([(80.75, yb + hb + 0.3), (80.75, 89.0), (9.5, 89.0), (9.5, yb + hb + 0.3)],
           color=BLUE, lw=1.8)
    c.text(44.5, 91.4, "循环 k 次（每圈 = 整图一次前向）", fs=8.8, color=BLUE)

    # exit: one more forward for the prediction
    c.arrow(80.75, yb - 0.4, 80.75, 60.5, color=BLUE, lw=1.8)
    c.text(82.0, 65.5, "选满 k 块之后", fs=8.2, color=BLUE, ha="left")
    c.box(67.0, 48.0, 28.0, 12.5, "再前向一次\n得到最终预测", BLUE, fs=9, fill=0.16)

    c.text(2.0, 56.0, "k = 10 时：11 次整图前向", fs=12, color=BLUE, bold=True, ha="left")
    c.text(2.0, 51.0, "10 次用于选块，1 次用于最终预测。\n每圈都要把整张图重新算一遍，\n已读过的块也不能复用上一圈的结果。", fs=8.5, color=DARK, ha="left", va="center")

    # bottom: zeroed patches still go through every layer
    kept = set(int(v) for v in KEPT[2600])
    c.token_grid(2.0, 8.0, 19.6, 32.7, kept=kept, zeros=True, fs=8, base_alpha=0.30)
    c.text(1.6, 4.6, "14×14 网格：10 块已读（红），186 块置零（灰）", fs=8.0, color=DARK, ha="left")
    c.arrow(22.3, 24.7, 26.6, 24.7, color=DARK, lw=1.8)
    c.box(27.0, 13.0, 18.5, 23.5, "12 层网络\n197 个 token 都算\n186 个位置的值是 0", BLUE, fs=8.6)

    c.text(48.0, 30.0, "置零 ≠ 不算：没读的块照样经过每一层", fs=10.5, bold=True, ha="left")
    c.text(48.0, 22.5, "矩阵乘法不会跳过零：每个位置仍然要做\nLayerNorm、注意力和 MLP，只是贡献为 0。\n所以省掉的是信息，不是算力。", fs=8.5, color=DARK, ha="left", va="center")
    c.text(48.0, 14.0, "真正的省算力要等“丢 token”版：\n未读的块直接不进序列，一次前向只算已读块。", fs=8.5, color=GREEN, ha="left", va="center")
    c.fig.savefig("results/report_fig02_sequential.png", dpi=200)
    plt.close(c.fig)


# --------------------------------------------------------------------------
# figure 3: KV cache, bidirectional attention, token-dropped ViT
# --------------------------------------------------------------------------
def fig03():
    c = Canvas(11, 3.6)
    c.text(2.0, 96.5, "KV cache 的两件事：没读到的不算，算过的不重算",
           fs=13, bold=True, ha="left")

    for x, w in [(0.6, 31.6), (33.4, 31.6), (66.2, 33.0)]:
        c.ax.add_patch(FancyBboxPatch((x, 2.0), w, 89.0,
                                      boxstyle="round,pad=0,rounding_size=1.2",
                                      fc="#FAFAFA", ec="#CFCFCF", lw=1.0, zorder=1))

    def tokens(P, xs, y, w, h, hot):
        for i, x in enumerate(xs):
            col = RED if i == hot else BLUE
            c.ax.add_patch(FancyBboxPatch((P + x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                          fc=col, ec="none", alpha=0.85 if i == hot else 0.22, zorder=3))
            c.ax.add_patch(FancyBboxPatch((P + x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                          fc="none", ec=col, lw=1.5, zorder=3.1))

    # ---------------- (a) LLM decoding with a KV cache ----------------
    P = 0.6
    c.text(P + 1.6, 78.0, "(a) LLM 解码：KV cache", fs=9.5, bold=True, ha="left")
    tw, th, ty = 2.9, 7.5, 61.0
    xs = [2.6 + i * 4.0 for i in range(6)]
    tokens(P, xs, ty, tw, th, hot=5)
    c.text(P + xs[5] + tw / 2, ty + th / 2, "新", fs=8.5, color="white", bold=True, zorder=6)
    c.ax.add_patch(FancyBboxPatch((P + 2.6, 40.5), 19.0, 5.8, boxstyle="round,pad=0,rounding_size=0.8",
                                  fc=BLUE, ec="none", alpha=0.16, zorder=3))
    c.ax.add_patch(FancyBboxPatch((P + 2.6, 40.5), 19.0, 5.8, boxstyle="round,pad=0,rounding_size=0.8",
                                  fc="none", ec=BLUE, lw=1.4, zorder=3.1))
    c.text(P + 12.1, 43.4, "已缓存 K/V：不再重算", fs=8.2, color=BLUE)
    c.arrow(P + xs[5] + tw / 2 + 0.3, ty - 0.2, P + 21.2, 46.6, color=RED, lw=1.6, rad=-0.18)
    c.text(P + 22.6, 51.0, "读缓存", fs=8.2, color=RED, ha="left")
    c.arrow(P + xs[5] + tw / 2, ty + th + 0.3, P + xs[5] + tw / 2, 72.0, color=RED, lw=1.6)
    c.text(P + xs[5] + tw / 2 + 1.6, 70.4, "只算这一列", fs=8.2, color=RED, ha="left")
    c.text(P + 15.5, 29.0, "算量 ∝ 新读入的 token 数，\n缓存的不再算", fs=9, va="center")
    c.text(P + 15.5, 17.0, "每步只做 1 个 token 的 QKV\n和它对缓存的注意力", fs=8.2, color="#555")

    # ---------------- (b) bidirectional ViT ----------------
    P = 33.4
    c.text(P + 1.6, 78.0, "(b) 双向 ViT：互相注意，缓存会过期", fs=9.5, bold=True, ha="left")
    tw, th, ty = 2.9, 7.5, 61.0
    xs = [3.0 + i * 4.2 for i in range(6)]
    centres = [P + x + tw / 2 for x in xs]
    tokens(P, xs, ty, tw, th, hot=5)
    c.text(centres[5], ty + th / 2, "新", fs=8.5, color="white", bold=True, zorder=6)
    for i in range(5):
        c.arrow(centres[5], ty + th + 0.3, centres[i], ty + th + 0.3,
                color=RED, lw=1.0, ms=8, style="<|-|>", rad=0.10, zorder=2.5)
    c.ax.add_patch(FancyBboxPatch((P + 2.6, 40.5), 17.4, 5.8, boxstyle="round,pad=0,rounding_size=0.8",
                                  fc=GREY, ec="none", alpha=0.14, zorder=3))
    c.ax.add_patch(FancyBboxPatch((P + 2.6, 40.5), 17.4, 5.8, boxstyle="round,pad=0,rounding_size=0.8",
                                  fc="none", ec=GREY, lw=1.3, ls="--", zorder=3.1))
    c.ax.plot([P + 2.9, P + 19.7], [40.8, 46.0], color=RED, lw=1.6, alpha=0.8, zorder=4)
    c.ax.plot([P + 2.9, P + 19.7], [46.0, 40.8], color=RED, lw=1.6, alpha=0.8, zorder=4)
    c.text(P + 20.8, 43.4, "旧 K/V 全部过期", fs=8.2, color=GREY, ha="left")
    c.text(P + 15.5, 29.0, "新 token 改变所有旧 token\n的表示，缓存会过期", fs=9, va="center")
    c.text(P + 15.5, 17.0, "每一层的 K/V 都要重算，\n所以缓存省不了串行前向", fs=8.2, color="#555")

    # ---------------- (c) token-dropped ViT ----------------
    P = 66.2
    c.text(P + 1.6, 78.0, "(c) 丢 token ViT：不读的块不进网络", fs=9.5, bold=True, ha="left")
    tw, th = 2.5, 7.2
    kept_x = [3.0 + i * 3.5 for i in range(4)]
    for i, x in enumerate(kept_x):
        col = BLUE if i == 0 else GREEN
        c.ax.add_patch(FancyBboxPatch((P + x, 64.0), tw, th, boxstyle="round,pad=0,rounding_size=0.6",
                                      fc=col, ec="none", alpha=0.24, zorder=3))
        c.ax.add_patch(FancyBboxPatch((P + x, 64.0), tw, th, boxstyle="round,pad=0,rounding_size=0.6",
                                      fc="none", ec=col, lw=1.5, zorder=3.1))
    c.text(P + kept_x[0] + tw / 2, 67.6, "C", fs=8.2, color=BLUE, bold=True, zorder=6)
    c.text(P + 9.6, 60.5, "CLS + 已读的 3 块", fs=8.2, color=DARK)
    c.box(P + 18.8, 42.0, 12.8, 30.0, "12 层网络\n只算这 4 个", GREY, fs=8.8, fill=0.12, r=1.2)
    c.arrow(P + 16.1, 67.6, P + 18.4, 67.6, color=DARK, lw=1.7, ms=10)
    for i in range(3):
        x = P + kept_x[i]
        c.ax.add_patch(FancyBboxPatch((x, 48.0), tw, th, boxstyle="round,pad=0,rounding_size=0.6",
                                      fc=GREY, ec="none", alpha=0.16, zorder=3))
        c.ax.add_patch(FancyBboxPatch((x, 48.0), tw, th, boxstyle="round,pad=0,rounding_size=0.6",
                                      fc="none", ec=GREY, lw=1.3, zorder=3.1))
        c.ax.plot([x + 0.3, x + tw - 0.3], [48.3, 54.9], color=RED, lw=1.4, zorder=4)
        c.ax.plot([x + 0.3, x + tw - 0.3], [54.9, 48.3], color=RED, lw=1.4, zorder=4)
    c.text(P + 8.2, 44.5, "未读的块：不进入序列", fs=8.2, color=GREY)
    c.text(P + 16.0, 27.0, "未读块不进入网络；\n一次前向的算量 ∝ 已读块数；\n全部读入（196 块）时与原模型相同", fs=8.2, va="center")

    c.fig.savefig("results/report_fig03_kvcache.png", dpi=200)
    plt.close(c.fig)


def fig09():
    c = Canvas(11, 6.0)
    c.text(1.5, 97.0, "单趟选块：1 次前向、0.24× 整图算力，选出 10 个块并给出预测",
           fs=13, bold=True, ha="left")
    c.text(1.5, 92.0, "串行：k=10 要 11 次整图前向    ｜    单趟：1 次前向", fs=8.8, color=DARK, ha="left")

    yb, hb = 72.0, 15.0
    boxes = [
        (1.0, 9.0, "输入图\n224×224", GREY),
        (12.2, 13.0, "patch\nembedding\n196 + CLS", GREY),
        (27.4, 15.0, "第 1–2 层：全图层\n197 个 token\n全部计算", GREEN),
        (44.6, 22.0, "选块（一次，无参数）\n第 3 层 CLS 的注意力打分\n取 top-10；其余 186 块\n合成 1 个汇总 token", ORANGE),
        (68.8, 16.0, "第 3–12 层：聚焦层\n只算 12 个 token\nCLS + 汇总 + 10 块", RED),
        (87.0, 12.0, "LayerNorm\n→ 取 CLS\n→ 线性头\n→ 10 类", GREY),
    ]
    for x, w, s, col in boxes:
        c.box(x, yb, w, hb, s, col, fs=8.2 if col is GREY else 8.5)
    for xa, xbb in [(10.3, 11.9), (25.5, 27.1), (42.7, 44.3), (66.9, 68.5), (85.1, 86.7)]:
        c.arrow(xa, yb + hb / 2, xbb, yb + hb / 2, color=DARK, lw=1.7)

    # two token maps: what the full-layer segment and the focus segment compute
    gy, gside_x = 40.0, 15.9
    gside_y = gside_x * 11 / 6.0
    kept = set(int(v) for v in KEPT[2600])
    c.token_grid(4.0, gy, gside_x, gside_y, kept=(), base=GREEN, base_alpha=0.22)
    c.text(4.0 + gside_x / 2, gy - 3.4, "全图层计算的 token\n197 个全算（浅绿）", fs=8.2, va="top")
    c.token_grid(24.0, gy, gside_x, gside_y, kept=kept)
    c.text(24.0 + gside_x / 2, gy - 3.4, "聚焦层计算的 token\n10 块（红）+ 1 汇总（紫）", fs=8.2, va="top")
    c.ax.add_patch(FancyBboxPatch((40.7, gy + gside_y - 3.4), 1.2, 2.2,
                                  boxstyle="round,pad=0,rounding_size=0.3",
                                  fc=PURPLE, ec="none", alpha=0.85, zorder=4))
    c.text(42.5, gy + gside_y - 2.3, "汇总 token", fs=8.0, color=PURPLE, ha="left")

    c.text(51.0, 55.0, "选块不需要第二次前向：第 3 层的注意力分数已经算出来了，\n直接拿来排序；被选的 10 块在 3–12 层里单独算，\n没选的 186 块只以 1 个汇总 token 出现。", fs=8.4, ha="left", va="center", color=DARK)

    # FLOPs bar
    c.text(1.5, 27.5, "k=10 时每一段的 FLOPs（整图一次前向 = 1.00×；段宽 = 各段相对大小）：", fs=9, bold=True, ha="left")
    segs = [("patch embedding", 0.013, GREY, 2.6), ("第 1–2 层：全图层", 0.17, GREEN, 0.0),
            ("选块", 0.0, ORANGE, 1.4), ("第 3–12 层：聚焦层", 0.05, RED, 0.0)]
    scale = 60.0 / 0.24
    x = 1.5
    by, bh = 18.0, 6.0
    for name, val, col, minw in segs:
        w = max(val * scale, minw)
        c.ax.add_patch(Rectangle((x, by), w, bh, fc=col, ec="white", lw=1.0, alpha=0.85, zorder=3))
        if w > 8:
            c.text(x + w / 2, by + bh / 2, f"{name}\n{val:.2f}×", fs=8.2, color="white",
                   bold=True, zorder=4)
        x += w + 0.5
    c.text(x + 1.0, by + bh / 2, "= 0.24×", fs=11, bold=True, va="center", ha="left")
    c.arrow(2.9, by - 0.4, 2.9, 15.2, color=GREY, lw=1.0, ms=8)
    c.text(2.6, 13.4, "patch embedding 0.01×", fs=8.0, color=GREY, ha="left")
    c.arrow(49.0, by - 0.4, 49.0, 15.2, color=ORANGE, lw=1.0, ms=8)
    c.text(50.5, 13.4, "选块 0.00×", fs=8.0, color=ORANGE, ha="left")
    c.text(1.5, 10.4, "合计 0.24×（FlopCounter 实测 0.237×）", fs=8.5, color=DARK, ha="left")
    c.text(73.5, 21.0, "对照：\n串行丢 token 0.72×\n串行稠密 12×\n作者代码 23×", fs=8.5, color=DARK, ha="left", va="center")

    c.fig.savefig("results/report_fig09_onepass_arch.png", dpi=200)
    plt.close(c.fig)


if __name__ == "__main__":
    fig01()
    fig02()
    fig03()
    fig09()
    print("wrote 4 figures")
