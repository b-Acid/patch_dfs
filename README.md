# PatchDFS

复现 GDFS（ICML 2023）和 DIME（ICLR 2024）的**图像分类按需取样**实验，并回答一个后续问题：把“少看 patch”真正变成“少算 FLOPs、少花延迟”，需要改什么。

GDFS 和 DIME 都在每张图上串行地一块一块选：读一块、把网络重跑一遍、再决定下一块。这样做省了输入像素，但 ViT 的每一次前向仍要处理全部 token，所以单图延迟随预算**上升**。本仓库把这条链路拆成三件事分别测量：

1. **复现**：GDFS 的 MNIST / CIFAR-10，DIME 的 MNIST / Imagenette，按作者超参数重写为纯 PyTorch。
2. **算力账本**：同一模型、同一数据、同一张卡上，把串行选块、每步多块、单趟选择的前向次数、FLOPs、延迟拆开对比。
3. **两个改型**：丢 token 的 ViT（每次前向只算已读块）和单趟选择 ViT（前 L 层看全图，后 12−L 层只看 k 块，全程一次前向）。

课程报告：[docs/report.pdf](docs/report.pdf)（源文件 [docs/report.tex](docs/report.tex)）。全部数字的权威来源是 [docs/RESULTS.md](docs/RESULTS.md)。

## 主要结果

**复现**（单种子，测试集）：

| 实验 | 结果 |
| --- | --- |
| GDFS MNIST | 10 / 20 / 50 像素 0.876 / 0.943 / 0.973，与论文 Table 3 差 0.007 以内 |
| GDFS CIFAR-10 | 10 / 20 / 30 块 0.785 / 0.846 / 0.862，高于论文 greedy 曲线 |
| DIME MNIST | 惩罚式平均 12.4 块 0.944，预算式 10 块 0.82 左右 |
| DIME Imagenette | 10 / 15 / 30 块 0.892 / 0.926 / 0.961，整图 0.994 |

**算力与延迟**（Imagenette，ViT-S/16，k=10，延迟在空闲 RTX 5060 上重测）：

| 方法 | 前向次数 | FLOPs | batch 1 | batch 256 | 准确率 |
| --- | --- | --- | --- | --- | --- |
| 作者代码（eager，每步两次前向） | 23 | 23× | 129.3 ms | 2757 ms | — |
| 头融合（eager） | 12 | 12× | 67.4 ms | 1438 ms | 0.892 |
| 丢 token（eager） | 11 | 0.72× | 82.3 ms | 260.6 ms | 0.852 |
| 丢 token（compiled） | 11 | 0.72× | 8.62 ms | 103.7 ms | 0.852 |
| 单趟 L=2（compiled） | 1 | 0.24× | 0.87 ms | 29.55 ms | 0.915 |
| 单趟 L=4（compiled） | 1 | 0.39× | 1.01 ms | 43.04 ms | 0.949 |
| 整图一次（compiled，参照） | 1 | 1× | 1.18 ms | 100.58 ms | 0.994 |

三条结论：

- 串行选块的成本是**前向次数 × 每次前向的固定成本**。丢 token 把 FLOPs 压到 0.72×，但 batch 1 下每次前向仍有约 98 µs 的固定开销，所以 11 次前向仍比整图一次贵 7.3 倍。
- 吞吐场景（batch 256）里丢 token 有效：延迟降到整图一次的 0.92×；单图场景无效，必须减少前向次数。
- 单趟选择把前向次数降到 1，k=10 时 FLOPs 0.24×，batch 1 延迟 0.74×、batch 256 0.29×；代价是 k 较小时准确率低于串行稠密。

机制本身不是新的（EViT、DynamicViT、A-ViT 都做过单趟 token 选择）。这里的贡献是**同一模型、同一数据、同一张卡上的分解测量**：把“串行 / 每步多块 / 单趟”的前向次数和固定成本分开，说明单图延迟的地板在哪里。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/patchdfs/` | 模型、数据、训练、评测、算力账本 |
| `configs/` | 四类实验配置（GDFS / DIME / 丢 token / 单趟） |
| `scripts/` | 环境、数据、训练启动脚本；`scripts/figures/` 是报告图与延迟脚本 |
| `tests/` | 27 个单元测试（CPU 可跑） |
| `docs/` | 报告、结果、复现说明 |
| `results/` | 汇总 JSON、报告图；原始 `.npz` / checkpoint 不提交 |
| `third_party/` | 作者代码只读副本，仅作对照 |

## 安装

需要 CUDA GPU（报告用 RTX PRO 5000 训练、RTX 5060 测延迟），Python 3.12。实测组合：torch 2.7.0+cu128、torchvision 0.22.0、timm 1.0.15。

```bash
git clone git@github.com:b-Acid/patch_dfs.git
cd patch_dfs
bash scripts/setup_env.sh          # conda create + pip install -e ".[dev]"
```

如果已有环境，也可以手动：

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

数据和 ViT 权重默认放在 `./data`，可用 `PATCHDFS_DATA` 改：

```bash
export PATCHDFS_DATA=/path/to/data
bash scripts/prepare_data.sh
```

脚本会自动下载 MNIST、CIFAR-10、Imagenette2-320（约 350 MB）和 timm 的 ViT-S/16 预训练权重。国内网络可先 `export HF_ENDPOINT=https://hf-mirror.com`。

跑测试：

```bash
PYTHONPATH=src python -m pytest -q      # 27 passed
```

## 复现实验

所有启动脚本都读 `PATCHDFS_DATA`、`PATCHDFS_ARTIFACTS`（默认 `./artifacts`）、`PATCHDFS_GPU`（默认 0）。训练默认前台运行，按需自己加 `nohup` / `tmux`。

```bash
# GDFS
bash scripts/run_gdfs.sh mnist 0
bash scripts/run_gdfs.sh cifar10 0

# DIME
bash scripts/run_dime.sh mnist 0
bash scripts/run_dime.sh imagenette 0

# DIME + 丢 token ViT
CONFIG=configs/sparse/imagenette_token.yaml RUN_NAME=imagenette_token \
  bash scripts/run_dime.sh imagenette 0

# 单趟选择 ViT（默认读 artifacts/dime/imagenette/seed0/final.pt 当蒸馏教师）
bash scripts/run_onepass.sh 0
GLANCE=4 RUN_NAME=imagenette_onepass_g4 bash scripts/run_onepass.sh 0
```

静态基线：

```bash
bash scripts/run_baselines.sh center
bash scripts/run_baselines.sh random
bash scripts/run_baselines.sh cae --budgets 5 10 15 20 30
```

单次冒烟（不下载完整数据，只验证前向/反向）：

```bash
bash scripts/smoke_test.sh
```

各实验的 GPU 小时、种子、早停轮数和已知偏差见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

## 生成报告

报告图只读 `results/` 下的 JSON：

```bash
python scripts/figures/report_figs_data.py
python scripts/figures/report_figs_concept.py    # 需要 results/sparse/.../test_kept.npz
cd docs && xelatex report.tex && xelatex report.tex
```

定性图需要训练产物和测试集：

```bash
python scripts/figures/fig_qualitative.py 803,1900,2600,3300 \
  results/dime/imagenette/seed0/test_masks.npz \
  results/dime/imagenette_token/seed0/test_masks.npz
```

延迟脚本（必须在空闲 GPU 上跑，否则数字不可比）：

```bash
python scripts/figures/lat_compile_vit.py
python scripts/figures/lat_onepass.py artifacts/sparse/imagenette_onepass_g2/seed0/best.pt 2 1
```

## 硬件与测量口径

训练和精度评测在 RTX PRO 5000（72 GB）上完成。测延迟时该卡上同时有多个任务，读数受干扰，所以所有延迟改到一张空闲的 RTX 5060（8 GB，WSL2）上重测。两张卡不同，**绝对毫秒不跨卡比较**，只比较同一张卡上的相对倍数；FLOPs 与硬件无关。

5060 上无法锁频，测量有会话间波动，因此 `results/latency_logs/master_5060.json` 保存了同一会话、三次取中位数的整组数据，报告只引用这一组。复现延迟时请保持同一会话、同一 batch、同一编译模式，并确认卡上没有其他计算任务。

## 结果文件

- `results/latency_logs/master_5060.json`：报告用的延迟主表。
- `results/compute_ledger/`：FLOPs 与前向次数账本。
- `results/{gdfs,dime,sparse}/`：每次运行的 `config.json` 和 `summary.json`（原始 `.npz` / `.pt` 未提交）。
- `results/report_fig*.png`：报告图。

说明见 [results/README.md](results/README.md)。

## 作者代码

`third_party/dynamic-selection`（GDFS）和 `third_party/DIME` 是作者仓库的只读副本，用来核对实验设定；本项目不 import 它们，也不把它们作为依赖安装。各自许可证见目录内 `LICENSE`。上游 commit 记录在 `UPSTREAM_COMMIT.txt`。

## 已知边界

- 复现主结果是单种子；单趟选择给了三种子（0/1/2），种子间差异 0.5 个点以内。
- 丢 token 版比置零版低 2–4 个点，未达到“差 ≤ 3 个点”的验收线。
- 静态基线每个预算独立训练，曲线不单调；随机掩码每个预算只抽一组。
- 稀疏卷积只给了上界计算，没有实现卷积核。
- 单图延迟的地板由每层固定成本决定，任何只省 token 数、不减少前向次数的改型都绕不过去。
