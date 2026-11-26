# 复现说明

本文给出从零复现报告数字的完整步骤。结果本身见 [RESULTS.md](RESULTS.md)，报告见 [report.pdf](report.pdf)。

## 1. 硬件与软件

| 角色 | 机器 | 说明 |
| --- | --- | --- |
| 训练 / 精度 | RTX PRO 5000 72 GB | bf16；训练时卡上还有并行任务，GPU 小时数只作量级参考 |
| 延迟 | RTX 5060 8 GB（WSL2） | 空闲，无其他计算任务；不能锁频 |
| Python | 3.12 | conda 环境 |
| PyTorch | 2.7.0+cu128 | sm_120（Blackwell）可用 |
| torchvision | 0.22.0 | |
| timm | 1.0.15 | ViT-S/16 预训练权重 |
| 其他 | numpy、pyyaml、matplotlib、pandas、pytest、ruff | 见 `pyproject.toml` |

延迟的绝对毫秒只在同一张卡、同一会话内比较。跨卡比较没有意义。

## 2. 环境

```bash
bash scripts/setup_env.sh
conda activate patchdfs
```

手动安装：

```bash
conda create -y -n patchdfs python=3.12
conda activate patchdfs
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

验证：

```bash
python -c "import torch, timm; print(torch.__version__, torch.version.cuda, timm.__version__)"
PYTHONPATH=src python -m pytest -q     # 27 passed
```

## 3. 数据

默认根目录 `./data`，用 `PATCHDFS_DATA` 覆盖：

```bash
export PATCHDFS_DATA=/path/to/data
bash scripts/prepare_data.sh
```

下载内容与布局：

```
$PATCHDFS_DATA/
  MNIST/                 # torchvision 自动下载
  cifar-10-batches-py/   # torchvision 自动下载
  imagenette2-320/
    train/               # 9469 张，按 9:1 切训练/验证（np.random.seed(0)）
    val/                 # 3925 张测试
```

验证/测试划分固定为 `np.random.seed(0)`，见 `src/patchdfs/data/mnist.py` 的 `_split_indices`。ViT 权重由 timm 从 Hugging Face 拉取；国内网络先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 4. 跑实验

所有脚本共用 `scripts/common.sh`，可覆盖：

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `PATCHDFS_DATA` | `./data` | 数据与权重 |
| `PATCHDFS_ARTIFACTS` | `./artifacts` | checkpoint、日志、选中掩码 |
| `PATCHDFS_GPU` | `0` | `CUDA_VISIBLE_DEVICES` |
| `PYTHON` | `python` | 解释器 |

### 4.1 GDFS

```bash
bash scripts/run_gdfs.sh mnist 0
bash scripts/run_gdfs.sh cifar10 0
```

产物在 `artifacts/gdfs/{mnist,cifar10}/seed0/`：
- `pretrained.pt`、`best.pt`、`final.pt`
- `test_selections.npz`（测试集每个预算的选中块与预测）
- `summary.json`（验证/测试准确率、轮数、GPU 小时）
- `log.jsonl`

冒烟：`bash scripts/run_gdfs.sh cifar10 0 --smoke`。

### 4.2 DIME

```bash
bash scripts/run_dime.sh mnist 0
bash scripts/run_dime.sh imagenette 0
```

产物在 `artifacts/dime/{mnist,imagenette}/seed0/`，除 checkpoint 外还有 `test_masks.npz`（每个预算/λ/置信度的掩码）和 `summary_eval.json`。

### 4.3 丢 token ViT

```bash
CONFIG=configs/sparse/imagenette_token.yaml RUN_NAME=imagenette_token \
  bash scripts/run_dime.sh imagenette 0
```

和稠密 DIME 用同一训练协议，只把前向换成 `TokenDroppedViT`。产物在 `artifacts/dime/imagenette_token/seed0/`。

多块推理（不重训，只改推理时的每步块数）：

```bash
for m in 1 2 4; do
  python -m patchdfs.dime.eval --config configs/sparse/imagenette_token.yaml \
    --ckpt artifacts/dime/imagenette_token/seed0/final.pt \
    --run-dir artifacts/dime/imagenette_token/seed0 --data-root "${PATCHDFS_DATA:-data}" \
    --multi $m --tag multi$m
done
```

### 4.4 单趟选择 ViT

先有稠密 DIME 教师：

```bash
bash scripts/run_dime.sh imagenette 0
```

再训单趟模型：

```bash
bash scripts/run_onepass.sh 0                                  # L=2
GLANCE=4 RUN_NAME=imagenette_onepass_g4 bash scripts/run_onepass.sh 0
K_TRAIN=10 RUN_NAME=imagenette_onepass_g2_k10 bash scripts/run_onepass.sh 0
```

产物在 `artifacts/sparse/<run>/seed0/`，包括 `best.pt`、`summary.json`、`test_kept.npz`（每个预算保留的块索引）。

### 4.5 静态基线

```bash
bash scripts/run_baselines.sh center
bash scripts/run_baselines.sh random
bash scripts/run_baselines.sh cae --budgets 5 10 15 20 30
```

## 5. 训练代价

| 实验 | 轮数 / 说明 | GPU 小时 |
| --- | --- | --- |
| GDFS MNIST | 预训练 57 + 联合 163 | 1.38 |
| DIME MNIST | 预训练 28 + 联合 167 | 1.40 |
| GDFS CIFAR-10 | 预训练 38 + 联合 63 | 3.35 |
| GDFS CIFAR-10 CAE | 14 个预算 | 6.32 |
| DIME Imagenette | 预训练 13 + 联合 93 | 14.2 |
| 丢 token Imagenette | 80 轮 | 11.3 |
| 单趟 Imagenette | 60 轮，每个 L | 0.12–0.2 |
| 延迟测量 | 一次性，见第 6 节 | 分钟级 |

GPU 小时是在有并行任务的 PRO 5000 上记录的，只用于估算，不是干净基准。

## 6. 延迟与 FLOPs

FLOPs 用 `torch.utils.flop_counter.FlopCounterMode` 实测：

```bash
python -m patchdfs.bench.run_ledger --run-dir results/compute_ledger \
  --dime-vit artifacts/dime/imagenette/seed0/final.pt \
  --dime-token artifacts/dime/imagenette_token/seed0/final.pt --batch 1
```

延迟必须满足：空闲卡、同一会话、同一 batch、同一编译模式、至少三次取中位数。

```bash
python scripts/figures/lat_compile_vit.py
python scripts/figures/lat_onepass.py artifacts/sparse/imagenette_onepass_g2/seed0/best.pt 2 1
python scripts/figures/lat_onepass.py artifacts/sparse/imagenette_onepass_g4/seed0/best.pt 4 1
```

报告用的主表来自 `results/latency_logs/master_5060.json`，产生脚本是仓库外的 `lat_5060_master.py`（同一会话内跑完全部方法，三次取中位数）。仓库里的脚本用于复核单个方法；不同会话的数字不要混在一张表里。

## 7. 生成图与报告

```bash
python scripts/figures/report_figs_data.py
python scripts/figures/report_figs_concept.py
cd docs && xelatex report.tex && xelatex report.tex
```

- `report_figs_data.py`：只读 `results/` 的 JSON，CPU 即可。
- `report_figs_concept.py`：概念图；需要 `results/sparse/imagenette_onepass_g4/seed0/test_kept.npz`（仓库保留了这个文件）。
- `fig_qualitative.py`：定性图，需要测试集和两个 `test_masks.npz`，见脚本头部的路径参数。
- `export_assets.py`：导出定性图用的四张测试图。

## 8. 已知偏差与边界

- 主复现是单种子。GDFS 的论文读数是 Fig.3 / Table 3，误差约 ±0.02；DIME 的论文读数从 Fig.2 / Fig.4 读出。
- 丢 token 版比置零版低 2–4 个点，未达到计划里“差 ≤ 3 个点”的线。
- 静态基线每个预算独立训练，因此曲线不单调；随机掩码每个预算只抽一组，论文是 5 次平均。
- 稀疏卷积只计算了理论上界，没有实现稀疏卷积核。
- 单趟选择的机制与 EViT 相同，贡献是测量与分解，不是新机制。
- 编译后的 logit 与 eager 有 bf16 舍入差；精度评测一律用 eager，延迟才用 compiled。

## 9. 常见问题

**timm 下载失败**：`export HF_ENDPOINT=https://hf-mirror.com`，或手动把 `vit_small_patch16_224` 权重放进 timm 缓存。

**显存不足**：把 config 的 `batch_size` / `eval_batch_size` 减半；DIME Imagenette 在 8 GB 卡上不能训练，只在 72 GB 卡上跑。

**torch.compile 很慢或报 Triton 错**：第一次编译需要几十秒；5060 只有 12 个 SM，Triton 会提示 `Not enough SMs to use max_autotune_gemm mode`，不影响 `reduce-overhead`。

**WSL 下延迟偏高**：确认没有 Xwayland / 其他进程占用 GPU，`nvidia-smi` 的显存占用应为 0 MiB 左右；同一会话内重跑整组，不要跨会话拼接。
