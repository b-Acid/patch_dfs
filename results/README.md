# 结果文件

本目录只保留**汇总证据**和报告图，不提交原始 checkpoint、`.npz` 掩码和训练日志。这些大文件在每次运行后写入本机的 `artifacts/`，可用 `docs/REPRODUCE.md` 的命令重新生成。

| 路径 | 内容 |
| --- | --- |
| `latency_logs/master_5060.json` | 报告用的延迟主表：同一会话、同一张空闲 RTX 5060、三次取中位数 |
| `latency_logs/token_5060.json` | 丢 token ViT 单次前向随 token 数的延迟 |
| `latency_logs/onepass_*.json` | 单趟 L=2 / L=4 的延迟 |
| `compute_ledger/ledger_b{1,256}.json` | FLOPs、前向次数、固定成本的账本 |
| `gdfs/{mnist,cifar10}/seed0/` | GDFS 的 `config.json`、`summary.json` |
| `gdfs/cifar10_baselines/` | 中心裁剪 / 随机掩码 / CAE 的汇总 |
| `dime/{mnist,imagenette,imagenette_token}/seed0/` | DIME 的配置、训练汇总、评测汇总 |
| `dime_eval/` | 丢 token 版每步多块的评测 |
| `sparse/imagenette_onepass_*/` | 单趟选择的汇总；`g4/seed0/test_kept.npz` 保留给概念图 |
| `sparse_conv_bound.json` | 稀疏卷积的 FLOPs 上界 |
| `tables.md` | 由 `collect.py` 生成的表格汇总 |
| `report_fig*.png` | 报告图 |
| `report_assets/` | 定性图用的四张 Imagenette 测试图 |

## 口径

- 延迟：空闲 RTX 5060，bf16；compiled 指 `torch.compile(mode="reduce-overhead")`。绝对毫秒不跨卡比较。
- FLOPs：`FlopCounterMode` 实测，与硬件无关。
- 准确率：单种子测试集；单趟选择另有种子 0/1/2 的结果，见 `sparse/imagenette_onepass_g2/seed{0,1,2}`。
- 旧 RTX PRO 5000 延迟数据在 `RESULTS.md` 第 4 节标为“旧数据”，只作对照，不再进入结论。

原始数据（`.npz`、`.pt`、`.log`）不提交；需要时按 `docs/REPRODUCE.md` 重跑。
