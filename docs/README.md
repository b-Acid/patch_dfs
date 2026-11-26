# 文档

| 文件 | 内容 |
| --- | --- |
| [report.pdf](report.pdf) / [report.tex](report.tex) | 课程报告（XeLaTeX，中文） |
| [RESULTS.md](RESULTS.md) | 全部实验数字的权威来源；含 5060 延迟主表与旧卡对照 |
| [REPRODUCE.md](REPRODUCE.md) | 环境、数据、训练、评测、延迟测量的完整步骤 |

报告重新编译：

```bash
cd docs
xelatex report.tex
xelatex report.tex
```

图在 `../results/report_fig*.png`，生成脚本见 `../scripts/figures/`。
