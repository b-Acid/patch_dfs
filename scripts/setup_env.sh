#!/usr/bin/env bash
# Create a conda environment with the versions used for the reported runs.
# PyTorch wheels are CUDA 12.8 builds (RTX 40/50 series); change the index for another CUDA.
set -euo pipefail
ENV_NAME="${ENV_NAME:-patchdfs}"
conda create -y -n "$ENV_NAME" python=3.12
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
python -c "import torch, timm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'timm', timm.__version__)"
