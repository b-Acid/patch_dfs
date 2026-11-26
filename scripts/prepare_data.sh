#!/usr/bin/env bash
# Download MNIST, CIFAR-10, Imagenette2-320 and the ViT-S/16 weights into PATCHDFS_DATA.
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$PYTHON" scripts/prepare_data.py
