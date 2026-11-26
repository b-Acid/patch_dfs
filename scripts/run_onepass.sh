#!/usr/bin/env bash
# Train the single-pass selection ViT on Imagenette (teacher: dense DIME run of the same seed).
#   scripts/run_onepass.sh 0
#   GLANCE=4 RUN_NAME=imagenette_onepass_g4 scripts/run_onepass.sh 0
#   K_TRAIN=10 RUN_NAME=imagenette_onepass_g2_k10 scripts/run_onepass.sh 0
set -euo pipefail
source "$(dirname "$0")/common.sh"
SEED="${1:-0}"
shift || true
CONFIG="${CONFIG:-configs/sparse/imagenette_onepass.yaml}"
NAME="${RUN_NAME:-imagenette_onepass}"
RUN="$PATCHDFS_ARTIFACTS/sparse/$NAME/seed$SEED"
"$PYTHON" -m patchdfs.sparse.onepass_train --config "$CONFIG" --seed "$SEED" \
  --data-root "$PATCHDFS_DATA" --run-dir "$RUN" \
  --glance-layers "${GLANCE:-2}" --epochs "${EPOCHS:-60}" --lr "${LR:-1e-5}" \
  --k-train "${K_TRAIN:-1,30}" "$@"
