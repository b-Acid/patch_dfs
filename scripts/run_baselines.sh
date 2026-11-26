#!/usr/bin/env bash
# Train a CIFAR-10 static baseline (random mask / centre crop / CAE).
#   scripts/run_baselines.sh random
#   scripts/run_baselines.sh cae --budgets 5 10 15
set -euo pipefail
source "$(dirname "$0")/common.sh"
METHOD="${1:?method: random|center|cae}"
shift
RUN="$PATCHDFS_ARTIFACTS/gdfs/cifar10_baselines/$METHOD"
"$PYTHON" -m patchdfs.gdfs.train_baselines --method "$METHOD" \
  --data-root "$PATCHDFS_DATA" --run-dir "$RUN" "$@"
