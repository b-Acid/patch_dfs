#!/usr/bin/env bash
# Train GDFS on MNIST or CIFAR-10.
#   scripts/run_gdfs.sh mnist 0
#   scripts/run_gdfs.sh cifar10 0 --smoke
set -euo pipefail
source "$(dirname "$0")/common.sh"
DS="${1:?dataset: mnist|cifar10}"
SEED="${2:-0}"
shift $(( $# < 2 ? 1 : 2 ))
RUN="$PATCHDFS_ARTIFACTS/gdfs/$DS/seed$SEED"
"$PYTHON" -m patchdfs.gdfs.train --config "configs/gdfs/$DS.yaml" --seed "$SEED" \
  --data-root "$PATCHDFS_DATA" --run-dir "$RUN" "$@"
