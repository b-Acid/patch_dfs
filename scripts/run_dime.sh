#!/usr/bin/env bash
# Train DIME on MNIST or Imagenette. Set CONFIG to use one of the sparse variants:
#   scripts/run_dime.sh mnist 0
#   CONFIG=configs/sparse/imagenette_token.yaml RUN_NAME=imagenette_token scripts/run_dime.sh imagenette 0
set -euo pipefail
source "$(dirname "$0")/common.sh"
DS="${1:?dataset: mnist|imagenette}"
SEED="${2:-0}"
shift $(( $# < 2 ? 1 : 2 ))
CONFIG="${CONFIG:-configs/dime/$DS.yaml}"
NAME="${RUN_NAME:-$DS}"
RUN="$PATCHDFS_ARTIFACTS/dime/$NAME/seed$SEED"
"$PYTHON" -m patchdfs.dime.train --config "$CONFIG" --seed "$SEED" \
  --data-root "$PATCHDFS_DATA" --run-dir "$RUN" "$@"
