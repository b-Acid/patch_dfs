#!/usr/bin/env bash
# Forward+backward smoke test on the selected GPU.
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$PYTHON" scripts/smoke_test.py
