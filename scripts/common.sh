#!/usr/bin/env bash
# Shared environment for the launcher scripts. Override any variable before calling.
set -euo pipefail

PATCHDFS_ROOT="${PATCHDFS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PATCHDFS_ROOT
export PATCHDFS_DATA="${PATCHDFS_DATA:-$PATCHDFS_ROOT/data}"
export PATCHDFS_ARTIFACTS="${PATCHDFS_ARTIFACTS:-$PATCHDFS_ROOT/artifacts}"
export PATCHDFS_GPU="${PATCHDFS_GPU:-0}"
export PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PATCHDFS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$PATCHDFS_DATA" "$PATCHDFS_ARTIFACTS/logs"
cd "$PATCHDFS_ROOT"
