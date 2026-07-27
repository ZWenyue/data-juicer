#!/usr/bin/env bash
# P4 acceptance: dual-arm both → LeRobot state16 / action14.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_env.sh"

OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_p4}"
export OUT_DIR MUJOCO_GL
mkdir -p "$OUT_DIR"

echo "=== accept_p4_pipeline → $OUT_DIR"
bash "$REPO_ROOT/tests_au/ops/mapper/accept_p4_pipeline.sh"
