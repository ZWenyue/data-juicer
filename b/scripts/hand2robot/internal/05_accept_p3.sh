#!/usr/bin/env bash
# P3 acceptance: Render → Caption stub → Export + action invariance.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_env.sh"

OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_p3}"
export OUT_DIR SIDE MUJOCO_GL
mkdir -p "$OUT_DIR"

echo "=== accept_p3_pipeline → $OUT_DIR"
bash "$REPO_ROOT/tests_au/ops/mapper/accept_p3_pipeline.sh"
