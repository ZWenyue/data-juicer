#!/usr/bin/env bash
# P2 acceptance: depth-aware occlusion.
# Usage:
#   bash b/scripts/hand2robot/05_accept_depth.sh
#   SIDE=left bash b/scripts/hand2robot/05_accept_depth.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_depth}"
export OUT_DIR SIDE MUJOCO_GL
mkdir -p "$OUT_DIR"

echo "=== accept_depth_occlusion → $OUT_DIR"
bash "$REPO_ROOT/tests_au/ops/mapper/accept_depth_occlusion.sh"
