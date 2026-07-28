#!/usr/bin/env bash
# Acceptance: P2 depth-aware occlusion.
# Usage:
#   bash tests_au/ops/mapper/accept_depth_occlusion.sh
#   OUT_DIR=b/d/hand2robot/runs/accept_depth SIDE=right bash tests_au/ops/mapper/accept_depth_occlusion.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Prefer hand2robot env helper when present.
if [[ -f "$REPO_ROOT/b/scripts/hand2robot/_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/b/scripts/hand2robot/_env.sh"
else
  cd "$REPO_ROOT"
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/b/d/hand2robot/runs/accept_depth}"
SIDE="${SIDE:-right}"
mkdir -p "$OUT_DIR"

echo "=== accept_depth_occlusion → $OUT_DIR side=$SIDE"
python "$SCRIPT_DIR/accept_depth_occlusion.py" \
  --output-dir "$OUT_DIR" \
  --side "$SIDE" \
  --gl-backend "$MUJOCO_GL"

echo "Report: $OUT_DIR/accept_depth_occlusion_report.json"
