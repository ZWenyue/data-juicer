#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ -f "$REPO_ROOT/b/scripts/hand2robot/_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/b/scripts/hand2robot/_env.sh"
else
  cd "$REPO_ROOT"
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/b/d/hand2robot/runs/accept_p3}"
SIDE="${SIDE:-right}"
mkdir -p "$OUT_DIR"

echo "=== accept_p3_pipeline → $OUT_DIR side=$SIDE"
python "$SCRIPT_DIR/accept_p3_pipeline.py" \
  --output-dir "$OUT_DIR" \
  --side "$SIDE" \
  --gl-backend "$MUJOCO_GL"

echo "Report: $OUT_DIR/accept_p3_pipeline_report.json"
