#!/usr/bin/env bash
# Acceptance: synthetic hand→robot render mapper.
# Usage:
#   bash b/scripts/hand2robot/05_accept_render.sh
#   OUT_DIR=b/d/hand2robot/accept_out SIDE=right bash b/scripts/hand2robot/05_accept_render.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_render}"
mkdir -p "$OUT_DIR"

echo "=== accept_hand_to_robot_render → $OUT_DIR"
python tests_au/ops/mapper/accept_hand_to_robot_render.py \
  --output-dir "$OUT_DIR" \
  --side "$SIDE" \
  --gl-backend "$MUJOCO_GL"

echo "Report: $OUT_DIR/accept_hand_to_robot_render_report.json"
