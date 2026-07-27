#!/usr/bin/env bash
# Acceptance: calibrate_hand_to_robot (auto Galaxea if present, else synthetic).
# Usage:
#   bash b/scripts/hand2robot/05_accept_calibrate.sh
#   bash b/scripts/hand2robot/05_accept_calibrate.sh synthetic
#   bash b/scripts/hand2robot/05_accept_calibrate.sh /path/to/Handle_Plates_...
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_env.sh"

OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_calibrate}"
MODE="${1:-}"
mkdir -p "$OUT_DIR"

ARGS=(
  --output-dir "$OUT_DIR"
  --side "$SIDE"
  --gl-backend "$MUJOCO_GL"
  --episode "${EPISODE:-2}"
)

if [[ "$MODE" == "synthetic" ]]; then
  ARGS+=(--synthetic)
elif [[ -n "$MODE" && -d "$MODE/data" ]]; then
  ARGS+=(--lerobot-root "$MODE")
elif [[ -n "$MODE" && -f "$MODE" ]]; then
  ARGS+=(--data-path "$MODE")
fi

echo "=== accept_calibrate_hand_to_robot → $OUT_DIR"
python tests_au/ops/mapper/accept_calibrate_hand_to_robot.py "${ARGS[@]}"

echo "Report: $OUT_DIR/calibrate_hand_to_robot_report.json"
