#!/usr/bin/env bash
# Calibrate / diagnose on Galaxea R1 Lite LeRobot (GT joints → FK/IK).
# This validates kinematics; it does NOT replace ego-hand retarget calibration.
#
# Usage:
#   bash b/scripts/hand2robot/03_calibrate_galaxea.sh
#   LEROBOT_ROOT=/path/to/Handle_Plates_... EPISODE=2 SIDE=right \
#     bash b/scripts/hand2robot/03_calibrate_galaxea.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_env.sh"

LEROBOT_ROOT="${LEROBOT_ROOT:-/mnt/r/DATA/pre_train_v1/Galaxea_R1_Lite/Handle_Plates_20250619_001}"
EPISODE="${EPISODE:-2}"
MAX_FRAMES="${MAX_FRAMES:-80}"
STRIDE="${STRIDE:-3}"
REPORT_DIR="${REPORT_DIR:-$OUT_ROOT/calib_galaxea_${SIDE}_ep$(printf '%06d' "$EPISODE")}"
OUTPUT_CALIB="${OUTPUT_CALIB:-$REPORT_DIR/r1_${SIDE}_galaxea_calibrated.yaml}"

[[ -d "$LEROBOT_ROOT/data" ]] || { echo "not a LeRobot root: $LEROBOT_ROOT" >&2; exit 1; }
[[ -f "$INIT_CALIB" ]] || { echo "missing init calib: $INIT_CALIB" >&2; exit 1; }
[[ -f "$MODEL_XML" ]] || {
  echo "missing model: $MODEL_XML — run setup.sh first" >&2
  exit 1
}

mkdir -p "$REPORT_DIR"
echo "=== galaxea calibrate root=$LEROBOT_ROOT episode=$EPISODE side=$SIDE"
python -m data_juicer._au.tools.calibrate_hand_to_robot \
  --lerobot-root "$LEROBOT_ROOT" \
  --episode "$EPISODE" \
  --side "$SIDE" \
  --max-frames "$MAX_FRAMES" \
  --stride "$STRIDE" \
  --init-calib "$INIT_CALIB" \
  --model "$MODEL_XML" \
  --output-calib "$OUTPUT_CALIB" \
  --report-dir "$REPORT_DIR" \
  --maxiter "${MAXITER:-40}" \
  --n-anchors "${N_ANCHORS:-6}" \
  --gl-backend "$MUJOCO_GL"

echo "Report: $REPORT_DIR/calibrate_hand_to_robot_report.json"
python -c "
import json
from pathlib import Path
r=json.loads(Path('$REPORT_DIR/calibrate_hand_to_robot_report.json').read_text())
print('decision:', r.get('decision'))
fk=r.get('galaxea_fk_ik') or {}
print('ik_from_fk:', fk.get('ik_from_fk_site_success'),
      'fk_aligned_m:', fk.get('fk_aligned_pos_median_m'),
      'calib_ik:', (r.get('metrics_after') or {}).get('ik_success_rate'))
"
