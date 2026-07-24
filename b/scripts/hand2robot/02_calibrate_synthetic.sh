#!/usr/bin/env bash
# Synthetic smoke calibration (no ego meta required).
# Usage:
#   bash b/scripts/hand2robot/02_calibrate_synthetic.sh
#   SIDE=left bash b/scripts/hand2robot/02_calibrate_synthetic.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

REPORT_DIR="${REPORT_DIR:-$OUT_ROOT/calib_synthetic_${SIDE}}"
OUTPUT_CALIB="${OUTPUT_CALIB:-$REPORT_DIR/r1_${SIDE}_calibrated.yaml}"
mkdir -p "$REPORT_DIR"

[[ -f "$INIT_CALIB" ]] || { echo "missing init calib: $INIT_CALIB" >&2; exit 1; }
[[ -f "$MODEL_XML" ]] || {
  echo "missing model: $MODEL_XML — run 01_build_assets.sh first" >&2
  exit 1
}

echo "=== synthetic calibrate side=$SIDE"
python -m data_juicer._au.tools.calibrate_hand_to_robot \
  --synthetic \
  --side "$SIDE" \
  --init-calib "$INIT_CALIB" \
  --model "$MODEL_XML" \
  --output-calib "$OUTPUT_CALIB" \
  --report-dir "$REPORT_DIR" \
  --maxiter "${MAXITER:-40}" \
  --n-anchors "${N_ANCHORS:-6}" \
  --gl-backend "$MUJOCO_GL"

echo "Report: $REPORT_DIR/calibrate_hand_to_robot_report.json"
echo "YAML:   $OUTPUT_CALIB"
