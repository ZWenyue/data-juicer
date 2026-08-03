#!/usr/bin/env bash
# Calibrate hand→robot YAML from an ego-pipeline sample
# (pkl / json / jsonl / parquet with hand_action_tags + cam_c2w).
#
# Usage:
#   DATA_PATH=/path/to/sample.pkl \
#     bash b/scripts/hand2robot/04_calibrate_ego.sh
#   DATA_PATH=... SAMPLE_IDX=0 VIDEO_IDX=0 SIDE=right \
#     bash b/scripts/hand2robot/04_calibrate_ego.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_env.sh"

DATA_PATH="${DATA_PATH:-}"
if [[ -z "$DATA_PATH" ]]; then
  echo "Set DATA_PATH to a pipeline sample (pkl/json/jsonl/parquet)." >&2
  echo "Example:" >&2
  echo "  DATA_PATH=demos/ego_hand_action_annotation/output/xxx.pkl \\" >&2
  echo "    bash b/scripts/hand2robot/04_calibrate_ego.sh" >&2
  exit 2
fi
[[ -f "$DATA_PATH" ]] || { echo "file not found: $DATA_PATH" >&2; exit 1; }
[[ -f "$INIT_CALIB" ]] || { echo "missing init calib: $INIT_CALIB" >&2; exit 1; }
[[ -f "$MODEL_XML" ]] || {
  echo "missing model: $MODEL_XML — run setup.sh first" >&2
  exit 1
}

SAMPLE_IDX="${SAMPLE_IDX:-0}"
VIDEO_IDX="${VIDEO_IDX:-0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
STRIDE="${STRIDE:-2}"
VERSION="${VERSION:-v2}"
REPORT_DIR="${REPORT_DIR:-$OUT_ROOT/calib_ego_${SIDE}_${VERSION}}"
OUTPUT_CALIB="${OUTPUT_CALIB:-$CALIB_DIR/r1_${SIDE}_${VERSION}.yaml}"

mkdir -p "$REPORT_DIR" "$(dirname "$OUTPUT_CALIB")"
echo "=== ego calibrate data=$DATA_PATH sample=$SAMPLE_IDX video=$VIDEO_IDX side=$SIDE"
python -m data_juicer._au.tools.calibrate_hand_to_robot \
  --data-path "$DATA_PATH" \
  --sample-idx "$SAMPLE_IDX" \
  --video-idx "$VIDEO_IDX" \
  --side "$SIDE" \
  --max-frames "$MAX_FRAMES" \
  --stride "$STRIDE" \
  --init-calib "$INIT_CALIB" \
  --model "$MODEL_XML" \
  --output-calib "$OUTPUT_CALIB" \
  --report-dir "$REPORT_DIR" \
  --maxiter "${MAXITER:-80}" \
  --n-anchors "${N_ANCHORS:-8}" \
  --gl-backend "$MUJOCO_GL" \
  ${OPTIMIZE_BASE_ORIENT:+--optimize-base-orient} \
  ${OPTIMIZE_AXIS:+--optimize-axis} \
  ${FIT_WORKSPACE:+--fit-workspace} \
  ${ANCHOR_BASE:+--anchor-base}

echo "Report: $REPORT_DIR/calibrate_hand_to_robot_report.json"
echo "YAML:   $OUTPUT_CALIB"
echo "Next: set CALIBRATION_PATH=$OUTPUT_CALIB when running process.sh"
