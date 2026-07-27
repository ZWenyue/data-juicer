#!/usr/bin/env bash
# P1 human-hand calibration on EgoDex LeRobot (real egocentric wrist poses).
#
# Usage:
#   bash b/scripts/hand2robot/04_calibrate_egodex.sh
#   EGODEX_ROOT=/mnt/r/DATA/EgoDex/test_lerobot EPISODE=2 SIDE=right \
#     bash b/scripts/hand2robot/04_calibrate_egodex.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_env.sh"

EGODEX_ROOT="${EGODEX_ROOT:-/mnt/r/DATA/EgoDex/test_lerobot}"
EPISODE="${EPISODE:-2}"
MAX_FRAMES="${MAX_FRAMES:-120}"
STRIDE="${STRIDE:-4}"
VERSION="${VERSION:-egodex_v1}"
REPORT_DIR="${REPORT_DIR:-$OUT_ROOT/calib_egodex_${SIDE}_ep$(printf '%06d' "$EPISODE")}"
OUTPUT_CALIB="${OUTPUT_CALIB:-$CALIB_DIR/r1_${SIDE}_${VERSION}.yaml}"

[[ -d "$EGODEX_ROOT/data" ]] || { echo "not EgoDex root: $EGODEX_ROOT" >&2; exit 1; }
[[ -f "$INIT_CALIB" ]] || { echo "missing init calib: $INIT_CALIB" >&2; exit 1; }
[[ -f "$MODEL_XML" ]] || {
  echo "missing model: $MODEL_XML — run setup.sh first" >&2
  exit 1
}

mkdir -p "$REPORT_DIR" "$(dirname "$OUTPUT_CALIB")"
echo "=== EgoDex P1 calib root=$EGODEX_ROOT episode=$EPISODE side=$SIDE"

EXTRA=()
if [[ "${EGODEX_EXTRINSICS_W2C:-0}" == "1" ]]; then
  EXTRA+=(--egodex-extrinsics-w2c)
fi
if [[ "${OPTIMIZE_BASE_ORIENT:-0}" == "1" ]]; then
  EXTRA+=(--optimize-base-orient)
fi

python -m data_juicer._au.tools.calibrate_hand_to_robot \
  --egodex-root "$EGODEX_ROOT" \
  --episode "$EPISODE" \
  --side "$SIDE" \
  --max-frames "$MAX_FRAMES" \
  --stride "$STRIDE" \
  --min-wrist-conf "${MIN_WRIST_CONF:-0.5}" \
  --init-calib "$INIT_CALIB" \
  --model "$MODEL_XML" \
  --output-calib "$OUTPUT_CALIB" \
  --report-dir "$REPORT_DIR" \
  --maxiter "${MAXITER:-80}" \
  --n-anchors "${N_ANCHORS:-10}" \
  --gl-backend "$MUJOCO_GL" \
  "${EXTRA[@]}"

echo "Report: $REPORT_DIR/calibrate_hand_to_robot_report.json"
echo "YAML:   $OUTPUT_CALIB"
python - <<PY
import json
from pathlib import Path
r=json.loads(Path("$REPORT_DIR/calibrate_hand_to_robot_report.json").read_text())
print("decision:", r.get("decision"))
p1=r.get("p1_human_hand") or {}
print("P1:", json.dumps(p1, indent=2))
print("checks:", json.dumps(r.get("checks"), indent=2))
PY
