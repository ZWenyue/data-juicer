#!/usr/bin/env bash
# Calibrate hand→robot YAML for a human-hand dataset.
#
# Modes (SOURCE):
#   egodex     — EgoDex LeRobot root (default if EGODEX_ROOT / DATASET looks like EgoDex)
#   ego        — pipeline sample pkl/json/jsonl/parquet (needs DATA_PATH)
#   galaxea    — Galaxea robot LeRobot (proxy calib / FK-IK)
#   synthetic  — no data, smoke only
#
# Usage:
#   # EgoDex (most common)
#   DATASET=/mnt/r/DATA/EgoDex/test_lerobot SIDE=right EPISODE=2 \
#     bash b/scripts/hand2robot/calibrate.sh
#
#   # Already-processed ego sample
#   SOURCE=ego DATA_PATH=/path/to/sample.pkl SIDE=right \
#     bash b/scripts/hand2robot/calibrate.sh
#
#   SOURCE=synthetic SIDE=right bash b/scripts/hand2robot/calibrate.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

INTERNAL="$SCRIPT_DIR/internal"

# Infer SOURCE from inputs when not set.
SOURCE="${SOURCE:-}"
DATASET="${DATASET:-${EGODEX_ROOT:-${DATASET_PATH:-}}}"
if [[ -z "$SOURCE" ]]; then
  if [[ -n "${DATA_PATH:-}" ]]; then
    SOURCE=ego
  elif [[ -n "$DATASET" && -d "$DATASET/data" && -f "$DATASET/meta/info.json" ]]; then
    SOURCE=egodex
  elif [[ -n "${LEROBOT_ROOT:-}" ]]; then
    SOURCE=galaxea
  else
    SOURCE=egodex
  fi
fi

calibrate_one_side() {
  local side="$1"
  export SIDE="$side"
  # Refresh side-dependent defaults from _env conventions
  export INIT_CALIB="${INIT_CALIB_OVERRIDE:-$CALIB_DIR/r1_${side}_v1.yaml}"
  export MODEL_XML="${MODEL_XML_OVERRIDE:-$MODEL_DIR/r1_lite_arm_${side}.xml}"
  case "$SOURCE" in
    egodex)
      export EGODEX_ROOT="${EGODEX_ROOT:-${DATASET:-/mnt/r/DATA/EgoDex/test_lerobot}}"
      export EPISODE="${EPISODE:-2}"
      export VERSION="${VERSION:-egodex_v2}"
      export OUTPUT_CALIB="${OUTPUT_CALIB_OVERRIDE:-$CALIB_DIR/r1_${side}_${VERSION}.yaml}"
      echo "=== calibrate SOURCE=egodex root=$EGODEX_ROOT episode=$EPISODE side=$side"
      bash "$INTERNAL/04_calibrate_egodex.sh"
      ;;
    ego)
      if [[ -z "${DATA_PATH:-}" ]]; then
        echo "SOURCE=ego requires DATA_PATH=/path/to/sample.{pkl,json,jsonl,parquet}" >&2
        exit 2
      fi
      export VERSION="${VERSION:-v2}"
      export OUTPUT_CALIB="${OUTPUT_CALIB_OVERRIDE:-$CALIB_DIR/r1_${side}_${VERSION}.yaml}"
      echo "=== calibrate SOURCE=ego data=$DATA_PATH side=$side"
      bash "$INTERNAL/04_calibrate_ego.sh"
      ;;
    galaxea)
      export LEROBOT_ROOT="${LEROBOT_ROOT:-$DATASET}"
      echo "=== calibrate SOURCE=galaxea root=$LEROBOT_ROOT side=$side"
      bash "$INTERNAL/03_calibrate_galaxea.sh"
      ;;
    synthetic)
      echo "=== calibrate SOURCE=synthetic side=$side"
      bash "$INTERNAL/02_calibrate_synthetic.sh"
      ;;
    *)
      echo "Unknown SOURCE=$SOURCE (use egodex|ego|galaxea|synthetic)" >&2
      exit 2
      ;;
  esac
}

if [[ "$SIDE" == "both" ]]; then
  echo "SIDE=both → calibrate left then right (merge YAML separately if needed)"
  calibrate_one_side left
  calibrate_one_side right
  echo
  echo "Per-side YAMLs written. For dual-arm process use:"
  echo "  CALIBRATION_PATH=$CALIB_DIR/r1_both_egodex_v2.yaml"
  echo "  (or merge left/right into a both YAML)"
else
  calibrate_one_side "$SIDE"
  echo
  echo "Calibrate done. Use this YAML in process.sh:"
  echo "  CALIBRATION_PATH=${OUTPUT_CALIB:-$CALIB_DIR/r1_${SIDE}_egodex_v2.yaml}"
fi