#!/usr/bin/env bash
# Build VLA A/B manifest (baseline vs arm-no-depth vs arm-depth).
# Usage:
#   bash b/scripts/hand2robot/08_build_vla_ab_manifest.sh
#   DATASET_PATH=./demos/ego_hand_action_annotation/data/demo-dataset.jsonl \
#     CALIBRATION_PATH=b/d/hand2robot/calibration/r1_right_egodex_v2.yaml \
#     bash b/scripts/hand2robot/08_build_vla_ab_manifest.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_env.sh"

DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/demos/ego_hand_action_annotation/data/demo-dataset.jsonl}"
CALIBRATION_PATH="${CALIBRATION_PATH:-$CALIB_DIR/r1_${SIDE}_egodex_v2.yaml}"
RUN_ROOT="${RUN_ROOT:-$OUT_ROOT/vla_ab}"
MANIFEST="${MANIFEST:-$RUN_ROOT/manifest.json}"

[[ -f "$DATASET_PATH" ]] || { echo "missing dataset: $DATASET_PATH" >&2; exit 1; }

mkdir -p "$RUN_ROOT"
python -m data_juicer._au.tools.build_vla_ab_manifest \
  --output "$MANIFEST" \
  --dataset-path "$DATASET_PATH" \
  --run-root "$RUN_ROOT" \
  --calibration-path "$CALIBRATION_PATH" \
  --side "$SIDE"

echo "Manifest: $MANIFEST"
