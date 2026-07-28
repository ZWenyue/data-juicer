#!/usr/bin/env bash
# Process a human-hand dataset → robot-view LeRobot (render + export + finalize).
#
# DATASET can be:
#   - EgoDex LeRobot root  (auto: videos → jsonl)
#   - directory of mp4s    (auto → jsonl)
#   - a single .mp4
#   - an existing .jsonl   (videos field per line)
#
# Usage:
#   DATASET=/mnt/r/DATA/EgoDex/test_lerobot SIDE=right MAX_VIDEOS=2 \
#     bash b/scripts/hand2robot/process.sh
#
#   DATASET=/path/to/videos.jsonl \
#   CALIBRATION_PATH=b/d/hand2robot/calibration/r1_right_egodex_v1.yaml \
#   SIDE=right \
#     bash b/scripts/hand2robot/process.sh
#
#   SIDE=both CALIBRATION_PATH=b/d/hand2robot/calibration/r1_both_egodex_v1.yaml \
#   DATASET=... bash b/scripts/hand2robot/process.sh
#
# Env:
#   RUN_DIR / OUT_ROOT / MAX_VIDEOS / OFFSET / SKIP_FINALIZE=1 / BUILD_VLA_MANIFEST=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

INTERNAL="$SCRIPT_DIR/internal"
DATASET="${DATASET:-${DATASET_PATH:-}}"
if [[ -z "$DATASET" ]]; then
  cat >&2 <<'EOF'
Set DATASET to one of:
  - EgoDex LeRobot root, e.g. /mnt/r/DATA/EgoDex/test_lerobot
  - a folder of videos
  - a .jsonl / .mp4

Example:
  DATASET=/mnt/r/DATA/EgoDex/test_lerobot SIDE=right MAX_VIDEOS=2 \
    bash b/scripts/hand2robot/process.sh
EOF
  exit 2
fi

# Default calib for this SIDE
if [[ -z "${CALIBRATION_PATH:-}" ]]; then
  if [[ "$SIDE" == "both" ]]; then
    CALIBRATION_PATH="$CALIB_DIR/r1_both_egodex_v1.yaml"
  else
    CALIBRATION_PATH="$CALIB_DIR/r1_${SIDE}_egodex_v1.yaml"
  fi
fi
[[ -f "$CALIBRATION_PATH" ]] || {
  echo "missing calib: $CALIBRATION_PATH" >&2
  echo "Run: DATASET=... SIDE=$SIDE bash b/scripts/hand2robot/calibrate.sh" >&2
  exit 1
}

RUN_DIR="${RUN_DIR:-$OUT_ROOT/process_${SIDE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
JSONL="$RUN_DIR/input_dataset.jsonl"

echo "=== [1/3] prepare jsonl from DATASET=$DATASET"
PREP_ARGS=( -m data_juicer._au.tools.prepare_hand_dataset_jsonl --input "$DATASET" --output "$JSONL" )
if [[ -n "${MAX_VIDEOS:-}" && "${MAX_VIDEOS}" != "0" ]]; then
  PREP_ARGS+=( --max-videos "$MAX_VIDEOS" )
fi
if [[ -n "${OFFSET:-}" && "${OFFSET}" != "0" ]]; then
  PREP_ARGS+=( --offset "$OFFSET" )
fi
python "${PREP_ARGS[@]}"

echo "=== [2/3] dj-process (smooth → render → caption stub → export staging)"
export DATASET_PATH="$JSONL"
export CALIBRATION_PATH
export RUN_DIR
export SIDE
bash "$INTERNAL/07_process_ego_to_robot.sh"

LEROBOT_DIR="$RUN_DIR/lerobot_dataset"
if [[ "${SKIP_FINALIZE:-0}" != "1" ]]; then
  echo "=== [3/3] finalize LeRobot dataset → $LEROBOT_DIR"
  python - <<PY
from data_juicer._au.ops.mapper.export_robot_render_lerobot_mapper import ExportRobotRenderLeRobotMapper
ExportRobotRenderLeRobotMapper.finalize_dataset(
    "$LEROBOT_DIR",
    fps=10,
    robot_type="r1_lite_ego_retarget",
)
print("finalized:", "$LEROBOT_DIR")
PY
else
  echo "=== skip finalize (SKIP_FINALIZE=1); staging left under $LEROBOT_DIR/staging"
fi

if [[ "${BUILD_VLA_MANIFEST:-0}" == "1" ]]; then
  echo "=== optional VLA A/B manifest"
  DATASET_PATH="$JSONL" CALIBRATION_PATH="$CALIBRATION_PATH" RUN_ROOT="$RUN_DIR/vla_ab" \
    bash "$INTERNAL/08_build_vla_ab_manifest.sh"
fi

echo
echo "Process done."
echo "  jsonl:     $JSONL"
echo "  frames:    $RUN_DIR/frames"
echo "  robot:     $RUN_DIR/robot_frames"
echo "  lerobot:   $LEROBOT_DIR"
echo "  calib:     $CALIBRATION_PATH"
if [[ -f "$LEROBOT_DIR/meta/info.json" ]]; then
  python - <<PY
import json
from pathlib import Path
info=json.loads(Path("$LEROBOT_DIR/meta/info.json").read_text())
feat=info.get("features") or {}
print("  info: episodes=", info.get("total_episodes"),
      "state=", (feat.get("observation.state") or {}).get("shape"),
      "action=", (feat.get("action") or {}).get("shape"),
      "dual_arm=", info.get("dual_arm"))
PY
fi
