#!/usr/bin/env bash
# Run ego → robot-view processing with the hand2robot recipe.
#
# Usage:
#   # edit dataset_path / model paths in configs/ego_to_robot_recipe.yaml first
#   bash b/scripts/hand2robot/07_process_ego_to_robot.sh
#
#   DATASET_PATH=/path/to/ego.jsonl \
#   CALIBRATION_PATH=b/d/hand2robot/calibration/r1_right_v2.yaml \
#   SIDE=right \
#     bash b/scripts/hand2robot/07_process_ego_to_robot.sh
#
#   CONFIG=b/scripts/hand2robot/configs/ego_to_robot_recipe.yaml \
#     bash b/scripts/hand2robot/07_process_ego_to_robot.sh --help
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

CONFIG="${CONFIG:-$H2R_SCRIPT_DIR/configs/ego_to_robot_recipe.yaml}"
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
[[ -f "$MODEL_XML" ]] || {
  echo "missing model: $MODEL_XML — run 01_build_assets.sh first" >&2
  exit 1
}

# Optional: materialize a run-specific config with path overrides.
RUN_DIR="${RUN_DIR:-$OUT_ROOT/ego_process}"
mkdir -p "$RUN_DIR"
RUNTIME_CONFIG="$RUN_DIR/ego_to_robot_runtime.yaml"

python - <<PY
from pathlib import Path
import yaml

cfg_path = Path("$CONFIG")
raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

dataset = "${DATASET_PATH:-}"
calib = "${CALIBRATION_PATH:-}"
export_path = "${EXPORT_PATH:-}"
side = "${SIDE}"
model_xml = "${MODEL_XML}"
run_dir = Path("$RUN_DIR")

if dataset:
    raw["dataset_path"] = dataset
if export_path:
    raw["export_path"] = export_path
else:
    raw["export_path"] = str(run_dir / "processed")

# Patch process list entries by op name.
process = raw.get("process") or []
for item in process:
    if not isinstance(item, dict) or len(item) != 1:
        continue
    name, kwargs = next(iter(item.items()))
    if not isinstance(kwargs, dict):
        continue
    if name == "video_extract_frames_mapper":
        kwargs["frame_dir"] = str(run_dir / "frames")
    elif name == "video_hand_action_compute_mapper":
        kwargs["hand_type"] = side
    elif name == "video_hand_to_robot_render_mapper":
        kwargs["hand_type"] = side
        kwargs["robot_model_paths"] = {side: model_xml}
        if calib:
            kwargs["calibration_path"] = calib
        kwargs["output_root"] = str(run_dir / "robot_frames")
        kwargs["gl_backend"] = "${MUJOCO_GL}"
    elif name == "video_hand_action_caption_stub_mapper":
        kwargs["hand_type"] = side
        kwargs["frame_field"] = "robot_render_frames"
    elif name == "export_robot_render_lerobot_mapper":
        kwargs["output_dir"] = str(run_dir / "lerobot_dataset")
        kwargs["frame_field"] = "robot_render_frames"
        kwargs["encode_video_from_frames"] = True
        kwargs["robot_type"] = kwargs.get("robot_type") or "r1_lite_ego_retarget"
    elif name == "export_to_lerobot_mapper":
        kwargs["output_dir"] = str(run_dir / "lerobot_dataset")
        kwargs["frame_field"] = "robot_render_frames"
        kwargs["robot_type"] = kwargs.get("robot_type") or "r1_lite_ego_retarget"

out = Path("$RUNTIME_CONFIG")
out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"wrote {out}")
print(f"dataset_path={raw.get('dataset_path')}")
print(f"export_path={raw.get('export_path')}")
PY

echo "=== dj-process config=$RUNTIME_CONFIG"
# Prefer installed entry point; fall back to tools/process_data.py
if command -v dj-process >/dev/null 2>&1; then
  dj-process --config "$RUNTIME_CONFIG" "$@"
else
  python tools/process_data.py --config "$RUNTIME_CONFIG" "$@"
fi

echo "Done."
echo "  processed:  $RUN_DIR/processed"
echo "  frames:     $RUN_DIR/frames"
echo "  robot:      $RUN_DIR/robot_frames"
echo "  lerobot:    $RUN_DIR/lerobot_dataset"
