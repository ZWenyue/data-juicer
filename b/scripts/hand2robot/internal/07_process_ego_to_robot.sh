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
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_env.sh"

CONFIG="${CONFIG:-$H2R_SCRIPT_DIR/configs/ego_to_robot_recipe.yaml}"
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
if [[ "${SIDE}" == "both" ]]; then
  MODEL_XML_RIGHT="${MODEL_XML_RIGHT:-$MODEL_DIR/r1_lite_arm_right.xml}"
  MODEL_XML_LEFT="${MODEL_XML_LEFT:-$MODEL_DIR/r1_lite_arm_left.xml}"
  [[ -f "$MODEL_XML_RIGHT" ]] || { echo "missing model: $MODEL_XML_RIGHT" >&2; exit 1; }
  [[ -f "$MODEL_XML_LEFT" ]] || { echo "missing model: $MODEL_XML_LEFT" >&2; exit 1; }
else
  [[ -f "$MODEL_XML" ]] || {
    echo "missing model: $MODEL_XML — run setup.sh first" >&2
    exit 1
  }
fi

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
model_xml_right = "${MODEL_XML_RIGHT:-}"
model_xml_left = "${MODEL_XML_LEFT:-}"
run_dir = Path("$RUN_DIR")
moge_model = "${MOGE_MODEL_PATH:-Ruicheng/moge-2-vitl}"
mano_right = "${MANO_RIGHT_PATH:-/mnt/r/share/zwy/Projects/mano_v1_2/models/MANO_RIGHT.pkl}"
mano_left = "${MANO_LEFT_PATH:-/mnt/r/share/zwy/Projects/mano_v1_2/models/MANO_LEFT.pkl}"
hawor_ckpt = "${HAWOR_MODEL_PATH:-$HOME/.cache/data_juicer/models/HaWor/hawor.ckpt}"
hawor_cfg = "${HAWOR_CONFIG_PATH:-$HOME/.cache/data_juicer/models/HaWor/model_config.yaml}"
hawor_det = "${HAWOR_DETECTOR_PATH:-$HOME/.cache/data_juicer/models/HaWor/detector.pt}"
frame_num = int("${FRAME_NUM:-30}")
frame_method = "${FRAME_SAMPLING_METHOD:-uniform}"

def _as_export_file(path: str, default_name: str = "processed.jsonl") -> str:
    """dj-process Exporter requires .jsonl/.json/.parquet suffix."""
    p = Path(path)
    if p.suffix.lower() in {".jsonl", ".json", ".parquet"}:
        return str(p)
    # Directory path → write default_name inside it.
    if path.endswith("/") or path.endswith(chr(92)) or p.is_dir():
        return str(p / default_name)
    # Extension-less path (e.g. .../processed) → .../processed.jsonl
    return str(p) + ".jsonl"

if dataset:
    raw["dataset_path"] = dataset
if export_path:
    raw["export_path"] = _as_export_file(export_path)
else:
    raw["export_path"] = str(run_dir / "processed.jsonl")

# Fail loud by default (override with SKIP_OP_ERROR=1)
raw["skip_op_error"] = "${SKIP_OP_ERROR:-0}" in ("1", "true", "True")

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
        kwargs["frame_sampling_method"] = frame_method
        if frame_method == "uniform":
            kwargs["frame_num"] = frame_num
    elif name == "video_camera_calibration_moge_mapper":
        kwargs["model_path"] = moge_model
    elif name == "video_hand_reconstruction_hawor_mapper":
        kwargs["hawor_model_path"] = hawor_ckpt
        kwargs["hawor_config_path"] = hawor_cfg
        kwargs["hawor_detector_path"] = hawor_det
        kwargs["mano_right_path"] = mano_right
        kwargs["mano_left_path"] = mano_left
    elif name == "video_hand_action_compute_mapper":
        kwargs["hand_type"] = side
    elif name == "video_hand_to_robot_render_mapper":
        kwargs["hand_type"] = side
        if side == "both":
            kwargs["robot_model_paths"] = {"right": model_xml_right, "left": model_xml_left}
        else:
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

# Executor: Ray is required for MegaSaM's conda runtime_env (lietorch).
# Override with EXECUTOR_TYPE=default only for debugging (MegaSaM will fail
# unless the active env already has lietorch).
executor_type = "${EXECUTOR_TYPE:-}"
if executor_type:
    raw["executor_type"] = executor_type
if raw.get("executor_type") == "ray":
    raw.setdefault("ray_address", "auto")
    for item in process:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        name, kwargs = next(iter(item.items()))
        if name == "video_camera_pose_megasam_mapper" and isinstance(kwargs, dict):
            kwargs.setdefault("runtime_env", {"conda": "mega-sam"})

out = Path("$RUNTIME_CONFIG")
out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"wrote {out}")
print(f"dataset_path={raw.get('dataset_path')}")
print(f"export_path={raw.get('export_path')}")
print(f"executor_type={raw.get('executor_type')} moge_model={moge_model} frame={frame_method}/{frame_num} skip_op_error={raw['skip_op_error']}")
PY

# Ray + mega-sam: lietorch lives only in conda env mega-sam.
need_ray=0
if [[ "${EXECUTOR_TYPE:-}" == "ray" ]]; then
  need_ray=1
elif [[ -z "${EXECUTOR_TYPE:-}" ]] && grep -qE '^executor_type:[[:space:]]*['\''"]?ray' "$RUNTIME_CONFIG" 2>/dev/null; then
  need_ray=1
fi
if [[ "$need_ray" == "1" ]]; then
  if ! ray status >/dev/null 2>&1; then
    echo "=== starting local Ray head (required for MegaSaM runtime_env=mega-sam)"
    ray stop --force >/dev/null 2>&1 || true
    ray start --head
  else
    echo "=== Ray cluster already up"
  fi
  # Sanity: mega-sam must import lietorch
  if ! conda run -n mega-sam python -c "import lietorch, droid_backends" >/dev/null 2>&1; then
    echo "ERROR: conda env mega-sam missing lietorch/droid_backends." >&2
    echo "  See b/vla_pipeline_env_setup.md §3" >&2
    exit 1
  fi
fi

echo "=== dj-process config=$RUNTIME_CONFIG"
# Prefer installed entry point; fall back to tools/process_data.py
if command -v dj-process >/dev/null 2>&1; then
  dj-process --config "$RUNTIME_CONFIG" "$@"
else
  python tools/process_data.py --config "$RUNTIME_CONFIG" "$@"
fi

echo "Done."
echo "  processed:  $RUN_DIR/processed.jsonl"
echo "  frames:     $RUN_DIR/frames"
if [[ -d "$RUN_DIR/robot_frames" ]] && find "$RUN_DIR/robot_frames" -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | grep -q .; then
  n_robot=$(find "$RUN_DIR/robot_frames" -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l | tr -d ' ')
  echo "  robot:      $RUN_DIR/robot_frames  ($n_robot images)"
else
  echo "  robot:      MISSING — $RUN_DIR/robot_frames was not created" >&2
  echo "  (dj-process 'Done' only means the job exited; render may have skipped.)" >&2
fi
echo "  lerobot:    $RUN_DIR/lerobot_dataset"
