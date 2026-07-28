#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${1:-b/d/hand2robot/calib_out}"
SIDE="${2:-right}"
MODE="${3:-}"  # empty=auto Galaxea|synthetic, "synthetic", or path to ego sample / lerobot root
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ -f /mnt/r/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /mnt/r/miniforge3/etc/profile.d/conda.sh
  conda activate data_juicer
fi

ARGS=(--output-dir "$OUT_DIR" --side "$SIDE" --gl-backend "$MUJOCO_GL")
if [[ "$MODE" == "synthetic" ]]; then
  ARGS+=(--synthetic)
elif [[ -n "$MODE" && -d "$MODE/data" ]]; then
  ARGS+=(--lerobot-root "$MODE")
elif [[ -n "$MODE" ]]; then
  ARGS+=(--data-path "$MODE")
fi

python tests_au/ops/mapper/accept_calibrate_hand_to_robot.py "${ARGS[@]}"
