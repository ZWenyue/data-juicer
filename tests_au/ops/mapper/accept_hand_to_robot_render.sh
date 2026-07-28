#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${1:-b/d/hand2robot/accept_out}"
SIDE="${2:-right}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Prefer project data_juicer conda env when available.
if [[ -f /mnt/r/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /mnt/r/miniforge3/etc/profile.d/conda.sh
  conda activate data_juicer
fi

python tests_au/ops/mapper/accept_hand_to_robot_render.py \
  --output-dir "$OUT_DIR" \
  --side "$SIDE" \
  --gl-backend "$MUJOCO_GL"
