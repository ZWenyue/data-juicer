#!/usr/bin/env bash
# Shared env for hand→robot scripts. Source from sibling scripts:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
set -euo pipefail

H2R_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# b/scripts/hand2robot → repo root is ../../..
REPO_ROOT="$(cd "$H2R_SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Prefer conda env "data_juicer", else DJ_VENV / .venv.
activate_dj_python() {
  if [[ -f /mnt/r/miniforge3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /mnt/r/miniforge3/etc/profile.d/conda.sh
    if conda env list 2>/dev/null | grep -qE '^data_juicer\s'; then
      conda activate data_juicer
      return 0
    fi
  fi
  if [[ -n "${DJ_VENV:-}" && -x "${DJ_VENV}/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "${DJ_VENV}/bin/activate"
    return 0
  fi
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
    return 0
  fi
  echo "[warn] using system python; set DJ_VENV or activate data_juicer" >&2
}

activate_dj_python

SIDE="${SIDE:-right}"
CALIB_DIR="${CALIB_DIR:-$REPO_ROOT/b/d/hand2robot/calibration}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/b/d/urdf/generated}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/b/d/hand2robot/runs}"
INIT_CALIB="${INIT_CALIB:-$CALIB_DIR/r1_${SIDE}_v1.yaml}"
MODEL_XML="${MODEL_XML:-$MODEL_DIR/r1_lite_arm_${SIDE}.xml}"

mkdir -p "$OUT_ROOT"

echo "[hand2robot] REPO_ROOT=$REPO_ROOT"
echo "[hand2robot] SIDE=$SIDE MUJOCO_GL=$MUJOCO_GL"
echo "[hand2robot] INIT_CALIB=$INIT_CALIB"
echo "[hand2robot] MODEL_XML=$MODEL_XML"
