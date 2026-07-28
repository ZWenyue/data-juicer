#!/usr/bin/env bash
# Acceptance: merge one task × one episode from Sim + Galaxea unified80.
#
#   bash tests_au/pipeline/robot_clean/accept_merge_lerobot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="${DJ_VENV:-/mnt/r/VENV/dj}/bin/python"
ROOT_SIM="${ROOT_SIM:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/process_clean/unified80}"
ROOT_GLX="${ROOT_GLX:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_clean/260711_unified80}"
OUT="${OUT:-$SCRIPT_DIR/outputs/merge_lerobot_accept}"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"

echo "=== accept_merge_lerobot: merge 1 task × 1 ep from each root ==="
"$PY" -m data_juicer._au.pipeline.robot_clean.merge_lerobot \
  --root "$ROOT_SIM" \
  --root "$ROOT_GLX" \
  --output "$OUT" \
  --video-policy keep \
  --link-mode symlink \
  --max-tasks-per-root 1 \
  --max-episodes-per-task 1

echo "=== validate ==="
"$PY" "$SCRIPT_DIR/accept_merge_lerobot.py" \
  --merged "$OUT" \
  --expect-episodes 2 \
  --expect-min-frames 2 \
  --require-videos

echo "ACCEPT PASS: $OUT"
