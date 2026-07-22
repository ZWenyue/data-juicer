#!/usr/bin/env bash
# Acceptance: pad-only unified80 export (no cleaning), keep all episodes.
#
#   bash tests_au/pipeline/robot_clean/accept_pad_unified.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="${DJ_VENV:-$REPO_ROOT/.venv}/bin/python"
SRC="${SRC:-/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002}"
OUT="${OUT:-$SCRIPT_DIR/outputs/pad_unified_accept}"
EMBODIMENT="${EMBODIMENT:-galaxea_r1_lite}"
MAX_EPS="${MAX_EPS:-2}"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$SRC/data" ]]; then
  echo "ERROR: SRC not found or missing data/: $SRC" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "ERROR: python not executable: $PY" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"

echo "=== accept_pad_unified: keep-all export (max_episodes=$MAX_EPS) ==="
"$PY" -m data_juicer._au.pipeline.robot_clean.export_unified \
  --keep-all \
  --dataset "$SRC" \
  --output "$OUT" \
  --embodiment "$EMBODIMENT" \
  --max-episodes "$MAX_EPS"

TASK_OUT="$OUT/$(basename "$SRC")"
if [[ ! -d "$TASK_OUT/data" ]]; then
  # When --output already ends with task name, files go directly under OUT.
  TASK_OUT="$OUT"
fi

echo "=== validate ==="
"$PY" "$SCRIPT_DIR/accept_pad_unified.py" \
  --out "$TASK_OUT" \
  --src "$SRC" \
  --expect-episodes "$MAX_EPS" \
  --require-videos

echo "ACCEPT PASS: $TASK_OUT"
