#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="${DJ_VENV:-$REPO_ROOT/.venv}/bin/python"
DATASET="${DATASET_ROOT:-/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002}"
OUTPUT="$(mktemp -d "$REPO_ROOT/.robot-task-filter-XXXXXX")"
trap 'rm -rf "$OUTPUT"' EXIT

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$SCRIPT_DIR/accept_robot_task_filter.py" \
  --dataset "$DATASET" \
  --output "$OUTPUT"
