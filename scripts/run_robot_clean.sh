#!/usr/bin/env bash
# Thin wrapper around the production robot clean CLI.
#
# Examples:
#   bash scripts/run_robot_clean.sh \
#     --dataset /mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002 \
#     --output /tmp/robot_clean_out
#
#   bash scripts/run_robot_clean.sh \
#     --dataset ... --output ... --max-episodes 8 --export-unified-parquet
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${DJ_VENV:-${VENV:-/mnt/r/VENV/dj}}"

if [[ -x "$VENV/bin/python" ]]; then
  PYTHON="$VENV/bin/python"
else
  PYTHON="$(command -v python3)"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

exec "$PYTHON" -m data_juicer._au.pipeline.robot_clean.run_robot_clean "$@"
