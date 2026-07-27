#!/usr/bin/env bash
# One-time setup: build R1 Lite arm MJCF assets (optional smoke).
#
# Usage:
#   bash b/scripts/hand2robot/setup.sh
#   SMOKE=1 bash b/scripts/hand2robot/setup.sh   # also run local smoke
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "=== [1/2] build MJCF assets"
bash "$SCRIPT_DIR/internal/01_build_assets.sh"

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "=== [2/2] local smoke"
  bash "$SCRIPT_DIR/internal/06_run_smoke.sh"
else
  echo "=== skip smoke (set SMOKE=1 to enable)"
fi

echo
echo "Setup done."
echo "  models: $MODEL_DIR/r1_lite_arm_{left,right}.xml"
echo "Next:  bash b/scripts/hand2robot/calibrate.sh   # if you need a new calib"
echo "Then:  bash b/scripts/hand2robot/process.sh"
