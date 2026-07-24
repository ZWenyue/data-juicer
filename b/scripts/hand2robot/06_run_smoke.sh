#!/usr/bin/env bash
# End-to-end local smoke: build assets → synthetic calib → render accept → calib accept.
# Usage:
#   bash b/scripts/hand2robot/06_run_smoke.sh
#   SIDE=left bash b/scripts/hand2robot/06_run_smoke.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/b/d/hand2robot/runs/smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"
echo "=== smoke OUT_ROOT=$OUT_ROOT"

SIDE_ONLY=1 bash "$SCRIPT_DIR/01_build_assets.sh"
REPORT_DIR="$OUT_ROOT/calib_synthetic" bash "$SCRIPT_DIR/02_calibrate_synthetic.sh"
OUT_DIR="$OUT_ROOT/accept_render" bash "$SCRIPT_DIR/05_accept_render.sh"
OUT_DIR="$OUT_ROOT/accept_calibrate" bash "$SCRIPT_DIR/05_accept_calibrate.sh" synthetic

echo "==================================================================="
echo "Smoke finished under: $OUT_ROOT"
ls -la "$OUT_ROOT"
