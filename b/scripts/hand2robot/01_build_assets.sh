#!/usr/bin/env bash
# Build single-arm MJCF assets (left + right) from R1 Lite URDF/STL.
# Usage:
#   bash b/scripts/hand2robot/01_build_assets.sh
#   SIDE=right SIDE_ONLY=1 bash b/scripts/hand2robot/01_build_assets.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

OUT="${ASSET_OUT:-$MODEL_DIR}"
SMOKE_DIR="${SMOKE_DIR:-$OUT/smoke}"

ARGS=(
  -m data_juicer._au.tools.build_r1_arm_mjcf
  --out-dir "$OUT"
)

if [[ "${SIDE_ONLY:-}" == "1" ]]; then
  ARGS+=(--sides "$SIDE")
else
  ARGS+=(--sides left right)
fi

if [[ "${SMOKE_RENDER:-1}" == "1" ]]; then
  mkdir -p "$SMOKE_DIR"
  ARGS+=(--smoke-dir "$SMOKE_DIR")
fi

echo "=== build MJCF → $OUT"
python "${ARGS[@]}"

echo "Done. Manifest: $OUT/asset_manifest.json"
ls -la "$OUT"/r1_lite_arm_*.xml 2>/dev/null || true
