#!/usr/bin/env bash
# Acceptance: Check 1 Instruction Consistency (Cursor SDK pipeline).
# Unit logic is covered by pytest; this script hits live Cursor API when key is set.
#
# Env:
#   CURSOR_API_KEY   required for live run (else exit 0 skip)
#   DATA_DIR         LeRobot task dir (default: Connect_Router_Cables tst set)
#   ROOT             optional multi-task root; if set, first task under it is used
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# tests_au/pipeline -> repo root
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
PY="${VENV}/bin/python"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/outputs}"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Acceptance Test: Instruction Consistency (Cursor SDK) ==="

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python not found at $PY (activate/create .venv first)" >&2
  exit 1
fi

if [[ -z "${CURSOR_API_KEY:-}" ]]; then
  echo "CURSOR_API_KEY not set — skipping live API acceptance (unit tests still valid)."
  exit 0
fi

"$PY" -c "import cursor_sdk" 2>/dev/null || {
  echo "ERROR: cursor-sdk missing in $VENV. Install with:" >&2
  echo "  uv pip install --python $PY cursor-sdk" >&2
  exit 1
}

# Prefer ROOT (e.g. process_test) else single DATA_DIR
if [[ -n "${ROOT:-}" && -d "$ROOT" ]]; then
  # First task dir that contains data/ (avoid pipefail+SIGPIPE)
  DATA_DIR=""
  while IFS= read -r d; do
    if [[ -d "$d/data" ]]; then DATA_DIR="$d"; break; fi
  done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
fi
DATA_DIR="${DATA_DIR:-/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "WARNING: dataset not found at $DATA_DIR — skip."
  exit 0
fi

VIDEO=$(find "$DATA_DIR" \( -name "*.mp4" -o -name "*.avi" \) -print -quit || true)
if [[ -z "${VIDEO:-}" ]]; then
  echo "WARNING: No video under $DATA_DIR — skip."
  exit 0
fi
echo "DATA_DIR=$DATA_DIR"
echo "Video: $VIDEO"

POSITIONS="$OUT_DIR/accept_ic_eef_positions.npy"
export DATA_DIR
echo "=== Extract EE positions (xyz) from parquet for Stage 1 segmentation ==="
"$PY" - <<PY
import glob, os
import numpy as np
import pyarrow.parquet as pq

data_dir = os.environ["DATA_DIR"]
out = "$POSITIONS"
pfs = sorted(glob.glob(os.path.join(data_dir, "data", "chunk-*", "episode_*.parquet")))
assert pfs, f"no parquet under {data_dir}"
# Align with first video episode index if possible
base = os.path.splitext(os.path.basename("$VIDEO"))[0]  # episode_000040
cands = [p for p in pfs if base in p] or pfs
df = pq.read_table(cands[0]).to_pandas()
cols = list(df.columns)

def stack_xyz(col):
    return np.stack([np.asarray(v, dtype=float).reshape(-1)[:3] for v in df[col]])

if "observation.state" in cols:
    # unified 80: left EE xyz at [7:10]
    s = np.stack([np.asarray(v, dtype=float) for v in df["observation.state"]])
    arr = s[:, 7:10]
elif "observation.state.left_ee_pose" in cols:
    arr = stack_xyz("observation.state.left_ee_pose")
else:
    raise SystemExit(f"unsupported columns: {cols[:12]}...")

np.save(out, arr)
print(f"parquet={cands[0]}")
print(f"positions shape={arr.shape} -> {out}")
PY

INSTRUCTION="${INSTRUCTION:-connect the router cables}"
OUT_JSON="$OUT_DIR/accept_instruction_consistency_result.json"

echo ""
echo "=== Run Check 1 pipeline (composer-2.5, 2 experts for Stage 3 quorum) ==="
# Coarser Stage-1 cuts + ratio aggregation: avoid 20+ idle micro-segments / veto.
"$PY" -m data_juicer._au.pipeline.instruction_consistency.run_instruction_consistency \
  --video "$VIDEO" \
  --instruction "$INSTRUCTION" \
  --positions "$POSITIONS" \
  --primary-model "composer-2.5" \
  --expert-models "composer-2.5" "composer-2.5" \
  --voting-strategy majority \
  --confidence-threshold 0.7 \
  --num-frames 4 \
  --min-window 30 \
  --min-segment-frames 45 \
  --max-segments 6 \
  --inconsistent-ratio-threshold 0.35 \
  --output "$OUT_JSON"

echo ""
echo "Result:"
"$PY" -m json.tool "$OUT_JSON"
echo "=== ACCEPTANCE DONE ==="
