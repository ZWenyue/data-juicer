#!/usr/bin/env bash
# Acceptance: Galaxea meta → prompt_fields on DJ samples (+ optional unified export enrichment).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV=/mnt/r/VENV/dj
DATA_ROOT="${DATA_ROOT:-/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002}"
OUT_DIR="$SCRIPT_DIR/outputs"
POINTER="$OUT_DIR/prompt_pointer.jsonl"
MAX_EPISODES="${MAX_EPISODES:-2}"
EXPORT_OUT="$OUT_DIR/prompt_unified_lerobot"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

if [[ ! -d "$DATA_ROOT/data" ]]; then
  echo "FAIL: dataset not found at $DATA_ROOT" >&2
  exit 1
fi

echo "=== Step 1: Build pointer JSONL (max $MAX_EPISODES episodes) ==="
"$VENV/bin/python" - <<PY
import glob, json, os
root = "$DATA_ROOT"
out = "$POINTER"
files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet")))
files = files[: int("$MAX_EPISODES")]
assert files, f"no parquet under {root}"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    for i, p in enumerate(files):
        f.write(json.dumps({"id": f"ep_{i:04d}", "parquet_path": p, "text": f"ep_{i:04d}"}) + "\n")
print(f"wrote {len(files)} pointers -> {out}")
PY

echo ""
echo "=== Step 2: dj-process (embodiment prompt mapper) ==="
rm -f "$OUT_DIR"/accept_prompt_result.jsonl*
"$VENV/bin/dj-process" --config "$SCRIPT_DIR/accept_robot_embodiment_prompt_mapper.yaml"

echo ""
echo "=== Step 3: Export unified LeRobot meta with prompt_fields ==="
rm -rf "$EXPORT_OUT"
"$VENV/bin/python" "$SCRIPT_DIR/export_unified_parquets.py" \
  --root "$DATA_ROOT" \
  --out "$EXPORT_OUT" \
  --embodiment galaxea_r1_lite \
  --max_tasks 1 \
  --max_episodes "$MAX_EPISODES"

EPISODES_JSONL="$(find "$EXPORT_OUT" -path '*/meta/episodes.jsonl' | head -n 1)"
if [[ -z "$EPISODES_JSONL" ]]; then
  echo "FAIL: episodes.jsonl not found under $EXPORT_OUT" >&2
  exit 1
fi

echo ""
echo "=== Step 4: Verify ==="
"$VENV/bin/python" "$SCRIPT_DIR/accept_robot_embodiment_prompt_mapper.py" \
  --dj-export-glob "$OUT_DIR/accept_prompt_result.jsonl*" \
  --episodes-jsonl "$EPISODES_JSONL"
