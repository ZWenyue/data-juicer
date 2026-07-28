#!/usr/bin/env bash
# Acceptance test for robot_sudden_change_filter with real LeRobot dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV=/mnt/r/VENV/dj
DATASET=/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002

cd "$REPO_ROOT"

echo "=== Step 1: Convert LeRobot parquet -> per-episode JSONL ==="
"$VENV/bin/python" "$SCRIPT_DIR/convert_lerobot_episodes.py" \
    --dataset_dir "$DATASET" \
    --output "$SCRIPT_DIR/outputs/lerobot_episodes.jsonl"

echo ""
echo "=== Step 2: Run dj-process ==="
"$VENV/bin/dj-process" --config "$SCRIPT_DIR/accept_robot_sudden_change_filter.yaml"

echo ""
echo "=== Step 3: Verify outputs ==="
"$VENV/bin/python" - <<'PYEOF'
import json, sys

result_path = "tests_au/ops/filter/outputs/accept_result.jsonl"
stats_path  = "tests_au/ops/filter/outputs/accept_stats.jsonl"

with open(result_path) as f:
    results = [json.loads(l) for l in f if l.strip()]
with open(stats_path) as f:
    stats = [json.loads(l) for l in f if l.strip()]

print(f"Input: 16 episodes -> Output: {len(results)} episodes kept")
print(f"Stats exported: {len(stats)} rows")

required_keys = [
    "sudden_change_keep", "sudden_change_flagged_ratio",
    "sudden_change_num_flagged", "sudden_change_max_run",
    "sudden_change_max_residual", "sudden_change_max_acc",
    "sudden_change_max_jerk",
]

errors = []
for i, s in enumerate(stats):
    stat_dict = s.get("__dj__stats__", s)
    for k in required_keys:
        if k not in stat_dict:
            errors.append(f"Row {i}: missing key '{k}'")

if errors:
    for e in errors:
        print(f"  FAIL: {e}", file=sys.stderr)
    sys.exit(1)

print("All 7 stats keys present in every row.")

assert len(stats) == 16, f"Expected 16 stats rows, got {len(stats)}"
assert len(results) == 16, f"Expected 16 result rows (frame_mask keeps all), got {len(results)}"

kept_ids = [r["id"] for r in results]
print(f"Output episodes: {kept_ids}")
print("ACCEPTANCE PASSED")
PYEOF
