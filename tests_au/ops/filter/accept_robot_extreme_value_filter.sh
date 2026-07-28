#!/usr/bin/env bash
# Acceptance test for robot_extreme_value_filter (Stage 3) with real LeRobot dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV=/mnt/r/VENV/dj
DATASET=/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002

if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET" >&2
    exit 1
fi

cd "$REPO_ROOT"

echo "=== Step 1: Convert LeRobot parquet -> per-episode JSONL ==="
"$VENV/bin/python" "$SCRIPT_DIR/convert_lerobot_episodes.py" \
    --dataset_dir "$DATASET" \
    --output "$SCRIPT_DIR/outputs/lerobot_episodes.jsonl"

echo ""
echo "=== Step 2: Precompute per-embodiment percentiles ==="
"$VENV/bin/python" "$SCRIPT_DIR/compute_embodiment_percentiles.py" \
    --dataset_dir "$DATASET" \
    --output "$SCRIPT_DIR/outputs/percentiles.json" \
    --embodiment galaxea_r1_lite

echo ""
echo "=== Step 3: Run dj-process (Stage 3 only) ==="
"$VENV/bin/dj-process" --config "$SCRIPT_DIR/accept_robot_extreme_value_filter.yaml"

echo ""
echo "=== Step 4: Verify outputs ==="
"$VENV/bin/python" - <<'PYEOF'
import json, sys

result_path = "tests_au/ops/filter/outputs/accept_s3_result.jsonl"
stats_path  = "tests_au/ops/filter/outputs/accept_s3_stats.jsonl"

with open(result_path) as f:
    results = [json.loads(l) for l in f if l.strip()]
with open(stats_path) as f:
    stats = [json.loads(l) for l in f if l.strip()]

print(f"Input: 16 episodes -> Output: {len(results)} episodes kept")
print(f"Stats exported: {len(stats)} rows")

required_keys = [
    "extreme_value_keep", "extreme_value_flagged_frames",
    "extreme_value_flagged_ratio", "extreme_value_num_check_dims",
    "extreme_value_num_frames",
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

print(f"All {len(required_keys)} stats keys present in every row.")

# Check meta reports
meta_errors = []
for i, r in enumerate(results):
    meta = r.get("__dj__meta__", {})
    if "extreme_value_report" not in meta:
        meta_errors.append(f"Row {i}: missing meta 'extreme_value_report'")
    if "valid_frame_mask" not in meta:
        meta_errors.append(f"Row {i}: missing meta 'valid_frame_mask'")

if meta_errors:
    for e in meta_errors:
        print(f"  FAIL: {e}", file=sys.stderr)
    sys.exit(1)

print("Meta report + valid_frame_mask present in every row.")

assert len(stats) == 16, f"Expected 16 stats rows, got {len(stats)}"
assert len(results) == 16, f"Expected 16 result rows (frame_mask keeps all), got {len(results)}"

# Per-episode summary
print("\n--- Per-episode Stage 3 summary ---")
print(f"{'Episode':<20s}  {'ev_ratio':>8s}  {'ev_frames':>9s}  {'check_dims':>10s}  {'total':>6s}")
for s in stats:
    d = s.get("__dj__stats__", s)
    ep_id = s.get("id", "?")
    print(f"{ep_id:<20s}  "
          f"{d['extreme_value_flagged_ratio']:8.4f}  "
          f"{d['extreme_value_flagged_frames']:9d}  "
          f"{d['extreme_value_num_check_dims']:10d}  "
          f"{d['extreme_value_num_frames']:6d}")

print("\nACCEPTANCE PASSED")
PYEOF
