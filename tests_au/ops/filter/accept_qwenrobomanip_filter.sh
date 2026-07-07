#!/usr/bin/env bash
# Combined acceptance test: Stage 1 (Sudden Change) + Stage 2 (State-Action Alignment) + Stage 3 (Extreme Value Filtering)
# Validates that all three custom filters run correctly in a single dj-process pipeline.
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
echo "=== Step 2: Precompute per-embodiment percentiles (for Stage 3) ==="
"$VENV/bin/python" "$SCRIPT_DIR/compute_embodiment_percentiles.py" \
    --dataset_dir "$DATASET" \
    --output "$SCRIPT_DIR/outputs/percentiles.json" \
    --embodiment galaxea_r1_lite

echo ""
echo "=== Step 3: Run dj-process (Stage 1 + Stage 2 + Stage 3 cascade) ==="
"$VENV/bin/dj-process" --config "$SCRIPT_DIR/accept_qwenrobomanip_filter.yaml"

echo ""
echo "=== Step 4: Verify outputs ==="
"$VENV/bin/python" - <<'PYEOF'
import json, sys

result_path = "tests_au/ops/filter/outputs/accept_combined_result.jsonl"
s3_stats_path = "tests_au/ops/filter/outputs/accept_combined_s3_stats.jsonl"

with open(result_path) as f:
    results = [json.loads(l) for l in f if l.strip()]
with open(s3_stats_path) as f:
    stats = [json.loads(l) for l in f if l.strip()]

print(f"Input: 16 episodes -> Output: {len(results)} episodes kept")
print(f"Stats exported (after Stage 3): {len(stats)} rows")

# ---- Check all 18 stats keys (7 Stage-1 + 6 Stage-2 + 5 Stage-3) ----
s1_keys = [
    "sudden_change_keep", "sudden_change_flagged_ratio",
    "sudden_change_num_flagged", "sudden_change_max_run",
    "sudden_change_max_residual", "sudden_change_max_acc",
    "sudden_change_max_jerk",
]
s2_keys = [
    "state_action_alignment_keep", "state_action_min_da",
    "state_action_mean_da", "state_action_num_flagged_dims",
    "state_action_num_checked_dims", "state_action_max_abs_lag",
]
s3_keys = [
    "extreme_value_keep", "extreme_value_flagged_frames",
    "extreme_value_flagged_ratio", "extreme_value_num_check_dims",
    "extreme_value_num_frames",
]
all_keys = s1_keys + s2_keys + s3_keys

errors = []
for i, s in enumerate(stats):
    stat_dict = s.get("__dj__stats__", s)
    for k in all_keys:
        if k not in stat_dict:
            errors.append(f"Row {i}: missing stats key '{k}'")

if errors:
    for e in errors:
        print(f"  FAIL: {e}", file=sys.stderr)
    sys.exit(1)

print(f"All {len(all_keys)} stats keys present in every row.")

# ---- Check meta reports ----
meta_errors = []
for i, r in enumerate(results):
    meta = r.get("__dj__meta__", {})
    if "sudden_change_report" not in meta:
        meta_errors.append(f"Row {i}: missing meta 'sudden_change_report'")
    if "state_action_alignment_report" not in meta:
        meta_errors.append(f"Row {i}: missing meta 'state_action_alignment_report'")
    if "extreme_value_report" not in meta:
        meta_errors.append(f"Row {i}: missing meta 'extreme_value_report'")
    if "valid_frame_mask" not in meta:
        meta_errors.append(f"Row {i}: missing meta 'valid_frame_mask'")

if meta_errors:
    for e in meta_errors:
        print(f"  FAIL: {e}", file=sys.stderr)
    sys.exit(1)

print("All 3 meta reports + valid_frame_mask present in every row.")

# ---- Counts ----
assert len(stats) == 16, f"Expected 16 stats rows, got {len(stats)}"
assert len(results) == 16, (
    f"Expected 16 result rows (frame_mask+flag_only keeps all), got {len(results)}"
)

# ---- Per-episode summary ----
print("\n--- Per-episode combined summary (Stage 1 + Stage 2 + Stage 3) ---")
print(f"{'Episode':<20s}  {'flagged%':>8s}  {'min_DA':>7s}  {'mean_DA':>8s}  "
      f"{'checked':>7s}  {'flagged':>7s}  {'max_lag':>7s}  "
      f"{'ev_ratio':>8s}  {'ev_frames':>9s}")
for s in stats:
    d = s.get("__dj__stats__", s)
    ep_id = s.get("id", "?")
    print(f"{ep_id:<20s}  "
          f"{d['sudden_change_flagged_ratio']:8.3f}  "
          f"{d['state_action_min_da']:7.3f}  "
          f"{d['state_action_mean_da']:8.3f}  "
          f"{d['state_action_num_checked_dims']:7d}  "
          f"{d['state_action_num_flagged_dims']:7d}  "
          f"{d['state_action_max_abs_lag']:7d}  "
          f"{d['extreme_value_flagged_ratio']:8.4f}  "
          f"{d['extreme_value_flagged_frames']:9d}")

print("\nACCEPTANCE PASSED")
PYEOF
