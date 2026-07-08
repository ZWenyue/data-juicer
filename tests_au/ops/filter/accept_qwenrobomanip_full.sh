#!/usr/bin/env bash
# FULL acceptance test: Stage 1 + Stage 2 + Stage 3 (Filters) + Stage 5 (Mapper) in one pipeline.
# Reuses the existing convert / percentile helper scripts; does NOT modify accept_qwenrobomanip_filter.*.
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
echo "=== Step 3: Run dj-process (Stage 1 + 2 + 3 Filters + Stage 5 Mapper) ==="
"$VENV/bin/dj-process" --config "$SCRIPT_DIR/accept_qwenrobomanip_full.yaml"

echo ""
echo "=== Step 4: Verify outputs ==="
"$VENV/bin/python" - <<'PYEOF'
import json
import sys

import numpy as np

result_path = "tests_au/ops/filter/outputs/accept_full_result.jsonl"
s3_stats_path = "tests_au/ops/filter/outputs/accept_full_s3_stats.jsonl"
src_path = "tests_au/ops/filter/outputs/lerobot_episodes.jsonl"

with open(result_path) as f:
    results = [json.loads(l) for l in f if l.strip()]
with open(s3_stats_path) as f:
    stats = [json.loads(l) for l in f if l.strip()]
with open(src_path) as f:
    src = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

print(f"Input: {len(src)} episodes -> Output: {len(results)} episodes kept")

# ---- All 18 stats keys (7 Stage-1 + 6 Stage-2 + 5 Stage-3) still present ----
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
    d = s.get("__dj__stats__", s)
    for k in all_keys:
        if k not in d:
            errors.append(f"Row {i}: missing stats key '{k}'")

# ---- Stage 1/2/3 meta reports + valid_frame_mask still present after Stage 5 ----
for i, r in enumerate(results):
    meta = r.get("__dj__meta__", {})
    for k in ("sudden_change_report", "state_action_alignment_report",
              "extreme_value_report", "valid_frame_mask"):
        if k not in meta:
            errors.append(f"Row {i}: missing meta '{k}'")

# ---- Stage 5 (no pose_layout) must be a PURE NO-OP on joint-space data ----
#      i.e. states/actions unchanged AND no base_frame_alignment_report side-effect.
for r in results:
    ep = r["id"]
    if ep in src:
        s_in = np.array(src[ep]["states"], dtype=float)
        s_out = np.array(r["states"], dtype=float)
        a_in = np.array(src[ep]["actions"], dtype=float)
        a_out = np.array(r["actions"], dtype=float)
        if s_in.shape == s_out.shape and not np.allclose(s_in, s_out, atol=1e-9):
            errors.append(f"{ep}: Stage 5 modified joint-space states (should pass-through)")
        if a_in.shape == a_out.shape and not np.allclose(a_in, a_out, atol=1e-9):
            errors.append(f"{ep}: Stage 5 modified joint-space actions (should pass-through)")
    if "base_frame_alignment_report" in r.get("__dj__meta__", {}):
        errors.append(f"{ep}: Stage 5 wrote a report on pure pass-through (unexpected side-effect)")

if errors:
    for e in errors:
        print("  FAIL:", e, file=sys.stderr)
    sys.exit(1)

print(f"All {len(all_keys)} stats keys present in every row.")
print("Stage 1/2/3 meta reports + valid_frame_mask preserved after Stage 5.")
print("Stage 5 verified as pure pass-through (no data change, no side-effect) on joint-space data.")

assert len(stats) == 16, f"Expected 16 stats rows, got {len(stats)}"
assert len(results) == 16, f"Expected 16 result rows, got {len(results)}"

print("\nACCEPTANCE PASSED")
PYEOF
