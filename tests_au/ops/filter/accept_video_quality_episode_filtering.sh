#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${DJ_VENV:-/mnt/r/VENV/dj}"
DATASET="/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

if [ -x "$VENV/bin/python" ]; then
    PYTHON="$VENV/bin/python"
    DJ_PROCESS="$VENV/bin/dj-process"
else
    PYTHON="$(which python3)"
    DJ_PROCESS="$(which dj-process)"
fi

cd "$REPO_ROOT"

echo "=== Step 1: Ensure episode JSONL exists ==="
EPISODES_JSONL="$OUTPUT_DIR/lerobot_episodes.jsonl"
if [ ! -f "$EPISODES_JSONL" ]; then
    "$PYTHON" "$SCRIPT_DIR/convert_lerobot_episodes.py" \
        --dataset_dir "$DATASET" \
        --output "$EPISODES_JSONL"
    echo "Converted $(wc -l < "$EPISODES_JSONL") episodes."
else
    echo "Episodes JSONL already exists ($(wc -l < "$EPISODES_JSONL") lines), skipping conversion."
fi

echo ""
echo "=== Step 2: Run Episode-Level Quality Pipeline ==="
"$DJ_PROCESS" --config "$SCRIPT_DIR/accept_video_quality_episode_filtering.yaml"

echo ""
echo "=== Step 3: Verify results ==="
"$PYTHON" <<'PYEOF'
import json
import sys

result_path = "tests_au/ops/filter/outputs/check3_episode_result.jsonl"

with open(result_path) as f:
    results = [json.loads(line) for line in f]

print(f"Kept episodes: {len(results)}")

errors = []
for i, row in enumerate(results):
    meta = row.get("__dj__meta__", {})
    stats = row.get("__dj__stats__", {})
    ep_id = row.get("id", f"row_{i}")

    # Check key_frame_report present
    if "key_frame_report" not in meta:
        errors.append(f"{ep_id}: missing meta key 'key_frame_report'")

    # Check episode filter report present
    if "video_quality_episode_report" not in meta:
        errors.append(f"{ep_id}: missing meta key 'video_quality_episode_report'")

    # Check stats
    for sk in ["video_quality_episode_keep",
               "video_quality_episode_bad_ratio",
               "video_quality_episode_good_frames",
               "video_quality_episode_keyframe_overlap"]:
        if sk not in stats:
            errors.append(f"{ep_id}: missing stats key '{sk}'")

    # All kept episodes must have keep=True
    if not stats.get("video_quality_episode_keep", False):
        errors.append(f"{ep_id}: kept but stats say keep=False")

    # Report consistency
    report_str = meta.get("video_quality_episode_report")
    if report_str:
        report = json.loads(report_str) if isinstance(report_str, str) else report_str
        if not report.get("keep", False):
            errors.append(f"{ep_id}: kept but report says keep=False")
        bad_ratio = report.get("bad_ratio", 0)
        good_frames = report.get("good_frames", 0)
        print(f"  {ep_id}: bad_ratio={bad_ratio:.3f}, good_frames={good_frames}")

if errors:
    print(f"\n*** {len(errors)} ERRORS ***")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"\nAll {len(results)} kept episodes verified.")
    print("ACCEPTANCE PASSED")
PYEOF
