#!/usr/bin/env bash
# Pilot: convert → percentiles → dj-analyze → threshold report
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
DATASET="${DATASET:-/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot/Connect_Router_Cables_20250625_002}"
OUT="$SCRIPT_DIR/outputs/pilot_analyze"

if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET" >&2
    exit 1
fi
if [ ! -x "$VENV/bin/python" ] || [ ! -x "$VENV/bin/dj-analyze" ]; then
    echo "ERROR: Need python and dj-analyze in $VENV" >&2
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$OUT"

# Keep HuggingFace/datasets caches inside the workspace so acceptance can run in
# sandboxed or shared environments without writing to a user-level cache.
export HF_HOME="${HF_HOME:-$OUT/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$OUT/.hf_datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OUT/.cache}"
export TMPDIR="${TMPDIR:-/tmp/dj_pilot_accept_${USER:-user}}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

echo "=== Step 1: Convert LeRobot parquet -> per-episode JSONL ==="
"$VENV/bin/python" "$SCRIPT_DIR/convert_lerobot_episodes.py" \
    --dataset_dir "$DATASET" \
    --output "$OUT/lerobot_episodes.jsonl"

echo ""
echo "=== Step 2: Precompute per-embodiment percentiles (for Stage 3) ==="
"$VENV/bin/python" "$SCRIPT_DIR/compute_embodiment_percentiles.py" \
    --dataset_dir "$DATASET" \
    --output "$OUT/percentiles.json" \
    --embodiment galaxea_r1_lite

echo ""
echo "=== Step 3: Run dj-analyze (Stage 1 + Stage 2 + Stage 3 stats) ==="
rm -f "$OUT/analyze_result.jsonl" "$OUT/analyze_result_stats.jsonl"
if ! "$VENV/bin/dj-analyze" --config "$SCRIPT_DIR/analyze_qwenrobomanip_pilot.yaml"; then
    if [ -s "$OUT/analyze_result_stats.jsonl" ]; then
        echo "WARNING: dj-analyze failed after stats export; continuing with analyze_result_stats.jsonl" >&2
    else
        echo "ERROR: dj-analyze failed before stats export" >&2
        exit 1
    fi
fi

echo ""
echo "=== Step 4: Summarize thresholds ==="
"$VENV/bin/python" - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "tests_au/ops/filter")
from summarize_threshold_report import (
    REQUIRED_KEYS,
    _stat_dict,
    load_stats_jsonl,
    merge_stats_jsonl,
    suggest_thresholds,
    write_report,
)

out = Path("tests_au/ops/filter/outputs/pilot_analyze")
merged_path = out / "merged_stats.jsonl"
s1, s2, s3 = out / "s1_stats.jsonl", out / "s2_stats.jsonl", out / "s3_stats.jsonl"
analyze_result = out / "analyze_result.jsonl"
analyze_stats = out / "analyze_result_stats.jsonl"

rows = None
if s1.exists() and s2.exists() and s3.exists():
    rows = merge_stats_jsonl([str(s1), str(s2), str(s3)])
    print(f"Merged stats from s1/s2/s3 ({len(rows)} episodes)")
else:
    stats_source = analyze_result if analyze_result.exists() else analyze_stats
    try:
        rows = load_stats_jsonl(str(stats_source))
        print(f"Loaded stats from {stats_source.name} ({len(rows)} episodes)")
    except Exception as exc:
        if s1.exists() or s2.exists() or s3.exists():
            partial = [str(p) for p in (s1, s2, s3) if p.exists()]
            rows = merge_stats_jsonl(partial)
            print(f"Partial merge from {len(partial)} stage file(s) ({len(rows)} episodes): {exc}")
        else:
            raise

# If analyze_result exists but lacks cascade keys, prefer full s1+s2+s3 merge.
if s1.exists() and s2.exists() and s3.exists():
    missing_any = False
    for row in rows:
        stats = _stat_dict(row)
        if any(key not in stats for key in REQUIRED_KEYS):
            missing_any = True
            break
    if missing_any:
        rows = merge_stats_jsonl([str(s1), str(s2), str(s3)])
        print(f"Re-merged s1/s2/s3 due to missing stage keys ({len(rows)} episodes)")

with open(merged_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

sug = suggest_thresholds(rows, probe_alpha=0.1)
write_report(
    rows,
    sug,
    str(out / "threshold_report.md"),
    str(out / "clean_recipe_suggested.yaml"),
    probe_alpha=0.1,
)
print("Wrote report from", len(rows), "episodes")
PY

echo ""
echo "=== Step 5: Verify outputs ==="
"$VENV/bin/python" - <<'PY'
from pathlib import Path

out = Path("tests_au/ops/filter/outputs/pilot_analyze")
report = out / "threshold_report.md"
recipe = out / "clean_recipe_suggested.yaml"

text = report.read_text(encoding="utf-8")
for section in ("Stage 1", "Stage 2", "Stage 3", "建议"):
    assert section in text, f"missing section '{section}' in {report}"
assert recipe.is_file(), f"missing {recipe}"

print("ACCEPTANCE PASSED:", report)
PY
