#!/usr/bin/env bash
# Full corpus: pointer JSONL → global percentiles → dj-analyze → threshold report
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
ROOT="${ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot}"
OUT="$SCRIPT_DIR/outputs/full_analyze"

if [ ! -d "$ROOT" ]; then
    echo "ERROR: Dataset root not found at $ROOT" >&2
    exit 1
fi
if [ ! -x "$VENV/bin/python" ] || [ ! -x "$VENV/bin/dj-analyze" ]; then
    echo "ERROR: Need python and dj-analyze in $VENV" >&2
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$OUT"

export HF_HOME="${HF_HOME:-$OUT/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$OUT/.hf_datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OUT/.cache}"
export TMPDIR="${TMPDIR:-/tmp/dj_full_accept_${USER:-user}}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

echo "=== Step 1: Pointer JSONL (all tasks under $ROOT) ==="
"$VENV/bin/python" "$SCRIPT_DIR/convert_lerobot_episodes.py" \
    --mode pointer \
    --root "$ROOT" \
    --output "$OUT/lerobot_episodes_ptr.jsonl"

echo ""
echo "=== Step 2: Global percentiles (all tasks) ==="
"$VENV/bin/python" "$SCRIPT_DIR/compute_embodiment_percentiles.py" \
    --root "$ROOT" \
    --output "$OUT/percentiles.json" \
    --embodiment galaxea_r1_lite

echo ""
echo "=== Step 3: dj-analyze (loader + Stage 1/2/3) ==="
rm -f "$OUT/analyze_result.jsonl" "$OUT/analyze_result_stats.jsonl"
if ! "$VENV/bin/dj-analyze" --config "$SCRIPT_DIR/analyze_qwenrobomanip_full.yaml"; then
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

out = Path("tests_au/ops/filter/outputs/full_analyze")
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
    rows = load_stats_jsonl(str(stats_source))
    print(f"Loaded stats from {stats_source.name} ({len(rows)} episodes)")

if s1.exists() and s2.exists() and s3.exists():
    missing_any = any(
        any(key not in _stat_dict(row) for key in REQUIRED_KEYS) for row in rows
    )
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
print("Suggested:", {k: sug[k] for k in ("max_flagged_ratio", "da_threshold", "alpha")})
PY

echo ""
echo "=== Step 5: Verify ==="
"$VENV/bin/python" - <<'PY'
from pathlib import Path

out = Path("tests_au/ops/filter/outputs/full_analyze")
report = out / "threshold_report.md"
recipe = out / "clean_recipe_suggested.yaml"
ptr = out / "lerobot_episodes_ptr.jsonl"
assert ptr.is_file() and ptr.stat().st_size < 200_000_000, "pointer JSONL missing or unexpectedly huge"
text = report.read_text(encoding="utf-8")
for section in ("Stage 1", "Stage 2", "Stage 3", "建议"):
    assert section in text, f"missing section '{section}'"
assert recipe.is_file()
n = sum(1 for _ in open(ptr))
print(f"ACCEPTANCE PASSED: {n} pointer episodes, report={report}")
PY
