#!/usr/bin/env bash
# Full clean acceptance: Stage1/2/3 + Stage5 + Check3(episode) + unified 80-dim.
#
# Usage:
#   bash tests_au/ops/filter/accept_qwenrobomanip_full_clean.sh
#   MAX_EPISODES=8 bash tests_au/ops/filter/accept_qwenrobomanip_full_clean.sh
#   EXPORT_UNIFIED_PARQUET=1 bash tests_au/ops/filter/accept_qwenrobomanip_full_clean.sh
#
# Env:
#   DATASET                  LeRobot task dir (default: Galaxea Connect_Router test set)
#   MAX_EPISODES             Cap episodes in pointer JSONL (default: all)
#   VIDEO_KEY                Camera folder under videos/ (default: observation.images.head_rgb)
#   EXPORT_UNIFIED_PARQUET   If 1, also export LeRobot 80-dim parquet layout for KEPT episodes
#   UNIFIED_OUT              Output root for optional parquet export
#   VENV                     Python/dj-process env (default: /mnt/r/VENV/dj)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${VENV:-/mnt/r/VENV/dj}"
DATASET="${DATASET:-/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002}"
OUT_DIR="$SCRIPT_DIR/outputs"
POINTER="$OUT_DIR/full_clean_pointer.jsonl"
RESULT="$OUT_DIR/full_clean_result.jsonl"
PERCENTILES="$OUT_DIR/percentiles.json"
VIDEO_KEY="${VIDEO_KEY:-observation.images.head_rgb}"
MAX_EPISODES="${MAX_EPISODES:-}"
EXPORT_UNIFIED_PARQUET="${EXPORT_UNIFIED_PARQUET:-0}"
UNIFIED_OUT="${UNIFIED_OUT:-$OUT_DIR/full_clean_unified80}"
EMBODIMENT="${EMBODIMENT:-galaxea_r1_lite}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: python not found in $VENV" >&2
  exit 1
fi
if [[ ! -d "$DATASET/data" ]]; then
  echo "ERROR: Dataset not found at $DATASET" >&2
  exit 1
fi

PYTHON="$VENV/bin/python"
DJ_PROCESS="$VENV/bin/dj-process"
cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Step 1: Build pointer JSONL (parquet_path + videos) ==="
"$PYTHON" - <<PY
import glob
import json
import os
from pathlib import Path

root = Path("$DATASET").resolve()
out = Path("$POINTER")
video_key = "$VIDEO_KEY"
max_ep = "$MAX_EPISODES".strip()

files = sorted(glob.glob(str(root / "data" / "chunk-*" / "episode_*.parquet")))
if max_ep:
    files = files[: int(max_ep)]
assert files, f"no parquet under {root}"

out.parent.mkdir(parents=True, exist_ok=True)
n_vid = 0
with open(out, "w", encoding="utf-8") as f:
    for pf in files:
        p = Path(pf)
        # episode_000012.parquet -> episode_000012
        stem = p.stem
        # data/chunk-000/... -> chunk-000
        chunk = p.parent.name
        vid = root / "videos" / chunk / video_key / f"{stem}.mp4"
        rec = {
            "id": stem,
            "text": stem,
            "parquet_path": str(p.resolve()),
        }
        if vid.is_file():
            rec["videos"] = [str(vid.resolve())]
            n_vid += 1
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"wrote {len(files)} pointers -> {out}")
print(f"with videos[{video_key}]: {n_vid}/{len(files)}")
if n_vid == 0:
    print("WARNING: no videos attached; Check3 Op1 will skip (fail-open keep)")
PY

echo ""
echo "=== Step 2: Precompute embodiment percentiles (Stage 3) ==="
"$PYTHON" "$SCRIPT_DIR/compute_embodiment_percentiles.py" \
  --dataset_dir "$DATASET" \
  --output "$PERCENTILES" \
  --embodiment "$EMBODIMENT"

echo ""
echo "=== Step 3: dj-process (S1/S2/S3/S5 + Check3 + unified80) ==="
rm -f "$OUT_DIR"/full_clean_result.jsonl*
"$DJ_PROCESS" --config "$SCRIPT_DIR/accept_qwenrobomanip_full_clean.yaml"

echo ""
echo "=== Step 4: Verify cleaned JSONL + 80-dim fields ==="
"$PYTHON" - <<'PYEOF'
import glob
import json
import sys

import numpy as np

paths = sorted(glob.glob("tests_au/ops/filter/outputs/full_clean_result.jsonl*"))
assert paths, "missing export full_clean_result.jsonl"
# Prefer the main export (not a stats sidecar if naming collides)
result_path = paths[0]
rows = [json.loads(l) for l in open(result_path) if l.strip()]
assert rows, "empty export"
print(f"Kept episodes: {len(rows)} from {result_path}")

required_stats = [
    "sudden_change_keep",
    "state_action_alignment_keep",
    "extreme_value_keep",
    "video_quality_episode_keep",
]
required_meta = [
    "sudden_change_report",
    "state_action_alignment_report",
    "extreme_value_report",
    "key_frame_report",
    "video_quality_episode_report",
]
errors = []
for r in rows:
    ep = r.get("id", "?")
    stats = r.get("__dj__stats__", {}) or {}
    meta = r.get("__dj__meta__", {}) or {}
    for k in required_stats:
        if k not in stats:
            errors.append(f"{ep}: missing stats.{k}")
    for k in required_meta:
        if k not in meta:
            errors.append(f"{ep}: missing meta.{k}")
    for key in ("unified_states", "unified_actions", "unified_dim_mask"):
        if key not in r:
            errors.append(f"{ep}: missing {key}")
            continue
    if any(e.startswith(f"{ep}: missing unified") for e in errors[-3:]):
        continue
    s = np.asarray(r["unified_states"], dtype=float)
    a = np.asarray(r["unified_actions"], dtype=float)
    m = np.asarray(r["unified_dim_mask"], dtype=float)
    if s.ndim != 2 or s.shape[1] != 80:
        errors.append(f"{ep}: unified_states shape {s.shape}")
    elif a.shape != s.shape or m.shape != s.shape:
        errors.append(f"{ep}: unified shape mismatch")
    elif int(m[0].sum()) != 24:
        errors.append(f"{ep}: expected 24 active action dims, got {int(m[0].sum())}")
    # Kept rows must pass episode gate
    if not stats.get("video_quality_episode_keep", False):
        errors.append(f"{ep}: kept but video_quality_episode_keep=False")

if errors:
    print(f"\n*** {len(errors)} ERRORS ***", file=sys.stderr)
    for e in errors[:40]:
        print("  FAIL:", e, file=sys.stderr)
    if len(errors) > 40:
        print(f"  ... and {len(errors) - 40} more", file=sys.stderr)
    sys.exit(1)

print("Stage1/2/3 + Check3 stats/meta present; unified 80-dim OK (action_mask_active=24).")
print("ACCEPTANCE PASSED (JSONL clean + 80-dim)")
PYEOF

if [[ "$EXPORT_UNIFIED_PARQUET" == "1" ]]; then
  echo ""
  echo "=== Step 5 (optional): Export LeRobot 80-dim parquet for KEPT episodes ==="
  mkdir -p "$UNIFIED_OUT"
  "$PYTHON" - <<PY
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

repo = Path("$REPO_ROOT")
sys.path.insert(0, str(repo))

from data_juicer._au.utils.embodiment_layout import (
    UNIFIED_DIM,
    load_embodiment_config,
    pack_episode_to_80,
)

result_path = Path("$RESULT")
# dj-process may shard; take primary file
if not result_path.is_file():
    cands = sorted(result_path.parent.glob(result_path.name + "*"))
    assert cands, f"no result at {result_path}"
    result_path = cands[0]

rows = [json.loads(l) for l in open(result_path) if l.strip()]
cfg = load_embodiment_config("$EMBODIMENT")
out_root = Path("$UNIFIED_OUT")
src_root = Path("$DATASET").resolve()
task_name = src_root.name

# Mirror LeRobot layout under OUT/<task>/data/chunk-*/...
n = 0
for r in rows:
    src = Path(r["parquet_path"]).resolve()
    chunk = src.parent.name
    stem = src.name
    dst = out_root / task_name / "data" / chunk / stem
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(str(src)).to_pandas()
    states, actions, dim_mask = pack_episode_to_80(df, cfg)
    assert states.shape[1] == UNIFIED_DIM

    keep_cols = {}
    for col in (
        "timestamp", "frame_index", "episode_index", "index",
        "coarse_task_index", "task_index", "coarse_quality_index", "quality_index",
    ):
        if col in df.columns:
            keep_cols[col] = df[col].tolist()

    table = pa.table(
        {
            **keep_cols,
            "observation.state": [row.tolist() for row in states],
            "action": [row.tolist() for row in actions],
            "action_dim_mask": [row.tolist() for row in dim_mask],
        }
    )
    pq.write_table(table, dst)
    n += 1

# Minimal meta + videos symlink
meta_dir = out_root / task_name / "meta"
meta_dir.mkdir(parents=True, exist_ok=True)
src_info = src_root / "meta" / "info.json"
info = json.loads(src_info.read_text()) if src_info.is_file() else {}
info.update(
    {
        "total_episodes": n,
        "unified_embodiment": "$EMBODIMENT",
        "unified_dim": UNIFIED_DIM,
        "source_clean_jsonl": str(result_path),
    }
)
(meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

videos_link = out_root / task_name / "videos"
src_videos = src_root / "videos"
if src_videos.is_dir() and not videos_link.exists():
    videos_link.symlink_to(src_videos, target_is_directory=True)

summary = {
    "kept_episodes": n,
    "out": str(out_root / task_name),
    "result_jsonl": str(result_path),
}
(out_root / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Exported {n} kept episodes -> {out_root / task_name}")
print(f"summary -> {out_root / 'export_summary.json'}")
PY
  echo "Optional LeRobot 80-dim parquet export DONE."
else
  echo ""
  echo "Skip optional LeRobot parquet export (set EXPORT_UNIFIED_PARQUET=1 to enable)."
  echo "JSONL already contains unified_states / unified_actions / unified_dim_mask."
fi

echo ""
echo "ALL DONE"
