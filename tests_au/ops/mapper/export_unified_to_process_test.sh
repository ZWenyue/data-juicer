#!/usr/bin/env bash
# Export Galaxea → unified 80-dim per-episode parquet (LeRobot layout) under process_test.
#
# Output layout (same as source):
#   OUT/<task>/data/chunk-XXX/episode_YYYYYY.parquet
#   OUT/<task>/meta/info.json
#
# Each parquet frame has:
#   observation.state[80], action[80], observation.state_dim_mask[80]
#   (+ timestamp / frame_index / episode_index / ... when present)
#
# Examples:
#   MAX_TASKS=1 bash tests_au/ops/mapper/export_unified_to_process_test.sh
#   bash tests_au/ops/mapper/export_unified_to_process_test.sh
#   ROOT=/path/to/one_task bash tests_au/ops/mapper/export_unified_to_process_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${VENV:-/mnt/r/VENV/dj}"
ROOT="${ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot_press}"
OUT="${OUT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_test}"
EMBODIMENT="${EMBODIMENT:-galaxea_r1_lite}"
NP="${NP:-8}"
MAX_TASKS="${MAX_TASKS:-}"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: ROOT not found: $ROOT" >&2
  exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: Need python in $VENV" >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$OUT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Export unified 80-dim parquet (1 episode = 1 file) ==="
echo "ROOT=$ROOT"
echo "OUT=$OUT"
echo "EMBODIMENT=$EMBODIMENT  workers=$NP  MAX_TASKS=${MAX_TASKS:-all}"

ARGS=(
  --root "$ROOT"
  --out "$OUT"
  --embodiment "$EMBODIMENT"
  --num_workers "$NP"
)
if [[ -n "$MAX_TASKS" ]]; then
  ARGS+=(--max_tasks "$MAX_TASKS")
fi

"$VENV/bin/python" "$SCRIPT_DIR/export_unified_parquets.py" "${ARGS[@]}"

echo ""
echo "=== Verify LeRobot layout (data + meta + videos symlink) ==="
"$VENV/bin/python" - <<PY
import glob, json, os
import numpy as np
import pyarrow.parquet as pq

out = "$OUT"
tasks = [
    d for d in sorted(os.listdir(out))
    if os.path.isdir(os.path.join(out, d)) and os.path.isdir(os.path.join(out, d, "data"))
]
assert tasks, f"no task dirs under {out}"
task = os.path.join(out, tasks[0])
assert os.path.islink(os.path.join(task, "videos")) or os.path.isdir(os.path.join(task, "videos")), "videos/ missing"
assert os.path.exists(os.path.join(task, "videos")), "videos link broken"
meta = os.path.join(task, "meta")
for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"):
    assert os.path.isfile(os.path.join(meta, name)), f"missing meta/{name}"
info = json.load(open(os.path.join(meta, "info.json")))
assert "video_path" in info
assert info.get("total_videos", 0) > 0
assert any(k.startswith("observation.images.") for k in info["features"])

files = sorted(glob.glob(os.path.join(task, "data", "chunk-*", "episode_*.parquet")))
assert files, f"no episode parquet under {task}"
t = pq.read_table(files[0])
for col in ("observation.state", "action", "observation.state_dim_mask"):
    assert col in t.column_names, f"missing {col}"
s0 = np.asarray(t.column("observation.state")[0].as_py(), dtype=float)
m0 = np.asarray(t.column("observation.state_dim_mask")[0].as_py(), dtype=float)
assert s0.shape == (80,)
assert np.allclose(s0[m0 == 0], 0)
n_ep = sum(1 for _ in open(os.path.join(meta, "episodes.jsonl")) if _.strip())
print(f"task={tasks[0]}")
print(f"layout=data/+meta/+videos(symlink)  episodes.jsonl={n_ep}  parquet_files={len(files)}")
print(f"videos -> {os.readlink(os.path.join(task, 'videos')) if os.path.islink(os.path.join(task,'videos')) else '(dir)'}")
print(f"sample={files[0]}")
print(f"rows={t.num_rows} state_dim=80 mask_active={int(m0.sum())}/80")
print("EXPORT PARQUET OK")
PY
