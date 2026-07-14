#!/usr/bin/env bash
# Acceptance: Galaxea parquet → 80-dim unified_states/actions + unified_dim_mask on export.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV=/mnt/r/VENV/dj
DATA_ROOT="${DATA_ROOT:-/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002}"
OUT_DIR="$SCRIPT_DIR/outputs"
POINTER="$OUT_DIR/unified_pointer.jsonl"
MAX_EPISODES="${MAX_EPISODES:-4}"

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
echo "=== Step 2: dj-process (loader + unified mapper) ==="
rm -f "$OUT_DIR"/accept_unified_result.jsonl*
"$VENV/bin/dj-process" --config "$SCRIPT_DIR/accept_robot_unified_state_mapper.yaml"

echo ""
echo "=== Step 3: Verify exported 80-dim vectors + mask ==="
"$VENV/bin/python" - <<'PYEOF'
import glob
import json
import os
import sys

import numpy as np

out_files = sorted(glob.glob("tests_au/ops/mapper/outputs/accept_unified_result.jsonl*"))
assert out_files, "No export found"
# prefer non-stats companion if any; take first matching result
out_path = out_files[0]
rows = [json.loads(l) for l in open(out_path) if l.strip()]
assert rows, "empty export"
print(f"Exported episodes: {len(rows)} from {out_path}")

errors = []
for r in rows:
    ep = r.get("id", "?")
    for key in ("unified_states", "unified_actions", "unified_dim_mask"):
        if key not in r:
            errors.append(f"{ep}: missing {key}")
            continue
    if errors and errors[-1].startswith(f"{ep}:"):
        continue
    s = np.asarray(r["unified_states"], dtype=float)
    a = np.asarray(r["unified_actions"], dtype=float)
    m = np.asarray(r["unified_dim_mask"], dtype=float)
    if s.ndim != 2 or s.shape[1] != 80:
        errors.append(f"{ep}: unified_states shape {s.shape}")
    if a.shape != s.shape:
        errors.append(f"{ep}: actions shape {a.shape} != states {s.shape}")
    if m.shape != s.shape:
        errors.append(f"{ep}: mask shape {m.shape} != states {s.shape}")
    if not np.all((m == 0) | (m == 1)):
        errors.append(f"{ep}: mask not binary")
    # R1 Lite shoulder_roll pad slots (left=1, right=30)
    if not np.allclose(m[:, 1], 0) or not np.allclose(m[:, 30], 0):
        errors.append(f"{ep}: shoulder_roll mask should be 0")
    if not np.allclose(s[:, 1], 0) or not np.allclose(s[:, 30], 0):
        errors.append(f"{ep}: shoulder_roll values should be 0")
    # active joints present (skip pad slot 1)
    if np.allclose(s[:, [0, 2, 3, 4, 5, 6]], 0):
        errors.append(f"{ep}: left joints unexpectedly all zero")
    # Galaxea occupancy
    if int(m[0].sum()) != 39:
        errors.append(f"{ep}: expected 39 active dims, got {int(m[0].sum())}")
    # 16-dim native still present from loader
    if "states" in r and len(r["states"][0]) != 16:
        errors.append(f"{ep}: native states not 16-dim")
    meta = r.get("__dj__meta__", {}) or {}
    occ = meta.get("unified_dim_occupancy")
    if occ is None:
        errors.append(f"{ep}: missing meta unified_dim_occupancy")
    else:
        occ = json.loads(occ) if isinstance(occ, str) else occ
        if occ.get("num_active") != 39:
            errors.append(f"{ep}: meta num_active={occ.get('num_active')}")

if errors:
    for e in errors:
        print("  FAIL:", e, file=sys.stderr)
    sys.exit(1)

print(f"shapes ok: states={s.shape}, mask_active={int(m[0].sum())}/80")
print("ACCEPTANCE PASSED")
PYEOF
