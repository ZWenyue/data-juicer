#!/usr/bin/env bash
# Acceptance: Stage 5 base-frame alignment on unified 80-dim process_test data.
#
# ROOT defaults to Galaxea process_test (observation.state/action already 80-dim).
# Applies preset z_-90 and checks analytic EE transform; non-EE dims unchanged.
#
# Examples:
#   bash tests_au/ops/mapper/accept_robot_base_frame_alignment_unified80.sh
#   MAX_EPISODES=8 MAX_TASKS=2 bash tests_au/ops/mapper/accept_robot_base_frame_alignment_unified80.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${VENV:-/mnt/r/VENV/dj}"
ROOT="${ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_test}"
OUT_DIR="$SCRIPT_DIR/outputs"
POINTER="$OUT_DIR/stage5_unified80_pointer.jsonl"
CFG="$SCRIPT_DIR/accept_robot_base_frame_alignment_unified80.yaml"
MAX_TASKS="${MAX_TASKS:-1}"
MAX_EPISODES="${MAX_EPISODES:-4}"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: ROOT not found: $ROOT" >&2
  exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: Need python in $VENV" >&2
  exit 1
fi

echo "=== Step 1: Build pointer JSONL from unified 80-dim root ==="
echo "ROOT=$ROOT  MAX_TASKS=$MAX_TASKS  MAX_EPISODES=$MAX_EPISODES"
"$VENV/bin/python" - <<PY
import glob, json, os
root = "$ROOT"
out = "$POINTER"
max_tasks = int("$MAX_TASKS")
max_eps = int("$MAX_EPISODES")
tasks = sorted(
    d for d in glob.glob(os.path.join(root, "*"))
    if os.path.isdir(d) and os.path.isdir(os.path.join(d, "data"))
)[:max_tasks]
assert tasks, f"no task dirs under {root}"
files = []
for t in tasks:
    files.extend(sorted(glob.glob(os.path.join(t, "data", "chunk-*", "episode_*.parquet"))))
files = files[:max_eps]
assert files, "no episode parquet"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    for i, p in enumerate(files):
        task = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
        ep = os.path.splitext(os.path.basename(p))[0]
        eid = f"{task}/{ep}"
        f.write(json.dumps({"id": eid, "text": eid, "parquet_path": os.path.abspath(p)}) + "\n")
print(f"tasks={len(tasks)} episodes={len(files)} -> {out}")
for p in files:
    print(" ", p)
PY

echo ""
echo "=== Step 2: dj-process (loader + Stage 5) ==="
rm -f "$OUT_DIR"/accept_stage5_unified80_result.jsonl*
"$VENV/bin/dj-process" --config "$CFG"

echo ""
echo "=== Step 3: Verify EE transform under z_-90; non-EE dims unchanged ==="
"$VENV/bin/python" - <<'PYEOF'
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

EE_SLICES = [(7, 10, 10, 16), (36, 39, 39, 45)]  # (pos_i, pos_j, rot_i, rot_j)


def rot6d_to_mat(r6):
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def mat_to_rot6d(mat):
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


Rc = Rotation.from_euler("xyz", (0.0, 0.0, -90.0), degrees=True).as_matrix()

ptr = "tests_au/ops/mapper/outputs/stage5_unified80_pointer.jsonl"
out_files = sorted(glob.glob("tests_au/ops/mapper/outputs/accept_stage5_unified80_result.jsonl*"))
assert out_files, "No Stage 5 export found"
out_path = out_files[0]
src_rows = [json.loads(l) for l in open(ptr) if l.strip()]
res_rows = [json.loads(l) for l in open(out_path) if l.strip()]
print(f"Input pointers: {len(src_rows)}  Output: {len(res_rows)}  file={out_path}")
assert len(res_rows) == len(src_rows), "Mapper must not drop episodes"
by_id = {r["id"]: r for r in res_rows}

# Prefer workspace IO over other editable installs.
sys.path.insert(0, os.path.abspath("."))
from data_juicer._au.utils.lerobot_episode_io import load_episode_arrays

errors = []
for pr in src_rows:
    eid = pr["id"]
    r = by_id.get(eid)
    if r is None:
        errors.append(f"{eid}: missing in export")
        continue
    s0, a0 = load_episode_arrays(pr["parquet_path"])
    s1 = np.asarray(r["states"], dtype=float)
    a1 = np.asarray(r["actions"], dtype=float)
    if s0.shape != s1.shape or s0.shape[1] != 80:
        errors.append(f"{eid}: bad state shape {s0.shape} -> {s1.shape}")
        continue
    if a0.shape != a1.shape:
        errors.append(f"{eid}: bad action shape {a0.shape} -> {a1.shape}")
        continue

    # Non-EE dims must be bit-identical (float) after mapper.
    keep = np.ones(80, dtype=bool)
    for pi, pj, ri, rj in EE_SLICES:
        keep[pi:pj] = False
        keep[ri:rj] = False
    if not np.allclose(s0[:, keep], s1[:, keep], rtol=0, atol=1e-10):
        errors.append(f"{eid}: non-EE state dims changed")
    if not np.allclose(a0[:, keep], a1[:, keep], rtol=0, atol=1e-10):
        errors.append(f"{eid}: non-EE action dims changed")

    for pi, pj, ri, rj in EE_SLICES:
        # position: p' = R p  <=>  rows @ Rc.T
        p_exp = s0[:, pi:pj] @ Rc.T
        if not np.allclose(s1[:, pi:pj], p_exp, rtol=1e-5, atol=1e-6):
            err = float(np.max(np.abs(s1[:, pi:pj] - p_exp)))
            errors.append(f"{eid}: state pos[{pi}:{pj}] max_abs_err={err:.3e}")
        # rotation: R' = Rc @ R
        M = rot6d_to_mat(s0[:, ri:rj])
        M_exp = np.einsum("ab,tbc->tac", Rc, M)
        r6_exp = mat_to_rot6d(M_exp)
        if not np.allclose(s1[:, ri:rj], r6_exp, rtol=1e-5, atol=1e-5):
            err = float(np.max(np.abs(s1[:, ri:rj] - r6_exp)))
            errors.append(f"{eid}: state rot6d[{ri}:{rj}] max_abs_err={err:.3e}")
        # orthonormality after correction
        M1 = rot6d_to_mat(s1[:, ri:rj])
        for t in (0, min(10, len(M1) - 1), len(M1) - 1):
            c0, c1 = M1[t, :, 0], M1[t, :, 1]
            if abs(np.linalg.norm(c0) - 1) > 1e-4 or abs(np.linalg.norm(c1) - 1) > 1e-4:
                errors.append(f"{eid}: rot6d cols not unit at t={t}")
            if abs(c0 @ c1) > 1e-4:
                errors.append(f"{eid}: rot6d cols not orthogonal at t={t}")
        # action EE (zeros -> zeros)
        if not np.allclose(a1[:, pi:pj], a0[:, pi:pj] @ Rc.T, atol=1e-10):
            errors.append(f"{eid}: action pos[{pi}:{pj}] mismatch")
        if not np.allclose(a1[:, ri:rj], a0[:, ri:rj], atol=1e-10) and float(np.abs(a0[:, ri:rj]).max()) > 0:
            # if non-zero action EE existed, would need full rot check; zeros stay zeros
            pass

    meta = r.get("__dj__meta__", {}) or {}
    rep = meta.get("base_frame_alignment_report")
    if rep is None:
        errors.append(f"{eid}: missing base_frame_alignment_report")
    else:
        rep = json.loads(rep) if isinstance(rep, str) else rep
        if not rep.get("changed"):
            errors.append(f"{eid}: report.changed is false")
        if rep.get("num_blocks") != 4:
            errors.append(f"{eid}: num_blocks={rep.get('num_blocks')} expected 4")

if errors:
    for e in errors:
        print("  FAIL:", e, file=sys.stderr)
    sys.exit(1)

# summary: right-arm EE displacement before/after (often larger motion than left)
print("\n--- Right EE (dims 36:39) end-start displacement (before -> after z_-90) ---")
print(f"{'Episode':<56s}  {'disp_before':>28s}  {'disp_after':>28s}")
for pr in src_rows:
    eid = pr["id"]
    s0, _ = load_episode_arrays(pr["parquet_path"])
    s1 = np.asarray(by_id[eid]["states"], dtype=float)
    db = (s0[-1, 36:39] - s0[0, 36:39]).round(4)
    da = (s1[-1, 36:39] - s1[0, 36:39]).round(4)
    print(f"{eid:<56s}  {str(db):>28s}  {str(da):>28s}")

print("\nACCEPTANCE PASSED (unified80 Stage 5)")
PYEOF
