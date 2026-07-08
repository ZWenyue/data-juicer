#!/usr/bin/env bash
# Acceptance test: Stage 5 (Base Frame & EEF Orientation Alignment) on synthetic EEF poses.
# Generates misaligned poses (forward=+y), runs Stage 5 (preset z_-90), verifies forward=+x.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV=/mnt/r/VENV/dj

cd "$REPO_ROOT"

echo "=== Step 1: Generate synthetic EEF pose dataset (forward=+y) ==="
"$VENV/bin/python" "$SCRIPT_DIR/gen_synthetic_pose_dataset.py" \
    --output "$SCRIPT_DIR/outputs/synth_pose.jsonl" \
    --num_episodes 8 --num_frames 60 --forward_axis y --seed 0

echo ""
echo "=== Step 2: Run dj-process (Stage 5 base frame alignment) ==="
"$VENV/bin/dj-process" --config "$SCRIPT_DIR/accept_robot_base_frame_alignment_mapper.yaml"

echo ""
echo "=== Step 3: Verify poses re-aligned to +x forward ==="
"$VENV/bin/python" - <<'PYEOF'
import glob
import json
import os

import numpy as np

in_path = "tests_au/ops/mapper/outputs/synth_pose.jsonl"
out_glob = "tests_au/ops/mapper/outputs/accept_stage5_result.jsonl*"
out_files = sorted(glob.glob(out_glob))
assert out_files, f"No output found matching {out_glob}"
out_path = out_files[0]

src = [json.loads(l) for l in open(in_path) if l.strip()]
res = [json.loads(l) for l in open(out_path) if l.strip()]
print(f"Input: {len(src)} episodes -> Output: {len(res)} episodes")
assert len(res) == len(src), "Mapper must not drop episodes"

by_id_src = {r["id"]: r for r in src}
errors = []
for r in res:
    ep = r["id"]
    s_before = np.array(by_id_src[ep]["states"])
    s_after = np.array(r["states"])
    disp_before = s_before[-1, :3] - s_before[0, :3]
    disp_after = s_after[-1, :3] - s_after[0, :3]

    # before: dominant along +y ; after: dominant along +x
    if not (abs(disp_before[1]) > abs(disp_before[0])):
        errors.append(f"{ep}: input not +y-dominant: {disp_before.round(4)}")
    if not (disp_after[0] > abs(disp_after[1]) and disp_after[0] > 0):
        errors.append(f"{ep}: output not +x-forward: {disp_after.round(4)}")

    # rot6d columns must stay orthonormal after correction
    r6 = s_after[0, 3:9]
    c0, c1 = r6[:3], r6[3:6]
    if abs(np.linalg.norm(c0) - 1) > 1e-5 or abs(np.linalg.norm(c1) - 1) > 1e-5 or abs(c0 @ c1) > 1e-5:
        errors.append(f"{ep}: rot6d not orthonormal after correction")

    # meta report present + changed
    meta = r.get("__dj__meta__", {})
    rep = meta.get("base_frame_alignment_report")
    if rep is None:
        errors.append(f"{ep}: missing base_frame_alignment_report")
    else:
        rep = json.loads(rep)
        if not rep.get("changed") or rep.get("num_blocks") != 2:
            errors.append(f"{ep}: report changed/num_blocks wrong: {rep.get('changed')}/{rep.get('num_blocks')}")

if errors:
    import sys
    for e in errors:
        print("  FAIL:", e, file=sys.stderr)
    sys.exit(1)

# summary table
print("\n--- Per-episode displacement (before -> after) ---")
print(f"{'Episode':<18s}  {'disp_before(xyz)':>26s}  {'disp_after(xyz)':>26s}")
for r in res:
    ep = r["id"]
    db = (np.array(by_id_src[ep]["states"])[-1, :3] - np.array(by_id_src[ep]["states"])[0, :3]).round(3)
    da = (np.array(r["states"])[-1, :3] - np.array(r["states"])[0, :3]).round(3)
    print(f"{ep:<18s}  {str(db):>26s}  {str(da):>26s}")

print("\nACCEPTANCE PASSED")
PYEOF
