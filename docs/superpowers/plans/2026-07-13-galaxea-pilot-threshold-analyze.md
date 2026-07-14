# Galaxea Pilot Threshold Analyze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For one Galaxea task (`Connect_Router_Cables_20250625_002`), convert decomposed-column LeRobot parquet to 16-dim JSONL, run Stage 1/2/3 via `dj-analyze`, and emit a threshold recommendation report plus a suggested clean recipe.

**Architecture:** Extend existing convert/percentile helper scripts to auto-detect unified vs decomposed columns and pack arms+grippers into a 16-dim layout. Reuse `_au` robot filters via a new analyze YAML. Add a pure-Python summarizer that reads exported stats JSONL and writes `threshold_report.md` + `clean_recipe_suggested.yaml`. Wire everything in one acceptance shell script.

**Tech Stack:** Python 3, numpy, pyarrow, Data-Juicer (`dj-analyze`), unittest/pytest, bash.

**Spec:** `docs/superpowers/specs/2026-07-13-galaxea-pilot-threshold-analyze-design.md`

**Commit policy:** Do **not** create git commits unless the user explicitly asks. Skip all “Commit” steps below by default.

---

## File map

| File | Responsibility |
|------|----------------|
| `tests_au/ops/filter/convert_lerobot_episodes.py` | Parquet → per-episode JSONL; unified or decomposed→16d |
| `tests_au/ops/filter/compute_embodiment_percentiles.py` | Same layout loading; write `percentiles.json` |
| `tests_au/ops/filter/analyze_qwenrobomanip_pilot.yaml` | `dj-analyze` recipe for Stage 1/2/3 |
| `tests_au/ops/filter/summarize_threshold_report.py` | Stats → markdown report + suggested YAML |
| `tests_au/ops/filter/accept_analyze_qwenrobomanip_pilot.sh` | One-shot pilot pipeline |
| `tests_au/ops/filter/test_convert_lerobot_episodes.py` | Unit tests for convert layout |
| `tests_au/ops/filter/test_summarize_threshold_report.py` | Unit tests for summarizer |

Shared layout constants live in convert script and are imported by percentiles (avoid drift).

---

### Task 1: Shared 16-dim packing helpers in convert script

**Files:**
- Modify: `tests_au/ops/filter/convert_lerobot_episodes.py`
- Test: `tests_au/ops/filter/test_convert_lerobot_episodes.py`

- [ ] **Step 1: Write the failing test**

Create `tests_au/ops/filter/test_convert_lerobot_episodes.py`:

```python
# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

import numpy as np

from tests_au.ops.filter.convert_lerobot_episodes import (
    ARM_SLOT_LEFT,
    ARM_SLOT_RIGHT,
    pack_decomposed_to_16,
)

REAL_DATASET_DIR = (
    "/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot"
    "/Connect_Router_Cables_20250625_002"
)
REAL_DATA_AVAILABLE = os.path.isdir(
    os.path.join(REAL_DATASET_DIR, "data", "chunk-000")
)


class TestPackDecomposed(unittest.TestCase):
    def test_pack_shapes_and_slots(self):
        T = 5
        left_arm = np.arange(T * 6, dtype=float).reshape(T, 6)
        right_arm = np.arange(100, 100 + T * 6, dtype=float).reshape(T, 6)
        left_g = np.full((T,), 0.3)
        right_g = np.full((T,), 0.7)
        out = pack_decomposed_to_16(left_arm, left_g, right_arm, right_g)
        self.assertEqual(out.shape, (T, 16))
        np.testing.assert_array_equal(out[:, ARM_SLOT_LEFT], left_arm)
        np.testing.assert_array_equal(out[:, ARM_SLOT_RIGHT], right_arm)
        self.assertTrue(np.allclose(out[:, 7], 0.3))
        self.assertTrue(np.allclose(out[:, 15], 0.7))
        self.assertTrue(np.allclose(out[:, 1], 0.0))
        self.assertTrue(np.allclose(out[:, 9], 0.0))

    def test_pack_accepts_column_vector_gripper(self):
        T = 3
        left_arm = np.zeros((T, 6))
        right_arm = np.zeros((T, 6))
        out = pack_decomposed_to_16(
            left_arm, np.ones((T, 1)) * 0.5, right_arm, np.zeros((T, 1))
        )
        self.assertEqual(out.shape, (T, 16))
        self.assertTrue(np.allclose(out[:, 7], 0.5))


@unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
class TestConvertReal(unittest.TestCase):
    def test_convert_one_episode_shape(self):
        from tests_au.ops.filter.convert_lerobot_episodes import convert

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "eps.jsonl")
            # convert all episodes is heavy; use internal helper on one file
            import glob
            import json

            import pyarrow.parquet as pq

            from tests_au.ops.filter.convert_lerobot_episodes import (
                episode_arrays_from_df,
            )

            pf = sorted(
                glob.glob(
                    os.path.join(
                        REAL_DATASET_DIR, "data", "chunk-000", "episode_*.parquet"
                    )
                )
            )[0]
            df = pq.read_table(pf).to_pandas()
            states, actions = episode_arrays_from_df(df)
            self.assertEqual(states.ndim, 2)
            self.assertEqual(states.shape[1], 16)
            self.assertEqual(actions.shape[1], 16)
            self.assertEqual(states.shape[0], len(df))
            convert(REAL_DATASET_DIR, out)
            with open(out) as f:
                n = sum(1 for _ in f)
            self.assertGreater(n, 0)
            with open(out) as f:
                row = json.loads(f.readline())
            self.assertEqual(len(row["states"][0]), 16)
            self.assertEqual(len(row["actions"][0]), 16)


if __name__ == "__main__":
    unittest.main()
```

Note: if `tests_au` is not a package, prefer importing via path insert in the test file:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_lerobot_episodes import (
    ARM_SLOT_LEFT,
    ARM_SLOT_RIGHT,
    pack_decomposed_to_16,
    episode_arrays_from_df,
    convert,
)
```

Use this path-insert style in all tests under `tests_au/ops/filter/` for consistency with local scripts.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/r/share/zwy/Projects/data-juicer
.venv/bin/python -m pytest tests_au/ops/filter/test_convert_lerobot_episodes.py::TestPackDecomposed -v
```

Expected: FAIL (import error / missing `pack_decomposed_to_16`).

- [ ] **Step 3: Implement packing + dual-format convert**

Replace `tests_au/ops/filter/convert_lerobot_episodes.py` with:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LeRobot v2.1 dataset (per-frame parquet) to per-episode JSONL for DJ.

Supports:
  - unified columns: observation.state / action
  - Galaxea decomposed columns packed into 16-dim arms+grippers layout
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

ARM_SLOT_LEFT = [0, 2, 3, 4, 5, 6]
ARM_SLOT_RIGHT = [8, 10, 11, 12, 13, 14]

DECOMP_STATE = {
    "left_arm": "observation.state.left_arm",
    "left_gripper": "observation.state.left_gripper",
    "right_arm": "observation.state.right_arm",
    "right_gripper": "observation.state.right_gripper",
}
DECOMP_ACTION = {
    "left_arm": "action.left_arm",
    "left_gripper": "action.left_gripper",
    "right_arm": "action.right_arm",
    "right_gripper": "action.right_gripper",
}


def _as_TxD(col, expected_last_dim=None):
    """Stack a pandas column of scalars/arrays into (T, D)."""
    arr = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in col])
    if expected_last_dim is not None and arr.shape[-1] != expected_last_dim:
        raise ValueError(
            f"Expected last dim {expected_last_dim}, got {arr.shape} from column"
        )
    return arr


def pack_decomposed_to_16(left_arm, left_gripper, right_arm, right_gripper):
    """Pack Galaxea R1 Lite arm+gripper channels into (T, 16)."""
    la = np.asarray(left_arm, dtype=float)
    ra = np.asarray(right_arm, dtype=float)
    if la.ndim != 2 or la.shape[1] != 6:
        raise ValueError(f"left_arm must be (T,6), got {la.shape}")
    if ra.ndim != 2 or ra.shape[1] != 6:
        raise ValueError(f"right_arm must be (T,6), got {ra.shape}")
    lg = np.asarray(left_gripper, dtype=float).reshape(-1)
    rg = np.asarray(right_gripper, dtype=float).reshape(-1)
    T = la.shape[0]
    if not (len(lg) == T and len(rg) == T and ra.shape[0] == T):
        raise ValueError("Mismatched T across arm/gripper arrays")
    out = np.zeros((T, 16), dtype=float)
    out[:, ARM_SLOT_LEFT] = la
    out[:, 7] = lg
    out[:, ARM_SLOT_RIGHT] = ra
    out[:, 15] = rg
    return out


def _has_unified(df):
    return "observation.state" in df.columns and "action" in df.columns


def _has_decomposed(df):
    needed = list(DECOMP_STATE.values()) + list(DECOMP_ACTION.values())
    return all(c in df.columns for c in needed)


def episode_arrays_from_df(df):
    """Return (states, actions) as float arrays (T, D)."""
    if _has_unified(df):
        states = np.stack([np.asarray(v, dtype=float) for v in df["observation.state"]])
        actions = np.stack([np.asarray(v, dtype=float) for v in df["action"]])
        return states, actions
    if _has_decomposed(df):
        states = pack_decomposed_to_16(
            _as_TxD(df[DECOMP_STATE["left_arm"]], 6),
            _as_TxD(df[DECOMP_STATE["left_gripper"]]),
            _as_TxD(df[DECOMP_STATE["right_arm"]], 6),
            _as_TxD(df[DECOMP_STATE["right_gripper"]]),
        )
        actions = pack_decomposed_to_16(
            _as_TxD(df[DECOMP_ACTION["left_arm"]], 6),
            _as_TxD(df[DECOMP_ACTION["left_gripper"]]),
            _as_TxD(df[DECOMP_ACTION["right_arm"]], 6),
            _as_TxD(df[DECOMP_ACTION["right_gripper"]]),
        )
        return states, actions
    missing = []
    for c in ["observation.state", "action"] + list(DECOMP_STATE.values()) + list(
        DECOMP_ACTION.values()
    ):
        if c not in df.columns:
            missing.append(c)
    raise ValueError(
        "Neither unified nor decomposed state/action columns found. "
        f"Missing examples: {missing[:8]}"
    )


def _to_nested_list(arr):
    return np.asarray(arr, dtype=float).tolist()


def convert(dataset_dir: str, output: str):
    pattern = os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found: {pattern}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for pf in parquet_files:
            df = pq.read_table(pf).to_pandas()
            ep_idx = int(df["episode_index"].iloc[0])
            ep_id = f"episode_{ep_idx:06d}"
            states, actions = episode_arrays_from_df(df)
            record = {
                "id": ep_id,
                "episode_index": ep_idx,
                "num_frames": int(states.shape[0]),
                "states": _to_nested_list(states),
                "actions": _to_nested_list(actions),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(parquet_files)} episodes -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.dataset_dir, args.output)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests_au/ops/filter/test_convert_lerobot_episodes.py -v
```

Expected: `TestPackDecomposed` PASS; `TestConvertReal` PASS if dataset mounted (may be slow on full convert — if too slow, change real test to only call `episode_arrays_from_df` on one parquet and skip full `convert`).

If full convert is too slow in CI/local, trim `test_convert_one_episode_shape` to **not** call `convert()`; only `episode_arrays_from_df` + assert shapes. Keep a separate optional smoke that convert writes ≥1 line.

- [ ] **Step 5: Commit (skip unless user asks)**

---

### Task 2: Percentiles script supports decomposed layout

**Files:**
- Modify: `tests_au/ops/filter/compute_embodiment_percentiles.py`
- Test: extend `tests_au/ops/filter/test_convert_lerobot_episodes.py` or add assert in Task 2 via a small function test inside the same file

- [ ] **Step 1: Write failing test for load path**

Append to `test_convert_lerobot_episodes.py` (or create `test_compute_embodiment_percentiles.py`):

```python
@unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
class TestPercentilesDecomposed(unittest.TestCase):
    def test_load_all_frames_decomposed(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from compute_embodiment_percentiles import load_all_frames

        states, actions = load_all_frames(REAL_DATASET_DIR)
        self.assertEqual(states.shape[1], 16)
        self.assertEqual(actions.shape[1], 16)
        self.assertGreater(states.shape[0], 100)
```

- [ ] **Step 2: Run test — expect FAIL** on old unified-only loader

```bash
.venv/bin/python -m pytest tests_au/ops/filter/test_convert_lerobot_episodes.py::TestPercentilesDecomposed -v
```

- [ ] **Step 3: Update `compute_embodiment_percentiles.py`**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute per-embodiment q01/q99 percentiles from LeRobot v2.1 parquet data."""

import argparse
import glob
import json
import os
import sys

import numpy as np

# Reuse convert helpers (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_lerobot_episodes import episode_arrays_from_df  # noqa: E402


def load_all_frames(dataset_dir):
    """Load all frames from all episodes, return (states, actions) as np arrays."""
    import pyarrow.parquet as pq

    pattern = os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found: {pattern}")

    all_states = []
    all_actions = []
    for pf in parquet_files:
        df = pq.read_table(pf).to_pandas()
        states, actions = episode_arrays_from_df(df)
        all_states.append(states)
        all_actions.append(actions)

    return np.vstack(all_states), np.vstack(all_actions)


def compute_percentiles(states, actions):
    """Compute q01 and q99 per dimension."""
    return {
        "state": {
            "q01": np.percentile(states, 1, axis=0).tolist(),
            "q99": np.percentile(states, 99, axis=0).tolist(),
        },
        "action": {
            "q01": np.percentile(actions, 1, axis=0).tolist(),
            "q99": np.percentile(actions, 99, axis=0).tolist(),
        },
        "num_frames": int(states.shape[0]),
        "num_episodes": None,
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute per-embodiment percentiles.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embodiment", default="default")
    args = parser.parse_args()

    print(f"Loading frames from {args.dataset_dir} ...")
    states, actions = load_all_frames(args.dataset_dir)
    print(
        f"Loaded {states.shape[0]} frames, "
        f"state_dim={states.shape[1]}, action_dim={actions.shape[1]}"
    )

    pct = compute_percentiles(states, actions)
    pct["num_episodes"] = len(
        glob.glob(os.path.join(args.dataset_dir, "data", "chunk-*", "episode_*.parquet"))
    )

    result = {args.embodiment: pct}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote percentiles to {args.output}")
    print(f"  Embodiment: {args.embodiment}")
    print(f"  Frames: {pct['num_frames']}, Episodes: {pct['num_episodes']}")
    print(f"  State dim: {pct['state_dim']}, Action dim: {pct['action_dim']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Re-run tests**

```bash
.venv/bin/python -m pytest tests_au/ops/filter/test_convert_lerobot_episodes.py -v
```

Expected: PASS (percentiles load may take a minute on full task).

- [ ] **Step 5: Commit (skip unless user asks)**

---

### Task 3: Summarizer (report + suggested YAML)

**Files:**
- Create: `tests_au/ops/filter/summarize_threshold_report.py`
- Test: `tests_au/ops/filter/test_summarize_threshold_report.py`

- [ ] **Step 1: Write failing tests**

```python
# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_threshold_report import (  # noqa: E402
    load_stats_jsonl,
    suggest_thresholds,
    write_report,
)


def _write_stats(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestSummarize(unittest.TestCase):
    def test_suggest_stage2_picks_in_band(self):
        # 10 eps: min_da from 0.50..0.95
        rows = []
        for i, da in enumerate([0.50, 0.55, 0.62, 0.64, 0.66, 0.68, 0.70, 0.80, 0.90, 0.95]):
            rows.append(
                {
                    "id": f"ep{i}",
                    "__dj__stats__": {
                        "sudden_change_flagged_ratio": 0.01 * i,
                        "sudden_change_max_run": i,
                        "state_action_min_da": da,
                        "state_action_mean_da": da,
                        "extreme_value_flagged_ratio": 0.08,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "stats.jsonl")
            _write_stats(p, rows)
            stats = load_stats_jsonl(p)
            sug = suggest_thresholds(stats)
            self.assertIn("da_threshold", sug)
            self.assertIn(sug["da_threshold"], (0.60, 0.65, 0.70))
            self.assertIn("max_flagged_ratio", sug)
            self.assertIn("alpha", sug)
            md = os.path.join(td, "threshold_report.md")
            yml = os.path.join(td, "clean_recipe_suggested.yaml")
            write_report(stats, sug, md, yml, probe_alpha=0.1)
            text = open(md).read()
            self.assertIn("Stage 1", text)
            self.assertIn("Stage 2", text)
            self.assertIn("Stage 3", text)
            self.assertIn("建议", text)
            self.assertTrue(os.path.isfile(yml))
            ytxt = open(yml).read()
            self.assertIn("robot_sudden_change_filter", ytxt)
            self.assertIn("robot_state_action_alignment_filter", ytxt)
            self.assertIn("robot_extreme_value_filter", ytxt)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/python -m pytest tests_au/ops/filter/test_summarize_threshold_report.py -v
```

- [ ] **Step 3: Implement summarizer**

Create `tests_au/ops/filter/summarize_threshold_report.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize Stage 1/2/3 analyze stats and suggest cleaning thresholds."""
import argparse
import json
import os
from typing import Dict, List

import numpy as np

S1_KEYS = ["sudden_change_flagged_ratio", "sudden_change_max_run"]
S2_KEYS = ["state_action_min_da", "state_action_mean_da"]
S3_KEYS = ["extreme_value_flagged_ratio"]
DA_CANDIDATES = (0.60, 0.65, 0.70)
TARGET_KEEP = 0.90


def _stat_dict(row: dict) -> dict:
    return row.get("__dj__stats__", row)


def load_stats_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    # validate keys
    missing = []
    for i, r in enumerate(rows):
        d = _stat_dict(r)
        for k in S1_KEYS + S2_KEYS + S3_KEYS:
            if k not in d:
                missing.append(f"row{i}:{k}")
    if missing:
        raise ValueError("Missing stats keys: " + ", ".join(missing[:20]))
    return rows


def _col(rows, key):
    return np.array([float(_stat_dict(r)[key]) for r in rows], dtype=float)


def _pcts(x):
    return {
        "min": float(np.min(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def suggest_thresholds(rows: List[dict], probe_alpha: float = 0.1) -> Dict:
    n = len(rows)
    ratio = _col(rows, "sudden_change_flagged_ratio")
    # max_flagged_ratio ≈ 90% keep if episode_discard on ratio > thr
    sorted_r = np.sort(ratio)
    # keep episodes with ratio <= thr; want keep ≈ TARGET_KEEP → thr = percentile
    thr_s1 = float(np.percentile(sorted_r, TARGET_KEEP * 100))
    thr_s1 = float(np.clip(thr_s1, 0.05, 0.5))

    min_da = _col(rows, "state_action_min_da")
    best_da = 0.65
    best_drop = None
    da_table = []
    for cand in DA_CANDIDATES:
        drop = int(np.sum(min_da < cand))
        drop_frac = drop / n
        da_table.append({"da_threshold": cand, "drop": drop, "drop_frac": drop_frac})
        # prefer drop_frac in [0.05, 0.15]
        score = abs(drop_frac - 0.10)
        if best_drop is None or score < best_drop:
            best_drop = score
            best_da = cand

    ev = _col(rows, "extreme_value_flagged_ratio")
    mean_ev = float(np.mean(ev))
    if mean_ev > 0.15:
        alpha = 0.2 if mean_ev <= 0.25 else 0.3
    elif mean_ev < 0.05:
        alpha = probe_alpha  # keep probe; optionally note can tighten
    else:
        alpha = probe_alpha

    return {
        "n_episodes": n,
        "max_flagged_ratio": round(thr_s1, 4),
        "da_threshold": best_da,
        "alpha": alpha,
        "s1_ratio_pcts": _pcts(ratio),
        "s1_max_run_pcts": _pcts(_col(rows, "sudden_change_max_run")),
        "s2_min_da_pcts": _pcts(min_da),
        "s2_da_table": da_table,
        "s3_ev_pcts": _pcts(ev),
        "s3_mean_flagged_ratio": mean_ev,
        "s1_keep_frac_if_discard": float(np.mean(ratio <= thr_s1)),
        "s2_keep_frac": 1.0 - next(
            t["drop_frac"] for t in da_table if t["da_threshold"] == best_da
        ),
    }


def write_report(rows, sug, md_path, yaml_path, probe_alpha=0.1):
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    lines = []
    lines.append("# Galaxea Pilot Threshold Report\n")
    lines.append(
        "> 建议值基于**本任务**分布的启发式；全量 227 任务清洗前请用 5–10 任务或全量复核。\n"
    )
    lines.append(f"- Episodes analyzed: **{sug['n_episodes']}**\n")
    lines.append(f"- Stage 3 probe alpha used in analyze: **{probe_alpha}**\n")

    lines.append("\n## Stage 1 — Sudden change\n")
    lines.append("| metric | min | p50 | p90 | p95 | max | mean |\n|---|---|---|---|---|---|---|\n")
    for name, key in [
        ("flagged_ratio", "s1_ratio_pcts"),
        ("max_run", "s1_max_run_pcts"),
    ]:
        p = sug[key]
        lines.append(
            f"| {name} | {p['min']:.4g} | {p['p50']:.4g} | {p['p90']:.4g} | "
            f"{p['p95']:.4g} | {p['max']:.4g} | {p['mean']:.4g} |\n"
        )
    lines.append(
        f"\n**建议 `max_flagged_ratio` = `{sug['max_flagged_ratio']}`** "
        f"（若改用 `episode_discard`，预计保留率 ≈ {sug['s1_keep_frac_if_discard']:.1%}）。\n"
    )
    lines.append(
        "当前分析用 `frame_mask`：不丢 episode，请结合 flagged 帧占比解读。\n"
    )

    lines.append("\n## Stage 2 — State-action alignment\n")
    p = sug["s2_min_da_pcts"]
    lines.append(
        f"min_da: p50={p['p50']:.4f}, p90={p['p90']:.4f}, mean={p['mean']:.4f}\n\n"
    )
    lines.append("| da_threshold | drop eps | drop frac |\n|---|---|---|\n")
    for t in sug["s2_da_table"]:
        lines.append(
            f"| {t['da_threshold']:.2f} | {t['drop']} | {t['drop_frac']:.1%} |\n"
        )
    lines.append(
        f"\n**建议 `da_threshold` = `{sug['da_threshold']}`** "
        f"（预计保留率 ≈ {sug['s2_keep_frac']:.1%}）。\n"
    )

    lines.append("\n## Stage 3 — Extreme value\n")
    p = sug["s3_ev_pcts"]
    lines.append(
        f"extreme_value_flagged_ratio: mean={sug['s3_mean_flagged_ratio']:.4f}, "
        f"p50={p['p50']:.4f}, p90={p['p90']:.4f}\n"
    )
    lines.append(
        f"\n**建议 `alpha` = `{sug['alpha']}`** "
        f"（目标帧标记率约 5%–15%；夹爪维保持 `exempt_dims: [7, 15]`）。\n"
    )

    lines.append("\n## Suggested clean params (summary)\n")
    lines.append("```yaml\n")
    lines.append(f"max_flagged_ratio: {sug['max_flagged_ratio']}\n")
    lines.append(f"da_threshold: {sug['da_threshold']}\n")
    lines.append(f"alpha: {sug['alpha']}\n")
    lines.append("```\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    yaml_text = f"""# Auto-suggested from pilot analyze — REVIEW before production use
project_name: 'galaxea-clean-suggested'
dataset_path: 'tests_au/ops/filter/outputs/pilot_analyze/lerobot_episodes.jsonl'
export_path: 'tests_au/ops/filter/outputs/pilot_analyze/clean_result.jsonl'
np: 4
executor_type: default
keep_stats_in_res_ds: true
text_keys: 'id'

custom_operator_paths:
  - 'data_juicer/_au'

process:
  - robot_sudden_change_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      threshold_mode: 'mad'
      mad_scale_residual: 6.0
      mad_scale_acc: 6.0
      mad_scale_jerk: 6.0
      max_flagged_ratio: {sug['max_flagged_ratio']}
      max_run_length: 10
      min_frames: 30
      exclusion_strategy: 'frame_mask'
  - robot_state_action_alignment_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      shared_dims: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
      action_is_delta: false
      max_lag: 15
      da_threshold: {sug['da_threshold']}
      eps_mode: 'range_frac'
      eps_frac: 0.01
      min_active_frames: 10
      min_frames: 20
      exclusion_strategy: 'flag_only'
  - robot_extreme_value_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      percentile_source: 'stats_json'
      percentile_stats_path: 'tests_au/ops/filter/outputs/pilot_analyze/percentiles.json'
      embodiment: 'galaxea_r1_lite'
      alpha: {sug['alpha']}
      exempt_dims: [7, 15]
      exclusion_strategy: 'frame_mask'
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    print(f"Wrote {md_path}")
    print(f"Wrote {yaml_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True, help="JSONL with __dj__stats__")
    parser.add_argument("--report", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--probe-alpha", type=float, default=0.1)
    args = parser.parse_args()
    rows = load_stats_jsonl(args.stats)
    sug = suggest_thresholds(rows, probe_alpha=args.probe_alpha)
    write_report(rows, sug, args.report, args.recipe, probe_alpha=args.probe_alpha)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
.venv/bin/python -m pytest tests_au/ops/filter/test_summarize_threshold_report.py -v
```

- [ ] **Step 5: Commit (skip unless user asks)**

---

### Task 4: Analyze YAML + acceptance shell

**Files:**
- Create: `tests_au/ops/filter/analyze_qwenrobomanip_pilot.yaml`
- Create: `tests_au/ops/filter/accept_analyze_qwenrobomanip_pilot.sh`

- [ ] **Step 1: Write analyze recipe**

```yaml
# Pilot analyze: Stage 1/2/3 stats only (for threshold tuning)
project_name: 'galaxea-pilot-analyze'
dataset_path: 'tests_au/ops/filter/outputs/pilot_analyze/lerobot_episodes.jsonl'
export_path: 'tests_au/ops/filter/outputs/pilot_analyze/analyze_result.jsonl'
np: 4
executor_type: default
keep_stats_in_res_ds: true
text_keys: 'id'

custom_operator_paths:
  - 'data_juicer/_au'

process:
  - robot_sudden_change_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      threshold_mode: 'mad'
      mad_scale_residual: 6.0
      mad_scale_acc: 6.0
      mad_scale_jerk: 6.0
      max_flagged_ratio: 0.3
      max_run_length: 10
      min_frames: 30
      exclusion_strategy: 'frame_mask'
      stats_export_path: 'tests_au/ops/filter/outputs/pilot_analyze/s1_stats.jsonl'
  - robot_state_action_alignment_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      shared_dims: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
      action_is_delta: false
      max_lag: 15
      da_threshold: 0.65
      eps_mode: 'range_frac'
      eps_frac: 0.01
      min_active_frames: 10
      min_frames: 20
      exclusion_strategy: 'flag_only'
      stats_export_path: 'tests_au/ops/filter/outputs/pilot_analyze/s2_stats.jsonl'
  - robot_extreme_value_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      percentile_source: 'stats_json'
      percentile_stats_path: 'tests_au/ops/filter/outputs/pilot_analyze/percentiles.json'
      embodiment: 'galaxea_r1_lite'
      alpha: 0.1
      exempt_dims: [7, 15]
      exclusion_strategy: 'frame_mask'
      stats_export_path: 'tests_au/ops/filter/outputs/pilot_analyze/s3_stats.jsonl'
```

- [ ] **Step 2: Write acceptance shell**

```bash
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

echo "=== Step 1: Convert ==="
"$VENV/bin/python" "$SCRIPT_DIR/convert_lerobot_episodes.py" \
  --dataset_dir "$DATASET" \
  --output "$OUT/lerobot_episodes.jsonl"

echo "=== Step 2: Percentiles ==="
"$VENV/bin/python" "$SCRIPT_DIR/compute_embodiment_percentiles.py" \
  --dataset_dir "$DATASET" \
  --output "$OUT/percentiles.json" \
  --embodiment galaxea_r1_lite

echo "=== Step 3: dj-analyze ==="
"$VENV/bin/dj-analyze" --config "$SCRIPT_DIR/analyze_qwenrobomanip_pilot.yaml"

echo "=== Step 4: Summarize thresholds ==="
# Prefer merged analyze_result; fall back to merging s1/s2/s3 if needed
STATS_SRC="$OUT/analyze_result.jsonl"
if [ ! -f "$STATS_SRC" ]; then
  echo "ERROR: missing $STATS_SRC" >&2
  exit 1
fi
"$VENV/bin/python" "$SCRIPT_DIR/summarize_threshold_report.py" \
  --stats "$STATS_SRC" \
  --report "$OUT/threshold_report.md" \
  --recipe "$OUT/clean_recipe_suggested.yaml" \
  --probe-alpha 0.1

# Verify report sections
"$VENV/bin/python" - <<PY
from pathlib import Path
p = Path("tests_au/ops/filter/outputs/pilot_analyze/threshold_report.md")
t = p.read_text(encoding="utf-8")
for s in ("Stage 1", "Stage 2", "Stage 3", "建议"):
    assert s in t, f"missing section {s}"
assert Path("tests_au/ops/filter/outputs/pilot_analyze/clean_recipe_suggested.yaml").is_file()
print("ACCEPTANCE PASSED:", p)
PY
```

Make executable: `chmod +x tests_au/ops/filter/accept_analyze_qwenrobomanip_pilot.sh`

**Note on stats source:** `dj-analyze` exports stats in `export_path` when configured. If export only has nested `__dj__stats__` after all ops, one file is enough. If Stage keys only appear in per-op `stats_export_path` files, update summarizer CLI to accept `--stats` as the analyze result **or** merge s1/s2/s3 by `id` before `suggest_thresholds`. Prefer merging in the shell if analyze_result lacks later-stage keys:

```python
# optional merge helper inside summarize_threshold_report.py
def merge_stats_jsonl(paths):
    by_id = {}
    for path in paths:
        for row in load_stats_jsonl_raw(path):  # without full-key validation
            eid = row.get("id")
            st = row.get("__dj__stats__", {})
            by_id.setdefault(eid, {"id": eid, "__dj__stats__": {}})
            by_id[eid]["__dj__stats__"].update(st)
    return list(by_id.values())
```

Use merge of `s1_stats.jsonl`, `s2_stats.jsonl`, `s3_stats.jsonl` in the acceptance script if `analyze_result.jsonl` validation fails missing keys.

- [ ] **Step 3: Dry-run unit tests only (no full accept yet)**

```bash
.venv/bin/python -m pytest tests_au/ops/filter/test_convert_lerobot_episodes.py tests_au/ops/filter/test_summarize_threshold_report.py -v
```

- [ ] **Step 4: Commit (skip unless user asks)**

---

### Task 5: End-to-end acceptance on Connect_Router

**Files:** none new (run existing)

- [ ] **Step 1: Run acceptance**

```bash
cd /mnt/r/share/zwy/Projects/data-juicer
bash tests_au/ops/filter/accept_analyze_qwenrobomanip_pilot.sh
```

Expected: prints `ACCEPTANCE PASSED` and creates:
- `tests_au/ops/filter/outputs/pilot_analyze/threshold_report.md`
- `tests_au/ops/filter/outputs/pilot_analyze/clean_recipe_suggested.yaml`

- [ ] **Step 2: Spot-check report**

```bash
head -n 80 tests_au/ops/filter/outputs/pilot_analyze/threshold_report.md
```

Confirm suggested `max_flagged_ratio`, `da_threshold`, `alpha` are present and numeric.

- [ ] **Step 3: If `dj-analyze` fails on custom ops**

Debug checklist:
1. `custom_operator_paths: ['data_juicer/_au']` and `_au/__init__.py` imports all three filters.
2. Working directory is repo root when invoking `dj-analyze`.
3. Percentiles path exists before analyze.
4. If Analyzer skips filters that write only meta — these filters write `__dj__stats__` scalars; should be fine.

- [ ] **Step 4: Commit (skip unless user asks)**

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Decomposed → 16-dim convert | Task 1 |
| Percentiles on same layout | Task 2 |
| `dj-analyze` Stage 1/2/3 recipe | Task 4 |
| `threshold_report.md` + suggestions | Task 3 |
| `clean_recipe_suggested.yaml` | Task 3 |
| One-shot accept shell | Task 4–5 |
| Unit tests convert + summarize | Task 1, 3 |
| Real Connect_Router path | Task 5 |
| No trunk `data_juicer/` edits | All tasks |
| Pilot-only percentiles caveat in report | Task 3 (`write_report` disclaimer) |

## Placeholder / consistency self-review

- No TBD left.
- Function names consistent: `pack_decomposed_to_16`, `episode_arrays_from_df`, `load_stats_jsonl`, `suggest_thresholds`, `write_report`.
- Stats keys match existing `_au` filter exports (`sudden_change_*`, `state_action_*`, `extreme_value_*`).
- Output dir fixed: `tests_au/ops/filter/outputs/pilot_analyze/`.
