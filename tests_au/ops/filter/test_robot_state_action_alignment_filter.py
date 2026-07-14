# -*- coding: utf-8 -*-
import glob
import json
import os
import unittest

import numpy as np

from data_juicer._au.ops.filter.robot_state_action_alignment_filter import (
    RobotStateActionAlignmentFilter,
)
from data_juicer._au.utils.lerobot_episode_io import load_episode_arrays
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

REAL_DATASET_DIR = "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset" "/Connect_Router_Cables_20250625_002"
REAL_DATA_AVAILABLE = os.path.isdir(os.path.join(REAL_DATASET_DIR, "data", "chunk-000"))

# Galaxea unified layout: arms are position-comparable in both state & action.
GALAXEA_ARM_DIMS = list(range(0, 7)) + list(range(8, 15))


def _col(vec):
    """Turn a 1-D sequence into (T, 1) list-of-list to mimic single-dim signal."""
    return np.asarray(vec, dtype=float).reshape(-1, 1).tolist()


def _load_episode_parquet(episode_idx: int):
    """Load a single episode parquet and return (states, actions) as list-of-list.

    Uses the shared loader (handles both unified ``observation.state``/``action``
    columns and Galaxea decomposed per-arm columns) instead of hardcoding a
    column name, since real datasets may use either schema.
    """
    pf = os.path.join(
        REAL_DATASET_DIR,
        "data",
        "chunk-000",
        f"episode_{episode_idx:06d}.parquet",
    )
    states, actions = load_episode_arrays(pf)
    return states.tolist(), actions.tolist()


# Shared config for synthetic single-dim scenarios.
BASE = dict(
    signal_source="top_level",
    eps_mode="abs",
    eps_abs=0.005,
    min_frames=20,
    min_active_frames=10,
    max_lag=15,
    da_threshold=0.65,
)

STATS_KEYS = [
    "state_action_alignment_keep",
    "state_action_min_da",
    "state_action_mean_da",
    "state_action_num_flagged_dims",
    "state_action_num_checked_dims",
    "state_action_max_abs_lag",
]


class RobotStateActionAlignmentFilterTest(DataJuicerTestCaseBase):

    def _stats(self, op, states, actions):
        sample = {Fields.stats: {}, "states": states, "actions": actions}
        return op.compute_stats_single(sample)

    def _report(self, sample, op):
        return json.loads(sample[Fields.meta][op.report_field])

    # ---- aligned (identical trend): keep, DA == 1 ----
    def test_aligned_keep(self):
        rng = np.random.RandomState(0)
        walk = np.cumsum(rng.randn(200)) * 0.05
        op = RobotStateActionAlignmentFilter(**BASE)
        s = self._stats(op, _col(walk), _col(walk))
        self.assertTrue(s[Fields.stats]["state_action_alignment_keep"])
        self.assertAlmostEqual(s[Fields.stats]["state_action_min_da"], 1.0, places=6)
        self.assertEqual(s[Fields.stats]["state_action_num_flagged_dims"], 0)
        self.assertTrue(op.process_single(s))

    # ---- independent random walks: misaligned -> drop ----
    def test_independent_walks_drop(self):
        state = np.cumsum(np.random.RandomState(0).randn(200)) * 0.05
        action = np.cumsum(np.random.RandomState(999).randn(200)) * 0.05
        op = RobotStateActionAlignmentFilter(**BASE)
        s = self._stats(op, _col(state), _col(action))
        self.assertFalse(s[Fields.stats]["state_action_alignment_keep"])
        self.assertLess(s[Fields.stats]["state_action_min_da"], 0.65)
        self.assertGreaterEqual(s[Fields.stats]["state_action_num_flagged_dims"], 1)
        self.assertFalse(op.process_single(s))

    # ---- action leads state by a fixed lag L: recovered by xcorr -> keep ----
    def test_temporal_lag_recovered_keep(self):
        rng = np.random.RandomState(7)
        base = np.cumsum(rng.randn(200)) * 0.05
        L = 5
        state = base
        action = np.roll(base, -L)  # action leads state by L frames
        op = RobotStateActionAlignmentFilter(**BASE)
        s = self._stats(op, _col(state), _col(action))
        self.assertTrue(s[Fields.stats]["state_action_alignment_keep"])
        rep = self._report(s, op)
        self.assertEqual(rep["pairs"][0]["dim_lag"]["0"], L)
        self.assertTrue(op.process_single(s))

    # ---- delta actions: integrate -> keep; without integration -> drop ----
    def test_delta_action_integration(self):
        state = np.cumsum(np.random.RandomState(0).randn(200)) * 0.05
        delta = np.concatenate([[0.0], np.diff(state)])

        cfg_on = dict(BASE)
        cfg_on["action_is_delta"] = True
        op_on = RobotStateActionAlignmentFilter(**cfg_on)
        s_on = self._stats(op_on, _col(state), _col(delta))
        self.assertTrue(s_on[Fields.stats]["state_action_alignment_keep"])
        self.assertAlmostEqual(s_on[Fields.stats]["state_action_min_da"], 1.0, places=6)

        # Without integration, position is compared against velocity -> low DA.
        op_off = RobotStateActionAlignmentFilter(**BASE)
        s_off = self._stats(op_off, _col(state), _col(delta))
        self.assertFalse(s_off[Fields.stats]["state_action_alignment_keep"])
        self.assertLess(
            s_off[Fields.stats]["state_action_min_da"],
            s_on[Fields.stats]["state_action_min_da"],
        )

    # ---- held action + noisy state: must not be mistaken for misalignment ----
    # Regression: when action is exactly constant (controller holding position)
    # and state only has small sensor jitter around it, sign(diff(action))==0
    # can never equal sign(diff(state))!=0. Gating "active" on OR let this
    # jitter alone dominate the active-frame set and force DA -> 0 even though
    # there is no real state/action disagreement (action never asked for any
    # direction to begin with). Gating on AND excludes these hold-phase frames.
    def test_held_action_with_state_jitter_keep(self):
        rng = np.random.RandomState(0)
        t_len = 200
        state = np.full(t_len, 0.3) + rng.normal(scale=0.01, size=t_len)
        action = np.full(t_len, 0.3)  # controller holding position: never moves
        cfg = dict(BASE)
        # eps far below the state jitter scale: |ds| clears it on nearly every
        # frame, while |da| is exactly 0 on every frame (action never moves).
        cfg["eps_abs"] = 1e-6
        op = RobotStateActionAlignmentFilter(**cfg)
        s = self._stats(op, _col(state), _col(action))
        # no dim should even be checked: action never exceeds eps, so there is
        # no "active" frame with a real trend on both sides to compare.
        self.assertEqual(s[Fields.stats]["state_action_num_checked_dims"], 0)
        self.assertTrue(op.process_single(s))

    # ---- static (never moving) dim: skipped, not failed -> keep ----
    def test_static_dim_skipped_keep(self):
        static = np.ones(200) * 0.3
        op = RobotStateActionAlignmentFilter(**BASE)
        s = self._stats(op, _col(static), _col(static))
        self.assertEqual(s[Fields.stats]["state_action_num_checked_dims"], 0)
        self.assertTrue(op.process_single(s))

    # ---- short trajectory: safe keep, no check ----
    def test_short_trajectory_keep(self):
        op = RobotStateActionAlignmentFilter(**BASE)
        s = self._stats(op, _col([0.0, 1.0, 2.0, 3.0, 4.0]), _col([0.0, 1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(s[Fields.stats]["state_action_num_checked_dims"], 0)
        self.assertTrue(op.process_single(s))
        rep = self._report(s, op)
        self.assertTrue(rep["pairs"][0]["insufficient_length"])

    # ---- flag_only strategy: never dropped, only annotated ----
    def test_flag_only_strategy(self):
        state = np.cumsum(np.random.RandomState(0).randn(200)) * 0.05
        action = np.cumsum(np.random.RandomState(999).randn(200)) * 0.05
        cfg = dict(BASE)
        cfg["exclusion_strategy"] = "flag_only"
        op = RobotStateActionAlignmentFilter(**cfg)
        s = self._stats(op, _col(state), _col(action))
        self.assertFalse(s[Fields.stats]["state_action_alignment_keep"])
        self.assertTrue(op.process_single(s))  # kept despite flags

    # ---- dimension selection: exempt a bad dim -> keep ----
    def test_exempt_dims_keep(self):
        good = np.cumsum(np.random.RandomState(0).randn(200)) * 0.05
        bad = np.cumsum(np.random.RandomState(999).randn(200)) * 0.05
        state = np.stack([good, bad], axis=1).tolist()
        action = np.stack([good, good], axis=1).tolist()  # dim1 mismatched

        op_all = RobotStateActionAlignmentFilter(**BASE)
        s_all = self._stats(op_all, state, action)
        self.assertFalse(s_all[Fields.stats]["state_action_alignment_keep"])

        cfg = dict(BASE)
        cfg["exempt_dims"] = [1]
        op_ex = RobotStateActionAlignmentFilter(**cfg)
        s_ex = self._stats(op_ex, state, action)
        self.assertTrue(s_ex[Fields.stats]["state_action_alignment_keep"])
        self.assertEqual(s_ex[Fields.stats]["state_action_num_checked_dims"], 1)

    # ---- episode_discard pipeline: aligned kept, misaligned dropped, short kept ----
    def test_pipeline_episode_discard(self):
        good = np.cumsum(np.random.RandomState(0).randn(200)) * 0.05
        bad = np.cumsum(np.random.RandomState(999).randn(200)) * 0.05
        ds_list = [
            {"id": "aligned", "states": _col(good), "actions": _col(good)},
            {"id": "misaligned", "states": _col(good), "actions": _col(bad)},
            {"id": "short", "states": _col([0.0, 1.0, 2.0]), "actions": _col([0.0, 1.0, 2.0])},
        ]
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        op = RobotStateActionAlignmentFilter(**BASE)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        kept_ids = sorted(r["id"] for r in dataset.select_columns(["id"]).to_list())
        self.assertEqual(kept_ids, ["aligned", "short"])

    # ------------------------------------------------------------------ #
    # Real dataset tests (skipped if data not available)
    # ------------------------------------------------------------------ #
    @unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not available")
    def test_real_episode_computes_stats(self):
        states, actions = _load_episode_parquet(1)
        op = RobotStateActionAlignmentFilter(
            signal_source="top_level",
            shared_dims=GALAXEA_ARM_DIMS,
            eps_mode="range_frac",
            eps_frac=0.01,
            min_frames=20,
            min_active_frames=10,
            max_lag=15,
            da_threshold=0.65,
        )
        sample = {Fields.stats: {}, "states": states, "actions": actions}
        sample = op.compute_stats_single(sample)
        for key in STATS_KEYS:
            self.assertIn(key, sample[Fields.stats], f"Missing stats key: {key}")
        rep = self._report(sample, op)
        self.assertIn("pairs", rep)
        self.assertGreaterEqual(rep["pairs"][0]["num_frames"], 100)

    @unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not available")
    def test_real_dataset_pipeline(self):
        import pyarrow.parquet as pq

        pattern = os.path.join(REAL_DATASET_DIR, "data", "chunk-000", "episode_*.parquet")
        parquet_files = sorted(glob.glob(pattern))
        self.assertGreater(len(parquet_files), 0)

        ds_list = []
        for pf in parquet_files:
            df = pq.read_table(pf).to_pandas()
            ep_idx = int(df["episode_index"].iloc[0])
            states, actions = load_episode_arrays(pf)
            ds_list.append(
                {
                    "id": f"episode_{ep_idx:06d}",
                    "states": states.tolist(),
                    "actions": actions.tolist(),
                }
            )

        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        op = RobotStateActionAlignmentFilter(
            signal_source="top_level",
            shared_dims=GALAXEA_ARM_DIMS,
            eps_mode="range_frac",
            eps_frac=0.01,
            min_frames=20,
            min_active_frames=10,
            max_lag=15,
            da_threshold=0.65,
            exclusion_strategy="flag_only",
        )
        dataset = dataset.map(op.compute_stats)
        result = dataset.filter(op.process)
        # flag_only keeps all rows; verify pipeline completes + stats present
        self.assertEqual(result.num_rows, len(parquet_files))
        for row in dataset.to_list():
            stats = row.get(Fields.stats, {})
            for key in STATS_KEYS:
                self.assertIn(key, stats, f"Missing stats key: {key}")


if __name__ == "__main__":
    unittest.main()
