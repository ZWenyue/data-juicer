# -*- coding: utf-8 -*-
import json
import os
import unittest

import numpy as np

from data_juicer._au.ops.filter.robot_sudden_change_filter import (
    RobotSuddenChangeFilter,
)
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

REAL_DATASET_DIR = (
    "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset"
    "/Connect_Router_Cables_20250625_002"
)
REAL_DATA_AVAILABLE = os.path.isdir(
    os.path.join(REAL_DATASET_DIR, "data", "chunk-000")
)


def _col(vec):
    """Turn a 1-D sequence into (T, 1) list-of-list to mimic single-dim states."""
    return np.asarray(vec, dtype=float).reshape(-1, 1).tolist()


def _load_episode_parquet(episode_idx: int):
    """Load a single episode parquet and return (states, actions) as list-of-list."""
    import pyarrow.parquet as pq

    pf = os.path.join(
        REAL_DATASET_DIR,
        "data",
        "chunk-000",
        f"episode_{episode_idx:06d}.parquet",
    )
    table = pq.read_table(pf)
    df = table.to_pandas()
    states = df["observation.state"].tolist()
    actions = df["action"].tolist()
    return states, actions


MANUAL = dict(
    signal_source="top_level",
    threshold_mode="manual",
    residual_threshold=0.3,
    acc_threshold=0.3,
    jerk_threshold=0.3,
    max_flagged_ratio=0.0,
    max_run_length=0,
    min_frames=4,
)

STATS_KEYS = [
    "sudden_change_keep",
    "sudden_change_flagged_ratio",
    "sudden_change_num_flagged",
    "sudden_change_max_run",
    "sudden_change_max_residual",
    "sudden_change_max_acc",
    "sudden_change_max_jerk",
]


class RobotSuddenChangeFilterTest(DataJuicerTestCaseBase):

    def _stats(self, op, states):
        sample = {Fields.stats: {}, "states": states}
        sample = op.compute_stats_single(sample)
        return sample

    def _report(self, sample):
        return json.loads(sample[Fields.meta]["sudden_change_report"])

    def _mask(self, sample):
        return json.loads(sample[Fields.meta]["valid_frame_mask"])

    # ---- smooth sine: keep ----
    def test_smooth_sine_keep(self):
        t = np.arange(40)
        x = 0.1 * np.sin(2 * np.pi * t / 20)
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col(x))
        self.assertTrue(s[Fields.stats]["sudden_change_keep"])
        self.assertEqual(s[Fields.stats]["sudden_change_num_flagged"], 0)
        self.assertTrue(op.process_single(s))

    # ---- slow drift: keep ----
    def test_slow_drift_keep(self):
        t = np.arange(40)
        x = 0.01 * t
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col(x))
        self.assertTrue(s[Fields.stats]["sudden_change_keep"])
        self.assertEqual(s[Fields.stats]["sudden_change_num_flagged"], 0)

    # ---- single spike: drop ----
    def test_single_spike_drop(self):
        t = np.arange(40)
        x = 0.1 * np.sin(2 * np.pi * t / 20)
        x[20] += 1.0
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col(x))
        self.assertFalse(s[Fields.stats]["sudden_change_keep"])
        report = self._report(s)
        flagged = report["blocks"][0]["flagged_frame_ids"]
        self.assertIn(20, flagged)
        self.assertFalse(op.process_single(s))

    # ---- step change: drop ----
    def test_step_change_drop(self):
        t = np.arange(40)
        x = 0.1 * np.sin(2 * np.pi * t / 20) + (t >= 20).astype(float) * 1.0
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col(x))
        self.assertFalse(s[Fields.stats]["sudden_change_keep"])
        self.assertGreaterEqual(s[Fields.stats]["sudden_change_num_flagged"], 1)

    # ---- angular wrapping: on→keep, off→drop ----
    def test_angular_wrapping(self):
        t = np.arange(40)
        true = 0.3 * t
        wrapped = ((true + np.pi) % (2 * np.pi)) - np.pi

        op_on = RobotSuddenChangeFilter(angular_dims=[0], **MANUAL)
        s_on = self._stats(op_on, _col(wrapped))
        self.assertTrue(s_on[Fields.stats]["sudden_change_keep"])
        self.assertEqual(s_on[Fields.stats]["sudden_change_num_flagged"], 0)

        op_off = RobotSuddenChangeFilter(**MANUAL)
        s_off = self._stats(op_off, _col(wrapped))
        self.assertFalse(s_off[Fields.stats]["sudden_change_keep"])

    # ---- short trajectory: keep ----
    def test_short_trajectory_keep(self):
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col([0.0, 1.0, 0.0]))
        self.assertTrue(s[Fields.stats]["sudden_change_keep"])
        report = self._report(s)
        blk = report["blocks"][0]
        self.assertTrue(blk["insufficient_length"])
        self.assertEqual(s[Fields.stats]["sudden_change_num_flagged"], 0)

    # ---- frame_mask strategy: keep sample + mask flagged frames ----
    def test_frame_mask_strategy(self):
        t = np.arange(40)
        x = 0.1 * np.sin(2 * np.pi * t / 20)
        x[20] += 1.0
        cfg = dict(MANUAL)
        cfg["exclusion_strategy"] = "frame_mask"
        op = RobotSuddenChangeFilter(**cfg)
        s = self._stats(op, _col(x))
        self.assertTrue(op.process_single(s))
        mask_dict = self._mask(s)
        mask = mask_dict["states"]
        self.assertEqual(len(mask), 40)
        self.assertFalse(mask[20])

    # ---- episode_discard pipeline: good/spike/short ----
    def test_pipeline_episode_discard(self):
        t = np.arange(40)
        good = 0.1 * np.sin(2 * np.pi * t / 20)
        spike = good.copy()
        spike[20] += 1.0
        ds_list = [
            {"id": "good", "states": _col(good)},
            {"id": "spike", "states": _col(spike)},
            {"id": "short", "states": _col([0.0, 1.0, 0.0])},
        ]
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(
                name=Fields.stats, column=[{}] * dataset.num_rows
            )
        op = RobotSuddenChangeFilter(**MANUAL)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        kept_ids = sorted(
            dataset.select_columns(["id"]).to_list(), key=lambda r: r["id"]
        )
        self.assertEqual(kept_ids, [{"id": "good"}, {"id": "short"}])

    # ------------------------------------------------------------------ #
    # Real dataset tests (skipped if data not available)
    # ------------------------------------------------------------------ #
    @unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not available")
    def test_real_episode_computes_stats(self):
        """Load one real episode and verify stats completeness + operator stability."""
        states, actions = _load_episode_parquet(1)
        op = RobotSuddenChangeFilter(
            signal_source="top_level",
            threshold_mode="mad",
            mad_scale_residual=6.0,
            mad_scale_acc=6.0,
            mad_scale_jerk=6.0,
            max_flagged_ratio=0.3,
            max_run_length=10,
            min_frames=30,
        )
        sample = {
            Fields.stats: {},
            "states": states,
            "actions": actions,
        }
        sample = op.compute_stats_single(sample)

        for key in STATS_KEYS:
            self.assertIn(key, sample[Fields.stats], f"Missing stats key: {key}")

        report = self._report(sample)
        self.assertIn("blocks", report)
        self.assertGreaterEqual(report["num_frames"], 100)
        self.assertGreaterEqual(len(report["blocks"]), 1)

    @unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not available")
    def test_real_dataset_pipeline(self):
        """Load all 16 episodes through map→filter pipeline."""
        import glob

        import pyarrow.parquet as pq

        pattern = os.path.join(
            REAL_DATASET_DIR, "data", "chunk-000", "episode_*.parquet"
        )
        parquet_files = sorted(glob.glob(pattern))
        self.assertGreater(len(parquet_files), 0)

        ds_list = []
        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            ep_idx = int(df["episode_index"].iloc[0])
            ds_list.append({
                "id": f"episode_{ep_idx:06d}",
                "states": df["observation.state"].tolist(),
                "actions": df["action"].tolist(),
            })

        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(
                name=Fields.stats, column=[{}] * dataset.num_rows
            )

        op = RobotSuddenChangeFilter(
            signal_source="top_level",
            threshold_mode="mad",
            mad_scale_residual=6.0,
            mad_scale_acc=6.0,
            mad_scale_jerk=6.0,
            max_flagged_ratio=0.3,
            max_run_length=10,
            min_frames=30,
            exclusion_strategy="frame_mask",
        )
        dataset = dataset.map(op.compute_stats)
        result = dataset.filter(op.process)

        # frame_mask keeps all samples; verify pipeline completes + stats present
        self.assertEqual(result.num_rows, len(parquet_files))

        for row in dataset.to_list():
            stats = row.get(Fields.stats, {})
            for key in STATS_KEYS:
                self.assertIn(key, stats, f"Missing stats key: {key}")
            self.assertGreater(
                stats["sudden_change_max_residual"], 0,
                "Expected non-zero residual for real robot data",
            )


if __name__ == "__main__":
    unittest.main()
