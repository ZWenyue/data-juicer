# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest

import numpy as np

from data_juicer._au.ops.filter.robot_extreme_value_filter import (
    RobotExtremeValueFilter,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

REAL_DATASET_DIR = (
    "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset"
    "/Connect_Router_Cables_20250625_002"
)
REAL_DATA_AVAILABLE = os.path.isdir(
    os.path.join(REAL_DATASET_DIR, "data", "chunk-000")
)

STATS_KEYS = [
    "extreme_value_keep",
    "extreme_value_flagged_frames",
    "extreme_value_flagged_ratio",
    "extreme_value_num_check_dims",
    "extreme_value_num_frames",
]


def _make_normal_trajectory(T=50, D=4, seed=42):
    """Smooth sinusoidal trajectory — no extreme values."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, T)
    states = np.column_stack([np.sin(t + i * 0.5) for i in range(D)])
    states += rng.normal(0, 0.01, states.shape)
    actions = np.diff(states, axis=0, prepend=states[:1])
    return states.tolist(), actions.tolist()


def _make_sample(states, actions, robot_type="test_robot"):
    return {
        "id": "test_ep",
        "states": states,
        "actions": actions,
        "robot_type": robot_type,
        Fields.stats: {},
        Fields.meta: {},
    }


def _write_percentiles_json(path, embodiment, D_state, D_action, q01_val=-2.0, q99_val=2.0):
    """Write a simple percentiles JSON file with uniform q01/q99 across dims."""
    data = {
        embodiment: {
            "state": {
                "q01": [q01_val] * D_state,
                "q99": [q99_val] * D_state,
            },
            "action": {
                "q01": [q01_val] * D_action,
                "q99": [q99_val] * D_action,
            },
        }
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class RobotExtremeValueFilterTest(DataJuicerTestCaseBase):

    def _run(self, op, sample):
        sample = op.compute_stats_single(sample)
        keep = op.process_single(sample)
        return sample, keep

    # ---- Test 1: Normal trajectory, param mode — all frames kept ----
    def test_normal_trajectory_param_mode(self):
        states, actions = _make_normal_trajectory(T=50, D=4)
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
            exclusion_strategy="frame_mask",
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertTrue(keep)
        self.assertEqual(sample[Fields.stats]["extreme_value_flagged_frames"], 0)
        self.assertAlmostEqual(sample[Fields.stats]["extreme_value_flagged_ratio"], 0.0)
        for k in STATS_KEYS:
            self.assertIn(k, sample[Fields.stats])

    # ---- Test 2: Injected extreme values — flagged ----
    def test_injected_extremes_flagged(self):
        states, actions = _make_normal_trajectory(T=50, D=4)
        arr = np.array(states)
        arr[10, 0] = 100.0  # extreme
        arr[20, 2] = -100.0  # extreme
        states = arr.tolist()
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
            exclusion_strategy="frame_mask",
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertTrue(keep)
        self.assertGreaterEqual(sample[Fields.stats]["extreme_value_flagged_frames"], 2)

    # ---- Test 3: Self mode — per-episode percentiles ----
    def test_self_mode(self):
        states, actions = _make_normal_trajectory(T=100, D=4)
        arr = np.array(states)
        arr[50, 0] = 50.0  # outlier beyond self-computed q99
        states = arr.tolist()
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="self",
            alpha=0.1,
            exclusion_strategy="frame_mask",
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertTrue(keep)
        self.assertGreaterEqual(sample[Fields.stats]["extreme_value_flagged_frames"], 1)

    # ---- Test 4: stats_json mode ----
    def test_stats_json_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pct_path = os.path.join(tmpdir, "percentiles.json")
            _write_percentiles_json(pct_path, "test_robot", 4, 4)

            states, actions = _make_normal_trajectory(T=50, D=4)
            op = RobotExtremeValueFilter(
                signal_source="top_level",
                percentile_source="stats_json",
                percentile_stats_path=pct_path,
                embodiment_field="robot_type",
                alpha=0.1,
                exclusion_strategy="frame_mask",
            )
            sample, keep = self._run(op, _make_sample(states, actions))
            self.assertTrue(keep)
            self.assertEqual(sample[Fields.stats]["extreme_value_flagged_frames"], 0)

    # ---- Test 5: Exempt dims — gripper values don't trigger ----
    def test_exempt_dims(self):
        states, actions = _make_normal_trajectory(T=50, D=4)
        arr = np.array(states)
        arr[:, 2] = 100.0  # dim 2 = extreme everywhere, but exempt
        states = arr.tolist()
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
            exempt_dims=[2],
            exclusion_strategy="frame_mask",
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertTrue(keep)
        self.assertEqual(sample[Fields.stats]["extreme_value_flagged_frames"], 0)

    # ---- Test 6: Frame remove — arrays trimmed ----
    def test_frame_remove(self):
        states, actions = _make_normal_trajectory(T=50, D=4)
        arr_s = np.array(states)
        arr_s[10, 0] = 100.0
        arr_s[11, 0] = 100.0
        states = arr_s.tolist()
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
            exclusion_strategy="frame_remove",
            min_frames=4,
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertTrue(keep)
        new_T = len(sample["states"])
        self.assertLess(new_T, 50)
        self.assertGreaterEqual(new_T, 48)  # 2 frames removed

    # ---- Test 7: Episode discard — high flagged ratio ----
    def test_episode_discard_high_ratio(self):
        states, actions = _make_normal_trajectory(T=50, D=4)
        arr = np.array(states)
        arr[:30, 0] = 100.0  # 60% of frames extreme
        states = arr.tolist()
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
            exclusion_strategy="episode_discard",
            max_flagged_ratio=0.3,
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertFalse(keep)

    # ---- Test 8: Episode discard — low flagged ratio ----
    def test_episode_discard_low_ratio(self):
        states, actions = _make_normal_trajectory(T=50, D=4)
        arr = np.array(states)
        arr[0, 0] = 100.0  # 1 frame out of 50 = 2% < 30%
        states = arr.tolist()
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
            exclusion_strategy="episode_discard",
            max_flagged_ratio=0.3,
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertTrue(keep)

    # ---- Test 9: Too few frames — skip with defaults ----
    def test_too_few_frames(self):
        states = [[0.0, 0.0]] * 3
        actions = [[0.0, 0.0]] * 3
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-1.0, -1.0],
            state_q99=[1.0, 1.0],
            action_q01=[-1.0, -1.0],
            action_q99=[1.0, 1.0],
            min_frames=4,
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertTrue(keep)
        self.assertEqual(sample[Fields.stats]["extreme_value_flagged_frames"], 0)

    # ---- Test 10: Meta report present and parseable ----
    def test_meta_report_present(self):
        states, actions = _make_normal_trajectory(T=50, D=4)
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
        )
        sample, _ = self._run(op, _make_sample(states, actions))
        report_str = sample[Fields.meta].get("extreme_value_report")
        self.assertIsNotNone(report_str)
        report = json.loads(report_str)
        self.assertIn("flagged_frames", report)
        self.assertIn("alpha", report)

    # ---- Test 11: Mask AND merge with existing mask ----
    def test_mask_and_merge(self):
        states, actions = _make_normal_trajectory(T=20, D=4)
        arr = np.array(states)
        arr[5, 0] = 100.0
        states = arr.tolist()
        existing_mask = [True] * 20
        existing_mask[10] = False  # pre-existing mask from Stage 1

        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
            exclusion_strategy="frame_mask",
        )
        sample = _make_sample(states, actions)
        sample[Fields.meta]["valid_frame_mask"] = json.dumps(existing_mask)
        sample, keep = self._run(op, sample)
        self.assertTrue(keep)
        mask = json.loads(sample[Fields.meta]["valid_frame_mask"])
        self.assertFalse(mask[5])   # flagged by extreme value
        self.assertFalse(mask[10])  # preserved from Stage 1 mask

    # ---- Test 12: check_dims include ----
    def test_check_dims_include(self):
        states, actions = _make_normal_trajectory(T=50, D=4)
        arr = np.array(states)
        arr[10, 3] = 100.0  # dim 3 is extreme, but we only check dim 0
        states = arr.tolist()
        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="param",
            state_q01=[-2.0] * 4,
            state_q99=[2.0] * 4,
            action_q01=[-2.0] * 4,
            action_q99=[2.0] * 4,
            alpha=0.1,
            check_dims={"include": [0]},
            exclusion_strategy="frame_mask",
        )
        sample, keep = self._run(op, _make_sample(states, actions))
        self.assertTrue(keep)
        self.assertEqual(sample[Fields.stats]["extreme_value_flagged_frames"], 0)

    # ---- Test 13: Validation errors ----
    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            RobotExtremeValueFilter(signal_source="invalid")
        with self.assertRaises(ValueError):
            RobotExtremeValueFilter(percentile_source="invalid")
        with self.assertRaises(ValueError):
            RobotExtremeValueFilter(exclusion_strategy="invalid")
        with self.assertRaises(ValueError):
            RobotExtremeValueFilter(percentile_source="stats_json")
        with self.assertRaises(ValueError):
            RobotExtremeValueFilter(percentile_source="param")

    # ---- Test 14: Real data (skip if unavailable) ----
    @unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
    def test_real_data(self):
        import pyarrow.parquet as pq

        pf = os.path.join(REAL_DATASET_DIR, "data", "chunk-000", "episode_000000.parquet")
        table = pq.read_table(pf)
        df = table.to_pandas()
        states = df["observation.state"].tolist()
        actions = df["action"].tolist()

        D_s = len(states[0])
        D_a = len(actions[0])

        op = RobotExtremeValueFilter(
            signal_source="top_level",
            percentile_source="self",
            alpha=0.1,
            exempt_dims=[7, 15],
            exclusion_strategy="frame_mask",
        )
        sample = _make_sample(states, actions, robot_type="galaxea_r1_lite")
        sample, keep = self._run(op, sample)
        self.assertTrue(keep)
        self.assertEqual(sample[Fields.stats]["extreme_value_num_frames"], len(states))
        self.assertEqual(
            sample[Fields.stats]["extreme_value_num_check_dims"],
            (D_s - 2) + (D_a - 2),  # 2 exempt dims (7, 15) if they exist
        )


if __name__ == "__main__":
    unittest.main()
