"""Regression tests for conservative robot-clean threshold analysis."""

import unittest
from tempfile import TemporaryDirectory

import numpy as np

from data_juicer._au.pipeline.robot_clean.analyze import (
    _sample_evenly,
    suggest_numeric,
    suggest_video,
)
from data_juicer._au.pipeline.robot_clean.config import CleanConfig
from data_juicer._au.pipeline.robot_clean.recipe import build_process_ops


class RobotCleanAnalyzeTest(unittest.TestCase):

    def test_numeric_thresholds_do_not_reject_a_fixed_healthy_tail(self):
        rows = []
        for i in range(20):
            rows.append(
                {
                    "stats": {
                        "sudden_change_flagged_ratio": 0.01 + i * 0.001,
                        "sudden_change_max_run": 1 + i % 4,
                        "state_action_min_da": 0.95,
                        "state_action_mean_da": 0.98,
                        "extreme_value_flagged_ratio": 0.01,
                    }
                }
            )
        rows.append(
            {
                "stats": {
                    "sudden_change_flagged_ratio": 0.8,
                    "sudden_change_max_run": 100,
                    "state_action_min_da": 0.1,
                    "state_action_mean_da": 0.5,
                    "extreme_value_flagged_ratio": 0.01,
                }
            }
        )

        result = suggest_numeric(rows, probe_alpha=0.1)

        self.assertEqual(result["threshold_policy"], "conservative_observed_envelope")
        self.assertGreaterEqual(result["max_flagged_ratio"], 0.3)
        self.assertGreaterEqual(result["max_run_length"], 10)
        self.assertEqual(result["wash"]["numeric_union_episode_drop_frac"], 0.0)

    def test_video_threshold_does_not_mark_a_fixed_percentile_bad(self):
        blur = np.linspace(9.8, 10.2, 100)
        black = np.full(100, 200.0)
        result = suggest_video(
            {
                "_blur": blur,
                "_black": black,
                "_per_ep": [
                    {
                        "blur": blur.tolist(),
                        "black": black.tolist(),
                        "n_sampled": len(blur),
                    }
                ],
                "probe_sampling_fps": 2.0,
                "original_fps": 15.0,
            }
        )

        self.assertEqual(result["blur_flag_frac_at_suggested"], 0.0)
        self.assertEqual(result["wash"]["episode_drop_frac"], 0.0)

    def test_video_probe_is_evenly_distributed(self):
        items = [f"episode_{i:03d}" for i in range(100)]
        sampled = _sample_evenly(items, 8)

        self.assertEqual(len(sampled), 8)
        self.assertEqual(sampled[0], items[0])
        self.assertEqual(sampled[-1], items[-1])
        self.assertGreater(items.index(sampled[1]), 1)

    def test_recipe_uses_semantic_keyframes_without_default_rejection(self):
        with TemporaryDirectory() as output:
            ops = build_process_ops(CleanConfig(dataset="/dataset", output_dir=output))

        keyframe = next(
            op["robot_key_frame_detector_mapper"]
            for op in ops
            if "robot_key_frame_detector_mapper" in op
        )
        gate = next(
            op["robot_video_quality_episode_filter"]
            for op in ops
            if "robot_video_quality_episode_filter" in op
        )
        self.assertEqual(keyframe["gripper_dims"], [7, 15])
        self.assertEqual(keyframe["exempt_dims"], [7, 15])
        self.assertEqual(keyframe["gripper_delta_threshold_frac"], 0.05)
        self.assertIsNone(gate["max_keyframe_overlap"])


if __name__ == "__main__":
    unittest.main()
