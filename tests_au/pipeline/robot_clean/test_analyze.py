"""Regression tests for robot-clean threshold analysis."""

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

    def test_numeric_thresholds_reject_extreme_tail_and_record_reasons(self):
        rows = []
        for i in range(20):
            rows.append(
                {
                    "id": f"healthy_{i:02d}",
                    "stats": {
                        "sudden_change_flagged_ratio": 0.01 + i * 0.001,
                        "sudden_change_max_run": 1 + i % 4,
                        "state_action_min_da": 0.95,
                        "state_action_mean_da": 0.98,
                        "extreme_value_flagged_ratio": 0.01,
                    },
                }
            )
        rows.append(
            {
                "id": "bad_outlier",
                "stats": {
                    "sudden_change_flagged_ratio": 0.8,
                    "sudden_change_max_run": 100,
                    "state_action_min_da": 0.1,
                    "state_action_mean_da": 0.5,
                    "extreme_value_flagged_ratio": 0.01,
                },
            }
        )

        result = suggest_numeric(rows, probe_alpha=0.1)

        self.assertEqual(
            result["threshold_policy"], "s1_percentile_auto_reject_and_review"
        )
        self.assertLess(result["max_flagged_ratio"], 0.8)
        self.assertLess(result["max_run_length"], 100)
        self.assertAlmostEqual(result["wash"]["numeric_union_episode_drop_frac"], 1 / 21)
        decision = next(item for item in result["episode_decisions"] if item["id"] == "bad_outlier")
        self.assertEqual(decision["decision"], "auto_reject")
        self.assertIn("stage1_flagged_ratio", decision["auto_reject_reasons"])
        self.assertIn("stage1_max_run", decision["auto_reject_reasons"])

    def test_numeric_threshold_percentiles_must_be_ordered(self):
        rows = [
            {
                "stats": {
                    "sudden_change_flagged_ratio": 0.1,
                    "sudden_change_max_run": 1,
                    "state_action_min_da": 0.9,
                    "state_action_mean_da": 0.9,
                    "extreme_value_flagged_ratio": 0.0,
                }
            }
        ]

        with self.assertRaises(ValueError):
            suggest_numeric(
                rows,
                probe_alpha=0.1,
                s1_auto_reject_percentile=97.5,
                s1_review_percentile=99.0,
            )

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
