"""Unit tests for RobotKeyFrameDetectorMapper."""

import json
import unittest

import numpy as np

from data_juicer._au.ops.mapper.robot_key_frame_detector_mapper import (
    RobotKeyFrameDetectorMapper,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

REPORT_FIELD = "key_frame_report"


def _make_sample(states, gripper=None):
    sample = {
        "states": states.tolist() if isinstance(states, np.ndarray) else states,
        Fields.meta: {},
    }
    if gripper is not None:
        sample["gripper"] = (
            gripper.tolist() if isinstance(gripper, np.ndarray) else gripper
        )
    return sample


class RobotKeyFrameDetectorMapperTest(DataJuicerTestCaseBase):

    def test_gripper_closure_detected(self):
        """A clear gripper closure event should be detected."""
        T = 50
        states = np.random.randn(T, 8) * 0.01
        gripper = np.ones(T) * 80.0
        gripper[20:30] = np.linspace(80, 10, 10)
        gripper[30:] = 10.0
        sample = _make_sample(states, gripper)

        op = RobotKeyFrameDetectorMapper(
            gripper_field="gripper",
            gripper_delta_threshold=5.0,
            gripper_close_direction="decrease",
            keyframe_window=2,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertGreater(report["num_gripper_events"], 0)
        self.assertTrue(len(report["protected_frame_indices"]) > 0)

    def test_gripper_dims_from_states(self):
        """Gripper events detected from state dimensions."""
        T = 50
        states = np.random.randn(T, 10) * 0.01
        states[:, 7] = 80.0
        states[15:25, 7] = np.linspace(80, 5, 10)
        states[25:, 7] = 5.0
        sample = _make_sample(states)

        op = RobotKeyFrameDetectorMapper(
            gripper_dims=[7],
            gripper_delta_threshold=5.0,
            gripper_close_direction="decrease",
            keyframe_window=2,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertGreater(report["num_gripper_events"], 0)

    def test_scale_relative_gripper_threshold_on_normalized_states(self):
        """Relative threshold should detect normalized packed grippers."""
        T = 50
        states = np.zeros((T, 16))
        states[:, [7, 15]] = 1.0
        states[20:, 7] = 0.0
        sample = _make_sample(states)

        op = RobotKeyFrameDetectorMapper(
            gripper_dims=[7, 15],
            gripper_delta_threshold=5.0,
            gripper_delta_threshold_frac=0.05,
            exempt_dims=[7, 15],
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertEqual(report["num_gripper_events"], 1)
        self.assertIn(19, report["gripper_event_frames"])

    def test_velocity_peaks_detected(self):
        """Velocity peaks (sudden acceleration) should be detected."""
        T = 100
        states = np.cumsum(np.random.randn(T, 6) * 0.01, axis=0)
        states[50:55] += np.random.randn(5, 6) * 5.0
        sample = _make_sample(states)

        op = RobotKeyFrameDetectorMapper(
            gripper_dims=None,
            state_velocity_percentile=90.0,
            keyframe_window=3,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertGreater(report["num_velocity_peaks"], 0)
        peak_frames = report["velocity_peak_frames"]
        self.assertTrue(any(45 <= f <= 60 for f in peak_frames))

    def test_protection_window_expansion(self):
        """Protection window should expand around keyframe indices."""
        T = 30
        states = np.random.randn(T, 4) * 0.01
        gripper = np.ones(T) * 80.0
        gripper[10] = 10.0
        sample = _make_sample(states, gripper)

        op = RobotKeyFrameDetectorMapper(
            gripper_field="gripper",
            gripper_delta_threshold=5.0,
            keyframe_window=3,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        protected = report["protected_frame_indices"]
        for idx in range(7, 13):
            self.assertIn(idx, protected)

    def test_short_trajectory_skipped(self):
        """Trajectories shorter than min_frames should be skipped."""
        states = np.random.randn(3, 4)
        sample = _make_sample(states)
        op = RobotKeyFrameDetectorMapper(
            min_frames=4, report_field=REPORT_FIELD
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertTrue(report.get("skipped", False))

    def test_no_events(self):
        """Smooth trajectory with no gripper changes should have no events."""
        T = 50
        states = np.cumsum(np.random.randn(T, 4) * 0.01, axis=0)
        gripper = np.ones(T) * 50.0 + np.random.randn(T) * 0.1
        sample = _make_sample(states, gripper)

        op = RobotKeyFrameDetectorMapper(
            gripper_field="gripper",
            gripper_delta_threshold=5.0,
            keyframe_window=3,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertEqual(report["num_gripper_events"], 0)

    def test_exempt_dims(self):
        """Exempt dims should be excluded from velocity peak calculation."""
        T = 50
        states = np.random.randn(T, 6) * 0.01
        states[:, 5] = np.linspace(0, 100, T)
        sample = _make_sample(states)

        op_with_exempt = RobotKeyFrameDetectorMapper(
            exempt_dims=[5],
            state_velocity_percentile=95.0,
            keyframe_window=2,
            report_field=REPORT_FIELD,
        )
        result = op_with_exempt.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        peaks_exempt = report["num_velocity_peaks"]

        op_without = RobotKeyFrameDetectorMapper(
            exempt_dims=[],
            state_velocity_percentile=95.0,
            keyframe_window=2,
            report_field=REPORT_FIELD,
        )
        sample2 = _make_sample(states)
        result2 = op_without.process_single(sample2)
        report2 = json.loads(result2[Fields.meta][REPORT_FIELD])
        peaks_no_exempt = report2["num_velocity_peaks"]

        self.assertLessEqual(peaks_exempt, peaks_no_exempt)


if __name__ == "__main__":
    unittest.main()
