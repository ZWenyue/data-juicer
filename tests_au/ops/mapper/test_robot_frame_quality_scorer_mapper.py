"""Unit tests for RobotFrameQualityScorerMapper."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from data_juicer._au.ops.mapper.robot_frame_quality_scorer_mapper import (
    RobotFrameQualityScorerMapper,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

REPORT_FIELD = "frame_quality_report"


def _make_video(frames, path, fps=15):
    """Write a list of BGR numpy frames to an mp4 file."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def _black_frame(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _white_frame(h=120, w=160):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _sharp_frame(h=120, w=160, seed=42):
    rng = np.random.RandomState(seed)
    frame = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (100, 80), (0, 255, 0), 2)
    cv2.putText(
        frame, "SHARP", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
    )
    return frame


def _blurry_frame(h=120, w=160, seed=42):
    frame = _sharp_frame(h, w, seed)
    return cv2.GaussianBlur(frame, (31, 31), 15)


class RobotFrameQualityScorerMapperTest(DataJuicerTestCaseBase):

    def _make_sample(self, frames, tmpdir):
        path = os.path.join(tmpdir, "test.mp4")
        _make_video(frames, path)
        return {
            "videos": [path],
            Fields.meta: {},
        }

    def test_all_normal_frames(self):
        """All sharp frames should produce no bad frame detections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [_sharp_frame() for _ in range(10)]
            sample = self._make_sample(frames, tmpdir)
            op = RobotFrameQualityScorerMapper(
                blackness_threshold=10.0,
                blur_threshold=50.0,
                report_field=REPORT_FIELD,
            )
            result = op.process_single(sample)
            report = json.loads(result[Fields.meta][REPORT_FIELD])
            self.assertEqual(report["num_frames"], 10)
            self.assertEqual(report["num_black"], 0)
            self.assertEqual(report["num_corrupt"], 0)
            self.assertFalse(any(report["bad_frame_mask"]))

    def test_black_frames_detected(self):
        """Black frames at the start should be flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [_black_frame()] * 3 + [_sharp_frame()] * 7
            sample = self._make_sample(frames, tmpdir)
            op = RobotFrameQualityScorerMapper(
                blackness_threshold=10.0,
                blur_threshold=50.0,
                report_field=REPORT_FIELD,
            )
            result = op.process_single(sample)
            report = json.loads(result[Fields.meta][REPORT_FIELD])
            self.assertGreaterEqual(report["num_black"], 3)
            for i in range(3):
                self.assertTrue(report["bad_frame_mask"][i])

    def test_blurry_frames_detected(self):
        """Very blurred frames should be flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [_sharp_frame()] * 3 + [_blurry_frame()] * 2 + [_sharp_frame()] * 3
            sample = self._make_sample(frames, tmpdir)
            op = RobotFrameQualityScorerMapper(
                blackness_threshold=10.0,
                blur_threshold=50.0,
                report_field=REPORT_FIELD,
            )
            result = op.process_single(sample)
            report = json.loads(result[Fields.meta][REPORT_FIELD])
            self.assertGreaterEqual(report["num_blurred"], 1)

    def test_mixed_defects(self):
        """Episode with both black and blurry frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = (
                [_black_frame()] * 2
                + [_sharp_frame()] * 4
                + [_blurry_frame()] * 2
                + [_sharp_frame()] * 2
            )
            sample = self._make_sample(frames, tmpdir)
            op = RobotFrameQualityScorerMapper(
                blackness_threshold=10.0,
                blur_threshold=50.0,
                report_field=REPORT_FIELD,
            )
            result = op.process_single(sample)
            report = json.loads(result[Fields.meta][REPORT_FIELD])
            self.assertEqual(report["num_frames"], 10)
            self.assertGreaterEqual(report["num_black"], 2)
            self.assertTrue(len(report["bad_frame_indices"]) >= 2)

    def test_no_video_key(self):
        """Sample without video should produce error report."""
        sample = {Fields.meta: {}}
        op = RobotFrameQualityScorerMapper(report_field=REPORT_FIELD)
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertEqual(report["num_frames"], 0)
        self.assertIn("error", report)

    def test_per_frame_scores_present(self):
        """Output should include per-frame blackness and blur scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [_sharp_frame()] * 5
            sample = self._make_sample(frames, tmpdir)
            op = RobotFrameQualityScorerMapper(report_field=REPORT_FIELD)
            result = op.process_single(sample)
            report = json.loads(result[Fields.meta][REPORT_FIELD])
            self.assertIn("per_frame_scores", report)
            scores = report["per_frame_scores"]
            self.assertEqual(len(scores["blackness"]), 5)
            self.assertEqual(len(scores["blur_laplacian_var"]), 5)
            self.assertEqual(len(scores["corrupt"]), 5)

    def test_pyav_decoder_on_mp4v(self):
        """Explicit pyav backend must still score synthetic mp4v clips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [_sharp_frame()] * 6
            sample = self._make_sample(frames, tmpdir)
            op = RobotFrameQualityScorerMapper(
                decoder="pyav",
                blackness_threshold=10.0,
                blur_threshold=50.0,
                report_field=REPORT_FIELD,
            )
            result = op.process_single(sample)
            report = json.loads(result[Fields.meta][REPORT_FIELD])
            self.assertNotIn("error", report)
            self.assertEqual(report["num_frames"], 6)

    def test_galaxea_av1_decodes_with_auto(self):
        """Real Galaxea AV1 should load frames via PyAV (OpenCV alone fails)."""
        av1 = (
            "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/"
            "Connect_Router_Cables_20250625_002/videos/chunk-000/"
            "observation.images.head_rgb/episode_000000.mp4"
        )
        if not os.path.isfile(av1):
            self.skipTest(f"AV1 sample not found: {av1}")
        sample = {"videos": [av1], Fields.meta: {}}
        op = RobotFrameQualityScorerMapper(
            decoder="auto",
            # Speed: subsample ~1 fps for this acceptance-like check
            sampling_fps=1.0,
            original_fps=15.0,
            resize_for_scoring=[160, 90],
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertNotIn("error", report, msg=report)
        self.assertGreater(report["num_frames"], 0)
        self.assertIn("bad_frame_indices", report)


if __name__ == "__main__":
    unittest.main()
