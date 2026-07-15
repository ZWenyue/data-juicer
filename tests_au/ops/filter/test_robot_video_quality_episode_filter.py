"""Unit tests for RobotVideoQualityEpisodeFilter."""

import json
import unittest

from data_juicer._au.ops.filter.robot_video_quality_episode_filter import (
    RobotVideoQualityEpisodeFilter,
)
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

QR_FIELD = "frame_quality_report"
KR_FIELD = "key_frame_report"
REPORT_FIELD = "video_quality_episode_report"
STATS_KEYS = [
    "video_quality_episode_keep",
    "video_quality_episode_bad_ratio",
    "video_quality_episode_good_frames",
    "video_quality_episode_keyframe_overlap",
]


def _make_quality_report(num_frames, bad_indices):
    return json.dumps({
        "num_frames": num_frames,
        "bad_frame_indices": bad_indices,
        "bad_frame_mask": [i in bad_indices for i in range(num_frames)],
        "num_black": len(bad_indices),
        "num_blurred": 0,
        "num_corrupt": 0,
    })


def _make_keyframe_report(protected_indices, skipped=False):
    return json.dumps({
        "protected_frame_indices": protected_indices,
        "skipped": skipped,
    })


def _make_sample(num_frames, bad_indices, protected_indices=None):
    meta = {
        QR_FIELD: _make_quality_report(num_frames, bad_indices),
    }
    if protected_indices is not None:
        meta[KR_FIELD] = _make_keyframe_report(protected_indices)
    return {
        "id": "ep_test",
        Fields.stats: {},
        Fields.meta: meta,
    }


class RobotVideoQualityEpisodeFilterTest(DataJuicerTestCaseBase):

    def _run(self, op, sample):
        sample = op.compute_stats_single(sample)
        keep = op.process_single(sample)
        return sample, keep

    def test_healthy_episode_kept(self):
        """Episode with few bad frames should be kept."""
        sample = _make_sample(
            num_frames=100, bad_indices=[3, 7], protected_indices=[50, 60]
        )
        op = RobotVideoQualityEpisodeFilter(
            max_bad_ratio=0.1,
            min_good_frames=20,
            max_keyframe_overlap=0,
        )
        result, keep = self._run(op, sample)
        self.assertTrue(keep)
        self.assertTrue(result[Fields.stats]["video_quality_episode_keep"])
        for k in STATS_KEYS:
            self.assertIn(k, result[Fields.stats])

    def test_bad_ratio_exceeded(self):
        """Episode with too many bad frames should be rejected."""
        bad = list(range(15))
        sample = _make_sample(
            num_frames=100, bad_indices=bad, protected_indices=[50]
        )
        op = RobotVideoQualityEpisodeFilter(
            max_bad_ratio=0.1,
            min_good_frames=20,
            max_keyframe_overlap=0,
        )
        result, keep = self._run(op, sample)
        self.assertFalse(keep)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertIn("bad_ratio_exceeded", report["reject_reasons"])

    def test_too_few_good_frames(self):
        """Episode with too few good frames should be rejected."""
        bad = list(range(90))
        sample = _make_sample(
            num_frames=100, bad_indices=bad, protected_indices=[]
        )
        op = RobotVideoQualityEpisodeFilter(
            max_bad_ratio=1.0,
            min_good_frames=20,
            max_keyframe_overlap=999,
        )
        result, keep = self._run(op, sample)
        self.assertFalse(keep)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertIn("too_few_good_frames", report["reject_reasons"])

    def test_keyframe_contaminated(self):
        """Bad frame overlapping a protected keyframe should reject."""
        sample = _make_sample(
            num_frames=100, bad_indices=[10, 20], protected_indices=[10, 50]
        )
        op = RobotVideoQualityEpisodeFilter(
            max_bad_ratio=0.5,
            min_good_frames=5,
            max_keyframe_overlap=0,
        )
        result, keep = self._run(op, sample)
        self.assertFalse(keep)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertIn("keyframe_contaminated", report["reject_reasons"])
        self.assertEqual(
            result[Fields.stats]["video_quality_episode_keyframe_overlap"], 1
        )

    def test_multiple_rejection_reasons(self):
        """Multiple conditions can trigger simultaneously."""
        bad = list(range(50))
        sample = _make_sample(
            num_frames=100, bad_indices=bad, protected_indices=[5, 10, 15]
        )
        op = RobotVideoQualityEpisodeFilter(
            max_bad_ratio=0.1,
            min_good_frames=60,
            max_keyframe_overlap=0,
        )
        result, keep = self._run(op, sample)
        self.assertFalse(keep)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertGreaterEqual(len(report["reject_reasons"]), 2)
        self.assertIn("bad_ratio_exceeded", report["reject_reasons"])
        self.assertIn("too_few_good_frames", report["reject_reasons"])
        self.assertIn("keyframe_contaminated", report["reject_reasons"])

    def test_no_quality_report_defaults_keep(self):
        """Missing upstream quality report should default to keep."""
        sample = {
            "id": "ep_no_qr",
            Fields.stats: {},
            Fields.meta: {},
        }
        op = RobotVideoQualityEpisodeFilter()
        result, keep = self._run(op, sample)
        self.assertTrue(keep)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertTrue(report["skipped"])

    def test_no_keyframe_report(self):
        """Missing keyframe report should skip keyframe overlap check."""
        sample = _make_sample(
            num_frames=100, bad_indices=[3], protected_indices=None
        )
        op = RobotVideoQualityEpisodeFilter(
            max_bad_ratio=0.1,
            max_keyframe_overlap=0,
        )
        result, keep = self._run(op, sample)
        self.assertTrue(keep)
        self.assertEqual(
            result[Fields.stats]["video_quality_episode_keyframe_overlap"], 0
        )

    def test_pipeline_integration(self):
        """Filter should work through the DJ pipeline."""
        ds_list = []

        ds_list.append(_make_sample(100, [1, 2], [50]))
        ds_list[-1]["id"] = "ep_good"

        ds_list.append(_make_sample(100, list(range(20)), [5, 10]))
        ds_list[-1]["id"] = "ep_bad_ratio"

        ds_list.append(_make_sample(100, [], []))
        ds_list[-1]["id"] = "ep_perfect"

        dataset = Dataset.from_list(ds_list)
        op = RobotVideoQualityEpisodeFilter(
            max_bad_ratio=0.1,
            min_good_frames=20,
            max_keyframe_overlap=0,
        )
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        result = dataset.to_list()
        kept_ids = [r["id"] for r in result]
        self.assertIn("ep_good", kept_ids)
        self.assertNotIn("ep_bad_ratio", kept_ids)
        self.assertIn("ep_perfect", kept_ids)


if __name__ == "__main__":
    unittest.main()
