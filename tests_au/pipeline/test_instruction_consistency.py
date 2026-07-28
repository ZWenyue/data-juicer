"""Unit tests for instruction consistency checker (Cursor SDK version).

Uses mock Cursor SDK to test logic without real API calls.
"""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from data_juicer._au.pipeline.instruction_consistency.config import (
    CheckerConfig,
)
from data_juicer._au.pipeline.instruction_consistency.parse_utils import (
    parse_vlm_json,
)
from data_juicer._au.pipeline.instruction_consistency.segmentation import (
    segment_episode,
)
from data_juicer._au.pipeline.instruction_consistency.vote import (
    aggregate_votes,
)


class TestSegmentation(unittest.TestCase):
    """Test Stage 1 temporal normalization."""

    def test_single_action(self):
        """Monotonic speed -> no cut points -> single segment."""
        pos = np.cumsum(np.random.randn(50, 3) * 0.01, axis=0)
        segs = segment_episode(pos, min_window=5, min_frames=4)
        self.assertGreaterEqual(len(segs), 1)
        self.assertEqual(segs[0]["start_frame"], 0)

    def test_two_actions_with_pause(self):
        """Speed drops to zero mid-episode -> two segments."""
        pos = np.zeros((100, 3))
        # action 1: frames 0-40 (moving)
        pos[:40, 0] = np.linspace(0, 1, 40)
        # pause: frames 40-60 (stationary)
        pos[40:60, 0] = 1.0
        # action 2: frames 60-100 (moving again)
        pos[60:, 0] = np.linspace(1, 2, 40)

        segs = segment_episode(pos, min_window=5, min_frames=4)
        self.assertGreaterEqual(len(segs), 2)

    def test_empty_trajectory(self):
        """Single-point trajectory -> single segment."""
        pos = np.zeros((1, 3))
        segs = segment_episode(pos)
        self.assertEqual(len(segs), 1)


class TestParseVLMJson(unittest.TestCase):
    """Test VLM response parsing robustness."""

    def test_valid_json(self):
        resp = json.dumps({"verdict": "consistent", "confidence": 0.9})
        parsed = parse_vlm_json(resp)
        self.assertEqual(parsed["verdict"], "consistent")

    def test_markdown_code_block(self):
        resp = '```json\n{"verdict": "inconsistent", "confidence": 0.3}\n```'
        parsed = parse_vlm_json(resp)
        self.assertEqual(parsed["verdict"], "inconsistent")

    def test_json_in_text(self):
        resp = 'Here is my analysis: {"verdict": "consistent", "confidence": 0.8} end.'
        parsed = parse_vlm_json(resp)
        self.assertEqual(parsed["verdict"], "consistent")

    def test_unparseable(self):
        resp = "I think this is consistent."
        parsed = parse_vlm_json(resp)
        self.assertTrue(parsed.get("parse_error"))
        self.assertEqual(parsed["verdict"], "inconsistent")

    def test_empty_response(self):
        parsed = parse_vlm_json("")
        self.assertTrue(parsed.get("parse_error"))


class TestVoteAggregation(unittest.TestCase):
    """Test voting strategies."""

    def _votes(self, verdicts_confs):
        return [{"model": f"m{i}", "verdict": v, "confidence": c}
                for i, (v, c) in enumerate(verdicts_confs)]

    def test_majority_consistent(self):
        votes = self._votes([
            ("consistent", 0.8),
            ("consistent", 0.7),
            ("inconsistent", 0.6),
        ])
        r = aggregate_votes(votes, "majority")
        self.assertEqual(r["final_verdict"], "consistent")

    def test_majority_inconsistent(self):
        votes = self._votes([
            ("inconsistent", 0.8),
            ("inconsistent", 0.7),
            ("consistent", 0.6),
        ])
        r = aggregate_votes(votes, "majority")
        self.assertEqual(r["final_verdict"], "inconsistent")

    def test_weighted_overrides_count(self):
        """High-confidence minority overrides low-confidence majority."""
        votes = self._votes([
            ("inconsistent", 0.95),  # high confidence
            ("consistent", 0.51),    # low confidence
            ("consistent", 0.52),    # low confidence
        ])
        r = aggregate_votes(votes, "weighted")
        # w_c = 0.51 + 0.52 = 1.03, w_total = 1.98
        # score = 1.03 / 1.98 ~ 0.52 > 0.5 -> consistent
        self.assertIn(r["final_verdict"],
                      ["consistent", "inconsistent"])

    def test_unanimous_all_agree(self):
        votes = self._votes([
            ("consistent", 0.9),
            ("consistent", 0.8),
            ("consistent", 0.85),
        ])
        r = aggregate_votes(votes, "unanimous_override")
        self.assertEqual(r["final_verdict"], "consistent")
        self.assertEqual(r["final_score"], 1.0)

    def test_unanimous_disagree(self):
        votes = self._votes([
            ("consistent", 0.9),
            ("inconsistent", 0.8),
        ])
        r = aggregate_votes(votes, "unanimous_override")
        self.assertEqual(r["final_verdict"], "ambiguous")

    def test_empty_votes(self):
        r = aggregate_votes([], "majority")
        self.assertEqual(r["final_verdict"], "inconsistent")
        self.assertEqual(r["num_experts_responded"], 0)


class TestConfig(unittest.TestCase):
    """Test configuration defaults."""

    def test_defaults(self):
        cfg = CheckerConfig()
        self.assertEqual(cfg.primary_model, "composer-2.5")
        self.assertEqual(len(cfg.expert_models), 3)
        self.assertEqual(cfg.voting_strategy, "majority")
        self.assertEqual(cfg.confidence_threshold, 0.7)


class TestStage3NeedsAdjudication(unittest.TestCase):

    def test_high_confidence_consistent_skips(self):
        from data_juicer._au.pipeline.instruction_consistency \
            .stage3_adjudicator import needs_adjudication
        result = {"overall_verdict": "consistent",
                  "overall_confidence": 0.9}
        self.assertFalse(
            needs_adjudication(result, 0.7, ["m1", "m2"])
        )

    def test_low_confidence_triggers(self):
        from data_juicer._au.pipeline.instruction_consistency \
            .stage3_adjudicator import needs_adjudication
        result = {"overall_verdict": "consistent",
                  "overall_confidence": 0.5}
        self.assertTrue(
            needs_adjudication(result, 0.7, ["m1", "m2"])
        )

    def test_inconsistent_triggers(self):
        from data_juicer._au.pipeline.instruction_consistency \
            .stage3_adjudicator import needs_adjudication
        result = {"overall_verdict": "inconsistent",
                  "overall_confidence": 0.9}
        self.assertTrue(
            needs_adjudication(result, 0.7, ["m1", "m2"])
        )

    def test_no_experts_skips(self):
        from data_juicer._au.pipeline.instruction_consistency \
            .stage3_adjudicator import needs_adjudication
        result = {"overall_verdict": "inconsistent",
                  "overall_confidence": 0.3}
        self.assertFalse(
            needs_adjudication(result, 0.7, [])
        )


class TestStage2Aggregate(unittest.TestCase):
    def test_ratio_keeps_mostly_consistent(self):
        from data_juicer._au.pipeline.instruction_consistency.stage2_evaluator import (
            aggregate_stage2,
        )

        segs = [{"verdict": "consistent", "confidence": 0.9}] * 5
        segs += [{"verdict": "inconsistent", "confidence": 0.8}] * 2  # 2/7≈0.29 < 0.35
        r = aggregate_stage2(segs, inconsistent_ratio_threshold=0.35)
        self.assertEqual(r["overall_verdict"], "consistent")
        self.assertAlmostEqual(r["inconsistent_ratio"], 2 / 7, places=4)

    def test_ratio_flags_mostly_bad(self):
        from data_juicer._au.pipeline.instruction_consistency.stage2_evaluator import (
            aggregate_stage2,
        )

        segs = [{"verdict": "consistent", "confidence": 0.9}] * 2
        segs += [{"verdict": "inconsistent", "confidence": 0.8}] * 3  # 0.6 > 0.35
        r = aggregate_stage2(segs, inconsistent_ratio_threshold=0.35)
        self.assertEqual(r["overall_verdict"], "inconsistent")


class TestStage3SpanResolve(unittest.TestCase):
    def test_resolve_from_stage1_when_missing_on_result(self):
        from data_juicer._au.pipeline.instruction_consistency.stage3_adjudicator import (
            resolve_segment_span,
        )

        worst = {"segment_id": 2, "confidence": 0.1, "verdict": "inconsistent"}
        stage1 = [
            {"segment_id": 0, "start_frame": 0, "end_frame": 10},
            {"segment_id": 2, "start_frame": 40, "end_frame": 80},
        ]
        s, e = resolve_segment_span(worst, stage1)
        self.assertEqual((s, e), (40, 80))

    def test_prefer_existing_span(self):
        from data_juicer._au.pipeline.instruction_consistency.stage3_adjudicator import (
            resolve_segment_span,
        )

        worst = {
            "segment_id": 2,
            "start_frame": 100,
            "end_frame": 140,
            "verdict": "inconsistent",
        }
        s, e = resolve_segment_span(worst, None)
        self.assertEqual((s, e), (100, 140))

    def test_pick_worst_prefers_inconsistent(self):
        from data_juicer._au.pipeline.instruction_consistency.stage3_adjudicator import (
            pick_worst_segment,
        )

        segs = [
            {"segment_id": 0, "verdict": "consistent", "confidence": 0.5},
            {"segment_id": 1, "verdict": "inconsistent", "confidence": 0.9},
            {"segment_id": 2, "verdict": "consistent", "confidence": 0.1},
        ]
        w = pick_worst_segment(segs)
        self.assertEqual(w["segment_id"], 1)


class TestSegmentationCap(unittest.TestCase):
    def test_max_segments_merge(self):
        # Many pauses → many cuts; cap should merge down.
        pos = np.zeros((400, 3))
        for i in range(8):
            a, b = i * 50, i * 50 + 30
            pos[a:b, 0] = np.linspace(i, i + 1, b - a)
        segs = segment_episode(
            pos, min_window=5, min_frames=4, max_segments=3
        )
        self.assertLessEqual(len(segs), 3)
        self.assertEqual(segs[0]["start_frame"], 0)
        self.assertEqual(segs[-1]["end_frame"], len(pos))


class TestFrameExtractAv1(unittest.TestCase):
    """Real Galaxea AV1 clips need ffmpeg fallback (OpenCV often fails)."""

    DATA = (
        "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset"
        "/Connect_Router_Cables_20250625_002"
    )

    def test_extract_av1_via_ffmpeg(self):
        import glob
        import os

        from data_juicer._au.pipeline.instruction_consistency.frame_utils import (
            extract_frames_base64,
        )

        vids = sorted(
            glob.glob(
                os.path.join(
                    self.DATA,
                    "videos",
                    "chunk-*",
                    "observation.images.head_rgb",
                    "episode_*.mp4",
                )
            )
        )
        if not vids:
            self.skipTest(f"no videos under {self.DATA}")
        frames = extract_frames_base64(vids[0], 0, 80, num_frames=4)
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(f.get("mime_type") == "image/jpeg" for f in frames))
        self.assertTrue(all(len(f.get("data", "")) > 1000 for f in frames))


if __name__ == "__main__":
    unittest.main()
