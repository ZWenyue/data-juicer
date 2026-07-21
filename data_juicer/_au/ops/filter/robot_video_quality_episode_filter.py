import json
from typing import Optional

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields

OP_NAME = "robot_video_quality_episode_filter"


@OPERATORS.register_module(OP_NAME)
class RobotVideoQualityEpisodeFilter(Filter):
    """Episode-level video quality gate.

    Reads the per-frame quality report and keyframe report produced by
    upstream Mappers, then makes a binary keep/reject decision for the
    entire episode based on three criteria:

    1. Bad-frame ratio must not exceed ``max_bad_ratio``.
    2. Good-frame count must be at least ``min_good_frames``.
    3. When ``max_keyframe_overlap`` is configured, overlap between bad frames
       and protected keyframes must not exceed it.  The gate is disabled by
       default because a single overlap is not sufficient evidence that an
       otherwise healthy episode is unusable.

    If all three criteria are met the episode is kept *unchanged* (no
    frames are deleted).  Otherwise the entire episode is discarded.
    """

    def __init__(
        self,
        quality_report_field: str = "frame_quality_report",
        keyframe_report_field: str = "key_frame_report",
        max_bad_ratio: float = 0.1,
        min_good_frames: int = 20,
        max_keyframe_overlap: Optional[int] = None,
        report_field: str = "video_quality_episode_report",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.quality_report_field = quality_report_field
        self.keyframe_report_field = keyframe_report_field
        self.max_bad_ratio = max_bad_ratio
        self.min_good_frames = min_good_frames
        self.max_keyframe_overlap = max_keyframe_overlap
        self.report_field = report_field

    def _write_default_stats(self, sample, keep=True):
        sample[Fields.stats]["video_quality_episode_keep"] = keep
        sample[Fields.stats]["video_quality_episode_bad_ratio"] = 0.0
        sample[Fields.stats]["video_quality_episode_good_frames"] = 0
        sample[Fields.stats]["video_quality_episode_keyframe_overlap"] = 0
        return sample

    def compute_stats_single(self, sample, context=False):
        if Fields.stats not in sample or sample[Fields.stats] is None:
            sample[Fields.stats] = {}
        meta = sample.setdefault(Fields.meta, {})

        qr_str = meta.get(self.quality_report_field)
        if qr_str is None:
            self._write_default_stats(sample, keep=True)
            meta[self.report_field] = json.dumps(
                {"keep": True, "skipped": True, "reason": "no_quality_report"},
                ensure_ascii=False,
            )
            return sample

        qr = json.loads(qr_str) if isinstance(qr_str, str) else qr_str

        if "error" in qr:
            self._write_default_stats(sample, keep=True)
            meta[self.report_field] = json.dumps(
                {"keep": True, "skipped": True, "reason": "quality_report_error"},
                ensure_ascii=False,
            )
            return sample

        num_frames = qr.get("num_frames", 0)
        bad_frame_indices = set(qr.get("bad_frame_indices", []))
        num_bad = len(bad_frame_indices)
        num_good = num_frames - num_bad
        bad_ratio = num_bad / num_frames if num_frames > 0 else 0.0

        protected_indices = set()
        kr_str = meta.get(self.keyframe_report_field)
        if kr_str is not None:
            kr = json.loads(kr_str) if isinstance(kr_str, str) else kr_str
            if not kr.get("skipped", False):
                protected_indices = set(kr.get("protected_frame_indices", []))

        keyframe_overlap = len(bad_frame_indices & protected_indices)

        reject_reasons = []
        if bad_ratio > self.max_bad_ratio:
            reject_reasons.append("bad_ratio_exceeded")
        if num_good < self.min_good_frames:
            reject_reasons.append("too_few_good_frames")
        if (
            self.max_keyframe_overlap is not None
            and keyframe_overlap > self.max_keyframe_overlap
        ):
            reject_reasons.append("keyframe_contaminated")

        keep = len(reject_reasons) == 0

        sample[Fields.stats]["video_quality_episode_keep"] = bool(keep)
        sample[Fields.stats]["video_quality_episode_bad_ratio"] = float(bad_ratio)
        sample[Fields.stats]["video_quality_episode_good_frames"] = int(num_good)
        sample[Fields.stats]["video_quality_episode_keyframe_overlap"] = int(
            keyframe_overlap
        )

        report = {
            "keep": bool(keep),
            "bad_ratio": float(bad_ratio),
            "good_frames": int(num_good),
            "total_frames": int(num_frames),
            "num_bad": int(num_bad),
            "keyframe_overlap": int(keyframe_overlap),
            "reject_reasons": reject_reasons,
        }
        meta[self.report_field] = json.dumps(report, ensure_ascii=False)
        return sample

    def process_single(self, sample):
        return sample.get(Fields.stats, {}).get(
            "video_quality_episode_keep", True
        )
