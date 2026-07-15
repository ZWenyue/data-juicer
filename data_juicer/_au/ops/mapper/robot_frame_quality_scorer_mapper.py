import json

import numpy as np

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader

# Relative import: custom_operator_paths loads this package as `_au`.
from ...utils.video_decode import load_video_frames

cv2 = LazyLoader("cv2", "opencv-contrib-python")

OP_NAME = "robot_frame_quality_scorer_mapper"


@OPERATORS.register_module(OP_NAME)
class RobotFrameQualityScorerMapper(Mapper):
    """Score each frame in a robot episode for visual quality defects.

    Detects black frames (low mean intensity), blurred frames (low Laplacian
    variance), and optionally corrupted frames (decode failure).  Writes a
    per-frame quality report into Fields.meta as a JSON string.

    Video decoding defaults to ``decoder='auto'`` (PyAV → OpenCV → ffmpeg CLI)
    so Galaxea / LeRobot AV1 demos work when OpenCV HW/soft AV1 paths fail.
    """

    def __init__(
        self,
        blackness_threshold: float = 10.0,
        blur_threshold: float = 50.0,
        corrupt_check_enabled: bool = True,
        video_field_index: int = 0,
        sampling_fps: float = None,
        original_fps: float = None,
        resize_for_scoring: list = None,
        decoder: str = "auto",
        report_field: str = "frame_quality_report",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.blackness_threshold = blackness_threshold
        self.blur_threshold = blur_threshold
        self.corrupt_check_enabled = corrupt_check_enabled
        self.video_field_index = video_field_index
        self.sampling_fps = sampling_fps
        self.original_fps = original_fps
        self.resize_for_scoring = (
            tuple(resize_for_scoring) if resize_for_scoring else None
        )
        self.decoder = decoder
        self.report_field = report_field

    def _load_frames_from_video(self, video_path):
        return load_video_frames(
            video_path,
            decoder=self.decoder,
            sampling_fps=self.sampling_fps,
            original_fps=self.original_fps,
        )

    def _score_frame(self, frame):
        if frame is None:
            return {
                "blackness": 0.0,
                "blur_laplacian_var": 0.0,
                "corrupt": True,
            }
        if self.resize_for_scoring is not None:
            frame = cv2.resize(
                frame, self.resize_for_scoring, interpolation=cv2.INTER_AREA
            )
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        blackness = float(np.mean(gray))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_var = float(laplacian.var())

        return {
            "blackness": blackness,
            "blur_laplacian_var": blur_var,
            "corrupt": False,
        }

    def process_single(self, sample):
        meta = sample.setdefault(Fields.meta, {})

        video_keys = sample.get(self.video_key)
        if not video_keys:
            meta[self.report_field] = json.dumps(
                {"error": "no_video_key", "num_frames": 0},
                ensure_ascii=False,
            )
            return sample

        if isinstance(video_keys, list):
            idx = min(self.video_field_index, len(video_keys) - 1)
            video_path = video_keys[idx]
        else:
            video_path = video_keys

        frames = self._load_frames_from_video(video_path)
        num_frames = len(frames)

        if num_frames == 0:
            meta[self.report_field] = json.dumps(
                {"error": "no_frames_loaded", "num_frames": 0},
                ensure_ascii=False,
            )
            return sample

        blackness_scores = []
        blur_scores = []
        corrupt_flags = []
        bad_frame_mask = []

        for frame in frames:
            scores = self._score_frame(frame)
            b = scores["blackness"]
            bl = scores["blur_laplacian_var"]
            c = scores["corrupt"]
            blackness_scores.append(b)
            blur_scores.append(bl)
            corrupt_flags.append(c)

            is_bad = (
                c or b < self.blackness_threshold or bl < self.blur_threshold
            )
            bad_frame_mask.append(is_bad)

        bad_indices = [i for i, v in enumerate(bad_frame_mask) if v]
        num_black = sum(
            1 for b in blackness_scores if b < self.blackness_threshold
        )
        num_blurred = sum(
            1 for bl in blur_scores if bl < self.blur_threshold
        )
        num_corrupt = sum(1 for c in corrupt_flags if c)

        report = {
            "num_frames": num_frames,
            "num_black": num_black,
            "num_blurred": num_blurred,
            "num_corrupt": num_corrupt,
            "bad_frame_indices": bad_indices,
            "bad_frame_mask": bad_frame_mask,
            "per_frame_scores": {
                "blackness": blackness_scores,
                "blur_laplacian_var": blur_scores,
                "corrupt": corrupt_flags,
            },
        }
        meta[self.report_field] = json.dumps(report, ensure_ascii=False)
        return sample
