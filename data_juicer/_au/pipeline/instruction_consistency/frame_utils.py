"""Frame extraction and SDKImage wrapping for Cursor SDK.

Galaxea / LeRobot demos are often AV1. Many OpenCV builds fail with:
  "Your platform doesn't support hardware accelerated AV1 decoding"
even when system ffmpeg can soft-decode. This module tries OpenCV first,
then falls back to ffmpeg CLI.
"""
from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def _sample_indices(start_frame: int, end_frame: int, num_frames: int) -> np.ndarray:
    if end_frame <= start_frame:
        return np.array([], dtype=int)
    n = max(1, int(num_frames))
    return np.unique(np.linspace(start_frame, end_frame - 1, num=n, dtype=int))


def _ffprobe_stream_field(video_path: str, field: str) -> str:
    if not shutil.which("ffprobe"):
        return ""
    try:
        return subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                f"stream={field}",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=60,
        ).strip()
    except Exception:
        return ""


def _probe_codec(video_path: str) -> str:
    return (_ffprobe_stream_field(video_path, "codec_name") or "").lower()


def _probe_nb_frames(video_path: str) -> int:
    """Best-effort frame count via ffprobe, else OpenCV metadata."""
    for field in ("nb_frames", "nb_read_packets"):
        raw = _ffprobe_stream_field(video_path, field)
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    # count_packets is slower but reliable when nb_frames is N/A
    if shutil.which("ffprobe"):
        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_packets",
                    "-show_entries",
                    "stream=nb_read_packets",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=120,
            ).strip()
            if out.isdigit() and int(out) > 0:
                return int(out)
        except Exception as e:
            logger.debug(f"ffprobe count_packets failed: {e}")

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    return 0


def _bgr_to_jpeg_b64(frame_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf).decode("utf-8")


def _extract_via_opencv(video_path: str, indices: np.ndarray) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frames.append({"data": _bgr_to_jpeg_b64(frame), "mime_type": "image/jpeg"})
    cap.release()
    return frames


def _extract_via_ffmpeg(video_path: str, indices: np.ndarray) -> list[dict]:
    """Decode selected frames with system ffmpeg (AV1 soft-decode capable)."""
    if not shutil.which("ffmpeg") or len(indices) == 0:
        return []

    # One ffmpeg filter pass: keep only needed frame numbers (in display order).
    uniq = np.unique(indices.astype(int))
    select_expr = "+".join(f"eq(n\\,{int(i)})" for i in uniq)

    with tempfile.TemporaryDirectory(prefix="dj_ic_frames_") as tmp:
        # %04d starts at 1 for the first selected frame.
        pattern = str(Path(tmp) / "f_%04d.jpg")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            f"select='{select_expr}'",
            "-vsync",
            "vfr",
            "-q:v",
            "3",
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        except Exception as e:
            logger.warning(f"ffmpeg frame extract failed for {video_path}: {e}")
            return []

        paths = sorted(Path(tmp).glob("f_*.jpg"))
        if not paths:
            logger.warning(f"ffmpeg produced 0 frames for {video_path}")
            return []

        # Map selected order back: ffmpeg emits in ascending n order == uniq order.
        out = []
        for p in paths:
            raw = p.read_bytes()
            out.append({"data": base64.b64encode(raw).decode("utf-8"), "mime_type": "image/jpeg"})
        return out


def extract_frames_base64(
    video_path: str,
    start_frame: int,
    end_frame: int,
    num_frames: int = 8,
) -> list[dict]:
    """Extract uniformly-sampled frames from a video segment.

    Returns list of {"data": base64_str, "mime_type": "image/jpeg"}.
    Prefers OpenCV; on empty read (typical AV1 + broken OpenCV), uses ffmpeg.
    """
    if not video_path or not Path(video_path).is_file():
        return []

    total = _probe_nb_frames(video_path)
    end = min(int(end_frame), total) if total > 0 else int(end_frame)
    start = max(0, int(start_frame))
    indices = _sample_indices(start, end if total > 0 else end, num_frames)
    if len(indices) == 0:
        return []

    codec = _probe_codec(video_path)
    # OpenCV/FFmpeg builds on this host often cannot decode AV1 → skip straight to CLI ffmpeg.
    if codec in ("av1", "av01"):
        logger.debug(f"codec={codec}; using ffmpeg for {video_path}")
        return _extract_via_ffmpeg(video_path, indices)

    frames = _extract_via_opencv(video_path, indices)
    if len(frames) >= max(1, len(indices) // 2):
        return frames

    logger.warning(
        f"OpenCV got {len(frames)}/{len(indices)} frames (codec={codec or 'unknown'}) "
        f"from {video_path}; falling back to ffmpeg."
    )
    ff = _extract_via_ffmpeg(video_path, indices)
    return ff if ff else frames


def frames_to_sdk_images(frames: list[dict]):
    """Convert base64 frame dicts to Cursor SDKImage objects.

    Falls back to raw dicts if cursor_sdk is not installed (for testing).
    """
    try:
        from cursor_sdk import SDKImage

        return [SDKImage(data=f["data"], mime_type=f["mime_type"]) for f in frames]
    except ImportError:
        return frames
