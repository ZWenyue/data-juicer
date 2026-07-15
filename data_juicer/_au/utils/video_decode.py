# -*- coding: utf-8 -*-
"""Video frame loading with AV1-capable backends.

OpenCV builds on many machines fail to soft-decode AV1 (missing HW accel +
broken software path). PyAV (FFmpeg/libdav1d) and the system ``ffmpeg`` CLI
can decode Galaxea / LeRobot AV1 demos. Prefer PyAV; fall back to OpenCV then
optional ffmpeg pipe for full-sequence dumps.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from data_juicer.utils.lazy_loader import LazyLoader

cv2 = LazyLoader("cv2", "opencv-contrib-python")

# Prefer a normal import so we use the active venv's PyAV (AV1 via libdav1d)
# instead of LazyLoader trying to pip/uv-install into the wrong environment.
try:
    import av as _av_mod
except ImportError:  # pragma: no cover
    _av_mod = LazyLoader("av")


class _AvProxy:
    def __getattr__(self, name):
        return getattr(_av_mod, name)


av = _AvProxy()


def _sampling_step(fps: float, sampling_fps: Optional[float]) -> int:
    if sampling_fps and fps > 0 and sampling_fps < fps:
        return max(1, int(round(fps / sampling_fps)))
    return 1


def _load_via_pyav(
    video_path: str,
    sampling_fps: Optional[float] = None,
    original_fps: Optional[float] = None,
) -> List[np.ndarray]:
    try:
        av.logging.set_level(av.logging.ERROR)
    except Exception:
        pass

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if fps <= 0:
            fps = float(original_fps or 15.0)
        step = _sampling_step(fps, sampling_fps)
        frames: List[np.ndarray] = []
        for idx, frame in enumerate(container.decode(stream)):
            if idx % step != 0:
                continue
            frames.append(frame.to_ndarray(format="bgr24"))
        return frames
    finally:
        container.close()


def _load_via_opencv(
    video_path: str,
    sampling_fps: Optional[float] = None,
    original_fps: Optional[float] = None,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames: List[np.ndarray] = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or float(original_fps or 15.0)
        step = _sampling_step(fps, sampling_fps)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0 and frame is not None:
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    return frames


def _load_via_ffmpeg_cli(
    video_path: str,
    sampling_fps: Optional[float] = None,
    original_fps: Optional[float] = None,
) -> List[np.ndarray]:
    """Decode all (or fps-subsampled) frames via system ffmpeg → PNG sequence."""
    if not shutil.which("ffmpeg"):
        return []

    fps = float(original_fps or 15.0)
    # Prefer probed fps when available.
    if shutil.which("ffprobe"):
        try:
            raw = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=r_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=60,
            ).strip()
            if "/" in raw:
                num, den = raw.split("/", 1)
                if float(den) != 0:
                    fps = float(num) / float(den)
            elif raw:
                fps = float(raw)
        except Exception:
            pass

    vf_parts = []
    if sampling_fps and sampling_fps > 0 and sampling_fps < fps:
        vf_parts.append(f"fps={sampling_fps}")

    with tempfile.TemporaryDirectory(prefix="dj_vdec_") as tmp:
        pattern = str(Path(tmp) / "f_%06d.png")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
        ]
        if vf_parts:
            cmd.extend(["-vf", ",".join(vf_parts)])
        cmd.extend(["-vsync", "vfr", pattern])
        try:
            subprocess.run(cmd, check=True, timeout=600, capture_output=True)
        except Exception as e:
            logger.warning(f"ffmpeg CLI decode failed for {video_path}: {e}")
            return []

        paths = sorted(Path(tmp).glob("f_*.png"))
        frames: List[np.ndarray] = []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        return frames


def load_video_frames(
    video_path: str,
    decoder: str = "auto",
    sampling_fps: Optional[float] = None,
    original_fps: Optional[float] = None,
) -> List[np.ndarray]:
    """Load video frames as BGR ``uint8`` arrays.

    :param decoder: ``auto`` | ``pyav`` | ``opencv`` | ``ffmpeg``.
        ``auto`` tries PyAV first, then OpenCV, then ffmpeg CLI.
    """
    if not video_path or not Path(video_path).is_file():
        return []

    mode = (decoder or "auto").lower()
    order = {
        "auto": ("pyav", "opencv", "ffmpeg"),
        "pyav": ("pyav",),
        "opencv": ("opencv",),
        "ffmpeg": ("ffmpeg",),
    }.get(mode)
    if order is None:
        raise ValueError(
            f"Unknown decoder={decoder!r}; expected auto|pyav|opencv|ffmpeg"
        )

    loaders = {
        "pyav": _load_via_pyav,
        "opencv": _load_via_opencv,
        "ffmpeg": _load_via_ffmpeg_cli,
    }
    last_err = None
    for name in order:
        try:
            frames = loaders[name](video_path, sampling_fps, original_fps)
        except Exception as e:
            last_err = e
            logger.warning(f"video decode via {name} failed for {video_path}: {e}")
            continue
        if frames:
            if name != order[0]:
                logger.info(
                    f"Decoded {len(frames)} frames from {video_path} via {name}"
                )
            return frames

    if last_err is not None:
        logger.warning(f"All decoders failed for {video_path}: last={last_err}")
    return []
