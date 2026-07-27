# -*- coding: utf-8 -*-
"""Hand mask generation and robot/frame compositing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass
class DepthAligner:
    """Affine map scene depth → metric meters: ``scale * d + bias``.

    MoGe-2 is often already near-metric; wrist MANO z is used to fit residual
    scale/bias per clip when both are available.
    """

    scale: float = 1.0
    bias: float = 0.0

    def __call__(self, scene_depth: np.ndarray) -> np.ndarray:
        d = np.asarray(scene_depth, dtype=np.float32)
        return (self.scale * d + self.bias).astype(np.float32)


def fit_depth_aligner(
    scene_depths: Sequence[float],
    metric_depths_m: Sequence[float],
    min_pairs: int = 4,
    max_scale: float = 5.0,
) -> DepthAligner:
    """Least-squares affine fit ``metric ≈ scale * scene + bias``.

    Falls back to identity when too few pairs or the fit is unstable.
    """
    s = np.asarray(scene_depths, dtype=np.float64).reshape(-1)
    m = np.asarray(metric_depths_m, dtype=np.float64).reshape(-1)
    ok = np.isfinite(s) & np.isfinite(m) & (s > 1e-4) & (m > 1e-4)
    s, m = s[ok], m[ok]
    if s.size < min_pairs:
        return DepthAligner(1.0, 0.0)
    # Robust: median ratio as scale seed, then 1-D LS with bias.
    A = np.stack([s, np.ones_like(s)], axis=1)
    try:
        coef, _, _, _ = np.linalg.lstsq(A, m, rcond=None)
        scale, bias = float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:
        return DepthAligner(1.0, 0.0)
    if not np.isfinite(scale) or not np.isfinite(bias) or abs(scale) < 1e-3 or abs(scale) > max_scale:
        # Degenerate → median ratio, zero bias.
        ratio = np.median(m / s)
        if not np.isfinite(ratio) or abs(ratio) < 1e-3 or abs(ratio) > max_scale:
            return DepthAligner(1.0, 0.0)
        return DepthAligner(float(ratio), 0.0)
    return DepthAligner(scale, bias)


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize for depth maps (preserves discontinuities)."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth must be HxW, got {d.shape}")
    if d.shape == (height, width):
        return d
    return cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST)


def bbox_to_mask(bbox: Sequence[float], img_shape: Tuple[int, ...], expand_ratio: float = 1.3) -> np.ndarray:
    """Expand a detection box into an elliptical boolean mask."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1) * expand_ratio, (y2 - y1) * expand_ratio
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (max(int(w * 0.5), 1), max(int(h * 0.5), 1)), 0, 0, 360, 255, -1)
    return mask > 0


def project_joints_mask(
    joints_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    dilate_ksize: int = 15,
) -> np.ndarray:
    """Project MANO joints to image and fill convex hull (+ dilation)."""
    joints = np.asarray(joints_cam, dtype=np.float64)
    z = np.clip(joints[:, 2], 1e-4, None)
    u = fx * joints[:, 0] / z + cx
    v = fy * joints[:, 1] / z + cy
    points = np.stack([u, v], axis=-1).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, img_w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, img_h - 1)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        for p in points:
            cv2.circle(mask, tuple(p), 8, 255, -1)

    if dilate_ksize > 0:
        k = dilate_ksize if dilate_ksize % 2 == 1 else dilate_ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
    return mask > 0


def alpha_blend(background: np.ndarray, foreground: np.ndarray, mask: np.ndarray, edge_blur: int = 3) -> np.ndarray:
    alpha = mask.astype(np.float32)
    if edge_blur > 0:
        k = edge_blur * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha_3 = np.stack([alpha] * 3, axis=-1)
    out = alpha_3 * foreground.astype(np.float32) + (1.0 - alpha_3) * background.astype(np.float32)
    return out.astype(np.uint8)


def _inpaint_hand(
    video_frame: np.ndarray,
    hand_mask: Optional[np.ndarray],
    inpaint_method: str = "telea",
) -> np.ndarray:
    result = video_frame.copy()
    if hand_mask is None or not np.any(hand_mask):
        return result
    inpaint_mask = hand_mask.astype(np.uint8) * 255
    flag = cv2.INPAINT_TELEA if inpaint_method != "ns" else cv2.INPAINT_NS
    return cv2.inpaint(result, inpaint_mask, 5, flag)


def composite_with_depth(
    video_frame: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    robot_depth_m: np.ndarray,
    scene_depth: np.ndarray,
    depth_aligner: Optional[Union[DepthAligner, Callable[[np.ndarray], np.ndarray]]] = None,
    hand_mask: Optional[np.ndarray] = None,
    epsilon_m: float = 0.02,
    max_invalid_ratio: float = 0.2,
    edge_blur: int = 3,
    inpaint_method: str = "telea",
) -> Tuple[np.ndarray, dict]:
    """P2 depth-aware composite.

    Robot is drawn only where ``robot_depth <= aligned_scene_depth + epsilon``.
    Returns ``(image, meta)`` with ``ok`` / ``depth_invalid_ratio`` / ``quality_flag``.
    """
    inpainted = _inpaint_hand(video_frame, hand_mask, inpaint_method=inpaint_method)
    h, w = inpainted.shape[:2]
    robot_d = resize_depth(robot_depth_m, h, w)
    scene_raw = resize_depth(scene_depth, h, w)
    aligner = depth_aligner if depth_aligner is not None else DepthAligner()
    scene_m = np.asarray(aligner(scene_raw), dtype=np.float32)

    rmask = np.asarray(robot_mask, dtype=bool)
    if rmask.shape != (h, w):
        rmask = cv2.resize(rmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    valid = (
        rmask
        & np.isfinite(robot_d)
        & np.isfinite(scene_m)
        & (robot_d > 0)
        & (scene_m > 0)
    )
    n_robot = int(rmask.sum())
    invalid_ratio = 1.0 - (float(valid.sum()) / max(n_robot, 1))
    meta = {
        "ok": True,
        "quality_flag": "ok",
        "depth_invalid_ratio": float(invalid_ratio),
        "visible_pixel_count": 0,
        "robot_pixel_count": n_robot,
    }
    if n_robot == 0:
        meta["quality_flag"] = "no_robot_mask"
        return inpainted, meta
    if invalid_ratio > max_invalid_ratio:
        meta["ok"] = False
        meta["quality_flag"] = "depth_invalid"
        return inpainted, meta

    visible = valid & (robot_d <= scene_m + float(epsilon_m))
    meta["visible_pixel_count"] = int(visible.sum())
    # Resize robot RGB if needed.
    fg = robot_rgb
    if fg.shape[:2] != (h, w):
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    return alpha_blend(inpainted, fg, visible, edge_blur=edge_blur), meta


def merge_render_layers(
    rgbs: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    depths: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Merge multiple robot renders into one foreground layer.

    When depth is available, the nearest robot pixel wins at each location.
    Otherwise later layers overwrite earlier ones inside their masks.
    """
    if not rgbs or not masks or len(rgbs) != len(masks):
        raise ValueError("rgbs/masks must be non-empty and have the same length")

    h, w = rgbs[0].shape[:2]
    merged_rgb = np.zeros_like(rgbs[0])
    merged_mask = np.zeros((h, w), dtype=bool)

    use_depth = depths is not None and len(depths) == len(rgbs) and any(d is not None for d in depths)
    merged_depth = np.full((h, w), np.inf, dtype=np.float32) if use_depth else None

    for i, (rgb, mask) in enumerate(zip(rgbs, masks)):
        if rgb.shape[:2] != (h, w):
            raise ValueError("all rgbs must share the same resolution")
        cur_mask = np.asarray(mask, dtype=bool)
        if cur_mask.shape != (h, w):
            cur_mask = cv2.resize(cur_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not np.any(cur_mask):
            continue

        if use_depth and depths is not None and depths[i] is not None:
            cur_depth = resize_depth(np.asarray(depths[i], dtype=np.float32), h, w)
            replace = cur_mask & np.isfinite(cur_depth) & (cur_depth > 0) & (cur_depth < merged_depth)
            if np.any(replace):
                merged_rgb[replace] = rgb[replace]
                merged_depth[replace] = cur_depth[replace]
                merged_mask[replace] = True
            merged_mask |= cur_mask
        else:
            merged_rgb[cur_mask] = rgb[cur_mask]
            merged_mask |= cur_mask

    if merged_depth is not None:
        finite = np.isfinite(merged_depth)
        merged_depth[~finite] = 0.0
    return merged_rgb, merged_mask, merged_depth


def composite_robot_on_frame(
    video_frame: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    hand_mask: Optional[np.ndarray] = None,
    edge_blur: int = 3,
    inpaint_method: str = "telea",
) -> np.ndarray:
    """P0/P1 compositing without depth occlusion."""
    result = _inpaint_hand(video_frame, hand_mask, inpaint_method=inpaint_method)
    return alpha_blend(result, robot_rgb, robot_mask, edge_blur=edge_blur)
