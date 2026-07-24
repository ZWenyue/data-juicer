# -*- coding: utf-8 -*-
"""Hand mask generation and robot/frame compositing."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


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


def composite_robot_on_frame(
    video_frame: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    hand_mask: Optional[np.ndarray] = None,
    edge_blur: int = 3,
    inpaint_method: str = "telea",
) -> np.ndarray:
    """P0/P1 compositing without depth occlusion."""
    result = video_frame.copy()
    if hand_mask is not None:
        inpaint_mask = hand_mask.astype(np.uint8) * 255
        flag = cv2.INPAINT_TELEA if inpaint_method != "ns" else cv2.INPAINT_NS
        result = cv2.inpaint(result, inpaint_mask, 5, flag)
    return alpha_blend(result, robot_rgb, robot_mask, edge_blur=edge_blur)
