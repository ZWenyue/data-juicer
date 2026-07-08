#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 Stage 5（Base Frame & EEF Orientation Alignment）验收生成合成 EEF 位姿数据。

真实 Galaxea 数据集是「关节空间」的（无 EEF 位姿），无法直接验证 Stage 5 的位姿
旋转校正。这里合成一批「在错位世界系里记录」的 EEF 位姿轨迹：机器人沿其自身
「前向」运动，但被记录到一个绕 z 轴旋转过的世界系里（forward = +y 而非 +x）。
Stage 5 应能用 R_corr（如 preset z_-90）把它对齐回「+x = 前向」的规范系。

每条 episode 的样本布局（统一 state-action 表示的简化版，见 note_data.md）：

    state  = [ pos(3), rot6d(6) ]                 -> 10 维，绝对位姿
    action = [ delta_rotvec(3), delta_trans(3) ]  -> 6 维，相机/世界系 delta 位姿

其中 pos 沿「记录系前向」轴线性推进（默认 +y），叠加小幅噪声；rot6d 为一个绕 z
轴缓慢转动的朝向；action 为相邻帧 delta（旋转向量 + 平移）。
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

# 记录系里「前向」对应的轴（默认 +y：模拟一个绕 z 轴 +90° 的错位世界系）
_AXIS = {"x": 0, "y": 1, "z": 2}


def _matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """(T,3,3) -> (T,6)，取旋转矩阵前两列（与算子实现一致）。"""
    return np.concatenate([mat[:, :, 0], mat[:, :, 1]], axis=1)


def _gen_episode(ep_idx: int, num_frames: int, forward_axis: str, rng: np.random.Generator):
    fa = _AXIS[forward_axis]

    # ---- 绝对位姿：pos 沿 forward_axis 线性推进 + 微噪声 ----
    t = np.linspace(0.0, 1.0, num_frames)
    pos = np.zeros((num_frames, 3), dtype=np.float64)
    pos[:, fa] = 0.6 * t + 0.05 * ep_idx  # 前向推进
    # 另一水平轴上给一点缓慢漂移，纵向抬升一点
    other = (fa + 1) % 3
    pos[:, other] = 0.05 * np.sin(2 * np.pi * t)
    pos[:, 2] += 0.1 * t
    pos += rng.normal(scale=1e-3, size=pos.shape)

    # ---- 朝向：绕 z 轴从 0 缓慢转到 ~20°，转成 rot6d ----
    yaw = np.deg2rad(np.linspace(0.0, 20.0, num_frames))
    R_abs = Rotation.from_euler("z", yaw).as_matrix()  # (T,3,3)
    rot6d = _matrix_to_rot6d(R_abs)  # (T,6)
    state = np.concatenate([pos, rot6d], axis=1)  # (T,10)

    # ---- delta 动作：相邻帧的 delta 旋转向量 + delta 平移 ----
    delta_rotvec = np.zeros((num_frames, 3), dtype=np.float64)
    for k in range(1, num_frames):
        dR = R_abs[k] @ R_abs[k - 1].T
        delta_rotvec[k] = Rotation.from_matrix(dR).as_rotvec()
    delta_trans = np.zeros((num_frames, 3), dtype=np.float64)
    delta_trans[1:] = pos[1:] - pos[:-1]
    action = np.concatenate([delta_rotvec, delta_trans], axis=1)  # (T,6)

    return {
        "id": f"synth_pose_ep{ep_idx:03d}",
        "robot_type": "synthetic_arm",
        "forward_axis_recorded": forward_axis,
        "states": state.round(6).tolist(),
        "actions": action.round(6).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic EEF pose dataset for Stage 5.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--num_episodes", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=60)
    ap.add_argument(
        "--forward_axis", choices=list(_AXIS), default="y", help="Recorded-frame forward axis (misaligned). Default +y."
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n = 0
    with open(args.output, "w") as f:
        for ep in range(args.num_episodes):
            sample = _gen_episode(ep, args.num_frames, args.forward_axis, rng)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} synthetic pose episodes -> {args.output}")
    print(
        f"  layout: state=[pos(3),rot6d(6)] action=[drotvec(3),dtrans(3)] "
        f"frames={args.num_frames} forward_axis(recorded)=+{args.forward_axis}"
    )


if __name__ == "__main__":
    main()
