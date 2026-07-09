#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize a *non-trivial* example of the state-side EEF representation
used in note.md Sec. 7.02.3 / Sec. 8.1:

    s_EEF = (x, y, z) || (r1, r2)  in R^9   (position 3 + 6D rotation)

Unlike the earlier "identity rotation" example (r1=(1,0,0), r2=(0,1,0)),
here the end-effector is rotated 40 deg about the diagonal axis
n = (1,1,1)/sqrt(3), so that none of the 9 numbers is a trivial 0 or 1 --
this matches the correction already made in Sec. 7.2.50 ("rotation-matrix
entries are not always 0/1").

Left panel : 3D scene. Gray = base reference frame at the robot origin.
             Purple arrow = absolute position vector p (with dashed
             projections onto x/y/z showing the 3 numbers). Colored arrows
             at the tip of p = the EEF's local axes r1 (blue), r2 (orange)
             and the *derived* r3 = r1 x r2 (green, dashed outline to mark
             it is not stored, only computed).
Right panel: a value grid listing all 9 numbers (3 position + 6 rotation),
             plus the equivalent axis-angle reading of the same rotation
             for cross-reference with Sec. 7.2.52.

Run:
    /mnt/r/VENV/dj/bin/python state_eef_pose_example.py
Output:
    state_eef_pose_example.png (same directory, same base name)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401
from mpl_toolkits.mplot3d.proj3d import proj_transform  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

COLOR_POS = "#9467bd"   # purple: absolute position vector p
COLOR_R1 = "#1f77b4"    # blue:   local x axis r1 (stored)
COLOR_R2 = "#ff7f0e"    # orange: local y axis r2 (stored)
COLOR_R3 = "#2ca02c"    # green:  local z axis r3 = r1 x r2 (derived, not stored)
COLOR_REF = "#555555"   # gray:   base reference axes
COLOR_ZERO = "#bbbbbb"


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs2d, ys2d, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        return min(zs3d)


def add_vector(ax, origin, vec, color, lw=2.6, ls="-", mutation_scale=15):
    arrow = Arrow3D(
        [origin[0], origin[0] + vec[0]],
        [origin[1], origin[1] + vec[1]],
        [origin[2], origin[2] + vec[2]],
        mutation_scale=mutation_scale,
        lw=lw,
        arrowstyle="-|>",
        color=color,
        linestyle=ls,
    )
    ax.add_artist(arrow)


def rodrigues(axis, angle_deg):
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    theta = np.deg2rad(angle_deg)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def draw_left_panel(ax, p, R, axis_len=0.16):
    r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]

    # Base reference frame at robot origin (small gray triad)
    for e, name in zip((np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
                        (r"$\hat{x}_{base}$", r"$\hat{y}_{base}$", r"$\hat{z}_{base}$")):
        add_vector(ax, np.zeros(3), e * 0.12, COLOR_REF, lw=1.6, mutation_scale=10)
    ax.text(0.14, 0.0, -0.02, r"$\hat{x}_{base}$", color=COLOR_REF, fontsize=9)
    ax.text(0.0, 0.14, -0.02, r"$\hat{y}_{base}$", color=COLOR_REF, fontsize=9)
    ax.text(-0.06, -0.02, 0.14, r"$\hat{z}_{base}$", color=COLOR_REF, fontsize=9)
    ax.scatter([0], [0], [0], color=COLOR_REF, s=25)
    ax.text(0.0, -0.06, -0.02, "base origin", color=COLOR_REF, fontsize=8.5)

    # Absolute position vector p (purple), with dashed axis projections
    add_vector(ax, np.zeros(3), p, COLOR_POS, lw=2.8, mutation_scale=16)
    ax.plot([p[0], p[0]], [p[1], p[1]], [0, p[2]], color=COLOR_POS, lw=1.1, ls="--")
    ax.plot([p[0], p[0]], [0, p[1]], [0, 0], color=COLOR_POS, lw=1.1, ls="--")
    ax.plot([0, p[0]], [0, 0], [0, 0], color=COLOR_POS, lw=1.1, ls="--")
    ax.text(p[0] / 2, -0.055, -0.02, f"x={p[0]:.2f}", color=COLOR_POS, fontsize=8.5)
    ax.text(p[0] + 0.015, p[1] / 2, -0.02, f"y={p[1]:.2f}", color=COLOR_POS, fontsize=8.5)
    ax.text(p[0] + 0.015, p[1] + 0.01, p[2] / 2, f"z={p[2]:.2f}", color=COLOR_POS, fontsize=8.5)
    ax.text(p[0] * 0.55, p[1] * 0.55 - 0.02, p[2] * 0.55 + 0.03, r"$\mathbf{p}$",
            color=COLOR_POS, fontsize=12, weight="bold")

    # EEF local frame at the tip of p
    add_vector(ax, p, r1 * axis_len, COLOR_R1, lw=2.8)
    add_vector(ax, p, r2 * axis_len, COLOR_R2, lw=2.8)
    add_vector(ax, p, r3 * axis_len, COLOR_R3, lw=2.2, ls="--")
    ax.text(*(p + r1 * axis_len * 1.25), rf"$\mathbf{{r}}_1$=({r1[0]:.2f},{r1[1]:.2f},{r1[2]:.2f})",
            color=COLOR_R1, fontsize=9, weight="bold")
    ax.text(*(p + r2 * axis_len * 1.25 + np.array([0, 0, 0.05])),
            rf"$\mathbf{{r}}_2$=({r2[0]:.2f},{r2[1]:.2f},{r2[2]:.2f})",
            color=COLOR_R2, fontsize=9, weight="bold")
    ax.text(*(p + r3 * axis_len * 1.3 + np.array([-0.05, -0.05, 0])),
            rf"$\mathbf{{r}}_3$=({r3[0]:.2f},{r3[1]:.2f},{r3[2]:.2f})" "\n(derived, not stored)",
            color=COLOR_R3, fontsize=8.5, weight="bold")
    ax.scatter([p[0]], [p[1]], [p[2]], color="black", s=22)
    ax.text(p[0] + 0.02, p[1] - 0.05, p[2] + 0.02, "EEF", color="black", fontsize=9, weight="bold")

    ax.set_xlim(-0.05, 0.55)
    ax.set_ylim(-0.25, 0.35)
    ax.set_zlim(-0.05, 0.55)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=22, azim=-55)
    ax.set_title(
        "State EEF pose in the base frame\n"
        r"$\mathbf{p}$ = where the EEF origin is; $\mathbf{r}_1,\mathbf{r}_2,\mathbf{r}_3$ = which way it points",
        fontsize=11,
        pad=10,
    )


def draw_right_panel(ax, p, R):
    r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(0.2, 9.6, "9 numbers of  s_EEF = (x,y,z) || (r1,r2)", fontsize=12.5, weight="bold")

    # Position block
    ax.add_patch(Rectangle((0.2, 7.6), 9.6, 1.5, facecolor="#f2ecf9", edgecolor=COLOR_POS, lw=1.6))
    ax.text(0.4, 8.75, "Position  p  (3 numbers, meters, base frame)", color=COLOR_POS,
            fontsize=10.5, weight="bold")
    ax.text(0.4, 8.15,
            f"x = {p[0]:+.2f}   y = {p[1]:+.2f}   z = {p[2]:+.2f}",
            fontsize=11, family="monospace")
    ax.text(0.4, 7.75,
            "meaning: EEF origin is 0.38 m in front of the base, 0.09 m to its right, 0.34 m above it",
            fontsize=8.7, color="#333333")

    # r1 block
    ax.add_patch(Rectangle((0.2, 5.7), 9.6, 1.6, facecolor="#e8f1fb", edgecolor=COLOR_R1, lw=1.6))
    ax.text(0.4, 6.95, "r1 = local x-axis direction (3 numbers, stored)", color=COLOR_R1,
            fontsize=10.5, weight="bold")
    ax.text(0.4, 6.35,
            f"r1x = {r1[0]:+.3f}   r1y = {r1[1]:+.3f}   r1z = {r1[2]:+.3f}",
            fontsize=11, family="monospace")
    ax.text(0.4, 5.85,
            "meaning: mostly points along base-x, tilted toward +y and -z (roll/twist mix)",
            fontsize=8.7, color="#333333")

    # r2 block
    ax.add_patch(Rectangle((0.2, 3.8), 9.6, 1.6, facecolor="#fdf1e6", edgecolor=COLOR_R2, lw=1.6))
    ax.text(0.4, 5.05, "r2 = local y-axis direction (3 numbers, stored)", color=COLOR_R2,
            fontsize=10.5, weight="bold")
    ax.text(0.4, 4.45,
            f"r2x = {r2[0]:+.3f}   r2y = {r2[1]:+.3f}   r2z = {r2[2]:+.3f}",
            fontsize=11, family="monospace")
    ax.text(0.4, 3.95,
            "meaning: mostly points along base-y, tilted toward +z and -x",
            fontsize=8.7, color="#333333")

    # r3 (derived) block
    ax.add_patch(Rectangle((0.2, 1.9), 9.6, 1.6, facecolor="#eaf6ea", edgecolor=COLOR_R3, lw=1.6, ls="--"))
    ax.text(0.4, 3.15, "r3 = r1 x r2  (derived, NOT stored -- saves 3 dims)", color=COLOR_R3,
            fontsize=10.5, weight="bold")
    ax.text(0.4, 2.55,
            f"r3x = {r3[0]:+.3f}   r3y = {r3[1]:+.3f}   r3z = {r3[2]:+.3f}",
            fontsize=11, family="monospace")
    ax.text(0.4, 2.05,
            "meaning: gripper's approach/pointing axis, reconstructed via right-hand rule",
            fontsize=8.7, color="#333333")

    ax.text(0.2, 1.2,
            r"Equivalent axis-angle reading of $\mathbf{R}=[\mathbf{r}_1|\mathbf{r}_2|\mathbf{r}_3]$: "
            r"rotate $40°$ about axis $\hat{n}=(0.577,0.577,0.577)$"
            "\n(diagonal of base x,y,z) "
            r"$\to \boldsymbol{\omega} = 40°\cdot\hat{n} \approx (0.403,0.403,0.403)$ rad (cf. Sec. 7.2.52)",
            fontsize=8.3, color="#555555")


def main():
    p = np.array([0.38, -0.09, 0.34])
    R = rodrigues([1, 1, 1], 40.0)

    fig = plt.figure(figsize=(15, 7.4))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2)

    draw_left_panel(ax_left, p, R)
    draw_right_panel(ax_right, p, R)

    fig.suptitle(
        "State-side EEF example: position 3 + 6D rotation (9 numbers, none is a trivial 0/1)",
        fontsize=14,
        weight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_eef_pose_example.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
