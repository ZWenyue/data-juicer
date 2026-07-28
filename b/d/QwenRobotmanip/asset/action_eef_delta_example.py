#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize a non-trivial example of the action-side EEF representation used
in note.md Sec. 7.02.3 / Sec. 8.2:

    a_EEF = (tx, ty, tz) || (wx, wy, wz)  in R^6   (translation delta 3 +
                                                     rotation-vector delta 3)

Unlike a "tiny, hard to see" delta, this example uses a still-small but
clearly visualizable step: t = (4.5cm, -2.8cm, 6.0cm) and a rotation vector
omega = (0.15, -0.10, 0.28) rad (~19.1 deg about a tilted axis), so every
one of the 6 numbers is a genuine (non 0/1) value.

Left panel : 3D scene in the *reference camera frame*. Gray triad at the
             origin = current EEF orientation ("now"). Magenta arrow =
             translation delta t (dashed projections show tx, ty, tz).
             At the tip of t: the *rotated* frame r1', r2', r3' obtained by
             applying exp(omega) to the identity axes (blue/orange/green),
             plus the rotation axis n = omega/|omega| (dashed black) and an
             arc showing the rotation angle theta = |omega|.
Right panel: value grid listing all 6 numbers plus the derived
             axis-angle decomposition (n, theta) for cross-reference with
             Sec. 7.2.52 / Sec. 8.2.

Run:
    /mnt/r/VENV/dj/bin/python action_eef_delta_example.py
Output:
    action_eef_delta_example.png (same directory, same base name)
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

COLOR_T = "#d62728"     # red/magenta: translation delta t
COLOR_R1 = "#1f77b4"    # blue:   rotated local x axis
COLOR_R2 = "#ff7f0e"    # orange: rotated local y axis
COLOR_R3 = "#2ca02c"    # green:  rotated local z axis
COLOR_REF = "#555555"   # gray:   current ("now") frame, identity
COLOR_AXIS = "#111111"  # black:  rotation axis n
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


def rodrigues(axis, angle_rad):
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def orthonormal_basis_perp_to(n):
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def draw_left_panel(ax, t, omega, axis_len=0.032):
    theta = np.linalg.norm(omega)
    n = omega / theta
    Rw = rodrigues(n, theta)
    r1, r2, r3 = Rw[:, 0], Rw[:, 1], Rw[:, 2]

    origin = np.zeros(3)

    # Current ("now") identity frame at origin, gray
    for e, name, off in zip((np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
                             ("x", "y", "z"), (0, 0, 0)):
        add_vector(ax, origin, e * axis_len * 0.9, COLOR_REF, lw=1.6, mutation_scale=10)
    ax.text(axis_len * 1.0, -0.004, -0.004, "now: cam-x", color=COLOR_REF, fontsize=8)
    ax.text(-0.006, axis_len * 1.0, -0.004, "now: cam-y", color=COLOR_REF, fontsize=8)
    ax.text(-0.006, -0.004, axis_len * 1.05, "now: cam-z", color=COLOR_REF, fontsize=8)
    ax.scatter([0], [0], [0], color=COLOR_REF, s=25)
    ax.text(0.002, -0.012, -0.006, "EEF now", color=COLOR_REF, fontsize=8.5)

    # Translation delta t (magenta), dashed axis projections
    add_vector(ax, origin, t, COLOR_T, lw=2.8, mutation_scale=16)
    ax.plot([t[0], t[0]], [t[1], t[1]], [0, t[2]], color=COLOR_T, lw=1.1, ls="--")
    ax.plot([t[0], t[0]], [0, t[1]], [0, 0], color=COLOR_T, lw=1.1, ls="--")
    ax.plot([0, t[0]], [0, 0], [0, 0], color=COLOR_T, lw=1.1, ls="--")
    ax.text(t[0] / 2 - 0.006, -0.020, -0.006, f"tx={t[0]*1000:.0f}mm", color=COLOR_T, fontsize=8.3)
    ax.text(t[0] + 0.004, t[1] / 2 - 0.004, -0.006, f"ty={t[1]*1000:.0f}mm", color=COLOR_T, fontsize=8.3)
    ax.text(t[0] + 0.004, t[1] + 0.004, t[2] / 2, f"tz={t[2]*1000:.0f}mm", color=COLOR_T, fontsize=8.3)
    ax.text(t[0] * 0.5, t[1] * 0.5 - 0.006, t[2] * 0.5 + 0.008, r"$\Delta\mathbf{t}$",
            color=COLOR_T, fontsize=12, weight="bold")
    ax.scatter([t[0]], [t[1]], [t[2]], color="black", s=22)
    ax.text(t[0] + 0.003, t[1] - 0.011, t[2] - 0.006, "EEF next", color="black", fontsize=8.5, weight="bold")

    # Rotation axis n, drawn from the target point
    add_vector(ax, t, n * axis_len * 1.3, COLOR_AXIS, lw=1.6, ls=":", mutation_scale=10)
    ax.text(*(t + n * axis_len * 1.5 + np.array([0.006, 0, 0])),
            r"$\hat{n}=\boldsymbol{\omega}/\|\boldsymbol{\omega}\|$",
            color=COLOR_AXIS, fontsize=8.3)

    # Rotated frame r1', r2', r3' at the target point (shows the "twist")
    add_vector(ax, t, r1 * axis_len, COLOR_R1, lw=2.6)
    add_vector(ax, t, r2 * axis_len, COLOR_R2, lw=2.6)
    add_vector(ax, t, r3 * axis_len, COLOR_R3, lw=2.0, ls="--")
    ax.text(*(t + r1 * axis_len * 1.25), "r1'", color=COLOR_R1, fontsize=9, weight="bold")
    ax.text(*(t + r2 * axis_len * 1.25), "r2'", color=COLOR_R2, fontsize=9, weight="bold")
    ax.text(*(t + r3 * axis_len * 1.3), "r3'", color=COLOR_R3, fontsize=9, weight="bold")

    # Arc showing rotation angle theta around axis n, in the plane perp to n
    u, v = orthonormal_basis_perp_to(n)
    arc_r = axis_len * 0.55
    tt = np.linspace(0, theta, 40)
    arc_pts = np.array([t + arc_r * (np.cos(ti) * u + np.sin(ti) * v) for ti in tt])
    ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2], color=COLOR_AXIS, lw=1.3)
    mid = t + arc_r * 1.7 * (np.cos(theta / 2) * u + np.sin(theta / 2) * v) + np.array([0, -0.006, -0.004])
    ax.text(*mid, rf"$\theta=\|\boldsymbol{{\omega}}\|={np.degrees(theta):.1f}°$", color=COLOR_AXIS, fontsize=8.3)

    ax.set_xlim(-0.01, 0.06)
    ax.set_ylim(-0.045, 0.02)
    ax.set_zlim(-0.01, 0.07)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("cam X (m)")
    ax.set_ylabel("cam Y (m)")
    ax.set_zlabel("cam Z (m)")
    ax.view_init(elev=20, azim=-50)
    ax.set_title(
        "Action EEF delta in the reference camera frame\n"
        r"$\Delta\mathbf{t}$ = how far to move; $\boldsymbol{\omega}$ = axis $\times$ angle to twist",
        fontsize=11,
        pad=10,
    )


def draw_right_panel(ax, t, omega):
    theta = np.linalg.norm(omega)
    n = omega / theta
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(0.2, 9.6, "6 numbers of  a_EEF = (tx,ty,tz) || (wx,wy,wz)", fontsize=12.5, weight="bold")

    # Translation block
    ax.add_patch(Rectangle((0.2, 7.3), 9.6, 1.9, facecolor="#fbe9e9", edgecolor=COLOR_T, lw=1.6))
    ax.text(0.4, 8.85, "Translation delta  t  (3 numbers, meters, camera frame)", color=COLOR_T,
            fontsize=10.5, weight="bold")
    ax.text(0.4, 8.2,
            f"tx = {t[0]*1000:+.0f} mm   ty = {t[1]*1000:+.0f} mm   tz = {t[2]*1000:+.0f} mm",
            fontsize=11, family="monospace")
    ax.text(0.4, 7.55,
            "meaning: from this step to the next, the EEF should move +4.5cm along cam-x (right in\n"
            "image), -2.8cm along cam-y (up), +6.0cm along cam-z (toward the camera)",
            fontsize=8.5, color="#333333")

    # Rotation vector block
    ax.add_patch(Rectangle((0.2, 4.9), 9.6, 2.2, facecolor="#e8f1fb", edgecolor=COLOR_R1, lw=1.6))
    ax.text(0.4, 6.85, r"Rotation vector  $\boldsymbol{\omega}$  (3 numbers, rad, camera frame)",
            color=COLOR_R1, fontsize=10.5, weight="bold")
    ax.text(0.4, 6.2,
            f"wx = {omega[0]:+.3f}   wy = {omega[1]:+.3f}   wz = {omega[2]:+.3f}",
            fontsize=11, family="monospace")
    ax.text(0.4, 5.6,
            "meaning: rotate by angle theta=|w| about the axis n=w/|w|; each component is\n"
            "the axis direction scaled by the angle (in rad)",
            fontsize=8.5, color="#333333")
    ax.text(0.4, 5.05,
            f"axis n = ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f})    "
            f"angle theta = {theta:.3f} rad = {np.degrees(theta):.1f}°",
            fontsize=9.3, color="#333333", weight="bold")

    # Comparison note
    ax.add_patch(Rectangle((0.2, 2.9), 9.6, 1.7, facecolor="#f4f4f4", edgecolor=COLOR_ZERO, lw=1.4))
    ax.text(0.4, 4.3, "vs. state EEF (absolute, 9D, base frame):", fontsize=9.5, weight="bold",
            color="#333333")
    ax.text(0.4, 3.75,
            "state answers \"where/how is it NOW\" (absolute p, 6D rotation);",
            fontsize=8.6, color="#333333")
    ax.text(0.4, 3.25,
            "action answers \"how should it MOVE next\" (relative t, 3D rotation vector).",
            fontsize=8.6, color="#333333")

    ax.text(0.2, 2.2,
            "Small-angle regime: theta ~ 19 deg << 180 deg singularity of the rotation-vector\n"
            "representation, and exp(w) ~ I + [w]_x for even smaller steps (cf. Sec. 8.2.3).",
            fontsize=8.3, color="#555555")


def main():
    t = np.array([0.045, -0.028, 0.060])
    omega = np.array([0.15, -0.10, 0.28])

    fig = plt.figure(figsize=(15, 7.4))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2)

    draw_left_panel(ax_left, t, omega)
    draw_right_panel(ax_right, t, omega)

    fig.suptitle(
        "Action-side EEF example: translation 3 + rotation-vector 3 (6 numbers, none is a trivial 0/1)",
        fontsize=14,
        weight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_eef_delta_example.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
