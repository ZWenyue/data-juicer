#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the R_z(30 deg) rotation matrix example used in note.md.

Left panel : a 3D scene showing the reference frame {x, y, z} (gray) and
             the local frame {r1, r2, r3} (blue/orange/green) obtained by
             rotating the reference frame 30 deg about the z axis. The
             rotation angle theta=30 deg and the projections of r1, r2 onto
             the x/y axes (i.e. cos30 deg, sin30 deg) are annotated.
Right panel: the 3x3 matrix drawn as a color-coded grid, so that each of
             the 9 matrix entries can be read off directly and matched to
             the geometric picture on the left (same color per column).

Run:
    /mnt/r/VENV/dj/bin/python rotation_matrix_rz30_example.py
Output:
    rotation_matrix_rz30_example.png (same directory, same base name)
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

COLOR_R1 = "#1f77b4"  # blue: local x axis -> column 1 of R
COLOR_R2 = "#ff7f0e"  # orange: local y axis -> column 2 of R
COLOR_R3 = "#2ca02c"  # green: local z axis -> column 3 of R
COLOR_REF = "#555555"  # gray: reference-frame axes
COLOR_ZERO = "#bbbbbb"


class Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch that can be projected/drawn in 3D axes."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs2d, ys2d, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        return min(zs3d)


def add_vector(ax, origin, vec, color, lw=2.6, ls="-", mutation_scale=16):
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


def draw_left_panel(ax, theta_deg=30.0):
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)

    e1, e2, e3 = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    r1 = np.array([c, s, 0.0])  # column 1 of R
    r2 = np.array([-s, c, 0.0])  # column 2 of R
    r3 = np.array([0.0, 0.0, 1.0])  # column 3 of R

    origin = np.zeros(3)

    # Reference-frame axes (gray, solid)
    add_vector(ax, origin, e1 * 1.15, COLOR_REF, lw=1.8)
    add_vector(ax, origin, e2 * 1.15, COLOR_REF, lw=1.8)
    add_vector(ax, origin, e3 * 1.15, COLOR_REF, lw=1.8)
    ax.text(1.22, -0.05, -0.05, r"$\hat{x}$ (reference)", color=COLOR_REF, fontsize=10)
    ax.text(0.32, 1.28, 0, r"$\hat{y}$ (reference)", color=COLOR_REF, fontsize=10)
    ax.text(-0.75, -0.05, 1.22, r"$\hat{z}$ (reference)", color=COLOR_REF, fontsize=10)

    # Rotated local axes (= the 3 columns of the rotation matrix)
    add_vector(ax, origin, r1, COLOR_R1, lw=3.0)
    add_vector(ax, origin, r2, COLOR_R2, lw=3.0)
    add_vector(ax, origin, r3, COLOR_R3, lw=3.0)
    ax.text(r1[0] * 1.08 + 0.03, r1[1] * 1.08 - 0.12, 0.16, r"$\mathbf{r}_1$ (local x, col 1)",
            color=COLOR_R1, fontsize=10, weight="bold")
    ax.text(r2[0] * 1.35 - 0.08, r2[1] * 1.1 + 0.05, 0.16, r"$\mathbf{r}_2$ (local y, col 2)",
            color=COLOR_R2, fontsize=10, weight="bold")
    ax.text(0.45, 0.42, 1.05, r"$\mathbf{r}_3$ (local z, col 3)" "\n(coincides with $\hat{z}$)",
            color=COLOR_R3, fontsize=10, weight="bold")

    # Arc for the rotation angle theta, in the xy-plane, from e1 to r1
    arc_t = np.linspace(0, theta, 40)
    arc_r = 0.55
    ax.plot(arc_r * np.cos(arc_t), arc_r * np.sin(arc_t), np.zeros_like(arc_t), color="black", lw=1.4)
    ax.text(arc_r * np.cos(theta / 2) + 0.06, arc_r * np.sin(theta / 2) - 0.09, -0.05,
            r"$\theta=30°$", fontsize=11, color="black")

    # Projections of r1 onto x, y axes (= cos30, sin30), dashed guide lines
    ax.plot([r1[0], r1[0]], [r1[1], 0], [0, 0], color=COLOR_R1, lw=1.2, ls="--")
    ax.plot([0, r1[0]], [0, 0], [0, 0], color=COLOR_R1, lw=1.2, ls="--")
    ax.text(r1[0] / 2 - 0.08, -0.32, 0, rf"$\cos30°={c:.3f}$", color=COLOR_R1, fontsize=9)
    ax.plot([r1[0], r1[0]], [0, r1[1]], [0, 0], color=COLOR_R1, lw=1.2, ls=":")
    ax.text(r1[0] + 0.05, r1[1] / 2 - 0.18, -0.05, rf"$\sin30°={s:.3f}$", color=COLOR_R1, fontsize=9)

    # Projections of r2 onto x, y axes (= -sin30, cos30)
    ax.plot([r2[0], 0], [r2[1], r2[1]], [0, 0], color=COLOR_R2, lw=1.2, ls="--")
    ax.text(r2[0] - 0.55, r2[1] - 0.35, 0.1, rf"$-\sin30°={-s:.3f}$", color=COLOR_R2, fontsize=9)

    ax.set_xlim(-0.8, 1.3)
    ax.set_ylim(-0.4, 1.3)
    ax.set_zlim(-0.2, 1.3)
    ax.set_box_aspect((1, 1, 0.75))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=24, azim=-60)
    ax.set_title(
        r"Rotate about $\hat{z}$ by $30°$: reference axes (gray) vs. rotated"
        "\nlocal axes $\mathbf{r}_1,\mathbf{r}_2,\mathbf{r}_3$ (colored)",
        fontsize=11.5,
        pad=14,
    )


def draw_right_panel(ax, theta_deg=30.0):
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)

    # Consistent with R = [r1 | r2 | r3]
    # rows = reference-frame components (x, y, z); cols = local axes r1, r2, r3
    cell_text = [
        [r"$\cos30°$" + f"\n= {c:.3f}", r"$-\sin30°$" + f"\n= {-s:.3f}", "0"],
        [r"$\sin30°$" + f"\n= {s:.3f}", r"$\cos30°$" + f"\n= {c:.3f}", "0"],
        ["0", "0", "1"],
    ]
    col_colors = [COLOR_R1, COLOR_R2, COLOR_R3]
    row_labels = [r"ref. $\hat{x}$ component (row 1)", r"ref. $\hat{y}$ component (row 2)",
                  r"ref. $\hat{z}$ component (row 3)"]
    col_labels = [r"$\mathbf{r}_1$ local x", r"$\mathbf{r}_2$ local y", r"$\mathbf{r}_3$ local z"]

    n = 3
    cell = 1.0
    x0, y0 = 0.0, 0.0

    ax.set_xlim(-2.9, n * cell + 0.3)
    ax.set_ylim(-0.6, n * cell + 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Column headers (top), color-coded to match the arrows on the left
    for j in range(n):
        cx = x0 + j * cell + cell / 2
        ax.text(cx, n * cell + 0.55, col_labels[j], ha="center", va="bottom",
                color=col_colors[j], fontsize=11, weight="bold")
        ax.add_patch(Rectangle((x0 + j * cell + 0.08, n * cell + 0.05), cell - 0.16, 0.12,
                                facecolor=col_colors[j], edgecolor="none"))

    # Row headers (left)
    for i in range(n):
        cy = y0 + (n - 1 - i) * cell + cell / 2
        ax.text(x0 - 0.15, cy, row_labels[i], ha="right", va="center", fontsize=10, color=COLOR_REF)

    # Grid cells
    for i in range(n):
        for j in range(n):
            cx0 = x0 + j * cell
            cy0 = y0 + (n - 1 - i) * cell
            is_zero = cell_text[i][j] == "0"
            face = "white" if is_zero else _lighten(col_colors[j])
            edge = COLOR_ZERO if is_zero else col_colors[j]
            ax.add_patch(Rectangle((cx0, cy0), cell, cell, facecolor=face, edgecolor=edge, lw=1.8))
            txt_color = COLOR_ZERO if is_zero else "black"
            ax.text(cx0 + cell / 2, cy0 + cell / 2, cell_text[i][j], ha="center", va="center",
                    fontsize=11.5, color=txt_color, weight=("normal" if is_zero else "bold"))

    # Bracket-style matrix border
    ax.plot([x0 - 0.06, x0 - 0.06], [y0 - 0.06, y0 + n * cell + 0.06], color="black", lw=1.6)
    ax.plot([x0 - 0.06, x0 + 0.12], [y0 - 0.06, y0 - 0.06], color="black", lw=1.6)
    ax.plot([x0 - 0.06, x0 + 0.12], [y0 + n * cell + 0.06, y0 + n * cell + 0.06], color="black", lw=1.6)

    xr = x0 + n * cell
    ax.plot([xr + 0.06, xr + 0.06], [y0 - 0.06, y0 + n * cell + 0.06], color="black", lw=1.6)
    ax.plot([xr - 0.12, xr + 0.06], [y0 - 0.06, y0 - 0.06], color="black", lw=1.6)
    ax.plot([xr - 0.12, xr + 0.06], [y0 + n * cell + 0.06, y0 + n * cell + 0.06], color="black", lw=1.6)

    ax.set_title(
        r"$\mathbf{R}_z(30°) = [\mathbf{r}_1 \mid \mathbf{r}_2 \mid \mathbf{r}_3]$"
        "\nEach column = direction of a local axis in the reference frame;"
        "\neach row = the component along one reference axis",
        fontsize=11.5,
        pad=18,
    )


def _lighten(hex_color, factor=0.78):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i: i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    fig = plt.figure(figsize=(15, 7.2))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2)

    draw_left_panel(ax_left, theta_deg=30.0)
    draw_right_panel(ax_right, theta_deg=30.0)

    fig.suptitle(
        r"Rotation matrix example: geometry of $\mathbf{R}_z(30°)$ $\leftrightarrow$ its 9 entries",
        fontsize=14.5,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotation_matrix_rz30_example.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
