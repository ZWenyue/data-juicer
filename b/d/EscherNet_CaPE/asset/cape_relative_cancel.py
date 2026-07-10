#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why CaPE encodes ONLY the relative camera pose (the world origin cancels).

Left panel : the same reference camera (key, P2) and target camera (query,
             P1) drawn under TWO arbitrary choices of world frame W and W'.
             The relative transform P2^{-1} P1 is identical in both, so the
             attention score must be identical too.
Right panel : the 3-line algebra that makes this happen for 6DoF CaPE --
             query is embedded with phi(P1) as P1^{-T}, key with phi(P2) as
             P2, and their dot product collapses to a function of
             P2^{-1} P1 alone.

Run: /mnt/r/VENV/dj/bin/python cape_relative_cancel.py
Output: cape_relative_cancel.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrow, FancyBboxPatch  # noqa: E402


def cam_marker(ax, x, y, color, label, angle=0, label_dy=-0.2, label_ha="center"):
    # simple camera: a small triangle (frustum) + label
    L = 0.16
    dx, dy = np.cos(angle), np.sin(angle)
    px, py = -np.sin(angle), np.cos(angle)
    tip = np.array([x, y])
    b1 = tip + L * np.array([dx, dy]) + 0.6 * L * np.array([px, py])
    b2 = tip + L * np.array([dx, dy]) - 0.6 * L * np.array([px, py])
    ax.fill([tip[0], b1[0], b2[0]], [tip[1], b1[1], b2[1]], color=color, alpha=0.85, zorder=5)
    ax.scatter([x], [y], s=30, color=color, zorder=6)
    ax.text(x, y + label_dy, label, color=color, fontsize=9, ha=label_ha, weight="bold")


def draw_scene(ax, origin, ocolor, wlabel):
    ox, oy = origin
    # world axes
    ax.add_patch(FancyArrow(ox, oy, 0.45, 0, width=0.005, color=ocolor, length_includes_head=True,
                            head_width=0.045))
    ax.add_patch(FancyArrow(ox, oy, 0, 0.45, width=0.005, color=ocolor, length_includes_head=True,
                            head_width=0.045))
    ax.text(ox + 0.5, oy - 0.02, "x", color=ocolor, fontsize=8)
    ax.text(ox - 0.02, oy + 0.5, "y", color=ocolor, fontsize=8)
    ax.text(ox - 0.05, oy - 0.2, wlabel, color=ocolor, fontsize=10, weight="bold", ha="left")

    # reference camera (key) and target camera (query) at FIXED positions
    # relative to the world origin -- but the two worlds are offset/rotated,
    # so absolute coords differ while the RELATIVE transform is identical.
    key_pos = (ox + 1.05, oy + 1.0)
    qry_pos = (ox + 1.7, oy + 0.4)
    cam_marker(ax, *key_pos, "#ff7f0e", "reference (key) P2", angle=np.deg2rad(200),
               label_dy=0.18)
    cam_marker(ax, *qry_pos, "#1f77b4", "target (query) P1", angle=np.deg2rad(150),
               label_dy=-0.22)

    # relative arrow between them
    ax.annotate("", xy=qry_pos, xytext=key_pos,
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2))
    mid = ((key_pos[0] + qry_pos[0]) / 2, (key_pos[1] + qry_pos[1]) / 2)
    ax.text(mid[0] + 0.12, mid[1] + 0.02, r"$P_2^{-1}P_1$", color="#d62728", fontsize=11,
            weight="bold", ha="left")


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.4))

    # ---- Left: two world frames, same relative pose ----
    axL.set_xlim(-0.5, 4.0)
    axL.set_ylim(-0.9, 4.2)
    axL.set_aspect("equal")
    axL.axis("off")
    draw_scene(axL, (0.0, 0.1), "#555555", "world W")
    draw_scene(axL, (1.15, 2.1), "#2ca02c", "world W' (shifted/rotated)")
    axL.set_title("Same two cameras, two arbitrary world frames\n"
                  r"$\Rightarrow$ absolute poses differ, but $P_2^{-1}P_1$ is the same",
                  fontsize=12, weight="bold")
    axL.text(-0.4, -0.75, "CaPE makes the attention score depend on the red relative arrow only,\n"
                          "never on where the world origin happens to sit.",
             fontsize=9.3, color="#333333")

    # ---- Right: the algebra ----
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.axis("off")
    axR.set_title("6DoF CaPE: the world origin cancels algebraically",
                  fontsize=12, weight="bold")

    box = FancyBboxPatch((0.3, 5.9), 9.4, 3.2, boxstyle="round,pad=0.15",
                         facecolor="#f4f7fb", edgecolor="#1f77b4", lw=1.5)
    axR.add_patch(box)
    axR.text(0.6, 8.55, "Embed query with  phi(P1) = P1^{-T},   key with  phi(P2) = P2 :",
             fontsize=10.5, weight="bold", color="#1f77b4")
    axR.text(0.6, 7.75, r"$\langle \phi(P_1)\,v_1,\ \phi(P_2)\,v_2\rangle "
                        r"= (P_1^{-\top} v_1)^\top (P_2\, v_2)$", fontsize=12)
    axR.text(0.6, 6.95, r"$= v_1^\top\, P_1^{-1} P_2\, v_2$", fontsize=12)
    axR.text(0.6, 6.2, r"depends on $P_1^{-1}P_2$ (equivalently $P_2^{-1}P_1$) only "
                       r"$\Rightarrow$ world frame $W$ drops out.", fontsize=10.2, color="#333333")

    box2 = FancyBboxPatch((0.3, 1.4), 9.4, 3.9, boxstyle="round,pad=0.15",
                          facecolor="#f6fbf6", edgecolor="#2ca02c", lw=1.5)
    axR.add_patch(box2)
    axR.text(0.6, 4.8, "Sanity check (any shared left-multiply by W cancels):",
             fontsize=10.5, weight="bold", color="#2ca02c")
    axR.text(0.6, 4.0, r"$P_1 \to W P_1,\quad P_2 \to W P_2$", fontsize=12)
    axR.text(0.6, 3.15, r"$((W P_1)^{-\top}v_1)^\top (W P_2)v_2"
                        r"= v_1^\top P_1^{-1}\,(W^{-1}W)\,P_2\, v_2$", fontsize=11.5)
    axR.text(0.6, 2.15, r"$= v_1^\top P_1^{-1}P_2\, v_2$   (the $W$ terms cancel, unchanged).",
             fontsize=11.5)

    fig.suptitle("CaPE encodes the RELATIVE camera transform, independent of any coordinate system",
                 fontsize=13.5, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cape_relative_cancel.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
