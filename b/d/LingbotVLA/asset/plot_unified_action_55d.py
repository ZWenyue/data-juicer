#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""55-dim unified action representation bar chart for LingBot-VLA 2.0 note."""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

SEGMENTS = [
    ("Arm joint\n(14)", 14, "#4C78A8"),
    ("EEF pose\n(14)", 14, "#F58518"),
    ("Gripper\n(2)", 2, "#E45756"),
    ("Hand\n(12)", 12, "#72B7B2"),
    ("Waist\n(4)", 4, "#54A24B"),
    ("Head\n(2)", 2, "#EECA3B"),
    ("Mobility\n(3)", 3, "#B279A2"),
    ("Reserved\n(4)", 4, "#9D755D"),
]


def main():
    fig, ax = plt.subplots(figsize=(12, 2.8))
    x = 0
    for name, width, color in SEGMENTS:
        rect = FancyBboxPatch(
            (x, 0.25),
            width,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.92,
        )
        ax.add_patch(rect)
        ax.text(
            x + width / 2,
            0.5,
            name,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
        ax.text(x + width / 2, 0.05, f"d={width}", ha="center", va="bottom", fontsize=8, color="#333")
        x += width

    ax.set_xlim(-0.5, 55.5)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0.0, color="#ccc", linewidth=0.8)
    ax.set_yticks([])
    ax.set_xticks([0, 14, 28, 30, 42, 46, 48, 51, 55])
    ax.set_xlabel("Canonical state / action dimension index (total = 55)", fontsize=11)
    ax.set_title("LingBot-VLA 2.0 Unified Action Representation (55-D)", fontsize=13, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "unified_action_55d.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
