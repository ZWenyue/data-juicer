#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation ranking bar chart from LingBot-VLA 2.0 paper numbers."""
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # Action target
    ax = axes[0]
    labels = ["absQpos", "relQpos"]
    vals = [33.7, 55.0]
    colors = ["#E74C3C", "#27AE60"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylim(0, 70)
    ax.set_ylabel("Avg success rate (%)")
    ax.set_title("Action target\n(most effective)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=10)

    # Normalization
    ax = axes[1]
    labels = ["MinMax", "Q01–Q99", "MeanStd"]
    vals = [47.5, 47.4, 55.0]
    colors = ["#E67E22", "#F1C40F", "#27AE60"]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_ylim(0, 70)
    ax.set_title("Normalization\n(MeanStd best)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=10)

    # Loss
    ax = axes[2]
    labels = ["L1", "L2"]
    vals = [46.4, 55.0]
    colors = ["#E67E22", "#27AE60"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylim(0, 70)
    ax.set_title("Loss function\n(L2 better overall)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=10)

    fig.suptitle(
        "GM-100 Ablation Ranking (4 real-robot tasks, avg success %)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "ablation_ranking.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
