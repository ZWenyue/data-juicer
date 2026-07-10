#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dual-query distillation flow schematic for LingBot-VLA 2.0."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))


def box(ax, xy, w, h, text, fc, fs=9):
    r = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=fc, edgecolor="#333", linewidth=1.2, alpha=0.95,
    )
    ax.add_patch(r)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2, text=None, color="#444"):
    ax.add_patch(
        FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4)
    )
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.15, text, ha="center", fontsize=8, color=color)


def main():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Dual-Query Distillation: Geometry + Causal Temporal Priors", fontsize=13, pad=10)

    box(ax, (0.3, 3.2), 2.2, 1.6, "Observation\n$I_t$  &  $I_{t+T}$\n(+ language)", "#AED6F1", fs=9)
    box(ax, (3.2, 3.0), 2.6, 2.0, "Causal VLM\n(Qwen3-VL)\n+ Action Expert", "#D5F5E3", fs=9)

    box(ax, (6.5, 5.5), 2.0, 1.2, "Query $Q_t$\n(current)", "#F9E79F", fs=9)
    box(ax, (6.5, 2.0), 2.0, 1.2, "Query $Q_{t+T}$\n(future)", "#F5CBA7", fs=9)

    box(ax, (9.5, 6.2), 3.0, 1.2, "LingBot-Depth\n(frozen teacher)\n$D_t, D_{t+T}$", "#D7BDE2", fs=8)
    box(ax, (9.5, 3.8), 3.0, 1.2, "DINO-Video\n(frozen teacher)\n$Z_t, Z_{t+T}$", "#F5B7B1", fs=8)

    box(ax, (9.5, 0.6), 3.0, 1.4, "Action Expert\nFlow Matching\n$\\rightarrow$ action chunk", "#ABEBC6", fs=8)

    arrow(ax, (2.5, 4.0), (3.2, 4.0))
    arrow(ax, (5.8, 4.5), (6.5, 6.0), "append")
    arrow(ax, (5.8, 3.5), (6.5, 2.6), "append")
    arrow(ax, (8.5, 6.1), (9.5, 6.7), r"$L_{depth}$")
    arrow(ax, (8.5, 5.7), (9.5, 4.6), r"$L_{video}$")
    arrow(ax, (8.5, 2.4), (9.5, 6.5), color="#7D3C98")
    arrow(ax, (8.5, 2.2), (9.5, 4.2), color="#922B21")
    arrow(ax, (5.8, 3.2), (9.5, 1.5), "control path")

    ax.text(0.3, 7.3,
            r"Train: $L = L_{action} + \alpha L_{depth} + \beta L_{video}$"
            "\nInfer: drop teachers & queries; keep flow-matching denoising only.",
            fontsize=9, color="#333",
            bbox=dict(boxstyle="round", facecolor="#F8F9F9", edgecolor="#BBB"))

    out = os.path.join(HERE, "dual_query_flow.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
