#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE routing + loss-free bias load-balancing schematic for LingBot-VLA 2.0."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def box(ax, xy, w, h, text, fc, ec="#333", fs=9):
    r = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.2, alpha=0.95,
    )
    ax.add_patch(r)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color="#222")


def arrow(ax, p1, p2, color="#555"):
    ax.add_patch(
        FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3)
    )


def main():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "Token-level Sparse MoE in Action Expert (loss-free load balancing)",
        fontsize=13,
        pad=8,
    )

    box(ax, (0.4, 3.0), 1.8, 1.0, "Token $u_{\\ell,t}$\n(modulated FFN in)", "#D6EAF8", fs=8)

    box(ax, (2.8, 5.2), 2.2, 0.9, "Shared Expert\n$E^{(s)}$ (always on)", "#A9DFBF", fs=9)
    box(ax, (2.8, 3.0), 2.2, 1.0, "Router\n$z_j = u^\\top e_j$\n$s_j=\\sigma(z_j)$", "#F9E79F", fs=8)

    # routed experts
    experts = ["$E_1^{(r)}$", "$E_2^{(r)}$", "$E_3^{(r)}$", "...", f"$E_{{N_r}}^{{(r)}}$"]
    xs = [5.6, 7.0, 8.4, 9.6, 10.8]
    for i, (x, name) in enumerate(zip(xs, experts)):
        fc = "#F5B7B1" if i in (0, 2) else "#FADBD8"
        ec = "#C0392B" if i in (0, 2) else "#999"
        box(ax, (x - 0.45, 2.9), 0.9, 1.1, name + ("\nTop-K" if i in (0, 2) else ""), fc, ec=ec, fs=8)

    box(ax, (5.5, 5.2), 3.5, 0.9, "Mixture output\n$m = E^{(s)} + \\lambda \\sum_{j\\in\\mathcal{R}} g_j E_j^{(r)}$", "#D7BDE2", fs=8)

    # bias panel
    box(ax, (5.5, 0.4), 5.8, 1.6,
        "Loss-free bias update (DeepSeek-V3 style)\n"
        "Select: TopK($s_j + b_j$, K)   |   Mix weights: from unbiased $s_j$\n"
        "$b_j \\leftarrow b_j - \\gamma \\, \\mathrm{sign}(n_j - \\bar{n})$",
        "#FCF3CF", fs=8)

    arrow(ax, (2.2, 3.5), (2.8, 3.5))
    arrow(ax, (3.9, 5.2), (3.9, 4.0))
    arrow(ax, (5.0, 3.5), (5.15, 3.5))
    arrow(ax, (3.9, 5.65), (5.5, 5.65))
    arrow(ax, (8.0, 4.0), (7.2, 5.2))
    arrow(ax, (8.4, 2.9), (8.4, 2.0))

    ax.text(0.4, 6.4, "Always active", fontsize=8, color="#1E8449")
    ax.text(5.6, 4.2, "Only K routed experts fire per token", fontsize=8, color="#922B21")
    ax.text(0.4, 1.8, "Primary action loss untouched;\nbias only corrects routing assignment.", fontsize=8, color="#555")

    out = os.path.join(HERE, "moe_routing.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
