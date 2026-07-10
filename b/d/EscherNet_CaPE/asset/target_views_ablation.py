#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect of the number of *target* views on view-synthesis quality.

Reproduces the four ablation sub-plots (`images/ablation{1,5,10,20}.tex`):
for a fixed pre-selected single target view, generating additional
(duplicate) target views jointly reduces diffusion stochasticity and raises
PSNR. The paper marks >=15 target views as the sweet spot (orange), beyond
which gains are marginal.

Run: /mnt/r/VENV/dj/bin/python target_views_ablation.py
Output: target_views_ablation.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

X = np.array([1, 3, 5, 10, 15, 20, 25, 30, 50, 100])
DATA = {
    "1 reference view": [14.35, 16.32, 17.25, 17.49, 17.35, 17.42, 16.71, 16.54, 16.73, 17.12],
    "5 reference views": [21.78, 22.62, 22.59, 23.00, 23.04, 22.99, 22.96, 22.95, 22.95, 23.14],
    "10 reference views": [22.55, 22.94, 23.13, 23.45, 23.53, 23.53, 23.48, 23.43, 23.42, 23.59],
    "20 reference views": [23.06, 23.63, 23.60, 23.59, 23.87, 23.94, 23.84, 23.83, 23.89, 24.04],
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
SWEET = 15


def main():
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    for (label, ys), c in zip(DATA.items(), COLORS):
        ys = np.array(ys)
        ax.plot(X, ys, "-o", color=c, lw=2, ms=5, label=label)
        idx = list(X).index(SWEET)
        ax.scatter([SWEET], [ys[idx]], s=120, facecolor="orange", edgecolor=c, zorder=6, lw=1.5)

    ax.axvline(SWEET, color="orange", ls="--", lw=1.6)
    ax.text(SWEET + 1, 14.8, "sweet spot:\n>= 15 target views", color="#cc7000",
            fontsize=10.5, weight="bold")

    ax.set_xscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels([str(x) for x in X])
    ax.set_xlabel("# Target views generated jointly (log scale)", fontsize=12)
    ax.set_ylabel("PSNR (dB)  $\\uparrow$  on one fixed target view", fontsize=12)
    ax.set_title("More jointly-generated target views reduce diffusion stochasticity\n"
                 "(GSO; gains saturate beyond ~15 target views)", fontsize=12.5, weight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(fontsize=10, loc="lower right", ncol=2)
    ax.set_ylim(13, 25)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_views_ablation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
