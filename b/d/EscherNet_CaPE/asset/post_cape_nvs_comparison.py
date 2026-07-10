#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-CaPE camera conditioning comparison: fixed vs. varying intrinsics.

Reproduces the key numbers from PRoPE (Li et al., "Cameras as Relative
Positional Encoding", NeurIPS 2025), Table 1 (constant intrinsics per scene)
and Table 2 (varying intrinsics per scene), for the LVSM backbone on
RealEstate10K and Objaverse. Metric: PSNR (dB), higher is better.

The key story this figure tells: CaPE (QK-only, SE(3)-only) and GTA
(QKV, SE(3)-only) both beat the absolute Plucker raymap baseline when
intrinsics are constant -- but COLLAPSE below the raymap baseline once
intrinsics vary across views (no focal-length/FOV information to condition
on). PRoPE, which additionally encodes the full camera frustum (intrinsics
+ extrinsics), is the only relative method that stays robust in both
regimes.

Run: /mnt/r/VENV/dj/bin/python post_cape_nvs_comparison.py
Output: post_cape_nvs_comparison.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

METHODS = ["Plucker Raymap\n(absolute, APE)", "CaPE\n(QK-only, SE(3))", "GTA\n(QKV, SE(3))", "PRoPE\n(QKV, full frustum)"]
COLORS = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#1f77b4"]

# PSNR (dB), from PRoPE paper Table 1 (fixed intrinsics) / Table 2 (varying intrinsics)
DATA = {
    "RealEstate10K": {"Fixed intrinsics": [20.48, 21.11, 22.51, 22.80], "Varying intrinsics": [19.89, 15.94, 15.77, 21.42]},
    "Objaverse": {"Fixed intrinsics": [21.44, 19.68, 23.70, 23.70], "Varying intrinsics": [21.43, 16.78, 18.00, 22.98]},
}


def plot_dataset(ax, dataset_name):
    groups = list(DATA[dataset_name].keys())
    n_methods = len(METHODS)
    x = np.arange(len(groups))
    width = 0.19

    for i, method in enumerate(METHODS):
        vals = [DATA[dataset_name][g][i] for g in groups]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=COLORS[i], label=method, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("PSNR (dB) $\\uparrow$", fontsize=11)
    ax.set_title(dataset_name, fontsize=12.5, weight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.set_ylim(0, 27.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    plot_dataset(axes[0], "RealEstate10K")
    plot_dataset(axes[1], "Objaverse")

    # Callout: CaPE / GTA collapse below the absolute raymap baseline once
    # intrinsics vary -- the central motivation for PRoPE.
    axes[0].annotate(
        "CaPE & GTA collapse\nbelow raymap baseline!",
        xy=(1.0 - 0.19 * 0.5, 15.94),
        xytext=(-0.15, 6.0),
        fontsize=9.5,
        color="#c0392b",
        weight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6),
    )
    axes[0].annotate(
        "PRoPE stays robust",
        xy=(1.0 + 0.19 * 1.5, 21.42),
        xytext=(1.62, 25.2),
        fontsize=9.5,
        color="#1f77b4",
        weight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.6),
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle(
        "Camera conditioning under fixed vs. varying intrinsics (LVSM backbone, real numbers from PRoPE paper Tab.1/2)",
        fontsize=12.5,
        weight="bold",
        y=1.14,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "post_cape_nvs_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
