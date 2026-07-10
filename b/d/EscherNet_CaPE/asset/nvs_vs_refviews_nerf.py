#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EscherNet vs scene-specific neural rendering on NeRF-Synthetic.

Reproduces Tab. `tab:nerf2` (main paper) + the 6DoF row from `tab:nerf_6DoF`
(appendix): PSNR as a function of the number of reference views. The key
message is the CROSSOVER: with few reference views (<5) the generative
EscherNet wins; with many reference views (>10) the per-scene optimised
InstantNGP / 3D Gaussian Splatting overtake it and keep climbing while
EscherNet saturates.

Run: /mnt/r/VENV/dj/bin/python nvs_vs_refviews_nerf.py
Output: nvs_vs_refviews_nerf.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

REF = np.array([1, 2, 3, 5, 10, 20, 50, 100])
INGP = [10.92, 12.42, 14.27, 18.17, 22.96, 24.99, 26.86, 27.30]
GS = [9.44, 10.78, 12.87, 17.09, 23.04, 25.34, 26.98, 27.11]
ESCHER4 = [13.36, 14.95, 16.19, 17.16, 17.74, 17.91, 18.05, 18.15]
ESCHER6 = [13.73, 15.66, 16.91, 17.72, 18.47, 18.77, 19.24, 19.28]


def main():
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    ax.plot(REF, INGP, "-o", color="#7f7f7f", lw=2, label="InstantNGP (per-scene)")
    ax.plot(REF, GS, "-s", color="#555555", lw=2, label="3D Gaussian Splatting (per-scene)")
    ax.plot(REF, ESCHER4, "-^", color="#1f77b4", lw=2.4, label="EscherNet 4DoF (zero-shot)")
    ax.plot(REF, ESCHER6, "--D", color="#2ca02c", lw=2.0, label="EscherNet 6DoF (zero-shot)")

    # Shade the two regimes and mark the crossover band (~5-10 ref views)
    ax.axvspan(0.9, 5, color="#1f77b4", alpha=0.06)
    ax.axvspan(10, 110, color="#7f7f7f", alpha=0.06)
    ax.text(1.5, 25.4, "few views:\nEscherNet wins", color="#1f77b4", fontsize=10, weight="bold")
    ax.text(30, 14.5, "many views: per-scene\nmethods overtake", color="#555555",
            fontsize=10, weight="bold", ha="center")
    ax.annotate("crossover", xy=(6.5, 17.9), xytext=(6.5, 21.5), color="#d62728",
                fontsize=10, weight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))

    ax.set_xscale("log")
    ax.set_xticks(REF)
    ax.set_xticklabels([str(r) for r in REF])
    ax.set_xlabel("# Reference views (log scale)", fontsize=12)
    ax.set_ylabel("PSNR (dB)  $\\uparrow$", fontsize=12)
    ax.set_title("NeRF-Synthetic: generative EscherNet vs per-scene neural rendering\n"
                 "(EscherNet is zero-shot; InstantNGP / 3DGS optimise each scene)",
                 fontsize=12.5, weight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(8, 28.5)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvs_vs_refviews_nerf.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
