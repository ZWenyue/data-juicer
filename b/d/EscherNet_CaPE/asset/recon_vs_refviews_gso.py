#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single/multi-image 3D reconstruction on GSO (Tab. `tab:3D`, main paper).

Left  : Chamfer Distance (lower is better) vs # reference views.
Right : Volume IoU (higher is better) vs # reference views.

EscherNet (4DoF) is drawn as a line over 1/2/3/5/10 reference views and
compared against NeuS (3/5/10 views) and a cloud of single-reference-view
image-to-3D baselines (Point-E, Shape-E, One-2-3-45(-XL), DreamGaussian(-XL),
SyncDreamer) shown as scatter points at x=1.

Run: /mnt/r/VENV/dj/bin/python recon_vs_refviews_gso.py
Output: recon_vs_refviews_gso.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

ESCHER_REF = np.array([1, 2, 3, 5, 10])
ESCHER_CD = [0.0314, 0.0215, 0.0190, 0.0175, 0.0167]
ESCHER_IOU = [0.5974, 0.6868, 0.7189, 0.7423, 0.7478]

NEUS_REF = np.array([3, 5, 10])
NEUS_CD = [0.0366, 0.0245, 0.0195]
NEUS_IOU = [0.5352, 0.6742, 0.7264]

# single-reference-view baselines: name -> (chamfer, iou, cd_label_dy, iou_label_dy)
BASE1 = {
    "Point-E": (0.0447, 0.2503, 0.0011, 0.0),
    "Shape-E": (0.0448, 0.3762, -0.0013, 0.012),
    "One2345": (0.0632, 0.4209, 0.0, 0.0),
    "One2345-XL": (0.0667, 0.4016, 0.0, 0.0),
    "DreamGaussian": (0.0605, 0.3757, 0.0, -0.016),
    "DreamGaussian-XL": (0.0459, 0.4531, 0.0013, 0.0),
    "SyncDreamer": (0.0400, 0.5220, 0.0, 0.0),
}


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Chamfer (lower better) ---
    axL.plot(ESCHER_REF, ESCHER_CD, "-^", color="#1f77b4", lw=2.6, ms=8, label="EscherNet (4DoF)")
    axL.plot(NEUS_REF, NEUS_CD, "-s", color="#2ca02c", lw=2.0, ms=7, label="NeuS (multi-view fit)")
    for name, (cd, _, cd_dy, _) in BASE1.items():
        mk = "*" if name == "SyncDreamer" else "o"
        col = "#d62728" if name == "SyncDreamer" else "#999999"
        axL.scatter([1], [cd], marker=mk, s=(150 if name == "SyncDreamer" else 45), color=col, zorder=5)
        axL.annotate(name, xy=(1, cd), xytext=(1.15, cd + cd_dy), fontsize=7.5,
                     color=col, va="center")
    axL.set_xlabel("# Reference views", fontsize=12)
    axL.set_ylabel("Chamfer Distance  $\\downarrow$", fontsize=12)
    axL.set_title("3D reconstruction error (GSO)\nlower is better", fontsize=12.5, weight="bold")
    axL.set_xticks([1, 2, 3, 5, 10])
    axL.grid(True, ls="--", alpha=0.35)
    axL.legend(fontsize=10, loc="upper right")

    # --- IoU (higher better) ---
    axR.plot(ESCHER_REF, ESCHER_IOU, "-^", color="#1f77b4", lw=2.6, ms=8, label="EscherNet (4DoF)")
    axR.plot(NEUS_REF, NEUS_IOU, "-s", color="#2ca02c", lw=2.0, ms=7, label="NeuS (multi-view fit)")
    for name, (_, iou, _, iou_dy) in BASE1.items():
        mk = "*" if name == "SyncDreamer" else "o"
        col = "#d62728" if name == "SyncDreamer" else "#999999"
        axR.scatter([1], [iou], marker=mk, s=(150 if name == "SyncDreamer" else 45), color=col, zorder=5)
        axR.annotate(name, xy=(1, iou), xytext=(1.15, iou + iou_dy), fontsize=7.5, color=col, va="center")
    axR.set_xlabel("# Reference views", fontsize=12)
    axR.set_ylabel("Volume IoU  $\\uparrow$", fontsize=12)
    axR.set_title("3D reconstruction overlap (GSO)\nhigher is better", fontsize=12.5, weight="bold")
    axR.set_xticks([1, 2, 3, 5, 10])
    axR.grid(True, ls="--", alpha=0.35)
    axR.legend(fontsize=10, loc="lower right")

    fig.suptitle("EscherNet 3D reconstruction scales cleanly with reference views "
                 "(single-ref baselines shown at x=1)", fontsize=13.5, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon_vs_refviews_gso.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
