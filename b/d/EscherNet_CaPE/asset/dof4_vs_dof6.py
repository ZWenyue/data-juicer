#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4DoF vs 6DoF CaPE ablation (appendix Tab. `tab:NVS_6DoF`, `tab:3D_6DoF`).

Left  : novel-view-synthesis PSNR on GSO-30 and RTMV vs # reference views.
Right : 3D reconstruction (Chamfer / IoU) on GSO vs # reference views.

Take-away: 6DoF CaPE is consistently a touch better numerically (more
compact / expressive pose representation), even though the paper reports
4DoF is visually more consistent on real-world images (hence 4DoF is used
in the main paper).

Run: /mnt/r/VENV/dj/bin/python dof4_vs_dof6.py
Output: dof4_vs_dof6.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

REF = np.array([1, 2, 3, 5, 10])

GSO_4 = [20.24, 22.91, 24.09, 25.09, 25.90]
GSO_6 = [20.89, 23.92, 25.21, 26.59, 27.75]
RTMV_4 = [10.56, 12.66, 13.59, 14.52, 15.55]
RTMV_6 = [12.30, 14.18, 15.06, 15.71, 16.58]

CD_4 = [0.0314, 0.0215, 0.0190, 0.0175, 0.0167]
CD_6 = [0.0274, 0.0196, 0.0180, 0.0176, 0.0160]
IOU_4 = [0.5974, 0.6868, 0.7189, 0.7423, 0.7478]
IOU_6 = [0.6382, 0.7100, 0.7348, 0.7392, 0.7628]


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PSNR on GSO and RTMV
    axL.plot(REF, GSO_4, "--^", color="#1f77b4", lw=2, ms=7, label="GSO  4DoF")
    axL.plot(REF, GSO_6, "-^", color="#1f77b4", lw=2.6, ms=7, label="GSO  6DoF")
    axL.plot(REF, RTMV_4, "--s", color="#ff7f0e", lw=2, ms=7, label="RTMV 4DoF")
    axL.plot(REF, RTMV_6, "-s", color="#ff7f0e", lw=2.6, ms=7, label="RTMV 6DoF")
    axL.set_xlabel("# Reference views", fontsize=12)
    axL.set_ylabel("PSNR (dB)  $\\uparrow$", fontsize=12)
    axL.set_title("Novel view synthesis: 6DoF >= 4DoF\n(solid = 6DoF, dashed = 4DoF)",
                  fontsize=12.5, weight="bold")
    axL.set_xticks(REF)
    axL.grid(True, ls="--", alpha=0.35)
    axL.legend(fontsize=9.5, loc="lower right", ncol=2)

    # Right: Chamfer (left axis) and IoU (right axis)
    axR.plot(REF, CD_4, "--^", color="#d62728", lw=2, ms=7, label="Chamfer 4DoF")
    axR.plot(REF, CD_6, "-^", color="#d62728", lw=2.6, ms=7, label="Chamfer 6DoF")
    axR.set_xlabel("# Reference views", fontsize=12)
    axR.set_ylabel("Chamfer Distance  $\\downarrow$", color="#d62728", fontsize=12)
    axR.tick_params(axis="y", labelcolor="#d62728")
    axR.set_xticks(REF)
    axR.grid(True, ls="--", alpha=0.35)

    axR2 = axR.twinx()
    axR2.plot(REF, IOU_4, "--o", color="#2ca02c", lw=2, ms=7, label="IoU 4DoF")
    axR2.plot(REF, IOU_6, "-o", color="#2ca02c", lw=2.6, ms=7, label="IoU 6DoF")
    axR2.set_ylabel("Volume IoU  $\\uparrow$", color="#2ca02c", fontsize=12)
    axR2.tick_params(axis="y", labelcolor="#2ca02c")

    lines1, labels1 = axR.get_legend_handles_labels()
    lines2, labels2 = axR2.get_legend_handles_labels()
    axR.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    axR.set_title("3D reconstruction: 6DoF edges out 4DoF\n(solid = 6DoF, dashed = 4DoF)",
                  fontsize=12.5, weight="bold")

    fig.suptitle("4DoF vs 6DoF CaPE: 6DoF is slightly better numerically; "
                 "4DoF is visually more robust on real images",
                 fontsize=13, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dof4_vs_dof6.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
