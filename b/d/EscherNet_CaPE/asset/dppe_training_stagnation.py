#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPPE fixes PRoPE's late-stage training stagnation.

Reproduces the real per-checkpoint PSNR numbers from DPPE (Kenney & Suzuki,
"DPPE: Rethinking Camera-Based Positional Encoding for Scaling Multi-View
Transformers", 2026), Table 6, on MVImgNet2 (NVS task, large-scale training
run, 320k iterations). GTA and PRoPE both peak early (~184k-200k) and then
DEGRADE by 320k, because rotation R and translation t are coupled in the
same value-output dimensions, which the paper proves is non-identifiable
per-token (Prop. 1). DPPE (both variants) is only reported at the final
320k checkpoint in the paper (no intermediate curve was released), but its
value already exceeds BOTH baselines' peak checkpoints -- so it is plotted
as isolated endpoint markers, not a fabricated curve.

Run: /mnt/r/VENV/dj/bin/python dppe_training_stagnation.py
Output: dppe_training_stagnation.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

# Real checkpoint data, DPPE paper Table 6 (MVImgNet2, PSNR dB)
GTA_ITERS = [188_000, 196_000, 240_000, 320_000]
GTA_PSNR = [23.65, 23.60, 23.46, 23.39]

PROPE_ITERS = [184_000, 200_000, 240_000, 320_000]
PROPE_PSNR = [23.67, 23.61, 23.61, 23.14]

# DPPE variants: only the final 320k checkpoint is reported in the paper.
DPPE_DUAL_320K = 23.92
DPPE_TADD_320K = 23.98


def main():
    fig, ax = plt.subplots(figsize=(9.6, 6.2))

    ax.plot(GTA_ITERS, GTA_PSNR, "-o", color="#2ca02c", lw=2.2, ms=6, label="GTA (SE(3), QKV)")
    ax.plot(PROPE_ITERS, PROPE_PSNR, "-s", color="#1f77b4", lw=2.2, ms=6, label="PRoPE (full frustum, QKV)")

    ax.plot(320_000, DPPE_DUAL_320K, "*", color="#d62728", ms=20, mec="white", mew=0.8, label="DPPEdual @320k (reported endpoint only)")
    ax.plot(320_000, DPPE_TADD_320K, "*", color="#9467bd", ms=20, mec="white", mew=0.8, label="DPPEtAdd @320k (reported endpoint only)")

    # Peak markers
    ax.annotate("peak @188k\nthen degrades", xy=(188_000, 23.65), xytext=(168_000, 22.55),
                fontsize=9, color="#2ca02c", weight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.4))
    ax.annotate("peak @184k, then\nfalls off a cliff by 320k", xy=(184_000, 23.67), xytext=(210_000, 24.35),
                fontsize=9, color="#1f77b4", weight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.4))
    ax.annotate("DPPE variants exceed BOTH\nbaselines' peak -- no degradation observed",
                xy=(316_000, 23.93), xytext=(232_000, 23.05),
                fontsize=9.3, color="#333333", weight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.3))

    ax.set_xlabel("Training iteration", fontsize=11.5)
    ax.set_ylabel("PSNR (dB) $\\uparrow$  on MVImgNet2", fontsize=11.5)
    ax.set_title(
        "Late-stage training stagnation in PRoPE/GTA, and how DPPE avoids it\n"
        "(real checkpoint numbers from DPPE paper Tab.6; DPPE curve not fabricated -- only its 320k endpoint is reported)",
        fontsize=11.5, weight="bold",
    )
    ax.set_xlim(160_000, 340_000)
    ax.set_ylim(22.8, 24.6)
    ax.xaxis.set_major_formatter(lambda v, pos: f"{int(v/1000)}k")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=9.3, loc="lower left")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dppe_training_stagnation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
