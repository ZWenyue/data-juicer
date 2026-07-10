#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert selected LingBot-VLA 2.0 paper figures (PDF) to PNG for embedding in note.md.

Uses PyMuPDF (fitz).

Run:
    python3 convert_paper_figs.py
"""
import os

import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "TeX_Source", "figures"))

# (source pdf relative to SRC, output png name, zoom/dpi scale)
JOBS = [
    ("framework.pdf", "paper_framework.png", 2.5),
    ("data_demo.pdf", "paper_data_demo.png", 2.2),
    ("data_process.pdf", "paper_data_process.png", 2.5),
    ("data_dimension.pdf", "paper_data_dimension.png", 2.5),
    ("loss_mse_comparison.pdf", "paper_loss_mse.png", 2.5),
    ("gm100_ablation_barplot.pdf", "paper_gm100_ablation.png", 2.5),
    ("vis_distillation.pdf", "paper_vis_distillation.png", 2.5),
    ("mobile_BM_embodiment.pdf", "paper_mobile_bm.png", 2.5),
    ("bm_subtask_progress_domain_bar.pdf", "paper_bm_subtask.png", 2.5),
    ("fig_action_space_boxalign.pdf", "paper_action_space_boxalign.png", 2.2),
    ("fig_ac_seaborn_allstyle.pdf", "paper_action_norm_stats.png", 2.2),
    ("Data_Annotation/subtask_duration_stats.pdf", "paper_subtask_stats.png", 2.2),
    ("Data_Annotation/subtask_vocab_objects.pdf", "paper_object_cloud.png", 2.2),
]


def main():
    for src_name, out_name, zoom in JOBS:
        src_path = os.path.join(SRC, src_name)
        out_path = os.path.join(HERE, out_name)
        if not os.path.exists(src_path):
            print(f"[skip] missing {src_path}")
            continue
        doc = fitz.open(src_path)
        page = doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(out_path)
        print(f"[ok] {src_name} -> {out_name}  ({pix.width}x{pix.height})")
        doc.close()


if __name__ == "__main__":
    main()
