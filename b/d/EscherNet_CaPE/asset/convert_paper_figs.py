#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert selected EscherNet paper figures (PDF) to PNG for embedding in note.md.

Uses PyMuPDF (fitz), which is self-contained and does not require a system
Ghostscript / poppler install.

Run:
    /mnt/r/VENV/dj/bin/python convert_paper_figs.py
"""
import os

import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "TeX_Source", "images"))

# (source pdf relative to SRC, output png name, zoom/dpi scale)
JOBS = [
    ("teaser.pdf", "paper_teaser.png", 2.2),
    ("eschernet_arch.pdf", "paper_arch.png", 3.0),
    ("0123_clean.pdf", "paper_repr_zero123.png", 3.0),
    ("eschernet_clean.pdf", "paper_repr_eschernet.png", 3.0),
    ("nerf_clean.pdf", "paper_repr_nerf.png", 3.0),
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
        # white background to avoid transparent-black rendering in viewers
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(out_path)
        print(f"[ok] {src_name} -> {out_name}  ({pix.width}x{pix.height})")
        doc.close()


if __name__ == "__main__":
    main()
