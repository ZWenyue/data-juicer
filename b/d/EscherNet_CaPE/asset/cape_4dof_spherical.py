#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4DoF (object-centric) camera parameterisation used by 4DoF CaPE.

A camera looking at the object at the origin is described by 4 disentangled
numbers P = {alpha, beta, gamma, r}:
    alpha : azimuth   (rotation around the world +z axis)
    beta  : elevation (angle from the +z axis / equator)
    gamma : in-plane camera orientation around the look-at (optical) axis
    r     : radius    (camera distance to the object)

This 3D schematic draws the object, the viewing sphere, one camera on the
sphere, and annotates all four quantities.

Run: /mnt/r/VENV/dj/bin/python cape_4dof_spherical.py
Output: cape_4dof_spherical.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401
from mpl_toolkits.mplot3d.proj3d import proj_transform  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs2d, ys2d, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        return min(zs3d)


def av(ax, o, v, color, lw=2.2, ls="-", ms=14):
    ax.add_artist(Arrow3D([o[0], o[0] + v[0]], [o[1], o[1] + v[1]], [o[2], o[2] + v[2]],
                          mutation_scale=ms, lw=lw, arrowstyle="-|>", color=color, linestyle=ls))


def main():
    r = 1.0
    alpha = np.deg2rad(50)   # azimuth
    beta = np.deg2rad(35)    # elevation from equator
    # camera position on the sphere (elevation measured from xy-plane)
    cx = r * np.cos(beta) * np.cos(alpha)
    cy = r * np.cos(beta) * np.sin(alpha)
    cz = r * np.sin(beta)
    cam = np.array([cx, cy, cz])

    fig = plt.figure(figsize=(10, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    # viewing sphere (wireframe)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="#cccccc", lw=0.4, alpha=0.5)

    # world axes
    av(ax, [0, 0, 0], [1.35, 0, 0], "#555555", lw=1.5, ms=11)
    av(ax, [0, 0, 0], [0, 1.35, 0], "#555555", lw=1.5, ms=11)
    av(ax, [0, 0, 0], [0, 0, 1.35], "#555555", lw=1.5, ms=11)
    ax.text(1.4, 0.05, 0, "world x", color="#555555", fontsize=9)
    ax.text(-0.15, 1.42, 0, "world y", color="#555555", fontsize=9)
    ax.text(0, 0, 1.42, "world z (azimuth axis)", color="#555555", fontsize=9)

    # object at origin
    ax.scatter([0], [0], [0], color="#8c564b", s=120, marker="o")
    ax.text(0.02, 0.02, -0.16, "object", color="#8c564b", fontsize=10, weight="bold")

    # radius vector to camera (this line is also the look-at / optical axis)
    av(ax, [0, 0, 0], cam, "#1f77b4", lw=2.6, ms=15)
    ax.text(cam[0] * 0.5 + 0.02, cam[1] * 0.5 - 0.02, cam[2] * 0.5 - 0.16,
            "r (radius = optical axis)", color="#1f77b4", fontsize=9.5, weight="bold")

    # camera marker
    ax.scatter([cam[0]], [cam[1]], [cam[2]], color="#1f77b4", s=90, marker="s")
    ax.text(cam[0] + 0.03, cam[1] + 0.03, cam[2] + 0.05, "camera", color="#1f77b4",
            fontsize=10, weight="bold")

    # azimuth arc (in xy-plane from world-x to camera ground projection)
    ta = np.linspace(0, alpha, 30)
    ar = 0.55
    ax.plot(ar * np.cos(ta), ar * np.sin(ta), np.zeros_like(ta), color="#2ca02c", lw=2)
    ax.text(ar * np.cos(alpha / 2) + 0.05, ar * np.sin(alpha / 2), 0.02,
            r"$\alpha$ azimuth", color="#2ca02c", fontsize=10, weight="bold")
    # ground projection line
    ax.plot([0, cx], [0, cy], [0, 0], color="#2ca02c", ls="--", lw=1)
    ax.plot([cx, cx], [cy, cy], [0, cz], color="#999999", ls=":", lw=1)

    # elevation arc (vertical, from ground projection up to camera)
    tb = np.linspace(0, beta, 30)
    gx = np.cos(alpha) * (0.6 * np.cos(tb))
    gy = np.sin(alpha) * (0.6 * np.cos(tb))
    gz = 0.6 * np.sin(tb)
    ax.plot(gx, gy, gz, color="#9467bd", lw=2)
    ax.text(0.6 * np.cos(alpha) * np.cos(beta / 2) + 0.02,
            0.6 * np.sin(alpha) * np.cos(beta / 2),
            0.6 * np.sin(beta / 2) + 0.04, r"$\beta$ elevation", color="#9467bd",
            fontsize=10, weight="bold")

    # gamma: in-plane roll around the look-at axis (small circular arrow near camera)
    look = -cam / np.linalg.norm(cam)
    helper = np.array([0, 0, 1.0])
    right = np.cross(look, helper)
    right = right / np.linalg.norm(right)
    up = np.cross(look, right)
    tg = np.linspace(0, 1.6 * np.pi, 40)
    rg = 0.12
    circ = np.array([cam + rg * (np.cos(t) * right + np.sin(t) * up) for t in tg])
    ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], color="#ff7f0e", lw=2)
    ax.text(cam[0] + 0.05, cam[1] - 0.12, cam[2] + 0.14, r"$\gamma$ camera roll",
            color="#ff7f0e", fontsize=10, weight="bold")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.6, 1.3)
    ax.set_box_aspect((1, 1, 0.9))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    ax.set_title("4DoF object-centric camera pose  P = {alpha, beta, gamma, r}\n"
                 "(the 4 disentangled numbers encoded by 4DoF CaPE)",
                 fontsize=12.5, weight="bold", pad=12)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cape_4dof_spherical.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
