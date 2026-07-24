"""Methods figures: YOLO keypoints -> per-frame (r, theta) -> per-trial metrics.

Two standalone figures:
  yolo_geometry_diagram    the geometry per frame (eye = class 0, proboscis = class 1).
  yolo_geometry_equations  the per-frame equations + per-fly normalisation of r.

Equations mirror scripts/analysis/geom_features.py (enrich_trial / compute_fly_stats):
  r = sqrt(dx^2 + dy^2)                                       (line ~798)
  r_pct_robust = 100 * (r - r_p01) / (r_p99 - r_p01 + 1e-6)   (lines ~808-809, clip[0,100])

Run:
    python scripts/analysis/yolo_geometry_figure.py
Outputs (each PNG 300 dpi + vector PDF):
    figures/yolo_geometry_diagram.png / .pdf
    figures/yolo_geometry_equations.png / .pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parents[2] / "figures"
SOURCE = "source: scripts/analysis/geom_features.py · enrich_trial()"

# entity colours (blue/orange = a classic CVD-safe pair; text labels also carry
# identity, so colour is never the sole channel)
EYE = "#1f77b4"        # class 0
PROB = "#d9600b"       # class 1
DX_C = "#1b9e77"       # Delta x
DY_C = "#c0489a"       # Delta y
R_C = "#26262b"        # r vector
TH_C = "#e08214"       # theta arc
INK = "#1a1a1a"
MUTED = "#666666"

EYE_XY = (3.8, 3.0)
PROB_XY = (1.2, 1.4)


# --------------------------------------------------------------------------- #
# Diagram
# --------------------------------------------------------------------------- #
def draw_diagram(ax):
    (xe, ye), (xp, yp) = EYE_XY, PROB_XY

    ax.plot([xe, xe + 1.7], [ye, ye], ls=(0, (1, 2)), color=MUTED, lw=1.1, zorder=1)

    ax.plot([xe, xp], [ye, ye], ls=(0, (5, 3)), color=DX_C, lw=2.0, zorder=2)
    ax.plot([xp, xp], [ye, yp], ls=(0, (5, 3)), color=DY_C, lw=2.0, zorder=2)
    ax.text((xe + xp) / 2, ye + 0.16, r"$\Delta x = x_p - x_e$", color=DX_C,
            ha="center", va="bottom", fontsize=12.5)
    ax.text(xp - 0.12, (ye + yp) / 2, r"$\Delta y = y_p - y_e$", color=DY_C,
            ha="right", va="center", fontsize=12.5, rotation=90)

    ax.add_patch(FancyArrowPatch(
        (xe, ye), (xp, yp), arrowstyle="-|>", mutation_scale=22,
        lw=2.6, color=R_C, zorder=4, shrinkA=9, shrinkB=9))
    ax.text((xe + xp) / 2 + 0.22, (ye + yp) / 2 - 0.12, r"$r$", color=R_C,
            ha="left", va="top", fontsize=17, fontweight="bold")

    theta = np.degrees(np.arctan2(yp - ye, xp - xe))
    ax.add_patch(Arc((xe, ye), 1.7, 1.7, angle=0, theta1=theta, theta2=0,
                     color=TH_C, lw=2.4, zorder=3))
    mid = np.radians(theta / 2)
    ax.text(xe + 1.02 * np.cos(mid), ye + 1.02 * np.sin(mid), r"$\theta$",
            color=TH_C, ha="center", va="center", fontsize=17, fontweight="bold")

    for (x, y), c in [(EYE_XY, EYE), (PROB_XY, PROB)]:
        ax.plot(x, y, "o", ms=15, color=c, mec="white", mew=1.8, zorder=6)
    ax.annotate("eye  (class 0)\n$(x_e,\\ y_e)$", EYE_XY, (xe + 0.28, ye + 0.42),
                color=EYE, fontsize=12.5, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=EYE, lw=1.0))
    ax.annotate("proboscis  (class 1)\n$(x_p,\\ y_p)$", PROB_XY, (xp - 0.05, yp - 0.55),
                color=PROB, fontsize=12.5, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=PROB, lw=1.0))

    ax.text(0.02, 1.03, "image-pixel coords from YOLO bounding-box centres",
            transform=ax.transAxes, fontsize=10, style="italic", color=MUTED)
    ax.set_xlim(-0.2, 5.7)
    ax.set_ylim(-0.2, 4.5)
    ax.set_aspect("equal")
    ax.set_xlabel("pixel  $x$", fontsize=11.5, color="#333333")
    ax.set_ylabel("pixel  $y$", fontsize=11.5, color="#333333")
    ax.set_xticks(range(0, 6))
    ax.set_yticks(np.arange(0, 4.5, 0.5))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bcbcbc")
    ax.tick_params(colors=MUTED, labelsize=9.5)


# --------------------------------------------------------------------------- #
# Equations (per-frame + per-fly normalisation)
# --------------------------------------------------------------------------- #
def draw_equations(ax):
    ax.set_axis_off()
    blocks = [
        ("head", "Inputs"),
        ("eq",   r"$(x_e,\ y_e)=$ eye keypoint  (class 0)"),
        ("eq",   r"$(x_p,\ y_p)=$ proboscis keypoint  (class 1)"),
        ("head", "Displacement Vector"),
        ("eq",   r"$\Delta x = x_p - x_e \qquad \Delta y = y_p - y_e$"),
        ("head", "Distance — Proboscis Extension (px)"),
        ("big",  r"$r = \sqrt{\Delta x^{2} + \Delta y^{2}}$"),
        ("head", "Angle & Proboscis Direction"),
        ("big",  r"$\cos\theta = \dfrac{\Delta x}{r+\varepsilon} \qquad "
                 r"\sin\theta = \dfrac{\Delta y}{r+\varepsilon}$"),
        ("eq",   r"$\theta = \mathrm{atan2}(\Delta y,\ \Delta x)"
                 r"\qquad \varepsilon = 10^{-6}$"),
        ("head", "Per Fly Normalization"),
        ("eq",   r"$r_{\min},\ r_{\max},\ r_{p01},\ r_{p99},\ \mu_r,\ \sigma_r$"),
        ("head", "Robust Percent Extension"),
        ("big",  r"$r_{\%}^{\mathrm{robust}} = 100 \cdot "
                 r"\dfrac{r - r_{p01}}{r_{p99} - r_{p01} + \varepsilon}"
                 r"\ \ \mathrm{clip}\,[0,100]$"),
    ]
    gap = {"head": 0.050, "eq": 0.049, "big": 0.066}
    pre = {"head": 0.028, "eq": 0.0, "big": 0.008}
    y = 0.955
    for kind, text in blocks:
        y -= pre[kind]
        if kind == "head":
            ax.text(0.05, y, text, transform=ax.transAxes, fontsize=13,
                    fontweight="bold", color=INK, va="top")
        else:
            ax.text(0.11, y, text, transform=ax.transAxes,
                    fontsize=15 if kind == "big" else 13, color="#222222", va="top")
        y -= gap[kind]


def _footer(fig):
    fig.text(0.975, 0.02, SOURCE, ha="right", va="center", fontsize=9,
             style="italic", color=MUTED)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"})

    # Figure 1 -- geometry diagram
    fig = plt.figure(figsize=(7.8, 6.8))
    draw_diagram(fig.add_axes([0.11, 0.10, 0.85, 0.74]))
    fig.suptitle("YOLO Keypoints → Per-Frame Geometry", x=0.5, y=0.955,
                 fontsize=16, fontweight="bold", color=INK)
    _footer(fig)
    _save(fig, "yolo_geometry_diagram")

    # Figure 2 -- equations + normalisation
    fig = plt.figure(figsize=(8.6, 9.4))
    draw_equations(fig.add_axes([0.04, 0.03, 0.92, 0.905]))
    fig.suptitle("Per-Frame Equations & Normalization", x=0.5, y=0.975,
                 fontsize=16, fontweight="bold", color=INK)
    _footer(fig)
    _save(fig, "yolo_geometry_equations")


if __name__ == "__main__":
    main()
