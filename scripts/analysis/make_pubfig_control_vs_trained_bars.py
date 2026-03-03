#!/usr/bin/env python3
"""Generate control vs trained PER% comparison bar plots for Hexanol and EB datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
try:
    from pptx import Presentation
except Exception:  # pragma: no cover
    Presentation = None
import numpy as np

ROOT = Path("/home/ramanlab/Documents/cole")
OUT_DIR = ROOT / "Data/imagesForFigures/imagesForOptoHEXTesting"

BAR_ORDER = [
    ("3-Octonol", "3-Octanol"),
    ("Apple Cider Vinegar", "ACV"),
    ("Benzaldehyde", "Benzaldehyde"),
    ("Citral", "Citral"),
    ("Ethyl Butyrate", "Ethyl Butyrate"),
    ("Hexanol", "Hexanol"),
    ("Linalool", "Linalool"),
]


def _register_arial() -> None:
    for font_path in [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/ariali.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbi.ttf",
    ]:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def plot_comparison(
    *,
    title: str,
    trained_values: dict[str, float],
    control_values: dict[str, float],
    trained_n: int,
    control_n: int,
    out_path: Path,
    trained_label: str,
    control_label: str,
) -> None:
    labels = [label for _, label in BAR_ORDER]
    trained = [float(trained_values.get(key, 0.0)) for key, _ in BAR_ORDER]
    control = [float(control_values.get(key, 0.0)) for key, _ in BAR_ORDER]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_control = ax.bar(
        x - width / 2,
        control,
        width,
        color="#9e9e9e",
        edgecolor="black",
        linewidth=0.8,
        label=f"{control_label} (n = {control_n})",
    )
    bars_trained = ax.bar(
        x + width / 2,
        trained,
        width,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
        label=f"{trained_label} (n = {trained_n})",
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("PER%", fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold", loc="center", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(loc="upper right", frameon=True, fontsize=14)

    _save_all(fig, out_path)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def _save_all(fig: plt.Figure, out_path: Path) -> None:
    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    svg_path = out_path.with_suffix(".svg")
    eps_path = out_path.with_suffix(".eps")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(svg_path, bbox_inches="tight")
    with matplotlib.rc_context(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 12,
            "ps.fonttype": 42,
            "ps.useafm": False,
        }
    ):
        fig.savefig(eps_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    _save_pptx(out_path, svg_path, out_path.with_suffix(".pptx"))


def _save_pptx(png_path: Path, svg_path: Path, pptx_path: Path) -> None:
    if Presentation is None:
        return
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_w = prs.slide_width
    slide_h = prs.slide_height

    img = Image.open(png_path)
    w_px, h_px = img.size
    emu_per_px = 914400 / 96.0
    img_w = w_px * emu_per_px
    img_h = h_px * emu_per_px
    max_w = slide_w * 0.95
    max_h = slide_h * 0.95
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    img_w *= scale
    img_h *= scale
    left = (slide_w - img_w) / 2
    top = (slide_h - img_h) / 2

    try:
        slide.shapes.add_picture(str(svg_path), left, top, width=img_w, height=img_h)
    except Exception:
        slide.shapes.add_picture(str(png_path), left, top, width=img_w, height=img_h)

    prs.save(str(pptx_path))


def main() -> None:
    _register_arial()
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "font.family": "Arial",
            "font.size": 14,
            "svg.fonttype": "none",
        }
    )

    # Hex trained vs Hex control
    hex_trained = {
        "Hexanol": 62.0,
        "Ethyl Butyrate": 35.0,
        "Apple Cider Vinegar": 13.0,
        "3-Octonol": 30.0,
        "Benzaldehyde": 5.0,
        "Citral": 5.0,
        "Linalool": 35.0,
    }
    hex_control = {
        "Hexanol": 11.0,
        "Apple Cider Vinegar": 8.0,
        "Benzaldehyde": 0.0,
        "3-Octonol": 16.0,
        "Ethyl Butyrate": 83.0,
        "Citral": 0.0,
        "Linalool": 33.0,
    }
    plot_comparison(
        title="PER% — Hexanol Trained vs Control",
        trained_values=hex_trained,
        control_values=hex_control,
        trained_n=20,
        control_n=6,
        out_path=OUT_DIR / "hex_trained_vs_hex_control_percents.png",
        trained_label="Trained",
        control_label="Control",
    )

    # EB trained vs EB control
    eb_trained = {
        "Hexanol": 50.0,
        "Ethyl Butyrate": 65.0,
        "Apple Cider Vinegar": 0.0,
        "3-Octonol": 32.0,
        "Benzaldehyde": 11.0,
        "Citral": 5.0,
        "Linalool": 16.0,
    }
    eb_control = {
        "Ethyl Butyrate": 50.0,
        "Hexanol": 39.0,
        "Apple Cider Vinegar": 0.0,
        "3-Octonol": 36.0,
        "Benzaldehyde": 21.0,
        "Citral": 14.0,
        "Linalool": 0.0,
    }
    plot_comparison(
        title="PER% — Ethyl Butyrate Trained vs Control",
        trained_values=eb_trained,
        control_values=eb_control,
        trained_n=19,
        control_n=14,
        out_path=OUT_DIR / "eb_trained_vs_eb_control_percents.png",
        trained_label="Trained",
        control_label="Control",
    )


if __name__ == "__main__":
    main()
