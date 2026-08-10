"""Methods figure: preparing a tethered fly for automated PER conditioning.

A 3x2 photo storyboard, one panel per mounting step:

  A  Cold Anesthesia On Abdomen     fly chilled, dorsum up, thorax exposed
  B  UV Glue Application to Thorax  glue bead carried on an insect pin
  C  Tethered & Inverted            fly on the pin, legs still mobile
  D  Tarsal Immobilization          tarsi sunk into an adhesive bolus
  E  Head Fixation                  head glued, antennae and proboscis free
  F  Fly Mounted                    finished preparation under UV/IR

Panels A-E are rebuilt from the raw stereoscope photographs (``1.png`` ...
``5.png``).  Panel F is the archived frame extracted from the original thesis
PDF and is reproduced unchanged -- it is the only panel with no replacement
photograph.

Each photograph is cropped to the largest square that still contains the fly
(see ``square_crop_box``) and given a gentle percentile contrast stretch, so
the animal reads clearly against the pale stereoscope background.

Run:
    python scripts/analysis/fly_mounting_workflow_figure.py
    python scripts/analysis/fly_mounting_workflow_figure.py \
        --photo-dir ~/Documents/cole/Results --out figures/fig_fly_mounting_workflow
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_F_ASSET = Path(__file__).resolve().parent / "assets" / "fly_mounting_panel_f_uv.jpg"
DEFAULT_PHOTO_DIR = Path.home() / "Documents" / "cole" / "Results"
DEFAULT_OUT = REPO_ROOT / "figures" / "fig_fly_mounting_workflow"

# --- palette, carried over from the published figure -----------------------
HEADER_BG = "#1f2933"
HEADER_FG = "#ffffff"
PANEL_EDGE = "#2b333d"
CONNECTOR = "#5a6675"
AMBER = "#e8a33d"
CYAN = "#2fbfd6"
LABEL_FG = "#111418"
LABEL_BG = "#ffffff"
LABEL_ALPHA = 0.78

FONT_STACK = ["Liberation Sans", "Arial", "DejaVu Sans"]

# keep every label editable in Illustrator / Inkscape, as the published figure was
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})

# --- page geometry, in figure fractions (510.24 x 376.59 pt page) ----------
FIG_SIZE_IN = (7.0866, 5.2304)
PANEL_W = 0.290688          # 148.32 pt
PANEL_H = 0.393850          # 148.32 pt -> square on the page
HEADER_H = 0.051176         # 19.27 pt
PANEL_X = (0.012, 0.354672, 0.697333)
PANEL_Y = (0.535000, 0.020000)   # bottom edge of the image axes, per row


@dataclass(frozen=True)
class Annotation:
    """A leader line from a text label to a feature in the photograph.

    ``xy`` (arrow tip) and ``xytext`` (label anchor) are both in panel axes
    fractions, with (0, 0) at the bottom-left of the cropped photograph.
    """

    text: str
    xy: tuple[float, float]
    xytext: tuple[float, float]
    color: str
    ha: str = "left"
    va: str = "center"
    rad: float = 0.25


@dataclass(frozen=True)
class PanelSpec:
    letter: str
    title: str
    source: str
    crop: tuple[float, float, float, float]  # normalised (x0, y0, x1, y1) ROI
    annotations: tuple[Annotation, ...] = ()
    badge: str | None = None
    enhance: bool = True
    contrast: float = 1.0


PANELS: tuple[PanelSpec, ...] = (
    PanelSpec(
        letter="A",
        title="Cold Anesthesia On Abdomen",
        source="1.png",
        crop=(0.0, 0.0, 1.0, 1.0),
        annotations=(
            Annotation(
                "Thorax\nAccessible",
                xy=(0.30, 0.72),
                xytext=(0.07, 0.13),
                color=AMBER,
                rad=-0.28,
            ),
        ),
    ),
    PanelSpec(
        letter="B",
        title="UV Glue Application to Thorax",
        source="2.png",
        crop=(0.06, 0.04, 0.84, 0.92),
        annotations=(
            Annotation(
                "UV-Curable Glue\nOn Insect Pin",
                xy=(0.30, 0.57),
                xytext=(0.34, 0.11),
                color=AMBER,
                ha="center",
                va="bottom",
                rad=-0.24,
            ),
        ),
    ),
    PanelSpec(
        letter="C",
        title="Tethered & Inverted",
        source="3.png",
        crop=(0.0, 0.0, 1.0, 0.72),
        annotations=(
            Annotation(
                "Legs Not Yet\nImmobilized",
                xy=(0.26, 0.60),
                xytext=(0.03, 0.95),
                color=CYAN,
                va="top",
                rad=0.28,
            ),
        ),
    ),
    PanelSpec(
        letter="D",
        title="Tarsal Immobilization",
        source="4.png",
        crop=(0.0, 0.0, 1.0, 1.0),
        annotations=(
            Annotation(
                "Tarsi Bonded\nVia UV-Glue",
                xy=(0.15, 0.67),
                xytext=(0.04, 0.14),
                color=AMBER,
                va="bottom",
                rad=-0.30,
            ),
        ),
    ),
    PanelSpec(
        letter="E",
        title="Head Fixation",
        source="5.png",
        crop=(0.02, 0.0, 0.82, 1.0),
        annotations=(
            Annotation(
                "Antenna's Free",
                xy=(0.53, 0.85),
                xytext=(0.03, 0.96),
                color=CYAN,
                va="top",
                rad=0.26,
            ),
            # The published figure also carried a "Proboscis Free" callout, but
            # both replacement photographs are dorso-lateral views in which the
            # proboscis is not resolved, so there is nothing honest to point at.
        ),
    ),
    PanelSpec(
        letter="F",
        title="Fly Mounted",
        source=PANEL_F_ASSET.name,
        crop=(0.0, 0.0, 1.0, 1.0),
        badge="IR Illumination",
        enhance=False,
    ),
)


# ---------------------------------------------------------------------------
# image preparation
# ---------------------------------------------------------------------------
def square_crop_box(
    width: int, height: int, roi: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    """Largest square crop covering ``roi``, clamped to the image.

    ``roi`` is ``(x0, y0, x1, y1)`` in fractions of the image, with the origin
    at the top-left (PIL convention).  The square is centred on the ROI and
    slid back inside the frame if that would push it over an edge; when the ROI
    is bigger than the short edge the square shrinks to the short edge.
    """
    x0, y0, x1, y1 = roi
    left_px, right_px = x0 * width, x1 * width
    top_px, bottom_px = y0 * height, y1 * height

    side = min(max(right_px - left_px, bottom_px - top_px), width, height)
    cx = (left_px + right_px) / 2.0
    cy = (top_px + bottom_px) / 2.0

    left = cx - side / 2.0
    top = cy - side / 2.0
    left = min(max(left, 0.0), width - side)
    top = min(max(top, 0.0), height - side)

    left_i, top_i = int(round(left)), int(round(top))
    side_i = int(round(side))
    return left_i, top_i, left_i + side_i, top_i + side_i


def stretch_contrast(
    img: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.5,
    strength: float = 1.0,
) -> np.ndarray:
    """Per-channel percentile contrast stretch (grey-world white balance).

    The stereoscope photographs carry a heavy blue-cyan cast -- the red channel
    tops out near 0.65 while blue reaches 0.90 -- so a shared stretch just
    amplifies the cast and the fly stays buried in teal.  Stretching each
    channel onto its own percentile range neutralises the background and lifts
    the animal off it.  ``strength`` blends between the original (0.0) and the
    full stretch (1.0).
    """
    arr = np.asarray(img, dtype=float)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if strength <= 0.0:
        return arr

    axes = (0, 1) if arr.ndim == 3 else None
    lo = np.percentile(arr, low_pct, axis=axes, keepdims=True)
    hi = np.percentile(arr, high_pct, axis=axes, keepdims=True)
    span = hi - lo
    # channels with no dynamic range are passed through untouched
    safe = np.where(span < 1e-6, 1.0, span)
    stretched = np.where(span < 1e-6, arr, np.clip((arr - lo) / safe, 0.0, 1.0))

    out = (1.0 - strength) * arr + strength * stretched
    return np.clip(out, 0.0, 1.0)


def prepare_panel_image(path: Path, panel: PanelSpec) -> np.ndarray:
    """Load, square-crop and (optionally) contrast-stretch one panel photo."""
    if not Path(path).exists():
        raise FileNotFoundError(f"panel {panel.letter} photograph not found: {path}")
    with Image.open(path) as im:
        im = im.convert("RGB")
        box = square_crop_box(im.width, im.height, panel.crop)
        arr = np.asarray(im.crop(box), dtype=float) / 255.0
    if panel.enhance:
        arr = stretch_contrast(arr, strength=panel.contrast)
    return arr


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
def panel_rect(index: int) -> tuple[float, float, float, float]:
    """``(left, bottom, width, height)`` of panel ``index``'s image axes."""
    row, col = divmod(index, 3)
    return (PANEL_X[col], PANEL_Y[row], PANEL_W, PANEL_H)


def header_rect(index: int) -> tuple[float, float, float, float]:
    left, bottom, width, _ = panel_rect(index)
    return (left, bottom + PANEL_H, width, HEADER_H)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def _draw_header(fig, panel: PanelSpec, index: int) -> None:
    ax = fig.add_axes(header_rect(index))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=HEADER_BG, zorder=0)
    )
    ax.text(
        0.030,
        0.44,
        panel.letter,
        color=HEADER_FG,
        fontsize=9.5,
        fontweight="bold",
        family=FONT_STACK,
        ha="left",
        va="center",
        zorder=2,
    )
    ax.text(
        0.135,
        0.44,
        panel.title,
        color=HEADER_FG,
        fontsize=8.0,
        family=FONT_STACK,
        ha="left",
        va="center",
        zorder=2,
    )


def _draw_annotation(ax, ann: Annotation) -> None:
    ax.annotate(
        "",
        xy=ann.xy,
        xytext=ann.xytext,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.42,head_width=0.17",
            color=ann.color,
            linewidth=1.0,
            shrinkA=6,
            shrinkB=2,
            connectionstyle=f"arc3,rad={ann.rad}",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
        ),
        zorder=5,
    )
    ax.text(
        ann.xytext[0],
        ann.xytext[1],
        ann.text,
        transform=ax.transAxes,
        fontsize=6.3,
        family=FONT_STACK,
        color=LABEL_FG,
        ha=ann.ha,
        va=ann.va,
        linespacing=1.25,
        zorder=6,
        bbox=dict(
            boxstyle="square,pad=0.28",
            facecolor=LABEL_BG,
            edgecolor="none",
            alpha=LABEL_ALPHA,
        ),
    )


def _draw_panel(fig, panel: PanelSpec, index: int, image: np.ndarray):
    ax = fig.add_axes(panel_rect(index))
    ax.imshow(image, interpolation="lanczos", aspect="equal", zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.7)

    for ann in panel.annotations:
        _draw_annotation(ax, ann)

    if panel.badge:
        ax.text(
            0.975,
            0.965,
            panel.badge,
            transform=ax.transAxes,
            fontsize=5.8,
            family=FONT_STACK,
            style="italic",
            color="#ffffff",
            ha="right",
            va="top",
            zorder=6,
            bbox=dict(
                boxstyle="square,pad=0.3",
                facecolor=HEADER_BG,
                edgecolor="none",
                alpha=0.75,
            ),
        )
    return ax


def _connector(fig, start: tuple[float, float], end: tuple[float, float], rad: float = 0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>,head_length=0.36,head_width=0.16",
        mutation_scale=14,
        color=CONNECTOR,
        linewidth=1.25,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    fig.patches.append(arrow)
    return arrow


def _draw_connectors(fig) -> None:
    gap_mid_x = (PANEL_X[1] - (PANEL_X[0] + PANEL_W)) / 2.0
    row0_mid_y = PANEL_Y[0] + PANEL_H / 2.0
    row1_mid_y = PANEL_Y[1] + PANEL_H / 2.0

    # A -> B, B -> C  (top row)
    for col in (0, 1):
        x0 = PANEL_X[col] + PANEL_W + gap_mid_x * 0.35
        x1 = PANEL_X[col + 1] - gap_mid_x * 0.35
        _connector(fig, (x0, row0_mid_y), (x1, row0_mid_y))

    # D -> E, E -> F  (bottom row)
    for col in (0, 1):
        x0 = PANEL_X[col] + PANEL_W + gap_mid_x * 0.35
        x1 = PANEL_X[col + 1] - gap_mid_x * 0.35
        _connector(fig, (x0, row1_mid_y), (x1, row1_mid_y))

    # C wraps down and back to D: right column -> under the top row -> left column
    band_y = PANEL_Y[1] + PANEL_H + HEADER_H + 0.038
    c_x = PANEL_X[2] + PANEL_W / 2.0
    d_x = PANEL_X[0] + PANEL_W / 2.0
    fig.add_artist(
        plt.Line2D(
            [c_x, c_x, d_x],
            [PANEL_Y[0] - 0.012, band_y, band_y],
            transform=fig.transFigure,
            color=CONNECTOR,
            linewidth=1.25,
            solid_joinstyle="miter",
            zorder=1,
        )
    )
    _connector(fig, (d_x, band_y), (d_x, PANEL_Y[1] + PANEL_H + HEADER_H + 0.004))


def build_figure(photo_dir: Path | str, panel_f_path: Path | str | None = None):
    """Assemble the six-panel figure."""
    photo_dir = Path(photo_dir)
    panel_f_path = Path(panel_f_path) if panel_f_path is not None else PANEL_F_ASSET

    fig = plt.figure(figsize=FIG_SIZE_IN, facecolor="white")
    for index, panel in enumerate(PANELS):
        source = panel_f_path if panel.letter == "F" else photo_dir / panel.source
        image = prepare_panel_image(source, panel)
        _draw_header(fig, panel, index)
        _draw_panel(fig, panel, index, image)
    _draw_connectors(fig)
    return fig


def render(
    photo_dir: Path | str,
    out_stem: Path | str,
    panel_f_path: Path | str | None = None,
    dpi: int = 400,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
) -> list[Path]:
    """Build the figure and write it out; returns the paths written."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(photo_dir, panel_f_path=panel_f_path)
    written: list[Path] = []
    try:
        for fmt in formats:
            path = out_stem.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=dpi, facecolor="white")
            written.append(path)
    finally:
        plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--photo-dir",
        type=Path,
        default=DEFAULT_PHOTO_DIR,
        help="directory holding the panel A-E photographs 1.png .. 5.png",
    )
    parser.add_argument(
        "--panel-f",
        type=Path,
        default=PANEL_F_ASSET,
        help="archived UV/IR frame used for panel F",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output path stem")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args(argv)

    written = render(args.photo_dir, args.out, panel_f_path=args.panel_f, dpi=args.dpi)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
