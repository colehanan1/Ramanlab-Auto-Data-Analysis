"""One line, in words: what the raw PER traces actually plot.

The figures under ``Results/New-Opto-Fly-Figures/Raw-Testing-PER-Traces`` plot a
single column whose axis reads "Max Distance x Angle %". This sheet states that
column as one sentence-shaped equation --

    PER %  =  distance from eye to proboscis  x  f(angle)

-- where f puts a *positive* angle on a log scale between 1 and 2, and leaves
everything else at 1. A small curve shows f's shape, and that is the whole sheet.

``scripts/analysis/per_trace_equation_figure.py`` is the rigorous companion: it
states every equation and defines every symbol. This one deliberately carries no
symbols. The curve and the 1 -> 2 bounds printed here are still produced by
calling ``envelope_combined._angle_multiplier`` -- the same function the pipeline
calls -- so the simplification cannot drift from the pipeline.

Run:
    python scripts/analysis/distance_angle_multiplier_simple.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from scripts.analysis import envelope_combined as ec  # noqa: E402

OUT_DIR = REPO_ROOT / "figures"
STEM = "distance_angle_multiplier_simple"

INK = "#0b0b0b"
MUTED = "#6f6d68"
FAINT = "#d7d5d0"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"     # the angle term, and everything explaining it
BAND = "#eef4fc"       # takeaway strip

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,       # editable text in Illustrator / Inkscape
        "ps.fonttype": 42,
        "svg.fonttype": "none",   # keep <text> as text, not paths, in the SVG
    }
)

TITLE = "What the raw PER traces plot"
TRACE_AXIS_NAME = "Max Distance x Angle %"


@dataclass(frozen=True)
class Term:
    """One piece of the one-line equation, with the note printed beneath it."""

    text: str
    note: str
    x: float          # figure fraction, left edge
    size: float
    color: str
    bold: bool = False


def multiplier_curve() -> tuple[np.ndarray, np.ndarray]:
    """Angle percentage -> multiplier, evaluated by the production function."""
    angle_pct = np.linspace(-100.0, 100.0, 401)
    with np.errstate(invalid="ignore"):
        return angle_pct, ec._angle_multiplier(angle_pct)


def _multiplier_at(angle_pct: float) -> float:
    with np.errstate(invalid="ignore"):
        return float(ec._angle_multiplier(np.array([float(angle_pct)]))[0])


# The two bounds the sheet prints, read off the production function rather than
# typed in: no angle -> MULT_MIN, the fly's largest positive angle -> MULT_MAX.
MULT_MIN = _multiplier_at(0.0)
MULT_MAX = _multiplier_at(100.0)

EQUATION_TERMS: tuple[Term, ...] = (
    Term("PER %", f"the trace axis:\n“{TRACE_AXIS_NAME}”", 0.045, 21.0, INK, bold=True),
    Term("=", "", 0.150, 21.0, MUTED),
    Term(
        "distance from eye to proboscis",
        "how far the proboscis sits from the eye,\nin pixels, one value per frame",
        0.183,
        21.0,
        INK,
    ),
    Term("×", "", 0.558, 21.0, MUTED),
    Term(
        "f(angle)",
        f"{MULT_MIN:.0f} → {MULT_MAX:.0f} on a log scale,\npositive angles only",
        0.594,
        21.0,
        ACCENT,
    ),
)

TAKEAWAY = (
    f"A positive angle multiplies the distance by up to {MULT_MAX:.0f}×; no angle, or an angle the "
    f"other way, multiplies it by {MULT_MIN:.0f}×. The result is then scaled per fly to 0–100 %."
)


def _draw_curve(ax) -> None:
    """f, drawn by evaluating the production function."""
    angle_pct, mult = multiplier_curve()
    ax.plot(angle_pct, mult, color=ACCENT, lw=2.4, solid_capstyle="round", zorder=3)
    ax.fill_between(angle_pct, MULT_MIN, mult, color=ACCENT, alpha=0.10, zorder=1)

    ax.set_xlim(-100, 100)
    ax.set_ylim(MULT_MIN - 0.12, MULT_MAX + 0.10)
    ax.set_xticks([-100, 0, 100])
    ax.set_xticklabels(["negative", "0", "positive"], fontsize=8)
    ax.set_yticks([MULT_MIN, MULT_MAX])
    ax.set_yticklabels([f"{MULT_MIN:.0f}", f"{MULT_MAX:.0f}"], fontsize=8.5)
    ax.tick_params(colors=MUTED, length=2.5, pad=2)
    ax.axvline(0.0, color=FAINT, lw=1.0, zorder=0)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    ax.set_title("f(angle)", fontsize=9.5, color=ACCENT, pad=4.0)
    ax.set_xlabel("angle", fontsize=8.5, color=MUTED, labelpad=2.0)


def build_figure():
    fig = plt.figure(figsize=(11.4, 3.35))

    fig.text(0.045, 0.94, TITLE, fontsize=14.5, fontweight="bold", va="top", ha="left")
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.835, 0.835], color=FAINT, lw=1.0))

    baseline = 0.585
    for term in EQUATION_TERMS:
        fig.text(term.x, baseline, term.text, fontsize=term.size, color=term.color,
                 fontweight="bold" if term.bold else "normal", va="baseline", ha="left")
        if term.note:
            fig.text(term.x, baseline - 0.115, term.note, fontsize=9.2, color=MUTED,
                     va="top", ha="left", linespacing=1.5)

    _draw_curve(fig.add_axes([0.818, 0.325, 0.137, 0.35]))

    fig.add_artist(Rectangle((0.045, 0.05), 0.91, 0.115, transform=fig.transFigure,
                             facecolor=BAND, edgecolor="none", zorder=0))
    fig.text(0.5, 0.1075, TAKEAWAY, fontsize=11.0, color=INK, ha="center", va="center")
    return fig


def render(out_dir: Path | str = OUT_DIR) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    written: list[Path] = []
    for suffix in (".png", ".svg"):
        path = out_dir / f"{STEM}{suffix}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"where to write {STEM}.png/.svg (default: {OUT_DIR})")
    args = parser.parse_args(argv)
    for path in render(args.out_dir):
        print(f"[OK] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
