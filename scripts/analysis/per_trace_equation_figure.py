"""Methods figures: the arithmetic behind the raw PER trace, on two sheets.

config_new.yaml's ``analysis.combined.combined_base.envelopes`` block writes
Raw-Testing-PER-Traces / Raw-Training-PER-Traces. Those figures plot one
column -- ``combined_pct`` (exported as ``combined_base``; the two names are
aliases, see ``envelope_combined._COLUMN_LOOKUP_ALIASES``).

Sheet 1 (``per_trace_equations``) states every equation between the YOLO
eye/proboscis coordinates and that column, and nothing else:

  1. eye->proboscis pixel distance        (envelope_combined._d_px_from_coords)
  2. proboscis angle about the eye        (envelope_combined._compute_angle_deg)
  3. centre on the retracted reference    (_find_reference_angle)
  4. scale to +/-100% per fly             (_fly_max_centered)
  5. logarithmic angle multiplier         (_angle_multiplier)
  6. weight the distance in the px domain (combine_distance_angle, pass 1)
  7. normalize per fly to 0-100           (combine_distance_angle, pass 2)

Sheet 2 (``per_trace_symbols``) defines every symbol used above, with the
constants each one carries.

Constants are read from config/config_new.yaml and from envelope_combined at
runtime, and the multiplier curve is drawn by calling the production function
-- nothing here re-derives the pipeline's numbers.

Run:
    python scripts/analysis/per_trace_equation_figure.py
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fbpipe.config import load_raw_config  # noqa: E402
from fbpipe.utils.columns import EYE_CLASS, PROBOSCIS_CLASS  # noqa: E402
from scripts.analysis import envelope_combined as ec  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "config_new.yaml"
OUT_DIR = REPO_ROOT / "figures"
STEM_EQUATIONS = "per_trace_equations"
STEM_SYMBOLS = "per_trace_symbols"

INK = "#0b0b0b"
MUTED = "#6f6d68"
FAINT = "#d7d5d0"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"     # step badges + the multiplier curve

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,       # editable text in Illustrator / Inkscape
        "ps.fonttype": 42,
        "svg.fonttype": "none",   # keep <text> as text, not paths, in the SVG
    }
)


@dataclass(frozen=True)
class Settings:
    """Every number the figures print, as read from config / production code."""

    anchor_x: float
    anchor_y: float
    norm_floor_px: float
    norm_min_px: float
    norm_max_px: float
    window_sec: float
    fps_default: float
    measure_cols: tuple[str, ...]
    out_dir: str
    eye_class: int
    proboscis_class: int

    @property
    def window_frames(self) -> int:
        return max(int(round(self.window_sec * self.fps_default)), 1)


@dataclass(frozen=True)
class EqStep:
    """One numbered equation on sheet 1."""

    number: int
    title: str
    equation: str


@dataclass(frozen=True)
class Symbol:
    """One row of sheet 2: the symbol, what to call it, what it is."""

    latex: str
    name: str
    meaning: str


def load_settings(config_path: Path | str = CONFIG_PATH) -> Settings:
    raw = load_raw_config(config_path)
    limits = raw.get("distance_limits", {})
    combined_base = raw["analysis"]["combined"]["combined_base"]
    wide = combined_base["wide"]
    testing_block = combined_base["envelopes"][0]
    return Settings(
        anchor_x=float(raw["anchor_x"]),
        anchor_y=float(raw["anchor_y"]),
        # The floor lives in the analysis module, not the config.
        norm_floor_px=float(ec.WEIGHTED_EFFECTIVE_MAX_FLOOR),
        norm_min_px=float(limits["class2_min"]),
        norm_max_px=float(limits["class2_max"]),
        window_sec=float(raw["window_sec"]),
        fps_default=float(raw["fps_default"]),
        measure_cols=tuple(wide["measure_cols"]),
        out_dir=str(testing_block["out_dir"]),
        eye_class=int(EYE_CLASS),
        proboscis_class=int(PROBOSCIS_CLASS),
    )


def multiplier_curve() -> tuple[np.ndarray, np.ndarray]:
    """Angle percentage -> multiplier, evaluated by the production function."""
    angle_pct = np.linspace(-100.0, 100.0, 401)
    return angle_pct, ec._angle_multiplier(angle_pct)


def _fmt(value: float) -> str:
    return f"{value:g}"


# Meanings are drawn with wrap=False (matplotlib's wrap ignores figure-fraction
# text), so they are wrapped here instead. MEANING_COLS fits the symbol sheet's
# right-hand column at 8.6 pt.
MEANING_COLS = 78


def _wrap(text: str, cols: int = MEANING_COLS) -> str:
    return "\n".join(textwrap.wrap(text, width=cols))


# ── sheet 1: the equations ────────────────────────────────────────────────
def steps(settings: Settings) -> list[EqStep]:
    """The seven equations, in pipeline order."""
    return [
        EqStep(
            1,
            "Eye-to-proboscis distance",
            r"$d(t)=\sqrt{\left(x_p-x_e\right)^2+\left(y_p-y_e\right)^2}$",
        ),
        EqStep(
            2,
            "Proboscis angle about the eye",
            r"$\theta(t)=\mathrm{atan2}\left(\left|u\times v\right|,\ u\cdot v\right)"
            r"\times\frac{180}{\pi}$",
        ),
        EqStep(
            3,
            "Centre on the retracted pose",
            r"$\Delta\theta(t)=\theta(t)-\theta_{ref},"
            r"\qquad \theta_{ref}=\theta\left(\arg\min_t\ d(t)\right)$",
        ),
        EqStep(
            4,
            "Scale the angle per fly",
            r"$A(t)=100\,\frac{\Delta\theta(t)}{\mathrm{max}_{\mathrm{fly}}"
            r"\left|\Delta\theta\right|}\ \in[-100,100]$",
        ),
        EqStep(
            5,
            "Angle multiplier",
            r"$m(t)=1+\ln\left(1+\frac{A(t)}{100}\left(e-1\right)\right),"
            r"\qquad m(t)=1\ \ \mathrm{for}\ A(t)\leq 0$",
        ),
        EqStep(
            6,
            "Weight in the pixel domain",
            r"$d_w(t)=d(t)\cdot m(t)$",
        ),
        EqStep(
            7,
            "Normalize to the raw PER trace",
            r"$\mathrm{PER}(t)=100\,\frac{d_w(t)-d_w^{\min}}"
            r"{\max\left(d_w^{\max},\ " + _fmt(settings.norm_floor_px) + r"\right)-d_w^{\min}}$",
        ),
    ]


# ── sheet 2: what the symbols mean ────────────────────────────────────────
def symbols(settings: Settings) -> list[Symbol]:
    """Every symbol that appears in the seven equations."""
    cols = " = ".join(settings.measure_cols)
    return [
        Symbol(
            r"$t$",
            "time",
            _wrap(
                f"Frame time in seconds, {_fmt(settings.fps_default)} fps "
                "(fps_default). Odor is commanded on at 30 s and off at 60 s."
            ),
        ),
        Symbol(
            r"$e(t)=\left(x_e,\ y_e\right)$",
            "eye position",
            _wrap(
                f"Centroid of the class-{settings.eye_class} (eye) YOLO-OBB "
                "detection for that fly, in image pixels."
            ),
        ),
        Symbol(
            r"$p(t)=\left(x_p,\ y_p\right)$",
            "proboscis position",
            _wrap(
                f"Centroid of the class-{settings.proboscis_class} (proboscis) "
                "YOLO-OBB detection bound to that eye, in image pixels."
            ),
        ),
        Symbol(
            r"$a$",
            "rig anchor",
            _wrap(
                f"Fixed point in the frame, ({_fmt(settings.anchor_x)}, "
                f"{_fmt(settings.anchor_y)}) px from anchor_x / anchor_y in "
                "config_new.yaml. Mirrored rigs resolve their own anchor. It "
                "supplies the direction the angle is measured from."
            ),
        ),
        Symbol(
            r"$u=a-e(t)$",
            "reference vector",
            _wrap("Points from the eye to the rig anchor."),
        ),
        Symbol(
            r"$v=p(t)-e(t)$",
            "proboscis vector",
            _wrap("Points from the eye to the proboscis."),
        ),
        Symbol(
            r"$d(t)$",
            "raw distance",
            _wrap(
                "Eye-to-proboscis separation in pixels. Only "
                f"{_fmt(settings.norm_min_px)}-{_fmt(settings.norm_max_px)} px "
                "passes the tracking gates; rejected frames enter as NaN."
            ),
        ),
        Symbol(
            r"$\theta(t)$",
            "extension angle",
            _wrap(
                "Unsigned angle between u and v, 0-180 deg -- which way the "
                "proboscis points, independent of how far it reaches."
            ),
        ),
        Symbol(
            r"$\theta_{ref}$",
            "retracted reference",
            _wrap(
                "Theta at the fly's minimum-distance frame (proboscis fully "
                "retracted), searched across all of that fly's trials: one "
                "reference per fly, not per trial."
            ),
        ),
        Symbol(
            r"$\Delta\theta(t)$",
            "centred angle",
            _wrap("Angle relative to that retracted pose; 0 at the reference frame."),
        ),
        Symbol(
            r"$\mathrm{max}_{\mathrm{fly}}\left|\Delta\theta\right|$",
            "per-fly angle scale",
            _wrap(
                "Largest angular excursion that fly ever made, pooled over its "
                "trials. Each fly is scaled by its own range."
            ),
        ),
        Symbol(
            r"$A(t)$",
            "angle percentage",
            _wrap(
                "Centred angle as a percent of that scale, clipped to "
                "[-100, 100]. Positive = swung toward extension."
            ),
        ),
        Symbol(
            r"$e$",
            "Euler's number",
            _wrap(
                "2.71828..., the constant inside the log, chosen so the "
                "multiplier lands exactly on 2 at A = +100."
            ),
        ),
        Symbol(
            r"$m(t)$",
            "angle multiplier",
            _wrap(
                "1x at A <= 0 rising logarithmically to 2x at A = +100 "
                "(+50 -> 1.62x). An inward angle never attenuates the distance."
            ),
        ),
        Symbol(
            r"$d_w(t)$",
            "weighted distance",
            _wrap(
                "Distance times multiplier, still in pixels -- angle and reach "
                "are combined before any normalization."
            ),
        ),
        Symbol(
            r"$d_w^{\min},\ d_w^{\max}$",
            "per-fly bounds",
            _wrap(
                "Smallest and largest weighted distance for that fly number, "
                "pooled over all of its trials."
            ),
        ),
        Symbol(
            r"$" + _fmt(settings.norm_floor_px) + r"$",
            "normalization floor",
            _wrap(
                f"WEIGHTED_EFFECTIVE_MAX_FLOOR: the denominator uses at least "
                f"{_fmt(settings.norm_floor_px)} px, so a fly that never "
                "extends is not stretched to 100%."
            ),
        ),
        Symbol(
            r"$\mathrm{PER}(t)$",
            "raw PER trace",
            _wrap(
                f"Percent extension, 0-100, one value per frame. This is the "
                f"plotted column: {cols}."
            ),
        ),
    ]


def _draw_multiplier_inset(fig, rect) -> None:
    ax = fig.add_axes(rect)
    angle_pct, mult = multiplier_curve()
    ax.plot(angle_pct, mult, color=ACCENT, lw=1.8, solid_capstyle="round")
    ax.axvline(0.0, color=FAINT, lw=0.8, zorder=0)
    ax.set_xlim(-100, 100)
    ax.set_ylim(0.9, 2.15)
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([1.0, 2.0])
    ax.set_yticklabels(["1x", "2x"])
    ax.tick_params(labelsize=6.5, colors=MUTED, length=2.5, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    ax.set_title("m  vs  A(t)", fontsize=7.5, color=MUTED, pad=3.0)
    ax.set_facecolor(SURFACE)


# Both sheets are laid out in inches from the top of the page and converted to
# figure fractions here: the two pages differ in height, and fixed fractions
# would slide the header around (they did -- the title overlapped its subtitle
# on the shorter sheet).
def _fy(fig, inches_from_top: float) -> float:
    return 1.0 - inches_from_top / fig.get_figheight()


HEADER_IN = 1.30   # height of the header block, incl. its rule


def _header(fig, title: str, subtitle: str, trailer: str) -> None:
    fig.text(0.055, _fy(fig, 0.38), title, fontsize=21, fontweight="bold",
             va="top", ha="left")
    fig.text(0.055, _fy(fig, 0.78), subtitle, fontsize=11, color=MUTED,
             va="top", ha="left")
    fig.text(0.055, _fy(fig, 1.02), trailer, fontsize=8.5, color=ACCENT,
             va="top", ha="left", family="monospace")
    rule = _fy(fig, HEADER_IN)
    fig.add_artist(plt.Line2D([0.055, 0.945], [rule, rule], color=FAINT, lw=1.0))


def _provenance(fig, y: float) -> None:
    fig.text(0.055, y,
             "Constants read at draw time from config/config_new.yaml (anchor_x, anchor_y,\n"
             "distance_limits, window_sec, fps_default) and from "
             "scripts/analysis/envelope_combined.py\n(WEIGHTED_EFFECTIVE_MAX_FLOOR, "
             "_angle_multiplier -- the plotted curve is that function, evaluated).",
             fontsize=7.4, color=MUTED, va="top", ha="left", family="monospace",
             linespacing=1.5)


EQ_ROW_IN = 0.82      # per equation row
EQ_FOOTER_IN = 0.62   # the "symbols are on the other sheet" line


def build_equation_figure(settings: Settings | None = None):
    """Sheet 1: the seven equations, no prose."""
    settings = settings or load_settings()
    rows = steps(settings)

    height = HEADER_IN + len(rows) * EQ_ROW_IN + EQ_FOOTER_IN
    fig = plt.figure(figsize=(8.6, height))
    _header(
        fig,
        "Raw PER trace",
        "The seven equations behind the plotted trace",
        "analysis.combined.combined_base.envelopes  ->  " + Path(settings.out_dir).name,
    )

    x_text = 0.115
    for i, step in enumerate(rows):
        # centre of the row, in inches from the top of the page
        centre = HEADER_IN + (i + 0.5) * EQ_ROW_IN + 0.06
        y = _fy(fig, centre)

        fig.text(0.062, y, str(step.number), fontsize=11, fontweight="bold",
                 color="#ffffff", va="center", ha="center",
                 bbox=dict(boxstyle="circle,pad=0.42", facecolor=ACCENT, edgecolor="none"))
        fig.text(x_text, _fy(fig, centre - 0.26), step.title.upper(), fontsize=8.5,
                 fontweight="bold", color=MUTED, va="center", ha="left")
        fig.text(x_text, y, step.equation, fontsize=15, va="center", ha="left")

        if i < len(rows) - 1:
            rule = _fy(fig, centre + EQ_ROW_IN / 2.0)
            fig.add_artist(plt.Line2D([x_text, 0.945], [rule, rule], color=FAINT, lw=0.6))

    fig.text(0.055, _fy(fig, height - 0.30),
             "Symbols are defined on the companion sheet (per_trace_symbols).",
             fontsize=8.2, color=MUTED, va="baseline", ha="left")
    return fig


SYM_LINE_IN = 0.175   # one wrapped line of a meaning
SYM_PAD_IN = 0.21     # padding around each symbol row
SYM_FOOTER_IN = 1.85  # the "stops at PER(t)" block + provenance


def _symbol_row_height(symbol: Symbol) -> float:
    """Rows grow with their meaning so nothing overlaps its neighbour."""
    lines = symbol.meaning.count("\n") + 1
    return max(0.42, SYM_PAD_IN + SYM_LINE_IN * lines)


def build_symbol_figure(settings: Settings | None = None):
    """Sheet 2: what every symbol in those equations means."""
    settings = settings or load_settings()
    rows = symbols(settings)
    heights = [_symbol_row_height(s) for s in rows]

    height = HEADER_IN + sum(heights) + SYM_FOOTER_IN
    fig = plt.figure(figsize=(8.6, height))
    _header(
        fig,
        "Raw PER trace -- symbols",
        "Every term in the seven equations, and the constant it carries",
        "analysis.combined.combined_base.envelopes  ->  " + Path(settings.out_dir).name,
    )

    x_sym = 0.055
    x_name = 0.245
    x_meaning = 0.410
    cursor = HEADER_IN + 0.22
    for i, (symbol, row_h) in enumerate(zip(rows, heights)):
        y = _fy(fig, cursor)
        fig.text(x_sym, y, symbol.latex, fontsize=12.5, va="top", ha="left")
        fig.text(x_name, _fy(fig, cursor + 0.03), symbol.name, fontsize=9,
                 fontweight="bold", va="top", ha="left")
        fig.text(x_meaning, _fy(fig, cursor + 0.03), symbol.meaning, fontsize=8.6,
                 color=MUTED, va="top", ha="left", linespacing=1.45)
        cursor += row_h
        if i < len(rows) - 1:
            rule = _fy(fig, cursor - 0.10)
            fig.add_artist(plt.Line2D([x_sym, 0.945], [rule, rule], color=FAINT, lw=0.5))

    # The multiplier is the one term whose shape is worth seeing.
    inset_h = 0.62 / height
    _draw_multiplier_inset(fig, [0.815, _fy(fig, height - 0.70), 0.125, inset_h])

    footer_top = height - SYM_FOOTER_IN + 0.55
    rule = _fy(fig, footer_top)
    fig.add_artist(plt.Line2D([0.055, 0.945], [rule, rule], color=FAINT, lw=1.0))
    fig.text(0.055, _fy(fig, footer_top + 0.22), "The raw trace stops at PER(t).",
             fontsize=9.5, fontweight="bold", va="top", ha="left")
    fig.text(0.055, _fy(fig, footer_top + 0.42),
             "A rolling RMS over "
             f"window_sec {_fmt(settings.window_sec)} s ({settings.window_frames} frames at "
             f"{_fmt(settings.fps_default)} fps) and a Hilbert envelope of that RMS are also "
             "computed by\nenvelope_combined, but they are not applied to the raw traces -- "
             "those plot PER(t) as defined above.",
             fontsize=8.2, color=MUTED, va="top", ha="left", linespacing=1.5)
    _provenance(fig, _fy(fig, footer_top + 0.90))
    return fig


def save_figure(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("png", "svg", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        written.append(path)
    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the raw-PER-trace equation and symbol sheets."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(args.config)
    for builder, stem in (
        (build_equation_figure, STEM_EQUATIONS),
        (build_symbol_figure, STEM_SYMBOLS),
    ):
        fig = builder(settings)
        for path in save_figure(fig, args.out_dir, stem):
            print(f"wrote {path}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
