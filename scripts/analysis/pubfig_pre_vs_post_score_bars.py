#!/usr/bin/env python3
"""Publication version of the pre-vs-post mean PER score bars.

Redraws ``<cohort>_score_bars.png`` from the ``<cohort>_summary.csv`` that
``pretest_vs_test_comparison.py`` writes beside it, so every mean, SEM and star
is the one already computed there (Wilcoxon signed-rank, paired within fly,
uncorrected). Only the styling changes, for a small printed panel:

* no title (the caption carries it), all text black and enlarged;
* black-outlined bars in the validated pre/post blue/orange pair;
* SEM drawn upward only;
* a fixed y max (default 4). A bar plus SEM above it raises rather than being
  clipped, so the fixed axis can never hide data.

Writes PNG (600 dpi), PDF and editable SVG.

Usage
-----
    python3 scripts/analysis/pubfig_pre_vs_post_score_bars.py \\
        --summary-csv .../IAA-Sensitivity-12-1/IAA-Sensitivity-12-1_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import significance_brackets  # noqa: E402
from scripts.analysis.pretest_vs_test_comparison import POST_COLOR, PRE_COLOR  # noqa: E402

BLACK = "#000000"
FIGSIZE = (4.4, 3.1)  # inches: a small panel, text sized to read at that size
TICK_PT, LABEL_PT, LEGEND_PT, STAR_PT = 10.5, 12, 10.5, 15
EDGE_LW = 0.9
BAR_W = 0.38
PAIR_GAP = 0.02

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": TICK_PT,
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "xtick.labelcolor": BLACK,
    "ytick.labelcolor": BLACK,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.frameon": False,
    "legend.fontsize": LEGEND_PT,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    # Journals reject Type-3 fonts; editable text in the PDF and SVG.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def _n_text(summary: pd.DataFrame) -> str:
    counts = sorted(set(int(v) for v in summary["n_flies"]))
    return f"n = {counts[0]} flies" if len(counts) == 1 else f"n = {counts[0]}–{counts[-1]} flies"


def build_figure(summary: pd.DataFrame, *, trained_odor: str = "", ymax: float = 4.0,
                 figsize: tuple[float, float] = FIGSIZE):
    """The styled figure, unsaved. Raises if any bar + SEM exceeds ``ymax``."""
    means = summary[["mean_pre", "mean_post"]].to_numpy(dtype=float)
    sems = np.nan_to_num(summary[["sem_pre", "sem_post"]].to_numpy(dtype=float))
    tops = means + sems
    if np.nanmax(tops) > ymax:
        worst = summary["odor"].iloc[int(np.nanargmax(tops.max(axis=1)))]
        raise ValueError(f"{worst}: mean + SEM {np.nanmax(tops):.2f} exceeds y max {ymax}")
    # Bars anchor at 0; drop below it only if a mean actually does (-1 is a real score).
    ymin = 0.0 if np.nanmin(means) >= 0 else -1.0

    odors = list(summary["odor"])
    x = np.arange(len(odors), dtype=float)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        for col, (color, sign) in enumerate(((PRE_COLOR, -1), (POST_COLOR, 1))):
            pos = x + sign * (BAR_W / 2 + PAIR_GAP / 2)
            ax.bar(pos, means[:, col], BAR_W, color=color, edgecolor=BLACK,
                   linewidth=EDGE_LW, zorder=2)
            cap = BAR_W * 0.22
            for xi, m, e in zip(pos, means[:, col], sems[:, col]):
                if not np.isfinite(m) or e <= 0:
                    continue
                # Upward only: from the bar top to mean + SEM, capped at the top.
                ax.plot([xi, xi], [m, m + e], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", zorder=3)
                ax.plot([xi - cap, xi + cap], [m + e, m + e], color=BLACK,
                        linewidth=1.0, zorder=3)

        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(ymin, ymax + 0.001, 1.0))
        ax.set_ylabel("Mean PER score", fontsize=LABEL_PT)
        ax.set_xticks(x)
        ax.set_xticklabels(odors, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(odors) - 0.4)
        ax.tick_params(axis="x", length=0, pad=4)
        for label, odor in zip(ax.get_xticklabels(), odors):
            label.set_color(BLACK)
            if trained_odor and str(odor).casefold().startswith(trained_odor.casefold()):
                label.set_fontweight("bold")

        handles = [Patch(facecolor=PRE_COLOR, edgecolor=BLACK, linewidth=EDGE_LW, label="Pre-test"),
                   Patch(facecolor=POST_COLOR, edgecolor=BLACK, linewidth=EDGE_LW, label="Post-training")]
        legend = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                           ncol=2, handlelength=1.2, handleheight=0.9, columnspacing=1.2,
                           borderaxespad=0.2)
        for t in legend.get_texts():
            t.set_color(BLACK)

        pvals = [None if pd.isna(p) else float(p) for p in summary["p_score"]]
        significance_brackets.draw(ax, x, 2 * BAR_W, pvals, fontsize=STAR_PT, expand=False)
        for t in ax.texts:
            t.set_color(BLACK)
        for line in ax.get_lines():
            line.set_color(BLACK)
        # After the brackets: the helper treats every axes text as ink to clear,
        # so an n label drawn first would push the bracket up to the ceiling.
        ax.text(1.0, 1.0, _n_text(summary), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=LEGEND_PT, color=BLACK)
    return fig


def render(summary: pd.DataFrame, *, out_stem: Path, trained_odor: str = "",
           ymax: float = 4.0) -> list[Path]:
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(summary, trained_odor=trained_odor, ymax=ymax)
    written = []
    with plt.rc_context(_RC):
        for ext in (".png", ".pdf", ".svg"):
            path = out_stem.with_suffix(ext)
            fig.savefig(path)
            written.append(path)
    plt.close(fig)
    return written


def _cohort_from_summary(path: Path) -> str:
    return path.name.removesuffix("_summary.csv")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary-csv", required=True, type=Path)
    p.add_argument("--out-stem", type=Path, default=None,
                   help="Default: <cohort>_score_bars_pub beside the summary CSV.")
    p.add_argument("--trained-odor", default=None,
                   help="Odor to bold. Default: derived from the cohort name.")
    p.add_argument("--ymax", type=float, default=4.0)
    args = p.parse_args(argv)

    cohort = _cohort_from_summary(args.summary_csv)
    trained = args.trained_odor
    if trained is None:
        from scripts.analysis.envelope_visuals import _trained_label
        trained = _trained_label(cohort) or ""
    out_stem = args.out_stem or args.summary_csv.with_name(f"{cohort}_score_bars_pub")
    for path in render(pd.read_csv(args.summary_csv), out_stem=out_stem,
                       trained_odor=trained, ymax=args.ymax):
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
