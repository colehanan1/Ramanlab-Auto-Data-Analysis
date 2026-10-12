#!/usr/bin/env python3
"""Trained-odor (CS+) PER score, pre-test vs post-training, one panel per dataset.

Publication / grant style, matching ``pubfig_pre_vs_post_score_bars.py``: no
title, black text, black-outlined bars, SEM drawn upward only, sized to stay
legible when printed small. Each panel is one dataset's trained odor, headed by
the odor and its vial concentration in bold ("Isoamyl Acetate 1%", read from
the dataset name) with n underneath. Starvation time is not drawn; it goes to the .csv, taken from the
dataset name unless the panel spec overrides it ("DATASET=ODOR=HOURS") -- the
*-Sensitivity-1-* datasets are 0 h.

Colour: the hue is the odor (gold = Isoamyl Acetate, blue = 3-Octanol, the
repo's odor identities, IAA stepped darker so it reads on white); within a hue,
lighter = pre-test, darker = post-training. Each pair passes the dataviz
validator (``--ordinal``) and the four together pass the categorical checks;
the light steps sit under 3:1 on white, so every bar is outlined and labelled.

Stats: Wilcoxon signed-rank, pre vs post paired within fly (uncorrected); no
star when fewer than 6 flies changed (the test cannot reach p<0.05 there).

Usage
-----
    python3 scripts/analysis/pubfig_cs_plus_pre_post.py \\
        --panel "IAA-Sensitivity-1-1=Isoamyl Acetate=0" \\
        --panel "3Oct-Sensitivity-1-0.1=3-Octanol=0"
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.analysis import significance_brackets  # noqa: E402
from scripts.analysis.pretest_vs_test_comparison import _sem, load_paired_scores, wilcoxon_p  # noqa: E402

CSV = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
FLAGGED = CSV / "flagged-flys-truth.csv"
DEFAULT_OUT = Path("/home/ramanlab/Documents/cole/Results/Figures/CS-plus_pre-vs-post_pubfig")

BLACK = "#000000"
#: (pre, post) per odor. Validated: each pair --ordinal, all four categorical.
ODOR_COLORS = {
    "isoamyl acetate": ("#d4a90f", "#856600"),
    "3-octanol": ("#4f97d6", "#0d5c96"),
}
FALLBACK = ("#a8a7a1", "#46453f")
TICK_PT, LABEL_PT, HEAD_PT, SUB_PT, STAR_PT = 10.5, 12, 10, 8.5, 15
EDGE_LW, BAR_W = 0.9, 0.62
_HOURS_RE = re.compile(r"^(?P<odor>[A-Za-z0-9]+)-Sensitivity-(?P<h>[\d.]+)-(?P<c>[\d.]+)$")
DEFAULT_PANELS = ("IAA-Sensitivity-1-1=Isoamyl Acetate=0",
                  "3Oct-Sensitivity-1-0.1=3-Octanol=0")

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": TICK_PT, "text.color": BLACK, "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": BLACK, "ytick.color": BLACK,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


def parse_panel(spec: str) -> tuple[str, str, float | None]:
    """``"DATASET=ODOR[=HOURS]"`` -> (dataset, odor, hours or None)."""
    parts = spec.split("=")
    if len(parts) not in (2, 3):
        raise ValueError(f"panel spec must be DATASET=ODOR[=HOURS]: {spec!r}")
    return parts[0].strip(), parts[1].strip(), float(parts[2]) if len(parts) == 3 else None


def odor_colors(odor: str) -> tuple[str, str]:
    return ODOR_COLORS.get(odor.casefold(), FALLBACK)


@dataclasses.dataclass(frozen=True)
class Panel:
    dataset: str
    odor: str
    n: int
    mean_pre: float
    sem_pre: float
    mean_post: float
    sem_post: float
    p: float | None
    hours: float | None
    conc: str | None


def panel_summary(paired: pd.DataFrame, *, dataset: str, odor: str,
                  hours: float | None = None) -> Panel:
    sub = paired[(paired["dataset_canon"] == dataset)
                 & (paired["odor"].str.casefold() == odor.casefold())]
    if sub.empty:
        raise ValueError(f"no paired pre/post trials for {odor!r} in {dataset!r}")
    pre = sub["score_pre"].to_numpy(dtype=float)
    post = sub["score_post"].to_numpy(dtype=float)
    m = _HOURS_RE.search(dataset)
    return Panel(dataset, odor, len(sub), float(pre.mean()), _sem(pre), float(post.mean()),
                 _sem(post), wilcoxon_p(pre, post),
                 hours if hours is not None else (float(m["h"]) if m else None),
                 m["c"] if m else None)


def load_panels(specs: Sequence[tuple[str, str, float | None]], *, predictions_csv: Path,
                flagged_flies_csv: Path, tmp_dir: Path) -> list[Panel]:
    """Paired scores for each (dataset, odor), flagged flies dropped, labels normalised."""
    import scripts.analysis.starvation_pre_post_comparison as spc

    groups = [spc.Group(ds, ds, "") for ds, _o, _h in specs]
    preds = spc.drop_flagged(spc.subset_and_relabel(pd.read_csv(predictions_csv), groups),
                             flagged_flies_csv)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subset = tmp_dir / "predictions_subset_relabelled.csv"
    preds.to_csv(subset, index=False)
    paired = load_paired_scores(subset, flagged_flies_csv=str(flagged_flies_csv))
    return [panel_summary(paired, dataset=ds, odor=odor, hours=h) for ds, odor, h in specs]


def build_figure(panels: Sequence[Panel], *, ymax: float | None = None):
    tops = [max(p.mean_pre + p.sem_pre, p.mean_post + p.sem_post) for p in panels]
    # 3.75 by default: room above a bar topping out near 2.5 for the bracket and
    # the in-panel header, while the last labelled tick (and the spine) stops at 3.
    # Taller data raises it: top of bar + SEM, plus the header's share.
    ymax = ymax or max(3.75, float(max(tops)) + 1.25)
    ymin = 0.0 if min(min(p.mean_pre, p.mean_post) for p in panels) >= 0 else -1.0
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(2.05 * len(panels) + 0.5, 2.4),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        x = np.array([0.0, 1.0])
        for ax, p in zip(axes, panels):
            pre_c, post_c = odor_colors(p.odor)
            means, sems = [p.mean_pre, p.mean_post], [p.sem_pre, p.sem_post]
            ax.bar(x, means, BAR_W, color=[pre_c, post_c], edgecolor=BLACK,
                   linewidth=EDGE_LW, zorder=2)
            for xi, m, e in zip(x, means, sems):
                if e > 0:  # SEM upward only, capped
                    ax.plot([xi, xi], [m, m + e], color=BLACK, linewidth=1.0, zorder=3)
                    ax.plot([xi - 0.12, xi + 0.12], [m + e] * 2, color=BLACK, linewidth=1.0,
                            zorder=3)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(-0.6, 1.6)
            ax.set_xticks(x)
            ax.set_xticklabels(["Pre", "Post"])
            ax.tick_params(axis="x", length=0, pad=4)
            significance_brackets.draw(ax, [0.5], 1.0, [p.p], fontsize=STAR_PT, expand=False)
            for t in ax.texts:
                t.set_color(BLACK)
            for line in ax.get_lines():
                line.set_color(BLACK)
            # Panel header INSIDE the axes, in the headroom above the bars (drawn
            # after the bracket, which would otherwise treat it as data to clear).
            head = f"{p.odor} {p.conc}%" if p.conc else p.odor
            ax.text(0.5, 0.955, head, transform=ax.transAxes, ha="center",
                    va="top", fontsize=HEAD_PT, fontweight="bold", color=BLACK)
            ax.text(0.5, 0.86, f"n = {p.n} flies", transform=ax.transAxes, ha="center",
                    va="top", fontsize=SUB_PT, color=BLACK)
        top_tick = float(np.floor(ymax))
        axes[0].set_yticks(np.arange(ymin, top_tick + 0.001, 1.0))
        for ax in axes:
            ax.spines["left"].set_bounds(ymin, top_tick)
        axes[0].set_ylabel("Mean PER score", fontsize=LABEL_PT)
        for ax in axes[1:]:
            ax.tick_params(axis="y", labelleft=False)
        fig.subplots_adjust(wspace=0.3)
    return fig


def render(panels: Sequence[Panel], *, out_stem: Path, ymax: float | None = None) -> list[Path]:
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(panels, ymax=ymax)
    written = []
    with plt.rc_context(_RC):
        for ext in (".png", ".pdf", ".svg"):
            fig.savefig(out_stem.with_suffix(ext))
            written.append(out_stem.with_suffix(ext))
    plt.close(fig)
    table = out_stem.with_suffix(".csv")  # the numbers behind every bar
    pd.DataFrame([dataclasses.asdict(p) for p in panels]).rename(
        columns={"p": "p_wilcoxon"}).to_csv(table, index=False)
    written.append(table)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", action="append", default=None,
                    help='"DATASET=ODOR[=HOURS]", one per panel, left to right; '
                         'HOURS overrides the starvation time read from the dataset name.')
    ap.add_argument("--predictions-csv", type=Path, default=CSV / "model_predictions.csv")
    ap.add_argument("--flagged-flies-csv", type=Path, default=FLAGGED)
    ap.add_argument("--out-stem", type=Path, default=DEFAULT_OUT / "IAA-0h_vs_3Oct-0h_cs_plus_pre_post")
    ap.add_argument("--ymax", type=float, default=None)
    args = ap.parse_args(argv)
    specs = [parse_panel(s) for s in (args.panel or DEFAULT_PANELS)]
    panels = load_panels(specs, predictions_csv=args.predictions_csv,
                         flagged_flies_csv=args.flagged_flies_csv,
                         tmp_dir=args.out_stem.parent / "_inputs")
    for p in panels:
        print(f"{p.dataset:26s} {p.odor:16s} n={p.n:3d} pre {p.mean_pre:.2f}±{p.sem_pre:.2f} "
              f"post {p.mean_post:.2f}±{p.sem_post:.2f} p={p.p}")
    for path in render(panels, out_stem=args.out_stem, ymax=args.ymax):
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
