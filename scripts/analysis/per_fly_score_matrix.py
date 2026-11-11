#!/usr/bin/env python3
"""Every fly's PER score, pre-test and post-training, for every odor -- one row per fly.

A score matrix (not a yes/no reaction matrix): each cell is the model's PER
score for that fly and odor, coloured with the repo's fixed SCORE_COLORS
(purple = no response ... green = strong; >= 2 counts as a response), the
number printed inside. Each odor is a [pre | post] pair; the trained odor comes
first and is outlined. A last column shows the trained-odor change
(post - pre): green = went up, purple = went down.

Rows are named ``<batch folder>  fly <n>`` and ordered earliest day first, then
batch, then rig (rig 1 = the folder without a ``_rig_`` suffix), then fly, so
one rig's flies sit together. A thin dark line separates rigs, a thick one
separates batches/days. Flagged flies (FLY-State 0/-1) are left out unless
``--include-flagged`` (then their label says so).

Usage
-----
    python3 scripts/analysis/per_fly_score_matrix.py --dataset IAA-Sensitivity-1-1
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scripts.analysis.starvation_pre_post_comparison as spc  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.day_vs_day_report import folder_batch, folder_date, trained_odor  # noqa: E402
from scripts.analysis.pretest_vs_test_comparison import INK, INK_MUTED, _RC_CONTEXT, load_paired_scores  # noqa: E402
from scripts.analysis.score_summary import SCORE_COLORS, _score_cmap  # noqa: E402

CSV = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
FLAGGED = CSV / "flagged-flys-truth.csv"
OUT_ROOT = Path("/home/ramanlab/Documents/cole/Results/Figures")
_RIG_RE = re.compile(r"_rig_(\d+)$")
#: Change column: the findings figures' diverging pair (purple down, green up),
#: validated with the dataviz validator; gray-white midpoint = no change.
CHANGE_CMAP = LinearSegmentedColormap.from_list("change", ["#8a4f9e", "#f4f3f0", "#2f8a44"])
PHASE_LABEL = {"pre": "pre", "post": "post"}


def folder_key(folder: str) -> tuple[dt.date, int, int]:
    """(day, batch, rig) of a batch folder; rig 1 is the folder with no ``_rig_`` suffix."""
    m = _RIG_RE.search(folder)
    return folder_date(folder), folder_batch(folder), int(m.group(1)) if m else 1


def score_matrix(paired: pd.DataFrame, *, trained: str) -> pd.DataFrame:
    """Rows (folder, fly_number) in day/batch/rig/fly order; columns (odor, pre|post)."""
    p = paired.copy()
    p["fly_number"] = p["fly_number"].astype(int)
    long = p.melt(id_vars=["fly", "fly_number", "odor"], value_vars=["score_pre", "score_post"],
                  var_name="phase", value_name="score")
    long["phase"] = long["phase"].str.replace("score_", "", regex=False)
    m = long.pivot_table(index=["fly", "fly_number"], columns=["odor", "phase"], values="score",
                         aggfunc="first")
    odors = [trained] + sorted((o for o in m.columns.get_level_values(0).unique() if o != trained),
                               key=str.casefold)
    odors = [o for o in odors if o in m.columns.get_level_values(0)]
    cols = [(o, ph) for o in odors for ph in ("pre", "post") if (o, ph) in m.columns]
    rows = sorted(m.index, key=lambda k: (*folder_key(k[0]), k[1]))
    return m.loc[rows, cols]


def boundaries(rows: Sequence[tuple[str, int]]) -> tuple[list[int], list[int]]:
    """Row indices where a new rig starts (same batch) and where a new batch/day starts."""
    rig, batch = [], []
    for i in range(1, len(rows)):
        (d0, b0, r0), (d1, b1, r1) = folder_key(rows[i - 1][0]), folder_key(rows[i][0])
        if (d0, b0) != (d1, b1):
            batch.append(i)
        elif r0 != r1:
            rig.append(i)
    return rig, batch


def build_figure(m: pd.DataFrame, *, trained: str, title: str, flagged: set | None = None,
                 n_removed: int = 0):
    flagged = flagged or set()
    vals = m.to_numpy(float)
    n_rows, n_cols = vals.shape
    odors = list(dict.fromkeys(m.columns.get_level_values(0)))
    cmap, norm = _score_cmap()
    change = None
    if (trained, "pre") in m.columns and (trained, "post") in m.columns:
        change = (m[(trained, "post")] - m[(trained, "pre")]).to_numpy(float)
    rig_b, batch_b = boundaries(list(m.index))
    with plt.rc_context(_RC_CONTEXT):
        fig = plt.figure(figsize=(0.46 * n_cols + 5.4, 0.25 * n_rows + 2.9))
        gs = fig.add_gridspec(1, 3, width_ratios=[n_cols, 1.3, 0.45], wspace=0.06)
        ax = fig.add_subplot(gs[0, 0])
        cx = fig.add_subplot(gs[0, 1], sharey=ax)
        lx = fig.add_subplot(gs[0, 2])
        ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        for i in range(n_rows):
            for j in range(n_cols):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                            color="white" if v >= 3 or v <= -1 else INK)
        # cell gaps (surface), pair gaps (wider), rig / batch separators (ink)
        for j in range(1, n_cols):
            new_odor = m.columns[j][0] != m.columns[j - 1][0]
            ax.axvline(j - 0.5, color="white", linewidth=3.5 if new_odor else 0.8, zorder=2)
        for i in range(1, n_rows):
            ax.axhline(i - 0.5, color="white", linewidth=0.8, zorder=2)
        for axis in (ax, cx):
            for i in rig_b:
                axis.axhline(i - 0.5, color=INK_MUTED, linewidth=1.2, zorder=3)
            for i in batch_b:
                axis.axhline(i - 0.5, color=INK, linewidth=2.6, zorder=3)
        if (trained, "pre") in m.columns:
            j0 = list(m.columns).index((trained, "pre"))
            ax.add_patch(Rectangle((j0 - 0.5, -0.5), 2, n_rows, fill=False, edgecolor=INK,
                                   linewidth=2.2, zorder=4, clip_on=False))
        # x axis on top: pre/post ticks, odor names above each pair
        ax.xaxis.tick_top()
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([PHASE_LABEL[ph] for _o, ph in m.columns], fontsize=7.5)
        ax.tick_params(axis="x", length=0, pad=2)
        for o in odors:
            js = [j for j, (oo, _ph) in enumerate(m.columns) if oo == o]
            # title is the dataset name (main), so it carries the cohort's wording
            label = f"{o}\n({phase_terms.odor_tag(title, 'trained')})" if o == trained else o
            ax.text(np.mean(js), -1.5, label, ha="center", va="bottom", fontsize=8.5, color=INK,
                    fontweight="bold" if o == trained else "normal", rotation=0,
                    wrap=True)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"{f}  fly {n}" + ("  (flagged)" if (f, n) in flagged else "")
                            for f, n in m.index], fontsize=7.5)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

        if change is not None:
            lim = max(2.0, float(np.nanmax(np.abs(change)))) if np.isfinite(change).any() else 2.0
            cx.imshow(change[:, None], cmap=CHANGE_CMAP, vmin=-lim, vmax=lim, aspect="auto",
                      interpolation="nearest")
            for i, v in enumerate(change):
                if np.isfinite(v):
                    cx.text(0, i, f"{v:+.0f}" if v else "0", ha="center", va="center", fontsize=7.5,
                            color="white" if abs(v) > 0.55 * lim else INK,
                            fontweight="bold" if v else "normal")
            for i in range(1, n_rows):
                cx.axhline(i - 0.5, color="white", linewidth=0.8, zorder=2)
            cx.xaxis.tick_top()
            cx.set_xticks([0])
            cx.set_xticklabels(["post − pre"], fontsize=7.5)
            cx.text(0, -1.5, f"{trained}\nchange", ha="center", va="bottom", fontsize=8.5,
                    color=INK, fontweight="bold")
        cx.tick_params(axis="both", length=0, labelleft=False)
        for sp in cx.spines.values():
            sp.set_visible(False)

        # score legend: one swatch per score, -1 .. 5
        scores = list(range(-1, len(SCORE_COLORS) - 1))
        lx.imshow(np.array(scores)[::-1, None], cmap=cmap, norm=norm, aspect="auto")
        lx.set_yticks(range(len(scores)))
        lx.set_yticklabels([str(s) for s in scores[::-1]], fontsize=8)
        lx.yaxis.tick_right()
        lx.set_xticks([])
        lx.set_title("PER\nscore", fontsize=8.5, color=INK)
        lx.text(1.9, len(scores) - 1 - (scores.index(2) - 0.5), "← ≥ 2 = response",
                fontsize=7, color=INK_MUTED, va="center", transform=lx.transData, clip_on=False)
        lx.tick_params(length=0)
        for sp in lx.spines.values():
            sp.set_visible(False)
        n_flies = n_rows
        note = (f"{n_flies} flies" + (f" ({n_removed} flagged flies left out)" if n_removed else "")
                + " · rows: earliest day first, then batch, then rig, then fly · "
                  "thin line = new rig, thick line = new batch")
        fig.suptitle(f"{title} — every fly's PER score, "
                     f"{phase_terms.pre_label(title, 'pre-test', case='lower')} vs "
                     f"{phase_terms.post_label(title, 'post-training', case='lower')}", x=0.01,
                     ha="left", fontsize=12, fontweight="bold", color=INK,
                     y=1 - 0.12 / fig.get_size_inches()[1])
        fig.text(0.01, 1 - 0.42 / fig.get_size_inches()[1], note, ha="left", va="top",
                 fontsize=9, color=INK)
        H = fig.get_size_inches()[1]
        fig.subplots_adjust(top=1 - 1.3 / H, bottom=0.02, left=0.24, right=0.93)
        # Score key: a short block at the top (layout first, then shrink -- subplots_adjust
        # would otherwise stretch it back to the full matrix height).
        b = lx.get_position()
        h = min(b.height, 2.4 / H)
        lx.set_position([b.x0 + 0.01, b.y1 - h, b.width, h])
    return fig


def load(dataset: str, *, predictions_csv: Path, flagged_csv: Path, include_flagged: bool,
         tmp_dir: Path, after: dt.date | None = None):
    groups = [spc.Group(dataset, dataset, "")]
    preds = spc.subset_and_relabel(pd.read_csv(predictions_csv), groups)
    if after is not None:
        preds = preds[preds["fly"].map(folder_date) > after]
    from fbpipe.config import canon_fly_number, load_flagged_fly_exclusions

    excluded = load_flagged_fly_exclusions(flagged_csv) if Path(flagged_csv).exists() else set()
    keys = {(f, int(canon_fly_number(n))) for d, f, n in excluded if d == dataset}
    present = {(f, int(n)) for f, n in zip(preds["fly"], preds["fly_number"])}
    flagged_here = keys & present
    if not include_flagged:
        preds = spc.drop_flagged(preds, flagged_csv)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "predictions_subset_relabelled.csv"
    preds.to_csv(path, index=False)
    paired = load_paired_scores(path, flagged_flies_csv="" if include_flagged else str(flagged_csv))
    return paired, flagged_here


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="IAA-Sensitivity-1-1")
    ap.add_argument("--after", type=dt.date.fromisoformat, default=None,
                    help="only batch folders recorded strictly after this date")
    ap.add_argument("--include-flagged", action="store_true",
                    help="keep flagged flies (their row label says '(flagged)')")
    ap.add_argument("--trained-odor", default=None, help="default: read from the dataset name")
    ap.add_argument("--predictions-csv", type=Path, default=CSV / "model_predictions.csv")
    ap.add_argument("--flagged-flies-csv", type=Path, default=FLAGGED)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    trained = args.trained_odor or trained_odor(args.dataset)
    out = args.out_dir or OUT_ROOT / f"{args.dataset}_per_fly_scores"
    paired, flagged = load(args.dataset, predictions_csv=args.predictions_csv,
                           flagged_csv=args.flagged_flies_csv, include_flagged=args.include_flagged,
                           tmp_dir=out / "inputs", after=args.after)
    if paired.empty:
        raise SystemExit(f"no paired pre/post scores for {args.dataset}")
    m = score_matrix(paired, trained=trained)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (f"{args.dataset}_per_fly_scores" + ("_with_flagged" if args.include_flagged else ""))
    table = m.copy()
    table.columns = [f"{o} {ph}" for o, ph in table.columns]
    table.to_csv(f"{stem}.csv")
    fig = build_figure(m, trained=trained, title=args.dataset,
                       flagged=flagged if args.include_flagged else set(),
                       n_removed=0 if args.include_flagged else len(flagged))
    with plt.rc_context(_RC_CONTEXT):
        for ext in (".png", ".pdf"):
            fig.savefig(f"{stem}{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"{len(m)} flies, {m.shape[1] // 2} odors -> {stem}.png / .pdf / .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
