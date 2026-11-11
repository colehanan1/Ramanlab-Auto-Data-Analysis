#!/usr/bin/env python3
"""Every fly's response to one odor, pre- and post-exposure side by side.

For one cohort, one figure per odor: two heatmaps, pre- and post-exposure. Rows
are the flies scored in both phases (the score figures' fly x odor pairs, as on
the cumulative-responders figures), in one order for both panels: strongest first,
by each fly's mean extension during the odor, pre + post. A cell is the graded
rasters' value (``binarized_per_rasters.build_trials``): the proboscis extension
above the fly's resting position, counted only while above its own red line
(theta), 0 below it, grey where the tracker lost the proboscis. A dot marks a fly
the model scores as responding in that phase. The whole 90 s trial is drawn, so
extensions after the odor show too.

Unlike Graded-PER-Rasters (one figure per phase with every odor, rows ordered by
the exposure trials), this puts one odor's two phases next to each other, the
draft "E" of ``Mean-Trace-Options_DRAFT`` (2026-11-10).

Outputs (``--out-dir``)::

    <Odor>_every_fly_heatmap.png   one per odor, named as Mean-Traces_* names them
    every_fly_heatmap_order.csv    each odor's row order, with the flies' scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scripts.analysis.pretest_vs_test_comparison as pvt  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.binarized_per_rasters import (  # noqa: E402
    build_trials,
    resolve_threshold_rule,
)
from scripts.analysis.cumulative_responders import (  # noqa: E402
    PHASES,
    odor_order,
    paired_trials,
    phase_labels,
    sensitized_odor,
)

FOLDER = "Every-Fly-Heatmaps_Pre-Test_vs_Post-Training"
TAG = "every_fly_heatmap"
MAX_TIME_S = 90.0
VMAX = 40.0                   # % of range: one scale for every odor and cohort
COLORS = {"pre": pvt.PRE_COLOR, "post": pvt.POST_COLOR}
INK, INK_2, NAN_BG = "#0b0b0b", "#52514e", "#d9d8d3"
#: Sequential single-hue ramp, surface at zero (the draft's, light to dark blue).
RAMP = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
DPI = 200

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
}


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _read_wide(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)[:n]
    return np.concatenate([a, np.full(n - a.size, np.nan)]) if a.size < n else a


def load_rows(pretest: Path, testing: Path, predictions: Path, *, config: Path, dataset: str,
              n_frames: int, flagged_flies_csv: str = "") -> dict:
    """The graded row of every paired trial, by phase key: ``{"pre": {(fly, fly_number,
    odor): row}, "post": {...}, "paired": the pairs, "windows": [(on_s, off_s), ...]}``.

    A row is ``build_trials``' graded value, NaN where the trace is NaN (no
    tracking), frame 0 = the trial's start, cut or NaN-padded to ``n_frames``."""
    paired = paired_trials(predictions, dataset=dataset, flagged_flies_csv=flagged_flies_csv)
    out: dict = {"pre": {}, "post": {}, "paired": paired, "windows": []}
    if paired.empty:
        return out
    rule = resolve_threshold_rule(config)
    for key, trial_type, trial_col, _ in PHASES:
        trials = build_trials(_read_wide(pretest if key == "pre" else testing),
                              dataset=dataset, trial_type=trial_type, rule=rule)
        index = ({(str(r.fly).strip(), int(r.fly_number), int(r.trial_index)): r
                  for r in trials.itertuples(index=False)} if not trials.empty else {})
        for p in paired.itertuples(index=False):
            r = index.get((p.fly, int(p.fly_number), int(getattr(p, trial_col))))
            if r is None:
                continue
            row = np.asarray(r.graded, dtype=float).copy()
            row[~np.isfinite(np.asarray(r.trace, dtype=float))] = np.nan
            out[key][(p.fly, int(p.fly_number), p.odor)] = _fit(row, n_frames)
            out["windows"].append((float(r.odor_on_s), float(r.odor_off_s)))
    return out


def fly_order(pre: Mapping[tuple, np.ndarray], post: Mapping[tuple, np.ndarray], *, odor: str,
              on: int, off: int) -> list[tuple[str, int]]:
    """The flies drawn in both phases, strongest first: mean extension during the odor
    (frames ``on`` to ``off``), pre + post; ties by fly."""
    flies = sorted({(f, n) for f, n, o in pre if o == odor} & {(f, n) for f, n, o in post
                                                               if o == odor})

    def strength(fly: tuple[str, int]) -> float:
        with np.errstate(all="ignore"):
            vals = [np.nanmean(rows[(*fly, odor)][on:off]) for rows in (pre, post)]
        return float(np.nansum(vals))

    return sorted(flies, key=lambda fly: (-strength(fly), fly))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot_odor(path: Path, odor: str, order: Sequence[tuple[str, int]], rows: Mapping,
              responding: Mapping[str, Mapping[tuple, bool]], *, sensitized: bool,
              dataset: str, label: str, labels: Mapping[str, str], threshold: int,
              fps: float, on_s: float, off_s: float, vmax: float = VMAX) -> None:
    n = len(order)
    height = max(4.4, 2.3 + 0.1 * n)
    cmap = LinearSegmentedColormap.from_list("every_fly", RAMP)
    cmap.set_bad(NAN_BG)
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(11.5, height))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.035], wspace=0.14, figure=fig,
                      top=1 - 1.25 / height, bottom=0.65 / height, left=0.07, right=0.93)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        im = None
        for ax, key in zip(axes, ("pre", "post")):
            img = np.vstack([rows[key][(f, num, odor)] for f, num in order])
            im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                           extent=(0, img.shape[1] / fps, n, 0), interpolation="nearest")
            resp = np.array([responding[key].get((f, num), False) for f, num in order])
            ys = np.flatnonzero(resp) + 0.5
            ax.scatter(np.full(ys.size, MAX_TIME_S * 1.024), ys, s=9, color=COLORS[key],
                       clip_on=False, marker="o", lw=0)
            for x in (on_s, off_s):
                ax.axvline(x, color=INK, lw=0.9, ls=(0, (3, 2)))
            ax.set_xlim(0, MAX_TIME_S)
            ax.set_xticks(range(0, int(MAX_TIME_S) + 1, 15))
            ax.set_yticks([])
            ax.set_xlabel("Time (s)")
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{labels[key]}: {int(resp.sum())} of {n} flies respond (●)",
                         loc="left", color=INK, fontsize=10.5)
        axes[0].set_ylabel(f"{n} flies, same order in both panels")
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))   # its own column: no shrink
        cb.set_label("Extension above rest while above\nthe red line (% of range)")
        cb.outline.set_visible(False)
        where = f"{dataset}, {label}" if label else dataset
        name = f"{odor} ({phase_terms.odor_tag(dataset)})" if sensitized else odor
        fig.text(0.07, 1 - 0.15 / height, f"{name}: every fly, pre- vs post-exposure",
                 fontweight="bold", fontsize=12, color=INK, va="top")
        fig.text(0.07, 1 - 0.47 / height,
                 f"{where}. Rows: the flies scored in both phases, strongest during the odor "
                 "(pre + post) first. Cell: extension above rest while above the red line,\nas "
                 "on Graded-PER-Rasters; grey = no tracking. Dashed: odor on and off. ● = "
                 f"model score {threshold} or more, as on the score bars.",
                 fontsize=8.6, color=INK_2, va="top")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def odor_file(odor: str) -> str:
    return f"{str(odor).replace('/', '-').replace(' ', '_')}_{TAG}.png"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pretest-wide-csv", required=True, type=Path)
    p.add_argument("--testing-wide-csv", required=True, type=Path)
    p.add_argument("--predictions-csv", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path,
                   help="the red line: analysis.combined.combined_base.wide of this config")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--flagged-flies-csv", default="")
    p.add_argument("--binary-threshold", type=int, default=pvt.BINARY_THRESHOLD,
                   help="model score counted as responding (the score figures' own)")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--vmax", type=float, default=VMAX, help="top of the colour scale, %%")
    p.add_argument("--label", default="", help="which flies, for the titles")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    n_frames = int(round(MAX_TIME_S * args.fps))
    rows = load_rows(args.pretest_wide_csv, args.testing_wide_csv, args.predictions_csv,
                     config=args.config, dataset=args.dataset, n_frames=n_frames,
                     flagged_flies_csv=str(args.flagged_flies_csv or ""))
    paired = rows["paired"]
    if paired.empty or not rows["pre"]:
        print(f"[every-fly] {args.dataset}: no fly was scored in both phases; nothing to draw")
        return 0
    on_s = float(np.median([a for a, _ in rows["windows"]]))
    off_s = float(np.median([z for _, z in rows["windows"]]))
    labels = phase_labels(args.dataset)
    odors = sorted(paired["odor"].unique())
    sensitized = sensitized_odor(args.dataset, odors)
    responding = {key: {(p.fly, int(p.fly_number), p.odor):
                        bool(float(getattr(p, score)) >= args.binary_threshold)
                        for p in paired.itertuples(index=False)}
                  for key, _, _, score in PHASES}
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob(f"*_{TAG}.png"), out / f"{TAG}_order.csv"]:
        old.unlink(missing_ok=True)
    order_rows = []
    for odor in odor_order(odors, sensitized):
        order = fly_order(rows["pre"], rows["post"], odor=odor,
                          on=int(round(on_s * args.fps)), off=int(round(off_s * args.fps)))
        if not order:
            continue
        flags = {key: {(f, n): v for (f, n, o), v in responding[key].items() if o == odor}
                 for key in ("pre", "post")}
        plot_odor(out / odor_file(odor), odor, order, rows, flags,
                  sensitized=odor == sensitized, dataset=args.dataset, label=args.label,
                  labels=labels, threshold=args.binary_threshold, fps=args.fps, on_s=on_s,
                  off_s=off_s, vmax=args.vmax)
        scores = paired[paired["odor"] == odor].set_index(["fly", "fly_number"])
        for rank, (f, n) in enumerate(order, 1):
            order_rows.append({"odor": odor, "rank": rank, "fly": f, "fly_number": n,
                               "score_pre": scores.loc[(f, n), "score_pre"],
                               "score_post": scores.loc[(f, n), "score_post"],
                               "responding_pre": flags["pre"][(f, n)],
                               "responding_post": flags["post"][(f, n)]})
    pd.DataFrame(order_rows).to_csv(out / f"{TAG}_order.csv", index=False)
    print(f"[every-fly] {args.dataset}: {len(odors)} odors -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
