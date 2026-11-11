#!/usr/bin/env python3
"""How far, and how many flies, above their red line at each moment, pre vs post.

For one cohort: one figure per odor plus one with every odor, two panels each
(the drafts "B" and "C" of ``Mean-Trace-Options_DRAFT``, 2026-11-10):

    left   mean over flies of the graded rasters' value: each trace's extension
           above its resting position, counted only while above its own red line
           (theta), 0 otherwise, so a small dip below rest cannot cancel a response
           the way it does on the mean trace (mean +/- SEM)
    right  % of flies above their red line at each moment, of the flies tracked
           then (1 s running mean, 95 % Wilson band)

Pre- and post-exposure on the same axes; the flies are the score figures' fly x odor
pairs (as on the cumulative-responders figures), one trial each. Both panels sit
near 0 in the last seconds before the odor by construction: that is where each
trial's resting position and red line are measured (hatched).

Outputs (``--out-dir``)::

    <Odor>_above_threshold.png     one per odor, named as Mean-Traces_* names them
    All_odors_above_threshold.png  every odor, one row each, sensitized odor first
    above_threshold_summary.csv    per odor x phase: flies, and both measures averaged
                                   before and during the odor
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

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scripts.analysis.pretest_vs_test_comparison as pvt  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.binarized_per_rasters import resolve_threshold_rule  # noqa: E402
from scripts.analysis.cumulative_responders import (  # noqa: E402
    odor_order,
    phase_labels,
    sensitized_odor,
)
from scripts.analysis.every_fly_heatmaps import MAX_TIME_S, load_rows  # noqa: E402

FOLDER = "Above-Threshold-Traces_Pre-Test_vs_Post-Training"
TAG = "above_threshold"
COLORS = {"pre": pvt.PRE_COLOR, "post": pvt.POST_COLOR}
INK, INK_2, GRID, ODOR_BG = "#0b0b0b", "#52514e", "#e4e3df", "#efeee9"
DPI = 200

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.facecolor": "white", "figure.facecolor": "white",
}


# ---------------------------------------------------------------------------
# The two measures
# ---------------------------------------------------------------------------


def mean_extension(rows: Mapping[tuple, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """``(mean, sem, n flies)`` of the graded rows at each frame; an untracked frame
    counts as not above the line (0), as ``binarize`` treats it."""
    stack = np.nan_to_num(np.vstack(list(rows.values())), nan=0.0)
    n = stack.shape[0]
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(stack.shape[1])
    return stack.mean(axis=0), sem, n


def _wilson(k: np.ndarray, n: np.ndarray, z: float = 1.959964) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        p = np.where(n > 0, k / np.maximum(n, 1e-12), np.nan)
        den = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def percent_above(above: np.ndarray, tracked: np.ndarray, *, smooth_frames: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(%, 95 % low, 95 % high)`` of the tracked flies above their line at each frame,
    both counts smoothed by a ``smooth_frames`` running mean."""
    k = np.asarray(above, dtype=float).sum(axis=0)
    n = np.asarray(tracked, dtype=float).sum(axis=0)
    if smooth_frames > 1:
        w = np.ones(smooth_frames) / smooth_frames
        k, n = np.convolve(k, w, "same"), np.convolve(n, w, "same")
    lo, hi = _wilson(k, n)
    with np.errstate(all="ignore"):
        pct = np.where(n > 0, 100.0 * k / np.maximum(n, 1e-12), np.nan)
    return pct, 100.0 * lo, 100.0 * hi


def curves(rows: Mapping[tuple, np.ndarray], *, fps: float) -> dict:
    stack = np.vstack(list(rows.values()))
    mean, sem, n = mean_extension(rows)
    pct, lo, hi = percent_above(np.nan_to_num(stack, nan=0.0) > 0, np.isfinite(stack),
                                smooth_frames=max(1, int(round(fps))))
    raw, _, _ = percent_above(np.nan_to_num(stack, nan=0.0) > 0, np.isfinite(stack),
                              smooth_frames=1)
    return {"mean": mean, "sem": sem, "n": n, "pct": pct, "lo": lo, "hi": hi, "pct_raw": raw}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

PANELS = (("extension", "Extension above rest, while above\nthe red line (% of range, mean)"),
          ("percent", "Flies above their red line (%)"))


def _frame(ax, *, on_s: float, off_s: float, anchor_s: float | None, label_size: float) -> None:
    ax.axvspan(on_s, off_s, color=ODOR_BG, zorder=0, lw=0)
    if anchor_s:
        ax.axvspan(on_s - anchor_s, on_s, facecolor="none", edgecolor="#b9b7ae", hatch="////",
                   lw=0, zorder=0)
    ax.text((on_s + off_s) / 2, 0.985, "odor", ha="center", va="top", color=INK_2,
            fontsize=label_size, transform=ax.get_xaxis_transform())
    ax.set_xlim(0, MAX_TIME_S)
    ax.set_xticks(range(0, int(MAX_TIME_S) + 1, 10))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def draw(ax, by_phase: Mapping[str, dict], panel: str, *, labels: Mapping[str, str],
         fps: float, on_s: float, off_s: float, anchor_s: float | None, ylim: tuple,
         legend_size: float = 8.5, label_size: float = 8.5) -> None:
    _frame(ax, on_s=on_s, off_s=off_s, anchor_s=anchor_s, label_size=label_size)
    for key in ("pre", "post"):
        c = by_phase[key]
        t = np.arange(c["mean"].size) / fps
        if panel == "extension":
            y, lo, hi = c["mean"], c["mean"] - c["sem"], c["mean"] + c["sem"]
        else:
            y, lo, hi = c["pct"], c["lo"], c["hi"]
        ax.fill_between(t, lo, hi, color=COLORS[key], alpha=0.16, lw=0, zorder=2)
        ax.plot(t, y, color=COLORS[key], lw=1.6, zorder=3, label=f"{labels[key]} (n={c['n']})")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", fontsize=legend_size)


def y_limits(by_phase: Mapping[str, dict]) -> dict[str, tuple[float, float]]:
    """One range per panel for both phases of an odor, from 0 to the highest band."""
    top = {"extension": max(float(np.nanmax(c["mean"] + c["sem"])) for c in by_phase.values()),
           "percent": max(float(np.nanmax(c["hi"])) for c in by_phase.values())}
    return {k: (0.0, max(v * 1.12, 1.0)) for k, v in top.items()}


def _note(threshold: int, anchor_s: float | None) -> str:
    lines = ["Left: mean over flies of each trace's extension above its resting position, "
             "counted only while above its own red line (0 otherwise), ± SEM.",
             "Right: % of the tracked flies above their red line, 1 s running mean, 95% "
             "Wilson band. The flies scored in both phases, one trial each."]
    if anchor_s:
        lines.append(f"Hatched: the last {anchor_s:g} s before the odor, where each trial's "
                     "rest and red line are measured, so both sit near 0 there.")
    return "\n".join(lines)


def plot_odor(path: Path, by_phase: Mapping[str, dict], *, odor: str, sensitized: bool,
              dataset: str, label: str, labels: Mapping[str, str], fps: float, on_s: float,
              off_s: float, anchor_s: float | None, threshold: int) -> None:
    ylim = y_limits(by_phase)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
        for ax, (panel, ylab) in zip(axes, PANELS):
            draw(ax, by_phase, panel, labels=labels, fps=fps, on_s=on_s, off_s=off_s,
                 anchor_s=anchor_s, ylim=ylim[panel])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylab)
        where = f"{dataset}, {label}" if label else dataset
        name = f"{odor} ({phase_terms.odor_tag(dataset)})" if sensitized else odor
        fig.text(0.06, 0.985, f"{name}: above the red line, pre- vs post-exposure",
                 fontweight="bold", fontsize=12, color=INK, va="top")
        fig.text(0.06, 0.935, f"{where}.\n" + _note(threshold, anchor_s), fontsize=8.4,
                 color=INK_2, va="top")
        fig.subplots_adjust(left=0.06, right=0.99, wspace=0.2, top=0.74, bottom=0.1)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_all(path: Path, by_odor: Mapping[str, Mapping[str, dict]], order: Sequence[str],
             sensitized: str | None, *, dataset: str, label: str, labels: Mapping[str, str],
             fps: float, on_s: float, off_s: float, anchor_s: float | None,
             threshold: int) -> None:
    """Every odor, one row each: extension left, % of flies right; scales per row."""
    pw, ph = 4.6, 1.9
    left, wgap, right = 0.95, 0.75, 0.2
    head, title, hgap, bottom = 1.25, 0.38, 0.3, 0.6
    n = len(order)
    fw = left + 2 * pw + wgap + right
    fh = head + n * (title + ph) + (n - 1) * hgap + bottom
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(fw, fh))
        for r, odor in enumerate(order):
            by_phase = by_odor[odor]
            ylim = y_limits(by_phase)
            top = fh - head - r * (title + ph + hgap) - title
            for c, (panel, ylab) in enumerate(PANELS):
                ax = fig.add_axes(((left + c * (pw + wgap)) / fw, (top - ph) / fh, pw / fw,
                                   ph / fh))
                draw(ax, by_phase, panel, labels=labels, fps=fps, on_s=on_s, off_s=off_s,
                     anchor_s=anchor_s, ylim=ylim[panel], legend_size=7.5, label_size=7.5)
                ax.tick_params(labelsize=8)
                ax.set_ylabel(ylab, fontsize=7.5)
                if c == 0:
                    tag = f" ({phase_terms.odor_tag(dataset)})" if odor == sensitized else ""
                    ax.set_title(f"{odor}{tag}", loc="left", fontweight="bold", color=INK,
                                 fontsize=10, pad=6)
                if r == n - 1:
                    ax.set_xlabel("Time (s)", fontsize=9)
        where = f"{dataset}, {label}" if label else dataset
        fig.text(left / fw, 1 - 0.15 / fh, f"{where}: above the red line, pre- vs "
                 "post-exposure, every odor", fontweight="bold", fontsize=12, color=INK,
                 va="top")
        fig.text(left / fw, 1 - 0.42 / fh, _note(threshold, anchor_s)
                 + " Each row has its own y-scale.", fontsize=8.2, color=INK_2, va="top")
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
                   help="the score figures' responder call; named in the notes only")
    p.add_argument("--fps", type=float, default=40.0)
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
        print(f"[above-threshold] {args.dataset}: no fly was scored in both phases; "
              "nothing to draw")
        return 0
    anchor_s = getattr(resolve_threshold_rule(args.config), "anchor_s", None)
    on_s = float(np.median([a for a, _ in rows["windows"]]))
    off_s = float(np.median([z for _, z in rows["windows"]]))
    labels = phase_labels(args.dataset)
    odors = sorted(paired["odor"].unique())
    sensitized = sensitized_odor(args.dataset, odors)
    order = [o for o in odor_order(odors, sensitized)
             if all(any(k[2] == o for k in rows[p]) for p in ("pre", "post"))]
    on, off = int(round(on_s * args.fps)), int(round(off_s * args.fps))
    by_odor, summary = {}, []
    for odor in order:
        by_odor[odor] = {}
        for key in ("pre", "post"):
            c = curves({k: v for k, v in rows[key].items() if k[2] == odor}, fps=args.fps)
            by_odor[odor][key] = c
            summary.append({"odor": odor, "sensitized": odor == sensitized,
                            "phase": labels[key], "n_flies": c["n"],
                            "mean_extension_before": float(np.mean(c["mean"][:on])),
                            "mean_extension_during": float(np.mean(c["mean"][on:off])),
                            "pct_above_before": float(np.nanmean(c["pct_raw"][:on])),
                            "pct_above_during": float(np.nanmean(c["pct_raw"][on:off]))})
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob(f"*_{TAG}.png"), out / f"{TAG}_summary.csv"]:
        old.unlink(missing_ok=True)
    pd.DataFrame(summary).to_csv(out / f"{TAG}_summary.csv", index=False)
    common = dict(dataset=args.dataset, label=args.label, labels=labels, fps=args.fps,
                  on_s=on_s, off_s=off_s, anchor_s=anchor_s, threshold=args.binary_threshold)
    for odor in order:
        plot_odor(out / odor_file(odor), by_odor[odor], odor=odor,
                  sensitized=odor == sensitized, **common)
    plot_all(out / f"All_odors_{TAG}.png", by_odor, order, sensitized, **common)
    print(f"[above-threshold] {args.dataset}: {len(order)} odors -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
