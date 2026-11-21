#!/usr/bin/env python3
"""Mean PER trace of the flies scored as responding vs not, pre- vs post-exposure.

For one cohort: one figure per odor, plus one with every odor. Left, the trials the
model scores as responding (score >= ``--binary-threshold``); right, the rest; the
pre- and post-exposure means on the same axes. The groups are the score figures' own
(their fly x odor pairs and threshold, as on the cumulative-responders figures), and
each trace is the mean-trace figures' own: every trial minus the mean of its first
``--odor-on-s`` seconds (``baseline_correct``), then the mean +/- SEM over flies at
each frame (``_mean_sem_trace``). So the two panels are the cohort's
Mean-Traces_Pre-Test_vs_Post-Training curve, split in two.

Why: averaged over every fly, a response cancels against the small dip most flies
make at odor onset (``Mean-Trace-Options_DRAFT``, 2026-11-10). Split, both show.
The split uses the scores of the very trials drawn, so the left panel shows what a
response looks like, not whether there are more of them; how many respond is the
cumulative-responders figure's job.

Outputs (``--out-dir``)::

    <Odor>_responder_mean_traces.png     one per odor, named as Mean-Traces_* names them
    All_odors_responder_mean_traces.png  every odor, one row each, sensitized odor first
    responder_mean_traces_summary.csv    per odor x phase x group: flies, and the mean
                                         trace's average before / during / after the odor

With ``--with-median`` each PNG also gets ``<stem>_median.png`` (same figure, per-frame
median over the same per-fly traces, band = interquartile range) and
``<stem>_mean_vs_median.png`` (mean solid, median dashed, no bands); the summary gains
``median_*`` columns after the existing ones. Without the flag nothing changes.

Usage::

    python3 scripts/analysis/responder_mean_traces.py --dataset IAA-Sensitivity-1-1 \\
        --pretest-wide-csv _inputs/wide_pretest.parquet \\
        --testing-wide-csv _inputs/wide_testing.parquet \\
        --predictions-csv _inputs/predictions.csv \\
        --out-dir Responder-Mean-Traces_Pre-Test_vs_Post-Training
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
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
from fbpipe.analysis.traces import baseline_correct  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.binarized_per_rasters import parse_trial  # noqa: E402
from scripts.analysis.cumulative_responders import (  # noqa: E402
    PHASES,
    odor_order,
    paired_trials,
    phase_labels,
    sensitized_odor,
)
from scripts.analysis.dataset_means_specific_flies import (  # noqa: E402
    MAX_TIME_S,
    _mean_sem_trace,
    nan_pad_stack,
)

#: The folder start_window_report draws this into, beside Mean-Traces_Pre-Test_vs_Post-Training.
FOLDER = "Responder-Mean-Traces_Pre-Test_vs_Post-Training"
TAG = "responder_mean_traces"
#: (group, responding?, panel title)
GROUPS = (("responding", True, "Flies scored as responding"),
          ("not responding", False, "Flies scored as not responding"))
COLORS = {"pre": pvt.PRE_COLOR, "post": pvt.POST_COLOR}   # the score figures' validated pair
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
# Traces and groups
# ---------------------------------------------------------------------------


def phase_traces(wide: pd.DataFrame, paired: pd.DataFrame, *, phase: str,
                 baseline_frames: int, dataset: str | None = None
                 ) -> dict[tuple[str, int, str], np.ndarray]:
    """``{(fly, fly_number, odor): trace}`` for the paired trials of one phase.

    As the mean-trace figures take a trial (``collect_presentations``): frames after
    the last tracked one are cut, then the mean of the first ``baseline_frames`` is
    subtracted.
    """
    trial_type = {key: t for key, t, _, _ in PHASES}[phase]
    trial_col = {key: c for key, _, c, _ in PHASES}[phase]
    want = {(str(p.fly), int(p.fly_number), int(getattr(p, trial_col))): p.odor
            for p in paired.itertuples(index=False)}
    if dataset is not None and "dataset" in wide.columns:
        wide = wide[wide["dataset"].astype(str).str.strip().str.casefold()
                    == str(dataset).strip().casefold()]
    cols = sorted((c for c in wide.columns if str(c).startswith("dir_val_")),
                  key=lambda c: int(str(c).split("_")[-1]))
    values = wide[cols].to_numpy(dtype=float)
    out: dict[tuple[str, int, str], np.ndarray] = {}
    for i, (fly, num, label) in enumerate(zip(wide["fly"], wide["fly_number"],
                                               wide["trial_label"])):
        label = str(label).strip()
        parsed = parse_trial(label)
        if parsed is None or not label.lower().startswith(f"{trial_type}_"):
            continue
        key = (str(fly).strip(), int(num), parsed[0])
        odor = want.get(key)
        finite = np.isfinite(values[i])
        if odor is None or not finite.any():
            continue
        trace = values[i][: np.flatnonzero(finite)[-1] + 1]
        out[(key[0], key[1], odor)] = baseline_correct(trace, baseline_frames)
    return out


def _fit(a: np.ndarray, n: int, fill: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)[:n]
    return np.concatenate([a, np.full(n - a.size, fill)]) if a.size < n else a


def group_curves(traces: Mapping[tuple, np.ndarray], responding: Mapping[tuple, bool], *,
                 max_frames: int) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """``{group: (mean, sem, n flies)}``: ``_mean_sem_trace`` over each group's traces."""
    out = {}
    for group, want, _ in GROUPS:
        per_fly = {k: v for k, v in traces.items() if responding.get(k) is want}
        if per_fly:
            mean, sem = _mean_sem_trace(per_fly, max_frames)
        else:
            mean, sem = np.full(max_frames, np.nan), np.zeros(max_frames)
        out[group] = (_fit(mean, max_frames, np.nan), _fit(sem, max_frames, 0.0), len(per_fly))
    return out


def group_bands(traces: Mapping[tuple, np.ndarray], responding: Mapping[tuple, bool], *,
                max_frames: int, stat: str = "mean"
                ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """``{group: (centre, lower, upper, n flies)}`` for ``stat`` = ``"mean"`` (mean, mean -/+ SEM:
    exactly ``group_curves``) or ``"median"`` (per-frame ``nanmedian``, the 25th and 75th
    percentiles) over the same per-fly traces, padded and cut the same way."""
    if stat == "mean":
        return {g: (m, m - e, m + e, n)
                for g, (m, e, n) in group_curves(traces, responding,
                                                 max_frames=max_frames).items()}
    if stat != "median":
        raise ValueError(f"stat must be 'mean' or 'median', not {stat!r}")
    out = {}
    for group, want, _ in GROUPS:
        per_fly = {k: v for k, v in traces.items() if responding.get(k) is want}
        if per_fly:
            stacked = nan_pad_stack(list(per_fly.values()))[:, :max_frames]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN frames
                centre = np.nanmedian(stacked, axis=0)
                lo, hi = np.nanpercentile(stacked, [25, 75], axis=0)
        else:
            centre = lo = hi = np.full(max_frames, np.nan)
        out[group] = tuple(_fit(a, max_frames, np.nan) for a in (centre, lo, hi)) + (len(per_fly),)
    return out


def summary_rows(odor: str, curves: Mapping[str, Mapping[str, tuple]], *, sensitized: bool,
                 labels: Mapping[str, str], fps: float, odor_on_s: float,
                 odor_off_s: float, median_curves: Mapping | None = None) -> list[dict]:
    """``median_curves`` (``group_bands(stat="median")`` per phase) adds the same four window
    statistics of the median curve as ``median_*`` columns, after the existing ones."""
    on, off = int(round(odor_on_s * fps)), int(round(odor_off_s * fps))

    def avg(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

    rows = []
    for key in ("pre", "post"):
        for group, _, _ in GROUPS:
            mean, _, n = curves[key][group]
            rows.append({"odor": odor, "sensitized": sensitized, "phase": labels[key],
                         "group": group, "n_flies": n, "mean_before": avg(mean[:on]),
                         "mean_during": avg(mean[on:off]), "mean_after": avg(mean[off:]),
                         "peak_during": (float(np.nanmax(mean[on:off]))
                                         if np.isfinite(mean[on:off]).any() else float("nan"))})
            if median_curves is not None:
                med = median_curves[key][group][0]
                rows[-1].update({"median_before": avg(med[:on]), "median_during": avg(med[on:off]),
                                 "median_after": avg(med[off:]),
                                 "median_peak_during": (float(np.nanmax(med[on:off]))
                                                        if np.isfinite(med[on:off]).any()
                                                        else float("nan"))})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def as_bands(curves: Mapping[str, Mapping[str, tuple]]) -> dict:
    """``(mean, sem, n)`` curves as ``(centre, lower, upper, n)`` bands."""
    return {k: {g: (m, m - e, m + e, n) for g, (m, e, n) in groups.items()}
            for k, groups in curves.items()}


def _lines_only(*band_sets: Mapping) -> dict:
    """The centre lines of several band sets, as bands of zero width (no band drawn)."""
    return {f"{i}{k}": {g: (c, c, c, n) for g, (c, _, _, n) in groups.items()}
            for i, bands in enumerate(band_sets) for k, groups in bands.items()}


def y_range(curves: Mapping[str, Mapping[str, tuple]]) -> tuple[float, float]:
    """One range for both panels of an odor, so responders and the rest compare.
    ``curves`` are ``(centre, lower, upper, n)`` bands."""
    lo, hi = -1.0, 1.0
    for groups in curves.values():
        for centre, lower, upper, n in groups.values():
            if n and np.isfinite(centre).any():
                lo = min(lo, float(np.nanmin(lower)))
                hi = max(hi, float(np.nanmax(upper)))
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def draw_group(ax, curves: Mapping[str, Mapping[str, tuple]], group: str, *,
               labels: Mapping[str, str], fps: float, odor_on_s: float, odor_off_s: float,
               ylim: tuple[float, float], legend_size: float = 8.5,
               other: Mapping[str, Mapping[str, tuple]] | None = None,
               phases: Sequence[str] = ("pre", "post")) -> None:
    """``curves`` are ``(centre, lower, upper, n)`` bands, drawn as a line with a band. With
    ``other`` (the median bands) it is the mean-vs-median panel instead: ``curves`` solid in
    the phase colour, ``other`` black dashed, no bands; give it one phase per panel so the
    dashed line is never hidden under the other phase's lines."""
    ax.axvspan(odor_on_s, odor_off_s, color=ODOR_BG, zorder=0, lw=0)
    ax.axhline(0, color=INK_2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    for key in phases:
        mean, lower, upper, n = curves[key][group]
        t = np.arange(mean.size) / fps
        if n:
            if other is None:
                ax.fill_between(t, lower, upper, color=COLORS[key], alpha=0.18, lw=0, zorder=2)
            ax.plot(t, mean, color=COLORS[key], lw=1.6, zorder=3,
                    label=f"{labels[key]} (n={n})")
            if other is not None:
                ax.plot(t, other[key][group][0], color=INK, lw=1.2, ls="--", zorder=4)
        else:                                   # keep the count in the legend
            ax.plot([], [], color=COLORS[key], lw=1.6, label=f"{labels[key]} (n=0)")
    ax.set_xlim(0, MAX_TIME_S)
    ax.set_xticks(range(0, int(MAX_TIME_S) + 1, 10))
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles, names = ax.get_legend_handles_labels()
    if other is not None:                        # say which line style is which statistic
        handles += [plt.Line2D([], [], color=INK, lw=1.2, ls="--")]
        names += ["median (black dashed)"]
    ax.legend(handles, names, loc="upper left", fontsize=legend_size)
    ax.text((odor_on_s + odor_off_s) / 2, 0.985, "odor", ha="center", va="top", color=INK_2,
            fontsize=8.5, transform=ax.get_xaxis_transform())


def _y_label(odor_on_s: float) -> str:
    return f"PER, change from its own\n0–{odor_on_s:g} s mean (% of range)"


def _caveat(threshold: int) -> str:
    return (f"Split by the model score of these same trials (responding = {threshold} or more, "
            "as on the score bars), so the left side shows what a response looks like,\nnot "
            "whether there are more of them: see Cumulative-Responders for that.")


STATS = {  # figure kind -> (title word, what the band/lines are)
    "mean": ("mean", "Mean ± SEM over flies, one trial each."),
    "median": ("median", "Median and interquartile range (25th–75th percentile) over flies, "
                         "one trial each."),
    "both": ("mean vs median", "Solid line = mean, black dashed line = median, over the same "
                               "flies, one trial each; no bands; one phase per panel."),
}


def _odor_title(odor: str, sensitized: bool, dataset: str) -> str:
    return f"{odor} ({phase_terms.odor_tag(dataset)})" if sensitized else odor


def plot_odor(path: Path, curves: Mapping, *, odor: str, sensitized: bool, dataset: str,
              label: str, labels: Mapping[str, str], fps: float, odor_on_s: float,
              odor_off_s: float, threshold: int, stat: str = "mean",
              other: Mapping | None = None) -> None:
    """``curves`` are bands (``as_bands`` / ``group_bands``). ``stat`` "mean" or "median"
    draws that statistic with its band; "both" draws ``curves`` (mean) solid and ``other``
    (median) dashed."""
    word, note = STATS[stat]
    ylim = y_range(_lines_only(curves, other) if stat == "both" else curves)
    rows = (("pre",), ("post",)) if stat == "both" else (("pre", "post"),)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(len(rows), 2, figsize=(12.5, 5.6 if len(rows) == 1 else 9.4),
                                 squeeze=False)
        for r, phases in enumerate(rows):
            for ax, (group, _, title) in zip(axes[r], GROUPS):
                draw_group(ax, curves, group, labels=labels, fps=fps, odor_on_s=odor_on_s,
                           odor_off_s=odor_off_s, ylim=ylim, other=other, phases=phases)
                head = title if len(rows) == 1 else f"{title} · {labels[phases[0]]}"
                ax.set_title(head, loc="left", color=INK, fontsize=10.5, pad=8)
                if r == len(rows) - 1:
                    ax.set_xlabel("Time (s)")
                ax.set_ylabel(_y_label(odor_on_s))
        where = f"{dataset}, {label}" if label else dataset
        top_in = 1.25                            # inches kept for the title and caption
        h = fig.get_figheight()
        fig.text(0.06, 1 - 0.08 / h, f"{_odor_title(odor, sensitized, dataset)}: {word} PER "
                 "trace, responders vs the rest", fontweight="bold", fontsize=12, color=INK,
                 va="top")
        fig.text(0.06, 1 - 0.36 / h, f"{where}. {note}\n"
                 + _caveat(threshold), fontsize=8.6, color=INK_2, va="top")
        fig.subplots_adjust(left=0.06, right=0.99, wspace=0.22, hspace=0.35,
                            top=1 - top_in / h, bottom=0.55 / h)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_all(path: Path, by_odor: Mapping[str, Mapping], order: Sequence[str],
             sensitized: str | None, *, dataset: str, label: str,
             labels: Mapping[str, str], fps: float, odor_on_s: float, odor_off_s: float,
             threshold: int, stat: str = "mean", others: Mapping | None = None) -> None:
    """Every odor, one row each: responders left, the rest right, shared within a row.
    ``by_odor`` holds bands; ``stat`` and ``others`` as in ``plot_odor``."""
    word, note = STATS[stat]
    # mean vs median: a column per group and phase, so each panel holds one mean and its median
    cols = ([(g, gt if p == "pre" else "", (p,), f"{gt.replace('Flies scored as ', '')}"
              f" · {labels[p]}") for g, _, gt in GROUPS for p in ("pre", "post")]
            if stat == "both" else [(g, gt, ("pre", "post"), gt) for g, _, gt in GROUPS])
    pw, ph = (3.6, 1.9) if stat == "both" else (4.6, 1.9)       # panel
    left, wgap, right = 0.95, 0.55, 0.2
    head, title, hgap, bottom = 1.15, 0.38, 0.3, 0.6
    n = len(order)
    fw = left + len(cols) * pw + (len(cols) - 1) * wgap + right
    fh = head + n * (title + ph) + (n - 1) * hgap + bottom
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(fw, fh))
        for r, odor in enumerate(order):
            curves = by_odor[odor]
            other = others[odor] if others is not None else None
            ylim = y_range(_lines_only(curves, other) if stat == "both" else curves)
            top = fh - head - r * (title + ph + hgap) - title
            for c, (group, _, phases, _) in enumerate(cols):
                ax = fig.add_axes(((left + c * (pw + wgap)) / fw, (top - ph) / fh, pw / fw,
                                   ph / fh))
                draw_group(ax, curves, group, labels=labels, fps=fps, odor_on_s=odor_on_s,
                           odor_off_s=odor_off_s, ylim=ylim, legend_size=7.5,
                           other=other, phases=phases)
                ax.tick_params(labelsize=8)
                if c == 0:
                    ax.set_title(_odor_title(odor, odor == sensitized, dataset), loc="left",
                                 fontweight="bold", color=INK, fontsize=10, pad=6)
                    ax.set_ylabel("Change from own\n0–30 s mean (%)" if odor_on_s == 30
                                  else _y_label(odor_on_s), fontsize=8)
                if r == n - 1:
                    ax.set_xlabel("Time (s)", fontsize=9)
        for c, (_, _, _, ctitle) in enumerate(cols):
            fig.text((left + c * (pw + wgap)) / fw, 1 - (head - 0.12) / fh, ctitle,
                     fontsize=10.5 if stat != "both" else 9.5, color=INK, va="bottom")
        where = f"{dataset}, {label}" if label else dataset
        fig.text(left / fw, 1 - 0.15 / fh, f"{where}: {word} PER trace, responders vs the rest, "
                 "every odor", fontweight="bold", fontsize=12, color=INK, va="top")
        fig.text(left / fw, 1 - 0.42 / fh, _caveat(threshold) + f" {note} Each row has its "
                 "own y-scale.", fontsize=8.2, color=INK_2,
                 va="top")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def odor_file(odor: str) -> str:
    return f"{str(odor).replace('/', '-').replace(' ', '_')}_{TAG}.png"


def _read_wide(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)


def sibling(path: Path, kind: str) -> Path:
    """``<stem>_median.png`` / ``<stem>_mean_vs_median.png`` beside ``path``."""
    return path.with_name(f"{path.stem}_{kind}.png")


def _clear(out: Path) -> None:
    """This script's own files from an earlier run, the ``--with-median`` ones included (a run
    without the flag must not leave them beside fresh mean figures): an odor gone from the
    data must not leave its old figure behind."""
    for old in [*out.glob(f"*_{TAG}.png"), *out.glob(f"*_{TAG}_median.png"),
                *out.glob(f"*_{TAG}_mean_vs_median.png"), out / f"{TAG}_summary.csv"]:
        old.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pretest-wide-csv", required=True, type=Path)
    p.add_argument("--testing-wide-csv", required=True, type=Path)
    p.add_argument("--predictions-csv", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--flagged-flies-csv", default="")
    p.add_argument("--binary-threshold", type=int, default=pvt.BINARY_THRESHOLD,
                   help="model score counted as responding (the score figures' own)")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--odor-on-s", type=float, default=30.0,
                   help="each trial loses the mean of its first this many seconds")
    p.add_argument("--odor-off-s", type=float, default=60.0)
    p.add_argument("--with-median", action="store_true",
                   help="also write a _median twin (median, IQR band) and a _mean_vs_median "
                        "figure for every PNG, and median_* summary columns")
    p.add_argument("--label", default="", help="which flies, for the titles, e.g. "
                                               "'start 9:40am-4:20pm'")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    paired = paired_trials(args.predictions_csv, dataset=args.dataset,
                           flagged_flies_csv=str(args.flagged_flies_csv or ""))
    if paired.empty:
        print(f"[responder-traces] {args.dataset}: no fly was scored in both phases; "
              "nothing to draw")
        return 0
    baseline_frames = max(1, int(round(args.odor_on_s * args.fps)))
    max_frames = int(round(MAX_TIME_S * args.fps))
    labels = phase_labels(args.dataset)
    odors = sorted(paired["odor"].unique())
    sensitized = sensitized_odor(args.dataset, odors)
    order = odor_order(odors, sensitized)

    by_odor: dict[str, dict[str, dict]] = {odor: {} for odor in order}
    by_median: dict[str, dict[str, dict]] = {odor: {} for odor in order}
    missing = 0
    for key, _, _, score_col in PHASES:
        traces = phase_traces(_read_wide(args.pretest_wide_csv if key == "pre"
                                         else args.testing_wide_csv),
                              paired, phase=key, baseline_frames=baseline_frames,
                              dataset=args.dataset)
        missing += len(paired) - len(traces)
        responding = {(str(p.fly), int(p.fly_number), p.odor):
                      bool(float(getattr(p, score_col)) >= args.binary_threshold)
                      for p in paired.itertuples(index=False)}
        for odor in order:
            mine = {k: v for k, v in traces.items() if k[2] == odor}
            by_odor[odor][key] = group_curves(mine, responding, max_frames=max_frames)
            if args.with_median:
                by_median[odor][key] = group_bands(mine, responding, max_frames=max_frames,
                                                   stat="median")
    if missing:
        print(f"[responder-traces] {args.dataset}: {missing} scored trials have no trace; "
              "left out")

    common = dict(labels=labels, fps=args.fps, odor_on_s=args.odor_on_s,
                  odor_off_s=args.odor_off_s)
    rows = [r for odor in order for r in summary_rows(
        odor, by_odor[odor], sensitized=odor == sensitized,
        median_curves=by_median[odor] if args.with_median else None, **common)]
    out.mkdir(parents=True, exist_ok=True)
    _clear(out)
    pd.DataFrame(rows).to_csv(out / f"{TAG}_summary.csv", index=False)
    mean_bands = {odor: as_bands(by_odor[odor]) for odor in order}
    for odor in order:
        path = out / odor_file(odor)
        one = dict(odor=odor, sensitized=odor == sensitized, dataset=args.dataset,
                   label=args.label, threshold=args.binary_threshold, **common)
        plot_odor(path, mean_bands[odor], **one)
        if args.with_median:
            plot_odor(sibling(path, "median"), by_median[odor], stat="median", **one)
            plot_odor(sibling(path, "mean_vs_median"), mean_bands[odor], stat="both",
                      other=by_median[odor], **one)
    path = out / f"All_odors_{TAG}.png"
    every = dict(dataset=args.dataset, label=args.label, threshold=args.binary_threshold,
                 **common)
    plot_all(path, mean_bands, order, sensitized, **every)
    if args.with_median:
        plot_all(sibling(path, "median"), by_median, order, sensitized, stat="median", **every)
        plot_all(sibling(path, "mean_vs_median"), mean_bands, order, sensitized, stat="both",
                 others=by_median, **every)
    print(f"[responder-traces] {args.dataset}: {len(order)} odors -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
