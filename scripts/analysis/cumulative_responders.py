#!/usr/bin/env python3
"""Cumulative % of flies responding to each odor, pre- vs post-exposure.

For one cohort: one figure per odor, plus one with every odor. Of the flies tested
on the odor in both phases, each curve is the % the model scores as responding
(score >= ``--binary-threshold``: the call, and the fly x odor pairs, of the score
figures, so a curve ends on the % responding they report), counted from the moment
each fly started: its first ``--min-run-s`` seconds above its own red line (theta)
after odor onset. A responder with no such moment during the odor (a late or a
brief response) is added after odor off, in a shaded zone of its own, so it is
neither dropped nor drawn as a response at odor off.

Why this and not a mean trace: responses are brief and start at different times,
and most flies pull the proboscis in a little at odor onset, for every odor and for
paraffin oil, so the mean trace of every fly cancels the responses out
(``Results/Sensitiviy-Results(11-15)/Mean-Trace-Options_DRAFT``, 2026-11-10).
Counting flies does not.

The red line is the pipeline's own (``binarized_per_rasters.build_trials`` with the
config's ``analysis.combined.combined_base.wide`` rule), the one the trace figures
and the graded rasters draw.

Outputs (``--out-dir``)::

    <Odor>_cumulative_responders.png     one per odor, named as Mean-Traces_* names them
    All_odors_cumulative_responders.png  every odor on one page, sensitized odor first
    cumulative_responders_summary.csv    per odor: flies, responders, %, no-onset counts,
                                         median onset, exact McNemar p (uncorrected)
    cumulative_responders_flies.csv      per fly x odor x phase: score, responding, onset

Usage::

    python3 scripts/analysis/cumulative_responders.py --dataset IAA-Sensitivity-1-1 \\
        --pretest-wide-csv _inputs/wide_pretest.parquet \\
        --testing-wide-csv _inputs/wide_testing.parquet \\
        --predictions-csv _inputs/predictions.csv --config config/config_new.yaml \\
        --out-dir Cumulative-Responders_Pre-Test_vs_Post-Training
"""

from __future__ import annotations

import argparse
import math
import re
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
from fbpipe.odor_constants import canon_dataset  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.auc_before_during import (  # noqa: E402
    MIN_FINITE_FRACTION,
    _mean_or_nan,
    _signed_peak,
    _window_frames,
)
from scripts.analysis.binarized_per_rasters import auc_ratio  # noqa: E402
from scripts.analysis.binarized_per_rasters import (  # noqa: E402
    build_trials,
    resolve_threshold_rule,
)
from scripts.analysis.envelope_visuals import _trained_label  # noqa: E402
from scripts.analysis.score_summary import _load_scores  # noqa: E402

#: The folder start_window_report draws this into, beside Mean-Traces_Pre-Test_vs_Post-Training.
FOLDER = "Cumulative-Responders_Pre-Test_vs_Post-Training"
TAG = "cumulative_responders"
MIN_RUN_S = 0.5
#: Where the odor is split in two for the per-half measures: seconds after its onset.
HALF_S = 15.0
#: The per-half measures (see :func:`half_measures`), one column per half in the flies CSV.
HALF_KEYS = ("auc", "auc_ratio", "auc_unsigned_ratio", "zpeak", "zpeak_ratio",
             "zpeak_unsigned_ratio")
HALF_COLUMNS = tuple(f"{k}_{h}_half" for k in HALF_KEYS for h in ("first", "second"))
#: (key, trial_type, trial column, score column) per phase, pre-exposure first.
PHASES = (("pre", "pretest", "trial_pre", "score_pre"),
          ("post", "testing", "trial_post", "score_post"))
COLORS = {"pre": pvt.PRE_COLOR, "post": pvt.POST_COLOR}   # the score figures' validated pair
INK, INK_2, GRID, ZONE = "#0b0b0b", "#52514e", "#e4e3df", "#efeee9"
CURVE_STEP_S = 0.025          # one frame at 40 fps
NO_ONSET_S = 3.0              # width of the zone after odor off
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
# When a response starts, and the curve
# ---------------------------------------------------------------------------


def first_run_s(binary: Sequence[bool], *, fps: float, start_s: float, end_s: float,
                min_run_s: float = MIN_RUN_S) -> float:
    """Seconds from ``start_s`` to the first ``min_run_s`` run of 1s in [start_s, end_s).

    Only frames inside the window count: a run already under way at ``start_s``
    starts at 0 if enough of it is inside, one cut by ``end_s`` must fit before it.
    NaN when there is none.
    """
    b = np.asarray(binary, dtype=bool)
    if not all(math.isfinite(v) for v in (fps, start_s, end_s)) or fps <= 0:
        return float("nan")
    a = max(0, int(round(start_s * fps)))
    z = min(b.size, int(round(end_s * fps)))
    run = max(1, int(round(min_run_s * fps)))
    if z - a < run:
        return float("nan")
    window = np.convolve(b[a:z].astype(np.int32), np.ones(run, dtype=np.int32), "valid")
    hits = np.flatnonzero(window == run)
    return float(hits[0] / fps) if hits.size else float("nan")


def half_aucs(trace: Sequence[float], *, fps: float, odor_on_s: float, odor_off_s: float,
              split_s: float = HALF_S) -> tuple[float, float]:
    """Signed area (% x s) in the first ``split_s`` seconds of the odor and in the rest.

    The AUC figures' own area (``auc_before_during.trial_auc``): the trace minus the
    median of the window just ahead of onset, as long as the odor window; missing
    frames skipped, mean deviation times the stretch's duration. With no missing
    frames the two halves add up to that figure's signed during-odor area.
    """
    env = np.asarray(trace, dtype=float)
    nan = float("nan")
    frames = _window_frames(env.size, fps, odor_on_s, odor_off_s)
    if frames is None:
        return nan, nan
    b0, a, z = frames
    before = env[b0:a]
    if np.isfinite(before).sum() < MIN_FINITE_FRACTION * before.size:
        return nan, nan
    baseline = float(np.nanmedian(before))
    m = min(z, a + int(round(split_s * fps)))
    first, second = env[a:m], env[m:z]
    return tuple(_mean_or_nan(seg - baseline) * seg.size / fps if seg.size else nan
                 for seg in (first, second))


def half_measures(trace: Sequence[float], *, fps: float, odor_on_s: float, odor_off_s: float,
                  split_s: float = HALF_S) -> dict[str, float]:
    """Each half of the odor (the first ``split_s`` s and the rest), by the measures of
    each cohort's AUC-Before-vs-During and PeakZ-Before-vs-During figures:

    ``auc``                   signed area, % x s (:func:`half_aucs`)
    ``auc_ratio``             raw area during / before, per second: the half's mean level
                              over the mean level of [0, onset), the AUC figures'
                              ``ratio_raw``; the two halves average to it
    ``auc_unsigned_ratio``    |deviation from the before median| during / before, per second
    ``zpeak``                 signed peak z in the half (z from the before window)
    ``zpeak_ratio``           that peak / |the before window's own peak|
    ``zpeak_unsigned_ratio``  |that peak| / |the before window's own peak|

    Both halves share one before window, the cohort figures' own, so they compare with
    each other and with the whole-odor figures.
    """
    nan = float("nan")
    out = dict.fromkeys(HALF_COLUMNS, nan)
    env = np.asarray(trace, dtype=float)
    frames = _window_frames(env.size, fps, odor_on_s, odor_off_s) if env.size else None
    if frames is None:
        return out
    out["auc_first_half"], out["auc_second_half"] = half_aucs(
        env, fps=fps, odor_on_s=odor_on_s, odor_off_s=odor_off_s, split_s=split_s)
    b0, a, z = frames
    m = min(z, a + int(round(split_s * fps)))
    halves = (("first", env[a:m]), ("second", env[m:z]))

    def area(seg: np.ndarray) -> float:          # bpr.window_area, frames already cut
        finite = seg[np.isfinite(seg)]
        return float(finite.sum() / fps) if finite.size else nan

    raw_before = area(env[:a])                   # [0, onset), as ratio_raw takes it
    before = env[b0:a]
    enough = np.isfinite(before).sum() >= MIN_FINITE_FRACTION * before.size
    if enough:
        dev_before = _mean_or_nan(np.abs(before - np.nanmedian(before)))
    for half, seg in halves:
        if seg.size:
            out[f"auc_ratio_{half}_half"] = auc_ratio(area(seg) * a / seg.size, raw_before)
            if enough:
                dev = _mean_or_nan(np.abs(seg - np.nanmedian(before)))
                window = (z - a) / fps          # both on the odor window's duration
                out[f"auc_unsigned_ratio_{half}_half"] = auc_ratio(dev * window,
                                                                   dev_before * window)
    finite = before[np.isfinite(before)]
    if finite.size < max(2, MIN_FINITE_FRACTION * before.size):
        return out
    mean, sd = float(finite.mean()), float(finite.std(ddof=1))
    if not sd > 0:
        return out
    peak_before = abs(_signed_peak((before - mean) / sd))
    for half, seg in halves:
        peak = _signed_peak((seg - mean) / sd) if seg.size else nan
        out[f"zpeak_{half}_half"] = peak
        if peak_before > 0:
            out[f"zpeak_ratio_{half}_half"] = peak / peak_before
            out[f"zpeak_unsigned_ratio_{half}_half"] = abs(peak) / peak_before
    return out


def cumulative_percent(onsets: Sequence[float], responding: Sequence[bool],
                       times: Sequence[float]) -> np.ndarray:
    """% of all flies that are responding and had started by each of ``times``."""
    onsets = np.asarray(onsets, dtype=float)
    responding = np.asarray(responding, dtype=bool)
    times = np.asarray(times, dtype=float)
    if onsets.size == 0:
        return np.zeros(times.size)
    timed = np.sort(onsets[responding & np.isfinite(onsets)])
    return 100.0 * np.searchsorted(timed, times, side="right") / onsets.size


# ---------------------------------------------------------------------------
# Which flies: the score figures' pairs
# ---------------------------------------------------------------------------


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fly"] = out["fly"].astype(str).str.strip()
    out["fly_number"] = pd.to_numeric(out["fly_number"], errors="coerce").astype("Int64")
    return out


def paired_trials(predictions_csv: Path | str, *, dataset: str,
                  flagged_flies_csv: str = "") -> pd.DataFrame:
    """One row per fly x odor scored in both phases: its two trial numbers and scores.

    The pairs are ``pretest_vs_test_comparison.load_paired_scores``'s, the score
    figures' own, so the % responding here cannot drift from theirs. The trial
    numbers come from the same ``_load_scores`` rows, first presentation kept as
    there; a score that does not match is an error, not a figure.
    """
    cols = ["fly", "fly_number", "odor", "trial_pre", "trial_post", "score_pre", "score_post"]
    paired = pvt.load_paired_scores(predictions_csv, flagged_flies_csv=flagged_flies_csv)
    canon = canon_dataset(str(dataset).strip())
    paired = paired[paired["dataset_canon"].astype(str) == canon]
    if paired.empty:
        return pd.DataFrame(columns=cols)
    out = _keyed(paired)[["fly", "fly_number", "odor", "score_pre", "score_post"]]
    predictions = pd.read_csv(predictions_csv)
    key = ["fly", "fly_number", "odor"]
    for _, trial_type, trial_col, score_col in PHASES:
        # Unflagged: the flagged table drops whole flies, so a kept fly's first
        # presentation is the same either way.
        scores = _load_scores(predictions, trial_types=(trial_type,))
        scores = _keyed(scores[scores["dataset_canon"].astype(str) == canon])
        scores = scores.rename(columns={"odor_display": "odor", "trial_num": trial_col,
                                        "score": "_score"})
        scores = scores.drop_duplicates(subset=key, keep="first")
        out = out.merge(scores[key + [trial_col, "_score"]], on=key, how="left",
                        validate="one_to_one")
        same = np.isclose(pd.to_numeric(out["_score"], errors="coerce").to_numpy(float),
                          pd.to_numeric(out[score_col], errors="coerce").to_numpy(float),
                          equal_nan=True)
        if out[trial_col].isna().any() or not same.all():
            raise RuntimeError(f"{dataset}: {trial_type} trials do not match the score "
                               f"figures' pairs ({int((~same).sum())} scores differ)")
        out = out.drop(columns="_score")
    for col in ("fly_number", "trial_pre", "trial_post"):
        out[col] = out[col].astype(int)
    return out[cols].sort_values(["fly", "fly_number", "odor"]).reset_index(drop=True)


def phase_labels(dataset: str) -> dict[str, str]:
    return {"pre": phase_terms.pre_label(dataset, pvt.PRE_LABEL),
            "post": phase_terms.post_label(dataset, pvt.POST_LABEL)}


def fly_table(paired: pd.DataFrame, trials: Mapping[str, pd.DataFrame], *,
              labels: Mapping[str, str], binary_threshold: float,
              min_run_s: float = MIN_RUN_S) -> pd.DataFrame:
    """Per fly x odor x phase: score, responding, and the onset of its first run.

    ``trials`` holds ``build_trials`` per phase key. A scored trial with no trace
    row keeps its score (it still counts as responding) and has no onset.
    """
    rows = []
    for key, _, trial_col, score_col in PHASES:
        t = trials.get(key, pd.DataFrame())
        index = ({(str(r.fly).strip(), int(r.fly_number), int(r.trial_index)): r
                  for r in t.itertuples(index=False)} if not t.empty else {})
        for p in paired.itertuples(index=False):
            trial = int(getattr(p, trial_col))
            score = float(getattr(p, score_col))
            r = index.get((p.fly, int(p.fly_number), trial))
            onset = (first_run_s(r.binary, fps=float(r.fps), start_s=float(r.odor_on_s),
                                 end_s=float(r.odor_off_s), min_run_s=min_run_s)
                     if r is not None else float("nan"))
            halves = (half_measures(r.trace, fps=float(r.fps), odor_on_s=float(r.odor_on_s),
                                    odor_off_s=float(r.odor_off_s))
                      if r is not None else dict.fromkeys(HALF_COLUMNS, float("nan")))
            rows.append({
                "fly": p.fly, "fly_number": int(p.fly_number), "odor": p.odor,
                "phase": labels[key], "trial": trial, "score": score,
                "responding": bool(score >= binary_threshold), "onset_s": onset,
                "odor_on_s": float(r.odor_on_s) if r is not None else float("nan"),
                "odor_off_s": float(r.odor_off_s) if r is not None else float("nan"),
                "theta": float(r.theta) if r is not None else float("nan"),
                **halves,
                "traced": r is not None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-odor numbers
# ---------------------------------------------------------------------------


def _norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def sensitized_odor(dataset: str, odors: Sequence[str]) -> str | None:
    """The odor of ``odors`` the cohort was exposed to (trained on), or None."""
    label = _trained_label(dataset)
    if not label:
        return None
    return next((o for o in odors if _norm(o) == _norm(label)), None)


def odor_order(odors: Sequence[str], sensitized: str | None) -> list[str]:
    rest = sorted(o for o in odors if o != sensitized)
    return ([sensitized] if sensitized in odors else []) + rest


def _phase_rows(flies: pd.DataFrame, odor: str, label: str) -> pd.DataFrame:
    return (flies[(flies["odor"] == odor) & (flies["phase"] == label)]
            .sort_values(["fly", "fly_number"]))


def odor_summary(flies: pd.DataFrame, order: Sequence[str], *, sensitized: str | None,
                 labels: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for odor in order:
        pre, post = (_phase_rows(flies, odor, labels[k]) for k in ("pre", "post"))
        if pre[["fly", "fly_number"]].values.tolist() != post[["fly", "fly_number"]].values.tolist():
            raise RuntimeError(f"{odor}: the two phases do not hold the same flies")
        row = {"odor": odor, "sensitized": odor == sensitized, "n_flies": len(pre)}
        resp, onset = {}, {}
        for k, g in (("pre", pre), ("post", post)):
            resp[k] = g["responding"].to_numpy(bool)
            onset[k] = g["onset_s"].to_numpy(float)
        for k in ("pre", "post"):
            row[f"responding_{k}"] = int(resp[k].sum())
        for k in ("pre", "post"):
            row[f"pct_{k}"] = 100.0 * float(resp[k].mean()) if resp[k].size else float("nan")
        for k in ("pre", "post"):
            row[f"no_onset_{k}"] = int((resp[k] & ~np.isfinite(onset[k])).sum())
        for k in ("pre", "post"):
            timed = resp[k] & np.isfinite(onset[k])
            row[f"median_onset_{k}_s"] = (float(np.median(onset[k][timed])) if timed.any()
                                          else float("nan"))
        p = pvt.mcnemar_p(resp["pre"], resp["post"])
        row["p_mcnemar"] = float("nan") if p is None else float(p)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def p_text(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "no fly changed, no test"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def y_top(summary: pd.DataFrame) -> float:
    """One y-range for every odor of the cohort, with room above the highest curve."""
    top = float(np.nanmax(summary[["pct_pre", "pct_post"]].to_numpy(float))) if len(summary) else 0
    return float(min(100.0, max(40.0, 20.0 * math.ceil((top + 8.0) / 20.0))))


def _spread(values: Sequence[float], gap: float) -> list[float]:
    """Label heights at least ``gap`` apart, centred on the values they label."""
    if len(values) != 2 or abs(values[0] - values[1]) >= gap:
        return list(values)
    mid = (values[0] + values[1]) / 2
    lo, hi = mid - gap / 2, mid + gap / 2
    return [lo, hi] if values[0] <= values[1] else [hi, lo]


def draw_panel(ax, flies_odor: pd.DataFrame, *, labels: Mapping[str, str], span: float,
               odor_off: float, ymax: float, end_size: float = 10.0,
               zone_label: bool = True) -> None:
    """Both phases of one odor: timed curve to the end of the odor, then the zone."""
    times = np.arange(0.0, span + CURVE_STEP_S / 2, CURVE_STEP_S)
    end = span + NO_ONSET_S
    ax.axvspan(span, end, color=ZONE, lw=0, zorder=0)
    ax.axvline(odor_off, color=INK_2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    if zone_label:
        ax.text(span + NO_ONSET_S / 2, 0.985, "no clear\nonset",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5,
                color=INK_2, linespacing=1.0)
    line = dict(lw=2.0, zorder=3, solid_capstyle="butt", solid_joinstyle="miter")
    totals = []
    for i, (key, *_rest) in enumerate(PHASES):
        g = flies_odor[flies_odor["phase"] == labels[key]]
        n = len(g)
        resp = g["responding"].to_numpy(bool)
        timed = cumulative_percent(g["onset_s"].to_numpy(float), resp, times)
        total = 100.0 * float(resp.sum()) / n if n else 0.0
        color = COLORS[key]
        label = f"{labels[key]}: {int(resp.sum())} of {n} flies"
        if total <= timed[-1] + 1e-9:                    # every responder timed: one line
            ax.step(np.append(times, end), np.append(timed, timed[-1]), where="post",
                    color=color, label=label, **line)
        else:                                            # pre and post rise apart
            x_rise = span + NO_ONSET_S * (i + 1) / 3
            ax.step(np.append(times, x_rise), np.append(timed, timed[-1]), where="post",
                    color=color, label=label, **line)
            ax.plot([x_rise, x_rise], [timed[-1], total], color=color, lw=2.0,
                    ls=(0, (1.2, 1.2)), zorder=3)
            ax.plot([x_rise, end], [total] * 2, color=color, **line)
        totals.append(total)
    for total, y in zip(totals, _spread(totals, 0.07 * ymax)):
        ax.text(end + 0.35, y, f"{total:.0f}%", va="center", ha="left", color=INK,
                fontsize=end_size, fontweight="bold", clip_on=False)
    ax.set_xlim(0, end)
    ax.set_ylim(0, ymax)
    ax.set_xticks(list(range(0, int(span) + 1, 5)))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


Y_LABEL = "Flies that have responded (%)"
X_LABEL = "Time from odor onset (s)"


def _odor_title(odor: str, row: pd.Series, dataset: str) -> str:
    return f"{odor} ({phase_terms.odor_tag(dataset)})" if bool(row["sensitized"]) else odor


def _definition(threshold: int, min_run_s: float) -> str:
    return (f"Flies the model scores {threshold} or more (as on the score bars), each counted "
            f"from its first {min_run_s:g} s above its own red line.\nShaded: responders "
            f"never {min_run_s:g} s above it during the odor.")


def plot_odor(path: Path, flies_odor: pd.DataFrame, row: pd.Series, *, dataset: str,
              label: str, labels: Mapping[str, str], span: float, odor_off: float,
              ymax: float, threshold: int, min_run_s: float) -> None:
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8.6, 4.9))
        draw_panel(ax, flies_odor, labels=labels, span=span, odor_off=odor_off, ymax=ymax)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.legend(loc="upper left", fontsize=9)
        where = f"{dataset}, {label}" if label else dataset
        ax.set_title(f"{_odor_title(row['odor'], row, dataset)}: flies responding since "
                     "odor onset", loc="left", fontweight="bold", color=INK, fontsize=12,
                     pad=46)
        ax.text(0, 1.03, f"{where}. The same {int(row['n_flies'])} flies in both phases; "
                f"exact McNemar {p_text(row['p_mcnemar'])}.\n"
                + _definition(threshold, min_run_s),
                transform=ax.transAxes, fontsize=8.6, color=INK_2, va="bottom")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_all(path: Path, flies: pd.DataFrame, summary: pd.DataFrame, *, dataset: str,
             label: str, labels: Mapping[str, str], span: float, odor_off: float,
             ymax: float, threshold: int, min_run_s: float) -> None:
    n = len(summary)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    # Placed by hand, in inches: room above each panel for its two title lines,
    # right of it for the end labels, below it for the tick labels and x label.
    pw, ph = 3.3, 2.4                       # panel
    left, right, wgap = 0.75, 0.55, 1.05    # wgap: end labels + the next panel's y ticks
    head, title, hgap, bottom = 1.0, 0.5, 0.65, 0.6
    fw = left + ncols * pw + (ncols - 1) * wgap + right
    fh = head + nrows * (title + ph) + (nrows - 1) * hgap + bottom
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(fw, fh))
        for i, (_, row) in enumerate(summary.iterrows()):
            r, c = divmod(i, ncols)
            top = fh - head - r * (title + ph + hgap) - title
            ax = fig.add_axes(((left + c * (pw + wgap)) / fw, (top - ph) / fh, pw / fw, ph / fh))
            draw_panel(ax, flies[flies["odor"] == row["odor"]], labels=labels, span=span,
                       odor_off=odor_off, ymax=ymax, end_size=9.0, zone_label=False)
            ax.set_title(row["odor"], loc="left", fontweight="bold", color=INK, fontsize=10.5,
                         pad=15)
            note = f"n = {int(row['n_flies'])} flies, {p_text(row['p_mcnemar'])}"
            if bool(row["sensitized"]):
                note = f"{phase_terms.odor_tag(dataset)}; {note}"
            ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=8, color=INK_2,
                    va="bottom")
            if c == 0:
                ax.set_ylabel(Y_LABEL, fontsize=9)
            if i + ncols >= n:                    # nothing drawn below this panel
                ax.set_xlabel(X_LABEL, fontsize=9)
            if i == 0:
                handles, texts = ax.get_legend_handles_labels()
        where = f"{dataset}, {label}" if label else dataset
        fig.text(left / fw, 1 - 0.18 / fh, f"{where}: flies responding since odor onset, "
                 "every odor", fontweight="bold", fontsize=12.5, color=INK, va="top")
        fig.text(left / fw, 1 - 0.48 / fh, _definition(threshold, min_run_s)
                 + " Exact McNemar p per odor, uncorrected.", fontsize=8.6, color=INK_2,
                 va="top")
        fig.legend(handles, [t.split(":")[0] for t in texts], loc="upper right", ncol=2,
                   fontsize=9.5, bbox_to_anchor=(1 - right / fw, 1 - 0.1 / fh))
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def odor_file(odor: str) -> str:
    return f"{str(odor).replace('/', '-').replace(' ', '_')}_{TAG}.png"


def _read_wide(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)


def _clear(out: Path) -> None:
    """This script's own files from an earlier run: an odor gone from the data must not
    leave its old figure behind."""
    for old in [*out.glob(f"*_{TAG}.png"), out / f"{TAG}_summary.csv", out / f"{TAG}_flies.csv"]:
        old.unlink(missing_ok=True)


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
    p.add_argument("--min-run-s", type=float, default=MIN_RUN_S,
                   help="time above the red line that starts a response")
    p.add_argument("--label", default="", help="which flies, for the titles, e.g. "
                                               "'start 9:40am-4:20pm'")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    paired = paired_trials(args.predictions_csv, dataset=args.dataset,
                           flagged_flies_csv=str(args.flagged_flies_csv or ""))
    if paired.empty:
        print(f"[cumulative] {args.dataset}: no fly was scored in both phases; nothing to draw")
        return 0
    rule = resolve_threshold_rule(args.config)
    trials = {key: build_trials(_read_wide(args.pretest_wide_csv if key == "pre"
                                           else args.testing_wide_csv),
                                dataset=args.dataset, trial_type=trial_type, rule=rule)
              for key, trial_type, _, _ in PHASES}
    labels = phase_labels(args.dataset)
    flies = fly_table(paired, trials, labels=labels, binary_threshold=args.binary_threshold,
                      min_run_s=args.min_run_s)
    untraced = int((~flies["traced"]).sum())
    if untraced:
        print(f"[cumulative] {args.dataset}: {untraced} scored trials have no trace; "
              "a responder among them counts with no onset")
    odors = sorted(flies["odor"].unique())
    sensitized = sensitized_odor(args.dataset, odors)
    summary = odor_summary(flies, odor_order(odors, sensitized), sensitized=sensitized,
                           labels=labels)
    durations = (flies["odor_off_s"] - flies["odor_on_s"]).to_numpy(float)
    durations = durations[np.isfinite(durations)]
    span = math.ceil(float(durations.max()) * 10) / 10 if durations.size else 30.0
    odor_off = float(np.median(durations)) if durations.size else 30.0
    ymax = y_top(summary)
    common = dict(dataset=args.dataset, label=args.label, labels=labels, span=span,
                  odor_off=odor_off, ymax=ymax, threshold=args.binary_threshold,
                  min_run_s=args.min_run_s)

    out.mkdir(parents=True, exist_ok=True)
    _clear(out)
    summary.to_csv(out / f"{TAG}_summary.csv", index=False)
    flies.drop(columns="traced").to_csv(out / f"{TAG}_flies.csv", index=False)
    for _, row in summary.iterrows():
        plot_odor(out / odor_file(row["odor"]), flies[flies["odor"] == row["odor"]], row,
                  **common)
    plot_all(out / f"All_odors_{TAG}.png", flies, summary, **common)
    print(f"[cumulative] {args.dataset}: {len(summary)} odors -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
