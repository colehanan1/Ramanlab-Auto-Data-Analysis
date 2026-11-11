"""PER trace before vs during the odor: area (AUC), peak z-score or the scorer's AUC log ratio.

For every pre-test and post-training trial of every fly that has both panels,
``before`` is the window ahead of odor onset and ``during`` is the odor window
itself (the trial's own rig timing).

``--metric auc`` (default) -- area under the trace, % x s. No threshold: the
zero line on the figures IS each trial's own reference.

    baseline  = median of the trace over the before window (the odor window's
                length, just ahead of onset)
    signed    = area of (trace - baseline); a dip below baseline counts negative
    unsigned  = area of |trace - baseline|; every excursion counts positive
    delta     = during - before, the odor-evoked change on that one trial

  Both windows are scored over the same duration, so the signed delta is just
  the raw-trace area difference and does not depend on the baseline at all.
  The median (not the mean) is the baseline so a spontaneous extension before
  the odor shows up as before-odor area instead of being folded into the
  reference. Two during / before ratios sit beside the deltas:

    ratio_raw       area under the raw trace during / before -- the
                    ``auc_ratio`` of Graded-PER-Rasters/*_binarized_trials.csv,
                    computed by the same functions, zero-substitute included
    ratio_unsigned  unsigned AUC during / unsigned AUC before

``--metric peakz`` -- peak z-score, in SDs of the same before window:

    z         = (trace - mean) / SD, both from the before window
    signed    = z at the frame of largest |z|, sign kept: a dip bigger than any
                rise reads negative
    unsigned  = that |z|
    before    = the before window's own peak -- its noise ceiling, the bar the
                odor window has to clear (the windows are the same length, so
                the same number of frames gets the same chance at a big value)
    ratio     = unsigned during / unsigned before

  The scale is the plain SD, not the pipeline's robust 1.4826 x MAD: on a fly
  that holds very still the MAD collapses towards 0 and z runs into the
  hundreds (163 in the 1pm-2pm cohort), whereas the SD keeps the before
  window's own peak near 3.5, where pure noise puts it.

``--metric logratio`` -- the wide table's thresholded ``AUC-Before`` /
``AUC-During`` (area ABOVE the response threshold theta, the red line on the
trace figures) and the scoring model's top input, ``AUC-During-Before-LogRatio``
= log((AUC-During + eps) / (AUC-Before + eps)). ``eps`` is the value stored on
the model (``fbr_auc_eps``), read from ``reaction_prediction.model_path`` in
``--config`` unless ``--auc-eps`` is given, so the column is the model's input
exactly. Thresholded areas are never negative, so there is no signed/unsigned
split.

There is no signed ratio for AUC or peak z: the signed before value sits near
0 and flips sign from trial to trial, so dividing by it is not interpretable.

Each fly contributes one trial per odor per phase, so a per-odor mean is a
mean over trials and flies alike. The "All odors" group pools every trial
(never a mean of odor means) and its n counts trials; it carries no p-value
because those trials are not independent (each fly gives seven).

With ``--predictions-csv`` every trial also gets its model score, and the
metric is plotted against it (pre-test, post-training and both pooled), with a
Spearman rank correlation per panel.

Outputs (``--out-dir``; <key> = auc | peakz | logratio,
<Stem> = AUC | PeakZ | AUCLogRatio):
    <key>_per_trial.csv                          one row per fly x odor x phase
    <key>_per_fly_change.csv                     each fly's pre, post, post - pre
    <key>_summary.csv                            mean / SEM / n / Wilcoxon p
    <Stem>_before_vs_during_dataset.png          before vs during, whole dataset
    <Stem>_during_dataset.png                    during window only, flies paired
    <Stem>_during_minus_before_dataset.png       delta, flies paired
    <Stem>_ratio_during_over_before_dataset.png  during / before, flies paired
    <Stem>_during_minus_before_per_fly.png       delta per fly x odor (heatmap;
                                                 logratio: <Stem>_log_ratio_per_fly)
  with --predictions-csv:
    <Stem>_vs_score_during_minus_before.png      delta by model score
    <Stem>_vs_score_during.png                   during window by model score
    <Stem>_vs_score_ratio.png                    during / before by model score
    <key>_vs_score_correlation.csv               Spearman rho per phase x measure
    <key>_vs_score_summary.csv                   n / mean / median per score

Usage
-----
    python3 scripts/analysis/auc_before_during.py \
        --pretest-wide-csv <folder>/_inputs/wide_pretest.parquet \
        --testing-wide-csv <folder>/_inputs/wide_testing.parquet \
        --predictions-csv <folder>/_inputs/predictions_raster.csv \
        --config config/config_new.yaml --metric auc \
        --out-dir <folder>/AUC-Before-vs-During --label "start 1pm-2pm"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fbpipe.utils.tables import read_table  # noqa: E402
from scripts.analysis import binarized_per_rasters as bpr  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis import pretest_vs_test_traces as pvt  # noqa: E402
from scripts.analysis import score_summary as ss  # noqa: E402

PHASES = ("Pre-test", "Post-training")
COMBINED = "Pre + post combined"
PANELS = (*PHASES, COMBINED)
PHASE_COLOR = {"Pre-test": pvt.PRE_COLOR, "Post-training": pvt.POST_COLOR}
WINDOWS = ("before", "during")
KINDS = ("signed", "unsigned")
ALL_ODORS = "All odors"
ID_COLUMNS = ["dataset", "fly", "fly_number", "phase", "odor", "trained", "trial_label",
              "odor_on_s", "odor_off_s"]

#: A window with less than this fraction of finite frames is not scored.
MIN_FINITE_FRACTION = 0.5

#: Where the scoring model stores its AUC epsilon (flybehavior_response.xgb_ordinal).
MODEL_EPS_ATTR = "fbr_auc_eps"

INK = "#2b2b28"
INK_MUTED = pvt.INK_MUTED


# ---------------------------------------------------------------------------
# What is measured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One row of panels: which column it plots, how it is labelled, its null."""

    measure: str
    ylabel: str
    title: str
    null: float = 0.0
    short: str = ""             # heatmap panel label; default: ylabel minus its unit

    @property
    def short_label(self) -> str:
        return self.short or self.ylabel.split(" (")[0]


@dataclass(frozen=True)
class Metric:
    """One family of per-trial columns and everything needed to plot it."""

    key: str                    # CLI value and table-file prefix
    stem: str                   # figure-file prefix
    prefix: str                 # column prefix
    name: str                   # "AUC", "peak z"
    unit: str                   # axis unit
    what: str                   # figure-title noun
    fmt: str                    # heatmap cell format
    kind_title: dict
    ratios: tuple               # Row per ratio column
    extra_columns: tuple        # reference values written to the per-trial CSV
    definition: str             # footnote, first sentence(s)
    ratio_note: str
    kinds: tuple = KINDS        # ("",) for a family with no signed/unsigned split
    heatmap: str = "delta"      # what the per-fly heatmap shows: "delta" | "ratio"

    def col(self, window: str, kind: str) -> str:
        return f"{self.prefix}{window}_{kind}" if kind else f"{self.prefix}{window}"

    def label(self, kind: str) -> str:
        return f"{kind.title()} {self.name}" if kind else self.name[:1].upper() + self.name[1:]

    @property
    def deltas(self) -> tuple[str, ...]:
        return tuple(self.col("delta", k) for k in self.kinds)

    @property
    def measures(self) -> tuple[str, ...]:
        windows = tuple(self.col(w, k) for k in self.kinds for w in WINDOWS)
        return windows + self.deltas + tuple(r.measure for r in self.ratios)

    @property
    def tested(self) -> tuple[str, ...]:
        """Measures with a null to test against: the deltas and the ratios."""
        return self.deltas + tuple(r.measure for r in self.ratios)

    def rows(self, window: str) -> tuple[Row, ...]:
        what = "during - before" if window == "delta" else "during odor"
        return tuple(
            Row(self.col(window, k), f"{self.label(k)}, {what} ({self.unit})",
                self.kind_title[k])
            for k in self.kinds
        )

    def heatmap_rows(self) -> tuple[Row, ...]:
        return self.rows("delta") if self.heatmap == "delta" else self.ratios

    @property
    def heatmap_file(self) -> str:
        return "during_minus_before" if self.heatmap == "delta" else "log_ratio"


AUC = Metric(
    key="auc", stem="AUC", prefix="", name="AUC", unit="% x s",
    what="PER-trace area", fmt="{:.0f}",
    kind_title={
        "signed": "Signed AUC: dips below baseline count negative",
        "unsigned": "Unsigned AUC: every excursion from baseline counts positive",
    },
    ratios=(
        Row("ratio_raw", "During / before (raw area)",
            "Raw-area ratio: area under the trace during / before "
            "(auc_ratio of the raster CSVs)", null=1.0),
        Row("ratio_unsigned", "During / before (unsigned AUC)",
            "Unsigned ratio: |deviation from baseline| during / before", null=1.0),
    ),
    extra_columns=("baseline", "auc_before_raw", "auc_during_raw"),
    definition=(
        "Trace: Max Distance x Angle % per trial. Baseline = median of the window "
        "before odor onset (no threshold). Before = the window of the odor window's "
        "length (~30 s) just ahead of onset; During = the odor window. AUC in % x s. "
    ),
    ratio_note=(
        "Raw ratio = area under the raw trace in the odor window / the window "
        "before it, the auc_ratio in Graded-PER-Rasters/*_binarized_trials.csv "
        "(a before-area of exactly 0 is replaced by 1). Unsigned ratio = unsigned "
        "AUC during / before. 1 = no change. No signed ratio: the signed "
        "before-area sits near 0 and flips sign, so the quotient is meaningless."
    ),
)

PEAKZ = Metric(
    key="peakz", stem="PeakZ", prefix="zpeak_", name="peak z", unit="SD",
    what="PER-trace peak z-score", fmt="{:.1f}",
    kind_title={
        "signed": "Signed peak z: the largest excursion from baseline, dips negative",
        "unsigned": "Unsigned peak z: size of the largest excursion, either direction",
    },
    ratios=(
        Row("zpeak_ratio_signed", "Signed during / |before| (peak z)",
            "Signed ratio: peak z during (sign kept) / |peak z| before", null=0.0),
        Row("zpeak_ratio_unsigned", "During / before (unsigned peak z)",
            "Unsigned ratio: |peak z| during / |peak z| before", null=1.0),
    ),
    extra_columns=("z_baseline_mean", "z_baseline_sd"),
    definition=(
        "Trace: Max Distance x Angle % per trial, z-scored against its own pre-odor "
        "window: z = (trace - mean) / SD of the window of the odor window's length "
        "(~30 s) just ahead of onset (no threshold). Peak = the frame of largest |z| "
        "in a window; signed keeps its sign (a dip bigger than any rise reads "
        "negative), unsigned is |z|. Before = the pre-odor window's own peak, its "
        "noise ceiling; During = the odor window. "
    ),
    ratio_note=(
        "Signed ratio = peak z during, sign kept / |peak z| before: the odor peak in "
        "units of the baseline's own largest excursion, + for a rise, - for a dip; "
        "pure noise lands near +/-1. The denominator is the before-peak's SIZE: its "
        "sign flips between trials, so dividing by it would scramble the direction. "
        "Unsigned ratio = |peak z| during / |peak z| before; 1 = the odor window's "
        "biggest excursion is no bigger than the baseline's own."
    ),
)

LOGRATIO = Metric(
    key="logratio", stem="AUCLogRatio", prefix="thr_auc_", name="thresholded AUC",
    unit="% x s", what="Thresholded AUC (the scorer's AUC-During-Before-LogRatio)",
    fmt="{:.1f}",
    kind_title={"": "Thresholded AUC: only the area above the response threshold "
                    "theta counts (never negative)"},
    ratios=(
        Row("thr_auc_log_ratio", "log((During + eps) / (Before + eps))",
            "AUC-During-Before-LogRatio: the scoring model's most-used input",
            null=0.0, short="AUC-During-Before-LogRatio"),
    ),
    extra_columns=(),
    definition=(
        "Thresholded AUC = the wide table's AUC-Before / AUC-During: area of the "
        "Max Distance x Angle % trace ABOVE the response threshold theta (the red "
        "line on the trace figures: median of the last 5 s before odor + max(3 x "
        "idle-noise SD, 5 %)) over the pre-odor and odor windows, in % x s; 0 when "
        "the trace never crosses theta. "
    ),
    ratio_note=(
        "Log ratio = log((AUC-During + eps) / (AUC-Before + eps)), eps = {eps:.2f} "
        "stored on the scoring model: exactly the model's AUC-During-Before-LogRatio "
        "input. 0 = no change; +1 = 2.7x, +2 = 7.4x."
    ),
    kinds=("",), heatmap="ratio",
)

METRICS = {m.key: m for m in (AUC, PEAKZ, LOGRATIO)}
_NULL_BY_MEASURE = {r.measure: r.null for m in METRICS.values() for r in m.ratios}

# Back-compatible names for the AUC family.
MEASURES = AUC.measures[:6]
RATIOS = tuple(r.measure for r in AUC.ratios)
RAW_AREAS = ("auc_before_raw", "auc_during_raw")
ALL_MEASURES = AUC.measures
KIND_TITLE = AUC.kind_title


def null_value(measure: str) -> float:
    """What "no odor effect" reads as: 1 for a plain ratio, 0 for the rest."""
    return _NULL_BY_MEASURE.get(measure, 0.0)


def model_auc_eps(config_path: Path | str) -> float | None:
    """``fbr_auc_eps`` stored on the model ``reaction_prediction.model_path`` names."""
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    model = (cfg.get("reaction_prediction") or {}).get("model_path")
    if not model or not Path(model).is_file():
        return None
    attrs = json.loads(Path(model).read_text()).get("learner", {}).get("attributes", {})
    value = attrs.get(MODEL_EPS_ATTR)
    return float(value) if value is not None else None


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


def _window_frames(
    n: int, fps: float, odor_on_s: float, odor_off_s: float
) -> tuple[int, int, int] | None:
    """``(before_start, odor_on, odor_off)`` frame indices, or None."""
    if not (math.isfinite(fps) and fps > 0):
        return None
    if not (math.isfinite(odor_on_s) and math.isfinite(odor_off_s)):
        return None
    a, z = int(round(odor_on_s * fps)), int(round(odor_off_s * fps))
    if not 0 < a < z <= n:
        return None
    return max(0, a - (z - a)), a, z


def _mean_or_nan(seg: np.ndarray) -> float:
    finite = seg[np.isfinite(seg)]
    if seg.size == 0 or finite.size < MIN_FINITE_FRACTION * seg.size:
        return float("nan")
    return float(finite.mean())


def trial_auc(
    trace: np.ndarray, *, fps: float, odor_on_s: float, odor_off_s: float
) -> dict[str, float]:
    """Signed and unsigned area before and during the odor for one trial.

    Missing frames are skipped and the area is the mean deviation times the
    window's duration, so a short tracking dropout does not shrink it. The raw
    areas and ``ratio_raw`` instead follow the raster CSV's own definition
    (:func:`binarized_per_rasters.window_area`) so they match it exactly.
    """
    out = {"baseline": float("nan"),
           **{m: float("nan") for m in ALL_MEASURES + RAW_AREAS}}
    env = np.asarray(trace, dtype=float)
    frames = _window_frames(env.size, fps, odor_on_s, odor_off_s)
    if frames is None:
        return out
    b0, a, z = frames
    before = env[b0:a]
    if np.isfinite(before).sum() < MIN_FINITE_FRACTION * before.size:
        return out
    baseline = float(np.nanmedian(before))
    out["baseline"] = baseline
    duration = (z - a) / fps
    for window, seg in (("before", before), ("during", env[a:z])):
        dev = seg - baseline
        out[f"{window}_signed"] = _mean_or_nan(dev) * duration
        out[f"{window}_unsigned"] = _mean_or_nan(np.abs(dev)) * duration
    for kind in KINDS:
        out[f"delta_{kind}"] = out[f"during_{kind}"] - out[f"before_{kind}"]
    out["auc_before_raw"] = bpr.window_area(env, fps=fps, start_s=0.0, end_s=odor_on_s)
    out["auc_during_raw"] = bpr.window_area(env, fps=fps, start_s=odor_on_s,
                                            end_s=odor_off_s)
    out["ratio_raw"] = bpr.auc_ratio(out["auc_during_raw"], out["auc_before_raw"])
    out["ratio_unsigned"] = bpr.auc_ratio(out["during_unsigned"], out["before_unsigned"])
    return out


def _signed_peak(z: np.ndarray) -> float:
    """``z`` at the frame of largest ``|z|``, sign kept."""
    finite = np.isfinite(z)
    if z.size == 0 or finite.sum() < MIN_FINITE_FRACTION * z.size:
        return float("nan")
    return float(z[np.nanargmax(np.abs(z))])


def trial_peak_z(
    trace: np.ndarray, *, fps: float, odor_on_s: float, odor_off_s: float
) -> dict[str, float]:
    """Signed and unsigned peak z-score before and during the odor, one trial."""
    out = {"z_baseline_mean": float("nan"), "z_baseline_sd": float("nan"),
           **{m: float("nan") for m in PEAKZ.measures}}
    env = np.asarray(trace, dtype=float)
    frames = _window_frames(env.size, fps, odor_on_s, odor_off_s)
    if frames is None:
        return out
    b0, a, z = frames
    before = env[b0:a]
    finite = before[np.isfinite(before)]
    if finite.size < max(2, MIN_FINITE_FRACTION * before.size):
        return out
    mean, sd = float(finite.mean()), float(finite.std(ddof=1))
    out["z_baseline_mean"], out["z_baseline_sd"] = mean, sd
    if not sd > 0:
        return out                      # a flat baseline has no noise to scale by
    for window, seg in (("before", before), ("during", env[a:z])):
        peak = _signed_peak((seg - mean) / sd)
        out[PEAKZ.col(window, "signed")] = peak
        out[PEAKZ.col(window, "unsigned")] = abs(peak)
    for kind in KINDS:
        out[PEAKZ.col("delta", kind)] = (out[PEAKZ.col("during", kind)]
                                         - out[PEAKZ.col("before", kind)])
    before_u = out[PEAKZ.col("before", "unsigned")]
    if before_u > 0:
        out["zpeak_ratio_unsigned"] = out[PEAKZ.col("during", "unsigned")] / before_u
        out["zpeak_ratio_signed"] = out[PEAKZ.col("during", "signed")] / before_u
    return out


def trial_threshold_auc(
    before: float, during: float, *, auc_eps: float | None
) -> dict[str, float]:
    """The wide table's thresholded areas and the scorer's log ratio of them.

    Same arithmetic as ``flybehavior_response.xgb_ordinal.auc_log_ratios``:
    negative areas are clipped to 0 before the log.
    """
    b, d = float(before), float(during)
    log_ratio = float("nan")
    if auc_eps is not None and math.isfinite(b) and math.isfinite(d):
        log_ratio = math.log(max(d, 0.0) + auc_eps) - math.log(max(b, 0.0) + auc_eps)
    return {"thr_auc_before": b, "thr_auc_during": d, "thr_auc_delta": d - b,
            "thr_auc_log_ratio": log_ratio}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


_THRESHOLD_COLUMNS = ("AUC-Before", "AUC-During")


def _threshold_auc_lookup(df: pd.DataFrame) -> dict[tuple[str, str, str, str], tuple]:
    """``(dataset, fly, fly number, trial label) -> (AUC-Before, AUC-During)``."""
    if not set(_THRESHOLD_COLUMNS) <= set(df.columns):
        return {}
    d = ev._normalise_fly_columns(
        df[["dataset", "fly", "fly_number", "trial_label", *_THRESHOLD_COLUMNS]].copy())
    d["dataset"] = d["dataset"].map(ev._canon_dataset)
    return {
        (str(ds), str(fly), str(num), str(label)): (before, during)
        for ds, fly, num, label, before, during in d.itertuples(index=False, name=None)
    }


def build_auc_table(
    pretest: pd.DataFrame,
    testing: pd.DataFrame,
    *,
    cohorts: Sequence[str] | None = None,
    auc_eps: float | None = None,
) -> pd.DataFrame:
    """One row per fly x odor x phase (flies with both phases), every metric.

    ``auc_eps`` is the scoring model's AUC epsilon; without it the log ratio
    stays missing (the thresholded areas themselves are still filled).
    """
    lookups = {phase: _threshold_auc_lookup(df)
               for phase, df in zip(PHASES, (pretest, testing))}
    nan2 = (float("nan"), float("nan"))
    records = []
    for pair in pvt.pair_flies(pretest, testing, cohorts=cohorts):
        for phase, traces in zip(PHASES, (pair.pre, pair.post)):
            for odor in pair.odors:
                trace = traces.get(odor)
                if trace is None:
                    continue
                on = trace.odor_on_s if trace.odor_on_s is not None else float("nan")
                off = trace.odor_off_s if trace.odor_off_s is not None else float("nan")
                timing = dict(fps=trace.fps, odor_on_s=on, odor_off_s=off)
                thr = lookups[phase].get(
                    (pair.dataset, pair.fly, str(pair.fly_number), trace.trial_label), nan2)
                records.append({
                    "dataset": pair.dataset,
                    "fly": pair.fly,
                    "fly_number": str(pair.fly_number),
                    "phase": phase,
                    "odor": odor,
                    "trained": pvt._is_trained_row(pair, odor),
                    "trial_label": trace.trial_label,
                    "odor_on_s": on,
                    "odor_off_s": off,
                    **trial_auc(trace.env, **timing),
                    **trial_peak_z(trace.env, **timing),
                    **trial_threshold_auc(*thr, auc_eps=auc_eps),
                })
    return pd.DataFrame.from_records(records)


_CONC_SUFFIX = re.compile(r"-\d+(?:\.\d+)?$")


def _bare_label(label: str) -> str:
    return _CONC_SUFFIX.sub("", str(label))


def attach_scores(t: pd.DataFrame, predictions_csv: Path | str) -> pd.DataFrame:
    """Model score per trial, joined on (dataset, fly, fly number, trial label).

    Batches after ``~/odor_pins.txt`` write ``testing_2_hexanol-0.1`` in
    model_predictions.csv but the subset tables may carry ``testing_2_hexanol``;
    the label without its concentration suffix is the fallback key.
    """
    scores = pvt.load_scores(predictions_csv)
    bare: dict[tuple[str, str, str, str], int] = {}
    for (ds, fly, num, label), s in scores.items():
        bare.setdefault((ds, fly, num, _bare_label(label)), s)
    matched: list[int | None] = []
    for key in t[["dataset", "fly", "fly_number", "trial_label"]].astype(str).itertuples(
        index=False, name=None
    ):
        s = scores.get(key)
        if s is None:
            s = bare.get((*key[:3], _bare_label(key[3])))
        matched.append(s)
    out = t.copy()
    out["score"] = pd.array(matched, dtype="Int64")
    return out


def per_trial_table(t: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    """The columns of one metric (plus ids and score) for its CSV."""
    cols = ID_COLUMNS + list(metric.extra_columns) + list(metric.measures)
    if "score" in t.columns:
        cols.append("score")
    return t[[c for c in cols if c in t.columns]]


def _odor_order(t: pd.DataFrame) -> list[str]:
    return pvt.order_odors(t["odor"].unique())


def _values(t: pd.DataFrame, phase: str, odor: str, measure: str) -> np.ndarray:
    sub = t.loc[t["phase"] == phase]
    if odor != ALL_ODORS:
        sub = sub.loc[sub["odor"] == odor]
    v = sub[measure].to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _sem(v: np.ndarray) -> float:
    return float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan")


def _wilcoxon(x: np.ndarray, y: np.ndarray | None = None) -> float:
    from scipy.stats import wilcoxon

    d = x if y is None else x - y
    d = d[np.isfinite(d)]
    if d.size < 2 or not np.any(d):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(d).pvalue)
        except ValueError:
            return float("nan")


_KEY = ["dataset", "fly", "fly_number", "odor"]


def paired_change(t: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Each fly's ``measure`` in both phases, side by side, plus post - pre."""
    wide = t.pivot_table(index=_KEY, columns="phase", values=measure, aggfunc="first",
                         dropna=False)
    wide = wide.reindex(columns=list(PHASES)).reset_index()
    wide.columns.name = None
    wide["change"] = wide["Post-training"] - wide["Pre-test"]
    return wide


def per_fly_change_table(t: pd.DataFrame, metric: Metric = AUC) -> pd.DataFrame:
    out = None
    for measure in metric.measures:
        p = paired_change(t, measure).rename(columns={
            "Pre-test": f"{measure}_pre",
            "Post-training": f"{measure}_post",
            "change": f"{measure}_post_minus_pre",
        })
        out = p if out is None else out.merge(p, on=_KEY, how="outer")
    trained = t.groupby(_KEY)["trained"].any().reset_index()
    return trained.merge(out, on=_KEY, how="right")


def summarise(t: pd.DataFrame, metric: Metric = AUC) -> pd.DataFrame:
    """Mean / SEM / n per phase x odor x measure, plus the pooled group.

    ``p_vs_null``: one-sample Wilcoxon of a delta or log ratio against 0, or a
    plain ratio against 1 (did the odor change the trace at all?).
    ``p_pre_vs_post``: Wilcoxon of the same measure paired by fly across
    phases. Both uncorrected, per odor only.
    """
    rows = []
    for odor in _odor_order(t) + [ALL_ODORS]:
        for measure in metric.measures:
            p_pair = float("nan")
            if odor != ALL_ODORS:
                pc = paired_change(t.loc[t["odor"] == odor], measure)
                p_pair = _wilcoxon(pc["Post-training"].to_numpy(float),
                                   pc["Pre-test"].to_numpy(float))
            has_null = measure in metric.tested and odor != ALL_ODORS
            for phase in PHASES:
                v = _values(t, phase, odor, measure)
                rows.append({
                    "phase": phase,
                    "odor": odor,
                    "measure": measure,
                    "n_trials": int(v.size),
                    "mean": float(v.mean()) if v.size else float("nan"),
                    "sem": _sem(v),
                    "median": float(np.median(v)) if v.size else float("nan"),
                    "p_vs_null": (_wilcoxon(v - null_value(measure)) if has_null
                                  else float("nan")),
                    "p_pre_vs_post": p_pair,
                })
    return pd.DataFrame(rows)


def _panel_rows(t: pd.DataFrame, panel: str) -> pd.DataFrame:
    return t if panel == COMBINED else t.loc[t["phase"] == panel]


def score_correlations(
    t: pd.DataFrame, measures: Sequence[str] = ALL_MEASURES
) -> pd.DataFrame:
    """Spearman rank correlation of each measure with the model score.

    Trial-level: a fly contributes up to seven trials per phase, so the p-value
    treats non-independent trials as independent and is descriptive only.
    """
    from scipy.stats import spearmanr

    rows = []
    for measure in measures:
        for panel in PANELS:
            sub = _panel_rows(t, panel)
            x = pd.to_numeric(sub["score"], errors="coerce").to_numpy(float)
            y = sub[measure].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            rho = p = float("nan")
            if ok.sum() >= 3 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
                res = spearmanr(x[ok], y[ok])
                rho, p = float(res.statistic), float(res.pvalue)
            rows.append({"phase": panel, "measure": measure, "n_trials": int(ok.sum()),
                         "rho": rho, "p": p})
    return pd.DataFrame(rows)


def score_summary_table(t: pd.DataFrame, metric: Metric = AUC) -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        sub = _panel_rows(t, panel)
        score = pd.to_numeric(sub["score"], errors="coerce")
        for measure in metric.measures:
            for s in ss.SCORES:
                v = sub.loc[score == s, measure].to_numpy(float)
                v = v[np.isfinite(v)]
                rows.append({
                    "phase": panel, "measure": measure, "score": s,
                    "n_trials": int(v.size),
                    "mean": float(v.mean()) if v.size else float("nan"),
                    "sem": _sem(v),
                    "median": float(np.median(v)) if v.size else float("nan"),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def wording_cohort(datasets: Iterable[str]) -> str | None:
    """The cohort a pooled figure is worded for: one of ``datasets`` when every
    one is a *-Sensitivity-* cohort (pre-/post-exposure, sensitized odor),
    else None (pre-test / post-training / CS+). A mix keeps the old words."""
    names = sorted({str(d) for d in datasets})
    return names[0] if phase_terms.all_sensitivity(names) else None


def _cohort(t: pd.DataFrame) -> str | None:
    return wording_cohort(t["dataset"].dropna() if "dataset" in t.columns else ())


def phase_label(phase: str, cohort: str | None = None) -> str:
    """A PHASES / PANELS value as a figure shows it. Only the label follows the
    cohort; the tables' ``phase`` column keeps PHASES."""
    if phase == PHASES[0]:
        return phase_terms.pre_label(cohort, phase)
    if phase == PHASES[1]:
        return phase_terms.post_label(cohort, phase)
    return phase


def _phase_words(cohort: str | None) -> tuple[str, str]:
    """The two phases mid-sentence: ("pre-test", "post-training") or pre-/post-exposure."""
    return (phase_terms.pre_label(cohort, "pre-test", case="lower"),
            phase_terms.post_label(cohort, "post-training", case="lower"))


def _group_x(groups: list[str]) -> np.ndarray:
    """Odors at 0..k-1; the pooled group set off by a gap."""
    x = np.arange(len(groups), dtype=float)
    if groups and groups[-1] == ALL_ODORS:
        x[-1] += 0.5
    return x


def _tick_labels(t: pd.DataFrame, groups: list[str], measure: str) -> list[str]:
    trained = set(t.loc[t["trained"], "odor"])
    tag = phase_terms.odor_tag(_cohort(t))
    labels = []
    for g in groups:
        n = _values(t, PHASES[0], g, measure).size
        name = f"{g}\n({tag})" if g in trained else g
        unit = "trials" if g == ALL_ODORS else "flies"
        labels.append(f"{name}\nn={n} {unit}")
    return labels


def _style_xaxis(ax, t: pd.DataFrame, groups: list[str], x: np.ndarray,
                 measure: str) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(_tick_labels(t, groups, measure), fontsize=9)
    trained = set(t.loc[t["trained"], "odor"])
    for tick, g in zip(ax.get_xticklabels(), groups):
        if g in trained:
            tick.set_fontweight("bold")
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)


def _style_axes(ax, null: float = 0.0) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(null, color=INK_MUTED, lw=0.8, zorder=1)
    ax.grid(axis="y", color="#e6e6e3", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


_BAR_NOTE = ("One trial per fly per odor per phase; bars = mean +/- SEM over trials; "
             "'All odors' pools every trial.")


def _footnote(fig, metric: Metric, extra: str) -> None:
    fig.text(0.01, 0.005, metric.definition + extra, fontsize=7.5,
             color=INK_MUTED, ha="left", va="bottom", wrap=True)


def _legend_under_title(fig, handles, ncol: int) -> None:
    fig.legend(handles=handles, loc="upper left", ncol=ncol, frameon=False, fontsize=9,
               bbox_to_anchor=(0.005, 0.962))


def _jitter(n: int, seed: int, spread: float) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-spread, spread, size=n)


def _layout(fig, n_rows: int, note: bool = False) -> None:
    """tight_layout with room for the title block and the wrapped footnote."""
    single = n_rows == 1
    bottom = (0.06 if note else 0.05) * (2.0 if single else 1.0)
    fig.tight_layout(rect=(0, bottom, 1, 0.88 if single else 0.93))


def plot_before_vs_during(t: pd.DataFrame, title: str, metric: Metric = AUC):
    groups = _odor_order(t) + [ALL_ODORS]
    x = _group_x(groups)
    slots = [(phase, window) for phase in PHASES for window in WINDOWS]
    width = 0.19
    offsets = (np.arange(len(slots)) - (len(slots) - 1) / 2) * (width + 0.015)
    n = len(metric.kinds)
    fig, axes = plt.subplots(n, 1, squeeze=False,
                             figsize=(max(10.0, 1.55 * len(groups) + 2.5),
                                      4.75 * n + (1.0 if n == 1 else 0)))
    for ax, kind in zip(axes[:, 0], metric.kinds):
        for gi, (odor, x0) in enumerate(zip(groups, x)):
            for si, ((phase, window), off) in enumerate(zip(slots, offsets)):
                v = _values(t, phase, odor, metric.col(window, kind))
                if not v.size:
                    continue
                color = PHASE_COLOR[phase]
                ax.bar(x0 + off, v.mean(), width, color=color if window == "during" else "white",
                       edgecolor=color, linewidth=1.6, zorder=2)
                ax.errorbar(x0 + off, v.mean(), yerr=_sem(v), color=INK, lw=1.1,
                            capsize=2.5, zorder=4)
                ax.scatter(x0 + off + _jitter(v.size, gi * 10 + si, width * 0.3), v,
                           s=7, color=INK, alpha=0.3 if odor != ALL_ODORS else 0.15,
                           edgecolors="none", zorder=3)
        _style_axes(ax)
        ax.set_ylabel(f"{metric.label(kind)} ({metric.unit})")
        ax.set_title(metric.kind_title[kind], fontsize=10.5, loc="left", color=INK)
        _style_xaxis(ax, t, groups, x, metric.col("delta", kind))
    cohort = _cohort(t)
    handles = [
        Patch(facecolor="white" if w == "before" else PHASE_COLOR[p],
              edgecolor=PHASE_COLOR[p], linewidth=1.6,
              label=f"{phase_label(p, cohort)}, {w} odor")
        for p in PHASES for w in WINDOWS
    ]
    _legend_under_title(fig, handles, ncol=4)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=12)
    _footnote(fig, metric, _BAR_NOTE + " Dots = individual trials.")
    _layout(fig, n)
    return fig


def _p_text(p: float) -> str:
    if not math.isfinite(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def plot_paired_dataset(t: pd.DataFrame, rows: Sequence[Row], title: str, note: str = "",
                        metric: Metric = AUC):
    """Per odor: pre-test vs post-training, each fly's two trials joined.

    Bars rise from the measure's null (0 for a delta, 1 for a ratio).
    """
    groups = _odor_order(t) + [ALL_ODORS]
    x = _group_x(groups)
    half = 0.2
    fig, axes = plt.subplots(len(rows), 1, squeeze=False,
                             figsize=(max(9.0, 1.3 * len(groups) + 2.5),
                                      4.75 * len(rows) + (1.0 if len(rows) == 1 else 0)))
    for ax, row in zip(axes[:, 0], rows):
        null = row.null
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for gi, (odor, x0) in enumerate(zip(groups, x)):
            pooled = odor == ALL_ODORS
            sub = t if pooled else t.loc[t["odor"] == odor]
            pc = paired_change(sub, row.measure)
            if not pooled:
                for pre, post in pc[list(PHASES)].to_numpy(float):
                    if np.isfinite(pre) and np.isfinite(post):
                        ax.plot([x0 - half, x0 + half], [pre, post], color="#b5b5b0",
                                lw=0.7, zorder=2)
            for side, phase in zip((-1, 1), PHASES):
                v = _values(t, phase, odor, row.measure)
                if not v.size:
                    continue
                color = PHASE_COLOR[phase]
                xc = x0 + side * half
                ax.bar(xc, v.mean() - null, 2 * half - 0.04, bottom=null, color=color,
                       alpha=0.22, edgecolor=color, linewidth=1.6, zorder=1)
                xs = xc + (_jitter(v.size, gi, 0.06) if pooled else np.zeros(v.size))
                ax.scatter(xs, v, s=6 if pooled else 12,
                           color=color, alpha=0.35 if pooled else 0.9,
                           edgecolors="white", linewidths=0.4, zorder=3)
                ax.errorbar(xc, v.mean(), yerr=_sem(v), color=INK, lw=1.3, capsize=3,
                            zorder=4)
                ax.plot([xc - half + 0.04, xc + half - 0.04], [v.mean()] * 2, color=color,
                        lw=2.2, zorder=4)
            if not pooled:
                p = _wilcoxon(pc["Post-training"].to_numpy(float), pc["Pre-test"].to_numpy(float))
                ax.text(x0, 0.985, _p_text(p), transform=trans, ha="center", va="top",
                        fontsize=8.5, color=INK)
        _style_axes(ax, null)
        ax.set_ylabel(row.ylabel)
        ax.set_title(row.title, fontsize=10.5, loc="left", color=INK, pad=14)
        _style_xaxis(ax, t, groups, x, row.measure)
    cohort = _cohort(t)
    handles = [Patch(facecolor=PHASE_COLOR[p], alpha=0.5, edgecolor=PHASE_COLOR[p],
                     label=phase_label(p, cohort)) for p in PHASES]
    handles.append(Line2D([], [], color="#b5b5b0", lw=0.9, label="same fly"))
    _legend_under_title(fig, handles, ncol=3)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=12)
    pre_name, post_name = _phase_words(cohort)
    _footnote(fig, metric, (note + " " if note else "") + _BAR_NOTE
              + " Dots = individual flies (one trial each); lines join a fly's "
                f"{pre_name} and {post_name} trial. p = Wilcoxon signed-rank, "
                "post vs pre paired by fly, uncorrected; none for the pooled group "
                "(seven trials per fly are not independent).")
    _layout(fig, len(rows), note=bool(note))
    return fig


def _fly_order(t: pd.DataFrame) -> list[tuple[str, str, str]]:
    keys = t[["dataset", "fly", "fly_number"]].drop_duplicates().itertuples(index=False,
                                                                             name=None)
    return sorted(keys, key=lambda k: (k[0], *ev._fly_sort_key(k[1], k[2])))


def plot_per_fly_matrix(t: pd.DataFrame, title: str, metric: Metric = AUC, note: str = ""):
    odors = _odor_order(t)
    flies = _fly_order(t)
    trained = set(t.loc[t["trained"], "odor"])
    cohort = _cohort(t)
    tag = phase_terms.odor_tag(cohort)
    # "(sensitized odor)" on its own line: as one rotated line it runs into the footnote
    sep = "\n" if phase_terms.is_sensitivity(cohort) else " "
    columns = (*PHASES, "Post - Pre")
    rows = metric.heatmap_rows()
    n_f, n_o, n_r = len(flies), len(odors), len(rows)
    fig = plt.figure(figsize=(3.2 + 3 * (0.62 * n_o + 0.3),
                              n_r * (0.3 * n_f + 1.9) + 0.6 + (0.6 if n_r == 1 else 0)))
    gs = GridSpec(n_r, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.08,
                  hspace=0.3, left=0.17, right=0.95, top=0.93 if n_r > 1 else 0.9,
                  bottom=0.08 if n_r > 1 else 0.16)
    row_index = {f: i for i, f in enumerate(flies)}
    col_index = {o: j for j, o in enumerate(odors)}
    for r, row in enumerate(rows):
        mats = {}
        for phase in PHASES:
            m = np.full((n_f, n_o), np.nan)
            sub = t.loc[t["phase"] == phase]
            for ds, fly, num, odor, v in sub[["dataset", "fly", "fly_number", "odor",
                                              row.measure]].itertuples(index=False, name=None):
                m[row_index[(ds, fly, num)], col_index[odor]] = v
            mats[phase] = m
        mats["Post - Pre"] = mats["Post-training"] - mats["Pre-test"]
        allv = np.concatenate([m[np.isfinite(m)] - (row.null if k in PHASES else 0.0)
                               for k, m in mats.items()])
        vmax = float(np.nanpercentile(np.abs(allv), 98)) if allv.size else 1.0
        vmax = max(vmax, 1e-9)
        short = row.short_label
        for c, col in enumerate(columns):
            ax = fig.add_subplot(gs[r, c])
            centre = row.null if col in PHASES else 0.0
            m = mats[col]
            im = ax.imshow(m - centre, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                           interpolation="nearest")
            for i in range(n_f):
                for j in range(n_o):
                    v = m[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, metric.fmt.format(v), ha="center", va="center",
                                fontsize=6,
                                color="white" if abs(v - centre) > 0.6 * vmax else INK)
            ax.set_xticks(range(n_o))
            ax.set_xticklabels([f"{o}{sep}({tag})" if o in trained else o for o in odors],
                               rotation=40, ha="right", fontsize=8)
            for tick, o in zip(ax.get_xticklabels(), odors):
                if o in trained:
                    tick.set_fontweight("bold")
            ax.set_yticks(range(n_f))
            if c == 0:
                ax.set_yticklabels([f"{fly} fly{num}" for _, fly, num in flies], fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            head = phase_label(col, cohort) if col in PHASES else "Post - Pre (change)"
            ax.set_title(f"{short}: {head}", fontsize=9.5, color=INK, loc="left")
        cax = fig.add_subplot(gs[r, 3])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"{short} (relative to {row.null:g})" if row.null else short,
                     fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=12)
    pre_name, post_name = _phase_words(cohort)
    _footnote(fig, metric, (note + " " if note else "")
              + f"Each cell is one trial's value (Post - Pre: the {post_name} value "
                f"minus the {pre_name} value). Colour scale clipped at the 98th percentile "
                "of |value| per row.")
    return fig


def plot_auc_vs_score(t: pd.DataFrame, rows: Sequence[Row], title: str, note: str = "",
                      metric: Metric = AUC):
    """Metric distribution at each model score: pre-test, post-training, pooled."""
    corr = score_correlations(t, [r.measure for r in rows]).set_index(["phase", "measure"])
    cohort = _cohort(t)
    fig, axes = plt.subplots(len(rows), len(PANELS),
                             figsize=(15.5, 4.9 * len(rows) + (1.0 if len(rows) == 1 else 0)),
                             sharey="row", squeeze=False)
    box_edge = "#8a8a85"
    for r, row in enumerate(rows):
        for c, panel in enumerate(PANELS):
            ax = axes[r, c]
            sub = _panel_rows(t, panel)
            score = pd.to_numeric(sub["score"], errors="coerce")
            labels = []
            for s in ss.SCORES:
                at = sub.loc[score == s]
                v = at[row.measure].to_numpy(float)
                ok = np.isfinite(v)
                v, phases = v[ok], at["phase"].to_numpy()[ok]
                labels.append(f"{s}\nn={v.size}")
                if not v.size:
                    continue
                if v.size >= 3:
                    ax.boxplot([v], positions=[s], widths=0.6, showfliers=False,
                               patch_artist=True, zorder=1,
                               boxprops=dict(facecolor="#f1f1ef", edgecolor=box_edge, lw=1),
                               medianprops=dict(color=INK, lw=1.6),
                               whiskerprops=dict(color=box_edge, lw=1),
                               capprops=dict(color=box_edge, lw=1))
                for pi, phase in enumerate(PHASES):
                    m = phases == phase
                    if m.any():
                        seed = 1000 * r + 100 * c + 10 * (s + 1) + pi
                        ax.scatter(s + _jitter(int(m.sum()), seed, 0.22), v[m], s=13,
                                   color=PHASE_COLOR[phase], alpha=0.75,
                                   edgecolors="white", linewidths=0.4, zorder=3)
                ax.scatter([s], [v.mean()], marker="D", s=30, color=INK,
                           edgecolors="white", linewidths=0.6, zorder=4)
            _style_axes(ax, row.null)
            ax.axvline(ss.REACTION_BOUNDARY_Y, color=INK_MUTED, lw=0.9, ls="--", zorder=1)
            ax.set_xticks(ss.SCORES)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_xlim(ss.SCORES[0] - 0.6, ss.SCORES[-1] + 0.6)
            k = corr.loc[(panel, row.measure)]
            rho = (f"Spearman rho = {k['rho']:.2f}, {_p_text(k['p'])}"
                   if math.isfinite(k["rho"]) else "Spearman rho: n/a")
            ax.text(0.02, 0.97, f"{rho}\nn = {int(k['n_trials'])} trials",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
            if r == 0:
                ax.set_title(phase_label(panel, cohort), fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(f"{row.ylabel}")
            if r == len(rows) - 1:
                ax.set_xlabel("Model score (-1 retraction ... 5 strongest PER)")
        # Wider than its axes: kept out of tight_layout, or it squeezes column 0.
        axes[r, 0].annotate(row.title, xy=(0, 1), xycoords="axes fraction",
                            xytext=(0, 22), textcoords="offset points", fontsize=10,
                            color=INK_MUTED, ha="left", va="bottom").set_in_layout(False)
    handles = [Line2D([], [], marker="o", ls="", color=PHASE_COLOR[p], markersize=6,
                      label=f"{phase_label(p, cohort)} trial") for p in PHASES]
    handles += [
        Line2D([], [], marker="D", ls="", color=INK, markersize=6, label="mean"),
        Patch(facecolor="#f1f1ef", edgecolor=box_edge, label="median + IQR (n >= 3)"),
        Line2D([], [], color=INK_MUTED, ls="--", lw=0.9, label="reaction boundary (score >= 2)"),
    ]
    _legend_under_title(fig, handles, ncol=5)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=12)
    _footnote(fig, metric, (note + " " if note else "")
              + "Each dot is one trial; whiskers 1.5 IQR. rho = Spearman rank correlation "
                "of the measure with the model score over trials; its p treats trials as "
                "independent although a fly gives up to 7 per phase (14 pooled), so read "
                "it as descriptive.")
    single = len(rows) == 1
    fig.tight_layout(rect=(0, 0.11 if single else 0.06, 1, 0.88 if single else 0.925))
    if not single:
        fig.subplots_adjust(hspace=0.42)          # room for the row titles
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pretest-wide-csv", required=True, type=Path)
    p.add_argument("--testing-wide-csv", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--metric", choices=sorted(METRICS), default=AUC.key,
                   help="auc: area under the trace; peakz: peak z-score; logratio: the "
                        "thresholded AUC and the scorer's AUC-During-Before-LogRatio.")
    p.add_argument("--predictions-csv", type=Path, default=None,
                   help="Model scores per trial; adds the metric-vs-score figures.")
    p.add_argument("--config", type=Path, default=None,
                   help="Pipeline config: per-dataset odor_remap display names, and the "
                        "scoring model whose stored AUC epsilon --metric logratio uses.")
    p.add_argument("--auc-eps", type=float, default=None,
                   help="Override the scoring model's AUC epsilon (--metric logratio).")
    p.add_argument("--cohorts", nargs="*", default=None,
                   help="Only these datasets (default: every fly with both phases).")
    p.add_argument("--label", default="",
                   help="Appended to the figure titles, e.g. 'start 1pm-2pm'.")
    return p


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metric = METRICS[args.metric]
    eps = args.auc_eps
    if eps is None and args.config is not None:
        eps = model_auc_eps(args.config)
    if metric is LOGRATIO:
        if eps is None:
            parser.error("--metric logratio needs --auc-eps, or a --config whose "
                         f"reaction_prediction.model_path stores {MODEL_EPS_ATTR}")
        metric = dataclasses.replace(metric, ratio_note=metric.ratio_note.format(eps=eps))
    if args.config is not None:
        pvt._register_odor_remaps(args.config)
    t = build_auc_table(read_table(args.pretest_wide_csv), read_table(args.testing_wide_csv),
                        cohorts=args.cohorts, auc_eps=eps)
    if t.empty:
        print("[auc-before-during] no fly has both a pre-test and a post-training panel")
        return 1
    if args.predictions_csv is not None:
        t = attach_scores(t, args.predictions_csv)
        print(f"[auc-before-during] scores matched for {int(t['score'].notna().sum())}"
              f" of {len(t)} trials")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key, stem = metric.key, metric.stem
    per_trial_table(t, metric).to_csv(out / f"{key}_per_trial.csv", index=False)
    per_fly_change_table(t, metric).to_csv(out / f"{key}_per_fly_change.csv", index=False)
    summarise(t, metric).to_csv(out / f"{key}_summary.csv", index=False)

    datasets = ", ".join(sorted(t["dataset"].unique()))
    base = f"{datasets}{' (' + args.label + ')' if args.label else ''}"
    n_flies = t[["dataset", "fly", "fly_number"]].drop_duplicates().shape[0]
    pre_name, post_name = _phase_words(_cohort(t))
    suffix = f"{pre_name} vs {post_name} ({n_flies} flies)"
    heat_note = metric.ratio_note if metric.heatmap == "ratio" else ""
    _save(plot_before_vs_during(t, f"{base}: {metric.what} before vs during odor, {suffix}",
                                metric),
          out / f"{stem}_before_vs_during_dataset.png")
    _save(plot_paired_dataset(t, metric.rows("during"),
                              f"{base}: {metric.what}, during odor only, {suffix}",
                              metric=metric),
          out / f"{stem}_during_dataset.png")
    _save(plot_paired_dataset(t, metric.rows("delta"),
                              f"{base}: {metric.what}, during minus before, {suffix}",
                              metric=metric),
          out / f"{stem}_during_minus_before_dataset.png")
    _save(plot_paired_dataset(t, metric.ratios,
                              f"{base}: {metric.what}, during / before, {suffix}",
                              note=metric.ratio_note, metric=metric),
          out / f"{stem}_ratio_during_over_before_dataset.png")
    _save(plot_per_fly_matrix(t, f"{base}: {metric.what}, per fly, {suffix}", metric,
                              note=heat_note),
          out / f"{stem}_{metric.heatmap_file}_per_fly.png")

    if "score" in t.columns and t["score"].notna().any():
        score_correlations(t, metric.measures).to_csv(
            out / f"{key}_vs_score_correlation.csv", index=False)
        score_summary_table(t, metric).to_csv(out / f"{key}_vs_score_summary.csv",
                                              index=False)
        for rows, name, what, note in (
            (metric.rows("delta"), "during_minus_before", "during minus before", ""),
            (metric.rows("during"), "during", "during odor", ""),
            (metric.ratios, "ratio", "during / before", metric.ratio_note),
        ):
            _save(plot_auc_vs_score(
                t, rows, f"{base}: {metric.what} ({what}) vs model score ({n_flies} flies)",
                note=note, metric=metric), out / f"{stem}_vs_score_{name}.png")
    print(f"[auc-before-during] {metric.key}: {n_flies} flies, {len(t)} trials -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
