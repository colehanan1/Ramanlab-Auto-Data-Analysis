"""Does conditioning vigor predict a fly's test response?

Control flies get the odor during "training" with nothing paired to it. This
module asks whether the flies that nonetheless extended most during that
odor-only exposure are the same flies that react to the odor at test — i.e.
whether the control cohort's test response is a trait of the individual rather
than anything the protocol did to it.

Four figures per cohort, all keyed on the same per-fly join:

``<dataset>_per_vs_training_auc.png|svg``
    Outcome = **binary PER**. Row 1: training AUC split by whether the fly
    reacted, one column per test presentation plus a pooled column
    (reacted on either). Row 2: P(react) against training AUC — a logistic
    fit with the observed rate in equal-count AUC bins (Wilson intervals).

``<dataset>_score_vs_training_auc.png|svg``
    Outcome = **score (−1…5)**. Row 1: training AUC against testing score with
    a Spearman fit. Row 2: mean testing score by equal-count AUC bin.

``<dataset>_trial_matrix.png|svg``
    Every *individual* conditioning trial against every testing outcome —
    Spearman ρ as a diverging heatmap, so a trial that carries the signal on
    its own is visible against the pooled mean. Benjamini-Hochberg over the
    whole grid, because the grid is a scan.

``<dataset>_trial_profile.png|svg``
    The same ρ values read as a trajectory: does the predictive power build
    across conditioning trials, or is it there from trial 1?

The join is ``(fly, fly_number)``: AUC from the wide envelope tables,
score/PER from the predictions CSV. Flies missing either side are dropped, so
the fly set is exactly the one the published figures use.

Usage::

    python scripts/analysis/training_auc_vs_control_response.py \\
        --training-wide-csv .../all_envelope_rows_wide_combined_base_training.parquet \\
        --testing-wide-csv .../all_envelope_rows_wide_combined_base.parquet \\
        --predictions-csv .../model_predictions.csv \\
        --dataset Hex-Control-24-0.1 \\
        --out-dir .../New-Opto-Fly-Figures/Training-AUC-vs-Testing-Response
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.corrected_stats import (  # noqa: E402
    bh_adjust,
    holm_adjust,
    score_test,
    wilson_ci,
)
from scripts.analysis.score_scale_figure import SCORE_COLORS  # noqa: E402

LOGGER = logging.getLogger("training_auc_vs_control_response")

DPI = 400

# --------------------------------------------------------------------------- #
# Palette
#
# The two PER groups reuse the poles of the project's pinned PRGn score ramp
# (score_scale_figure.SCORE_COLORS): purple = the no-reaction arm, green = the
# reaction arm, so "green means the fly reacted" carries the same meaning here
# as in every score figure. Validated as a categorical pair on white:
# worst all-pairs CVD dE 16.7 (deutan), normal-vision dE 29.4, both >= 3:1
# contrast. Never a red/green split -- see score_summary.SCORE_COLORS' note.
# --------------------------------------------------------------------------- #
NO_PER_COLOR = SCORE_COLORS[-1]   # "#762a83" purple
PER_COLOR = SCORE_COLORS[4]       # "#1b7837" green
POINT_COLOR = "#2a78d6"           # single-series scatter (identity is the title)
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FIT_INK = "#52514e"

#: Diverging ramp for Spearman rho: warm = positive, cool = negative, neutral
#: gray at zero. Deliberately NOT the score ramp -- a rho of +0.5 must never
#: read as "score 3".
RHO_NEG = "#2a78d6"
RHO_MID = "#f0efec"
RHO_POS = "#e34948"

#: Ordinal ramp for the low/mid/high training-AUC bins. Ordinal, not
#: categorical: swapping "low" and "high" would change the meaning, so the
#: reader must see the order in the colour. One hue, light->dark; the light end
#: clears 2:1 on white (validated with the ordinal gate).
BIN_RAMP = ("#86b6ef", "#3987e5", "#1c5cab")
BIN_NAMES = {1: ("all",), 2: ("lower half", "upper half"),
             3: ("low", "medium", "high"),
             4: ("lowest", "low", "high", "highest")}

REACTION_BOUNDARY = 1.5           # score >= 2 is a reaction (score_summary)
SCORE_MIN, SCORE_MAX = -1, 5
DEFAULT_BINS = 3

_RC = {
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

_TRIAL_RE = re.compile(r"^(?:pretest|training|testing)_(\d+)_(.+)$", re.IGNORECASE)

ODOR_SHORT = {
    "hexanol": "Hex",
    "3-octonol": "3Oct",
    "3-octanol": "3Oct",
    "ethylbutyrate": "EB",
    "ethyl butyrate": "EB",
    "citral": "Citral",
    "linalool": "Lin",
    "acv": "ACV",
    "benzaldehyde": "Benz",
    "isoamylacetate": "IAA",
    "isoamyl acetate": "IAA",
    "lightonly": "Light",
}

ODOR_PRETTY = {
    "hexanol": "Hexanol",
    "3-octonol": "3-Octanol",
    "3-octanol": "3-Octanol",
    "ethylbutyrate": "Ethyl Butyrate",
    "citral": "Citral",
    "linalool": "Linalool",
    "acv": "Apple Cider Vinegar",
    "benzaldehyde": "Benzaldehyde",
    "isoamylacetate": "Isoamyl Acetate",
    "lightonly": "Light only",
}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def is_trained_arm(dataset: object) -> bool:
    """Whether *dataset* names a TRAINED arm rather than a control one.

    Cohorts are named ``<odor>-{Training,Control}-<starvation>-<conc>``, so the
    arm is in the name. Anything that does not clearly say "Training" reads as
    not-trained: the captions then keep their original control wording, which is
    the safer failure -- it never claims a pairing the protocol may not have had.
    """
    return "-training-" in f"-{str(dataset).strip().strip('-').lower()}-"


def cohort_noun(dataset: object) -> str:
    """The noun for this cohort's flies, e.g. "control" / "trained"."""
    return "trained" if is_trained_arm(dataset) else "control"


def protocol_line(dataset: object, odor_label: object) -> str:
    """One sentence describing what this cohort got during conditioning.

    The control wording ("presented with nothing paired to it") is a statement
    about the protocol, and it is FALSE for a trained arm -- there the odor was
    paired with the light. Printing it on a trained figure would be a fabricated
    claim about how the experiment was run, so the sentence follows the arm.
    """
    if is_trained_arm(dataset):
        return (
            f"Trained cohort: {odor_label} presented during conditioning paired "
            f"with the light stimulus."
        )
    return (
        f"Control cohort: {odor_label} presented during conditioning with nothing "
        f"paired to it."
    )

def parse_trial(label: object) -> tuple[int, str] | None:
    """``"testing_8_hexanol"`` -> ``(8, "hexanol")``; ``None`` if unparseable."""
    m = _TRIAL_RE.match(str(label).strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip().lower()


def odor_short(token: str) -> str:
    """Short axis/legend label for an odor token."""
    key = str(token).strip().lower()
    return ODOR_SHORT.get(key, key[:4].capitalize())


def odor_pretty(token: str) -> str:
    """Full-word odor name for titles."""
    key = str(token).strip().lower()
    return ODOR_PRETTY.get(key, key.capitalize())


def training_odor_token(training_df: pd.DataFrame) -> str:
    """The conditioning odor: the modal token across the training trials.

    Modal rather than first, so one mislabelled trial cannot rename the odor
    the whole cohort was conditioned on.
    """
    tokens = [
        parsed[1]
        for parsed in (parse_trial(v) for v in training_df["trial_label"])
        if parsed is not None
    ]
    if not tokens:
        raise ValueError("No parseable training trial labels")
    return str(pd.Series(tokens).value_counts().idxmax())


def presentation_trials(testing_df: pd.DataFrame, token: str) -> list[int]:
    """Trial numbers on which ``token`` was presented at test, in order."""
    want = str(token).strip().lower()
    nums = {
        parsed[0]
        for parsed in (parse_trial(v) for v in testing_df["trial_label"])
        if parsed is not None and parsed[1] == want
    }
    return sorted(nums)


def auc_slope(auc_by_trial: pd.Series) -> float:
    """OLS slope of AUC against trial number; NaN with fewer than two trials."""
    clean = pd.Series(auc_by_trial).dropna()
    if len(clean) < 2:
        return float("nan")
    x = np.asarray(clean.index, dtype=float)
    y = np.asarray(clean.to_numpy(), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


# --------------------------------------------------------------------------- #
# The join
# --------------------------------------------------------------------------- #


def _subset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Rows for one dataset, matched case-insensitively.

    The wide table spells it ``3Oct-Control-24-0.1``; people type
    ``3OCT-Control-24-0.1``. An exact match silently returns nothing, which
    looks like "this cohort has no flies" rather than a typo.
    """
    want = str(dataset).strip().casefold()
    names = df["dataset"].astype(str).str.strip()
    return df[names.str.casefold() == want].copy()


def resolve_dataset(df: pd.DataFrame, dataset: str) -> str:
    """The dataset's own spelling, so titles match the wide table."""
    hit = _subset(df, dataset)
    if hit.empty:
        return str(dataset).strip()
    return str(hit["dataset"].astype(str).str.strip().iloc[0])


def _with_trial_number(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_trial"] = [
        (p[0] if (p := parse_trial(v)) is not None else np.nan)
        for v in df["trial_label"]
    ]
    return df.dropna(subset=["_trial"])


def build_fly_table(
    training_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    *,
    dataset: str,
    odor_token: Optional[str] = None,
    testing_auc_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per fly: conditioning AUC (per trial and pooled) plus its test.

    Returns an empty frame when either side has no rows for ``dataset``. The
    merge is an inner join, so a fly with conditioning but no scored test
    trials is dropped rather than carried as NaN — which is what makes the
    emitted ``n`` the fly count the figures actually plot.
    """
    train = _subset(training_df, dataset)
    test = _subset(testing_df, dataset)
    if train.empty or test.empty:
        return pd.DataFrame()

    token = str(odor_token).strip().lower() if odor_token else training_odor_token(train)
    presentations = presentation_trials(test, token)
    if not presentations:
        LOGGER.warning("%s: odor %r never presented at test", dataset, token)
        return pd.DataFrame()

    train = _with_trial_number(train)
    train["AUC-During"] = pd.to_numeric(train["AUC-During"], errors="coerce")
    training_trials = sorted(int(t) for t in train["_trial"].unique())

    # One AUC per (fly, conditioning trial).
    per_trial = (
        train.groupby(["fly", "fly_number", "_trial"])["AUC-During"].mean().unstack("_trial")
    )
    per_trial.columns = [f"auc_t{int(c)}" for c in per_trial.columns]
    summary = per_trial.reset_index()

    auc_cols = [f"auc_t{t}" for t in training_trials]
    summary["mean_auc"] = summary[auc_cols].mean(axis=1)
    summary["auc_slope"] = [
        auc_slope(pd.Series(row.to_numpy(dtype=float), index=training_trials))
        for _, row in summary[auc_cols].iterrows()
    ]

    # Testing side: score and PER for each presentation of the odor.
    test = _with_trial_number(test)
    for trial in presentations:
        sub = test[test["_trial"] == trial]
        scores = sub.groupby(["fly", "fly_number"])["score"].mean().rename(f"score_{trial}")
        pers = (
            sub.groupby(["fly", "fly_number"])["prediction"].max().rename(f"per_{trial}")
        )
        summary = summary.merge(
            pd.concat([scores, pers], axis=1).reset_index(),
            on=["fly", "fly_number"],
            how="inner",
        )
    if summary.empty:
        return summary

    score_cols = [f"score_{t}" for t in presentations]
    per_cols = [f"per_{t}" for t in presentations]
    summary["mean_score"] = summary[score_cols].mean(axis=1)
    summary["per_any"] = summary[per_cols].max(axis=1)

    # Optional: the AUC the fly produced at test, from the full wide table.
    if testing_auc_df is not None:
        t_auc = _with_trial_number(_subset(testing_auc_df, dataset))
        if not t_auc.empty:
            t_auc["AUC-During"] = pd.to_numeric(t_auc["AUC-During"], errors="coerce")
            for trial in presentations:
                sub = t_auc[t_auc["_trial"] == trial]
                col = (
                    sub.groupby(["fly", "fly_number"])["AUC-During"]
                    .mean()
                    .rename(f"test_auc_{trial}")
                    .reset_index()
                )
                # Left join: a missing testing envelope must not drop the fly
                # from the score/PER analysis, only from the AUC column.
                summary = summary.merge(col, on=["fly", "fly_number"], how="left")
            summary["mean_test_auc"] = summary[
                [f"test_auc_{t}" for t in presentations]
            ].mean(axis=1)

    summary = summary.sort_values(["fly", "fly_number"]).reset_index(drop=True)
    summary.attrs["dataset"] = dataset
    summary.attrs["odor_token"] = token
    summary.attrs["presentations"] = presentations
    summary.attrs["training_trials"] = training_trials
    summary.attrs["odor_short"] = odor_short(token)
    return summary


def presentation_labels(table: pd.DataFrame) -> list[str]:
    """``["Hex 1", "Hex 2"]`` — the odor short name plus presentation index."""
    short = table.attrs.get("odor_short", "Odor")
    return [f"{short} {i + 1}" for i in range(len(table.attrs["presentations"]))]


def pooled_per_label(table: pd.DataFrame) -> str:
    short = table.attrs.get("odor_short", "Odor")
    idx = range(1, len(table.attrs["presentations"]) + 1)
    return f"{short} " + " or ".join(str(i) for i in idx)


def pooled_score_label(table: pd.DataFrame) -> str:
    short = table.attrs.get("odor_short", "Odor")
    idx = range(1, len(table.attrs["presentations"]) + 1)
    return f"{short} " + " & ".join(str(i) for i in idx) + " mean"


# --------------------------------------------------------------------------- #
# Logistic fit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LogisticFit:
    """P(react) as a function of training AUC."""

    n: int
    intercept: float
    slope: float
    p_lr: float
    odds_ratio_100: float
    converged: bool
    x_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    p_grid: np.ndarray = field(default_factory=lambda: np.array([]))


_EMPTY_FIT = LogisticFit(
    n=0, intercept=float("nan"), slope=float("nan"), p_lr=float("nan"),
    odds_ratio_100=float("nan"), converged=False,
)


def _penalised_logit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Newton fit with a whisper of ridge, for the separated small-n case.

    Only used when statsmodels refuses (perfect separation is common at n<20):
    the penalty keeps the coefficient finite so the curve can still be drawn,
    and the p-value is reported as NaN rather than from this fit.
    """
    X = np.column_stack([np.ones_like(x), x])
    scale = float(np.std(x)) or 1.0
    beta = np.zeros(2)
    ridge = np.diag([0.0, 1e-6 / (scale ** 2)])
    for _ in range(200):
        eta = np.clip(X @ beta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        grad = X.T @ (y - mu) - ridge @ beta
        hess = X.T @ (X * w[:, None]) + ridge
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def fit_logistic(x: Sequence[float], y: Sequence[float], *, n_grid: int = 200) -> LogisticFit:
    """Logistic regression of a 0/1 outcome on training AUC.

    ``p_lr`` is the likelihood-ratio test for the slope. With a degenerate
    outcome (every fly reacted, or none did) there is nothing to fit and the
    grids come back empty so callers draw no curve.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.unique(y).size < 2:
        return LogisticFit(
            n=int(x.size), intercept=float("nan"), slope=float("nan"),
            p_lr=float("nan"), odds_ratio_100=float("nan"), converged=False,
        )

    intercept = slope = float("nan")
    p_lr = float("nan")
    converged = False
    try:
        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
        intercept, slope = float(res.params[0]), float(res.params[1])
        p_lr = float(res.llr_pvalue)
        converged = bool(res.mle_retvals.get("converged", False))
    except Exception as exc:  # perfect separation, singular design, no statsmodels
        LOGGER.debug("statsmodels logit failed (%s); falling back to ridge", exc)
    if not np.isfinite(slope):
        intercept, slope = _penalised_logit(x, y)
        p_lr = float("nan")
        converged = False

    grid = np.linspace(float(np.min(x)), float(np.max(x)), n_grid)
    probs = 1.0 / (1.0 + np.exp(-np.clip(intercept + slope * grid, -35, 35)))
    return LogisticFit(
        n=int(x.size),
        intercept=intercept,
        slope=slope,
        p_lr=p_lr,
        odds_ratio_100=float(np.exp(np.clip(slope * 100.0, -700, 700))),
        converged=converged,
        x_grid=grid,
        p_grid=probs,
    )


# --------------------------------------------------------------------------- #
# Equal-count bins
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BinRate:
    """Observed response rate in one equal-count AUC bin."""

    index: int
    n: int
    k: int
    rate: float
    lo: float
    hi: float
    x_center: float
    x_lo: float
    x_hi: float


@dataclass(frozen=True)
class BinMean:
    """Mean outcome in one equal-count AUC bin."""

    index: int
    n: int
    mean: float
    sem: float
    x_center: float
    x_lo: float
    x_hi: float


def equal_count_bins(values: Sequence[float], n_bins: int) -> np.ndarray:
    """Split ``values`` into ``n_bins`` groups of (near-)equal size.

    Equal-count rather than equal-width: training AUC is heavily
    right-skewed, so equal-width bins put 15 of 18 flies in one box.
    NaNs get bin ``-1`` and are excluded downstream.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.size, -1, dtype=int)
    finite = np.flatnonzero(np.isfinite(v))
    if finite.size == 0 or n_bins < 1:
        return out
    order = finite[np.argsort(v[finite], kind="stable")]
    ranks = np.arange(order.size)
    out[order] = np.minimum((ranks * n_bins) // order.size, n_bins - 1)
    return out


def _bin_extent(v: np.ndarray) -> tuple[float, float, float]:
    return float(np.median(v)), float(np.min(v)), float(np.max(v))


def rate_by_bin(
    values: Sequence[float], outcome: Sequence[float], n_bins: int = DEFAULT_BINS
) -> list[BinRate]:
    """Observed 0/1 rate per equal-count bin, with Wilson intervals."""
    v = np.asarray(values, dtype=float)
    y = np.asarray(outcome, dtype=float)
    bins = equal_count_bins(v, n_bins)
    out: list[BinRate] = []
    for b in range(n_bins):
        sel = (bins == b) & np.isfinite(y)
        n = int(sel.sum())
        if n == 0:
            continue
        k = int(np.nansum(y[sel] >= 0.5))
        lo, hi = wilson_ci(k, n)
        centre, x_lo, x_hi = _bin_extent(v[sel])
        out.append(BinRate(b, n, k, k / n, lo, hi, centre, x_lo, x_hi))
    return out


def mean_by_bin(
    values: Sequence[float], outcome: Sequence[float], n_bins: int = DEFAULT_BINS
) -> list[BinMean]:
    """Mean outcome (and SEM) per equal-count bin."""
    v = np.asarray(values, dtype=float)
    y = np.asarray(outcome, dtype=float)
    bins = equal_count_bins(v, n_bins)
    out: list[BinMean] = []
    for b in range(n_bins):
        sel = (bins == b) & np.isfinite(y)
        n = int(sel.sum())
        if n == 0:
            continue
        vals = y[sel]
        sem = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        centre, x_lo, x_hi = _bin_extent(v[sel])
        out.append(BinMean(b, n, float(vals.mean()), sem, centre, x_lo, x_hi))
    return out


# --------------------------------------------------------------------------- #
# Per-trial correlation matrix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrialMatrix:
    """Spearman rho / raw p / BH-adjusted p for every trial x outcome cell."""

    rho: pd.DataFrame
    p_raw: pd.DataFrame
    p_bh: pd.DataFrame
    n: pd.DataFrame


def _outcome_columns(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    """``(presentation label, outcome kind, dataframe column)`` in draw order."""
    cols: list[tuple[str, str, str]] = []
    for label, trial in zip(presentation_labels(table), table.attrs["presentations"]):
        if f"test_auc_{trial}" in table.columns:
            cols.append((label, "AUC", f"test_auc_{trial}"))
        cols.append((label, "Score", f"score_{trial}"))
        cols.append((label, "PER", f"per_{trial}"))
    return cols


def _predictor_rows(table: pd.DataFrame) -> list[tuple[str, str]]:
    rows = [(f"Train {t}", f"auc_t{t}") for t in table.attrs["training_trials"]]
    rows.append(("Mean", "mean_auc"))
    return rows


def trial_correlation_matrix(table: pd.DataFrame) -> TrialMatrix:
    """Spearman each conditioning trial's AUC against each testing outcome.

    Spearman throughout — for the binary PER column it is the rank-biserial
    correlation, which is the same ordering statistic Mann-Whitney tests, so
    the heatmap and the group panels never disagree about direction.

    BH rather than Holm over the grid: this is an exploratory scan of every
    trial, and Holm at 30-42 tests would leave nothing visible even where the
    pooled effect is real.
    """
    rows = _predictor_rows(table)
    cols = _outcome_columns(table)
    index = pd.Index([r[0] for r in rows], name="Conditioning trial")
    columns = pd.MultiIndex.from_tuples(
        [(c[0], c[1]) for c in cols], names=["Presentation", "Outcome"]
    )

    rho = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    praw = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    counts = pd.DataFrame(0, index=index, columns=columns, dtype=int)

    for r_label, r_col in rows:
        x = pd.to_numeric(table.get(r_col), errors="coerce").to_numpy(dtype=float)
        for c_label, c_kind, c_col in cols:
            y = pd.to_numeric(table.get(c_col), errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            counts.loc[r_label, (c_label, c_kind)] = int(ok.sum())
            if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
                continue
            r, p = spearmanr(x[ok], y[ok])
            rho.loc[r_label, (c_label, c_kind)] = float(r)
            praw.loc[r_label, (c_label, c_kind)] = float(p)

    flat = praw.to_numpy(dtype=float).ravel()
    pbh = pd.DataFrame(
        np.asarray(bh_adjust(flat), dtype=float).reshape(praw.shape),
        index=index, columns=columns,
    )
    return TrialMatrix(rho=rho, p_raw=praw, p_bh=pbh, n=counts)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #


def _style_axes(ax, *, grid_axis: str = "y") -> None:
    """Hairline solid grid, two spines, ticks pointing out. Never dashed."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(direction="out", length=3, colors=MUTED)


def _annotate(ax, text: str, *, loc: str = "upper left") -> None:
    """Stat readout in ink, no box — the panel already has a frame."""
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.98, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va,
        fontsize=8, color=INK_SECONDARY, linespacing=1.45, zorder=6,
    )


def _p_text(p: float, adjusted: Optional[float] = None) -> str:
    if not np.isfinite(p):
        return "p n/a"
    body = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    if adjusted is not None and np.isfinite(adjusted):
        body += (
            "  (Holm < 0.001)" if adjusted < 0.001 else f"  (Holm {adjusted:.3f})"
        )
    return body


def _jitter(n: int, seed: int, width: float = 0.16) -> np.ndarray:
    """Deterministic horizontal jitter — the same figure every re-run."""
    return np.random.default_rng(seed).uniform(-width, width, n)


def _dot_group(ax, pos: float, values: np.ndarray, color: str, seed: int) -> None:
    """Every fly as a dot, with a median bar and an IQR spine behind it.

    A dot plot, not a boxplot: at n<10 a box draws quartiles from five points
    and reads as more certainty than there is.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.vlines(pos, q1, q3, color=color, linewidth=2.0, alpha=0.30, zorder=2)
    ax.hlines(med, pos - 0.28, pos + 0.28, color=color, linewidth=2.0, zorder=4)
    ax.scatter(
        pos + _jitter(values.size, seed), values, s=34, facecolor=color,
        edgecolor=SURFACE, linewidth=1.0, alpha=0.95, zorder=3,
    )


def _spearman_panel(ax, x: np.ndarray, y: np.ndarray, *, color: str) -> tuple[float, float, int]:
    """Scatter with a least-squares guide line; returns (rho, p, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    rho = p = float("nan")
    if n >= 3 and np.unique(x[ok]).size > 1 and np.unique(y[ok]).size > 1:
        rho, p = spearmanr(x[ok], y[ok])
    if n >= 2 and np.unique(x[ok]).size > 1:
        fit = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
        ax.plot(xs, np.polyval(fit, xs), color=FIT_INK, linewidth=1.2, zorder=2)
    ax.scatter(
        x[ok], y[ok], s=42, facecolor=color, edgecolor=SURFACE,
        linewidth=1.0, alpha=0.95, zorder=3,
    )
    return float(rho), float(p), n


def _score_axis(ax) -> None:
    ax.set_ylim(SCORE_MIN - 0.6, SCORE_MAX + 0.6)
    ax.set_yticks(list(range(SCORE_MIN, SCORE_MAX + 1)))
    ax.axhline(
        REACTION_BOUNDARY, color=PER_COLOR, linewidth=0.9, alpha=0.55, zorder=1
    )


def _auc_label(table: pd.DataFrame, col: str = "mean_auc") -> str:
    if col == "mean_auc":
        n = len(table.attrs.get("training_trials", []))
        return f"Training AUC-During\n(mean of {n} odor-only trials)"
    return "Training AUC-During"


def _suptitle(fig, title: str, subtitle: str) -> float:
    """Title block measured in inches, so it survives short figures.

    Figure-fraction offsets collapse on a 4-inch-tall panel row and the
    subtitle lands on top of the title; these are converted from inches.
    Returns the ``rect`` top for ``tight_layout``.
    """
    height = float(fig.get_figheight())
    fig.suptitle(
        title, fontsize=13, weight="bold", color=INK, y=1 - 0.30 / height
    )
    fig.text(
        0.5, 1 - 0.60 / height, subtitle, ha="center", va="top",
        fontsize=8.5, color=MUTED,
    )
    return 1 - 0.92 / height


def _relative_luminance(rgb: Sequence[float]) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(float(c)) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def stars(p: float) -> str:
    """``***`` / ``**`` / ``*`` / ``n.s.`` — the usual thresholds, strict.

    Strict ``<`` at every boundary: p = 0.05 is ``n.s.``, never rounded up
    into a star.
    """
    if p is None or not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def is_significant(p: float) -> bool:
    return bool(p is not None and np.isfinite(p) and p < 0.05)


def bin_colors(n_bins: int) -> list[str]:
    """Ordinal ramp steps for ``n_bins`` bins, always ending at the dark step."""
    if n_bins <= 1:
        return [BIN_RAMP[1]]
    if n_bins == 2:
        return [BIN_RAMP[0], BIN_RAMP[2]]
    if n_bins == 3:
        return list(BIN_RAMP)
    # Past three, interpolate within the same hue rather than adding one.
    ramp = LinearSegmentedColormap.from_list("bins", list(BIN_RAMP), N=256)
    return [
        matplotlib.colors.to_hex(ramp(i / (n_bins - 1))) for i in range(n_bins)
    ]


def bin_names(n_bins: int) -> tuple[str, ...]:
    return BIN_NAMES.get(n_bins, tuple(f"bin {i + 1}" for i in range(n_bins)))


def _ink_on(rgb: Sequence[float]) -> str:
    """Whichever of light/dark ink has more contrast against ``rgb``.

    Eyeballing this put white numerals on mid-red cells at rho ~= 0.65, where
    the ramp is nowhere near dark enough to carry them.
    """
    lum = _relative_luminance(rgb)
    return SURFACE if lum < 0.179 else INK


def _sig_bracket(ax, x1: float, x2: float, y: float, label: str, *, sig: bool,
                 drop: float = 0.0) -> None:
    """Comparison bracket with the star (or ``n.s.``) sitting on top of it."""
    colour = INK if sig else MUTED
    ax.plot(
        [x1, x1, x2, x2], [y - drop, y, y, y - drop],
        color=colour, linewidth=1.1, clip_on=False, zorder=6,
    )
    ax.text(
        (x1 + x2) / 2, y, f" {label}", ha="center", va="bottom",
        fontsize=11 if sig else 8.5, color=colour,
        weight="bold" if sig else "normal", clip_on=False, zorder=6,
    )


def _verdict(ax, text: str, *, sig: bool) -> None:
    """One plain-language line at the top of a panel saying what it means."""
    ax.text(
        0.5, 1.015, text, transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8.5, color=INK if sig else MUTED,
        weight="bold" if sig else "normal",
    )


def asym_err(
    values: Sequence[float], lows: Sequence[float], highs: Sequence[float],
    *, scale: float = 1.0,
) -> np.ndarray:
    """``[[below], [above]]`` error offsets, clamped at zero.

    At k = n the Wilson upper bound lands on 0.9999999999999998 while the
    point estimate is exactly 1.0, so the raw subtraction is -2e-16 and
    matplotlib refuses the whole errorbar with "yerr must not contain
    negative values". Clamp rather than special-case k == n: the same thing
    happens at the lower bound when a proportion is exactly 0.
    """
    v = np.asarray(values, dtype=float)
    lo = np.asarray(lows, dtype=float)
    hi = np.asarray(highs, dtype=float)
    return scale * np.vstack([
        np.clip(v - lo, 0.0, None), np.clip(hi - v, 0.0, None)
    ])


def bar_label_position(value: float, *, pad: float = 0.035) -> tuple[float, str]:
    """Where a bar's label goes: outside the bar end, on the far side of zero.

    A fixed ``value + pad`` puts the label *inside* a downward bar and clips it
    against the axis floor, which is what happened to the ``n.s.`` marks on the
    negative-rho trials.
    """
    if value < 0:
        return value - pad, "top"
    return value + pad, "bottom"


def _bar_ends(ax, xs, heights, colours, *, width: float = 0.62) -> None:
    """Thin bars with a surface gap between neighbours, anchored to zero."""
    ax.bar(
        xs, heights, width=width, color=colours, edgecolor=SURFACE,
        linewidth=1.6, zorder=3,
    )


# --------------------------------------------------------------------------- #
# Figure A — the plain-reading overview: bars + stars
# --------------------------------------------------------------------------- #


def figure_simple_overview(
    table: pd.DataFrame, *, dataset: str, odor_label: str,
    n_bins: int = DEFAULT_BINS,
) -> tuple[plt.Figure, list[dict]]:
    """Split flies by training vigor, then show what they did at test.

    The headline figure, built to be read without knowing what a Spearman
    correlation is: sort the flies into equal-sized low/medium/high training-AUC
    groups, then plot two bars per group — how many reacted, and how strongly.
    Significance is the low-vs-high comparison, marked with a star.
    """
    from scripts.analysis.corrected_stats import binary_test_battery

    presentations = list(table.attrs["presentations"])
    labels = presentation_labels(table)
    auc = table["mean_auc"].to_numpy(dtype=float)
    bins = equal_count_bins(auc, n_bins)
    colours = bin_colors(n_bins)
    names = bin_names(n_bins)
    order = [b for b in range(n_bins) if (bins == b).any()]
    stats: list[dict] = []

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            2, len(labels), figsize=(4.0 * len(labels), 7.6), squeeze=False
        )

        for i, (label, trial) in enumerate(zip(labels, presentations)):
            per = pd.to_numeric(table[f"per_{trial}"], errors="coerce").to_numpy(float) >= 0.5
            score = pd.to_numeric(table[f"score_{trial}"], errors="coerce").to_numpy(float)

            # ---- Row 1: how many flies reacted ------------------------- #
            ax = axes[0][i]
            rates, ticks = [], []
            for pos, b in enumerate(order):
                sel = bins == b
                n, k = int(sel.sum()), int(per[sel].sum())
                lo, hi = wilson_ci(k, n)
                rates.append({"bin": b, "name": names[b], "n": n, "k": k,
                              "rate": k / n, "wilson_lo": lo, "wilson_hi": hi,
                              "auc_median": float(np.median(auc[sel])),
                              "auc_range": [float(auc[sel].min()), float(auc[sel].max())]})
                ticks.append(
                    f"{names[b]}\n{auc[sel].min():.0f}–{auc[sel].max():.0f}\nn = {n}"
                )
            _bar_ends(ax, range(len(order)), [100 * r["rate"] for r in rates],
                      [colours[b] for b in order])
            ax.errorbar(
                range(len(order)), [100 * r["rate"] for r in rates],
                yerr=asym_err(
                    [r["rate"] for r in rates], [r["wilson_lo"] for r in rates],
                    [r["wilson_hi"] for r in rates], scale=100.0,
                ),
                fmt="none", ecolor=INK_SECONDARY, elinewidth=1.1, capsize=4, zorder=4,
            )
            for pos, r in enumerate(rates):
                # Inside the bar when it is tall enough to hold the label,
                # otherwise just above it -- white-on-white is not a label.
                inside = r["rate"] >= 0.12
                ax.text(
                    pos, 100 * r["rate"] + (-2.5 if inside else 2.0),
                    f"{r['k']}/{r['n']}", ha="center",
                    va="top" if inside else "bottom", fontsize=8,
                    color=SURFACE if inside else INK_SECONDARY,
                    weight="bold", zorder=5,
                )

            first, last = rates[0], rates[-1]
            bt = binary_test_battery(last["k"], last["n"], first["k"], first["n"])
            # Headline test is the trend over ALL flies, not the end bins.
            # Binning into thirds throws away most of the power: the same
            # cohort that gives Spearman p = 0.016 gives Fisher p = 0.24 on
            # 6-vs-6, and reporting the latter would call a real effect dead.
            rho_per, p_per = spearmanr(auc, per.astype(float))
            sig = is_significant(p_per)
            ax.set_ylim(0, 128)
            ax.set_yticks([0, 25, 50, 75, 100])
            _sig_bracket(ax, 0, len(order) - 1, 108,
                         f"{stars(p_per)}   p = {p_per:.3f}   (all {len(table)} flies)",
                         sig=sig, drop=4)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(ticks)
            ax.tick_params(axis="x", length=0, labelcolor=INK_SECONDARY)
            _style_axes(ax)
            ax.set_title(label, fontsize=11, color=INK, weight="bold", pad=26)
            _verdict(
                ax,
                ("Higher training extension → more flies react"
                 if sig else "No clear difference in how many react"),
                sig=sig,
            )
            if i == 0:
                ax.set_ylabel("Flies that reacted at test (%)")

            # ---- Row 2: how strongly they reacted ---------------------- #
            ax = axes[1][i]
            means = []
            for pos, b in enumerate(order):
                sel = (bins == b) & np.isfinite(score)
                vals = score[sel]
                sem = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
                means.append({"bin": b, "name": names[b], "n": int(vals.size),
                              "mean_score": float(vals.mean()), "sem": sem})
                ax.scatter(
                    pos + _jitter(int(vals.size), 40 + i, 0.15), vals, s=24,
                    facecolor=INK_SECONDARY, edgecolor=SURFACE, linewidth=0.8,
                    alpha=0.55, zorder=5,
                )
            _bar_ends(ax, range(len(order)), [m["mean_score"] for m in means],
                      [colours[b] for b in order])
            ax.errorbar(
                range(len(order)), [m["mean_score"] for m in means],
                yerr=[m["sem"] for m in means], fmt="none", ecolor=INK_SECONDARY,
                elinewidth=1.1, capsize=4, zorder=4,
            )
            lo_vals = score[(bins == order[0]) & np.isfinite(score)]
            hi_vals = score[(bins == order[-1]) & np.isfinite(score)]
            st = score_test(hi_vals, lo_vals, n_boot=2000)
            rho, p_rho = spearmanr(auc, score)
            sig_s = is_significant(p_rho)
            ax.axhline(REACTION_BOUNDARY, color=PER_COLOR, linewidth=0.9, alpha=0.55,
                       zorder=1)
            ax.set_ylim(SCORE_MIN - 0.6, SCORE_MAX + 1.6)
            ax.set_yticks(list(range(SCORE_MIN, SCORE_MAX + 1)))
            _sig_bracket(ax, 0, len(order) - 1, SCORE_MAX + 0.35,
                         f"{stars(p_rho)}   p = {p_rho:.3f}   (all {len(table)} flies)",
                         sig=sig_s, drop=0.25)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(ticks)
            ax.tick_params(axis="x", length=0, labelcolor=INK_SECONDARY)
            _style_axes(ax)
            _verdict(
                ax,
                ("Higher training extension → stronger reaction"
                 if sig_s else "No clear difference in reaction strength"),
                sig=sig_s,
            )
            ax.set_xlabel("Training extension group\n(AUC-During, mean of all trials)")
            if i == 0:
                ax.set_ylabel("Reaction score at test (−1 … 5)")

            stats.append({
                "presentation": label,
                "trial": int(trial),
                "n_flies": int(len(table)),
                "n_bins": n_bins,
                "per_bins": rates,
                "score_bins": means,
                # Headline: the full-sample trend that the panels are annotated with.
                "per_trend_all_flies": {
                    "spearman_rho": float(rho_per), "p": float(p_per),
                    "stars": stars(float(p_per)), "significant": sig,
                },
                "score_trend_all_flies": {
                    "spearman_rho": float(rho), "p": float(p_rho),
                    "stars": stars(float(p_rho)), "significant": sig_s,
                },
                # Secondary: the end-bin contrast the bars actually draw. Kept
                # for reference; underpowered at n/3 per bin, so never the
                # headline.
                "per_low_vs_high": {
                    "k_high": bt.k1, "n_high": bt.n1, "k_low": bt.k2, "n_low": bt.n2,
                    "p_fisher": bt.p_fisher, "odds_ratio": bt.odds_ratio,
                    "risk_difference": bt.risk_difference,
                    "risk_difference_ci": list(bt.risk_difference_ci),
                    "stars": stars(bt.p_fisher),
                    "significant": is_significant(bt.p_fisher),
                },
                "score_low_vs_high": {
                    "median_high": st.median1, "median_low": st.median2,
                    "p": st.p_value, "cliffs_delta": st.delta,
                    "cliffs_delta_ci": list(st.delta_ci),
                    "stars": stars(st.p_value),
                    "significant": is_significant(st.p_value),
                },
                "spearman_rho": float(rho),
                "spearman_p": float(p_rho),
                "spearman_stars": stars(float(p_rho)),
            })

        handles = [
            plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                       markerfacecolor=colours[b], markeredgecolor=SURFACE,
                       label=f"{names[b]} training extension")
            for b in order
        ] + [
            plt.Line2D([], [], color=INK_SECONDARY, linewidth=1.1,
                       label="95% CI (top) / SEM (bottom)"),
            plt.Line2D([], [], color=PER_COLOR, linewidth=0.9, alpha=0.55,
                       label="score ≥ 2 counts as a reaction"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=len(handles), frameon=False,
            bbox_to_anchor=(0.5, -0.004), fontsize=8, labelcolor=INK_SECONDARY,
        )
        top = _suptitle(
            fig,
            "Do flies that extend more during training react more at test?",
            f"{dataset} {cohort_noun(dataset)} cohort, n = {len(table)} flies, {odor_label} throughout.  "
            f"Bars sort the flies into {len(order)} equal-sized groups by training "
            "extension; the bracket tests the trend over every fly individually "
            "(Spearman), not the end bars.\n"
            "* p < 0.05    ** p < 0.01    *** p < 0.001    n.s. = not significant.",
        )
        fig.tight_layout(rect=(0, 0.05, 1, top - 0.015))
    return fig, stats


# --------------------------------------------------------------------------- #
# Figure B — which single conditioning trial predicts this test?
# --------------------------------------------------------------------------- #


def figure_trial_predictors(
    table: pd.DataFrame, *, presentation_index: int, dataset: str, odor_label: str
) -> tuple[plt.Figure, dict]:
    """One bar per conditioning trial: how well does *that* trial predict?

    Same numbers as the trial matrix, one presentation at a time and one bar
    per trial, so "trial 4 predicts, trial 3 does not" is readable at a glance.
    """
    mat = trial_correlation_matrix(table)
    labels = presentation_labels(table)
    label = labels[presentation_index]
    trial = int(table.attrs["presentations"][presentation_index])
    trials = list(table.attrs["training_trials"])
    rows = [f"Train {t}" for t in trials] + ["Mean"]
    bar_labels = [str(t) for t in trials] + ["Mean"]

    kinds = [k for k in ("PER", "Score", "AUC") if (label, k) in mat.rho.columns]
    titles = {
        "PER": "Predicts WHETHER the fly reacts",
        "Score": "Predicts HOW STRONGLY it reacts",
        "AUC": "Predicts the size of the test response",
    }
    panels: list[dict] = []

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1, len(kinds), figsize=(3.9 * len(kinds), 4.9), squeeze=False, sharey=True
        )
        for i, kind in enumerate(kinds):
            ax = axes[0][i]
            rhos = [float(mat.rho.loc[r, (label, kind)]) for r in rows]
            ps = [float(mat.p_raw.loc[r, (label, kind)]) for r in rows]
            # Trials in slot-1 blue; the pooled "Mean" bar takes the dark step
            # of the same hue, so it reads as a summary, not a ninth trial.
            colours = [BIN_RAMP[1]] * len(trials) + [BIN_RAMP[2]]
            xs = list(range(len(trials))) + [len(trials) + 0.55]
            _bar_ends(ax, xs, rhos, colours)

            for x, value, p in zip(xs, rhos, ps):
                if not np.isfinite(value):
                    continue
                mark = stars(p)
                sig = is_significant(p)
                y, va = bar_label_position(value)
                ax.text(
                    x, y, mark, ha="center", va=va,
                    fontsize=12 if sig else 8, color=INK if sig else MUTED,
                    weight="bold" if sig else "normal", zorder=5,
                )
            ax.axhline(0, color=AXIS, linewidth=0.9, zorder=2)
            # Floor has to clear the label hanging under the lowest bar.
            ax.set_ylim(min(-0.14, float(np.nanmin(rhos)) - 0.16), 1.02)
            ax.set_xticks(xs)
            ax.set_xticklabels(bar_labels)
            ax.set_xlim(-0.7, xs[-1] + 0.7)
            ax.tick_params(axis="x", length=0, labelcolor=INK_SECONDARY)
            _style_axes(ax)
            ax.set_title(titles[kind], fontsize=10, color=INK, weight="bold", pad=24)
            n_sig = sum(1 for p in ps[:-1] if is_significant(p))
            _verdict(
                ax,
                (f"{n_sig} of {len(trials)} training trials predict this"
                 if n_sig else "No single training trial predicts this"),
                sig=bool(n_sig),
            )
            ax.set_xlabel("Training trial  (last bar = mean of all trials)")
            if i == 0:
                ax.set_ylabel(
                    "Prediction strength\n(Spearman ρ: 0 = no link, 1 = perfect)"
                )
            panels.append({
                "outcome": kind,
                "bars": [
                    {"label": bl, "row": r, "rho": rv, "p_raw": pv,
                     "p_bh": float(mat.p_bh.loc[r, (label, kind)]),
                     "stars": stars(pv), "significant": is_significant(pv)}
                    for bl, r, rv, pv in zip(bar_labels, rows, rhos, ps)
                ],
            })

        top = _suptitle(
            fig,
            f"Which training trial predicts the {label} test? — {dataset}",
            f"Each bar is one conditioning trial's proboscis extension (AUC-During) "
            f"against the {label} test outcome, n = {len(table)} {cohort_noun(dataset)} flies.\n"
            "Taller bar = better prediction.   "
            "* p < 0.05    ** p < 0.01    *** p < 0.001    n.s. = not significant.",
        )
        fig.tight_layout(rect=(0, 0, 1, top - 0.015))

    return fig, {
        "presentation": label,
        "trial": trial,
        "odor": odor_label,
        "n_flies": int(len(table)),
        "panels": panels,
    }


# --------------------------------------------------------------------------- #
# Figure 1 — binary PER
# --------------------------------------------------------------------------- #


def figure_per(
    table: pd.DataFrame, *, dataset: str, odor_label: str, n_bins: int = DEFAULT_BINS
) -> tuple[plt.Figure, list[dict]]:
    """Training AUC vs *whether* the fly reacted at test."""
    presentations = list(table.attrs["presentations"])
    labels = presentation_labels(table)
    columns = [(lab, f"per_{t}") for lab, t in zip(labels, presentations)]
    columns.append((pooled_per_label(table), "per_any"))

    auc = table["mean_auc"].to_numpy(dtype=float)
    raw_p: list[float] = []
    stats: list[dict] = []

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            2, len(columns), figsize=(3.45 * len(columns), 6.9), squeeze=False
        )

        for i, (label, col) in enumerate(columns):
            per = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
            reacted = per >= 0.5
            no_auc = auc[~reacted & np.isfinite(per)]
            yes_auc = auc[reacted]
            # Responders first, so Cliff's delta is positive when the flies that
            # reacted are the ones that extended more -- a negative delta beside
            # a visibly higher green group reads as a bug.
            test = score_test(yes_auc, no_auc, n_boot=4000)
            raw_p.append(test.p_value)

            # -- Row 1: AUC by outcome -------------------------------------- #
            ax = axes[0][i]
            _dot_group(ax, 0, no_auc, NO_PER_COLOR, seed=10 + i)
            _dot_group(ax, 1, yes_auc, PER_COLOR, seed=20 + i)
            ax.set_xlim(-0.6, 1.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [f"no PER\nn = {no_auc.size}", f"PER\nn = {yes_auc.size}"]
            )
            ax.tick_params(axis="x", length=0, labelcolor=INK_SECONDARY)
            _style_axes(ax)
            ax.set_title(label, fontsize=10, color=INK, weight="bold", pad=21)
            if i == 0:
                ax.set_ylabel("Training AUC-During (mean)")
            delta = f"δ = {test.delta:+.2f}" if np.isfinite(test.delta) else "δ n/a"
            _annotate(ax, f"Mann–Whitney {_p_text(test.p_value)}\nCliff's {delta}")
            # The pooled column can reproduce a single presentation exactly when
            # one responder set contains the other. Say so above the panel --
            # unstated, it reads as a duplicated figure.
            if i == len(columns) - 1:
                same = [
                    lab for lab, t in zip(labels, presentations)
                    if np.array_equal(
                        np.nan_to_num(pd.to_numeric(table[f"per_{t}"],
                                                    errors="coerce").to_numpy(float)) >= 0.5,
                        reacted,
                    )
                ]
                if same:
                    note = f"same flies as {same[0]} — every other responder also reacted here"
                    ax.text(
                        0.5, 1.02, note, transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=7.5, color=MUTED,
                    )

            # -- Row 2: P(react) vs AUC ------------------------------------- #
            ax = axes[1][i]
            fit = fit_logistic(auc, per)
            rates = rate_by_bin(auc, per, n_bins)
            if fit.p_grid.size:
                ax.plot(
                    fit.x_grid, fit.p_grid, color=FIT_INK, linewidth=1.6, zorder=3
                )
            ax.scatter(
                auc[np.isfinite(per)], per[np.isfinite(per)],
                s=26, facecolor=[PER_COLOR if v else NO_PER_COLOR
                                 for v in reacted[np.isfinite(per)]],
                edgecolor=SURFACE, linewidth=0.9, alpha=0.9, zorder=4,
            )
            if rates:
                xs = [r.x_center for r in rates]
                ys = [r.rate for r in rates]
                err = asym_err(ys, [r.lo for r in rates], [r.hi for r in rates])
                ax.errorbar(
                    xs, ys, yerr=err, fmt="s", markersize=6.5, color=INK,
                    ecolor=MUTED, elinewidth=1.0, capsize=3, zorder=5,
                    markerfacecolor=SURFACE, markeredgewidth=1.4,
                )
            ax.set_ylim(-0.09, 1.09)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0", ".25", ".5", ".75", "1"])
            _style_axes(ax)
            ax.set_xlabel(_auc_label(table))
            if i == 0:
                ax.set_ylabel("P(PER at test)")
            or_txt = (
                f"OR / 100 AUC = {fit.odds_ratio_100:.2f}"
                if np.isfinite(fit.odds_ratio_100) and fit.odds_ratio_100 < 1e6
                else "OR / 100 AUC = ∞ (separated)"
            )
            _annotate(
                ax, f"logistic {_p_text(fit.p_lr)}\n{or_txt}", loc="upper left"
            )

            stats.append({
                "column": label,
                "trial": int(presentations[i]) if i < len(presentations) else None,
                "n_flies": int(np.isfinite(per).sum()),
                "n_per": int(yes_auc.size),
                "n_no_per": int(no_auc.size),
                "per_rate": float(yes_auc.size / max(np.isfinite(per).sum(), 1)),
                "median_auc_per": test.median1,
                "median_auc_no_per": test.median2,
                "iqr_auc_per": list(test.iqr1),
                "iqr_auc_no_per": list(test.iqr2),
                "mannwhitney_u": test.u_statistic,
                "mannwhitney_p": test.p_value,
                "mannwhitney_exact": test.exact,
                "cliffs_delta": test.delta,
                "cliffs_delta_ci": list(test.delta_ci),
                "hodges_lehmann_shift": test.shift,
                "hodges_lehmann_ci": list(test.shift_ci),
                "logistic_slope": fit.slope,
                "logistic_p": fit.p_lr,
                "logistic_converged": fit.converged,
                "odds_ratio_per_100_auc": fit.odds_ratio_100,
                "auc_bins": [
                    {"bin": r.index, "n": r.n, "k": r.k, "rate": r.rate,
                     "wilson_lo": r.lo, "wilson_hi": r.hi,
                     "auc_median": r.x_center, "auc_range": [r.x_lo, r.x_hi]}
                    for r in rates
                ],
            })

        adjusted = holm_adjust(raw_p)
        for entry, adj in zip(stats, adjusted):
            entry["mannwhitney_p_holm"] = float(adj)
        for ax, entry in zip(axes[0], stats):
            txt = ax.texts[0]
            txt.set_text(
                f"Mann–Whitney {_p_text(entry['mannwhitney_p'], entry['mannwhitney_p_holm'])}"
                f"\nCliff's δ = {entry['cliffs_delta']:+.2f}  (PER > no PER)"
            )

        # Row 1 plots one variable in every column, so it gets one scale --
        # set explicitly rather than via sharey, which would also couple the
        # tick formatter. Headroom keeps the stat readout off the top points.
        finite = auc[np.isfinite(auc)]
        if finite.size:
            span = float(finite.max() - finite.min()) or 1.0
            for ax in axes[0]:
                ax.set_ylim(float(finite.min()) - 0.08 * span,
                            float(finite.max()) + 0.34 * span)

        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                       markerfacecolor=NO_PER_COLOR, markeredgecolor=SURFACE,
                       label="no PER at test"),
            plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                       markerfacecolor=PER_COLOR, markeredgecolor=SURFACE,
                       label="PER at test"),
            plt.Line2D([], [], marker="s", linestyle="", markersize=6,
                       markerfacecolor=SURFACE, markeredgecolor=INK,
                       markeredgewidth=1.4,
                       label=f"observed rate ± 95% Wilson ({n_bins} equal-count AUC bins)"),
            plt.Line2D([], [], color=FIT_INK, linewidth=1.6, label="logistic fit"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=4, frameon=False,
            bbox_to_anchor=(0.5, -0.005), fontsize=8, labelcolor=INK_SECONDARY,
        )
        top = _suptitle(
            fig,
            f"Conditioning vigor vs whether the fly reacts — {dataset}",
            f"{protocol_line(dataset, odor_label)}  Each dot is one fly "
            f"(n = {len(table)}); bar = median, spine = IQR.",
        )
        fig.tight_layout(rect=(0, 0.045, 1, top))
    return fig, stats


# --------------------------------------------------------------------------- #
# Figure 2 — score
# --------------------------------------------------------------------------- #


def figure_score(
    table: pd.DataFrame, *, dataset: str, odor_label: str, n_bins: int = DEFAULT_BINS
) -> tuple[plt.Figure, list[dict]]:
    """Training AUC vs *how strongly* the fly reacted at test."""
    presentations = list(table.attrs["presentations"])
    labels = presentation_labels(table)
    columns = [(lab, f"score_{t}") for lab, t in zip(labels, presentations)]
    columns.append((pooled_score_label(table), "mean_score"))

    auc = table["mean_auc"].to_numpy(dtype=float)
    raw_p: list[float] = []
    stats: list[dict] = []

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            2, len(columns), figsize=(3.45 * len(columns), 6.9), squeeze=False
        )

        for i, (label, col) in enumerate(columns):
            score = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)

            # -- Row 1: AUC vs score ---------------------------------------- #
            ax = axes[0][i]
            rho, p, n = _spearman_panel(ax, auc, score, color=POINT_COLOR)
            raw_p.append(p)
            _score_axis(ax)
            _style_axes(ax)
            ax.set_title(label, fontsize=10, color=INK, weight="bold", pad=8)
            ax.set_xlabel(_auc_label(table))
            if i == 0:
                ax.set_ylabel("Testing score (−1 … 5)")
            # Lower right: a rising fit leaves the top-left occupied, and at
            # score 5 the annotation sat on top of a fly.
            _annotate(ax, f"Spearman ρ = {rho:+.2f}\n{_p_text(p)}\nn = {n}",
                      loc="lower right")

            # -- Row 2: mean score by AUC bin ------------------------------- #
            ax = axes[1][i]
            bins = mean_by_bin(auc, score, n_bins)
            bin_index = equal_count_bins(auc, n_bins)
            for b in bins:
                sel = (bin_index == b.index) & np.isfinite(score)
                ax.scatter(
                    b.index + _jitter(int(sel.sum()), 30 + i, 0.13),
                    score[sel], s=26, facecolor=POINT_COLOR, edgecolor=SURFACE,
                    linewidth=0.9, alpha=0.45, zorder=2,
                )
            if bins:
                ax.errorbar(
                    [b.index for b in bins], [b.mean for b in bins],
                    yerr=[b.sem for b in bins], fmt="o-", color=INK,
                    ecolor=MUTED, elinewidth=1.0, capsize=3, linewidth=1.4,
                    markersize=6.5, markerfacecolor=SURFACE, markeredgewidth=1.4,
                    zorder=4,
                )
                ax.set_xticks([b.index for b in bins])
                ax.set_xticklabels([
                    f"{'low' if b.index == 0 else 'high' if b.index == len(bins) - 1 else 'mid'}"
                    f"\n{b.x_lo:.0f}–{b.x_hi:.0f}\nn = {b.n}"
                    for b in bins
                ])
                ax.set_xlim(-0.6, len(bins) - 0.4)
            ax.tick_params(axis="x", length=0, labelcolor=INK_SECONDARY)
            _score_axis(ax)
            _style_axes(ax)
            ax.set_xlabel("Training AUC-During bin (equal count)")
            if i == 0:
                ax.set_ylabel("Testing score (−1 … 5)")

            stats.append({
                "column": label,
                "trial": int(presentations[i]) if i < len(presentations) else None,
                "n": n,
                "spearman_rho": rho,
                "spearman_p": p,
                "auc_bins": [
                    {"bin": b.index, "n": b.n, "mean_score": b.mean, "sem": b.sem,
                     "auc_median": b.x_center, "auc_range": [b.x_lo, b.x_hi]}
                    for b in bins
                ],
            })

        adjusted = holm_adjust(raw_p)
        for entry, adj in zip(stats, adjusted):
            entry["spearman_p_holm"] = float(adj)
        for ax, entry in zip(axes[0], stats):
            ax.texts[0].set_text(
                f"Spearman ρ = {entry['spearman_rho']:+.2f}\n"
                f"{_p_text(entry['spearman_p'], entry['spearman_p_holm'])}\n"
                f"n = {entry['n']}"
            )
        # One score scale for the whole figure, plus room for the readout.
        for ax in list(axes[0]) + list(axes[1]):
            ax.set_ylim(SCORE_MIN - 1.15, SCORE_MAX + 0.6)

        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                       markerfacecolor=POINT_COLOR, markeredgecolor=SURFACE,
                       label="one fly"),
            plt.Line2D([], [], color=FIT_INK, linewidth=1.2, label="least-squares guide"),
            plt.Line2D([], [], marker="o", linestyle="-", markersize=6, color=INK,
                       markerfacecolor=SURFACE, markeredgewidth=1.4,
                       label="bin mean ± SEM"),
            plt.Line2D([], [], color=PER_COLOR, linewidth=0.9, alpha=0.55,
                       label="reaction threshold (score ≥ 2)"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=4, frameon=False,
            bbox_to_anchor=(0.5, -0.005), fontsize=8, labelcolor=INK_SECONDARY,
        )
        top = _suptitle(
            fig,
            f"Conditioning vigor vs how strongly the fly reacts — {dataset}",
            f"{protocol_line(dataset, odor_label)}  Each dot is one fly "
            f"(n = {len(table)}).",
        )
        fig.tight_layout(rect=(0, 0.045, 1, top))
    return fig, stats


# --------------------------------------------------------------------------- #
# Figure 3 — per-trial correlation matrix
# --------------------------------------------------------------------------- #


def _rho_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "rho_diverging", [RHO_NEG, RHO_MID, RHO_POS], N=256
    )


def figure_trial_matrix(
    table: pd.DataFrame, *, dataset: str, odor_label: str
) -> tuple[plt.Figure, dict]:
    """Does one conditioning trial carry the signal, or only the mean of them?"""
    mat = trial_correlation_matrix(table)
    rho = mat.rho
    cmap = _rho_cmap()
    norm = Normalize(vmin=-1, vmax=1)

    n_rows, n_cols = rho.shape
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(0.95 * n_cols + 3.0, 0.80 * n_rows + 2.6))
        ax.imshow(
            rho.to_numpy(dtype=float), cmap=cmap, norm=norm,
            aspect="auto", interpolation="nearest",
        )

        for r in range(n_rows):
            for c in range(n_cols):
                value = rho.iat[r, c]
                if not np.isfinite(value):
                    ax.text(c, r, "–", ha="center", va="center",
                            fontsize=9, color=MUTED)
                    continue
                text_color = _ink_on(cmap(norm(value)))
                mark = ""
                if np.isfinite(mat.p_bh.iat[r, c]) and mat.p_bh.iat[r, c] < 0.05:
                    mark = "**"
                elif np.isfinite(mat.p_raw.iat[r, c]) and mat.p_raw.iat[r, c] < 0.05:
                    mark = "*"
                ax.text(
                    c, r, f"{value:+.2f}{mark}", ha="center", va="center",
                    fontsize=8.5, color=text_color,
                    weight="bold" if mark == "**" else "normal",
                )

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([c[1] for c in rho.columns], fontsize=8.5, color=INK)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(list(rho.index), fontsize=8.5, color=INK)
        ax.set_ylabel("Conditioning trial (AUC-During)", labelpad=8)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        # 2px surface gap between cells, rather than a border around each.
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2.0)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)

        # Presentation group labels above the outcome ticks, with a wider
        # surface gap marking where one presentation's block ends.
        seen: dict[str, list[int]] = {}
        for idx, (pres, _kind) in enumerate(rho.columns):
            seen.setdefault(pres, []).append(idx)
        for pres, idxs in seen.items():
            ax.text(
                float(np.mean(idxs)), -0.82, pres, ha="center", va="bottom",
                fontsize=9.5, weight="bold", color=INK,
            )
            if max(idxs) + 1 < n_cols:
                ax.axvline(max(idxs) + 0.5, color=SURFACE, linewidth=6, zorder=5)

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
            fraction=0.032, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1],
        )
        cbar.set_label("Spearman ρ", color=INK_SECONDARY, fontsize=8.5)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=2, colors=MUTED, labelsize=8)

        top = _suptitle(
            fig,
            f"Which conditioning trial predicts the test response? — {dataset}",
            f"Spearman ρ, {len(table)} {cohort_noun(dataset)} flies.  "
            "* raw p < 0.05  ·  ** Benjamini-Hochberg q < 0.05 across the grid.  "
            "PER is binary, so its ρ is the rank-biserial correlation.",
        )
        fig.tight_layout(rect=(0, 0, 1, top - 0.03))

    cells = []
    for r_label in rho.index:
        for c_label in rho.columns:
            cells.append({
                "conditioning_trial": r_label,
                "presentation": c_label[0],
                "outcome": c_label[1],
                "spearman_rho": float(rho.loc[r_label, c_label]),
                "p_raw": float(mat.p_raw.loc[r_label, c_label]),
                "p_bh": float(mat.p_bh.loc[r_label, c_label]),
                "n": int(mat.n.loc[r_label, c_label]),
            })
    payload = {
        "n_flies": int(len(table)),
        "n_tests": int(rho.size),
        "multiplicity": "benjamini-hochberg over the whole grid",
        "cells": cells,
    }
    return fig, payload


# --------------------------------------------------------------------------- #
# Figure 4 — rho across conditioning trials
# --------------------------------------------------------------------------- #


def figure_trial_profile(
    table: pd.DataFrame, *, dataset: str, odor_label: str
) -> tuple[plt.Figure, list[dict]]:
    """Read the matrix as a trajectory: is the signal there from trial 1?"""
    mat = trial_correlation_matrix(table)
    trials = list(table.attrs["training_trials"])
    labels = presentation_labels(table)
    kinds = [k for k in ("AUC", "Score", "PER")
             if any(c[1] == k for c in mat.rho.columns)]
    # Categorical slots 1..2 — identity is the presentation, never its rank.
    series_colors = {labels[0]: "#2a78d6"}
    if len(labels) > 1:
        series_colors[labels[1]] = "#eb6834"
    for extra, hue in zip(labels[2:], ["#1baf7a", "#eda100"]):
        series_colors[extra] = hue

    rows = [f"Train {t}" for t in trials]
    payload: list[dict] = []
    # The pooled predictor gets its own x slot rather than a horizontal rule
    # averaged over presentations -- averaging two rho values is not a rho.
    mean_x = max(trials) + 1.4
    floor = float(np.nanmin(mat.rho.to_numpy(dtype=float)))
    y_lo = min(-0.25, floor - 0.15)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1, len(kinds), figsize=(3.9 * len(kinds), 4.3),
            squeeze=False, sharey=True,
        )
        for i, kind in enumerate(kinds):
            ax = axes[0][i]
            ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
            ax.axvline(mean_x - 0.7, color=GRID, linewidth=0.8, zorder=1)
            panel = {"outcome": kind, "series": {}, "p_raw": {}, "mean": {}}
            endpoints: list[tuple[str, float]] = []
            for label in labels:
                if (label, kind) not in mat.rho.columns:
                    continue
                vals = [float(mat.rho.loc[r, (label, kind)]) for r in rows]
                ps = [float(mat.p_raw.loc[r, (label, kind)]) for r in rows]
                panel["series"][label] = vals
                panel["p_raw"][label] = ps
                colour = series_colors[label]
                ax.plot(
                    trials, vals, marker="o", markersize=5.5, linewidth=1.6,
                    color=colour, markerfacecolor=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1.0,
                    label=label, zorder=3,
                )
                # Ring the trials that clear raw p < 0.05.
                sig = [(t, v) for t, v, pv in zip(trials, vals, ps)
                       if np.isfinite(pv) and pv < 0.05]
                if sig:
                    ax.scatter(
                        [s[0] for s in sig], [s[1] for s in sig], s=96,
                        facecolor="none", edgecolor=colour,
                        linewidth=1.4, zorder=4,
                    )
                mean_rho = float(mat.rho.loc["Mean", (label, kind)])
                mean_p = float(mat.p_raw.loc["Mean", (label, kind)])
                panel["mean"][label] = {"rho": mean_rho, "p_raw": mean_p}
                if np.isfinite(mean_rho):
                    ax.scatter(
                        [mean_x], [mean_rho], s=52, marker="D",
                        facecolor=colour, edgecolor=SURFACE, linewidth=1.0,
                        zorder=4,
                    )
                    if np.isfinite(mean_p) and mean_p < 0.05:
                        ax.scatter(
                            [mean_x], [mean_rho], s=130, marker="D",
                            facecolor="none", edgecolor=colour, linewidth=1.4,
                            zorder=4,
                        )
                    endpoints.append((label, mean_rho))

            # Direct labels, nudged apart when two series land on top of
            # each other -- otherwise "Hex 1" and "Hex 2" overprint.
            endpoints.sort(key=lambda e: e[1])
            placed: list[float] = []
            gap = 0.085 * (1.05 - y_lo)
            for name, value in endpoints:
                y = value
                for prior in placed:
                    if abs(y - prior) < gap:
                        y = prior + gap
                placed.append(y)
                ax.annotate(
                    name, (mean_x, y), textcoords="offset points",
                    # Clear of the significance ring (s=130 -> ~6.4pt radius).
                    xytext=(13, 0), fontsize=8, color=INK_SECONDARY, va="center",
                )

            ax.set_ylim(y_lo, 1.05)
            ax.set_xticks(list(trials) + [mean_x])
            ax.set_xticklabels([str(t) for t in trials] + ["mean"])
            ax.set_xlim(min(trials) - 0.4, mean_x + 1.5)
            _style_axes(ax)
            ax.tick_params(axis="x", labelcolor=INK_SECONDARY)
            ax.set_title(
                {"AUC": "Testing AUC-During", "Score": "Testing score (−1…5)",
                 "PER": "Testing PER (0/1)"}[kind],
                fontsize=10, color=INK, weight="bold", pad=8,
            )
            ax.set_xlabel("Conditioning trial")
            if i == 0:
                ax.set_ylabel("Spearman ρ  (trial AUC vs outcome)")
            payload.append(panel)

        handles = [
            plt.Line2D([], [], color=series_colors[l], marker="o", markersize=5.5,
                       markeredgecolor=SURFACE, linewidth=1.6, label=l)
            for l in labels
        ] + [
            plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                       markerfacecolor="none", markeredgecolor=MUTED,
                       markeredgewidth=1.4, label="raw p < 0.05"),
            plt.Line2D([], [], marker="D", linestyle="", markersize=6,
                       markerfacecolor=MUTED, markeredgecolor=SURFACE,
                       label="predictor = mean AUC of all conditioning trials"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=len(handles), frameon=False,
            bbox_to_anchor=(0.5, -0.01), fontsize=8, labelcolor=INK_SECONDARY,
        )
        top = _suptitle(
            fig,
            f"Does the prediction build across conditioning? — {dataset}",
            f"Spearman ρ between a single conditioning trial's AUC and each testing "
            f"outcome, {len(table)} {cohort_noun(dataset)} flies, {odor_label} throughout.",
        )
        fig.tight_layout(rect=(0, 0.075, 1, top))
    return fig, payload


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _load_inputs(args) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    from fbpipe.analysis.traces import read_wide_table
    from scripts.analysis.envelope_visuals import _normalise_fly_columns

    keep = ["dataset", "fly", "fly_number", "trial_type", "trial_label", "AUC-During"]

    training = read_wide_table(args.training_wide_csv)
    training = _normalise_fly_columns(training[[c for c in keep if c in training.columns]].copy())
    if "trial_type" in training.columns:
        training = training[training["trial_type"].astype(str).str.strip() == "training"]

    testing = pd.read_csv(args.predictions_csv)
    testing = _normalise_fly_columns(testing)
    if "trial_type" in testing.columns:
        testing = testing[testing["trial_type"].astype(str).str.strip() == "testing"]

    testing_auc = None
    if args.testing_wide_csv:
        wide = read_wide_table(args.testing_wide_csv)
        wide = _normalise_fly_columns(wide[[c for c in keep if c in wide.columns]].copy())
        if "trial_type" in wide.columns:
            wide = wide[wide["trial_type"].astype(str).str.strip() == "testing"]
        testing_auc = wide

    return training, testing, testing_auc


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, svg: bool = True) -> list[Path]:
    paths = [out_dir / f"{stem}.png"]
    fig.savefig(paths[0], dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    if svg:
        paths.append(out_dir / f"{stem}.svg")
        fig.savefig(paths[1], bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    for p in paths:
        LOGGER.info("Saved %s", p)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--training-wide-csv", type=Path, required=True)
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--testing-wide-csv", type=Path, default=None,
                   help="Full wide table, for testing-side AUC. Omit to skip the "
                        "AUC outcome columns.")
    p.add_argument("--dataset", action="append", required=True,
                   help="Dataset to analyse (either arm); repeatable.")
    p.add_argument("--odor-token", default="",
                   help="Testing odor to score. Defaults to the conditioning odor.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bins", type=int, default=DEFAULT_BINS)
    p.add_argument("--no-svg", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    training, testing, testing_auc = _load_inputs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    for requested in args.dataset:
        dataset = resolve_dataset(training, requested)
        if dataset != requested:
            LOGGER.info("Resolved %r to %r", requested, dataset)
        table = build_fly_table(
            training, testing, dataset=dataset,
            odor_token=args.odor_token or None, testing_auc_df=testing_auc,
        )
        if table.empty:
            LOGGER.warning("No joinable flies for %s — skipping", dataset)
            continue
        token = table.attrs["odor_token"]
        label = odor_pretty(token)
        LOGGER.info(
            "%s: %d flies, conditioning trials %s, %s presented at test on %s",
            dataset, len(table), table.attrs["training_trials"], label,
            table.attrs["presentations"],
        )

        stem = dataset.replace("/", "_")
        svg = not args.no_svg
        # One folder per cohort, so a dataset's figures, per-fly CSV and stats
        # travel together.
        ds_dir = args.out_dir / stem
        ds_dir.mkdir(parents=True, exist_ok=True)

        # ---- The three plain-reading figures --------------------------- #
        fig, overview_stats = figure_simple_overview(
            table, dataset=dataset, odor_label=label, n_bins=args.bins
        )
        _save(fig, ds_dir, f"{stem}_01_training_extension_vs_test", svg=svg)

        predictor_stats = []
        for idx, pres in enumerate(table.attrs["presentations"]):
            fig, payload = figure_trial_predictors(
                table, presentation_index=idx, dataset=dataset, odor_label=label
            )
            _save(
                fig, ds_dir,
                f"{stem}_{idx + 2:02d}_which_training_trial_predicts_test_{pres}",
                svg=svg,
            )
            predictor_stats.append(payload)

        # ---- Detail versions, kept out of the cohort's main folder ------ #
        detail_dir = ds_dir / "supplementary"
        detail_dir.mkdir(parents=True, exist_ok=True)
        fig, per_stats = figure_per(
            table, dataset=dataset, odor_label=label, n_bins=args.bins
        )
        _save(fig, detail_dir, f"{stem}_per_vs_training_auc", svg=svg)

        fig, score_stats = figure_score(
            table, dataset=dataset, odor_label=label, n_bins=args.bins
        )
        _save(fig, detail_dir, f"{stem}_score_vs_training_auc", svg=svg)

        fig, matrix_stats = figure_trial_matrix(
            table, dataset=dataset, odor_label=label
        )
        _save(fig, detail_dir, f"{stem}_trial_matrix", svg=svg)

        fig, profile_stats = figure_trial_profile(
            table, dataset=dataset, odor_label=label
        )
        _save(fig, detail_dir, f"{stem}_trial_profile", svg=svg)

        table.to_csv(ds_dir / f"{stem}_per_fly.csv", index=False)
        meta[dataset] = {
            "dataset": dataset,
            "odor_token": token,
            "odor": label,
            "n_flies": int(len(table)),
            "conditioning_trials": list(table.attrs["training_trials"]),
            "testing_presentations": list(table.attrs["presentations"]),
            "flies": [
                {"fly": f, "fly_number": int(n)}
                for f, n in zip(table["fly"], table["fly_number"])
            ],
            "overview": overview_stats,
            "trial_predictors": predictor_stats,
            "supplementary": {
                "per": per_stats,
                "score": score_stats,
                "trial_matrix": matrix_stats,
                "trial_profile": profile_stats,
            },
        }

        per_ds = ds_dir / f"{stem}_stats.json"
        per_ds.write_text(
            json.dumps(meta[dataset], indent=2, default=float), encoding="utf-8"
        )
        LOGGER.info("Saved %s", per_ds)

    sidecar = args.out_dir / "training_auc_vs_control_response.json"
    sidecar.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    LOGGER.info("Saved %s", sidecar)


if __name__ == "__main__":
    main()
