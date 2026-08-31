#!/usr/bin/env python3
"""Naive vs every conditioning and test trial, for a control cohort's own odorant.

A control fly meets its cohort odorant ten times: never paired with light during
six conditioning trials, then twice in testing. The existing figures collapse
that to one bar per odor. This one keeps every presentation and puts a *naive*
bar — the same odorant at the same concentration in a random panel, never
conditioned — on the left of each, so "did repeated unpaired exposure move the
response off naive?" is a comparison the reader can see trial by trial.

Four figures per cohort, because the random panel shows each odorant twice and
averaging those two exposures would hide habituation inside the baseline::

    <dataset>_training_vs_naive-p1   Naive 1, Train 1..6
    <dataset>_training_vs_naive-p2   Naive 2, Train 1..6
    <dataset>_testing_vs_naive-p1    Naive 1, Test 1, Test 2
    <dataset>_testing_vs_naive-p2    Naive 2, Test 1, Test 2

Each is drawn twice: mean ordinal score (Mann-Whitney, Kruskal-Wallis omnibus)
and, under a ``_percent`` suffix, the response rate — the share of flies scoring
at or above 2 — with Wilson 95% intervals, Fisher's exact test and a
Fisher-Freeman-Halton omnibus. Both readings share one set of bars, so a bar's
``n`` is the same in the pair.

"Test 1" and "Test 2" are the two presentations of the cohort odorant in
testing — trials 1 and 8 in the v2 panel, not testing indices 1 and 2, which
are a different odor on every fly.

Training scores do not exist in ``model_predictions.csv``: the pipeline's
``predict_reactions`` step filters to ``trial_type == "testing"`` before it
scores. This driver therefore scores the training wide table itself into a
sidecar ``model_predictions_training.csv``, applying the same frozen-folder and
flagged-fly cuts, so the training bars are filtered exactly like the testing
ones. The sidecar is cached; ``--rescore`` rebuilds it.

Cohorts live in :data:`COHORTS`. Adding a control cohort whose odorant and
concentration a random panel also ran is one entry and nothing else; a cohort
with no concentration-matched panel sets ``naive_dataset=None`` and gets the two
naive-free figures instead of four. There is no ``RandomPanel-24-0.01``, so
``Hex-Control-24-0.01`` is deliberately unmatched rather than compared against a
neighbouring dose.

Run::

    python scripts/analysis/naive_vs_trial_score_bars.py                  # all four
    python scripts/analysis/naive_vs_trial_score_bars.py \\
        --dataset Hex-Control-24-0.1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact, kruskal, mannwhitneyu  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import odor_bar_palette as pal  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _canon_dataset,
    _display_label_ci,
    _extract_odor_from_label,
    _normalise_fly_columns,
    _trial_num,
    set_protocol,
)
from scripts.analysis.per_axis_labels import (  # noqa: E402
    PERCENT_Y_LABEL,
    SCORE_Y_LABEL,
)
from scripts.analysis.pubfig_naive_vs_trained import base_odor  # noqa: E402
from scripts.analysis.randompanel_conc_comparison import (  # noqa: E402
    _wilson_ci,
    fisher_freeman_halton_mc,
    holm_adjust,
)
from scripts.analysis.reaction_matrix_from_spreadsheet import (  # noqa: E402
    _normalise_trial_label,
)
from scripts.analysis.score_summary import _load_scores  # noqa: E402

PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv"
)
TRAINING_WIDE = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_combined_base_training.parquet"
)
TRAINING_PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions_training.csv"
)
MODEL_PATH = Path(
    "/home/ramanlab/Documents/cole/VSCode/FlyBehaviorScoring/outputs/"
    "ordinal_scorer/model_ordinal_xgb.json"
)
FLAGGED_FLIES_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/flagged-flys-truth.csv"
)
OUT_DIR = Path(
    "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Naive-vs-Trial-Score-Bars"
)
#: The *-Sensitivity-* variant. Its baseline is each fly's OWN pre-test panel,
#: not a separate concentration-matched cohort, so it gets its own folder: a
#: reader must never have to guess which baseline a figure used.
PRETEST_OUT_DIR = Path(
    "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Pre-Test-vs-Trial-Score-Bars"
)

BINARY_THRESHOLD = 2
#: A score of 2 or more is a reaction, matching score_summary and the pubfigs.
REACTION_BOUNDARY = 2
SCORE_MIN, SCORE_MAX = -1.0, 5.0
RATE_MIN, RATE_MAX = 0.0, 100.0
#: Headroom above the rate axis for the significance mark on a 100% bar.
RATE_HEADROOM = 12.0
METRICS = ("score", "percent")
ALPHA = 0.05

#: The naive bar is drawn open so it never reads as one of the conditioned
#: trials, matching ``pubfig_naive_vs_trained``.
NAIVE_FACE = "#ffffff"
NAIVE_HATCH = "///"
NAIVE_EDGE = "#333333"

# A little headroom so a tall bar's star does not touch the top spine.
Y_HEADROOM = 0.4

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.family": "Arial",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """One control cohort: its odorant, and where its naive baseline lives."""

    dataset: str
    odor: str
    concentration: str
    #: The random panel run at ``concentration``, or None when none was. None
    #: drops the naive bars rather than borrowing a neighbouring dose, which
    #: would confound concentration with conditioning.
    naive_dataset: str | None
    #: Use this cohort's OWN ``pretest`` trials as the baseline instead of a
    #: separate naive dataset. Within-subject, so no concentration matching and
    #: no cross-cohort confound. The pre-test presents each odor once, so there
    #: is a single baseline exposure rather than the naive panel's two.
    pretest_baseline: bool = False
    n_training: int = 6
    #: Testing trial indices at which the cohort odorant is presented, in order.
    testing_indices: tuple[int, ...] = (1, 8)
    #: Registry name when it differs from ``dataset`` -- how a filtered variant
    #: of a cohort gets its own figures without overwriting the unfiltered ones.
    variant_key: str = ""
    #: Folder-name date prefixes to drop from the cohort's own rows, e.g.
    #: ("august_26",). Applied to this cohort only, never to the naive panel:
    #: the two variants are meant to be read against one shared baseline.
    exclude_fly_dates: tuple[str, ...] = ()
    #: Also emit the condensed figure: naive, the first and last conditioning
    #: trial, and the first test. Off by default -- it is a reading of the same
    #: rows the per-trial figures already carry, useful only where the whole arc
    #: is the point.
    summary: bool = False

    @property
    def key(self) -> str:
        return self.variant_key or self.dataset

    @property
    def canon(self) -> str:
        return _canon_dataset(self.dataset)

    @property
    def naive_canon(self) -> str | None:
        return _canon_dataset(self.naive_dataset) if self.naive_dataset else None


#: The *-Sensitivity-* cohorts, baselined on their own pre-test panel. Kept in
#: a separate registry from COHORTS so the two figure sets never share a folder
#: and neither can silently acquire the other's cohorts.
PRETEST_COHORTS: dict[str, CohortSpec] = {
    spec.key: spec
    for spec in (
        CohortSpec(
            dataset="Hex-Sensitivity-24-0.1",
            odor="Hexanol",
            concentration="0.1%",
            naive_dataset=None,
            pretest_baseline=True,
        ),
        CohortSpec(
            dataset="IAA-Sensitivity-24-1",
            odor="Isoamyl Acetate",
            concentration="1%",
            naive_dataset=None,
            pretest_baseline=True,
        ),
        CohortSpec(
            dataset="3Oct-Sensitivity-24-0.1",
            odor="3-Octanol",
            concentration="0.1%",
            naive_dataset=None,
            pretest_baseline=True,
        ),
        CohortSpec(
            dataset="EB-Sensitivity-24-1",
            odor="Ethyl Butyrate",
            concentration="1%",
            naive_dataset=None,
            pretest_baseline=True,
        ),
    )
}


COHORTS: dict[str, CohortSpec] = {
    spec.key: spec
    for spec in (
        CohortSpec(
            dataset="Hex-Control-24-0.1",
            odor="Hexanol",
            concentration="0.1%",
            naive_dataset="RandomPanel-24-0.1",
        ),
        # Same cohort, minus the late-August folders. Kept alongside the
        # unfiltered entry rather than replacing it, so the effect of the cut is
        # itself visible.
        CohortSpec(
            dataset="Hex-Control-24-0.1",
            odor="Hexanol",
            concentration="0.1%",
            naive_dataset="RandomPanel-24-0.1",
            variant_key="Hex-Control-24-0.1-no-aug25-27",
            exclude_fly_dates=("august_25", "august_26", "august_27"),
            summary=True,
        ),
        CohortSpec(
            dataset="Hex-Control-24-0.01",
            odor="Hexanol",
            concentration="0.01%",
            naive_dataset=None,
        ),
        CohortSpec(
            dataset="EB-Control-24-1",
            odor="Ethyl Butyrate",
            concentration="1%",
            naive_dataset="RandomPanel-24-1",
        ),
        CohortSpec(
            dataset="3Oct-Control-24-0.1",
            odor="3-Octanol",
            concentration="0.1%",
            naive_dataset="RandomPanel-24-0.1",
        ),
    )
}


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


@dataclass
class Bar:
    label: str
    values: Sequence[float]
    color: str
    hatch: str = ""

    @property
    def n(self) -> int:
        return int(len(self.values))

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.n else float("nan")

    @property
    def sem(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.std(self.values, ddof=1) / np.sqrt(self.n))

    @property
    def responders(self) -> int:
        """Flies that reacted -- scored at or above :data:`REACTION_BOUNDARY`."""
        return int(sum(1 for v in self.values if float(v) >= REACTION_BOUNDARY))

    @property
    def rate(self) -> float:
        """Percentage of flies that reacted."""
        if not self.n:
            return float("nan")
        return 100.0 * self.responders / self.n

    @property
    def rate_ci(self) -> tuple[float, float]:
        """Wilson 95% interval on :attr:`rate`, as percentages.

        Wilson rather than normal-approximation: several bars sit at 0% or 100%,
        where the normal interval has zero width and claims a certainty the
        counts do not support.
        """
        if not self.n:
            return (float("nan"), float("nan"))
        low, high = _wilson_ci(self.responders, self.n)
        return (100.0 * low, 100.0 * high)

    @property
    def rate_err(self) -> tuple[float, float]:
        """The Wilson interval as (below, above) offsets, for ``yerr``."""
        low, high = self.rate_ci
        return (max(0.0, self.rate - low), max(0.0, high - self.rate))


@dataclass(frozen=True)
class Stat:
    label: str
    p_raw: float
    p_holm: float

    @property
    def stars(self) -> str:
        return _stars(self.p_holm)


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_testing_scores(csv_path: Path | str) -> pd.DataFrame:
    """Testing + random-panel scores, with ``occurrence`` numbering exposures."""
    set_protocol("v2")
    return _load_scores(Path(csv_path), threshold=None, flagged_flies_csv="")


def load_pretest_scores(csv_path: Path | str) -> pd.DataFrame:
    """Naive pre-training panel scores, in the same shape as the testing ones.

    Empty rather than raising when the file carries no pre-test rows: only the
    *-Sensitivity-* cohorts run a naive panel, and it is absent until a full
    pipeline run has produced it.
    """
    set_protocol("v2")
    try:
        return _load_scores(
            Path(csv_path), threshold=None, flagged_flies_csv="",
            trial_types=("pretest",),
        )
    except RuntimeError:
        return pd.DataFrame()


def load_training_scores(csv_path: Path | str) -> pd.DataFrame:
    """Conditioning-trial scores from the sidecar.

    ``_load_scores`` hard-filters to ``trial_type == "testing"`` and raises when
    nothing survives, so it cannot read this file. The columns it would add that
    matter here — canonical dataset, trial number, odor display — are cheap to
    derive directly, and conditioning trials need no occurrence numbering: the
    trial index already orders them.
    """
    df = pd.read_csv(Path(csv_path))
    required = {"dataset", "fly", "fly_number", "trial_label", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = _normalise_fly_columns(df)
    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].astype(str).apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)
    # Drop trials whose recording name carried no odor token -- a placeholder
    # folder ("flagged") logs bare ``training_N``, and _extract_odor_from_label
    # hands back the label itself for those. _load_scores makes the same cut on
    # the testing side; without it here the training bars would keep rows the
    # testing bars dropped.
    has_odor = df["trial"].apply(lambda t: _extract_odor_from_label(t) != str(t))
    df = df.loc[has_odor].copy()
    df["odor_display"] = df["trial"].apply(
        lambda t: _display_label_ci(_extract_odor_from_label(t))
    )
    return df.dropna(subset=["score"])


def _per_fly(frame: pd.DataFrame) -> list[float]:
    """One score per (fly, fly_number), so a doubly-logged trial counts once."""
    if frame.empty:
        return []
    out = frame.groupby(["fly", "fly_number"], as_index=False)["score"].mean()
    return [float(v) for v in out.sort_values(["fly", "fly_number"])["score"]]


def _odor_rows(
    frame: pd.DataFrame, dataset: str, odor: str, *, trial_type: str | None = None
) -> pd.DataFrame:
    """Rows of *dataset* whose odor is *odor*, matched on the base odorant name.

    A conditioned cohort tags the odorant with its concentration
    ("Hexanol (0.1%)") and the random panel does not, so both sides are stripped
    to the bare name before matching.
    """
    sub = frame[frame["dataset_canon"].astype(str) == _canon_dataset(dataset)]
    # Opt-in, NOT defaulted: the training frame carries trial_type == "training"
    # and the testing frame "testing", so a default of "testing" would silently
    # empty every conditioning bar. Only the pre-test path, which hands in a
    # frame carrying BOTH phases, needs to select one.
    if trial_type and "trial_type" in sub.columns:
        sub = sub[sub["trial_type"].astype(str).str.strip() == str(trial_type)]
    if sub.empty:
        return sub
    target = base_odor(odor).casefold()
    keep = sub["odor_display"].astype(str).map(lambda s: base_odor(s).casefold()) == target
    return sub.loc[keep]


def _excluded(fly: str, dates: Sequence[str]) -> bool:
    """Whether *fly*'s folder name starts with one of the excluded dates.

    Anchored at the start on purpose: the leading token is the recording date,
    and a date appearing later in the name is part of a batch or rig label.
    """
    name = str(fly).strip()
    return any(name.startswith(str(d)) for d in dates)


def _drop_excluded(frame: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    if not spec.exclude_fly_dates or frame.empty:
        return frame
    keep = ~frame["fly"].map(lambda f: _excluded(f, spec.exclude_fly_dates))
    return frame.loc[keep]


def naive_scores(scores: pd.DataFrame, spec: CohortSpec, *, exposure: int) -> pd.DataFrame:
    """Per-fly naive scores on exposure *exposure* of the cohort odorant."""
    if spec.pretest_baseline:
        frame = _odor_rows(scores, spec.dataset, spec.odor, trial_type="pretest")
        if frame.empty:
            return pd.DataFrame(columns=["fly", "fly_number", "score"])
        return (
            frame.groupby(["fly", "fly_number"], as_index=False)["score"]
            .mean()
            .sort_values(["fly", "fly_number"])
            .reset_index(drop=True)
        )
    if not spec.naive_dataset:
        return pd.DataFrame(columns=["fly", "fly_number", "score"])
    frame = _odor_rows(scores, spec.naive_dataset, spec.odor)
    if frame.empty:
        return pd.DataFrame(columns=["fly", "fly_number", "score"])
    if "occurrence" in frame.columns:
        frame = frame[pd.to_numeric(frame["occurrence"], errors="coerce") == int(exposure)]
    if frame.empty:
        return pd.DataFrame(columns=["fly", "fly_number", "score"])
    return (
        frame.groupby(["fly", "fly_number"], as_index=False)["score"]
        .mean()
        .sort_values(["fly", "fly_number"])
        .reset_index(drop=True)
    )


def naive_bar(scores: pd.DataFrame, spec: CohortSpec, exposure: int) -> Bar:
    values = [float(v) for v in naive_scores(scores, spec, exposure=exposure)["score"]]
    label = "Pre-test" if spec.pretest_baseline else f"Naive {exposure}"
    return Bar(label, values, NAIVE_FACE, NAIVE_HATCH)


def testing_scores(scores: pd.DataFrame, spec: CohortSpec) -> list[Bar]:
    """One bar per presentation of the cohort odorant in testing."""
    # Explicit for the pre-test variant, whose frame also carries pretest rows.
    frame = _drop_excluded(
        _odor_rows(scores, spec.dataset, spec.odor,
                   trial_type="testing" if spec.pretest_baseline else None),
        spec,
    )
    color = pal.odor_color(spec.odor) or pal.TRAIN_COLOR
    if spec.pretest_baseline:
        # The sensitivity panel randomises odor ORDER per fly, so the CS+ sits
        # at a different testing index for each one. Keying on the index
        # dropped every fly whose CS+ was not at index 1 — a pre-test bar of
        # n=7 beside a test bar of n=2. Key on the odor instead: one
        # presentation per fly, one bar.
        values = _per_fly(frame)
        return [Bar("Test", values, color)]
    bars: list[Bar] = []
    for position, index in enumerate(spec.testing_indices, start=1):
        at = (
            frame[pd.to_numeric(frame["trial_num"], errors="coerce") == index]
            if not frame.empty
            else frame
        )
        bars.append(Bar(f"Test {position}", _per_fly(at), color))
    return bars


def training_scores(training: pd.DataFrame, spec: CohortSpec) -> list[Bar]:
    """One bar per conditioning trial, 1..``spec.n_training``."""
    frame = _drop_excluded(_odor_rows(training, spec.dataset, spec.odor), spec)
    color = pal.odor_color(spec.odor) or pal.TRAIN_COLOR
    bars: list[Bar] = []
    for index in range(1, spec.n_training + 1):
        at = (
            frame[pd.to_numeric(frame["trial_num"], errors="coerce") == index]
            if not frame.empty
            else frame
        )
        bars.append(Bar(f"Train {index}", _per_fly(at), color))
    return bars


# ---------------------------------------------------------------------------
# Figure plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigurePlan:
    spec: CohortSpec
    phase: str  # "training" | "testing"
    naive_exposure: int | None

    @property
    def stem(self) -> str:
        if self.naive_exposure is None:
            return f"{self.spec.key}_{self.phase}"
        if self.spec.pretest_baseline:
            # One presentation per odor, so no "-pN": an exposure number would
            # imply a second pre-test that does not exist. The name also has to
            # say "pretest" — the filename is what reaches a figure caption.
            return f"{self.spec.key}_{self.phase}_vs_pretest"
        return f"{self.spec.key}_{self.phase}_vs_naive-p{self.naive_exposure}"

    def stem_for(self, metric: str) -> str:
        """The score figure keeps the bare stem; the rate figure is suffixed."""
        return self.stem if metric == "score" else f"{self.stem}_{metric}"

    @property
    def title(self) -> str:
        head = f"{self.spec.odor} ({self.spec.concentration}) — "
        if self.phase == "summary":
            head += "first and last conditioning, then first test"
        else:
            phase = "conditioning" if self.phase == "training" else "testing"
            head += f"{phase} trials"
        cut = (
            f", excl. {'/'.join(self.spec.exclude_fly_dates)}"
            if self.spec.exclude_fly_dates
            else ""
        )
        if self.naive_exposure is None:
            return (
                f"{head}\n{self.spec.dataset}{cut}"
                " (no concentration-matched naive panel)"
            )
        return (
            f"{head} vs pre-test"
            if self.spec.pretest_baseline else
            f"{head} vs naive exposure {self.naive_exposure}"
            f"\n{self.spec.dataset}{cut}, unpaired odor presentation"
        )


def summary_labels(spec: CohortSpec) -> tuple[str, ...]:
    """The condensed figure's trial bars: first and last conditioning, first test."""
    return ("Train 1", f"Train {spec.n_training}", "Test 1")


def figure_plans(spec: CohortSpec) -> list[FigurePlan]:
    """Four figures when a naive panel exists, two when none does.

    A cohort with ``summary=True`` gets one more per naive exposure: the arc from
    naive through the first and last conditioning trial to the first test, on one
    axes.
    """
    if spec.pretest_baseline:
        # One presentation per odor in the pre-test: a second exposure would be
        # an empty bar, not a missing one.
        plans = [FigurePlan(spec, "training", 1), FigurePlan(spec, "testing", 1)]
        if spec.summary:
            plans.append(FigurePlan(spec, "summary", 1))
        return plans
    if spec.naive_dataset is None:
        plans = [FigurePlan(spec, "training", None), FigurePlan(spec, "testing", None)]
        if spec.summary:
            plans.append(FigurePlan(spec, "summary", None))
        return plans
    plans = [
        FigurePlan(spec, "training", 1),
        FigurePlan(spec, "training", 2),
        FigurePlan(spec, "testing", 1),
        FigurePlan(spec, "testing", 2),
    ]
    if spec.summary:
        plans += [FigurePlan(spec, "summary", 1), FigurePlan(spec, "summary", 2)]
    return plans


def build_bars(
    plan: FigurePlan, scores: pd.DataFrame, training: pd.DataFrame
) -> list[Bar]:
    """The figure's bars, naive first when the plan has one."""
    if plan.phase == "summary":
        wanted = summary_labels(plan.spec)
        pool = {
            bar.label: bar
            for bar in (
                *training_scores(training, plan.spec),
                *testing_scores(scores, plan.spec),
            )
        }
        trials = [pool[label] for label in wanted if label in pool]
    elif plan.phase == "training":
        trials = training_scores(training, plan.spec)
    else:
        trials = testing_scores(scores, plan.spec)
    if plan.naive_exposure is None:
        return trials
    return [naive_bar(scores, plan.spec, plan.naive_exposure), *trials]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def omnibus(bars: Sequence[Bar]) -> float | None:
    """Kruskal-Wallis p across the populated bars, or None if fewer than two."""
    groups = [list(b.values) for b in bars if b.n]
    if len(groups) < 2:
        return None
    try:
        return float(kruskal(*groups).pvalue)
    except ValueError:
        # every observation identical -- no variance to test
        return None


def compare_to_naive(naive: Bar | None, trials: Sequence[Bar]) -> list[Stat]:
    """Each trial bar against the naive bar, Holm-corrected within the figure."""
    if naive is None or not naive.n:
        return []
    testable = [b for b in trials if b.n]
    if not testable:
        return []
    raw: list[float] = []
    for bar in testable:
        try:
            raw.append(float(mannwhitneyu(naive.values, bar.values).pvalue))
        except ValueError:
            raw.append(1.0)
    adjusted = holm_adjust(raw)
    return [
        Stat(bar.label, p, float(q)) for bar, p, q in zip(testable, raw, adjusted)
    ]


def compare_rates_to_naive(naive: Bar | None, trials: Sequence[Bar]) -> list[Stat]:
    """Each trial bar's response rate against naive, by Fisher's exact test.

    The rates are counts of responders, not a continuous measure, so the score
    figure's Mann-Whitney would be ranking a column of 0s and 1s.
    """
    if naive is None or not naive.n:
        return []
    testable = [b for b in trials if b.n]
    if not testable:
        return []
    raw: list[float] = []
    for bar in testable:
        table = [
            [naive.responders, naive.n - naive.responders],
            [bar.responders, bar.n - bar.responders],
        ]
        try:
            raw.append(float(fisher_exact(table).pvalue))
        except ValueError:
            raw.append(1.0)
    adjusted = holm_adjust(raw)
    return [Stat(bar.label, p, float(q)) for bar, p, q in zip(testable, raw, adjusted)]


def rate_omnibus(bars: Sequence[Bar]) -> float | None:
    """Fisher-Freeman-Halton across the populated bars' responder counts."""
    populated = [b for b in bars if b.n]
    if len(populated) < 2:
        return None
    # The test takes per-observation labels, not a table: one row per fly,
    # coded 1 if it reacted, tagged with the bar it came from.
    cats: list[int] = []
    groups: list[int] = []
    for index, bar in enumerate(populated):
        for value in bar.values:
            cats.append(1 if float(value) >= REACTION_BOUNDARY else 0)
            groups.append(index)
    try:
        return float(
            fisher_freeman_halton_mc(np.array(cats), np.array(groups), n_iter=20_000)
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Training-score sidecar
# ---------------------------------------------------------------------------


def _prepare_training_input(wide: pd.DataFrame) -> pd.DataFrame:
    """Frozen-folder and flagged-fly cuts, matching ``predict_reactions``."""
    from fbpipe.config import load_flagged_fly_exclusions
    from fbpipe.utils.frozen_folders import drop_frozen

    frame = drop_frozen(wide)
    if FLAGGED_FLIES_CSV.exists():
        exclusions = load_flagged_fly_exclusions(FLAGGED_FLIES_CSV)
        if exclusions:
            keys = pd.Series(
                list(
                    zip(
                        frame.get("dataset", pd.Series("", index=frame.index)).astype(str),
                        frame.get("fly", pd.Series("", index=frame.index)).astype(str),
                        frame.get("fly_number", pd.Series("", index=frame.index)).astype(str),
                    )
                ),
                index=frame.index,
                dtype=object,
            )
            frame = frame.loc[~keys.isin(exclusions)].copy()
    if "trial_type" in frame.columns:
        mask = frame["trial_type"].astype(str).str.strip().str.lower() == "training"
        frame = frame.loc[mask].copy()
    return frame


def ensure_training_predictions(
    training_wide: Path,
    model_path: Path,
    output_csv: Path,
    *,
    rescore: bool = False,
    runner: Callable[..., object] = subprocess.check_call,
) -> Path:
    """Score the conditioning trials into *output_csv*, reusing a cached file.

    ``predict_reactions`` never scores these rows, so this is the only place the
    training scores come from. The filters applied here mirror that step, or the
    training bars would be drawn from a different fly set than the testing ones.
    """
    if output_csv.exists() and not rescore:
        return output_csv
    if not training_wide.exists():
        raise FileNotFoundError(f"Training wide table not found: {training_wide}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from fbpipe.utils.tables import read_table

    frame = _prepare_training_input(read_table(training_wide))
    if frame.empty:
        raise RuntimeError(f"No training rows survived filtering in {training_wide}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix="_training_for_scoring.csv", delete=False
    )
    try:
        frame.to_csv(tmp.name, index=False)
    finally:
        tmp.close()
    try:
        runner(
            [
                "flybehavior-response",
                "predict-ordinal",
                "--data-csv",
                tmp.name,
                "--model-path",
                str(model_path),
                "--output-csv",
                str(output_csv),
                "--binary-threshold",
                str(BINARY_THRESHOLD),
            ]
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return output_csv


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _annotate_stars(
    ax, bars: Sequence[Bar], stats: Sequence[Stat], tops: Sequence[float], pad: float
) -> None:
    """Star each trial bar above its error bar.

    Every star is that bar against the naive bar, which the title already says.
    Six stacked brackets from a single reference bar would cost a third of the
    axes height to encode what one row of marks does.
    """
    by_label = {s.label: s for s in stats}
    for x, bar in enumerate(bars):
        stat = by_label.get(bar.label)
        if stat is None or not bar.n:
            continue
        ax.text(
            x,
            tops[x] + pad,
            stat.stars,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )


def render(
    plan: FigurePlan,
    scores: pd.DataFrame,
    training: pd.DataFrame,
    out_dir: Path,
    *,
    metric: str = "score",
) -> list[Path]:
    """Draw one figure. Returns the files written — empty when there is no data.

    ``metric="score"`` plots the mean ordinal score +/- SEM, tested by
    Mann-Whitney. ``metric="percent"`` plots the response rate — the share of
    flies scoring at or above :data:`REACTION_BOUNDARY` — with Wilson intervals
    and Fisher's exact test. Both read the same bars, so a bar's ``n`` is
    identical across the pair.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

    bars = build_bars(plan, scores, training)
    if not any(b.n for b in bars):
        return []

    naive = bars[0] if plan.naive_exposure is not None else None
    trials = bars[1:] if naive is not None else bars

    is_rate = metric == "percent"
    if is_rate:
        stats = compare_rates_to_naive(naive, trials)
        omni = rate_omnibus(bars)
        heights = [b.rate for b in bars]
        yerr = np.array([b.rate_err for b in bars], dtype=float).T
        tops = [b.rate_ci[1] for b in bars]
        y_label, y_min, y_max = PERCENT_Y_LABEL, RATE_MIN, RATE_MAX + RATE_HEADROOM
        y_ticks = np.arange(0.0, 101.0, 20.0)
        star_pad = 2.5
        omni_label = "Fisher–Freeman–Halton"
    else:
        stats = compare_to_naive(naive, trials)
        omni = omnibus(bars)
        heights = [b.mean for b in bars]
        yerr = np.array([[b.sem for b in bars], [b.sem for b in bars]], dtype=float)
        tops = [b.mean + b.sem for b in bars]
        y_label, y_min, y_max = SCORE_Y_LABEL, SCORE_MIN, SCORE_MAX + Y_HEADROOM
        y_ticks = np.arange(SCORE_MIN, SCORE_MAX + 1, 1.0)
        star_pad = 0.18
        omni_label = "Kruskal–Wallis"

    stem = plan.stem_for(metric)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    xs = np.arange(len(bars), dtype=float)

    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(0.95 * len(bars) + 1.8, 4.0))
        ax.bar(
            xs,
            heights,
            yerr=yerr,
            color=[b.color for b in bars],
            hatch=[b.hatch for b in bars],
            edgecolor=[NAIVE_EDGE if b.hatch else "none" for b in bars],
            width=0.72,
            capsize=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
            zorder=2,
        )
        ax.axhline(0.0, color="#999999", lw=0.6, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b.label}\nn={b.n}" for b in bars], fontsize=8)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        subtitle = "" if omni is None else f"{omni_label} p = {omni:.3g}"
        ax.set_title(f"{plan.title}\n{subtitle}".rstrip(), fontsize=9)
        if stats:
            _annotate_stars(ax, bars, stats, tops, star_pad)
        fig.tight_layout()
        for suffix in (".png", ".svg"):
            path = out_dir / f"{stem}{suffix}"
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    by_label = {s.label: s for s in stats}
    rows = []
    for bar in bars:
        stat = by_label.get(bar.label)
        low, high = bar.rate_ci
        rows.append(
            {
                "figure": stem,
                "metric": metric,
                "dataset": plan.spec.dataset,
                "variant": plan.spec.key,
                "excluded_fly_dates": "|".join(plan.spec.exclude_fly_dates),
                "odor": plan.spec.odor,
                "concentration": plan.spec.concentration,
                "naive_dataset": plan.spec.naive_dataset or "",
                "label": bar.label,
                "n": bar.n,
                "mean": bar.mean,
                "sem": bar.sem,
                "responders": bar.responders,
                "rate": bar.rate,
                "ci_low": low,
                "ci_high": high,
                "p_raw": stat.p_raw if stat else np.nan,
                "p_holm": stat.p_holm if stat else np.nan,
                "stars": stat.stars if stat else "",
                "omnibus_test": omni_label,
                "omnibus_p": omni if omni is not None else np.nan,
            }
        )
    stats_path = out_dir / f"{stem}_stats.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)
    written.append(stats_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    datasets: Sequence[str],
    *,
    predictions_csv: Path = PREDICTIONS_CSV,
    training_wide: Path = TRAINING_WIDE,
    training_predictions_csv: Path = TRAINING_PREDICTIONS_CSV,
    model_path: Path = MODEL_PATH,
    out_dir: Path = OUT_DIR,
    rescore: bool = False,
    metrics: Sequence[str] = METRICS,
    registry: Mapping[str, CohortSpec] | None = None,
) -> list[Path]:
    ensure_training_predictions(
        training_wide, model_path, training_predictions_csv, rescore=rescore
    )
    cohorts = COHORTS if registry is None else registry
    scores = load_testing_scores(predictions_csv)
    if any(cohorts[name].pretest_baseline for name in datasets):
        # The pre-test baseline lives in the same predictions file under
        # trial_type == "pretest"; _load_scores serves one phase per call, so
        # the two frames are concatenated and _odor_rows selects the phase.
        pretest = load_pretest_scores(predictions_csv)
        if not pretest.empty:
            scores = pd.concat([scores, pretest], ignore_index=True)
        else:
            print("[WARN] no pretest rows in the predictions CSV; "
                  "baseline bars will be empty.")
    training = load_training_scores(training_predictions_csv)

    written: list[Path] = []
    manifest: list[dict] = []
    for name in datasets:
        spec = cohorts[name]
        cohort_dir = out_dir / spec.key
        for plan in figure_plans(spec):
            for metric in metrics:
                files = render(plan, scores, training, cohort_dir, metric=metric)
                stem = plan.stem_for(metric)
                if not files:
                    print(f"[SKIP] {stem}: no rows")
                    continue
                written.extend(files)
                manifest.append({"figure": stem, "files": [str(p) for p in files]})
                print(f"[OK] {stem}")
    if manifest:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(set(COHORTS) | set(PRETEST_COHORTS)),
        help="Cohort to render; repeatable. Default: every cohort in the registry.",
    )
    parser.add_argument(
        "--pretest-baseline",
        action="store_true",
        help="Render the *-Sensitivity-* cohorts against their OWN pre-test "
             "panel instead of a separate naive dataset. Switches both the "
             "cohort registry and the output folder together, so the two sets "
             "can never be mixed.",
    )
    parser.add_argument("--predictions-csv", type=Path, default=PREDICTIONS_CSV)
    parser.add_argument("--training-wide-csv", type=Path, default=TRAINING_WIDE)
    parser.add_argument(
        "--training-predictions-csv", type=Path, default=TRAINING_PREDICTIONS_CSV
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Defaults to Naive-vs-Trial-Score-Bars, or "
             "Pre-Test-vs-Trial-Score-Bars under --pretest-baseline.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=list(METRICS),
        help="Which figures to draw; repeatable. Default: both score and percent.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Rebuild the training-score sidecar even if it already exists.",
    )
    args = parser.parse_args(argv)

    # Registry and output folder move together — a pre-test figure must never
    # land in the naive-baselined folder, where its baseline would be unreadable.
    registry = PRETEST_COHORTS if args.pretest_baseline else COHORTS
    default_out = PRETEST_OUT_DIR if args.pretest_baseline else OUT_DIR

    run(
        args.dataset or sorted(registry),
        predictions_csv=args.predictions_csv,
        training_wide=args.training_wide_csv,
        training_predictions_csv=args.training_predictions_csv,
        model_path=args.model_path,
        out_dir=args.out_dir or default_out,
        rescore=args.rescore,
        metrics=args.metric or METRICS,
        registry=registry,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
