"""Score bars that compare one odor *across* conditioning protocols.

Every other score-bar figure draws one cohort's odor panel. These two put bars
from different cohorts on the same axes:

``score_bars_CSplus_across_conditioning``
    The conditioned stimulus itself, after conditioning: hexanol in the
    hexanol-trained flies, air in the air-trained flies, and apple cider
    vinegar in the ACV-trained flies. Only hexanol conditioning moves its own
    CS+; the two appetitively neutral protocols land on zero.

``score_bars_EthylButyrate_across_conditioning``
    Ethyl butyrate in all four Oct/Nov cohorts. EB was never the CS+ in any of
    them, so the bars are expected to sit on top of each other — the figure is
    the evidence for that, and it draws no brackets because nothing separates.

Provenance
----------
No single predictions CSV reproduces every published bar, so each cohort is
read from the CSV its own published figure was rendered from:

============  ========  ====================================================
Cohort        Source    Why
============  ========  ====================================================
Hex-Training  frozen    Identical in both files, so the choice is free.
Hex-Control   frozen    The rebuild lost two Oct/Nov control flies and
                        rescored others: the live file returns n=13 / EB 2.46
                        even after reading ``Hex-Control-flagged`` back, where
                        the published bar is n=15 / EB 2.00.
AIR-Training  live      The published AIR figure used the live file; the
                        frozen copy scores this cohort differently (EB 2.33).
ACV-Training  live      Not present in the frozen copy at all.
============  ========  ====================================================

The cost is that Hex-Control is scored by a different model revision than
AIR/ACV. That is a real caveat for the EB figure and is why the EB comparison
is reported with its omnibus test rather than eyeballed; the conclusion (no
cohort separates) holds under either source for Hex-Control.

Run::

    /home/ramanlab/anaconda3/bin/python \
        scripts/analysis/cross_cohort_score_bars.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kruskal, mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import odor_bar_palette  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _canon_dataset,
    _display_odor,
    _is_testing_11_label,
    _normalise_fly_columns,
    _trial_num,
)
from scripts.analysis.reaction_matrix_from_spreadsheet import (  # noqa: E402
    _normalise_trial_label,
)
from scripts.analysis.per_axis_labels import SCORE_Y_LABEL as _SCORE_Y_LABEL  # noqa: E402

PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/model_predictions.csv"
)
# The May snapshot the published Oct/Nov Hex figures were rendered from. It is
# the only surviving copy of the full 15-fly Hex-Control cohort.
FROZEN_PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Results/Opto-Fly-Figures-OctNov/"
    "Matrix-PER-Reactions-Model/model_predictions_oct_nov.csv"
)
# AIR-Training trial labels carry no odor token, so the odor per trial is read
# back off the binary export written next to the reaction figure.
AIR_BINARY_CSV = Path(
    "/home/ramanlab/Documents/cole/Results/Opto-Fly-Figures/"
    "Matrix-PER-Reactions-Model/AIR-Training/binary_reactions_AIR-Training_unordered.csv"
)
FIGURES_DIR = Path("/home/ramanlab/Documents/cole/Results/Figures")

OCTNOV_PREFIXES = ("october_", "november_")

SCORE_MIN, SCORE_MAX = -1.0, 5.0
SCORE_Y_LABEL = _SCORE_Y_LABEL

# AIR is not an odorant and has no palette entry; it keeps the blue every other
# AIR bar in this project uses.
AIR_COLOR = "tab:blue"

EB_ODOR = "Ethyl Butyrate"

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}

__all__ = [
    "AIR_COLOR",
    "COHORTS",
    "Cohort",
    "OCTNOV_PREFIXES",
    "FROZEN_PREDICTIONS_CSV",
    "LIVE_HEX_CONTROL",
    "PREDICTIONS_CSV",
    "SCORE_MAX",
    "SCORE_MIN",
    "SCORE_Y_LABEL",
    "cohort_scores",
    "CS_PLUS_COHORT_ORDER",
    "cs_plus_omnibus",
    "cs_plus_pairwise",
    "cs_plus_pvalue",
    "cs_plus_table",
    "eb_omnibus",
    "eb_pairwise",
    "eb_table",
    "first_presentation_scores",
    "load_predictions",
    "load_sources",
    "render_bars",
    "significant_brackets",
    "stars",
]


@dataclass(frozen=True)
class Cohort:
    """One conditioning protocol, and how to pull its flies out of the CSV."""

    key: str
    label: str
    datasets: tuple[str, ...]
    date_prefixes: tuple[str, ...] | None
    color: str
    odor_source: str  # "trial_map" | "air_binary"
    source: str = "live"  # "live" | "frozen"; see the module docstring
    # The stimulus this cohort was conditioned to. ``None`` for an unpaired
    # control, which by definition has no conditioned stimulus.
    cs_plus: str | None = None


COHORTS: dict[str, Cohort] = {
    # The Hex cohorts span more than Oct/Nov in the predictions CSV, so they
    # carry the date filter the published oct_nov figures use.
    "Hex-Training": Cohort(
        key="Hex-Training",
        label="Hexanol\nconditioned",
        datasets=("Hex-Training",),
        date_prefixes=OCTNOV_PREFIXES,
        color=odor_bar_palette.HEX_COLOR,
        odor_source="trial_map",
        source="frozen",
        cs_plus="Hexanol",
    ),
    # Frozen, and therefore *without* ``-flagged``: the rebuild that scattered
    # this cohort into a flagged dataset is exactly what the frozen copy
    # predates. Reading it live needs ("Hex-Control", "Hex-Control-flagged")
    # and still only recovers 13 of the 15 flies.
    "Hex-Control": Cohort(
        key="Hex-Control",
        label="Hexanol\nunpaired control",
        datasets=("Hex-Control",),
        date_prefixes=OCTNOV_PREFIXES,
        color=odor_bar_palette.CTRL_COLOR,
        odor_source="trial_map",
        source="frozen",
    ),
    "AIR-Training": Cohort(
        key="AIR-Training",
        label="Air\nconditioned",
        datasets=("AIR-Training",),
        date_prefixes=None,
        color=AIR_COLOR,
        odor_source="air_binary",
        source="live",
        cs_plus="AIR",
    ),
    "ACV-Training": Cohort(
        key="ACV-Training",
        label="Apple cider vinegar\nconditioned",
        datasets=("ACV-Training",),
        date_prefixes=None,
        color=odor_bar_palette.ACV_COLOR,
        odor_source="trial_map",
        source="live",
        cs_plus="Apple Cider Vinegar",
    ),
}

# What a live read of Hex-Control needs to get as close as it can (n=13). Kept
# so the regression test can prove the live path is still the lossy one.
LIVE_HEX_CONTROL = Cohort(
    key="Hex-Control",
    label=COHORTS["Hex-Control"].label,
    datasets=("Hex-Control", "Hex-Control-flagged"),
    date_prefixes=OCTNOV_PREFIXES,
    color=odor_bar_palette.CTRL_COLOR,
    odor_source="trial_map",
    source="live",
)

# Figure 1's bars: every cohort that has a conditioned stimulus, each showing
# its own CS+. Hex-Control is absent because an unpaired control has none.
CS_PLUS_COHORT_ORDER = ("Hex-Training", "AIR-Training", "ACV-Training")

# Figure 2's bars, in plotted order.
EB_COHORT_ORDER = ("Hex-Training", "Hex-Control", "AIR-Training", "ACV-Training")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_predictions(path: Path | None = None) -> pd.DataFrame:
    """Testing-trial rows with an odor attached, one row per fly and trial.

    Mirrors ``score_bars_variants._load_predictions`` so the cohorts here come
    out identical to the ones the single-cohort figures publish.
    """
    df = pd.read_csv(path or PREDICTIONS_CSV)
    if "score" not in df.columns:
        raise RuntimeError("Predictions CSV is missing the 'score' column.")

    df = df[df["trial_type"].astype(str).str.strip().str.lower() == "testing"].copy()
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["trial_label"] = df["trial_label"].astype(str).str.strip()
    df = _normalise_fly_columns(df)
    df = df.loc[~df.get("_non_reactive", pd.Series(False, index=df.index)).astype(bool)]
    df = df.copy()

    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score", "trial_num"]).copy()
    df = df[~df["trial"].apply(_is_testing_11_label)].copy()
    df = df.drop_duplicates(subset=["dataset", "fly", "fly_number", "trial"], keep="first")
    df["trial_num"] = df["trial_num"].astype(int)
    return df


def load_sources(
    live_csv: Path | None = None, frozen_csv: Path | None = None
) -> dict[str, pd.DataFrame]:
    """Both predictions files, keyed by the ``Cohort.source`` they serve."""
    return {
        "live": load_predictions(live_csv or PREDICTIONS_CSV),
        "frozen": load_predictions(frozen_csv or FROZEN_PREDICTIONS_CSV),
    }


def _frame_for(preds: pd.DataFrame | dict[str, pd.DataFrame], cohort: Cohort) -> pd.DataFrame:
    """Pick a cohort's predictions frame.

    A bare DataFrame is used for every cohort, which keeps single-source
    overrides (and tests) working; a mapping dispatches on ``cohort.source``.
    """
    if isinstance(preds, dict):
        try:
            return preds[cohort.source]
        except KeyError:
            raise RuntimeError(
                f"Cohort {cohort.key!r} wants the {cohort.source!r} predictions "
                f"source; got {sorted(preds)}."
            ) from None
    return preds


def _air_odor_lookup(binary_csv: Path | None = None) -> pd.DataFrame:
    binary = _normalise_fly_columns(pd.read_csv(binary_csv or AIR_BINARY_CSV))
    binary["trial_num"] = pd.to_numeric(binary["trial_num"], errors="coerce")
    binary = binary.dropna(subset=["trial_num"]).copy()
    binary["trial_num"] = binary["trial_num"].astype(int)
    binary["odor_sent"] = binary["odor_sent"].astype(str).str.strip()
    key = ["fly", "fly_number", "trial_num"]
    return binary[key + ["odor_sent"]].drop_duplicates(subset=key, keep="first")


def cohort_scores(
    preds: pd.DataFrame | dict[str, pd.DataFrame],
    cohort: Cohort,
    *,
    binary_csv: Path | None = None,
) -> pd.DataFrame:
    """One row per (fly, trial) for a cohort, with the presented odor attached."""
    frame = _frame_for(preds, cohort)
    sub = frame[frame["dataset"].isin(cohort.datasets)].copy()
    if cohort.date_prefixes is not None:
        sub = sub[sub["fly"].str.startswith(cohort.date_prefixes)].copy()
    if sub.empty:
        raise RuntimeError(f"No rows for cohort {cohort.key!r}.")

    if cohort.odor_source == "air_binary":
        key = ["fly", "fly_number", "trial_num"]
        sub = sub.merge(_air_odor_lookup(binary_csv), on=key, how="left")
        sub = sub.rename(columns={"odor_sent": "odor"})
        missing = sub[sub["odor"].isna()]
        if not missing.empty:
            sample = ", ".join(
                f"{f}/{n} trial {t}"
                for f, n, t in missing[key].head(5).itertuples(index=False)
            )
            raise RuntimeError(
                f"{len(missing)} {cohort.key} trial(s) have no odor in the binary "
                f"export: {sample}"
            )
    else:
        # A cohort merged from several datasets ("-flagged") must map odors with
        # one canonical name, not each dataset's own.
        canon = _canon_dataset(cohort.key)
        sub["odor"] = [_display_odor(canon, t) for t in sub["trial"]]
        sub["odor"] = sub["odor"].replace({"3-Octonol": "3-Octanol"})

    sub["odor"] = sub["odor"].astype(str).str.strip()
    # A fly re-scored on the same trial would otherwise weight twice.
    fly_level = (
        sub.groupby(["trial_num", "odor", "fly", "fly_number"])["score"]
        .mean()
        .rename("fly_mean_score")
        .reset_index()
    )
    return fly_level


def first_presentation_scores(
    preds: pd.DataFrame | dict[str, pd.DataFrame],
    cohort: Cohort,
    odor: str,
    *,
    binary_csv: Path | None = None,
) -> pd.Series:
    """Per-fly scores for the *first* presentation of one odor in one cohort."""
    fly_level = cohort_scores(preds, cohort, binary_csv=binary_csv)
    sub = fly_level[fly_level["odor"].str.casefold() == str(odor).casefold()]
    if sub.empty:
        raise RuntimeError(f"{cohort.key} never presented {odor!r}.")
    first = sub["trial_num"].min()
    return sub.loc[sub["trial_num"] == first, "fly_mean_score"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def stars(p: float) -> str:
    """Star label for a p-value; empty when it is not significant.

    Matches ``score_bars_variants._stars``: a non-significant pair gets no
    bracket and no label rather than an "n.s." annotation.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _mw(a: pd.Series, b: pd.Series) -> float:
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def _holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _cs_plus_vectors(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> dict[str, pd.Series]:
    """Each conditioning cohort's response to its *own* conditioned stimulus."""
    vectors = {}
    for key in CS_PLUS_COHORT_ORDER:
        cohort = COHORTS[key]
        if cohort.cs_plus is None:
            raise RuntimeError(f"Cohort {key!r} has no conditioned stimulus to plot.")
        vectors[key] = first_presentation_scores(
            preds, cohort, cohort.cs_plus, binary_csv=binary_csv
        )
    return vectors


def cs_plus_pvalue(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> float:
    """Hexanol in hexanol-trained flies vs air in air-trained flies, uncorrected.

    Kept as the headline two-group comparison. The figure labels its brackets
    from :func:`cs_plus_pairwise`, which corrects across all three pairs.
    """
    hexanol = first_presentation_scores(preds, COHORTS["Hex-Training"], "Hexanol")
    air = first_presentation_scores(
        preds, COHORTS["AIR-Training"], "AIR", binary_csv=binary_csv
    )
    return _mw(hexanol, air)


def cs_plus_omnibus(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> dict[str, float]:
    """Kruskal-Wallis across the three conditioned-stimulus responses."""
    vectors = _cs_plus_vectors(preds, binary_csv=binary_csv)
    h, p = kruskal(*vectors.values())
    return {"H": float(h), "p": float(p), "k": len(vectors)}


def cs_plus_pairwise(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> pd.DataFrame:
    """Every CS+ pair, Mann-Whitney with a Holm correction."""
    return _pairwise(_cs_plus_vectors(preds, binary_csv=binary_csv))


def _pairwise(vectors: dict[str, pd.Series]) -> pd.DataFrame:
    keys = list(vectors)
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rows.append(
                {
                    "left": keys[i],
                    "right": keys[j],
                    "i": i,
                    "j": j,
                    "p": _mw(vectors[keys[i]], vectors[keys[j]]),
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm"] = _holm(out["p"].tolist())
    out["stars"] = out["p_holm"].apply(stars)
    return out


def _eb_vectors(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> dict[str, pd.Series]:
    return {
        key: first_presentation_scores(
            preds, COHORTS[key], EB_ODOR, binary_csv=binary_csv
        )
        for key in EB_COHORT_ORDER
    }


def eb_omnibus(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> dict[str, float]:
    """Kruskal-Wallis across the four EB cohorts."""
    vectors = _eb_vectors(preds, binary_csv=binary_csv)
    h, p = kruskal(*vectors.values())
    return {"H": float(h), "p": float(p), "k": len(vectors)}


def eb_pairwise(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> pd.DataFrame:
    """Every EB cohort pair, Mann-Whitney with a Holm correction."""
    return _pairwise(_eb_vectors(preds, binary_csv=binary_csv))


def significant_brackets(
    preds: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    which: str = "eb",
    binary_csv: Path | None = None,
) -> list[dict]:
    """Brackets to draw, one per significant pair. Empty when nothing separates.

    Narrow spans come first so the stack builds outward and a wide bracket
    never has to cross a narrow one.
    """
    if which == "cs_plus":
        pairs = cs_plus_pairwise(preds, binary_csv=binary_csv)
    elif which == "eb":
        pairs = eb_pairwise(preds, binary_csv=binary_csv)
    else:
        raise ValueError(f"Unknown comparison {which!r}; expected 'cs_plus' or 'eb'.")

    keep = pairs[pairs["stars"] != ""].copy()
    keep["span"] = (keep["j"] - keep["i"]).abs()
    keep = keep.sort_values(["span", "i"], kind="mergesort")
    return [
        {
            "i": int(r.i),
            "j": int(r.j),
            "p": float(r.p),
            "p_holm": float(r.p_holm),
            "stars": r.stars,
        }
        for r in keep.itertuples(index=False)
    ]


# ---------------------------------------------------------------------------
# Bar tables
# ---------------------------------------------------------------------------
def _bar_row(scores: pd.Series, *, label: str, odor: str, cohort: str, color: str) -> dict:
    return {
        "cohort": cohort,
        "label": label,
        "odor": odor,
        "mean_score": float(scores.mean()),
        "sem_score": float(scores.sem()) if len(scores) > 1 else 0.0,
        "n_flies": int(len(scores)),
        "color": color,
    }


def cs_plus_table(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> pd.DataFrame:
    """Figure 1: each protocol's response to its own conditioned stimulus.

    Bar colour tracks the stimulus presented, as everywhere else: hexanol
    green, apple cider vinegar orange, and AIR the blue reserved for the one
    non-odorant.
    """
    vectors = _cs_plus_vectors(preds, binary_csv=binary_csv)
    rows = []
    for key, scores in vectors.items():
        cohort = COHORTS[key]
        assert cohort.cs_plus is not None  # guaranteed by _cs_plus_vectors
        rows.append(
            _bar_row(
                scores,
                label=cohort.label,
                odor=cohort.cs_plus,
                cohort=cohort.key,
                color=odor_bar_palette.odor_color(cohort.cs_plus) or cohort.color,
            )
        )
    return pd.DataFrame(rows)



def eb_table(
    preds: pd.DataFrame | dict[str, pd.DataFrame], *, binary_csv: Path | None = None
) -> pd.DataFrame:
    """Figure 2: ethyl butyrate in every cohort.

    Every bar is the same odor, so every bar takes that odor's palette colour;
    the cohort is carried by the tick label. Colouring by cohort would make the
    palette's green mean "hexanol" in one figure and "EB in hexanol-trained
    flies" in this one.
    """
    eb_color = odor_bar_palette.odor_color(EB_ODOR) or odor_bar_palette.PINK
    rows = []
    for key in EB_COHORT_ORDER:
        cohort = COHORTS[key]
        scores = first_presentation_scores(
            preds, cohort, EB_ODOR, binary_csv=binary_csv
        )
        rows.append(
            _bar_row(
                scores,
                label=cohort.label,
                odor=EB_ODOR,
                cohort=cohort.key,
                color=eb_color,
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
# A two-line "mean\n(n=..)" annotation needs roughly this much of the score
# scale. Used to decide whether a label fits beside its bar or has to flip.
_ANNOTATION_HEIGHT = 0.72
_ANNOTATION_GAP = 0.14

# Bracket geometry, in score units: clearance above the tallest whisker it
# spans, the step between stacked levels, and the room its star label needs.
_BRACKET_GAP = 0.30
_BRACKET_STEP = 0.55
_BRACKET_TEXT = 0.30


def _annotation_y(mean: float, sem: float) -> tuple[float, str]:
    """Where to put a bar's ``mean``/``n`` label, and its vertical alignment.

    Labels sit outside the error bar: below a negative bar, above a positive
    one. When the preferred side has no room left on the score scale the label
    flips to the other side rather than being clamped — clamping walks the text
    back over the whisker it was supposed to clear.
    """
    below = mean - sem - _ANNOTATION_GAP
    above = mean + sem + _ANNOTATION_GAP
    fits_below = below - _ANNOTATION_HEIGHT >= SCORE_MIN
    fits_above = above + _ANNOTATION_HEIGHT <= SCORE_MAX

    if mean < 0:
        if fits_below:
            return below, "top"
        if fits_above:
            return above, "bottom"
    else:
        if fits_above:
            return above, "bottom"
        if fits_below:
            return below, "top"

    # Neither side fits: keep the label on the scale and inside the bar, where
    # it can still be read against the fill.
    if mean >= 0:
        return max(SCORE_MIN + _ANNOTATION_HEIGHT, min(mean - _ANNOTATION_GAP, SCORE_MAX)), "top"
    return min(SCORE_MAX - _ANNOTATION_HEIGHT, max(mean + _ANNOTATION_GAP, SCORE_MIN)), "bottom"


def render_bars(
    table: pd.DataFrame,
    *,
    title: str,
    brackets: Iterable[dict] | None = None,
    bar_width: float | None = None,
) -> plt.Figure:
    """One bar per row, SEM whiskers, mean and n annotated, optional brackets."""
    rows = table.reset_index(drop=True)
    n_bars = len(rows)
    # Two fat bars on a tall canvas read as a poster; keep them narrower.
    if bar_width is None:
        bar_width = 0.45 if n_bars <= 2 else 0.6
    fig_w = max(7.0, 1.8 * n_bars + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))

    x = np.arange(n_bars)
    means = rows["mean_score"].to_numpy(float)
    sems = rows["sem_score"].to_numpy(float)

    ax.bar(
        x,
        means,
        width=bar_width,
        yerr=sems,
        color=list(rows["color"]),
        edgecolor="black",
        linewidth=0.75,
        error_kw={"ecolor": "black", "elinewidth": 1.0, "capsize": 4},
    )
    # Scores run -1..5, so the bars need a visible baseline to read against.
    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(list(rows["label"]), fontsize=10)
    ax.set_ylim(SCORE_MIN, SCORE_MAX)
    ax.set_ylabel(SCORE_Y_LABEL)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.margins(x=0.12)

    # Brackets are laid out against the whiskers first, because a tall bar's
    # own label would otherwise push the stack straight off the top of the
    # scale — with three bars and two stacked brackets there is not enough
    # headroom above a 3.2 bar for both.
    bracket_list = list(brackets or [])
    whisker_tops = [float(m + s) for m, s in zip(means, sems)]
    placed = []
    for level, bracket in enumerate(bracket_list):
        i, j = int(bracket["i"]), int(bracket["j"])
        span_top = max(whisker_tops[k] for k in range(min(i, j), max(i, j) + 1))
        y = min(span_top + _BRACKET_GAP + _BRACKET_STEP * level, SCORE_MAX - 0.25)
        placed.append((i, j, y, bracket["stars"]))

    def _blocked(bar: int, lo: float, hi: float) -> bool:
        """Does an annotation spanning ``lo``..``hi`` run into a bracket?"""
        return any(
            min(i, j) <= bar <= max(i, j) and lo <= y + _BRACKET_TEXT <= hi
            for i, j, y, _ in placed
        )

    for xi, mean, sem, n in zip(x, means, sems, rows["n_flies"]):
        y, va = _annotation_y(float(mean), float(sem))
        lo, hi = (y, y + _ANNOTATION_HEIGHT) if va == "bottom" else (y - _ANNOTATION_HEIGHT, y)
        # A bar tall enough to hold its own label gives up the space above it
        # to the bracket rather than fighting it. Inside means below the *lower*
        # whisker end, not just below the bar top, or the label lands back on
        # the error bar it was moved to avoid.
        inside = float(mean) - float(sem) - _ANNOTATION_GAP
        if _blocked(int(xi), lo, hi) and inside - _ANNOTATION_HEIGHT >= 0.0:
            y, va = inside, "top"
        ax.text(
            xi,
            y,
            f"{mean:.2f}\n(n={int(n)})",
            ha="center",
            va=va,
            fontsize=9,
        )

    for i, j, bracket_y, star in placed:
        tip_y = bracket_y - 0.15
        ax.plot(
            [x[i], x[i], x[j], x[j]],
            [tip_y, bracket_y, bracket_y, tip_y],
            color="black",
            linewidth=0.9,
            clip_on=False,
        )
        ax.text(
            (x[i] + x[j]) / 2,
            bracket_y + 0.05,
            star,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    written = []
    for suffix in formats:
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    return written


def build_figures(
    *,
    predictions_csv: Path | None = None,
    frozen_csv: Path | None = None,
    binary_csv: Path | None = None,
    out_dir: Path | None = None,
    formats: Sequence[str] = ("png", "svg"),
) -> list[Path]:
    """Write both cross-cohort figures plus their value/stat CSVs."""
    out_dir = Path(out_dir) if out_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = load_sources(predictions_csv, frozen_csv)
    written: list[Path] = []

    # -- Figure 1: the CS+ itself ------------------------------------------
    cs_plus = cs_plus_table(preds, binary_csv=binary_csv)
    cs_brackets = significant_brackets(preds, which="cs_plus", binary_csv=binary_csv)
    cs_omnibus = cs_plus_omnibus(preds, binary_csv=binary_csv)
    cs_pairs = cs_plus_pairwise(preds, binary_csv=binary_csv)
    stem = "score_bars_CSplus_across_conditioning"
    with plt.rc_context(_RC_CONTEXT):
        fig = render_bars(
            cs_plus,
            title="Response to the Conditioned Stimulus After Training",
            brackets=cs_brackets,
        )
        written += _save(fig, out_dir, stem, formats)
        plt.close(fig)
    cs_plus.drop(columns=["color"]).to_csv(
        out_dir / f"{stem}.csv", index=False, float_format="%.6f"
    )
    written.append(out_dir / f"{stem}.csv")
    cs_pairs.drop(columns=["i", "j"]).to_csv(
        out_dir / f"{stem}_stats.csv", index=False, float_format="%.6f"
    )
    written.append(out_dir / f"{stem}_stats.csv")
    print(
        f"[INFO] CS+ across {cs_omnibus['k']} protocols: "
        f"Kruskal-Wallis H={cs_omnibus['H']:.3f} p={cs_omnibus['p']:.3g} "
        f"{stars(cs_omnibus['p']) or 'n.s.'}; {len(cs_brackets)} significant pair(s)"
    )
    for row in cs_pairs.itertuples(index=False):
        print(
            f"[INFO]   {row.left:14s} vs {row.right:14s} "
            f"p={row.p:.3g} p_holm={row.p_holm:.3g} {row.stars or 'n.s.'}"
        )

    # -- Figure 2: ethyl butyrate across cohorts ---------------------------
    eb = eb_table(preds, binary_csv=binary_csv)
    eb_brackets = significant_brackets(preds, which="eb", binary_csv=binary_csv)
    omnibus = eb_omnibus(preds, binary_csv=binary_csv)
    pairs = eb_pairwise(preds, binary_csv=binary_csv)
    stem = "score_bars_EthylButyrate_across_conditioning"
    with plt.rc_context(_RC_CONTEXT):
        fig = render_bars(
            eb,
            title="Ethyl Butyrate Response Across Conditioning Protocols",
            brackets=eb_brackets,
        )
        written += _save(fig, out_dir, stem, formats)
        plt.close(fig)
    eb.drop(columns=["color"]).to_csv(
        out_dir / f"{stem}.csv", index=False, float_format="%.6f"
    )
    written.append(out_dir / f"{stem}.csv")
    pairs.drop(columns=["i", "j"]).to_csv(
        out_dir / f"{stem}_stats.csv", index=False, float_format="%.6f"
    )
    written.append(out_dir / f"{stem}_stats.csv")
    print(
        f"[INFO] Ethyl butyrate across {omnibus['k']} cohorts: "
        f"Kruskal-Wallis H={omnibus['H']:.3f} p={omnibus['p']:.4f} "
        f"{stars(omnibus['p']) or 'n.s.'}; {len(eb_brackets)} significant pair(s)"
    )
    for row in pairs.itertuples(index=False):
        print(
            f"[INFO]   {row.left:14s} vs {row.right:14s} "
            f"p={row.p:.4f} p_holm={row.p_holm:.4f} {row.stars or 'n.s.'}"
        )

    for path in written:
        print(f"[INFO] wrote {path}")
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--frozen-csv", type=Path, default=None)
    parser.add_argument("--binary-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        default=None,
        help="Output format, repeatable. Default: png and svg.",
    )
    args = parser.parse_args(argv)
    build_figures(
        predictions_csv=args.predictions_csv,
        frozen_csv=args.frozen_csv,
        binary_csv=args.binary_csv,
        out_dir=args.out_dir,
        formats=tuple(args.formats) if args.formats else ("png", "svg"),
    )


if __name__ == "__main__":
    main()
