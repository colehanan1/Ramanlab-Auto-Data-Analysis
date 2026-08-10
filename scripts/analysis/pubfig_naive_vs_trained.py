#!/usr/bin/env python3
"""Naive vs trained vs control for one odorant at one concentration.

Two published figures each answer half the question:

* ``pubfig_mean_score_train_vs_ctrl_*`` — trained vs control within a cohort,
  every odor in the panel, but no untrained baseline for the odorant itself.
* ``randompanel_conc_score_comparison`` — how naive flies respond to each
  odorant at 10 / 1 / 0.1%, but nothing that was ever conditioned.

This one puts all three cohorts on one axes for a single odorant at a single
concentration, so "conditioning moved the response away from naive" is a
comparison the reader can actually see:

* left panel  — mean PER score per fly (+/- SEM), Kruskal-Wallis omnibus with
  Holm-corrected Mann-Whitney post-hoc;
* right panel — response rate, the fraction of flies scoring >= 2 (+/- Wilson
  95% CI), Fisher-Freeman-Halton omnibus with Holm-corrected Fisher post-hoc.

Only the **first presentation** of the odorant counts, in every cohort: the
trained bar is then the immediate post-training test and the naive bar is that
odorant's first appearance in the random panel. Both p-value families are
corrected within their own panel; brackets carry stars only, p-values go to the
sidecar CSV.

Run::

    python scripts/analysis/pubfig_naive_vs_trained.py            # both figures
    python scripts/analysis/pubfig_naive_vs_trained.py --only 3oct
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

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
from scripts.analysis.randompanel_conc_comparison import (  # noqa: E402
    _wilson_ci,
    fisher_freeman_halton_mc,
    holm_adjust,
)

FIGURES_DIR = Path("/home/ramanlab/Documents/cole/Results/Figures")
PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv"
)
CONFIG = REPO_ROOT / "config" / "config_new.yaml"
GENOTYPE = "GR5a-Old"

COHORTS = ("Naive", "Trained", "Control")
REACTION_BOUNDARY = 2  # score >= 2 is a reaction (matches score_summary)
SCORE_MIN, SCORE_MAX = -1.0, 5.0
SCORE_Y_LABEL = "Mean PER Score"
RATE_Y_LABEL = "Responding flies (%)"
ALPHA = 0.05

NAIVE_FACE = "#ffffff"     # open bar: same odor, never conditioned
NAIVE_HATCH = "///"

# "3-Octanol (0.1%) 2" -> "3-Octanol". Same job as odor_bar_palette's
# normalise_odor, but the display case is kept because this is a figure label.
_LABEL_NOISE = re.compile(r"\s*\([^)]*\)\s*|\s+\d+\s*$")
_BATCH_RE = re.compile(r"batch_(\d+)")


@dataclass(frozen=True)
class Comparison:
    """One figure: an odorant at a concentration, across three cohorts."""

    odor: str
    concentration: str
    naive_dataset: str
    train_dataset: str
    control_dataset: str
    batch: int | None = None
    cohort_note: str = ""

    @property
    def title(self) -> str:
        batch = f", batch {self.batch}" if self.batch is not None else ""
        return f"{self.odor} ({self.concentration}) — naive vs trained vs control{batch}"

    @property
    def stem(self) -> str:
        odor = self.odor.replace(" ", "-")
        conc = self.concentration.replace("%", "pct").replace(".", "-")
        batch = f"_batch{self.batch}" if self.batch is not None else ""
        return f"pubfig_naive_vs_trained_{odor}_{conc}{batch}"


COMPARISONS: dict[str, Comparison] = {
    "3oct": Comparison(
        odor="3-Octanol",
        concentration="0.1%",
        naive_dataset="RandomPanel-24-0.1",
        train_dataset="3OCT-Training-24-0.1",
        control_dataset="3OCT-Control-24-0.1",
    ),
    "eb": Comparison(
        odor="Ethyl Butyrate",
        concentration="1%",
        naive_dataset="RandomPanel-24-1",
        train_dataset="EB-Training-24-1",
        control_dataset="EB-Control-24-1",
        batch=1,
    ),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def base_odor(label: str) -> str:
    """Odorant behind a display label, concentration and presentation stripped."""
    name = str(label).strip()
    while True:
        stripped = _LABEL_NOISE.sub(" ", name).strip()
        if stripped == name:
            return name
        name = stripped


def _batch_of(fly: str) -> int | None:
    m = _BATCH_RE.search(str(fly))
    return int(m.group(1)) if m else None


def cohort_scores(
    df: pd.DataFrame,
    dataset: str,
    odor: str,
    *,
    genotype: str = GENOTYPE,
    batch: int | None = None,
) -> pd.DataFrame:
    """One row per fly: its score on the **first** presentation of ``odor``.

    ``df`` is ``score_summary._load_scores`` output. The naive panel labels the
    odorant plainly ("3-Octanol") while a conditioned cohort tags it with the
    concentration ("3-Octanol (0.1%)"), so both sides are matched on the base
    name.
    """
    frame = df[df["dataset_canon"].astype(str) == str(dataset)]
    if "fly_type" in frame.columns and genotype:
        frame = frame[frame["fly_type"].astype(str).str.strip() == genotype]
    if frame.empty:
        return pd.DataFrame(columns=["fly", "fly_number", "score"])

    target = base_odor(odor).casefold()
    frame = frame[
        frame["odor_display"].astype(str).map(lambda s: base_odor(s).casefold()) == target
    ]
    if "occurrence" in frame.columns:
        frame = frame[pd.to_numeric(frame["occurrence"], errors="coerce") == 1]
    if batch is not None:
        frame = frame[frame["fly"].map(_batch_of) == batch]
    if frame.empty:
        return pd.DataFrame(columns=["fly", "fly_number", "score"])

    # One value per fly even if a rig logged the same trial twice.
    out = (
        frame.groupby(["fly", "fly_number"], as_index=False)["score"]
        .mean()
        .sort_values(["fly", "fly_number"])
        .reset_index(drop=True)
    )
    return out[["fly", "fly_number", "score"]]


def load_groups(
    df: pd.DataFrame, comparison: Comparison, *, genotype: str = GENOTYPE
) -> dict[str, np.ndarray]:
    """Per-fly scores for the three cohorts of one comparison."""
    datasets = {
        "Naive": (comparison.naive_dataset, None),
        "Trained": (comparison.train_dataset, comparison.batch),
        "Control": (comparison.control_dataset, comparison.batch),
    }
    return {
        cohort: cohort_scores(
            df, dataset, comparison.odor, genotype=genotype, batch=batch
        )["score"].to_numpy(dtype=float)
        for cohort, (dataset, batch) in datasets.items()
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _pairs() -> list[tuple[str, str]]:
    return list(combinations(COHORTS, 2))


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def score_stats(groups: dict[str, np.ndarray]) -> dict:
    """Mean/SEM per cohort, Kruskal-Wallis omnibus, Holm-corrected Mann-Whitney."""
    clean = {c: _finite(groups.get(c, [])) for c in COHORTS}
    means, sems, ns = {}, {}, {}
    for cohort, values in clean.items():
        ns[cohort] = int(values.size)
        means[cohort] = float(values.mean()) if values.size else float("nan")
        sems[cohort] = (
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )

    usable = [v for v in clean.values() if v.size > 0]
    omnibus = float("nan")
    if len(usable) >= 2:
        try:
            omnibus = float(kruskal(*usable).pvalue)
        except ValueError:  # every observation identical -> no rank variation
            omnibus = 1.0

    raw = []
    for a, b in _pairs():
        va, vb = clean[a], clean[b]
        if va.size == 0 or vb.size == 0:
            raw.append(float("nan"))
        elif np.array_equal(np.sort(va), np.sort(vb)) and va.size == vb.size:
            raw.append(1.0)
        else:
            raw.append(float(mannwhitneyu(va, vb, alternative="two-sided").pvalue))
    adj = holm_adjust(raw)
    pairwise = {
        pair: {"p_raw": float(r), "p_adj": float(a)}
        for pair, r, a in zip(_pairs(), raw, adj)
    }
    return {
        "means": means, "sems": sems, "n": ns,
        "omnibus_p": omnibus, "pairwise": pairwise,
    }


def rate_stats(groups: dict[str, np.ndarray]) -> dict:
    """Response rate per cohort with Wilson CI, FFH omnibus, Holm-corrected Fisher."""
    clean = {c: _finite(groups.get(c, [])) for c in COHORTS}
    k, n, rate, ci = {}, {}, {}, {}
    for cohort, values in clean.items():
        n[cohort] = int(values.size)
        k[cohort] = int((values >= REACTION_BOUNDARY).sum())
        rate[cohort] = float(k[cohort] / n[cohort]) if n[cohort] else float("nan")
        ci[cohort] = _wilson_ci(k[cohort], n[cohort])

    cats, grps = [], []
    for idx, cohort in enumerate(COHORTS):
        values = clean[cohort]
        cats.extend((values >= REACTION_BOUNDARY).astype(int).tolist())
        grps.extend([idx] * values.size)
    omnibus = (
        fisher_freeman_halton_mc(np.asarray(cats), np.asarray(grps))
        if len(set(grps)) > 1 and cats
        else float("nan")
    )

    raw = []
    for a, b in _pairs():
        if n[a] == 0 or n[b] == 0:
            raw.append(float("nan"))
            continue
        table = [[k[a], n[a] - k[a]], [k[b], n[b] - k[b]]]
        raw.append(float(fisher_exact(table, alternative="two-sided")[1]))
    adj = holm_adjust(raw)
    pairwise = {
        pair: {"p_raw": float(r), "p_adj": float(a)}
        for pair, r, a in zip(_pairs(), raw, adj)
    }
    return {
        "k": k, "n": n, "rate": rate, "ci": ci,
        "omnibus_p": float(omnibus), "pairwise": pairwise,
    }


def _stars(p: float) -> str:
    """Stars for a p-value, empty when it is not significant."""
    if not np.isfinite(p) or p >= ALPHA:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def stats_rows(comparison: Comparison, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Long-form stats table: omnibus + every pair, for both metrics."""
    score = score_stats(groups)
    rate = rate_stats(groups)
    rows: list[dict] = []
    common = {
        "odor": comparison.odor,
        "concentration": comparison.concentration,
        "batch": comparison.batch,
        "naive_dataset": comparison.naive_dataset,
        "train_dataset": comparison.train_dataset,
        "control_dataset": comparison.control_dataset,
    }
    for metric, stats, value_of in (
        ("mean_score", score, lambda c: score["means"][c]),
        ("response_rate", rate, lambda c: rate["rate"][c]),
    ):
        rows.append(
            {
                **common,
                "metric": metric,
                "comparison": "omnibus",
                "group_a": "", "group_b": "",
                "value_a": np.nan, "value_b": np.nan,
                "n_a": sum(stats["n"].values()), "n_b": np.nan,
                "p_raw": stats["omnibus_p"], "p_adj": stats["omnibus_p"],
                "stars": _stars(stats["omnibus_p"]),
            }
        )
        for (a, b), p in stats["pairwise"].items():
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "comparison": f"{a} vs {b}",
                    "group_a": a, "group_b": b,
                    "value_a": value_of(a), "value_b": value_of(b),
                    "n_a": stats["n"][a], "n_b": stats["n"][b],
                    "p_raw": p["p_raw"], "p_adj": p["p_adj"],
                    "stars": _stars(p["p_adj"]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _bar_styles(odor: str) -> dict[str, dict]:
    """Face/edge/hatch per cohort: open naive, solid trained, grey control."""
    color = pal.odor_color(odor) or pal.TRAIN_COLOR
    return {
        "Naive": {
            "facecolor": NAIVE_FACE, "edgecolor": color,
            "hatch": NAIVE_HATCH, "linewidth": 1.4,
        },
        "Trained": {"facecolor": color, "edgecolor": "black", "linewidth": 0.9},
        "Control": {
            "facecolor": pal.CTRL_COLOR, "edgecolor": "black", "linewidth": 0.9,
        },
    }


def _draw_brackets(ax, pairwise: dict, tops: dict[str, float], *, span: float) -> None:
    """Star brackets for significant pairs, packed from the shortest span up."""
    drawn = [
        (a, b, _stars(p["p_adj"]))
        for (a, b), p in pairwise.items()
        if _stars(p["p_adj"])
    ]
    if not drawn:
        return
    index = {c: i for i, c in enumerate(COHORTS)}
    drawn.sort(key=lambda t: abs(index[t[1]] - index[t[0]]))

    step = span * 0.11
    level = max(tops.values()) + step * 0.55
    for a, b, stars in drawn:
        xa, xb = index[a], index[b]
        bar_top = max(tops[a], tops[b])
        y = max(level, bar_top + step * 0.45)
        ax.plot(
            [xa, xa, xb, xb],
            [y, y + step * 0.22, y + step * 0.22, y],
            color="black", linewidth=1.0, clip_on=False,
        )
        ax.text(
            (xa + xb) / 2.0, y + step * 0.28, stars,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
        level = y + step * 0.95


def _panel(ax, comparison: Comparison, groups: dict[str, np.ndarray], *, metric: str):
    styles = _bar_styles(comparison.odor)
    x = np.arange(len(COHORTS))

    if metric == "score":
        stats = score_stats(groups)
        heights = [stats["means"][c] for c in COHORTS]
        err = [[stats["sems"][c] for c in COHORTS], [stats["sems"][c] for c in COHORTS]]
        labels = [
            "" if not np.isfinite(h) else f"{h:.2f}" for h in heights
        ]
        y_label, y_lim = SCORE_Y_LABEL, (SCORE_MIN, SCORE_MAX)
        tops = {
            c: (stats["means"][c] + stats["sems"][c]) if np.isfinite(stats["means"][c]) else 0.0
            for c in COHORTS
        }
        n_by_cohort = stats["n"]
    else:
        stats = rate_stats(groups)
        heights = [100.0 * stats["rate"][c] if np.isfinite(stats["rate"][c]) else np.nan
                   for c in COHORTS]
        lo = [
            max(0.0, (stats["rate"][c] - stats["ci"][c][0]) * 100.0)
            if np.isfinite(stats["rate"][c]) else 0.0
            for c in COHORTS
        ]
        hi = [
            max(0.0, (stats["ci"][c][1] - stats["rate"][c]) * 100.0)
            if np.isfinite(stats["rate"][c]) else 0.0
            for c in COHORTS
        ]
        err = [lo, hi]
        labels = [
            "" if not np.isfinite(h) else f"{h:.0f}%" for h in heights
        ]
        y_label, y_lim = RATE_Y_LABEL, (0.0, 100.0)
        tops = {
            c: (heights[i] + hi[i]) if np.isfinite(heights[i]) else 0.0
            for i, c in enumerate(COHORTS)
        }
        n_by_cohort = stats["n"]

    for i, cohort in enumerate(COHORTS):
        ax.bar(
            x[i], 0.0 if not np.isfinite(heights[i]) else heights[i],
            width=0.62, zorder=2, **styles[cohort],
        )
    ax.errorbar(
        x, np.nan_to_num(heights, nan=0.0), yerr=err, fmt="none",
        ecolor="black", elinewidth=1.0, capsize=4, zorder=3,
    )
    for i, text in enumerate(labels):
        if text:
            ax.text(
                x[i], tops[i if isinstance(tops, list) else COHORTS[i]] + (y_lim[1] - y_lim[0]) * 0.02,
                text, ha="center", va="bottom", fontsize=9,
            )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n_by_cohort[c]})" for c in COHORTS])
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_lim)
    ax.set_xlim(-0.65, len(COHORTS) - 0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.85", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    _draw_brackets(ax, stats["pairwise"], tops, span=y_lim[1] - y_lim[0])
    return stats


def render_comparison(
    comparison: Comparison, groups: dict[str, np.ndarray]
) -> plt.Figure:
    """The two-panel figure: mean PER score and response rate."""
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
        score = _panel(axes[0], comparison, groups, metric="score")
        rate = _panel(axes[1], comparison, groups, metric="rate")
        for ax, letter in zip(axes, "AB"):
            ax.set_title(letter, loc="left", fontsize=11, fontweight="bold")
        fig.suptitle(comparison.title, fontsize=11, fontweight="bold", y=0.99)
        note = (
            "First presentation only, one trial per fly. A: mean ± SEM, "
            "Kruskal–Wallis omnibus with Holm-corrected Mann–Whitney post-hoc. "
            "B: flies scoring ≥ 2, ± Wilson 95% CI, Fisher–Freeman–Halton omnibus "
            "with Holm-corrected Fisher post-hoc. Brackets: * p<0.05, ** p<0.01, "
            "*** p<0.001; p-values in the sidecar CSV."
        )
        if comparison.cohort_note:
            note = f"{comparison.cohort_note} {note}"
        fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=6.5,
                 color="0.35", wrap=True)
        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig._pubfig_stats = {"score": score, "rate": rate}  # noqa: SLF001 (debug aid)
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_predictions(predictions_csv: Path, config: Path | None) -> pd.DataFrame:
    from scripts.analysis import pubfig_score_train_vs_control as pubfig
    from scripts.analysis import score_summary as ss

    pubfig._apply_config(config)
    return ss._load_scores(predictions_csv, threshold=None, flagged_flies_csv="")


def build(
    key: str,
    *,
    predictions_csv: Path = PREDICTIONS_CSV,
    config: Path | None = CONFIG,
    figures_dir: Path = FIGURES_DIR,
    genotype: str = GENOTYPE,
    df: pd.DataFrame | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Render one comparison; returns the PNG path and its stats table."""
    comparison = COMPARISONS[key]
    frame = _load_predictions(predictions_csv, config) if df is None else df
    groups = load_groups(frame, comparison, genotype=genotype)
    for cohort, values in groups.items():
        if values.size == 0:
            raise SystemExit(
                f"{comparison.stem}: no {cohort} flies for {comparison.odor} "
                f"({comparison.concentration}) — check the dataset names."
            )
    fig = render_comparison(comparison, groups)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{comparison.stem}.png"
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(figures_dir / f"{comparison.stem}{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    rows = stats_rows(comparison, groups)
    rows.to_csv(figures_dir / f"{comparison.stem}_stats.csv", index=False)
    return png, rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(COMPARISONS), default=None,
                        help="Render one comparison instead of all of them.")
    parser.add_argument("--predictions-csv", type=Path, default=PREDICTIONS_CSV)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--genotype", default=GENOTYPE)
    args = parser.parse_args(argv)

    keys = [args.only] if args.only else sorted(COMPARISONS)
    df = _load_predictions(args.predictions_csv, args.config)
    for key in keys:
        png, rows = build(
            key,
            predictions_csv=args.predictions_csv,
            config=args.config,
            figures_dir=args.figures_dir,
            genotype=args.genotype,
            df=df,
        )
        print(f"[SAVED] {png}")
        for _, row in rows[rows["comparison"] != "omnibus"].iterrows():
            print(
                f"  {row['metric']:<14} {row['comparison']:<20} "
                f"p_adj={row['p_adj']:.4f} {row['stars']}"
            )


if __name__ == "__main__":
    main()
