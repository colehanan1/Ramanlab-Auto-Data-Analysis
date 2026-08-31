"""Does conditioning vigor predict learning?

Pairs each fly's proboscis extension *during conditioning* (AUC-During over the
trained-odor training trials) with how it responded to that same odor at test,
and emits two figures per cohort role:

``training_vs_learning_scatter_<role>.png``
    One row per testing presentation of the trained odor. Left: training AUC vs
    testing score, with a Spearman fit. Right: training AUC split by whether
    the fly reacted (PER) at test, with Mann-Whitney.

``training_vs_learning_trajectory_<role>.png``
    Left: mean AUC across the conditioning trials, learners vs non-learners.
    Right: each fly's AUC slope over those trials vs its mean testing score —
    i.e. does habituating (or sensitising) during training predict learning?

Both roles (training cohort and its control) get the same treatment, since the
control's conditioning trials are odor-only and act as the baseline.

The two inputs are joined on ``(fly, fly_number)``: AUC comes from the training
wide table, scores/PER from the predictions CSV. Flies missing either side are
dropped rather than carried as NaN.

Usage::

    python scripts/analysis/training_vs_learning.py \
        --training-wide-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/all_envelope_rows_wide_combined_base_training.parquet \
        --predictions-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv \
        --train-dataset EB-Training-24-1 --control-dataset EB-Control-24-1 \
        --odor-short EB \
        --flagged-flies-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/flagged-flys-truth.csv \
        --out-dir /home/ramanlab/Documents/cole/Results/Figures/EB-24-1_training_vs_learning
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.analysis.traces import read_wide_table  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _normalise_fly_columns,
    compute_non_reactive_flags,
)

LOGGER = logging.getLogger("training_vs_learning")

DPI = 300
LEARNER_THRESHOLD = 1.0

# Learners green, non-learners red, per request. Note this is the red/green
# pairing the score palette deliberately avoids: a protanope sees the two as
# near-identical. Shape carries no backup signal here (every fly is a circle),
# so the group is legible only by colour.
LEARNER_COLOR = "#1b7837"
NONLEARNER_COLOR = "#cb181d"

_RC_CONTEXT = {
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}
_DATE_RE = re.compile(r"^([a-z]+)_(\d+)", re.IGNORECASE)
_TRIAL_RE = re.compile(r"^(?:pretest|training|testing)_(\d+)_(.+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def learner_mask(summary: pd.DataFrame, threshold: float) -> np.ndarray:
    """Which flies learned: mean testing score strictly above ``threshold``.

    Strict ``>`` matches the trajectory legend ("> 1.0" / "<= 1.0"), so a fly
    sitting exactly on the line lands in the same group in both places. A NaN
    score compares False and joins the non-learners.
    """
    return (
        pd.to_numeric(summary["mean_score"], errors="coerce")
        .gt(threshold)
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def point_colors(summary: pd.DataFrame, threshold: float) -> list[str]:
    """Per-fly scatter colour: green for learners, red for non-learners."""
    return [
        LEARNER_COLOR if is_learner else NONLEARNER_COLOR
        for is_learner in learner_mask(summary, threshold)
    ]


def _split_trial(label: str) -> tuple[int, str] | None:
    m = _TRIAL_RE.match(str(label).strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2).lower()


def trained_odor_token(training_df: pd.DataFrame) -> str:
    """Odor token of the conditioning trials (they are all the trained odor)."""
    tokens = [
        parsed[1]
        for parsed in (_split_trial(v) for v in training_df["trial_label"])
        if parsed is not None
    ]
    if not tokens:
        raise ValueError("No parseable training trial labels")
    return pd.Series(tokens).value_counts().idxmax()


def trained_presentations(testing_df: pd.DataFrame, token: str) -> list[int]:
    """Trial numbers on which ``token`` was presented at test, in order."""
    nums = {
        parsed[0]
        for parsed in (_split_trial(v) for v in testing_df["trial_label"])
        if parsed is not None and parsed[1] == token
    }
    return sorted(nums)


def slope_per_fly(auc_by_trial: pd.Series) -> float:
    """OLS slope of AUC-During against trial number; NaN with under 2 trials."""
    clean = auc_by_trial.dropna()
    if len(clean) < 2:
        return float("nan")
    x = np.asarray(clean.index, dtype=float)
    y = np.asarray(clean.to_numpy(), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------


def build_fly_summary(
    training_df: pd.DataFrame, testing_df: pd.DataFrame, dataset: str
) -> pd.DataFrame:
    """One row per fly: conditioning vigor plus its testing outcome."""
    train = training_df[training_df["dataset"].astype(str).str.strip() == dataset].copy()
    test = testing_df[testing_df["dataset"].astype(str).str.strip() == dataset].copy()
    if train.empty or test.empty:
        return pd.DataFrame()

    token = trained_odor_token(train)
    presentations = trained_presentations(test, token)

    train["_trial"] = [
        (p[0] if (p := _split_trial(v)) else np.nan) for v in train["trial_label"]
    ]
    train = train.dropna(subset=["_trial"])
    train["AUC-During"] = pd.to_numeric(train["AUC-During"], errors="coerce")

    rows = []
    for (fly, fly_number), grp in train.groupby(["fly", "fly_number"], sort=True):
        auc = grp.groupby("_trial")["AUC-During"].mean().sort_index()
        rows.append({
            "fly": fly,
            "fly_number": fly_number,
            "mean_auc": float(auc.mean()) if len(auc) else np.nan,
            "slope": slope_per_fly(auc),
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    test["_trial"] = [
        (p[0] if (p := _split_trial(v)) else np.nan) for v in test["trial_label"]
    ]
    for trial in presentations:
        sub = test[test["_trial"] == trial]
        scores = (
            sub.groupby(["fly", "fly_number"])["score"].mean().rename(f"score_{trial}")
        )
        pers = (
            sub.groupby(["fly", "fly_number"])["prediction"]
            .max()
            .rename(f"per_{trial}")
        )
        summary = summary.merge(
            pd.concat([scores, pers], axis=1).reset_index(),
            on=["fly", "fly_number"],
            how="inner",
        )

    if summary.empty:
        return summary
    score_cols = [f"score_{t}" for t in presentations]
    summary["mean_score"] = summary[score_cols].mean(axis=1)
    summary.attrs["presentations"] = presentations
    summary.attrs["odor_token"] = token
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _stat_box(ax, text: str) -> None:
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6"),
    )


def _scatter_with_fit(ax, x, y, colors) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    colors = np.asarray(colors, dtype=object)
    ok = np.isfinite(x) & np.isfinite(y)
    rho, p = (np.nan, np.nan)
    if ok.sum() >= 3:
        rho, p = spearmanr(x[ok], y[ok])
    ax.scatter(x[ok], y[ok], s=55, c=list(colors[ok]), edgecolor="black",
               linewidth=0.6, zorder=3)
    if ok.sum() >= 2:
        fit = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 100)
        ax.plot(xs, np.polyval(fit, xs), linestyle="--", color="0.35", linewidth=1.2)
    return float(rho), float(p)


def _per_box(ax, summary: pd.DataFrame, per_col: str, colors) -> float:
    reacted = summary[per_col].fillna(0).astype(float) >= 0.5
    groups = [
        summary.loc[~reacted, "mean_auc"].dropna().to_numpy(float),
        summary.loc[reacted, "mean_auc"].dropna().to_numpy(float),
    ]
    ax.boxplot(
        [g for g in groups], positions=[0, 1], widths=0.55, showfliers=False,
        medianprops=dict(color="black"), boxprops=dict(color="black"),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
    )
    rng = np.random.default_rng(0)
    colors = np.asarray(colors, dtype=object)
    for pos, mask in ((0, ~reacted), (1, reacted)):
        sub = summary.loc[mask]
        jitter = rng.uniform(-0.13, 0.13, len(sub))
        ax.scatter(pos + jitter, sub["mean_auc"], s=45,
                   c=list(colors[mask.to_numpy()]), edgecolor="black",
                   linewidth=0.5, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"no PER\n(n={int((~reacted).sum())})", f"PER\n(n={int(reacted.sum())})"
    ])
    if groups[0].size and groups[1].size:
        return float(mannwhitneyu(groups[0], groups[1], alternative="two-sided").pvalue)
    return float("nan")


def plot_scatter_figure(
    summary: pd.DataFrame, *, role: str, dataset: str, odor_short: str,
    threshold: float,
) -> tuple[plt.Figure, list[dict]]:
    presentations = summary.attrs["presentations"]
    colors = point_colors(summary, threshold)
    n_learn = int(learner_mask(summary, threshold).sum())
    n_non = len(summary) - n_learn
    stats: list[dict] = []

    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(
            len(presentations), 2, figsize=(14, 5.4 * len(presentations)), squeeze=False
        )
        for i, trial in enumerate(presentations):
            label = f"{odor_short} {i + 1}"
            ax = axes[i][0]
            rho, p = _scatter_with_fit(
                ax, summary["mean_auc"], summary[f"score_{trial}"], colors
            )
            n = int(np.isfinite(summary[f"score_{trial}"].to_numpy(float)).sum())
            _stat_box(ax, f"Spearman ρ={rho:.2f}\np={p:.3f}  (n={n})")
            ax.set_xlabel(f"Training AUC-During (mean over {odor_short} trials)")
            ax.set_ylabel(f"Testing {label} score (0–5)")
            ax.set_title(f"{role}: training extension vs {label} SCORE",
                         fontsize=12, weight="bold")
            ax.grid(alpha=0.25, linestyle=":", linewidth=0.6)
            if i == 0:
                handles = [
                    plt.Line2D([], [], marker="o", linestyle="", color=LEARNER_COLOR,
                               markeredgecolor="black",
                               label=f"learners ({odor_short} score > {threshold})"
                                     f"  (n={n_learn})"),
                    plt.Line2D([], [], marker="o", linestyle="", color=NONLEARNER_COLOR,
                               markeredgecolor="black",
                               label=f"non-learners (\u2264 {threshold})  (n={n_non})"),
                ]
                ax.legend(handles=handles, loc="lower right", fontsize=9)

            ax = axes[i][1]
            mw = _per_box(ax, summary, f"per_{trial}", colors)
            _stat_box(ax, f"Mann-Whitney\np={mw:.3f}")
            ax.set_ylabel("Training AUC-During")
            ax.set_title(f"{role}: training AUC by {label} PER (reacted?)",
                         fontsize=12, weight="bold")
            ax.grid(axis="y", alpha=0.25, linestyle=":", linewidth=0.6)

            stats.append({
                "label": label,
                "trial": int(trial),
                "n": n,
                "spearman_rho": rho,
                "spearman_p": p,
                "mannwhitney_p": mw,
            })

        fig.suptitle(
            f"Does training proboscis-extension predict {odor_short} learning?"
            f"  —  {role} ({dataset} protocol)",
            fontsize=15, weight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, stats


def plot_trajectory_figure(
    training_df: pd.DataFrame, summary: pd.DataFrame, *, role: str, dataset: str,
    odor_short: str, threshold: float,
) -> tuple[plt.Figure, dict]:
    train = training_df[training_df["dataset"].astype(str).str.strip() == dataset].copy()
    train["_trial"] = [
        (p[0] if (p := _split_trial(v)) else np.nan) for v in train["trial_label"]
    ]
    train["AUC-During"] = pd.to_numeric(train["AUC-During"], errors="coerce")
    keys = summary.set_index(["fly", "fly_number"])["mean_score"]
    train = train.join(
        keys.rename("_mean_score"), on=["fly", "fly_number"], how="inner"
    )
    learner = train["_mean_score"] > threshold

    colors = point_colors(summary, threshold)

    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax = axes[0]
        for mask, color, name in (
            (learner, LEARNER_COLOR, f"learners ({odor_short} score > {threshold})"),
            (~learner, NONLEARNER_COLOR, f"non-learners ({odor_short} score ≤ {threshold})"),
        ):
            sub = train[mask]
            if sub.empty:
                continue
            n_flies = sub[["fly", "fly_number"]].drop_duplicates().shape[0]
            grouped = sub.groupby("_trial")["AUC-During"]
            mean, sem = grouped.mean(), grouped.sem().fillna(0.0)
            ax.errorbar(
                mean.index, mean.to_numpy(), yerr=sem.to_numpy(), marker="o",
                capsize=4, color=color, linewidth=1.8,
                label=f"{name}  (n={n_flies})",
            )
        ax.set_xlabel(f"Training trial ({odor_short} presentation #)")
        ax.set_ylabel("Mean AUC-During ± SEM")
        ax.set_title(f"{role}: extension across training trials", fontsize=12, weight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(alpha=0.25, linestyle=":", linewidth=0.6)

        ax = axes[1]
        rho, p = _scatter_with_fit(
            ax, summary["slope"], summary["mean_score"], colors
        )
        n = int(np.isfinite(summary["slope"].to_numpy(float)).sum())
        _stat_box(ax, f"Spearman ρ={rho:.2f}\np={p:.3f}  (n={n})")
        ax.axvline(0.0, color="0.5", linestyle=":", linewidth=1.0)
        n_trials = int(train["_trial"].max()) if len(train) else 0
        ax.set_xlabel(
            f"Training AUC-During slope (trials 1→{n_trials})\n"
            "(+ = rising, − = habituating)"
        )
        ax.set_ylabel(f"Mean testing {odor_short} score "
                      f"({odor_short}1,{odor_short}2)")
        ax.set_title(f"{role}: extension trend vs learning", fontsize=12, weight="bold")
        ax.grid(alpha=0.25, linestyle=":", linewidth=0.6)

        fig.suptitle(
            f"Across-trial training dynamics vs {odor_short} learning — {role}",
            fontsize=15, weight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    return fig, {
        "slope_spearman_rho": float(rho),
        "slope_spearman_p": float(p),
        "n_learners": int(summary["mean_score"].gt(threshold).sum()),
        "n_non_learners": int(summary["mean_score"].le(threshold).sum()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_inputs(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = read_wide_table(args.training_wide_csv)
    keep = [c for c in ("dataset", "fly", "fly_number", "trial_type", "trial_label",
                        "AUC-During") if c in training.columns]
    training = _normalise_fly_columns(training[keep].copy())
    if "trial_type" in training.columns:
        training = training[
            training["trial_type"].astype(str).str.strip() == "training"
        ]

    testing = pd.read_csv(args.predictions_csv)
    testing = _normalise_fly_columns(testing)
    if "trial_type" in testing.columns:
        testing = testing[testing["trial_type"].astype(str).str.strip() == "testing"]

    if args.flagged_flies_csv:
        for name, frame in (("training", training), ("testing", testing)):
            flagged = compute_non_reactive_flags(
                frame, flagged_flies_csv=args.flagged_flies_csv
            )
            if flagged.any():
                LOGGER.info("Dropping %d flagged %s rows", int(flagged.sum()), name)
                if name == "training":
                    training = frame.loc[~flagged].copy()
                else:
                    testing = frame.loc[~flagged].copy()
    return training, testing


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--training-wide-csv", type=Path, required=True,
                   help="all_envelope_rows_wide_combined_base_training CSV/Parquet.")
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--train-dataset", required=True)
    p.add_argument("--control-dataset", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--odor-short", default="",
                   help="Short trained-odor label for titles (e.g. EB). "
                        "Defaults to the dataset prefix.")
    p.add_argument("--flagged-flies-csv", type=str, default="")
    p.add_argument("--learner-threshold", type=float, default=LEARNER_THRESHOLD)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    training, testing = _load_inputs(args)
    odor_short = args.odor_short or str(args.train_dataset).split("-")[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    for role, dataset in (("Training", args.train_dataset),
                          ("Control", args.control_dataset)):
        summary = build_fly_summary(training, testing, dataset)
        if summary.empty:
            LOGGER.warning("No joinable flies for %s — skipping", dataset)
            continue
        LOGGER.info(
            "%s (%s): %d flies, presentations %s",
            role, dataset, len(summary), summary.attrs["presentations"],
        )

        fig, pres_stats = plot_scatter_figure(
            summary, role=role, dataset=dataset, odor_short=odor_short,
            threshold=args.learner_threshold,
        )
        out = args.out_dir / f"training_vs_learning_scatter_{role.lower()}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved %s", out)

        fig, traj_stats = plot_trajectory_figure(
            training, summary, role=role, dataset=dataset, odor_short=odor_short,
            threshold=args.learner_threshold,
        )
        out = args.out_dir / f"training_vs_learning_trajectory_{role.lower()}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved %s", out)

        meta[role.lower()] = {
            "dataset": dataset,
            "n_flies": int(len(summary)),
            "odor_token": summary.attrs["odor_token"],
            "learner_threshold": float(args.learner_threshold),
            "presentations": pres_stats,
            **traj_stats,
        }

    sidecar = args.out_dir / "training_vs_learning.json"
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", sidecar)


if __name__ == "__main__":
    main()
