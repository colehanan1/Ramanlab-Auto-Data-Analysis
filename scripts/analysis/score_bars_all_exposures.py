#!/usr/bin/env python3
"""Mean ordinal PER score for *every* Hexanol and ACV exposure, train vs control.

``score_bars_variants`` collapses each odor to its first presentation, so the
published ``score_bars_Hex-Training_oct_nov_hex_acv`` figure carries one ACV bar
and one Hexanol bar. The Oct/Nov testing panel actually presents Apple Cider
Vinegar twice (trials 1 and 3) and Hexanol three times (trials 2, 4 and 5).
This driver keeps all five presentations, each still paired against the
Hex-Control cohort and tested on its own.

Bars are grouped by odor by default — both ACV exposures, then all three
Hexanol exposures — with the odor that was presented first leading, so the
figure reads the same left-to-right as the one-bar-per-odor version.
``--presentation-order`` instead lays them out in the order the fly met them
(ACV 1, Hexanol 1, ACV 2, Hexanol 2, Hexanol 3).

Run::

    python scripts/analysis/score_bars_all_exposures.py \
        --predictions-csv /home/ramanlab/Documents/cole/Results/\
Opto-Fly-Figures-OctNov/Matrix-PER-Reactions-Model/model_predictions_oct_nov.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import odor_bar_palette  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    DISPLAY_LABEL,
    _canon_dataset,
    _display_odor,
    _is_testing_11_label,
    _normalise_fly_columns,
    _trial_num,
)
from scripts.analysis.reaction_matrix_from_spreadsheet import (  # noqa: E402
    _normalise_trial_label,
)

PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/model_predictions.csv"
)
FIGURES_DIR = Path("/home/ramanlab/Documents/cole/Results/Figures")
ALLOWED_PREFIXES_OCTNOV = ("october_", "november_")
HEX_ACV_KEYS = {"hexanol", "apple cider vinegar"}

CTRL_COLOR = odor_bar_palette.CTRL_COLOR

SCORE_MIN, SCORE_MAX = -1, 5
SCORE_Y_LABEL = "Mean PER Score"
# The tallest Hexanol bar's error bar reaches ~4.3 and carries a value label
# above it, so the shared bracket row needs this much clearance...
BRACKET_GAP = 0.75
# ...and the axis needs headroom above the score maximum to hold that row, while
# still ticking only over the -1..5 the score is defined on.
Y_HEADROOM = 0.9

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""  # not significant: no bracket, no label


def _cohort_label(name: str, n_flies) -> str:
    """``"Training (n=13)"`` when every bar shares one fly count."""
    counts = {int(n) for n in pd.Series(n_flies).dropna()}
    return f"{name} (n={counts.pop()})" if len(counts) == 1 else name


# ---------------------------------------------------------------------------
# Loading and per-fly aggregation
# ---------------------------------------------------------------------------

def _load_predictions(predictions_csv: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv or PREDICTIONS_CSV)
    if "score" not in df.columns:
        raise RuntimeError("Predictions CSV is missing the 'score' column.")
    df = df[df["trial_type"].astype(str).str.strip().str.lower() == "testing"].copy()
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["trial_label"] = df["trial_label"].astype(str).str.strip()
    df = _normalise_fly_columns(df)
    df = df.loc[
        ~df.get("_non_reactive", pd.Series(False, index=df.index)).astype(bool)
    ].copy()
    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score", "trial_num"]).copy()
    df = df[~df["trial"].apply(_is_testing_11_label)].copy()
    df = df.drop_duplicates(subset=["dataset", "fly", "fly_number", "trial"], keep="first")
    df["odor"] = [_display_odor(ds, t) for ds, t in zip(df["dataset_canon"], df["trial"])]
    df["odor"] = df["odor"].replace({"3-Octonol": "3-Octanol"})
    return df


def _per_fly_scores(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """One row per (fly, trial_num, odor) for the given dataset."""
    sub = df[df["dataset_canon"] == dataset].copy()
    if sub.empty:
        return sub
    return (
        sub.groupby(["dataset_canon", "trial_num", "odor", "fly", "fly_number"])["score"]
        .mean()
        .rename("fly_mean_score")
        .reset_index()
    )


def _summary_from_fly_scores(fly_level: pd.DataFrame, *, trained_odor: str) -> pd.DataFrame:
    if fly_level.empty:
        return pd.DataFrame(
            columns=["trial_num", "odor", "mean_score", "sem_score", "n_flies", "is_trained"]
        )
    stats = (
        fly_level.groupby(["trial_num", "odor"])["fly_mean_score"]
        .agg(["mean", "sem", "count"])
        .rename(columns={"mean": "mean_score", "sem": "sem_score", "count": "n_flies"})
        .reset_index()
    )
    stats["sem_score"] = stats["sem_score"].fillna(0.0)
    stats["is_trained"] = stats["odor"].astype(str).str.casefold() == trained_odor.casefold()
    return stats.sort_values(["trial_num", "odor"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Subset, exposure numbering, bar order
# ---------------------------------------------------------------------------

def _keep_hex_acv(stats: pd.DataFrame) -> pd.DataFrame:
    """Every Hexanol and ACV presentation — deliberately not collapsed to the first."""
    if stats.empty:
        return stats
    odors = stats["odor"].astype(str).str.strip().str.casefold()
    return (
        stats.loc[odors.isin(HEX_ACV_KEYS)]
        .sort_values("trial_num", kind="mergesort")
        .reset_index(drop=True)
    )


def _exposure_labels(frame: pd.DataFrame) -> list[str]:
    """``"Hexanol 1"`` … ``"Hexanol 3"`` — the odor plus its exposure number.

    The number counts presentations of that odor in trial order, so it stays
    attached to the right bar however the bars are later sorted for display.
    """
    order = frame.groupby("odor")["trial_num"].rank(method="first").astype(int)
    return [f"{odor} {int(idx)}" for odor, idx in zip(frame["odor"], order)]


def _order_bars(stats: pd.DataFrame, *, group_by_odor: bool) -> pd.DataFrame:
    """Left-to-right bar order: odor blocks, or the fly's presentation order."""
    if stats.empty or not group_by_odor:
        return stats.sort_values("trial_num", kind="mergesort").reset_index(drop=True)
    # Odor blocks, ordered by when each odor was first presented, so the figure
    # opens on the same odor as the one-bar-per-odor version.
    first_seen = stats.groupby("odor")["trial_num"].transform("min")
    return (
        stats.assign(_first_seen=first_seen)
        .sort_values(["_first_seen", "trial_num"], kind="mergesort")
        .drop(columns="_first_seen")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mannwhitney_per_presentation(
    fly_train: pd.DataFrame, fly_ctrl: pd.DataFrame, presentations
) -> dict:
    """One two-sided test per (trial_num, odor) — each exposure stands alone."""
    out = {}
    for trial_num, odor in presentations:
        key = (int(trial_num), str(odor))
        if fly_train.empty or fly_ctrl.empty:
            out[key] = np.nan
            continue
        a = fly_train[
            (fly_train["trial_num"] == trial_num) & (fly_train["odor"] == odor)
        ]["fly_mean_score"].to_numpy(float)
        b = fly_ctrl[
            (fly_ctrl["trial_num"] == trial_num) & (fly_ctrl["odor"] == odor)
        ]["fly_mean_score"].to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            out[key] = np.nan
            continue
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            p = np.nan
        out[key] = float(p) if p is not None else np.nan
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_train_vs_ctrl(
    ax: plt.Axes,
    train_stats: pd.DataFrame,
    ctrl_stats: pd.DataFrame,
    *,
    title: str,
    p_values: dict,
) -> pd.DataFrame:
    merged = pd.merge(
        train_stats[
            ["trial_num", "odor", "mean_score", "sem_score", "n_flies", "is_trained"]
        ],
        ctrl_stats[["trial_num", "odor", "mean_score", "sem_score", "n_flies"]],
        on=["trial_num", "odor"],
        how="left",
        suffixes=("_train", "_ctrl"),
    )
    for col in ("mean_score_train", "mean_score_ctrl", "sem_score_train", "sem_score_ctrl"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["n_flies_train"] = merged["n_flies_train"].fillna(0).astype(int)
    merged["n_flies_ctrl"] = merged["n_flies_ctrl"].fillna(0).astype(int)
    merged["is_trained"] = merged["is_trained"].fillna(False).astype(bool)

    n = len(merged)
    x = np.arange(n)
    bar_w = 0.35
    train_colors = odor_bar_palette.training_bar_colors(
        merged["odor"], merged["is_trained"]
    )

    bars_train = ax.bar(
        x - bar_w / 2,
        merged["mean_score_train"].fillna(0).to_numpy(float),
        width=bar_w,
        yerr=merged["sem_score_train"].fillna(0).to_numpy(float),
        color=train_colors,
        edgecolor="black",
        linewidth=0.75,
        capsize=3,
        error_kw={"linewidth": 0.9},
        label="Training",
    )
    bars_ctrl = ax.bar(
        x + bar_w / 2,
        merged["mean_score_ctrl"].fillna(0).to_numpy(float),
        width=bar_w,
        yerr=merged["sem_score_ctrl"].fillna(0).to_numpy(float),
        color=CTRL_COLOR,
        edgecolor="black",
        linewidth=0.75,
        capsize=3,
        error_kw={"linewidth": 0.9},
        label="Control",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(_exposure_labels(merged), rotation=35, ha="right")
    for tick, odor, is_trained in zip(
        ax.get_xticklabels(), merged["odor"], merged["is_trained"]
    ):
        if bool(is_trained):
            tick.set_color(odor_bar_palette.trained_tick_color(odor))
            tick.set_weight("bold")

    ax.set_ylim(SCORE_MIN - 0.5, SCORE_MAX + Y_HEADROOM)
    ax.set_yticks(range(SCORE_MIN, SCORE_MAX + 1))
    ax.axhline(0, color="0.4", linewidth=0.7)
    ax.set_ylabel(SCORE_Y_LABEL, fontsize=12)
    ax.set_xlabel("Presented Odor", fontsize=12)
    ax.set_title(title, fontsize=13, weight="bold", pad=20)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.margins(x=0.04)
    odor_bar_palette.add_training_legend(
        ax,
        train_colors,
        ctrl_color=CTRL_COLOR,
        train_label=_cohort_label("Training", merged["n_flies_train"]),
        ctrl_label=_cohort_label("Control", merged["n_flies_ctrl"]),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.85,
    )

    for bars, mean_col, sem_col, n_col in (
        (bars_train, "mean_score_train", "sem_score_train", "n_flies_train"),
        (bars_ctrl, "mean_score_ctrl", "sem_score_ctrl", "n_flies_ctrl"),
    ):
        for bar, (_, row) in zip(bars, merged.iterrows()):
            if int(row[n_col]) == 0:
                continue
            mean_v = float(row[mean_col]) if pd.notna(row[mean_col]) else 0.0
            sem_v = float(row[sem_col]) if pd.notna(row[sem_col]) else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_v + sem_v + 0.18,
                f"{mean_v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    merged["p_value"] = [
        p_values.get((int(t), str(o)), np.nan)
        for t, o in zip(merged["trial_num"], merged["odor"])
    ]

    def _pair_top(row) -> float:
        tops = []
        for mean_col, sem_col in (
            ("mean_score_train", "sem_score_train"),
            ("mean_score_ctrl", "sem_score_ctrl"),
        ):
            mean_v = float(row[mean_col]) if pd.notna(row[mean_col]) else 0.0
            sem_v = float(row[sem_col]) if pd.notna(row[sem_col]) else 0.0
            tops.append(mean_v + sem_v)
        return max(tops)

    # One bracket row for the whole panel, above the tallest error bar and clear
    # of its value annotation. Three Hexanol bars of slightly different height
    # would otherwise staircase their brackets across the figure, and per-bar
    # heights near 4 push the tallest one off the top of the axes.
    significant = [
        (i, row) for i, (_, row) in enumerate(merged.iterrows()) if _stars(row["p_value"])
    ]
    if significant:
        bracket_y = max(_pair_top(row) for _, row in merged.iterrows()) + BRACKET_GAP
        tip_y = bracket_y - 0.15
        for i, row in significant:
            ax.plot(
                [x[i] - bar_w / 2, x[i] - bar_w / 2, x[i] + bar_w / 2, x[i] + bar_w / 2],
                [tip_y, bracket_y, bracket_y, tip_y],
                color="black",
                linewidth=0.9,
            )
            ax.text(
                x[i], bracket_y + 0.05, _stars(row["p_value"]),
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
    return merged


def _save(train_stats, ctrl_stats, p_values, *, title, out_stem, figures_dir):
    n_bars = max(1, len(train_stats))
    fig_w = max(7.0, 0.95 * n_bars + 3.0)
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(fig_w, 6.0))
        merged = _plot_train_vs_ctrl(
            ax, train_stats, ctrl_stats, title=title, p_values=p_values
        )
        fig.tight_layout()
        for ext in ("png", "svg"):
            out = figures_dir / f"{out_stem}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {out}")
        plt.close(fig)

    merged.insert(0, "exposure", _exposure_labels(merged))
    merged["stars"] = [_stars(p) for p in merged["p_value"]]
    out_csv = figures_dir / f"{out_stem}_stats.csv"
    merged.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=PREDICTIONS_CSV,
        help="Scored predictions to plot. Point at the archived Oct/Nov copy to "
        "reproduce the published cohort (the live CSV has since lost 4 controls).",
    )
    parser.add_argument(
        "--presentation-order",
        action="store_true",
        help="Lay bars out in trial order (ACV 1, Hexanol 1, ACV 2, ...) instead "
        "of grouping each odor's exposures together.",
    )
    parser.add_argument(
        "--out-stem", default="score_bars_Hex-Training_oct_nov_hex_acv_all_exposures"
    )
    args = parser.parse_args(argv)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    df = _load_predictions(args.predictions_csv)
    df = df[
        df["fly"].astype(str).str.lower().str.startswith(ALLOWED_PREFIXES_OCTNOV)
    ].copy()
    if df.empty:
        raise RuntimeError("No Oct/Nov flies in predictions CSV.")

    hex_fly = _per_fly_scores(df, "Hex-Training")
    ctrl_fly = _per_fly_scores(df, "Hex-Control")
    if hex_fly.empty or ctrl_fly.empty:
        raise RuntimeError("Empty Oct/Nov Hex-Training or Hex-Control fly-score table.")

    train_stats = _keep_hex_acv(
        _summary_from_fly_scores(hex_fly, trained_odor="Hexanol")
    )
    ctrl_stats = _keep_hex_acv(
        _summary_from_fly_scores(ctrl_fly, trained_odor="Hexanol")
    )

    presentations = [
        (int(r.trial_num), str(r.odor)) for r in train_stats.itertuples(index=False)
    ]
    p_values = _mannwhitney_per_presentation(hex_fly, ctrl_fly, presentations)
    for (trial_num, odor), p in p_values.items():
        if not pd.isna(p):
            print(f"  Mann-Whitney U: T{trial_num:<2d} {odor:22s} p={p:.5f} {_stars(p)}")

    group = not args.presentation_order
    hex_label = DISPLAY_LABEL.get("Hex-Training", "Hex-Training")
    _save(
        _order_bars(train_stats, group_by_odor=group),
        ctrl_stats,
        p_values,
        title=f"Mean PER Score – {hex_label} Training vs Control",
        out_stem=args.out_stem,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
