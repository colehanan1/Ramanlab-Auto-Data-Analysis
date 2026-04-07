"""Generate average ordinal-score plots from model predictions.

Reads a predictions CSV that contains a ``score`` column (ordinal -1..5) and
produces:

1. Per-odor bar chart of mean score by testing number (with SEM error bars).
2. Heatmap of mean score (datasets x testing numbers).
3. Summary CSV: ``score_summary_by_odor_testing.csv``.
4. Training-vs-control grouped bar charts of mean score by presented odor.
5. Training-vs-control summary CSV with score-based p-values.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import (
    DISPLAY_LABEL,
    ODOR_ORDER,
    _canon_dataset,
    _display_odor,
    _is_testing_11_label,
    _normalise_fly_columns,
    _trained_label,
    _trial_num,
    compute_non_reactive_flags,
    should_write,
)

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}

TRAINING_CONTROL_PAIRS = {
    "EB-Training": "EB-Control",
    "Hex-Training": "Hex-Control",
    "Hex-Training-24": "Hex-Control-24",
    "Hex-Training-24-2": "Hex-Control-24-2",
    "Hex-Training-24-02": "Hex-Control-24-02",
    "Hex-Training-36": "Hex-Control-36",
    "Benz-Training": "Benz-Control",
    "Benz-Training-24": "Benz-Control",
    "Benz-Training-24-2": "Benz-Control-24-2",
    "Benz-Training-24-02": "Benz-Control-24-02",
    "ACV-Training": "ACV-Control",
    "3OCT-Training": "3OCT-Control",
    "3OCT-Training-24-2": "3OCT-Control-24-2",
    "Cit-Training": "Cit-Control",
    "Lin-Training": "Lin-Control",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalise_trial_label(label: str) -> str:
    m = re.match(r"(testing_\d+)", str(label))
    return m.group(1) if m else str(label)


def _load_scores(
    csv_path: Path,
    *,
    threshold: float | None = None,
    flagged_flies_csv: str = "",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"dataset", "fly", "fly_number", "trial_label", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()

    # Filter to testing trials
    if "trial_type" in df.columns:
        mask = df["trial_type"].astype(str).str.strip().str.lower() == "testing"
        df = df.loc[mask].copy()

    if df.empty:
        raise RuntimeError("No testing trials found in predictions CSV.")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = _normalise_fly_columns(df)

    # Drop non-reactive flies
    if threshold is not None:
        flags = compute_non_reactive_flags(
            df, threshold=threshold, flagged_flies_csv=flagged_flies_csv
        )
        if flags.any():
            df = df.loc[~flags.astype(bool)].copy()

    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].astype(str).apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)

    # Drop testing_11
    t11_mask = df["trial"].apply(_is_testing_11_label)
    df = df.loc[~t11_mask].copy()

    # De-duplicate (same logic as reaction matrix)
    df = df.drop_duplicates(
        subset=["dataset", "fly", "fly_number", "trial"], keep="first"
    )

    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (dataset_canon, trial_num) and compute mean/sem/n."""
    grouped = (
        df.groupby(["dataset_canon", "trial_num"])["score"]
        .agg(["mean", "sem", "count"])
        .rename(columns={"mean": "mean_score", "sem": "sem_score", "count": "n_flies"})
        .reset_index()
    )
    grouped["sem_score"] = grouped["sem_score"].fillna(0.0)
    return grouped


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _empty_train_control_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "training_dataset",
            "control_dataset",
            "trial_num",
            "odor",
            "is_trained",
            "mean_score_train",
            "sem_score_train",
            "n_flies_train",
            "mean_score_ctrl",
            "sem_score_ctrl",
            "n_flies_ctrl",
            "score_p_value",
            "significance",
        ]
    )


def _compute_training_vs_control_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_train_control_summary()

    compare = df.copy()
    compare["odor"] = [
        _display_odor(dataset, trial)
        for dataset, trial in zip(compare["dataset_canon"], compare["trial"])
    ]

    present = set(compare["dataset_canon"])
    summaries: list[pd.DataFrame] = []

    for train_ds, ctrl_ds in TRAINING_CONTROL_PAIRS.items():
        if train_ds not in present or ctrl_ds not in present:
            continue

        pair_df = compare[compare["dataset_canon"].isin((train_ds, ctrl_ds))].copy()
        if pair_df.empty:
            continue

        fly_level = (
            pair_df.groupby(
                ["dataset_canon", "trial_num", "odor", "fly", "fly_number"]
            )["score"]
            .mean()
            .rename("fly_mean_score")
            .reset_index()
        )
        fly_samples = {
            (dataset, trial_num): group["fly_mean_score"].to_numpy(float)
            for (dataset, trial_num), group in fly_level.groupby(["dataset_canon", "trial_num"])
        }
        score_stats = (
            fly_level.groupby(["dataset_canon", "trial_num", "odor"])["fly_mean_score"]
            .agg(["mean", "sem", "count"])
            .rename(
                columns={
                    "mean": "mean_score",
                    "sem": "sem_score",
                    "count": "n_flies",
                }
            )
            .reset_index()
        )
        score_stats["sem_score"] = score_stats["sem_score"].fillna(0.0)

        train_scores = (
            score_stats[score_stats["dataset_canon"] == train_ds]
            .drop(columns="dataset_canon")
            .rename(
                columns={
                    "mean_score": "mean_score_train",
                    "sem_score": "sem_score_train",
                    "n_flies": "n_flies_train",
                }
            )
        )
        ctrl_scores = (
            score_stats[score_stats["dataset_canon"] == ctrl_ds]
            .drop(columns="dataset_canon")
            .rename(
                columns={
                    "mean_score": "mean_score_ctrl",
                    "sem_score": "sem_score_ctrl",
                    "n_flies": "n_flies_ctrl",
                }
            )
        )

        merged = pd.merge(train_scores, ctrl_scores, on=["trial_num", "odor"], how="outer")

        if merged.empty:
            continue

        merged["training_dataset"] = train_ds
        merged["control_dataset"] = ctrl_ds
        merged["is_trained"] = (
            merged["odor"].astype(str).str.casefold()
            == _trained_label(train_ds).casefold()
        )

        for col in (
            "sem_score_train",
            "sem_score_ctrl",
            "n_flies_train",
            "n_flies_ctrl",
        ):
            merged[col] = merged[col].fillna(0)

        for col in ("n_flies_train", "n_flies_ctrl"):
            merged[col] = merged[col].astype(int)

        p_values: list[float] = []
        significance: list[str] = []
        for row in merged.itertuples(index=False):
            train_vals = fly_samples.get((train_ds, row.trial_num), np.array([], dtype=float))
            ctrl_vals = fly_samples.get((ctrl_ds, row.trial_num), np.array([], dtype=float))
            if len(train_vals) == 0 or len(ctrl_vals) == 0:
                p = np.nan
            else:
                _, p = mannwhitneyu(
                    train_vals,
                    ctrl_vals,
                    alternative="two-sided",
                    method="auto",
                )
            p_values.append(p)
            significance.append(_sig_stars(p))

        merged["score_p_value"] = p_values
        merged["significance"] = significance
        merged = merged.sort_values(["trial_num", "odor"]).reset_index(drop=True)

        summaries.append(
            merged[
                [
                    "training_dataset",
                    "control_dataset",
                    "trial_num",
                    "odor",
                    "is_trained",
                    "mean_score_train",
                    "sem_score_train",
                    "n_flies_train",
                    "mean_score_ctrl",
                    "sem_score_ctrl",
                    "n_flies_ctrl",
                    "score_p_value",
                    "significance",
                ]
            ]
        )

    if not summaries:
        return _empty_train_control_summary()

    return pd.concat(summaries, ignore_index=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_bar_charts(
    df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, *, overwrite: bool
) -> None:
    """One bar-chart per dataset/odor: mean score per testing number."""
    present = df["dataset_canon"].unique().tolist()
    ordered = [o for o in ODOR_ORDER if o in present]
    extras = sorted(o for o in present if o not in ODOR_ORDER)

    for odor in ordered + extras:
        sub = summary[summary["dataset_canon"] == odor].sort_values("trial_num")
        if sub.empty:
            continue

        label = DISPLAY_LABEL.get(odor, odor)
        png_path = out_dir / f"mean_score_{odor.replace(' ', '_')}.png"
        if not should_write(png_path, overwrite):
            continue

        with plt.rc_context(_RC_CONTEXT):
            fig, ax = plt.subplots(figsize=(max(6, len(sub) * 0.7 + 2), 5))
            x = np.arange(len(sub))
            bars = ax.bar(
                x,
                sub["mean_score"].values,
                yerr=sub["sem_score"].values,
                capsize=4,
                color="#5B9BD5",
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"T{int(t)}" for t in sub["trial_num"]], fontsize=9
            )
            ax.set_ylim(-1.5, 5.5)
            ax.set_ylabel("Mean Score")
            ax.set_xlabel("Testing Trial")
            ax.set_title(f"Mean Ordinal Score - {label}", fontsize=13, weight="bold")
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            # Mark the reaction boundary
            ax.axhline(
                y=1.5, color="red", linewidth=0.8, linestyle=":", alpha=0.6,
                label="Reaction boundary (1/2)",
            )
            ax.legend(fontsize=8, loc="upper right")
            plt.tight_layout()
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def _draw_score_significance_brackets(
    ax: plt.Axes,
    x_positions: np.ndarray,
    bar_w: float,
    sub: pd.DataFrame,
) -> None:
    for i, row in enumerate(sub.itertuples(index=False)):
        p = float(row.score_p_value) if pd.notna(row.score_p_value) else np.nan
        stars = _sig_stars(p)
        if not stars:
            continue

        train_top = (
            float(row.mean_score_train) + float(row.sem_score_train)
            if pd.notna(row.mean_score_train)
            else 0.0
        )
        ctrl_top = (
            float(row.mean_score_ctrl) + float(row.sem_score_ctrl)
            if pd.notna(row.mean_score_ctrl)
            else 0.0
        )
        top = max(0.0, train_top, ctrl_top)
        bracket_y = top + 0.45
        tip_y = bracket_y - 0.08
        x_left = x_positions[i] - bar_w / 2
        x_right = x_positions[i] + bar_w / 2

        ax.plot(
            [x_left, x_left, x_right, x_right],
            [tip_y, bracket_y, bracket_y, tip_y],
            color="black",
            linewidth=0.9,
            clip_on=False,
        )
        ax.text(
            x_positions[i],
            bracket_y + 0.05,
            stars,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )


def _plot_training_vs_control_bars(
    summary: pd.DataFrame, out_dir: Path, *, overwrite: bool
) -> None:
    if summary.empty:
        return

    for train_ds in TRAINING_CONTROL_PAIRS:
        sub = summary[summary["training_dataset"] == train_ds].copy()
        if sub.empty:
            continue

        label = DISPLAY_LABEL.get(train_ds, train_ds)
        png_path = out_dir / f"mean_score_train_vs_ctrl_{train_ds.replace(' ', '_')}.png"
        if not should_write(png_path, overwrite):
            continue

        sub = sub.sort_values(["trial_num", "odor"]).reset_index(drop=True)
        x = np.arange(len(sub))
        bar_w = 0.35
        train_vals = sub["mean_score_train"].fillna(0.0).to_numpy(float)
        ctrl_vals = sub["mean_score_ctrl"].fillna(0.0).to_numpy(float)
        train_err = sub["sem_score_train"].fillna(0.0).to_numpy(float)
        ctrl_err = sub["sem_score_ctrl"].fillna(0.0).to_numpy(float)

        y_top = 0.0
        if len(sub):
            y_top = float(
                np.max(
                    np.concatenate(
                        [
                            np.maximum(train_vals + train_err, 0.0),
                            np.maximum(ctrl_vals + ctrl_err, 0.0),
                        ]
                    )
                )
            )

        with plt.rc_context(_RC_CONTEXT):
            fig, ax = plt.subplots(figsize=(max(7, len(sub) * 1.0 + 2), 5.5))
            ax.bar(
                x - bar_w / 2,
                train_vals,
                width=bar_w,
                yerr=train_err,
                capsize=4,
                color=[
                    "#1a3a6b" if bool(is_trained) else "#4a7fbf"
                    for is_trained in sub["is_trained"]
                ],
                edgecolor="black",
                linewidth=0.75,
                label="Training",
            )
            ax.bar(
                x + bar_w / 2,
                ctrl_vals,
                width=bar_w,
                yerr=ctrl_err,
                capsize=4,
                color="#b0b0b0",
                edgecolor="black",
                linewidth=0.75,
                label="Control",
            )

            labels = [
                str(odor).upper() if bool(is_trained) else str(odor)
                for odor, is_trained in zip(sub["odor"], sub["is_trained"])
            ]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            for tick, is_trained in zip(ax.get_xticklabels(), sub["is_trained"]):
                if bool(is_trained):
                    tick.set_color("#1a3a6b")
                    tick.set_weight("bold")

            ax.set_ylabel("Mean Score")
            ax.set_xlabel("Testing Trial / Presented Odor")
            ax.set_title(
                f"Mean Model Score - {label} (Training vs Control, Trials Separate)",
                fontsize=13,
                weight="bold",
            )
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            ax.axhline(
                y=1.5,
                color="red",
                linewidth=0.8,
                linestyle=":",
                alpha=0.6,
                label="Reaction boundary (score >= 2)",
            )
            ax.set_ylim(-1.5, max(6.1, y_top + 1.0))
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

            _draw_score_significance_brackets(ax, x, bar_w, sub)

            plt.tight_layout()
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def _plot_heatmap(
    summary: pd.DataFrame, out_dir: Path, *, overwrite: bool
) -> None:
    """Heatmap: datasets (rows) x testing numbers (columns), colored by mean score."""
    png_path = out_dir / "mean_score_heatmap.png"
    if not should_write(png_path, overwrite):
        return

    pivot = summary.pivot_table(
        index="dataset_canon",
        columns="trial_num",
        values="mean_score",
        aggfunc="first",
    )

    # Order rows by ODOR_ORDER
    ordered = [o for o in ODOR_ORDER if o in pivot.index]
    extras = sorted(o for o in pivot.index if o not in ODOR_ORDER)
    pivot = pivot.reindex(ordered + extras)

    # Sort columns numerically
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    display_idx = [DISPLAY_LABEL.get(o, o) for o in pivot.index]

    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(
            figsize=(max(8, len(pivot.columns) * 0.8 + 3), max(4, len(pivot) * 0.5 + 2))
        )
        norm = TwoSlopeNorm(vmin=-1, vcenter=1.5, vmax=5)
        im = ax.imshow(
            pivot.values,
            cmap="RdYlGn",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"T{int(c)}" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(display_idx)))
        ax.set_yticklabels(display_idx, fontsize=9)
        ax.set_xlabel("Testing Trial")
        ax.set_title("Mean Ordinal Score by Dataset and Trial", fontsize=13, weight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mean Score (-1 to 5)")

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if 0.5 < val < 4 else "white",
                    )

        plt.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_score_summary(
    csv_path: Path,
    out_dir: Path,
    *,
    overwrite: bool = True,
    non_reactive_threshold: float | None = None,
    flagged_flies_csv: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_scores(
        csv_path,
        threshold=non_reactive_threshold,
        flagged_flies_csv=flagged_flies_csv,
    )
    if df.empty:
        print("[score_summary] No data after filtering; nothing to plot.")
        return

    summary = _compute_summary(df)

    # Export CSV
    csv_out = out_dir / "score_summary_by_odor_testing.csv"
    if should_write(csv_out, overwrite):
        summary.to_csv(csv_out, index=False, float_format="%.4f")
        print(f"[score_summary] Wrote {csv_out}")

    train_ctrl_summary = _compute_training_vs_control_summary(df)
    if not train_ctrl_summary.empty:
        train_ctrl_csv = out_dir / "score_summary_train_vs_control.csv"
        if should_write(train_ctrl_csv, overwrite):
            train_ctrl_summary.to_csv(train_ctrl_csv, index=False, float_format="%.4f")
            print(f"[score_summary] Wrote {train_ctrl_csv}")

    _plot_bar_charts(df, summary, out_dir, overwrite=overwrite)
    _plot_heatmap(summary, out_dir, overwrite=overwrite)
    _plot_training_vs_control_bars(train_ctrl_summary, out_dir, overwrite=overwrite)
    print(f"[score_summary] Plots saved to {out_dir}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path", type=Path, required=True,
        help="Path to predictions CSV containing a 'score' column.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory for exported plots and summary CSV.",
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--non-reactive-threshold", type=float, default=None,
        help="Span threshold for non-reactive fly exclusion.",
    )
    parser.add_argument(
        "--flagged-flies-csv", type=str, default="",
        help="Path to flagged-flies CSV for exclusion.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    generate_score_summary(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        non_reactive_threshold=args.non_reactive_threshold,
        flagged_flies_csv=args.flagged_flies_csv,
    )


if __name__ == "__main__":
    main()
