"""Generate average ordinal-score plots from model predictions.

Reads a predictions CSV that contains a ``score`` column (ordinal -1..5) and
produces:

1. Per-odor bar chart of mean score by testing number (with SEM error bars).
2. Heatmap of mean score (datasets x testing numbers).
3. Summary CSV: ``score_summary_by_odor_testing.csv``.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import (
    DISPLAY_LABEL,
    ODOR_ORDER,
    _canon_dataset,
    _is_testing_11_label,
    _normalise_fly_columns,
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

    _plot_bar_charts(df, summary, out_dir, overwrite=overwrite)
    _plot_heatmap(summary, out_dir, overwrite=overwrite)
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
