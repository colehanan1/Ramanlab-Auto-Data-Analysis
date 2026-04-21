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
    _display_label_ci,
    _display_odor,
    _extract_odor_from_label,
    _is_light_only_label,
    _is_testing_11_label,
    _normalise_fly_columns,
    _trained_label,
    _trial_num,
    compute_non_reactive_flags,
    get_protocol,
    set_protocol,
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

_STATIC_PAIRS = {
    "EB-Training": "EB-Control",
    "Hex-Training": "Hex-Control",
    "Hex-Training-24": "Hex-Control-24",
    "Hex-Training-24-2": "Hex-Control-24-2",
    "Hex-Training-24-02": "Hex-Control-24-02",
    "Hex-Training-36": "Hex-Control-36",
    "Hex-Training-24-002": "Hex-Control-24-002",
    "Hex-Training-24-0002": "Hex-Control-24-0002",
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


def _auto_pairs(datasets: list[str]) -> dict[str, str]:
    pairs = dict(_STATIC_PAIRS)
    pat = re.compile(r"^(.+)-Training(.*)$", re.IGNORECASE)
    ds_set = set(datasets)
    for ds in datasets:
        if ds in pairs:
            continue
        m = pat.match(ds)
        if m:
            ctrl = f"{m.group(1)}-Control{m.group(2)}"
            if ctrl in ds_set:
                pairs[ds] = ctrl
    return pairs

TRAINING_CONTROL_PAIRS = dict(_STATIC_PAIRS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalise_trial_label(label: str) -> str:
    """Keep odor suffix for v2: testing_3_benzaldehyde → testing_3_benzaldehyde."""
    m = re.match(
        r"((?:testing|training)_\d+(?:_(?!fly\d|distances)[A-Za-z0-9._-]+?)?)"
        r"(?:_fly\d|_distances|$)",
        str(label), re.IGNORECASE,
    )
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

    # Drop light-only trials (testing_11 legacy, testing_9 v2)
    light_mask = df["trial"].apply(_is_light_only_label)
    df = df.loc[~light_mask].copy()

    # Add odor display name from trial label
    df["odor_display"] = df.apply(
        lambda r: _display_label_ci(_extract_odor_from_label(r["trial"])), axis=1
    )

    # For v2: build odor_col with "Name 1"/"Name 2" for trained odor only
    if get_protocol() == "v2":
        _odor_col_rows = []
        for (ds, fly, fn), grp in df.groupby(["dataset_canon", "fly", "fly_number"]):
            seen: dict[str, int] = {}
            for idx, row in grp.sort_values("trial_num").iterrows():
                od = row["odor_display"]
                seen[od] = seen.get(od, 0) + 1
                _odor_col_rows.append((idx, seen[od]))
        occ_series = pd.Series(dict(_odor_col_rows), name="occurrence")
        df = df.join(occ_series)
        # Find which odors appear more than once per fly
        max_occ = df.groupby(["dataset_canon", "odor_display"])["occurrence"].max()
        dup_set = set(max_occ[max_occ > 1].reset_index()["odor_display"])
        # Only number the trained odor; non-trained duplicates stay unnumbered
        def _should_number(row):
            if row["odor_display"] not in dup_set:
                return False
            trained = _trained_label(row["dataset_canon"])
            return row["odor_display"].casefold() == trained.casefold()
        df["odor_col"] = df.apply(
            lambda r: f"{r['odor_display']} {int(r['occurrence'])}" if _should_number(r) else r["odor_display"],
            axis=1,
        )
    else:
        df["odor_col"] = df["odor_display"]

    # De-duplicate (same logic as reaction matrix)
    df = df.drop_duplicates(
        subset=["dataset", "fly", "fly_number", "trial"], keep="first"
    )

    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (dataset_canon, odor_col) and compute mean/sem/n."""
    group_key = "odor_col" if "odor_col" in df.columns and get_protocol() == "v2" else "trial_num"
    grouped = (
        df.groupby(["dataset_canon", group_key])["score"]
        .agg(["mean", "sem", "count"])
        .rename(columns={"mean": "mean_score", "sem": "sem_score", "count": "n_flies"})
        .reset_index()
    )
    grouped["sem_score"] = grouped["sem_score"].fillna(0.0)
    if group_key == "odor_col":
        grouped = grouped.sort_values(["dataset_canon", "odor_col"], key=lambda s: s.str.casefold() if s.dtype == object else s)
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
    if get_protocol() == "v2" and "odor_col" in compare.columns:
        compare["odor"] = compare["odor_col"]
    else:
        compare["odor"] = [
            _display_odor(dataset, trial)
            for dataset, trial in zip(compare["dataset_canon"], compare["trial"])
        ]

    present = set(compare["dataset_canon"])
    active_pairs = _auto_pairs(list(present))
    summaries: list[pd.DataFrame] = []

    for train_ds, ctrl_ds in active_pairs.items():
        if train_ds not in present or ctrl_ds not in present:
            continue

        pair_df = compare[compare["dataset_canon"].isin((train_ds, ctrl_ds))].copy()
        if pair_df.empty:
            continue

        group_key = "odor" if get_protocol() == "v2" else "trial_num"
        group_cols = ["dataset_canon", group_key, "odor", "fly", "fly_number"] if group_key == "trial_num" else ["dataset_canon", "odor", "fly", "fly_number"]
        fly_level = (
            pair_df.groupby(group_cols)["score"]
            .mean()
            .rename("fly_mean_score")
            .reset_index()
        )
        sample_group_cols = ["dataset_canon", "odor"] if get_protocol() == "v2" else ["dataset_canon", "trial_num"]
        fly_samples = {
            tuple(key): group["fly_mean_score"].to_numpy(float)
            for key, group in fly_level.groupby(sample_group_cols)
        }
        stats_group = ["dataset_canon", "odor"] if get_protocol() == "v2" else ["dataset_canon", "trial_num", "odor"]
        score_stats = (
            fly_level.groupby(stats_group)["fly_mean_score"]
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

        merge_on = ["odor"] if get_protocol() == "v2" else ["trial_num", "odor"]
        merged = pd.merge(train_scores, ctrl_scores, on=merge_on, how="outer")

        if merged.empty:
            continue

        merged["training_dataset"] = train_ds
        merged["control_dataset"] = ctrl_ds
        merged["is_trained"] = (
            merged["odor"].astype(str).str.casefold()
            .str.startswith(_trained_label(train_ds).casefold())
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
            if get_protocol() == "v2":
                train_vals = fly_samples.get((train_ds, row.odor), np.array([], dtype=float))
                ctrl_vals = fly_samples.get((ctrl_ds, row.odor), np.array([], dtype=float))
            else:
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
        if get_protocol() == "v2":
            merged = merged.sort_values("odor", key=lambda s: s.str.casefold()).reset_index(drop=True)
        else:
            merged = merged.sort_values(["trial_num", "odor"]).reset_index(drop=True)

        out_cols = ["training_dataset", "control_dataset"]
        if "trial_num" in merged.columns:
            out_cols.append("trial_num")
        out_cols += [
            "odor", "is_trained",
            "mean_score_train", "sem_score_train", "n_flies_train",
            "mean_score_ctrl", "sem_score_ctrl", "n_flies_ctrl",
            "score_p_value", "significance",
        ]
        summaries.append(merged[[c for c in out_cols if c in merged.columns]])

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
        sort_col = "odor_col" if "odor_col" in summary.columns and get_protocol() == "v2" else "trial_num"
        sub = summary[summary["dataset_canon"] == odor].sort_values(
            sort_col, key=lambda s: s.str.casefold() if s.dtype == object else s
        )
        if sub.empty:
            continue

        label = DISPLAY_LABEL.get(odor, odor)
        png_path = out_dir / f"mean_score_{odor.replace(' ', '_')}.png"
        if not should_write(png_path, overwrite):
            continue

        with plt.rc_context(_RC_CONTEXT):
            fig, ax = plt.subplots(figsize=(max(6, len(sub) * 0.7 + 2), 5))
            x = np.arange(len(sub))
            # Color trained odor blue, non-trained gray
            trained = _trained_label(odor)
            if get_protocol() == "v2" and "odor_col" in sub.columns:
                is_trained = sub["odor_col"].str.casefold().str.startswith(trained.casefold())
            else:
                is_trained = pd.Series([False] * len(sub), index=sub.index)
            bar_colors = [
                "#1a3a6b" if t else "#b0b0b0" for t in is_trained
            ]
            bars = ax.bar(
                x,
                sub["mean_score"].values,
                yerr=sub["sem_score"].values,
                capsize=4,
                color=bar_colors,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(x)
            if get_protocol() == "v2" and "odor_col" in sub.columns:
                ax.set_xticklabels(
                    [f"{o}\n(n={int(n)})" for o, n in zip(sub["odor_col"], sub["n_flies"])],
                    fontsize=8, rotation=35, ha="right",
                )
            else:
                ax.set_xticklabels(
                    [f"T{int(t)}\n(n={int(n)})" for t, n in zip(sub["trial_num"], sub["n_flies"])],
                    fontsize=9,
                )
            ax.set_ylim(-1.5, 5.5)
            ax.set_ylabel("Mean Score")
            ax.set_xlabel("Presented Odor" if get_protocol() == "v2" else "Testing Trial")
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

    all_train_ds = set(summary["training_dataset"].unique()) if not summary.empty else set()
    for train_ds in all_train_ds:
        sub = summary[summary["training_dataset"] == train_ds].copy()
        if sub.empty:
            continue

        label = DISPLAY_LABEL.get(train_ds, train_ds)
        png_path = out_dir / f"mean_score_train_vs_ctrl_{train_ds.replace(' ', '_')}.png"
        if not should_write(png_path, overwrite):
            continue

        if "trial_num" in sub.columns:
            sub = sub.sort_values(["trial_num", "odor"]).reset_index(drop=True)
        else:
            sub = sub.sort_values("odor", key=lambda s: s.str.casefold()).reset_index(drop=True)
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

        train_color_trained = "#1a3a6b"   # dark blue (trained odor)
        train_color_other = "#7bafd4"     # lighter blue (non-trained)
        ctrl_color_trained = "#808080"    # dark gray (trained odor)
        ctrl_color_other = "#c8c8c8"     # lighter gray (non-trained)

        with plt.rc_context(_RC_CONTEXT):
            fig, ax = plt.subplots(figsize=(max(7, len(sub) * 1.0 + 2), 5.5))
            ax.bar(
                x - bar_w / 2,
                train_vals,
                width=bar_w,
                yerr=train_err,
                capsize=4,
                color=[
                    train_color_trained if bool(is_trained) else train_color_other
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
                color=[
                    ctrl_color_trained if bool(is_trained) else ctrl_color_other
                    for is_trained in sub["is_trained"]
                ],
                edgecolor="black",
                linewidth=0.75,
                label="Control",
            )

            n_train = sub["n_flies_train"].fillna(0).astype(int)
            n_ctrl = sub["n_flies_ctrl"].fillna(0).astype(int)
            labels = [
                f"{str(odor).upper() if bool(is_trained) else str(odor)}\n(n={nt}/{nc})"
                for odor, is_trained, nt, nc in zip(
                    sub["odor"], sub["is_trained"], n_train, n_ctrl
                )
            ]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            for tick, is_trained in zip(ax.get_xticklabels(), sub["is_trained"]):
                if bool(is_trained):
                    tick.set_color(train_color_trained)
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

    col_key = "odor_col" if "odor_col" in summary.columns and get_protocol() == "v2" else "trial_num"
    pivot = summary.pivot_table(
        index="dataset_canon",
        columns=col_key,
        values="mean_score",
        aggfunc="first",
    )

    # Order rows by ODOR_ORDER
    ordered = [o for o in ODOR_ORDER if o in pivot.index]
    extras = sorted(o for o in pivot.index if o not in ODOR_ORDER)
    pivot = pivot.reindex(ordered + extras)

    # Sort columns
    if col_key == "odor_col":
        pivot = pivot.reindex(sorted(pivot.columns, key=str.casefold), axis=1)
    else:
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
        if get_protocol() == "v2":
            ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8, rotation=35, ha="right")
        else:
            ax.set_xticklabels([f"T{int(c)}" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(display_idx)))
        ax.set_yticklabels(display_idx, fontsize=9)
        ax.set_xlabel("Presented Odor" if get_protocol() == "v2" else "Testing Trial")
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
    parser.add_argument(
        "--protocol", type=str, default="v2", choices=["v2", "legacy"],
        help="Protocol version (default: v2).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    set_protocol(args.protocol)
    generate_score_summary(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        non_reactive_threshold=args.non_reactive_threshold,
        flagged_flies_csv=args.flagged_flies_csv,
    )


if __name__ == "__main__":
    main()
