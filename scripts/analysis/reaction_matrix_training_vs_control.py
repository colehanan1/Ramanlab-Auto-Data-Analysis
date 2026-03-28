"""Generate reaction-matrix plots with Training vs Control bar charts.

Reads the **already-generated** binary reaction CSVs from the
Matrix-PER-Reactions-Model output directory for bar-chart values, and
rebuilds the matrix heatmap from the same predictions CSV used by
``reaction_matrix_from_spreadsheet.py``.

For each of EB-Training, Hex-Training, and Benz-Training this script
produces the same matrix heatmap but extends the bottom bar chart to
show **both** the training PER% (dark blue) and the corresponding
control PER% (gray) side by side for every odor.

Usage::

    python scripts/analysis/reaction_matrix_training_vs_control.py \
        --csv-path /path/to/model_predictions.csv \
        --out-dir  /path/to/Matrix-PER-Reactions-Model \
        --latency-sec 2.15

"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import (
    DISPLAY_LABEL,
    ODOR_ORDER,
    REACTION_RATE_ODOR_INDEX,
    _canon_dataset,
    _display_odor,
    _drop_testing_11,
    _fly_sort_key,
    _is_testing_11_label,
    _matrix_title,
    _normalise_fly_columns,
    _order_suffix,
    _style_trained_xticks,
    _trained_label,
    _trial_num,
    _trial_order_for,
    compute_non_reactive_flags,
    non_reactive_mask,
    resolve_dataset_output_dir,
    should_write,
)
from scripts.analysis.reaction_matrix_from_spreadsheet import (
    SpreadsheetMatrixConfig,
    _filter_trial_types,
    _normalise_trial_label,
)

# ---------------------------------------------------------------------------
# Training → Control mapping
# ---------------------------------------------------------------------------
TRAINING_CONTROL_PAIRS = {
    "EB-Training": "EB-Control",
    "Hex-Training": "Hex-Control",
    "Hex-Training-24": "Hex-Control-24",
    "Hex-Training-24-2": "Hex-Control-24-2",
    "Hex-Training-36": "Hex-Control-36",
    "Benz-Training": "Benz-Control",
    "Benz-Training-24": "Benz-Control",
    "Benz-Training-24-2": "Benz-Control-24-2",
    "ACV-Training": "ACV-Control",
    "3OCT-Training": "3OCT-Control",
    "Cit-Training": "Cit-Control",
    "Lin-Training": "Lin-Control",
}

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _load_raw_binary_csv(out_dir: Path, dataset: str) -> pd.DataFrame:
    """Load raw trial-level binary CSV for a dataset (needed for Fisher's test)."""
    ds_dir = resolve_dataset_output_dir(out_dir, dataset)
    candidates = list(ds_dir.glob(f"binary_reactions_{dataset}*.csv"))
    if not candidates:
        return pd.DataFrame()
    csv_path = None
    for c in candidates:
        if "unordered" in c.name:
            csv_path = c
            break
    if csv_path is None:
        csv_path = candidates[0]
    df = pd.read_csv(csv_path)
    if "odor_sent" not in df.columns or "during_hit" not in df.columns:
        return pd.DataFrame()
    return df


def _fisher_test_per_odor(
    train_raw: pd.DataFrame,
    ctrl_raw: pd.DataFrame,
    odors: Sequence[str],
) -> dict[str, float]:
    """Run Fisher's exact test for each odor, return {odor: p_value}."""
    results = {}
    for odor in odors:
        t = train_raw[train_raw["odor_sent"].str.casefold() == odor.casefold()]
        c = ctrl_raw[ctrl_raw["odor_sent"].str.casefold() == odor.casefold()]
        if t.empty or c.empty:
            results[odor] = 1.0
            continue
        # 2x2 table: [[train_react, train_no], [ctrl_react, ctrl_no]]
        t_react = int(t["during_hit"].sum())
        t_total = len(t)
        c_react = int(c["during_hit"].sum())
        c_total = len(c)
        table = [[t_react, t_total - t_react],
                 [c_react, c_total - c_react]]
        _, p = fisher_exact(table)
        results[odor] = p
    return results


def _draw_significance_brackets(
    ax: plt.Axes,
    x_positions: np.ndarray,
    bar_w: float,
    merged: pd.DataFrame,
    p_values: dict[str, float],
) -> None:
    """Draw bracket + stars between each train/control bar pair."""
    for i, (_, row) in enumerate(merged.iterrows()):
        odor = str(row["odor"])
        p = p_values.get(odor, 1.0)
        stars = _sig_stars(p)

        # Height of bracket: well above the taller bar's annotation
        rate_train = float(row["rate_train"])
        rate_ctrl = float(row["rate_ctrl"])
        top = max(rate_train, rate_ctrl)

        # Raise bracket high above annotations (% + n=X text takes ~18 units)
        bracket_y = top + 22
        tip_y = bracket_y - 2  # small downward ticks

        x_left = x_positions[i] - bar_w / 2
        x_right = x_positions[i] + bar_w / 2

        # Horizontal line
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [tip_y, bracket_y, bracket_y, tip_y],
            color="black", linewidth=0.9, clip_on=False,
        )
        # Stars text
        ax.text(
            x_positions[i], bracket_y + 0.5, stars,
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )


# ---------------------------------------------------------------------------
# Load pre-computed reaction-rate stats from existing binary CSVs
# ---------------------------------------------------------------------------

def _load_rates_from_binary_csv(
    out_dir: Path,
    dataset: str,
    dataset_canon: str,
    *,
    include_hexanol: bool,
) -> pd.DataFrame:
    """Build a stats DataFrame (odor, rate, num_trials, num_reactions, is_trained)
    from an existing ``binary_reactions_*_unordered.csv``."""

    ds_dir = resolve_dataset_output_dir(out_dir, dataset)
    candidates = list(ds_dir.glob(f"binary_reactions_{dataset}*.csv"))
    if not candidates:
        print(f"[WARN] No binary CSV found for {dataset} in {ds_dir}")
        return pd.DataFrame()

    csv_path = None
    for c in candidates:
        if "unordered" in c.name:
            csv_path = c
            break
    if csv_path is None:
        csv_path = candidates[0]

    print(f"[INFO] Reading rates from {csv_path}")
    df = pd.read_csv(csv_path)

    if df.empty or "odor_sent" not in df.columns or "during_hit" not in df.columns:
        print(f"[WARN] Binary CSV for {dataset} is empty or missing columns")
        return pd.DataFrame()

    hexanol_label = "Hexanol"
    if not include_hexanol:
        df = df[df["odor_sent"].str.strip().str.casefold() != hexanol_label.casefold()]

    stats = (
        df.groupby("odor_sent")["during_hit"]
        .agg(num_reactions="sum", num_trials="size")
        .reset_index()
        .rename(columns={"odor_sent": "odor"})
    )
    stats["rate"] = np.where(
        stats["num_trials"] > 0,
        (stats["num_reactions"] / stats["num_trials"]) * 100.0,
        0.0,
    )

    highlight = _trained_label(dataset_canon)
    stats["is_trained"] = stats["odor"].str.casefold() == highlight.casefold()

    stats["_key"] = stats["odor"].map(
        lambda v: REACTION_RATE_ODOR_INDEX.get(str(v).casefold(), len(REACTION_RATE_ODOR_INDEX))
    )
    stats = stats.sort_values(["_key", "odor"]).drop(columns="_key").reset_index(drop=True)
    return stats


# ---------------------------------------------------------------------------
# Grouped bar-chart: training (dark blue) + control (gray)
# ---------------------------------------------------------------------------

def plot_training_vs_control_bars(
    ax: plt.Axes,
    training_stats: pd.DataFrame,
    control_stats: pd.DataFrame,
    *,
    title: str,
    p_values: dict[str, float] | None = None,
) -> None:
    """Side-by-side bars: dark-blue training, gray control, per odor."""

    merged = pd.merge(
        training_stats[["odor", "rate", "num_trials", "is_trained"]],
        control_stats[["odor", "rate", "num_trials"]],
        on="odor",
        how="outer",
        suffixes=("_train", "_ctrl"),
    ).fillna(0)

    order_map = {odor: i for i, odor in enumerate(training_stats["odor"])}
    merged["_sort"] = merged["odor"].map(order_map).fillna(999)
    merged = merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    n = len(merged)
    x = np.arange(n)
    bar_w = 0.35

    train_color = "#1a3a6b"   # dark blue
    ctrl_color = "#b0b0b0"    # gray

    bars_train = ax.bar(
        x - bar_w / 2,
        merged["rate_train"].to_numpy(float),
        width=bar_w,
        color=[train_color if bool(t) else "#4a7fbf" for t in merged["is_trained"]],
        edgecolor="black",
        linewidth=0.75,
        label="Training",
    )
    bars_ctrl = ax.bar(
        x + bar_w / 2,
        merged["rate_ctrl"].to_numpy(float),
        width=bar_w,
        color=ctrl_color,
        edgecolor="black",
        linewidth=0.75,
        label="Control",
    )

    labels = [
        str(odor).upper() if bool(is_trained) else str(odor)
        for odor, is_trained in zip(merged["odor"], merged["is_trained"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for tick, is_trained in zip(ax.get_xticklabels(), merged["is_trained"]):
        if bool(is_trained):
            tick.set_color(train_color)
            tick.set_weight("bold")

    # Y-axis: 0-100% with headroom for annotations + brackets
    ax.set_ylim(0.0, 110)
    ax.set_ylabel("PER %")
    ax.set_xlabel("Odor")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.margins(x=0.04)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    def _annotate(bars, rate_col, trials_col):
        for bar, (_, row) in zip(bars, merged.iterrows()):
            rate = float(row[rate_col])
            trials = int(row[trials_col])
            if trials == 0:
                continue
            text_y = rate + 1.5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{rate:.0f}%\n(n={trials})",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    _annotate(bars_train, "rate_train", "num_trials_train")
    _annotate(bars_ctrl, "rate_ctrl", "num_trials_ctrl")

    # Draw significance brackets if p-values provided
    if p_values:
        _draw_significance_brackets(ax, x, bar_w, merged, p_values)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_training_vs_control_matrices(cfg: SpreadsheetMatrixConfig) -> None:
    # Load and prepare predictions (same as reaction_matrix_from_spreadsheet)
    df = pd.read_csv(cfg.csv_path)
    required = {"dataset", "fly", "fly_number", "trial_label", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = _filter_trial_types(df, allowed=("testing",))
    if df.empty:
        raise RuntimeError("No testing trials found in predictions CSV.")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["trial_label"] = df["trial_label"].astype(str).str.strip()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = _normalise_fly_columns(df)
    df["_non_reactive"] = compute_non_reactive_flags(df)

    flagged_mask = df["_non_reactive"].astype(bool)
    if flagged_mask.any():
        flagged = df.loc[flagged_mask, ["dataset", "fly", "fly_number"]].drop_duplicates()
        summaries = ", ".join(
            f"{r.dataset}::{r.fly}::{r.fly_number}" for r in flagged.itertuples(index=False)
        )
        print(f"[analysis] excluding non-reactive flies: {summaries}")
        df = df.loc[~flagged_mask].copy()

    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)
    df["during_hit"] = df["prediction"].fillna(0).astype(int)
    # Drop duplicate rows created by trial label normalization (e.g.
    # testing_1_fly1_distances_... and testing_1_fly1_angle_... both
    # map to testing_1 with identical predictions).
    df = df.drop_duplicates(subset=["dataset", "fly", "fly_number", "trial"], keep="first")
    df = _normalise_fly_columns(df)

    # --- Exclude flagged flies from matrix (before heatmap is built) ---
    df["_non_reactive"] = compute_non_reactive_flags(
        df, flagged_flies_csv=cfg.flagged_flies_csv
    )
    flagged_mask = df["_non_reactive"].astype(bool)
    if flagged_mask.any():
        flagged = df.loc[flagged_mask, ["dataset_canon", "fly", "fly_number"]].drop_duplicates()
        summaries = ", ".join(
            f"{r.dataset_canon}::{r.fly}::{r.fly_number}" for r in flagged.itertuples(index=False)
        )
        print(f"[INFO] Excluding non-reactive flies from matrix: {summaries}")
        df = df.loc[~flagged_mask].copy()

    # Same colormap as reaction_matrix_from_spreadsheet: white/black
    cmap = ListedColormap(["white", "black"])
    cmap.set_bad(color="0.7")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    present = df["dataset_canon"].unique().tolist()
    saved_unordered_pngs: list[Path] = []

    for order in cfg.trial_orders:
        order_suffix = _order_suffix(order)

        for train_ds, ctrl_ds in TRAINING_CONTROL_PAIRS.items():
            # Skip training datasets with no control data
            if train_ds in ("ACV-Training", "3OCT-Training"):
                print(f"[INFO] {train_ds} has no control data, skipping train vs control plots.")
                continue
            if train_ds not in present:
                print(f"[INFO] {train_ds} not in predictions CSV, skipping.")
                continue

            # --- Build the MATRIX from the training dataset ---
            subset = df[df["dataset_canon"] == train_ds].copy()
            subset = _normalise_fly_columns(subset)
            drop_mask = subset["trial"].apply(_is_testing_11_label)
            if drop_mask.any():
                subset = subset.loc[~drop_mask].copy()
                if subset.empty:
                    continue

            flagged_pairs = set()
            fm = non_reactive_mask(subset)
            if fm.any():
                flagged_pairs = {
                    (r.fly, r.fly_number)
                    for r in subset[fm][["fly", "fly_number"]].drop_duplicates().itertuples(index=False)
                }
                fly_pair_series = subset[["fly", "fly_number"]].apply(tuple, axis=1)
                subset = subset.loc[~fly_pair_series.isin(flagged_pairs)]
                if subset.empty:
                    continue

            fly_pairs = sorted(
                [(r.fly, r.fly_number)
                 for r in subset[["fly", "fly_number"]].drop_duplicates().itertuples(index=False)],
                key=lambda p: _fly_sort_key(*p),
            )
            trial_list = _drop_testing_11(_trial_order_for(list(subset["trial"].unique()), order))
            pretty_labels = [_display_odor(train_ds, t) for t in trial_list]

            during_matrix = np.full((len(fly_pairs), len(trial_list)), np.nan, dtype=float)
            fly_map = {p: i for i, p in enumerate(fly_pairs)}
            trial_map = {t: i for i, t in enumerate(trial_list)}
            for _, row in subset.iterrows():
                during_matrix[fly_map[(row["fly"], row["fly_number"])], trial_map[row["trial"]]] = int(row["during_hit"])

            # --- Load PRE-COMPUTED reaction rates from existing binary CSVs ---
            train_rate = _load_rates_from_binary_csv(
                cfg.out_dir, train_ds, train_ds,
                include_hexanol=cfg.include_hexanol,
            )
            ctrl_rate = _load_rates_from_binary_csv(
                cfg.out_dir, ctrl_ds, ctrl_ds,
                include_hexanol=cfg.include_hexanol,
            )

            # --- Fisher's exact test per odor ---
            p_values: dict[str, float] = {}
            train_raw = _load_raw_binary_csv(cfg.out_dir, train_ds)
            ctrl_raw = _load_raw_binary_csv(cfg.out_dir, ctrl_ds)
            if not train_raw.empty and not ctrl_raw.empty:
                all_odors = list(train_rate["odor"]) if not train_rate.empty else []
                p_values = _fisher_test_per_odor(train_raw, ctrl_raw, all_odors)
                for odor, p in p_values.items():
                    print(f"  Fisher's exact: {odor:25s} p={p:.4f} {_sig_stars(p)}")

            # --- Figure layout ---
            odor_label = DISPLAY_LABEL.get(train_ds, train_ds)
            trained_display = DISPLAY_LABEL.get(train_ds, train_ds)
            n_flies = len(fly_pairs)
            n_trials = len(trial_list)

            base_w = max(10.0, 0.70 * n_trials + 6.0)
            base_h = max(5.0, n_flies * 0.26 + 3.8)
            fig_w = base_w
            gap_scale = 0.6
            fig_h = base_h + cfg.row_gap * cfg.height_per_gap_in * gap_scale + cfg.bottom_shift_in
            xtick_fs = 9 if n_trials <= 10 else (8 if n_trials <= 16 else 7)

            with plt.rc_context(_RC_CONTEXT):
                fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
                gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.25], hspace=cfg.row_gap * gap_scale)

                ax_mat = fig.add_subplot(gs[0, 0])
                ax_bar = fig.add_subplot(gs[1, 0])

                # Matrix heatmap (white/black, same as original)
                ax_mat.imshow(during_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
                ax_mat.set_title(_matrix_title(train_ds), fontsize=14, weight="bold")
                _style_trained_xticks(ax_mat, pretty_labels, trained_display, xtick_fs)
                ax_mat.set_yticks([])
                ax_mat.set_ylabel(f"{n_flies} Flies", fontsize=11)
                for idx, pair in enumerate(fly_pairs):
                    if pair in flagged_pairs:
                        ax_mat.text(
                            -0.35, idx, "*",
                            ha="right", va="center", color="red",
                            fontsize=12, fontweight="bold", clip_on=False,
                        )

                # Bar chart: training vs control
                if not train_rate.empty and not ctrl_rate.empty:
                    plot_training_vs_control_bars(
                        ax_bar, train_rate, ctrl_rate,
                        title=f"Reaction Rates \u2013 {odor_label} (Training vs Control)",
                        p_values=p_values if p_values else None,
                    )
                elif not train_rate.empty:
                    from scripts.analysis.envelope_visuals import plot_reaction_rate_bars
                    plot_reaction_rate_bars(ax_bar, train_rate, title="Reaction Rates by Odor")
                else:
                    ax_bar.text(0.5, 0.5, "No odors available for rate summary",
                                ha="center", va="center", fontsize=11, transform=ax_bar.transAxes)
                    ax_bar.set_axis_off()

                # Shift bottom panel down
                shift_frac = cfg.bottom_shift_in / fig_h if fig_h else 0.0
                pos = ax_bar.get_position()
                ax_bar.set_position([pos.x0, max(0.05, pos.y0 - shift_frac), pos.width, pos.height])

                # Save
                odor_dir = resolve_dataset_output_dir(cfg.out_dir, train_ds)
                png_name = (
                    f"reaction_matrix_train_vs_ctrl_{train_ds.replace(' ', '_')}"
                    f"_{int(cfg.after_window_sec)}_latency_{cfg.latency_sec:.3f}s"
                )
                if order_suffix != "observed":
                    png_name += f"_{order_suffix}"
                png_path = odor_dir / f"{png_name}.png"
                if should_write(png_path, cfg.overwrite):
                    fig.savefig(png_path, dpi=300, bbox_inches="tight")
                    print(f"[SAVED] {png_path}")
                    if "unordered" in png_name:
                        saved_unordered_pngs.append(png_path)

                plt.close(fig)

    # --- Symlink all unordered plots into Figures dir ---
    figures_dir = Path("/home/ramanlab/Documents/cole/Results/Figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for png_path in saved_unordered_pngs:
        dest = figures_dir / png_path.name
        try:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(png_path)
            print(f"[SYMLINK] {dest} -> {png_path}")
        except OSError:
            import shutil
            shutil.copy2(png_path, dest)
            print(f"[COPIED] {dest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-path", type=Path, required=True,
                   help="Path to model_predictions.csv (same one used by reaction_matrix_from_spreadsheet)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Matrix-PER-Reactions-Model dir (reads binary CSVs from here, writes new PNGs here)")
    p.add_argument("--latency-sec", type=float, default=2.15)
    p.add_argument("--after-window-sec", type=float, default=30.0)
    p.add_argument("--row-gap", type=float, default=0.6)
    p.add_argument("--height-per-gap-in", type=float, default=3.0)
    p.add_argument("--bottom-shift-in", type=float, default=0.5)
    p.add_argument("--trial-order", action="append", choices=("observed", "trained-first"))
    p.add_argument("--exclude-hexanol", action="store_true")
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--flagged-flies-csv", type=str, default="",
                   help="Path to flagged-flies-truth CSV for excluding non-reactive flies.")
    args = p.parse_args(argv)

    trial_orders = tuple(args.trial_order) if args.trial_order else ("observed", "trained-first")

    cfg = SpreadsheetMatrixConfig(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        latency_sec=args.latency_sec,
        after_window_sec=args.after_window_sec,
        row_gap=args.row_gap,
        height_per_gap_in=args.height_per_gap_in,
        bottom_shift_in=args.bottom_shift_in,
        trial_orders=trial_orders,
        include_hexanol=not args.exclude_hexanol,
        overwrite=args.overwrite,
        flagged_flies_csv=args.flagged_flies_csv or "",
    )

    generate_training_vs_control_matrices(cfg)
    print("[DONE] All training-vs-control matrix plots generated.")


if __name__ == "__main__":
    main()
