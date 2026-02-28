"""Generate reaction matrices from a spreadsheet of manual predictions.

This utility mirrors the figure layout and file naming conventions used by
``scripts.analysis.envelope_visuals`` but sources binary reaction calls from a CSV file
instead of scoring Hilbert envelopes.  The spreadsheet must contain the
following columns::

    dataset, fly, fly_number, trial_label, prediction

Predictions are treated as binary where ``1`` indicates a reaction (rendered as
black squares) and ``0`` indicates no reaction (white squares).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import (
    DISPLAY_LABEL,
    ODOR_ORDER,
    compute_non_reactive_flags,
    non_reactive_mask,
    plot_reaction_rate_bars,
    reaction_rate_stats_from_rows,
    resolve_dataset_output_dir,
    should_write,
    _canon_dataset,
    _display_odor,
    _fly_row_label,
    _fly_sort_key,
    _normalise_fly_columns,
    _matrix_title,
    NON_REACTIVE_SPAN_PX,
    _order_suffix,
    _style_trained_xticks,
    _trial_num,
    _trial_order_for,
    _drop_testing_11,
)


_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}


@dataclass
class SpreadsheetMatrixConfig:
    """Configuration for building reaction matrices from a CSV spreadsheet."""

    csv_path: Path
    out_dir: Path
    latency_sec: float
    after_window_sec: float = 30.0
    row_gap: float = 0.6
    height_per_gap_in: float = 3.0
    bottom_shift_in: float = 0.5
    trial_orders: Sequence[str] = field(default_factory=lambda: ("observed", "trained-first"))
    include_hexanol: bool = True
    overwrite: bool = True
    non_reactive_threshold: float | None = None
    flagged_flies_csv: str = ""


def _filter_trial_types(
    df: pd.DataFrame, allowed: Iterable[str] = ("testing",)
) -> pd.DataFrame:
    if "trial_type" not in df.columns:
        return df

    allowed_normalised = {str(value).strip().lower() for value in allowed}
    if not allowed_normalised:
        return df

    mask = (
        df["trial_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(allowed_normalised)
    )
    return df.loc[mask].copy()


def _load_predictions(
    csv_path: Path, *, threshold: float, flagged_flies_csv: str = ""
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"dataset", "fly", "fly_number", "trial_label", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df = _filter_trial_types(df, allowed=("testing",))
    if df.empty:
        raise RuntimeError("Predictions CSV did not contain any testing trials to plot.")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["trial_label"] = df["trial_label"].astype(str).str.strip()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = _normalise_fly_columns(df)
    df["_non_reactive"] = compute_non_reactive_flags(
        df, threshold=threshold, flagged_flies_csv=flagged_flies_csv
    )

    flagged_mask = df["_non_reactive"].astype(bool)
    if flagged_mask.any():
        flagged = df.loc[flagged_mask, ["dataset", "fly", "fly_number"]]
        summaries = ", ".join(
            f"{row.dataset}::{row.fly}::{row.fly_number}"
            for row in flagged.itertuples(index=False)
        )
        source = "flagged-flies CSV" if flagged_flies_csv else f"span ≤ {threshold:g}px"
        print(
            f"[INFO] reaction_matrix_csv: excluding non-reactive flies ({source}): {summaries}"
        )
        df = df.loc[~flagged_mask].copy()
        if df.empty:
            raise RuntimeError("All rows were flagged non-reactive; nothing to plot.")

    invalid = df["prediction"].dropna().unique()
    if not set(invalid).issubset({0, 1}):
        raise ValueError(
            "Predictions must be binary (0 or 1). Found values: "
            + ", ".join(sorted(str(val) for val in set(invalid) - {0, 1}))
        )

    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"]
    df["trial_num"] = df["trial"].apply(_trial_num)
    df["during_hit"] = df["prediction"].fillna(0).astype(int)
    df["after_hit"] = df["during_hit"]
    return df


def _collect_unordered_matrices(out_dir: Path) -> None:
    """Collect all unordered matrix PNGs into a single 'all' folder."""
    all_dir = out_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)

    # Find all PNG files ending with "unordered.png"
    for png_file in out_dir.rglob("*unordered.png"):
        # Skip files in the "all" directory itself
        if png_file.parent == all_dir:
            continue

        # Create a symlink or copy to the all folder
        dest_path = all_dir / png_file.name
        # Use relative symlink to original file
        try:
            # Remove existing symlink if it exists
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            # Create symlink - use relative path from all_dir to png_file
            rel_path = os.path.relpath(png_file, all_dir)
            dest_path.symlink_to(rel_path)
        except (OSError, NotImplementedError):
            # Fallback to copy if symlinks not supported
            import shutil
            shutil.copy2(png_file, dest_path)


def generate_reaction_matrices_from_csv(cfg: SpreadsheetMatrixConfig) -> None:
    span_threshold = float(cfg.non_reactive_threshold) if cfg.non_reactive_threshold is not None else float(NON_REACTIVE_SPAN_PX)
    df = _load_predictions(
        cfg.csv_path, threshold=span_threshold, flagged_flies_csv=cfg.flagged_flies_csv
    )
    if df.empty:
        raise RuntimeError("Spreadsheet did not contain any rows to plot.")

    cmap = ListedColormap(["white", "black"])
    cmap.set_bad(color="0.7")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    present = df["dataset_canon"].unique().tolist()
    ordered_present = [odor for odor in ODOR_ORDER if odor in present]
    extras = sorted(odor for odor in present if odor not in ODOR_ORDER)

    # Collect reaction rate statistics for CSV export
    all_rate_stats = []

    for order in cfg.trial_orders:
        order_suffix = _order_suffix(order)
        for odor in ordered_present + extras:
            subset = df[df["dataset_canon"] == odor]
            if subset.empty:
                continue

            subset = subset.copy()
            subset = _normalise_fly_columns(subset)
            flagged_mask = non_reactive_mask(subset)
            flagged_pairs = {
                (row.fly, row.fly_number)
                for row in subset[flagged_mask][["fly", "fly_number"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
            if flagged_pairs:
                fly_pair_series = subset[["fly", "fly_number"]].apply(tuple, axis=1)
                keep_mask = ~fly_pair_series.isin(flagged_pairs)
                subset = subset.loc[keep_mask]
                if subset.empty:
                    print(
                        "[INFO] reaction_matrix_csv: skipping", odor, "because all flies were non-reactive."
                    )
                    continue
            fly_pairs = [
                (row.fly, row.fly_number)
                for row in subset[["fly", "fly_number"]].drop_duplicates().itertuples(index=False)
            ]
            fly_pairs.sort(key=lambda pair: _fly_sort_key(*pair))
            trial_list = _trial_order_for(list(subset["trial"].unique()), order)
            trial_list = _drop_testing_11(trial_list)
            pretty_labels = [_display_odor(odor, trial) for trial in trial_list]

            during_matrix = np.full((len(fly_pairs), len(trial_list)), np.nan, dtype=float)

            fly_map = {pair: idx for idx, pair in enumerate(fly_pairs)}
            trial_map = {trial: idx for idx, trial in enumerate(trial_list)}
            for _, row in subset.iterrows():
                key = (row["fly"], row["fly_number"])
                i = fly_map[key]
                j = trial_map[row["trial"]]
                value = int(row["during_hit"])
                during_matrix[i, j] = value

            odor_label = DISPLAY_LABEL.get(odor, odor)
            trained_display = DISPLAY_LABEL.get(odor, odor)
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
                gs = gridspec.GridSpec(
                    2,
                    1,
                    height_ratios=[3.0, 1.25],
                    hspace=cfg.row_gap * gap_scale,
                )

                ax_during = fig.add_subplot(gs[0, 0])
                ax_dc = fig.add_subplot(gs[1, 0])

                ax_during.imshow(
                    during_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
                )
                ax_during.set_title(_matrix_title(odor), fontsize=14, weight="bold")
                _style_trained_xticks(ax_during, pretty_labels, trained_display, xtick_fs)
                ax_during.set_yticks([])
                ax_during.set_ylabel(f"{n_flies} Flies", fontsize=11)
                for idx, pair in enumerate(fly_pairs):
                    if pair in flagged_pairs:
                        ax_during.text(
                            -0.35,
                            idx,
                            "*",
                            ha="right",
                            va="center",
                            color="red",
                            fontsize=12,
                            fontweight="bold",
                            clip_on=False,
                        )

                try:
                    rate_stats = reaction_rate_stats_from_rows(
                        subset,
                        odor,
                        include_hexanol=cfg.include_hexanol,
                        context=f"{odor_label} (spreadsheet/{order_suffix})",
                        trial_col="trial",
                        reaction_col="during_hit",
                    )
                except RuntimeError:
                    ax_dc.text(
                        0.5,
                        0.5,
                        "No odors available for rate summary",
                        ha="center",
                        va="center",
                        fontsize=11,
                        transform=ax_dc.transAxes,
                    )
                    ax_dc.set_axis_off()
                else:
                    plot_reaction_rate_bars(
                        ax_dc,
                        rate_stats,
                        title="Reaction Rates by Odor",
                    )
                    # Collect rate stats for CSV export
                    for _, row in rate_stats.iterrows():
                        all_rate_stats.append({
                            "dataset": odor,
                            "trial_order": order,
                            "odor_sent": str(row["odor"]),
                            "reaction_rate": float(row["rate"]),
                            "num_reactions": int(row["num_reactions"]),
                            "num_trials": int(row["num_trials"]),
                        })

                shift_frac = cfg.bottom_shift_in / fig_h if fig_h else 0.0
                for axis in (ax_dc,):
                    pos = axis.get_position()
                    new_y0 = max(0.05, pos.y0 - shift_frac)
                    axis.set_position([pos.x0, new_y0, pos.width, pos.height])

                odor_dir = resolve_dataset_output_dir(cfg.out_dir, odor)

                png_name = (
                    f"reaction_matrix_{odor.replace(' ', '_')}_{int(cfg.after_window_sec)}"
                    f"_latency_{cfg.latency_sec:.3f}s"
                )
                if order_suffix != "observed":
                    png_name += f"_{order_suffix}"
                png_path = odor_dir / f"{png_name}.png"
                if should_write(png_path, cfg.overwrite):
                    fig.savefig(png_path, dpi=300, bbox_inches="tight")

                row_key_name = f"row_key_{odor.replace(' ', '_')}_{int(cfg.after_window_sec)}"
                if order_suffix != "observed":
                    row_key_name += f"_{order_suffix}"
                row_key_path = odor_dir / f"{row_key_name}.txt"
                if should_write(row_key_path, cfg.overwrite):
                    with row_key_path.open("w", encoding="utf-8") as fh:
                        for idx, (fly, fly_number) in enumerate(fly_pairs):
                            label = _fly_row_label(fly, fly_number)
                            if (fly, fly_number) in flagged_pairs:
                                label = f"* {label}"
                            fh.write(f"Row {idx}: {label}\n")

                if order == "trained-first":
                    export = subset.copy()
                    export["odor_sent"] = export["trial"].apply(lambda t: _display_odor(odor, t))
                    order_map = {trial: idx for idx, trial in enumerate(trial_list)}
                    export["trial_ord"] = export["trial"].map(order_map).fillna(10**9).astype(int)
                    export = export.sort_values([
                        "fly",
                        "fly_number",
                        "trial_ord",
                        "trial_num",
                        "trial",
                    ])
                    if "probability" in export.columns:
                        export["prob_reaction"] = pd.to_numeric(export["probability"], errors="coerce")
                    elif "prob_reaction" not in export.columns:
                        export["prob_reaction"] = np.nan
                    export_cols = [
                        "dataset",
                        "fly",
                        "fly_number",
                        "trial_num",
                        "odor_sent",
                        "during_hit",
                        "after_hit",
                        "prob_reaction",
                    ]
                    export_path = odor_dir / f"binary_reactions_{odor.replace(' ', '_')}_{order_suffix}.csv"
                    if should_write(export_path, cfg.overwrite):
                        export.to_csv(export_path, columns=export_cols, index=False)

                plt.close(fig)

    # Export aggregated reaction rate statistics to CSV
    if all_rate_stats:
        stats_df = pd.DataFrame(all_rate_stats)

        # For each trial order, create a separate summary CSV
        for order in cfg.trial_orders:
            order_stats = stats_df[stats_df["trial_order"] == order].copy()
            if order_stats.empty:
                continue

            # Create pivot table: rows = datasets, columns = odor_sent, values = reaction_rate
            pivot = order_stats.pivot_table(
                index="dataset",
                columns="odor_sent",
                values="reaction_rate",
                aggfunc="first"  # Should only be one value per dataset-odor pair
            )

            # Sort datasets by ODOR_ORDER
            ordered_datasets = [d for d in ODOR_ORDER if d in pivot.index]
            extra_datasets = sorted(d for d in pivot.index if d not in ODOR_ORDER)
            all_datasets = ordered_datasets + extra_datasets
            if all_datasets:
                pivot = pivot.loc[all_datasets]

            # Reset index to make dataset a column
            pivot = pivot.reset_index()

            # Replace spaces with underscores in column names for CSV readability
            pivot.columns = [col.replace(' ', '_') for col in pivot.columns]

            # Save to CSV
            order_suffix = _order_suffix(order)
            csv_filename = f"reaction_rates_summary_{order_suffix}.csv"
            csv_path = cfg.out_dir / csv_filename

            if should_write(csv_path, cfg.overwrite):
                pivot.to_csv(csv_path, index=False, float_format="%.4f")
                print(f"[INFO] Exported reaction rate summary to {csv_path}")

    # Collect all unordered matrices into a single 'all' folder
    _collect_unordered_matrices(cfg.out_dir)
    print(f"[INFO] Collected unordered matrices to {cfg.out_dir / 'all'}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True, help="Path to the predictions spreadsheet.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for exported figures.")
    parser.add_argument("--latency-sec", type=float, default=2.15, help="Mean odor transit latency in seconds.")
    parser.add_argument("--after-window-sec", type=float, default=30.0, help="Window length used in filenames (seconds).")
    parser.add_argument("--row-gap", type=float, default=0.6, help="Vertical gap between matrix and bar charts.")
    parser.add_argument(
        "--height-per-gap-in",
        type=float,
        default=3.0,
        help="Figure height added per 1.0 of row gap (inches).",
    )
    parser.add_argument(
        "--bottom-shift-in",
        type=float,
        default=0.5,
        help="Downward shift applied to bar charts (inches).",
    )
    parser.add_argument(
        "--trial-order",
        action="append",
        choices=("observed", "trained-first"),
        help="Trial ordering strategy. Repeat to request multiple variants.",
    )
    parser.add_argument(
        "--exclude-hexanol",
        action="store_true",
        help="Exclude Hexanol from 'other' reaction counts.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")
    parser.add_argument(
        "--non-reactive-threshold",
        type=float,
        default=None,
        help="Override non-reactive span threshold in pixels.",
    )
    parser.add_argument(
        "--flagged-flies-csv",
        type=str,
        default="",
        help="Path to flagged-flies truth CSV for CSV-based exclusion.",
    )
    parser.set_defaults(overwrite=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    trial_orders: Sequence[str] = args.trial_order or ("observed", "trained-first")
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
        non_reactive_threshold=args.non_reactive_threshold,
        flagged_flies_csv=args.flagged_flies_csv or "",
    )
    generate_reaction_matrices_from_csv(cfg)


if __name__ == "__main__":
    main()
