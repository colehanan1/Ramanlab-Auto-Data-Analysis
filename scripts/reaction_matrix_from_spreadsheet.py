"""Generate reaction matrices from a spreadsheet of manual predictions.

This utility mirrors the figure layout and file naming conventions used by
``scripts.envelope_visuals`` but sources binary reaction calls from a CSV file
instead of scoring Hilbert envelopes.  The spreadsheet must contain the
following columns::

    dataset, fly, fly_number, trial_label, prediction

Predictions are treated as binary where ``1`` indicates a reaction (rendered as
black squares) and ``0`` indicates no reaction (white squares).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from envelope_visuals import (
    DISPLAY_LABEL,
    ODOR_ORDER,
    resolve_dataset_output_dir,
    should_write,
    _canon_dataset,
    _compute_category_counts,
    _display_odor,
    _fly_row_label,
    _fly_sort_key,
    _normalise_fly_columns,
    _order_suffix,
    _plot_category_counts,
    _style_trained_xticks,
    _trial_num,
    _trial_order_for,
)


_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
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
    overwrite: bool = False


def _load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"dataset", "fly", "fly_number", "trial_label", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["trial_label"] = df["trial_label"].astype(str).str.strip()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = _normalise_fly_columns(df)

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


def generate_reaction_matrices_from_csv(cfg: SpreadsheetMatrixConfig) -> None:
    df = _load_predictions(cfg.csv_path)
    if df.empty:
        raise RuntimeError("Spreadsheet did not contain any rows to plot.")

    cmap = ListedColormap(["white", "black"])
    cmap.set_bad(color="0.7")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    present = df["dataset_canon"].unique().tolist()
    ordered_present = [odor for odor in ODOR_ORDER if odor in present]
    extras = sorted(odor for odor in present if odor not in ODOR_ORDER)

    for order in cfg.trial_orders:
        order_suffix = _order_suffix(order)
        for odor in ordered_present + extras:
            subset = df[df["dataset_canon"] == odor]
            if subset.empty:
                continue

            subset = subset.copy()
            subset = _normalise_fly_columns(subset)
            fly_pairs = [
                (row.fly, row.fly_number)
                for row in subset[["fly", "fly_number"]].drop_duplicates().itertuples(index=False)
            ]
            fly_pairs.sort(key=lambda pair: _fly_sort_key(*pair))
            trial_list = _trial_order_for(list(subset["trial"].unique()), order)
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
            fig_h = base_h + cfg.row_gap * cfg.height_per_gap_in + cfg.bottom_shift_in

            xtick_fs = 9 if n_trials <= 10 else (8 if n_trials <= 16 else 7)

            during_counts = _compute_category_counts(
                during_matrix, pretty_labels, trained_display, cfg.include_hexanol
            )

            with plt.rc_context(_RC_CONTEXT):
                fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
                gs = gridspec.GridSpec(
                    2,
                    1,
                    height_ratios=[3.0, 1.25],
                    hspace=cfg.row_gap,
                )

                ax_during = fig.add_subplot(gs[0, 0])
                ax_dc = fig.add_subplot(gs[1, 0])

                ax_during.imshow(
                    during_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
                )
                ax_during.set_title(
                    f"{odor_label} — During (Spreadsheet)", fontsize=14, weight="bold"
                )
                _style_trained_xticks(ax_during, pretty_labels, trained_display, xtick_fs)
                ax_during.set_yticks([])
                ax_during.set_ylabel(f"{n_flies} Flies", fontsize=11)

                _plot_category_counts(ax_dc, during_counts, n_flies, "During — Fly Reaction Categories")

                legend_handles = [
                    Patch(facecolor="black", edgecolor="black", label="Prediction = 1"),
                    Patch(facecolor="white", edgecolor="black", label="Prediction = 0"),
                ]
                ax_during.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=9)

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
                    export_cols = [
                        "dataset",
                        "fly",
                        "fly_number",
                        "trial_num",
                        "odor_sent",
                        "during_hit",
                        "after_hit",
                    ]
                    export_path = odor_dir / f"binary_reactions_{odor.replace(' ', '_')}_{order_suffix}.csv"
                    if should_write(export_path, cfg.overwrite):
                        export.to_csv(export_path, columns=export_cols, index=False)

                plt.close(fig)


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
        help="Exclude Optogenetics Hexanol from 'other' reaction counts.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")
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
    )
    generate_reaction_matrices_from_csv(cfg)


if __name__ == "__main__":
    main()
