#!/usr/bin/env python3
"""
Standalone script to export reaction rates summary CSV from existing matrix data.
Run this to generate the CSV files without rerunning the entire pipeline.

Usage:
    python scripts/analysis/export_reaction_rates.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from fbpipe.config import load_raw_config
from scripts.analysis.envelope_visuals import (
    ODOR_ORDER,
    _canon_dataset,
    _display_odor,
    _order_suffix,
    _normalise_fly_columns,
    _trial_num,
    non_reactive_mask,
    reaction_rate_stats_from_rows,
)

def export_reaction_rates_from_matrix(
    matrix_npy: Path,
    codes_json: Path,
    out_dir: Path,
    include_hexanol: bool = True,
    trial_orders: list[str] = None,
    latency_sec: float = 0.0,
    fps_default: float = 40.0,
    before_sec: float = 30.0,
    during_sec: float = 30.0,
    after_window_sec: float = 30.0,
    threshold_std_mult: float = 4.0,
    min_samples_over: int = 20,
) -> None:
    """
    Generate reaction rates summary CSV from existing matrix data.

    This function:
    1. Loads the matrix and codes
    2. Scores each trial for reactions
    3. Collects reaction rate statistics for each dataset
    4. Exports to CSV files
    """
    import json

    if trial_orders is None:
        trial_orders = ["trained-first", "observed"]

    print(f"[INFO] Loading matrix from {matrix_npy}")
    print(f"[INFO] Loading codes from {codes_json}")

    # Load matrix data
    env_data = np.load(matrix_npy)
    with open(codes_json, "r", encoding="utf-8") as f:
        maps = json.load(f)

    row_code_to_info = {int(k): v for k, v in maps["row_code_to_info"].items()}

    # Build DataFrame
    records = []
    for row_idx, row_code in enumerate(maps["row_codes"]):
        info = row_code_to_info[row_code]
        records.append({
            "dataset": _canon_dataset(info["dataset"]),
            "fly": info["fly"],
            "fly_number": str(info.get("fly_number", "1")),
            "trial_label": info["trial_label"],
            "trial_type": info.get("trial_type", "testing").strip().lower(),
            "fps": float(info.get("fps", fps_default)),
            "_non_reactive": bool(info.get("non_reactive", False)),
            "_row_idx": row_idx,
        })

    df = pd.DataFrame(records)

    print(f"[INFO] Loaded {len(df)} trials from matrix")
    print(f"[INFO] Unique datasets: {sorted(df['dataset'].unique())}")

    # Filter to testing trials
    df_testing = df[df["trial_type"] == "testing"].copy()
    print(f"[INFO] Filtered to {len(df_testing)} testing trials")

    # Score trials
    def _score_trial(env: np.ndarray, fps: float) -> tuple[int, int]:
        """Score a trial for reactions during and after odor presentation."""
        latency_frames = int(latency_sec * fps)
        before_frames = int(before_sec * fps)
        during_frames = int(during_sec * fps)
        after_frames = int(after_window_sec * fps)

        odor_on_frame = before_frames + latency_frames
        odor_off_frame = odor_on_frame + during_frames
        after_end_frame = odor_off_frame + after_frames

        before_seg = env[:before_frames]
        during_seg = env[odor_on_frame:odor_off_frame]
        after_seg = env[odor_off_frame:after_end_frame]

        if len(before_seg) == 0 or len(during_seg) == 0:
            return 0, 0

        before_mean = float(np.nanmean(before_seg))
        before_std = float(np.nanstd(before_seg))
        threshold = before_mean + threshold_std_mult * before_std

        during_over = np.sum(during_seg > threshold)
        during_hit = 1 if during_over >= min_samples_over else 0

        after_hit = 0
        if len(after_seg) > 0:
            after_over = np.sum(after_seg > threshold)
            after_hit = 1 if after_over >= min_samples_over else 0

        return during_hit, after_hit

    print(f"[INFO] Scoring trials...")
    scores = []
    for _, row in df_testing.iterrows():
        row_idx = int(row["_row_idx"])
        env = env_data[row_idx]
        during_hit, after_hit = _score_trial(env, float(row["fps"]))
        scores.append({
            "dataset": row["dataset"],
            "fly": row["fly"],
            "fly_number": row["fly_number"],
            "trial": row["trial_label"],
            "trial_num": _trial_num(row["trial_label"]),
            "during_hit": during_hit,
            "after_hit": after_hit,
            "_non_reactive": bool(row["_non_reactive"]),
        })

    scores_df = pd.DataFrame(scores)
    print(f"[INFO] Scored {len(scores_df)} trials")

    # Collect reaction rate statistics
    all_rate_stats = []

    present = scores_df["dataset"].unique().tolist()
    ordered_present = [odor for odor in ODOR_ORDER if odor in present]
    extras = sorted(odor for odor in present if odor not in ODOR_ORDER)

    for order in trial_orders:
        order_suffix = _order_suffix(order)
        print(f"[INFO] Processing trial order: {order}")

        for odor in ordered_present + extras:
            subset = scores_df[scores_df["dataset"] == odor].copy()
            if subset.empty:
                continue

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
                    print(f"[INFO] Skipping {odor} ({order}): all flies non-reactive")
                    continue

            # Calculate reaction rates
            rate_context = f"{odor} ({order_suffix})"
            try:
                rate_stats = reaction_rate_stats_from_rows(
                    subset,
                    odor,
                    include_hexanol=include_hexanol,
                    context=rate_context,
                    trial_col="trial",
                    reaction_col="during_hit",
                )

                # Collect stats
                for _, row in rate_stats.iterrows():
                    all_rate_stats.append({
                        "dataset": odor,
                        "trial_order": order,
                        "odor_sent": str(row["odor"]),
                        "reaction_rate": float(row["rate"]),
                        "num_reactions": int(row["num_reactions"]),
                        "num_trials": int(row["num_trials"]),
                    })
            except RuntimeError as e:
                print(f"[WARN] Could not compute rates for {odor} ({order}): {e}")
                continue

    if not all_rate_stats:
        print("[ERROR] No reaction rate statistics collected!")
        return

    # Export to CSV
    stats_df = pd.DataFrame(all_rate_stats)
    print(f"[INFO] Collected {len(stats_df)} reaction rate entries")

    out_dir.mkdir(parents=True, exist_ok=True)

    for order in trial_orders:
        order_stats = stats_df[stats_df["trial_order"] == order].copy()
        if order_stats.empty:
            continue

        # Create pivot table: rows = datasets, columns = odor_sent, values = reaction_rate
        pivot = order_stats.pivot_table(
            index="dataset",
            columns="odor_sent",
            values="reaction_rate",
            aggfunc="first"
        )

        # Sort datasets by ODOR_ORDER
        ordered_datasets = [d for d in ODOR_ORDER if d in pivot.index]
        extra_datasets = sorted(d for d in pivot.index if d not in ODOR_ORDER)
        all_datasets = ordered_datasets + extra_datasets
        pivot = pivot.loc[all_datasets]

        # Reset index to make dataset a column
        pivot = pivot.reset_index()

        # Save to CSV
        order_suffix = _order_suffix(order)
        csv_filename = f"reaction_rates_summary_{order_suffix}.csv"
        csv_path = out_dir / csv_filename

        pivot.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"[SUCCESS] Exported {csv_path}")
        print(f"          Shape: {pivot.shape[0]} datasets x {pivot.shape[1]-1} odors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export reaction rate summaries from matrix artifacts.")
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument("--matrix-npy", default=None, help="Path to envelope_matrix_float16.npy")
    parser.add_argument("--codes-json", default=None, help="Path to code_maps.json")
    parser.add_argument("--out-dir", default=None, help="Output directory for CSV summaries")
    parser.add_argument(
        "--trial-order",
        action="append",
        dest="trial_orders",
        default=None,
        help="Trial order to export (repeatable).",
    )
    args = parser.parse_args()

    config_data = load_raw_config(args.config)
    combined_cfg = config_data.get("analysis", {}).get("combined", {})
    matrices_cfg = combined_cfg.get("matrices", {}) if isinstance(combined_cfg, dict) else {}

    matrix_npy_value = args.matrix_npy or matrices_cfg.get("matrix_npy", "")
    codes_json_value = args.codes_json or matrices_cfg.get("codes_json", "")
    out_dir_value = args.out_dir or matrices_cfg.get("out_dir", "")
    matrix_npy = Path(matrix_npy_value).expanduser() if matrix_npy_value else None
    codes_json = Path(codes_json_value).expanduser() if codes_json_value else None
    out_dir = Path(out_dir_value).expanduser() if out_dir_value else None
    trial_orders = args.trial_orders or matrices_cfg.get("trial_orders") or ["trained-first", "observed"]

    print("=" * 70)
    print("REACTION RATES SUMMARY EXPORT")
    print("=" * 70)

    if not matrix_npy:
        print("[ERROR] Matrix path not configured. Provide --matrix-npy or set analysis.combined.matrices.matrix_npy.")
        sys.exit(1)
    if not codes_json:
        print("[ERROR] Codes JSON not configured. Provide --codes-json or set analysis.combined.matrices.codes_json.")
        sys.exit(1)
    if not out_dir:
        print("[ERROR] Output directory not configured. Provide --out-dir or set analysis.combined.matrices.out_dir.")
        sys.exit(1)

    if not matrix_npy.exists():
        print(f"[ERROR] Matrix file not found: {matrix_npy}")
        sys.exit(1)

    if not codes_json.exists():
        print(f"[ERROR] Codes file not found: {codes_json}")
        sys.exit(1)

    export_reaction_rates_from_matrix(
        matrix_npy=matrix_npy,
        codes_json=codes_json,
        out_dir=out_dir,
        include_hexanol=True,
        trial_orders=list(trial_orders),
    )

    print("=" * 70)
    print("DONE!")
    print("=" * 70)
