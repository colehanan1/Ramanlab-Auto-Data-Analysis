#!/usr/bin/env python3
"""
Standalone script to export reaction rates summary CSV from wide CSV data.
This is a simpler version that reads from the all_envelope_rows_wide.csv file.

Usage:
    python scripts/analysis/export_reaction_rates_simple.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from fbpipe.config import load_raw_config
from scripts.analysis.envelope_visuals import ODOR_ORDER


def score_trial_from_envelope(
    envelope_values: np.ndarray,
    fps: float,
    latency_sec: float = 0.0,
    before_sec: float = 30.0,
    during_sec: float = 30.0,
    after_window_sec: float = 30.0,
    threshold_std_mult: float = 4.0,
    min_samples_over: int = 20,
) -> tuple[int, int]:
    """Score a trial for reactions during and after odor presentation."""
    latency_frames = int(latency_sec * fps)
    before_frames = int(before_sec * fps)
    during_frames = int(during_sec * fps)
    after_frames = int(after_window_sec * fps)

    odor_on_frame = before_frames + latency_frames
    odor_off_frame = odor_on_frame + during_frames
    after_end_frame = odor_off_frame + after_frames

    before_seg = envelope_values[:before_frames]
    during_seg = envelope_values[odor_on_frame:odor_off_frame]
    after_seg = envelope_values[odor_off_frame:after_end_frame]

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


def compute_reaction_rates(
    df: pd.DataFrame,
    dataset: str,
    trial_col: str = "trial_label",
) -> pd.DataFrame:
    """Compute reaction rates per odor for a dataset."""

    # Map trial labels to odor names
    def get_odor_name(trial_label: str) -> str:
        # Simple mapping from trial number to odor
        # You may need to adjust this based on your actual trial naming
        trial_label = str(trial_label).lower()

        # Common patterns
        if "acv" in trial_label or "vinegar" in trial_label:
            return "Apple Cider Vinegar"
        elif "hex" in trial_label:
            return "Hexanol"
        elif "benz" in trial_label or "benzaldehyde" in trial_label:
            return "Benzaldehyde"
        elif "oct" in trial_label:
            return "3-Octonol"
        elif "eb" in trial_label or "ethyl" in trial_label or "butyrate" in trial_label:
            return "Ethyl Butyrate"
        elif "lin" in trial_label or "linalool" in trial_label:
            return "Linalool"
        elif "cit" in trial_label or "citral" in trial_label:
            return "Citral"
        elif "air" in trial_label:
            return "AIR"
        else:
            # Try to extract from trial name
            return trial_label

    df = df.copy()
    df["odor"] = df[trial_col].apply(get_odor_name)

    # Compute reaction rates per odor
    stats = (
        df.groupby("odor")["during_hit"]
        .agg(num_reactions="sum", num_trials="size")
        .reset_index()
    )
    stats["reaction_rate"] = stats["num_reactions"] / stats["num_trials"]

    return stats


def export_reaction_rates_from_wide_csv(
    wide_csv: Path,
    out_dir: Path,
    latency_sec: float = 0.0,
    fps_default: float = 40.0,
    before_sec: float = 30.0,
    during_sec: float = 30.0,
    after_window_sec: float = 30.0,
    threshold_std_mult: float = 4.0,
    min_samples_over: int = 20,
) -> None:
    """
    Generate reaction rates summary CSV from wide CSV file.
    """
    print(f"[INFO] Loading wide CSV from {wide_csv}")
    df = pd.read_csv(wide_csv)

    print(f"[INFO] Loaded {len(df)} rows")

    # Filter to testing trials
    if "trial_type" in df.columns:
        df = df[df["trial_type"].str.lower().str.strip() == "testing"].copy()
        print(f"[INFO] Filtered to {len(df)} testing trials")

    # Get envelope columns
    env_cols = [c for c in df.columns if c.startswith("dir_val_")]
    if not env_cols:
        print("[ERROR] No envelope columns found (dir_val_*)")
        return

    print(f"[INFO] Found {len(env_cols)} envelope columns")

    # Score trials
    print(f"[INFO] Scoring trials...")
    scores = []
    for idx, row in df.iterrows():
        envelope = row[env_cols].values.astype(float)
        fps = float(row.get("fps", fps_default))

        during_hit, after_hit = score_trial_from_envelope(
            envelope, fps, latency_sec, before_sec, during_sec,
            after_window_sec, threshold_std_mult, min_samples_over
        )

        scores.append({
            "dataset": row["dataset"],
            "fly": row["fly"],
            "fly_number": row["fly_number"],
            "trial_label": row["trial_label"],
            "during_hit": during_hit,
            "after_hit": after_hit,
        })

    scores_df = pd.DataFrame(scores)
    print(f"[INFO] Scored {len(scores_df)} trials")

    # Filter out non-reactive flies if needed
    if "non_reactive_flag" in df.columns:
        non_reactive = df["non_reactive_flag"].fillna(False).astype(bool)
        scores_df = scores_df[~df.index.map(lambda i: non_reactive.iloc[i] if i < len(non_reactive) else False)]
        print(f"[INFO] After filtering non-reactive: {len(scores_df)} trials")

    # Compute reaction rates per dataset
    datasets = sorted(scores_df["dataset"].unique())
    print(f"[INFO] Processing {len(datasets)} datasets: {datasets}")

    all_stats = []
    for dataset in datasets:
        subset = scores_df[scores_df["dataset"] == dataset]
        stats = compute_reaction_rates(subset, dataset)

        for _, row in stats.iterrows():
            all_stats.append({
                "dataset": dataset,
                "odor_sent": row["odor"],
                "reaction_rate": row["reaction_rate"],
                "num_reactions": row["num_reactions"],
                "num_trials": row["num_trials"],
            })

    if not all_stats:
        print("[ERROR] No reaction rate statistics collected!")
        return

    # Create summary table
    summary_df = pd.DataFrame(all_stats)

    # Pivot to create the final format: datasets as rows, odors as columns
    pivot = summary_df.pivot_table(
        index="dataset",
        columns="odor_sent",
        values="reaction_rate",
        aggfunc="first"
    )

    # Sort datasets by ODOR_ORDER
    ordered_datasets = [d for d in ODOR_ORDER if d in pivot.index]
    extra_datasets = sorted(d for d in pivot.index if d not in ODOR_ORDER)
    all_datasets = ordered_datasets + extra_datasets
    if all_datasets:
        pivot = pivot.loc[all_datasets]

    # Reset index
    pivot = pivot.reset_index()

    # Save to CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "reaction_rates_summary_from_wide.csv"

    pivot.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[SUCCESS] Exported {csv_path}")
    print(f"          Shape: {pivot.shape[0]} datasets x {pivot.shape[1]-1} odors")
    print(f"\n[INFO] Preview:")
    print(pivot.to_string(max_rows=10, max_cols=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export reaction rate summary from a wide CSV.")
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument("--wide-csv", default=None, help="Path to the wide CSV file.")
    parser.add_argument("--out-dir", default=None, help="Output directory for CSV summaries.")
    args = parser.parse_args()

    config_data = load_raw_config(args.config)
    tools_cfg = config_data.get("tools", {}).get("export_reaction_rates_simple", {})
    if not isinstance(tools_cfg, dict):
        tools_cfg = {}
    wide_csv_value = args.wide_csv or tools_cfg.get("wide_csv") or config_data.get("analysis", {}).get("combined", {}).get("wide", {}).get("output_csv", "")
    out_dir_value = args.out_dir or tools_cfg.get("out_dir") or config_data.get("reaction_prediction", {}).get("matrix", {}).get("out_dir", "")
    wide_csv = Path(wide_csv_value).expanduser() if wide_csv_value else None
    out_dir = Path(out_dir_value).expanduser() if out_dir_value else None

    print("=" * 70)
    print("REACTION RATES SUMMARY EXPORT (from Wide CSV)")
    print("=" * 70)

    if not wide_csv:
        print("[ERROR] Wide CSV not configured. Provide --wide-csv or set tools.export_reaction_rates_simple.wide_csv.")
        sys.exit(1)
    if not out_dir:
        print("[ERROR] Output directory not configured. Provide --out-dir or set tools.export_reaction_rates_simple.out_dir.")
        sys.exit(1)
    if not wide_csv.exists():
        print(f"[ERROR] Wide CSV not found: {wide_csv}")
        sys.exit(1)

    export_reaction_rates_from_wide_csv(
        wide_csv=wide_csv,
        out_dir=out_dir,
        latency_sec=0.0,
        fps_default=40.0,
        before_sec=30.0,
        during_sec=30.0,
        after_window_sec=30.0,
        threshold_std_mult=4.0,
        min_samples_over=20,
    )

    print("=" * 70)
    print("DONE!")
    print("=" * 70)
