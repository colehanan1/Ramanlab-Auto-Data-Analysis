"""
Calculate frame-to-frame acceleration from combined distance% × angle multiplier.

Modification #4: This step detects potential model errors or dropped frames by identifying
physically implausible rapid changes in proboscis position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Settings, get_main_directories
from ..utils.columns import PROBOSCIS_DISTANCE_PCT_COL
from ..utils.fly_files import iter_fly_distance_csvs

# Acceleration threshold for flagging suspicious frames (% per frame)
ACCELERATION_THRESHOLD = 20.0

# Required column names
DISTANCE_PCT_COL = PROBOSCIS_DISTANCE_PCT_COL
ANGLE_MULT_COL = "angle_multiplier"
FRAME_COL = "frame"


def calculate_acceleration_for_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Calculate acceleration for a single CSV file.

    Args:
        csv_path: Path to CSV file in RMS_calculations

    Returns:
        DataFrame with new columns added, or None if missing required columns
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ACCEL] Error reading {csv_path.name}: {e}")
        return None

    # Check for required columns (with fallback aliases)
    dist_col = None
    for col in [
        DISTANCE_PCT_COL,
        "distance_percentage",
        "distance_pct",
        "distance_class1_class2_pct",
        "distance_percentage_2_8",
        "distance_pct_2_8",
    ]:
        if col in df.columns:
            dist_col = col
            break

    if dist_col is None:
        print(f"[ACCEL] Missing distance percentage column in {csv_path.name}")
        return None

    if ANGLE_MULT_COL not in df.columns:
        print(f"[ACCEL] Missing {ANGLE_MULT_COL} in {csv_path.name} (run compose_videos_rms first)")
        return None

    # Extract data
    dist_pct = pd.to_numeric(df[dist_col], errors="coerce").to_numpy()
    angle_mult = pd.to_numeric(df[ANGLE_MULT_COL], errors="coerce").to_numpy()

    # Calculate combined metric (distance% × angle multiplier)
    combined = dist_pct * angle_mult
    df["combined_distance_x_angle"] = combined

    # Calculate frame-to-frame acceleration (delta)
    acceleration = np.full(len(combined), np.nan)
    acceleration[1:] = np.diff(combined)  # Frame i - Frame i-1
    df["acceleration_pct_per_frame"] = acceleration

    # Flag high-acceleration frames
    df["acceleration_flag"] = np.abs(acceleration) > ACCELERATION_THRESHOLD

    return df


def main(cfg: Settings) -> None:
    """
    Calculate acceleration for all RMS calculation CSVs.

    Process:
    1. Find all CSVs in RMS_calculations directories
    2. Calculate combined distance × angle multiplier
    3. Calculate frame-to-frame acceleration
    4. Flag suspicious frames (acceleration > threshold)
    5. Save updated CSVs with new columns

    Modification #4: Adds acceleration analysis to detect model errors.
    """
    roots = get_main_directories(cfg)

    total_processed = 0
    total_flagged = 0

    print(f"[ACCEL] Starting acceleration calculation in {len(roots)} directories")

    for root in roots:
        print(f"[ACCEL] Processing root directory: {root}")
        for fly_dir in [p for p in root.iterdir() if p.is_dir()]:
            rms_dir = fly_dir / "RMS_calculations"
            if not rms_dir.is_dir():
                continue

            fly_processed = 0
            fly_flagged = 0

            for csv_path, token, _ in iter_fly_distance_csvs(rms_dir, recursive=False):
                # Skip if already processed (has acceleration column)
                try:
                    temp_df = pd.read_csv(csv_path, nrows=1)
                    if "acceleration_pct_per_frame" in temp_df.columns:
                        continue
                except Exception:
                    pass

                # Calculate acceleration
                df = calculate_acceleration_for_csv(csv_path)
                if df is None:
                    continue

                # Count flagged frames
                n_flagged = int(df["acceleration_flag"].sum())
                fly_flagged += n_flagged

                # Save updated CSV
                df.to_csv(csv_path, index=False)
                fly_processed += 1

                if n_flagged > 0:
                    print(f"[ACCEL] {csv_path.name}: {n_flagged} high-acceleration frames flagged")

            if fly_processed > 0:
                total_processed += fly_processed
                total_flagged += fly_flagged
                print(f"[ACCEL] {fly_dir.name}: {fly_processed} files processed, {fly_flagged} suspicious frames")

    print(f"[ACCEL] Complete: {total_processed} CSVs processed, {total_flagged} total flagged frames")


if __name__ == "__main__":
    from ..config import load_settings
    main(load_settings())
