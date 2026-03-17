from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from .columns import (
    EYE_CLASS,
    PROBOSCIS_CLASS,
    PROBOSCIS_CORNERS_COL,
    PROBOSCIS_DISTANCE_COL,
    PROBOSCIS_DISTANCE_PCT_COL,
    PROBOSCIS_TRACK_COL,
    PROBOSCIS_X_COL,
    PROBOSCIS_Y_COL,
    find_proboscis_distance_column,
)

_TRIAL_SLOT_REGEX = re.compile(r"(?P<prefix>.+)_fly(?P<slot>\d+)_distances\.csv$", re.IGNORECASE)
_ANGLE_EYE_PROB_ANCHOR_COL = f"angle_deg_c{EYE_CLASS}_c{PROBOSCIS_CLASS}_vs_anchor"


def is_three_fly_trial_csv(csv_path: Path) -> bool:
    match = _TRIAL_SLOT_REGEX.match(csv_path.name)
    if match is None:
        return False

    prefix = match.group("prefix")
    slots: set[int] = set()
    pattern = f"{prefix}_fly*_distances.csv"
    for sibling in csv_path.parent.glob(pattern):
        sibling_match = _TRIAL_SLOT_REGEX.match(sibling.name)
        if sibling_match is None or sibling_match.group("prefix") != prefix:
            continue
        try:
            slots.add(int(sibling_match.group("slot")))
        except ValueError:
            continue
    return len(slots) == 3


def csv_requires_three_fly_distance_sanitization(csv_path: Path, max_distance_px: float) -> bool:
    if max_distance_px <= 0 or not is_three_fly_trial_csv(csv_path):
        return False

    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return False

    dist_col = find_proboscis_distance_column(header)
    if dist_col is None:
        return False

    try:
        distances = pd.read_csv(csv_path, usecols=[dist_col])[dist_col]
    except Exception:
        return False

    return bool(pd.to_numeric(distances, errors="coerce").gt(max_distance_px).any())


def sanitize_three_fly_distance_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    max_distance_px: float,
) -> Tuple[pd.DataFrame, int]:
    if max_distance_px <= 0 or not is_three_fly_trial_csv(csv_path):
        return df, 0

    dist_col = find_proboscis_distance_column(df)
    if dist_col is None:
        return df, 0

    distances = pd.to_numeric(df[dist_col], errors="coerce")
    invalid_mask = distances.gt(max_distance_px)
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return df, 0

    clean_df = df.copy()
    for col in (
        dist_col,
        PROBOSCIS_DISTANCE_COL,
        "distance_2_6",
        PROBOSCIS_DISTANCE_PCT_COL,
        "distance_percentage",
        "distance_percent",
        "distance_pct",
        "distance_percentage_2_6",
        "distance_pct_2_6",
        PROBOSCIS_TRACK_COL,
        PROBOSCIS_X_COL,
        PROBOSCIS_Y_COL,
        PROBOSCIS_CORNERS_COL,
        _ANGLE_EYE_PROB_ANCHOR_COL,
    ):
        if col in clean_df.columns:
            clean_df.loc[invalid_mask, col] = pd.NA

    return clean_df, invalid_count
