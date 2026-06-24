from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple

import numpy as np
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
    find_eye_xy_columns,
    find_proboscis_distance_column,
    find_proboscis_xy_columns,
)
from .tables import read_schema_columns, read_table

# Per-fly distance files may be on disk as either .csv (legacy) or .parquet.
_TRIAL_SLOT_REGEX = re.compile(
    r"(?P<prefix>.+)_fly(?P<slot>\d+)_distances\.(?:csv|parquet)$", re.IGNORECASE
)
_ANGLE_EYE_PROB_ANCHOR_COL = f"angle_deg_c{EYE_CLASS}_c{PROBOSCIS_CLASS}_vs_anchor"

# Every column that describes a single proboscis detection. When a detection is
# rejected (geometry or velocity gate) all of these are blanked together so the
# point reads as a clean "no detection" downstream.
_PROBOSCIS_NAN_COLS: Tuple[str, ...] = (
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
)


def _blank_proboscis_rows(df: pd.DataFrame, mask, extra_cols: Tuple[str, ...] = ()) -> pd.DataFrame:
    """Return a copy of ``df`` with all proboscis columns NaN where ``mask`` is True."""

    clean_df = df.copy()
    for col in (*extra_cols, *_PROBOSCIS_NAN_COLS):
        if col in clean_df.columns:
            clean_df.loc[mask, col] = pd.NA
    return clean_df


def anisotropic_semi_axes(dx, dy, max_px: float, up_divisor: float):
    """Per-point semi-axes of the eye->proboscis gate.

    Left/right/down are unrestricted (semi-axis ``max_px``); only the upward
    direction is tightened to ``max_px / up_divisor`` (dy<0). ``up_divisor`` of 1
    makes a plain circle. Accepts scalars or arrays.
    """

    up = max_px / up_divisor if up_divisor else max_px
    a = float(max_px)  # no left/right restriction
    b = np.where(np.asarray(dy, dtype=float) < 0, up, max_px)  # up tight / down generous
    return a, b


def anisotropic_boundary_offsets(max_px: float, up_divisor: float, n: int = 72):
    """Trace the gate boundary as ``n`` (dx, dy) offsets from the eye, for drawing."""

    pts = []
    for k in range(n):
        th = 2.0 * math.pi * k / n
        cx, cy = math.cos(th), math.sin(th)
        a, b = anisotropic_semi_axes(cx, cy, max_px, up_divisor)
        a, b = float(a), float(b)
        r = 1.0 / math.sqrt((cx / a) ** 2 + (cy / b) ** 2)
        pts.append((r * cx, r * cy))
    return pts


def sanitize_eye_prob_geometry_dataframe(
    df: pd.DataFrame,
    max_distance_px: float,
    up_divisor: float = 4.0,
) -> Tuple[pd.DataFrame, int]:
    """Geometry gate: blank any proboscis outside the anisotropic region around its eye.

    Applies to every fly count. When eye+proboscis coordinates are available the
    gate is the quarter-ellipse blob from :func:`anisotropic_semi_axes`; otherwise
    it falls back to a plain radial circle using the precomputed distance column.
    """

    if max_distance_px <= 0:
        return df, 0

    dist_col = find_proboscis_distance_column(df)
    ex_col, ey_col = find_eye_xy_columns(df)
    px_col, py_col = find_proboscis_xy_columns(df)

    if ex_col and ey_col and px_col and py_col:
        ex = pd.to_numeric(df[ex_col], errors="coerce").to_numpy(dtype=float)
        ey = pd.to_numeric(df[ey_col], errors="coerce").to_numpy(dtype=float)
        px = pd.to_numeric(df[px_col], errors="coerce").to_numpy(dtype=float)
        py = pd.to_numeric(df[py_col], errors="coerce").to_numpy(dtype=float)
        dx = px - ex
        dy = py - ey
        a, b = anisotropic_semi_axes(dx, dy, max_distance_px, up_divisor)
        with np.errstate(invalid="ignore"):
            norm = (dx / a) ** 2 + (dy / b) ** 2
        invalid_mask = pd.Series(np.isfinite(norm) & (norm > 1.0), index=df.index)
    elif dist_col is not None:
        # Fallback: radial circle (no coordinates to build the anisotropic gate).
        invalid_mask = pd.to_numeric(df[dist_col], errors="coerce").gt(max_distance_px)
    else:
        return df, 0

    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return df, 0

    extra_cols = (dist_col,) if dist_col else ()
    return _blank_proboscis_rows(df, invalid_mask, extra_cols=extra_cols), invalid_count


def sanitize_proboscis_velocity_dataframe(
    df: pd.DataFrame,
    max_jump_px: float,
) -> Tuple[pd.DataFrame, int]:
    """Velocity gate: blank proboscis points that jump > ``max_jump_px`` between detections.

    Walks the rows in order, comparing each present proboscis position against
    the last *accepted* one. A point whose jump exceeds the limit is rejected and
    does not become the new reference, so a single hallucinated/switched point
    cannot drag the reference along with it.
    """

    if max_jump_px <= 0:
        return df, 0

    x_col, y_col = find_proboscis_xy_columns(df)
    if x_col is None or y_col is None:
        return df, 0

    xs = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    reject = np.zeros(len(df), dtype=bool)
    last_x = last_y = None
    for i in range(len(df)):
        x, y = xs[i], ys[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue  # already missing
        if last_x is not None and math.hypot(x - last_x, y - last_y) > max_jump_px:
            reject[i] = True
            continue  # suspect point: drop it, keep the previous reference
        last_x, last_y = x, y

    reject_count = int(reject.sum())
    if reject_count == 0:
        return df, 0

    return _blank_proboscis_rows(df, reject), reject_count


def sanitize_proboscis_dataframe(
    df: pd.DataFrame,
    *,
    max_distance_px: float,
    max_jump_px: float,
    up_divisor: float = 4.0,
) -> Tuple[pd.DataFrame, int, int]:
    """Apply geometry then velocity gates. Returns (df, geometry_count, velocity_count)."""

    df, geo_count = sanitize_eye_prob_geometry_dataframe(df, max_distance_px, up_divisor)
    df, vel_count = sanitize_proboscis_velocity_dataframe(df, max_jump_px)
    return df, geo_count, vel_count


def is_three_fly_trial_csv(csv_path: Path) -> bool:
    match = _TRIAL_SLOT_REGEX.match(csv_path.name)
    if match is None:
        return False

    prefix = match.group("prefix")
    slots: set[int] = set()
    # Discover sibling fly slots in either on-disk format (.parquet preferred,
    # .csv legacy). Slots are collected into a set, so a stem present as both
    # formats is naturally de-duplicated.
    siblings = list(csv_path.parent.glob(f"{prefix}_fly*_distances.parquet"))
    siblings += list(csv_path.parent.glob(f"{prefix}_fly*_distances.csv"))
    for sibling in siblings:
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
        header = pd.DataFrame(columns=read_schema_columns(csv_path))
    except Exception:
        return False

    dist_col = find_proboscis_distance_column(header)
    if dist_col is None:
        return False

    try:
        distances = read_table(csv_path, columns=[dist_col])[dist_col]
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
