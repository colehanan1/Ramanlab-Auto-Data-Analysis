from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

import pandas as pd

PROBOSCIS_CLASS = 8

PROBOSCIS_X_COL = f"x_class{PROBOSCIS_CLASS}"
PROBOSCIS_Y_COL = f"y_class{PROBOSCIS_CLASS}"
PROBOSCIS_TRACK_COL = f"track_id_class{PROBOSCIS_CLASS}"
PROBOSCIS_CORNERS_COL = f"corners_class{PROBOSCIS_CLASS}"
PROBOSCIS_DISTANCE_COL = f"distance_2_{PROBOSCIS_CLASS}"
PROBOSCIS_DISTANCE_PCT_COL = f"distance_percentage_2_{PROBOSCIS_CLASS}"
PROBOSCIS_MIN_DISTANCE_COL = f"min_distance_2_{PROBOSCIS_CLASS}"
PROBOSCIS_MAX_DISTANCE_COL = f"max_distance_2_{PROBOSCIS_CLASS}"

_PROBOSCIS_X_ALIASES = (
    PROBOSCIS_X_COL,
    f"x_class_{PROBOSCIS_CLASS}",
    f"class{PROBOSCIS_CLASS}_x",
    "x_proboscis",
    "proboscis_x",
    "proboscisx",
    "xclassproboscis",
    "xclassprob",
    # legacy
    "x_class6",
    "x_class_6",
    "class6_x",
)

_PROBOSCIS_Y_ALIASES = (
    PROBOSCIS_Y_COL,
    f"y_class_{PROBOSCIS_CLASS}",
    f"class{PROBOSCIS_CLASS}_y",
    "y_proboscis",
    "proboscis_y",
    "proboscisy",
    "yclassproboscis",
    "yclassprob",
    # legacy
    "y_class6",
    "y_class_6",
    "class6_y",
)

_PROBOSCIS_DISTANCE_ALIASES = (
    PROBOSCIS_DISTANCE_COL,
    f"distance_class2_class{PROBOSCIS_CLASS}",
    f"distance_class_2_{PROBOSCIS_CLASS}",
    "distance_proboscis",
    "proboscis_distance",
    "distance_prob",
    "distance",
    # legacy
    "distance_2_6",
    "distance_class2_class6",
    "distance_class_2_6",
)

_PROBOSCIS_DISTANCE_PCT_ALIASES = (
    PROBOSCIS_DISTANCE_PCT_COL,
    f"distance_percentage_class2_class{PROBOSCIS_CLASS}",
    f"distance_pct_class2_class{PROBOSCIS_CLASS}",
    "distance_percentage",
    "distance_percent",
    "distance_pct",
    "distance_proboscis_pct",
    "proboscis_distance_pct",
    # legacy
    "distance_percentage_2_6",
    "distance_pct_2_6",
)

_PROBOSCIS_MIN_DISTANCE_ALIASES = (
    PROBOSCIS_MIN_DISTANCE_COL,
    "min_distance_proboscis",
    # legacy
    "min_distance_2_6",
)

_PROBOSCIS_MAX_DISTANCE_ALIASES = (
    PROBOSCIS_MAX_DISTANCE_COL,
    "max_distance_proboscis",
    # legacy
    "max_distance_2_6",
)


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _resolve_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lookup = {_normalise(col): col for col in df.columns}
    for alias in aliases:
        key = _normalise(alias)
        if key in lookup:
            return lookup[key]
    return None


def find_proboscis_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    return _resolve_column(df, _PROBOSCIS_X_ALIASES), _resolve_column(df, _PROBOSCIS_Y_ALIASES)


def find_proboscis_distance_column(df: pd.DataFrame) -> Optional[str]:
    return _resolve_column(df, _PROBOSCIS_DISTANCE_ALIASES)


def find_proboscis_distance_percentage_column(df: pd.DataFrame) -> Optional[str]:
    return _resolve_column(df, _PROBOSCIS_DISTANCE_PCT_ALIASES)


def legacy_distance_columns(df: pd.DataFrame) -> Tuple[str, ...]:
    cols: list[str] = []
    for alias in ("distance_2_6", "distance_percentage_2_6", "min_distance_2_6", "max_distance_2_6"):
        col = _resolve_column(df, [alias])
        if col:
            cols.append(col)
    return tuple(cols)


def find_proboscis_min_distance_column(df: pd.DataFrame) -> Optional[str]:
    return _resolve_column(df, _PROBOSCIS_MIN_DISTANCE_ALIASES)


def find_proboscis_max_distance_column(df: pd.DataFrame) -> Optional[str]:
    return _resolve_column(df, _PROBOSCIS_MAX_DISTANCE_ALIASES)


__all__ = [
    "PROBOSCIS_CLASS",
    "PROBOSCIS_X_COL",
    "PROBOSCIS_Y_COL",
    "PROBOSCIS_TRACK_COL",
    "PROBOSCIS_CORNERS_COL",
    "PROBOSCIS_DISTANCE_COL",
    "PROBOSCIS_DISTANCE_PCT_COL",
    "PROBOSCIS_MIN_DISTANCE_COL",
    "PROBOSCIS_MAX_DISTANCE_COL",
    "find_proboscis_xy_columns",
    "find_proboscis_distance_column",
    "find_proboscis_distance_percentage_column",
    "find_proboscis_min_distance_column",
    "find_proboscis_max_distance_column",
    "legacy_distance_columns",
]
