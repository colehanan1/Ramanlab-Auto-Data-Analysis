"""Column names, file naming and table I/O for the abdomen geometry pipeline.

Naming is deliberately disjoint from production: production enumerates
``*_fly{N}_distances.{csv,parquet}``, ``*merged*``, ``angle_distance_rms_envelope/``
and ``RMS_calculations/``; nothing here matches those patterns, so the two
pipelines can share a batch directory without ever reading each other's files.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils.tables import read_table, write_table

TABLE_FORMATS = ("csv", "parquet")
GEOMETRY_SUBDIR_DEFAULT = "geometry"

_FLY_FILE_RE = re.compile(r"_fly(?P<slot>\d+)_geometry\.(?P<ext>csv|parquet)$", re.IGNORECASE)

# ---- per-frame table -------------------------------------------------------
FRAME = "frame"
TIMESTAMP = "timestamp"
ODOR_ON = "odor_on"
EYE_X, EYE_Y, EYE_SCALE = "eye_x", "eye_y", "eye_scale_px"
PRO_X, PRO_Y, PRO_CONF = "proboscis_x", "proboscis_y", "proboscis_conf"
PRO_DIST, PRO_ANGLE_ANCHOR = "proboscis_distance_px", "proboscis_angle_vs_anchor_deg"
ABD_X, ABD_Y, ABD_CONF = "abdomen_x", "abdomen_y", "abdomen_conf"
ABD_LEN, ABD_WID, EYE_ABD_DIST = "abdomen_length_px", "abdomen_width_px", "eye_abdomen_distance_px"
TIP_X, TIP_Y, TIP_CONF, ABD_TIP_DIST = "tip_x", "tip_y", "tip_conf", "abdomen_tip_distance_px"
ABD_ANGLE, ABD_BEND = "abdomen_angle_deg", "abdomen_bend_deg"
N_EYES, STATUS, PRO_STATUS = "n_eyes_detected", "geometry_status", "proboscis_status"

PER_FRAME_COLUMNS: List[str] = [
    FRAME, TIMESTAMP, ODOR_ON,
    EYE_X, EYE_Y, EYE_SCALE,
    PRO_X, PRO_Y, PRO_CONF, PRO_DIST, PRO_ANGLE_ANCHOR,
    ABD_X, ABD_Y, ABD_CONF, ABD_LEN, ABD_WID, EYE_ABD_DIST,
    TIP_X, TIP_Y, TIP_CONF, ABD_TIP_DIST,
    ABD_ANGLE, ABD_BEND,
    N_EYES, STATUS, PRO_STATUS,
]
PROBOSCIS_VALUE_COLUMNS = (PRO_X, PRO_Y, PRO_CONF, PRO_DIST, PRO_ANGLE_ANCHOR)

# ---- added by normalize ----------------------------------------------------
PRO_PCT = "proboscis_pct"
ABD_ANGLE_PCT = "abdomen_angle_pct"
ABD_LEN_PCT = "abdomen_length_pct_rest"
ABD_WID_PCT = "abdomen_width_pct_rest"
NORMALIZED_COLUMNS: List[str] = [PRO_PCT, ABD_ANGLE_PCT, ABD_LEN_PCT, ABD_WID_PCT]

# ---- status values ---------------------------------------------------------
STATUS_OK = "ok"                              # abdomen and tip paired: angle measurable
STATUS_ABDOMEN_ONLY = "abdomen_only:no_tip"   # box via the tip-less fallback
STATUS_NO_ABDOMEN = "no_abdomen"              # nothing passed and no reason recorded
PRO_OK = "ok"
PRO_MISSING = "missing"
PRO_VELOCITY = "velocity_gate"


def validate_table_format(fmt: str) -> str:
    f = str(fmt or "").strip().lower()
    if f not in TABLE_FORMATS:
        raise ValueError(f"table_format must be one of {TABLE_FORMATS}, got {fmt!r}")
    return f


def geometry_table_path(trial_dir: Path, trial_name: str, slot: int, fmt: str) -> Path:
    return Path(trial_dir) / f"{trial_name}_fly{int(slot)}_geometry.{validate_table_format(fmt)}"


def meta_path(trial_dir: Path, trial_name: str) -> Path:
    return Path(trial_dir) / f"{trial_name}_geometry_meta.json"


def annotated_video_path(trial_dir: Path, trial_name: str) -> Path:
    return Path(trial_dir) / f"{trial_name}_geometry_annotated.mp4"


def stats_path(geometry_dir: Path, slot: int) -> Path:
    return Path(geometry_dir) / f"fly{int(slot)}_geometry_stats.json"


def fly_slot_from_geometry_name(name: str) -> Optional[int]:
    m = _FLY_FILE_RE.search(str(name))
    return int(m.group("slot")) if m else None


def write_table_fmt(df: pd.DataFrame, path: Path, fmt: str) -> Path:
    """Write ``df`` as CSV or Parquet next to ``path`` (suffix replaced). A Parquet
    write never removes a CSV sibling -- the verified demo CSVs must survive the
    switch to Parquet."""
    fmt = validate_table_format(fmt)
    out = Path(path).with_suffix("." + fmt)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out, index=False)
        return out
    return write_table(df, out, replace_legacy_csv=False)


def read_table_any(path: Path) -> pd.DataFrame:
    """Read a geometry table; a missing literal path resolves to the .parquet/.csv sibling."""
    return read_table(path)


def iter_geometry_tables(trial_dir: Path) -> List[Tuple[int, Path]]:
    """``(slot, path)`` per fly in ``trial_dir``; Parquet wins over CSV for the same slot."""
    best: Dict[int, Path] = {}
    for p in sorted(Path(trial_dir).glob("*_fly*_geometry.*")):
        slot = fly_slot_from_geometry_name(p.name)
        if slot is None or not p.is_file():
            continue
        cur = best.get(slot)
        if cur is None or (p.suffix.lower() == ".parquet" and cur.suffix.lower() != ".parquet"):
            best[slot] = p
    return sorted(best.items())
