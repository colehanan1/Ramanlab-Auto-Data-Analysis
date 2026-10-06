"""Read a wide CSV through its Parquet sibling, as ``pd.read_csv`` would.

``build_wide_csv`` writes every wide table twice: the CSV, then a Parquet
sibling made from ``pd.read_csv`` of that CSV (envelope_combined.py, "Emit
Parquet siblings"). The sibling holds the same values in the same dtypes and
reads ~25x faster (under 1 s vs 20-25 s for the 1.3 GB testing table), but a
reader whose config names the ``.csv`` and whose output depends on read_csv's
exact frame kept parsing the CSV.

:func:`read_csv_via_parquet` keeps that contract. It returns the frame
``pd.read_csv(path)`` would, served from the sibling only when the sibling is
at least as new as the CSV -- the sidecar write only warns on failure, so a
stale sibling can sit beside tonight's CSV -- and falls back to the CSV when
the sibling is missing, stale or unreadable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tables import read_table

__all__ = ["read_csv_via_parquet"]


def _as_read_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Undo what a Parquet round trip changes relative to ``pd.read_csv``.

    read_csv marks a missing value in an object column as float NaN; pyarrow
    hands back None instead (string columns under pandas 2, bool columns with
    a gap under both). ``astype(str)``/``unique()`` tell the two apart ("nan"
    vs "None"), so put the NaN back. Numeric, bool and pandas-3 ``str``
    columns already come back exactly as read_csv builds them.
    """
    for name in df.columns[(df.dtypes == object).to_numpy()]:
        column = df[name]
        missing = column.isna()
        if missing.any():
            df[name] = column.where(~missing, np.nan)
    return df


def read_csv_via_parquet(path: str | Path) -> pd.DataFrame:
    """``pd.read_csv(path)``, from the ``.parquet`` sibling when it is current.

    A path that is not a ``.csv`` is read with :func:`fbpipe.utils.tables.read_table`.
    A missing CSV raises exactly as ``pd.read_csv`` does: the sibling never
    stands in for an absent CSV.
    """
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return read_table(path)
    sibling = path.with_suffix(".parquet")
    try:
        fresh = sibling.stat().st_mtime_ns >= path.stat().st_mtime_ns
    except OSError:  # no sibling (or no CSV: read_csv raises below)
        return pd.read_csv(path)
    if not fresh:
        print(f"[TABLES] {sibling.name} is older than {path.name}; reading the CSV.")
        return pd.read_csv(path)
    try:
        frame = pd.read_parquet(sibling, engine="pyarrow")
    except Exception as exc:  # a half-written or corrupt sidecar
        print(f"[TABLES] {sibling.name} is unreadable ({exc}); reading the CSV.")
        return pd.read_csv(path)
    return _as_read_csv(frame)
