"""Shared analysis helpers: wide-table loading and trace baseline correction.

The analysis scripts each independently re-parse the large wide envelope tables
(``all_envelope_rows_wide*.csv``, hundreds of MB) on every run. Reading those
through :func:`read_wide_table` makes them transparently use the Parquet sibling
when present (3-4x faster, columnar) while still falling back to CSV, with
optional column projection so a script can read only the columns it needs.

``baseline_correct`` is the single source of truth for the pre-odor baseline
subtraction that was previously copy-pasted across ``dataset_means`` and
``dataset_means_specific_flies``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.frozen_folders import FROZEN_COLUMN, drop_frozen
from ..utils.tables import read_table, resolve_existing


def _has_column(parquet_path: Path, column: str) -> bool:
    """Whether a Parquet file carries *column*, read from its footer only.

    Cheap (no row groups touched) and, on any read problem, answers False so the
    caller falls back to a projection that cannot raise.
    """
    try:
        import pyarrow.parquet as pq

        return column in pq.ParquetFile(parquet_path).schema_arrow.names
    except Exception:
        return False


def read_wide_table(
    path: str | Path,
    columns: Optional[Sequence[str]] = None,
    *,
    include_frozen: bool = False,
) -> pd.DataFrame:
    """Load a wide analysis table, preferring Parquet over CSV.

    Resolves *path* to an existing ``.parquet`` (preferred) or ``.csv`` sibling.
    When *columns* is given and the resolved file is Parquet, only those columns
    are read from disk (columnar projection); for CSV they are selected after
    read. Raises ``FileNotFoundError`` if neither format exists.

    Rows from frozen experiment folders are dropped BY DEFAULT — that is the
    "in the CSV, not in the plots" cut, applied here so a figure script gets it
    without opting in. Pass ``include_frozen=True`` for the complete record.
    """
    resolved = resolve_existing(path)
    if resolved is None:
        raise FileNotFoundError(
            f"[wide] No .parquet or .csv table found for '{path}'."
        )
    if columns is not None and resolved.suffix.lower() == ".parquet":
        # Read the marker alongside the projection, or the filter has nothing to
        # read and a narrow projection would silently restore the retired flies.
        # Only when the file HAS it: pyarrow raises ArrowInvalid on a projection
        # naming an absent column, and legacy tables have no `frozen`.
        want = list(columns)
        projection = want
        if not include_frozen and FROZEN_COLUMN not in want and _has_column(
            resolved, FROZEN_COLUMN
        ):
            projection = want + [FROZEN_COLUMN]
        df = read_table(resolved, columns=projection)
        df = drop_frozen(df, include_frozen=include_frozen)
        # Hand back exactly what was asked for; the marker was our business.
        return df[[c for c in want if c in df.columns]]
    df = read_table(resolved)
    df = drop_frozen(df, include_frozen=include_frozen)
    if columns is not None:
        present = [c for c in columns if c in df.columns]
        df = df[present]
    return df


def baseline_correct(
    trace: np.ndarray, baseline_frames: Optional[int]
) -> np.ndarray:
    """Subtract the pre-odor mean (first ``baseline_frames`` samples) from a 1-D trace.

    Returns *trace* unchanged when ``baseline_frames`` is falsy/non-positive or
    the baseline window holds no finite values.
    """
    if not baseline_frames or baseline_frames <= 0:
        return trace
    window = trace[: min(len(trace), int(baseline_frames))]
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return trace
    return trace - float(finite.mean())
