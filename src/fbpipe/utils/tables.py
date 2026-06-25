"""Single chokepoint for tabular I/O used throughout the pipeline.

All pipeline-produced tabular data is stored as Parquet (via pyarrow).  CSV
files that already exist on disk are still readable for backward compatibility.
Every write goes to Parquet, normalising the file extension automatically.

Public API
----------
read_table(path, **kwargs) -> pd.DataFrame
    Read a table from disk, dispatching on suffix.  If the literal path does
    not exist, resolve_existing is called to find a sibling with a supported
    suffix (preferring .parquet over .csv).

write_table(df, path, **kwargs) -> pathlib.Path
    Always write Parquet regardless of the suffix given in ``path``.  Normalises
    the path with table_path(), creates parent directories, and returns the
    path actually written.  The DataFrame index is NOT persisted so behaviour
    matches the existing ``to_csv(index=False)`` call sites.

table_path(path) -> pathlib.Path
    Return ``path`` with the suffix changed to ``.parquet``.

resolve_existing(path) -> pathlib.Path | None
    Given any logical path, find the existing file preferring ``.parquet``
    then ``.csv``, returning None if neither exists.

read_schema_columns(path) -> list[str]
    Cheaply read column names without loading row data.  Uses
    pyarrow.parquet.read_schema for Parquet; pd.read_csv(nrows=0) for CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

_SUPPORTED = (".parquet", ".csv")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def table_path(path: str | Path) -> Path:
    """Return *path* with suffix replaced by ``.parquet``."""
    return Path(path).with_suffix(".parquet")


def resolve_existing(path: str | Path) -> Optional[Path]:
    """Return the existing file for *path*, preferring ``.parquet`` over ``.csv``.

    The stem is extracted from *path* (ignoring whatever suffix it carries) and
    sibling candidates are checked in preference order.  Returns ``None`` if no
    candidate exists.
    """
    p = Path(path)
    stem = p.parent / p.stem
    for suffix in _SUPPORTED:
        candidate = stem.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a table from *path*, dispatching by file extension.

    Extension dispatch
    ------------------
    ``.parquet``
        ``pd.read_parquet(path, engine="pyarrow", **kwargs)``
        Supports ``columns=[...]`` for selective column reads.
    ``.csv``
        ``pd.read_csv(path, **kwargs)``

    Path resolution
    ---------------
    If the literal *path* does not exist on disk, :func:`resolve_existing` is
    called to locate a sibling ``.parquet`` or ``.csv`` file.  ``.parquet`` is
    preferred.

    Raises
    ------
    FileNotFoundError
        When neither a ``.parquet`` nor a ``.csv`` sibling can be found.
    """
    p = Path(path)

    if not p.is_file():
        resolved = resolve_existing(p)
        if resolved is None:
            raise FileNotFoundError(
                f"[TABLES] No file found for '{p}' "
                f"(tried {p.with_suffix('.parquet')} and {p.with_suffix('.csv')})."
            )
        print(f"[TABLES] '{p.name}' not found; resolved to '{resolved.name}'.")
        p = resolved

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p, engine="pyarrow", **kwargs)
    elif suffix == ".csv":
        return pd.read_csv(p, **kwargs)
    else:
        # Unexpected suffix — attempt parquet then CSV resolution
        resolved = resolve_existing(p)
        if resolved is not None:
            print(f"[TABLES] Unknown suffix '{suffix}'; resolved to '{resolved.name}'.")
            return read_table(resolved, **kwargs)
        raise FileNotFoundError(
            f"[TABLES] Unsupported file type '{suffix}' and no sibling found for '{p}'."
        )


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    replace_legacy_csv: bool = True,
    **kwargs,
) -> Path:
    """Write *df* to Parquet at *path* (suffix normalised to ``.parquet``).

    Always writes Parquet via pyarrow regardless of the suffix of *path*.  The
    DataFrame index is NOT persisted (``index=False``) so round-tripping matches
    the existing ``df.to_csv(index=False)`` call sites and no
    ``__index_level_0__`` column appears on read-back.

    Single source of truth: when ``replace_legacy_csv`` is True (the default) a
    same-stem ``.csv`` sibling left over from the legacy CSV pipeline is removed
    after the Parquet is written.  This prevents a stale ``.csv`` from shadowing
    the freshly-written ``.parquet`` for any consumer that resolves a literal
    ``.csv`` path (``read_table`` dispatches on suffix and only falls back to the
    Parquet when the ``.csv`` is absent).  Set ``replace_legacy_csv=False`` to
    keep the CSV (e.g. the migration tool's keep-both mode).

    Parent directories are created automatically.  Returns the Path written.
    """
    out = table_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False, **kwargs)
    if replace_legacy_csv:
        legacy_csv = out.with_suffix(".csv")
        if legacy_csv != out and legacy_csv.exists():
            try:
                legacy_csv.unlink()
            except OSError:
                pass
    return out


# ---------------------------------------------------------------------------
# Cheap schema read
# ---------------------------------------------------------------------------


def read_schema_columns(path: str | Path) -> list[str]:
    """Return column names for *path* WITHOUT loading row data.

    For Parquet files this uses ``pyarrow.parquet.read_schema`` which reads
    only the file footer (O(1) vs row count).  For CSV files this uses
    ``pd.read_csv(path, nrows=0)`` which reads only the header line.

    Path resolution mirrors :func:`read_table`: if the literal *path* does not
    exist on disk, :func:`resolve_existing` is used to locate a sibling
    ``.parquet`` or ``.csv`` file (preferring ``.parquet``).

    Raises
    ------
    FileNotFoundError
        If neither a ``.parquet`` nor a ``.csv`` sibling can be found.
    ValueError
        If the suffix is not supported.
    """
    p = Path(path)
    if not p.is_file():
        resolved = resolve_existing(p)
        if resolved is None:
            raise FileNotFoundError(
                f"[TABLES] No file found for '{p}' "
                f"(tried {p.with_suffix('.parquet')} and {p.with_suffix('.csv')})."
            )
        print(f"[TABLES] '{p.name}' not found; resolved to '{resolved.name}'.")
        p = resolved

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        schema = pq.read_schema(p)
        return schema.names
    elif suffix == ".csv":
        return pd.read_csv(p, nrows=0).columns.tolist()
    else:
        raise ValueError(
            f"[TABLES] read_schema_columns: unsupported suffix '{suffix}' for '{p}'."
        )


__all__ = [
    "read_table",
    "write_table",
    "table_path",
    "resolve_existing",
    "read_schema_columns",
]
