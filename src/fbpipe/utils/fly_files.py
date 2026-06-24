"""Helpers for locating per-fly distance CSV/Parquet exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

# Matches both .csv and .parquet fly-distance file names.
_FLY_SLOT_REGEX = re.compile(
    r"(fly(\d+)_distances)\.(csv|parquet)$", re.IGNORECASE
)

# Legacy regex kept for backward compatibility — callers that rely on the old
# .csv-only pattern should migrate to _FLY_SLOT_REGEX.
_FLY_SLOT_REGEX_CSV_ONLY = re.compile(r"(fly(\d+)_distances)\.csv$", re.IGNORECASE)


def fly_slot_from_name(name: str) -> Optional[Tuple[str, int]]:
    """Return the fly slot token and index from a CSV or Parquet file name.

    Accepts both ``fly<N>_distances.csv`` and ``fly<N>_distances.parquet``
    names so that callers do not need to know which format is on disk.
    """

    lowered = name.lower()
    match = _FLY_SLOT_REGEX.search(lowered)
    if not match:
        if "_distances" in lowered:
            print(
                f"[FLYFILES] File '{name}' has '_distances' but is missing the fly# token."
            )
        return None
    token = match.group(1)
    try:
        slot_idx = int(match.group(2))
    except ValueError:
        print(f"[FLYFILES] Could not parse fly index from '{name}'.")
        return None
    return token, slot_idx


def iter_fly_distance_csvs(
    base: Path, *, recursive: bool = True
) -> Iterator[Tuple[Path, str, int]]:
    """Yield per-fly distance files (CSV or Parquet) under ``base``.

    When both ``fly<N>_distances.csv`` and ``fly<N>_distances.parquet`` exist
    in the same directory for the same stem, only the Parquet file is yielded.

    Parameters
    ----------
    base:
        Root directory to search.
    recursive:
        When ``True`` (default) search using ``rglob``; otherwise only inspect
        the immediate directory.
    """

    base = base.expanduser().resolve()
    print(
        f"[FLYFILES] Scanning {base} for flyN_distances CSVs (recursive={recursive})."
    )

    # Collect all candidate paths (both .csv and .parquet).
    patterns = ("*.csv", "*.parquet")
    candidates: list[Path] = []
    for pattern in patterns:
        if recursive:
            candidates.extend(base.rglob(pattern))
        else:
            candidates.extend(base.glob(pattern))

    # Deduplicate by (resolved_parent, stem), preferring .parquet over .csv.
    # best_by_stem: (parent, stem) -> (preferred_path, slot)
    best_by_stem: Dict[Tuple[Path, str], Tuple[Path, Tuple[str, int]]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        slot = fly_slot_from_name(path.name)
        if not slot:
            continue
        resolved = path.resolve()
        key = (resolved.parent, resolved.stem)
        if key not in best_by_stem:
            best_by_stem[key] = (resolved, slot)
        else:
            existing_path, _ = best_by_stem[key]
            # Prefer .parquet over .csv
            if resolved.suffix.lower() == ".parquet":
                best_by_stem[key] = (resolved, slot)

    # Deduplicate by resolved path (handles symlinks etc.) and yield.
    seen_resolved: Dict[Path, Tuple[str, int]] = {}
    for (parent, stem), (resolved, slot) in best_by_stem.items():
        if resolved in seen_resolved:
            print(f"[FLYFILES] Already yielded {resolved}; skipping duplicate reference.")
            continue
        token, slot_idx = slot
        seen_resolved[resolved] = slot
        try:
            rel_path = resolved.relative_to(base.parent)
        except ValueError:
            rel_path = resolved
        print(
            f"[FLYFILES] Found '{token}' (index {slot_idx}) at {rel_path}."
        )
        yield resolved, token, slot_idx


