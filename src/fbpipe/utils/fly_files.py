"""Helpers for locating per-fly distance CSV exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

_FLY_SLOT_REGEX = re.compile(r"(fly(\d+)_distances)\.csv$", re.IGNORECASE)


def fly_slot_from_name(name: str) -> Optional[Tuple[str, int]]:
    """Return the fly slot token and index from a CSV file name."""

    lowered = name.lower()
    match = _FLY_SLOT_REGEX.search(lowered)
    if not match:
        if "_distances" in lowered:
            print(
                f"[FLYFILES] CSV '{name}' has '_distances' but is missing the fly# token."
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
    """Yield per-fly distance CSVs under ``base``.

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

    iterator: Iterable[Path]
    if recursive:
        iterator = base.rglob("*.csv")
    else:
        iterator = base.glob("*.csv")

    seen: Dict[Path, Tuple[str, int]] = {}
    for path in iterator:
        if not path.is_file():
            continue
        slot = fly_slot_from_name(path.name)
        if not slot:
            continue
        resolved = path.resolve()
        if resolved in seen:
            print(f"[FLYFILES] Already yielded {resolved}; skipping duplicate reference.")
            continue
        token, slot_idx = slot
        seen[resolved] = slot
        try:
            rel_path = resolved.relative_to(base.parent)
        except ValueError:
            rel_path = resolved
        print(
            f"[FLYFILES] Found '{token}' (index {slot_idx}) at {rel_path}."
        )
        yield path, token, slot_idx


