"""Helpers for discovering YOLO distance CSV exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

_FLY_SLOT_PATTERN = re.compile(r"_fly(\d+)_distances\.csv$", re.IGNORECASE)
_MERGED_SUFFIX = "_distances_merged.csv"

__all__ = [
    "extract_fly_slot",
    "group_distance_csvs",
    "gather_distance_csvs",
]

def extract_fly_slot(path: Path | str) -> Optional[int]:
    """Return the 1-based fly slot encoded in a distance CSV name."""

    name = Path(path).name
    match = _FLY_SLOT_PATTERN.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None

def _normalise_parent(path: Path) -> Path:
    try:
        return path.resolve()
    except Exception:  # pragma: no cover - defensive fallback
        return path

def group_distance_csvs(paths: Sequence[Path]) -> Dict[Path, List[Path]]:
    grouped: Dict[Path, List[Path]] = {}
    for path in paths:
        parent = _normalise_parent(path.parent)
        grouped.setdefault(parent, []).append(path)
    return grouped

def _sorted_by_slot(paths: Iterable[Path]) -> List[Path]:
    return sorted(
        paths,
        key=lambda p: (
            extract_fly_slot(p) if extract_fly_slot(p) is not None else 0,
            p.name.lower(),
        ),
    )

def gather_distance_csvs(base_dir: Path) -> List[Path]:
    """Return distance CSVs under ``base_dir`` with per-fly preference."""

    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    fly_specific = [p for p in base_dir.rglob("*_fly*_distances.csv") if p.is_file()]
    merged = [p for p in base_dir.rglob(f"*{_MERGED_SUFFIX}") if p.is_file()]

    specific_by_parent = group_distance_csvs(fly_specific)
    merged_by_parent = group_distance_csvs(merged)

    results: List[Path] = []
    consumed_specific_parents: set[Path] = set()

    for parent, merges in sorted(merged_by_parent.items(), key=lambda item: str(item[0])):
        specific = specific_by_parent.get(parent, [])
        if specific:
            consumed_specific_parents.add(parent)
            results.extend(_sorted_by_slot(specific))
        else:
            results.extend(sorted(merges, key=lambda p: p.name.lower()))

    for parent, paths in sorted(specific_by_parent.items(), key=lambda item: str(item[0])):
        if parent in consumed_specific_parents:
            continue
        results.extend(_sorted_by_slot(paths))

    return results
