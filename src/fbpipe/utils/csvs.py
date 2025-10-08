"""Helpers for discovering YOLO distance CSV exports."""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_FLY_SLOT_PATTERN = re.compile(r"_fly(\d+)_distances\.csv$", re.IGNORECASE)
_MERGED_SUFFIX = "_distances_merged.csv"
_SPECIFIC_BASE_PATTERN = re.compile(r"^(?:updated_)?(?P<base>.+?)_fly\d+_distances\.csv$", re.IGNORECASE)
_MERGED_BASE_PATTERN = re.compile(r"^(?:updated_)?(?P<base>.+?)_distances_merged\.csv$", re.IGNORECASE)

__all__ = [
    "extract_fly_slot",
    "group_distance_csvs",
    "gather_distance_csvs",
    "distance_base_name",
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


def distance_base_name(path: Path | str) -> str:
    """Return a lowercase identifier representing the distance CSV family."""

    name = Path(path).name
    match = _SPECIFIC_BASE_PATTERN.match(name)
    if match:
        return match.group("base").lower()
    match = _MERGED_BASE_PATTERN.match(name)
    if match:
        return match.group("base").lower()
    return Path(path).stem.lower()

def _sorted_by_slot(paths: Iterable[Path]) -> List[Path]:
    return sorted(
        paths,
        key=lambda p: (
            extract_fly_slot(p) if extract_fly_slot(p) is not None else 0,
            p.name.lower(),
        ),
    )

def gather_distance_csvs(base_dir: Path) -> List[Path]:
    """Return only per-fly distance CSV exports under ``base_dir``.

    The discovery logic ignores historical ``*_distances_merged.csv`` artefacts
    entirely so downstream steps never fall back to the deprecated format.
    """

    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    per_fly = [p for p in base_dir.rglob("*_fly*_distances.csv") if p.is_file()]

    by_parent: Dict[Path, Dict[str, List[Path]]] = {}

    for path in per_fly:
        parent = _normalise_parent(path.parent)
        base = distance_base_name(path)
        by_parent.setdefault(parent, {}).setdefault(base, []).append(path)

    debug_enabled = bool(os.environ.get("FBPIPE_DEBUG_CSV"))
    if debug_enabled:
        print("[CSV] gather_distance_csvs debug dump:")

    results: List[Path] = []
    for parent in sorted(by_parent.keys(), key=lambda p: str(p)):
        base_map = by_parent[parent]
        for base in sorted(base_map.keys()):
            chosen = _sorted_by_slot(base_map[base])
            results.extend(chosen)
            if debug_enabled:
                print(
                    f"[CSV] {parent} :: base={base} using specific="
                    f"{[p.name for p in chosen]}"
                )

    return results

