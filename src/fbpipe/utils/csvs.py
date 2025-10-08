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
    """Return distance CSVs under ``base_dir`` (per-fly exports only).

    If a trial only has merged exports available a ``FileNotFoundError`` is
    raised so callers do not silently fall back to merged data.
    """

    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    fly_specific = [p for p in base_dir.rglob("*_fly*_distances.csv") if p.is_file()]
    merged = [p for p in base_dir.rglob(f"*{_MERGED_SUFFIX}") if p.is_file()]

    by_parent: Dict[Path, Dict[str, Dict[str, List[Path]]]] = {}

    def _base_key(path: Path) -> Tuple[Path, str]:
        parent = _normalise_parent(path.parent)
        base = distance_base_name(path)
        return parent, base

    for path in fly_specific:
        parent, base = _base_key(path)
        parent_map = by_parent.setdefault(parent, {})
        bucket = parent_map.setdefault(base, {"specific": [], "merged": []})
        bucket["specific"].append(path)

    for path in merged:
        parent, base = _base_key(path)
        parent_map = by_parent.setdefault(parent, {})
        bucket = parent_map.setdefault(base, {"specific": [], "merged": []})
        bucket["merged"].append(path)

    debug_enabled = bool(os.environ.get("FBPIPE_DEBUG_CSV"))
    debug_rows: List[Tuple[str, str, List[str], List[str]]] = []

    results: List[Path] = []
    merged_only: List[Tuple[Path, str, List[Path]]] = []

    for parent in sorted(by_parent.keys(), key=lambda p: str(p)):
        base_map = by_parent[parent]
        for base in sorted(base_map.keys()):
            bucket = base_map[base]
            if bucket["specific"]:
                chosen = _sorted_by_slot(bucket["specific"])
                results.extend(chosen)
            elif bucket["merged"]:
                merged_only.append((parent, base, bucket["merged"]))

            if debug_enabled:
                debug_rows.append(
                    (
                        str(parent),
                        base,
                        [p.name for p in bucket["specific"]],
                        [p.name for p in bucket["merged"]],
                    )
                )

    if debug_enabled and debug_rows:
        print("[CSV] gather_distance_csvs debug dump:")
        for parent, base, specific, merged_list in debug_rows:
            if specific:
                print(
                    f"[CSV] {parent} :: base={base} "
                    f"using specific={specific} (merged candidates={merged_list})"
                )
            else:
                print(
                    f"[CSV] {parent} :: base={base} ignoring merged-only candidates={merged_list}"
                )

    if merged_only:
        details = []
        for parent, base, paths in merged_only:
            merged_names = ", ".join(sorted(p.name for p in paths))
            details.append(f"{parent} :: {base} → {merged_names}")
        raise FileNotFoundError(
            "Per-fly distance CSV exports are required but missing. "
            "Re-run YOLO inference to generate *_fly{N}_distances.csv files. "
            "Merged-only candidates encountered:\n" + "\n".join(details)
        )

    return results

