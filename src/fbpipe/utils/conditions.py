"""
Experimental condition detection and classification.

Modification #6: Separate results by experimental condition (control vs optogenetic).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Condition patterns (order matters - check specific before general)
# Pattern format: (regex_pattern, condition_name)
CONDITION_PATTERNS = [
    (r"opto.*hex", "opto_hex"),
    (r"hex.*control", "hex_control"),
    (r"opto.*benz", "opto_benz_1"),
    (r"benz.*control", "benz_control"),
    (r"opto.*eb", "opto_Eb"),
    (r"opto.*Eb", "opto_Eb"),  # Case variation
    (r"eb.*control", "Eb_control"),
    (r"Eb.*control", "Eb_control"),  # Case variation
]

# Month names for extraction
MONTHS_FULL = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
)

MONTHS_SHORT = ("jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec")

MONTH_MAP = dict(zip(MONTHS_SHORT, MONTHS_FULL))


def infer_condition_from_path(fly_dir: Path) -> Optional[str]:
    """
    Infer experimental condition from fly directory name or parent directory.

    Searches for condition markers in directory path components.

    Args:
        fly_dir: Path to fly directory (e.g., .../opto_EB/october_fly_1/)

    Returns:
        Condition string or None if no match found

    Examples:
        >>> infer_condition_from_path(Path("/data/opto_EB/october_fly_1/"))
        'opto_Eb'
        >>> infer_condition_from_path(Path("/data/controls/hex_control_batch/"))
        'hex_control'
        >>> infer_condition_from_path(Path("/data/september_opto_hex/"))
        'opto_hex'
    """
    # Check fly directory name and all parent directories
    path_str = str(fly_dir).lower()

    for pattern, condition in CONDITION_PATTERNS:
        if re.search(pattern, path_str, re.IGNORECASE):
            return condition

    # Check for metadata file (optional, if exists)
    metadata_file = fly_dir / "experiment_metadata.txt"
    if metadata_file.exists():
        try:
            content = metadata_file.read_text().lower()
            for pattern, condition in CONDITION_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    return condition
        except Exception:
            pass

    return None  # No condition detected


def extract_month_from_path(fly_dir: Path) -> Optional[str]:
    """
    Extract month identifier from fly directory name.

    Args:
        fly_dir: Path to fly directory (e.g., october_fly_1)

    Returns:
        Month string (e.g., "october") or None

    Examples:
        >>> extract_month_from_path(Path("october_fly_1"))
        'october'
        >>> extract_month_from_path(Path("jan_batch_1"))
        'january'
        >>> extract_month_from_path(Path("september_09_fly_1"))
        'september'
    """
    name_lower = fly_dir.name.lower()

    # Check full month names
    for month in MONTHS_FULL:
        if name_lower.startswith(month):
            return month

    # Check short month names
    for short, full in MONTH_MAP.items():
        if name_lower.startswith(short):
            return full

    return None


def create_condition_key(fly_dir: Path) -> str:
    """
    Create a combined month_condition key for organizing results.

    Args:
        fly_dir: Path to fly directory

    Returns:
        Combined key (e.g., "october_hex_control") or "unknown" if undetectable

    Examples:
        >>> create_condition_key(Path("october_fly_1"))
        'october_unknown'
        >>> create_condition_key(Path("/data/opto_hex/november_fly_1"))
        'november_opto_hex'
    """
    month = extract_month_from_path(fly_dir)
    condition = infer_condition_from_path(fly_dir)

    if month is None:
        return "unknown"

    if condition:
        return f"{month}_{condition}"
    else:
        return f"{month}_unknown"
