"""Canonical fly-type (genotype) resolution + new-type alerting.

The Pi rig records a free-text ``Fly Type:`` in each batch's
``session_metadata.txt`` (one per batch; all flies in a batch share it). The
recorded strings drift in spelling/casing across batches, so this module maps
every known variant to a small set of canonical labels used for grouping result
outputs (so different genotypes are never plotted together) and alerts via ntfy
when a brand-new, uncatalogued ``Fly Type:`` shows up.

ADDING A NEW FLY NAME
---------------------
Add the raw string under the right canonical key in ``KNOWN_FLY_TYPE_ALIASES``
below (or add a whole new canonical entry). That is the single place to edit:
it both fixes the grouping AND silences the "new fly type" ntfy alert for that
string. Normalisation ignores case and separators, so you only need one spelling
per genuinely different string.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from .notify import ntfy_notify

log = logging.getLogger("fbpipe.fly_type")

UNKNOWN_FLY_TYPE = "Unknown-FlyType"

# Canonical labels (used verbatim in folder names, file names, and plot titles).
GR5A_OLD = "GR5a-Old"
GR5A_GCAMP8 = "GR5a-GCaMP8"
GR5A_NEW = "GR5a-New"
CANTON_S = "Canton-S"

SESSION_METADATA_FILENAME = "session_metadata.txt"

# === THE CATALOG: canonical label -> every known raw spelling ================
# Edit here to add/adjust fly names.
KNOWN_FLY_TYPE_ALIASES: Dict[str, Tuple[str, ...]] = {
    GR5A_OLD: (
        "GR5a-Retinol",
        "GR5a-OLD",
        "GR5a-RET-old",
        "GR5a-Retinol-OLD",
    ),
    GR5A_GCAMP8: (
        "GR5a-Retinol-Gcamp86",
        "GR5a-RET-GCamp86",
        "GR5a-RET-GCAMP86",
        "GR5a-Retinol-GCamp86",
        "GR5a-Gcamp86",
    ),
    GR5A_NEW: (
        "GR5a-Retinol-New",
    ),
    CANTON_S: (
        "Canton-S",
        "Canton-s",
        "CantonS",
        "Canton",
    ),
}


def _normalize(s: str) -> str:
    """Lowercase and drop all non-alphanumerics so case/separator drift collapses."""

    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


# Precomputed lookups derived from the catalog.
_ALIAS_TO_CANON: Dict[str, str] = {
    _normalize(alias): canon
    for canon, aliases in KNOWN_FLY_TYPE_ALIASES.items()
    for alias in aliases
}
_KNOWN_NORM = set(_ALIAS_TO_CANON)

_FLY_TYPE_RE = re.compile(r"^[ \t]*fly[ \t_]*type[ \t]*:[ \t]*(.+?)[ \t]*$", re.IGNORECASE | re.MULTILINE)

# Raw strings already alerted on this run, so we notify at most once per variant.
_alerted_new: set[str] = set()


def _heuristic_canon(norm: str) -> str:
    """Best-effort canonical for an uncatalogued string (provisional)."""

    if "new" in norm:
        return GR5A_NEW
    if "gcamp" in norm:
        return GR5A_GCAMP8
    if "old" in norm or "retinol" in norm or "ret" in norm or "gr5a" in norm:
        return GR5A_OLD
    return UNKNOWN_FLY_TYPE


def canonical_fly_type(raw: Optional[str]) -> str:
    """Map a raw ``Fly Type:`` string to its canonical label.

    Catalogued strings (``KNOWN_FLY_TYPE_ALIASES``) resolve exactly; anything
    else falls back to a heuristic provisional mapping.
    """

    if not raw:
        return UNKNOWN_FLY_TYPE
    norm = _normalize(raw)
    if not norm:
        return UNKNOWN_FLY_TYPE
    if norm in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[norm]
    return _heuristic_canon(norm)


def is_known_fly_type(raw: Optional[str]) -> bool:
    """True if ``raw`` is in the catalog (case/separator-insensitive)."""

    if not raw:
        return False
    return _normalize(raw) in _KNOWN_NORM


def maybe_alert_new_fly_type(raw: Optional[str], context: str = "", notify: bool = True) -> bool:
    """If ``raw`` is an uncatalogued fly type, log + ntfy once. Returns True if new."""

    if not raw:
        return False
    norm = _normalize(raw)
    if not norm or norm in _KNOWN_NORM or norm in _alerted_new:
        return False

    _alerted_new.add(norm)
    provisional = canonical_fly_type(raw)
    where = f" ({context})" if context else ""
    log.warning(
        "[FLY-TYPE] New/uncatalogued Fly Type %r%s; provisionally grouped as %s. "
        "Add it to KNOWN_FLY_TYPE_ALIASES in fbpipe/utils/fly_type.py.",
        raw, where, provisional,
    )
    if notify:
        ntfy_notify(
            "New fly type detected",
            f"Uncatalogued Fly Type: '{raw}'{where}\n"
            f"Provisionally grouped as: {provisional}\n"
            f"Add it to KNOWN_FLY_TYPE_ALIASES in fbpipe/utils/fly_type.py.",
            priority="high",
            tags="microscope,warning",
        )
    return True


def read_fly_type_raw(session_metadata_path: Path | str) -> Optional[str]:
    """Return the raw ``Fly Type:`` value from a session_metadata.txt, or None."""

    try:
        text = Path(session_metadata_path).read_text(errors="ignore")
    except OSError:
        return None
    match = _FLY_TYPE_RE.search(text)
    return match.group(1).strip() if match else None


def fly_type_for_dir(path: Path | str, alert_new: bool = False, context: str = "") -> str:
    """Resolve the canonical fly type for a trial/batch directory.

    Walks ``path`` and its ancestors looking for ``session_metadata.txt`` (the
    rig writes it in the batch folder, one level above each per-trial folder),
    then canonicalises its ``Fly Type:``. When ``alert_new`` is True, an
    uncatalogued raw value triggers a one-time ntfy alert. Returns
    ``UNKNOWN_FLY_TYPE`` when no metadata is found.
    """

    p = Path(path)
    for directory in (p, *p.parents):
        candidate = directory / SESSION_METADATA_FILENAME
        if candidate.exists():
            raw = read_fly_type_raw(candidate)
            if alert_new:
                maybe_alert_new_fly_type(raw, context=context or str(directory))
            return canonical_fly_type(raw)
    return UNKNOWN_FLY_TYPE


__all__ = [
    "UNKNOWN_FLY_TYPE",
    "GR5A_OLD",
    "GR5A_GCAMP8",
    "GR5A_NEW",
    "CANTON_S",
    "SESSION_METADATA_FILENAME",
    "KNOWN_FLY_TYPE_ALIASES",
    "canonical_fly_type",
    "is_known_fly_type",
    "maybe_alert_new_fly_type",
    "read_fly_type_raw",
    "fly_type_for_dir",
]
