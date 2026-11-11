"""Pre/post phase names and the exposed-odor tag, by cohort protocol.

The ``*-Sensitivity-*`` cohorts run the odor panel, then an odor-only exposure
block (no light), then the panel again. Nothing is paired with the exposed
odor, so it is the *sensitized odor*, not a CS+, and the two panels are
pre-exposure and post-exposure. Figures used to call them pre-test /
post-training / CS+, the conditioning words.

Light-conditioned cohorts that also run a pre-test (``Hex-HexTest-Training-*``)
really are conditioning, so they keep their wording: every caller passes its
own text for those as ``other`` and gets it back unchanged.
"""
from __future__ import annotations

import re
from typing import Iterable

PRE_EXPOSURE = "Pre-exposure"
POST_EXPOSURE = "Post-exposure"
EXPOSURE = "exposure"
SENSITIZED_ODOR = "sensitized odor"

# "IAA-Sensitivity-1-1", "IAA-Sensitivity5s-1-1", "3Oct-Sensitivity-24-0.1",
# and free-text titles such as "IAA sensitivity".
_SENSITIVITY_RE = re.compile(r"(?:^|[-_ ])sensitivity", re.IGNORECASE)

_CASES = ("sentence", "lower", "title")


def is_sensitivity(cohort: str | None) -> bool:
    """True for a ``*-Sensitivity-*`` cohort (or a title naming one)."""
    return bool(_SENSITIVITY_RE.search(str(cohort or "")))


def all_sensitivity(cohorts: Iterable[str]) -> bool:
    """True when there is at least one cohort and every one is a sensitivity cohort."""
    names = [str(c) for c in cohorts]
    return bool(names) and all(is_sensitivity(c) for c in names)


def _cased(text: str, case: str) -> str:
    if case not in _CASES:
        raise ValueError(f"case must be one of {_CASES}, got {case!r}")
    if case == "lower":
        return text.lower()
    if case == "title":
        return "-".join(part[:1].upper() + part[1:] for part in text.split("-"))
    return text


def pre_label(cohort: str | None, other: str, *, case: str = "sentence") -> str:
    """``"Pre-exposure"`` for a sensitivity cohort, else ``other`` unchanged.

    ``case`` ("sentence", "lower" or "title" -> "Pre-Exposure") applies
    to the sensitivity wording only; pass ``other`` already cased.
    """
    return _cased(PRE_EXPOSURE, case) if is_sensitivity(cohort) else other


def post_label(cohort: str | None, other: str, *, case: str = "sentence") -> str:
    """``"Post-exposure"`` for a sensitivity cohort, else ``other`` unchanged."""
    return _cased(POST_EXPOSURE, case) if is_sensitivity(cohort) else other


def block_label(cohort: str | None, other: str = "training") -> str:
    """The block between the panels: ``"exposure"``, else ``other`` (``"training"``)."""
    return EXPOSURE if is_sensitivity(cohort) else other


def odor_tag(cohort: str | None, other: str = "CS+", *, capital: bool = False) -> str:
    """``"sensitized odor"`` for a sensitivity cohort, else ``other`` (``"CS+"``)."""
    if not is_sensitivity(cohort):
        return other
    return SENSITIZED_ODOR[:1].upper() + SENSITIZED_ODOR[1:] if capital else SENSITIZED_ODOR


__all__ = [
    "EXPOSURE",
    "POST_EXPOSURE",
    "PRE_EXPOSURE",
    "SENSITIZED_ODOR",
    "all_sensitivity",
    "block_label",
    "is_sensitivity",
    "odor_tag",
    "post_label",
    "pre_label",
]
