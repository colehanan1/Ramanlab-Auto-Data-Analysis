"""Resolve the geometric anchor for a trial, per recording rig.

The proboscis angle is measured as the unsigned angle between the eye->anchor
and eye->proboscis vectors, where the anchor is a fixed point in the rig (the
odor tube). rig_3 is physically mirrored -- both the flies and the odor tube sit
on the opposite side -- so measuring it against the right-edge anchor used by
the other rigs inverts the extension signal and flips its angle multiplier into
0.5-1.0 where it should be 1.0-2.0.

Adding another mirrored rig later is a one-line edit to ``MIRRORED_RIGS``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

# Right edge, mid-height: the non-mirrored rigs.
DEFAULT_ANCHOR: Tuple[float, float] = (1080.0, 540.0)
# Left edge, mid-height: the mirrored rigs.
MIRRORED_ANCHOR: Tuple[float, float] = (0.0, 540.0)

MIRRORED_RIGS = frozenset({"rig_3"})

# Matches "rig_3" but deliberately NOT "rig3" (no underscore). The underscore
# is required as a defensive measure: rig_token scans every ancestor path
# component, so an ancestor directory whose NAME merely mentions a rig --
# e.g. "EB-Training-24-1_excl_rig3", which exists on disk under
# /home/ramanlab/Documents/cole/Results/Figures -- could otherwise be
# misread as a rig_3 trial. This is latent, not an observed production bug:
# resolve_anchor is only ever called on data paths (cfg.root.iterdir(),
# _discover_month_folders, video_path), never on Results/Figures, so that
# directory was never actually reachable through this code. Every real rig
# directory on disk uses the underscore form (rig_2, rig_3), so requiring it
# loses no real matches.
_RIG_RE = re.compile(r"rig_(\d+)", re.IGNORECASE)


def rig_token(path: str | Path) -> Optional[str]:
    """Return the normalized rig token (e.g. ``"rig_3"``) for *path*.

    Scans path components from the deepest upward so the rig nearest the trial
    wins over any similarly named ancestor. Returns ``None`` when no component
    names a rig.
    """
    for part in reversed(Path(path).parts):
        m = _RIG_RE.search(part)
        if m:
            return f"rig_{m.group(1)}"
    return None


def resolve_anchor(path: str | Path) -> Tuple[float, float]:
    """Return the ``(x, y)`` anchor to measure angles against for *path*.

    Falls back to :data:`DEFAULT_ANCHOR` for unknown or rig-less paths, so
    anything not explicitly mirrored keeps its current behaviour.
    """
    return MIRRORED_ANCHOR if rig_token(path) in MIRRORED_RIGS else DEFAULT_ANCHOR
