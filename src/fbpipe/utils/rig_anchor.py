"""Resolve the geometric anchor for a trial, per recording rig.

The proboscis angle is measured as the unsigned angle between the eye->anchor
and eye->proboscis vectors, where the anchor is a fixed point in the rig (the
odor tube). rig_3 is physically mirrored -- both the flies and the odor tube sit
on the opposite side -- so measuring it against the right-edge anchor used by
the other rigs inverts the extension signal and flips its angle multiplier into
0.5-1.0 where it should be 1.0-2.0.

Adding another mirrored rig later is a one-line edit to ``MIRRORED_RIGS``.

Rigs without a ``rig_N`` folder suffix are identified by the ``Host:`` line in
the batch's ``session_metadata.txt`` instead: ``MIRRORED_HOSTS_AFTER`` maps a
host name (case-insensitive) to the date its rig was physically rearranged
into the mirrored geometry. Batches from that host recorded STRICTLY after
the date use the mirrored anchor; an explicit ``rig_N`` token always wins
over the host rule.
"""
from __future__ import annotations

import datetime as _dt
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

# Right edge, mid-height: the non-mirrored rigs.
DEFAULT_ANCHOR: Tuple[float, float] = (1080.0, 540.0)
# Left edge, mid-height: the mirrored rigs.
MIRRORED_ANCHOR: Tuple[float, float] = (0.0, 540.0)

MIRRORED_RIGS = frozenset({"rig_3"})

# Flybehavior2 was rearranged to match Flybehavior3/rig_3's mirrored geometry
# after 2026-07-25 (same cutoff as its rig_gate_overrides entry).
MIRRORED_HOSTS_AFTER: dict[str, _dt.date] = {"flybehavior2": _dt.date(2026, 7, 25)}

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


@lru_cache(maxsize=4096)
def _mirrored_by_host(path_str: str) -> bool:
    """True when *path_str* belongs to a batch whose recording host was
    rearranged into the mirrored geometry before this batch was recorded.

    Walks ancestors for the nearest ``session_metadata.txt`` (trial dirs and
    video files sit one level below it), mirroring ``rig_token``'s
    nearest-component-wins spirit. Missing metadata, an unlisted host, or an
    underivable date all resolve False so unknown paths keep their current
    behaviour.
    """
    # Imported here (not at module top) to keep this low-level module cheap to
    # import for the token-only fast path; rig_gates pulls in fbpipe.config.
    from .rig_gates import read_batch_date, read_batch_host

    p = Path(path_str)
    candidates = [p] if p.is_dir() else []
    candidates.extend(p.parents)
    for candidate in candidates[:4]:
        if not (candidate / "session_metadata.txt").is_file():
            continue
        host = read_batch_host(candidate)
        if host is None:
            return False
        cutoff = MIRRORED_HOSTS_AFTER.get(host.lower())
        if cutoff is None:
            return False
        batch_date = read_batch_date(candidate)
        return batch_date is not None and batch_date > cutoff
    return False


def resolve_anchor(path: str | Path) -> Tuple[float, float]:
    """Return the ``(x, y)`` anchor to measure angles against for *path*.

    An explicit ``rig_N`` path token wins first; rig-less paths consult the
    host+date rule (:data:`MIRRORED_HOSTS_AFTER`) via the batch's
    ``session_metadata.txt``. Falls back to :data:`DEFAULT_ANCHOR` for
    unknown paths, so anything not explicitly mirrored keeps its current
    behaviour.
    """
    token = rig_token(path)
    if token is not None:
        return MIRRORED_ANCHOR if token in MIRRORED_RIGS else DEFAULT_ANCHOR
    if _mirrored_by_host(str(path)):
        return MIRRORED_ANCHOR
    return DEFAULT_ANCHOR
