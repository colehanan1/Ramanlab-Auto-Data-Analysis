"""Which frames of a trial table the stimulus light was on.

Under the light YOLO misplaces the eye and the light-evoked PER is usually the
biggest extension a fly makes -- so, optionally, light frames are left out of
each fly's global min/max (``distance_limits.exclude_light_frames_from_minmax``).
The light window comes from the trial's ``_trial_meta.json`` (``light_on_s`` to
``light_last_s``, the end of the whole light span -- a pulse train included --
parsed from the sensors CSV in video time); a trial without one has no light.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from . import trial_metadata as _tm

LOGGER = logging.getLogger(__name__)

_COPY_RE = re.compile(r"^updated_(.+)_fly\d+_distances(?:_merged)?\.(?:parquet|csv)$")
_TABLE_RE = re.compile(r"^(.+)_fly\d+_distances\.(?:parquet|csv)$")
_LIGHT_KEYS = ("light_on_s", "light_off_s", "light_last_s")
_fresh_light_memo: dict[tuple[str, int], Optional[dict]] = {}


def trial_meta_for_table(table_path: Path, fly_dir: Optional[Path] = None) -> dict:
    """The trial metadata of a per-fly table: its own folder's, or -- for an
    ``RMS_calculations/updated_*`` copy -- the source trial folder's. {} if none.

    A cache from an older schema gets its light window re-derived (see
    :func:`_current_light`)."""
    table_path = Path(table_path)
    candidates = [table_path.parent / "_trial_meta.json"]
    m = _COPY_RE.match(table_path.name) or _TABLE_RE.match(table_path.name)
    if m:
        base = Path(fly_dir) if fly_dir is not None else table_path.parent.parent
        candidates.append(base / m.group(1) / "_trial_meta.json")
    for c in candidates:
        try:
            meta = json.loads(c.read_text())
        except (OSError, ValueError):
            continue
        return _current_light(meta, c)
    return {}


def _current_light(meta: dict, cache_path: Path) -> dict:
    """``meta`` with its light window re-derived when an older schema wrote it.

    This reader opens the cache itself rather than through
    ``load_trial_metadata``, so a schema bump would never reach it: a schema-3
    cache times some light windows ~30 s early and ends a pulse train at its
    first pulse. Such a cache gets its light fields re-derived from the trial's
    sensors log, in memory -- nothing is written; the other fields are kept. A
    dict without ``_schema_version`` (hand-made) is taken as it is, and so is a
    stale one whose trial cannot be re-read."""
    version = meta.get("_schema_version")
    if version is None or version == _tm._CACHE_SCHEMA_VERSION:
        return meta
    try:
        key = (str(cache_path), cache_path.stat().st_mtime_ns)
    except OSError:
        return meta
    if key not in _fresh_light_memo:
        try:
            fresh = _tm.load_trial_metadata(cache_path.parent, use_cache=False)
            _fresh_light_memo[key] = {k: getattr(fresh, k) for k in _LIGHT_KEYS}
        except Exception as exc:  # a refresh must never take a min/max step down
            LOGGER.warning("Cannot re-derive the light window of %s (%s); using the cached one",
                           cache_path.parent, exc)
            _fresh_light_memo[key] = None
    fresh_light = _fresh_light_memo[key]
    return meta if fresh_light is None else {**meta, **fresh_light}


def light_frame_mask(frames, meta: dict, before_s: float, after_s: float) -> np.ndarray:
    """True for frames from ``light_on_s - before_s`` to the light's end ``+ after_s``.

    The light ends at ``light_last_s`` -- its last logged event, so every
    pulse of a pulse train is covered -- or, in a dict without that key (a
    cache from before it, or a hand-made one), at ``light_off_s``. ``frames``
    are video frame numbers; times come from the trial fps (40 when absent).
    A light with no logged off time runs to the end."""
    frames = np.asarray(frames, dtype=float)
    on = meta.get("light_on_s")
    if on is None:
        return np.zeros(frames.shape, dtype=bool)
    fps = float(meta.get("fps") or 40.0)
    off = meta.get("light_last_s")
    if off is None:
        off = meta.get("light_off_s")
    first = (float(on) - before_s) * fps
    last = np.inf if off is None else (float(off) + after_s) * fps
    return (frames >= first - 1e-9) & (frames <= last + 1e-9)


def light_exclusion_applies(cfg, fly_dir) -> bool:
    """Whether light frames are left out of this batch's min/max: the option
    is on and the batch's dataset is listed (an empty list means every one)."""
    if not getattr(cfg, "minmax_exclude_light", False):
        return False
    listed = tuple(getattr(cfg, "minmax_light_datasets", ()) or ())
    return not listed or Path(fly_dir).parent.name in listed
