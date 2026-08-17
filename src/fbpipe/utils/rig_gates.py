"""Resolve per-rig/date proboscis gate overrides for a batch directory.

Some rigs frame the fly so that a genuine full proboscis extension travels
farther in pixels than the global gates allow (Flybehavior2 reaches 251 px
against a 160 px gate, silently blanking real PER frames). The config's
``rig_gate_overrides:`` list widens the gates for recordings from a named
host after a given date, without loosening every other rig.

A batch directory is matched by:

* host — the ``Host:`` line in the batch's ``session_metadata.txt``
  (``Host: Flybehavior2 | OS: ...``), compared case-insensitively;
* date — the earliest ``output_*_YYYYMMDD_HHMMSS.*`` sidecar filename stamp,
  falling back to the metadata's "First training trial start" / "Intake
  Logged" lines. ``after`` is strict: a batch recorded ON the cutoff date
  keeps the global gates.

Batches with no metadata, an unmatched host, or no derivable date always keep
the global gates — the override can only be reached by an explicit match.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Optional

from ..config import RigGateOverride, Settings

__all__ = [
    "RigGateOverride",
    "apply_rig_gate_overrides",
    "read_batch_date",
    "read_batch_host",
]

_HOST_RE = re.compile(r"^Host:\s*([^\s|]+)", re.MULTILINE)
_SIDECAR_STAMP_RE = re.compile(r"_(20\d{2})(\d{2})(\d{2})_\d{6}\.[A-Za-z0-9]+$")
# Labeled metadata lines only — the file is full of unrelated dates (fly birth,
# starvation, retinal prep), so a bare "first ISO date in file" would be wrong.
_META_DATE_RES = (
    re.compile(r"^First training trial start \(local\):\s*(\d{4})-(\d{2})-(\d{2})", re.MULTILINE),
    re.compile(r"^Intake Logged \(UTC\):\s*(\d{4})-(\d{2})-(\d{2})", re.MULTILINE),
)


def _read_metadata(batch_dir: Path) -> str | None:
    meta = batch_dir / "session_metadata.txt"
    if not meta.is_file():
        return None
    try:
        return meta.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_batch_host(batch_dir: Path) -> Optional[str]:
    """The recording host of a batch, from its session_metadata.txt, or None."""
    text = _read_metadata(batch_dir)
    if text is None:
        return None
    m = _HOST_RE.search(text)
    return m.group(1) if m else None


def read_batch_date(batch_dir: Path) -> Optional[date]:
    """The recording date of a batch, or None when it cannot be derived."""
    stamps = []
    for f in batch_dir.glob("output_*"):
        m = _SIDECAR_STAMP_RE.search(f.name)
        if m:
            try:
                stamps.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                continue
    if stamps:
        return min(stamps)

    text = _read_metadata(batch_dir)
    if text is None:
        return None
    for pattern in _META_DATE_RES:
        m = pattern.search(text)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


def apply_rig_gate_overrides(cfg: Settings, batch_dir: Path) -> Settings:
    """Effective Settings for one batch dir: a modified copy when a
    ``rig_gate_overrides`` entry matches, otherwise ``cfg`` itself."""
    overrides = getattr(cfg, "rig_gate_overrides", ()) or ()
    if not overrides:
        return cfg

    host = read_batch_host(batch_dir)
    if host is None:
        return cfg
    batch_date = read_batch_date(batch_dir)
    if batch_date is None:
        return cfg

    for ov in overrides:
        if ov.host.lower() != host.lower() or batch_date <= ov.after:
            continue
        eff = cfg
        if ov.max_eye_prob_distance_px is not None:
            eff = replace(
                eff,
                proboscis_filter=replace(
                    eff.proboscis_filter,
                    max_eye_prob_distance_px=ov.max_eye_prob_distance_px,
                ),
            )
        scalar_updates = {
            key: value
            for key, value in (
                ("class2_max", ov.class2_max),
                ("three_fly_max_eye_prob_distance_px", ov.three_fly_max_eye_prob_distance_px),
            )
            if value is not None
        }
        if scalar_updates:
            eff = replace(eff, **scalar_updates)
        return eff
    return cfg
