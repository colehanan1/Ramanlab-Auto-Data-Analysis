"""Downstream rejection of physically-impossible proboscis detections.

Runs *after* YOLO/tracking, operating only on the exported per-fly distance
CSVs. It never fabricates or moves a detection — it only blanks (NaNs) suspect
proboscis points so they read as "no detection" downstream and can be
interpolated later if desired.

Two gates (see :class:`fbpipe.config.ProboscisFilterSettings`):

* geometry — drop any proboscis farther than ``max_eye_prob_distance_px`` from
  its frozen eye (a circle of that radius, applied for every fly count).
* velocity — drop any proboscis that jumps more than ``max_jump_px`` from the
  previous accepted position between consecutive detections.

The pass is idempotent: a blanked point reads as missing on the next run, so
re-running never removes additional valid points.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..config import Settings, get_main_directories
from ..utils.frozen_folders import iter_live_batch_dirs
from ..utils.distance_sanity import sanitize_proboscis_dataframe
from ..utils.fly_files import DERIVED_TABLE_DIRS, iter_fly_distance_csvs
from ..utils.parallel import parallel_map
from ..utils.rig_gates import apply_rig_gate_overrides
from ..utils.tables import read_table, write_table

log = logging.getLogger("fbpipe.reject_proboscis")


def _copies_rms_copy_filter_rewrites(fly_dir: Path, entries: List[Path]) -> Dict[Path, str]:
    """RMS copies in *entries* that rms_copy_filter rewrites from a per-trial
    source in *entries*, mapped to that source's stem.

    rms_copy_filter writes every table it finds outside
    ``<fly_dir>/RMS_calculations`` to ``<fly_dir>/RMS_calculations/updated_<stem>
    .parquet``. Only that exact file of a source outside every
    ``DERIVED_TABLE_DIRS`` dir is mapped; orphan copies, legacy .csv copies,
    copies of copies and a nested batch's own copies (e.g. ``flagged/<batch>/
    RMS_calculations``) are not. *entries* are paths as iter_fly_distance_csvs
    yields them for *fly_dir*, i.e. below the resolved *fly_dir*.
    """
    base = fly_dir.expanduser().resolve()
    depth = len(base.parts)
    derived = frozenset(DERIVED_TABLE_DIRS)
    sources = {p.stem for p in entries if derived.isdisjoint(p.parts[depth:-1])}
    rms_dir = base / "RMS_calculations"
    rewritten: Dict[Path, str] = {}
    for p in entries:
        if p.parent != rms_dir or p.suffix != ".parquet" or not p.stem.startswith("updated_"):
            continue
        if p.stem[len("updated_"):] in sources:
            rewritten[p] = p.stem[len("updated_"):]
    return rewritten


def _process_fly_dir(fly_dir: Path, cfg: Settings) -> Tuple[int, int, int]:
    """Process a single fly directory: sanitize all distance CSVs it contains.

    This is a module-level function so it is picklable by joblib's loky backend.
    All values it needs are derived from ``cfg`` (passed explicitly) or from
    ``fly_dir`` — no outer-scope closure variables are captured.

    Returns ``(files_modified, geometry_points_removed, velocity_points_removed)``
    so :func:`main` can report an aggregate summary across workers.
    """
    cfg = apply_rig_gate_overrides(cfg, fly_dir)
    pf = cfg.proboscis_filter
    max_dist = float(pf.max_eye_prob_distance_px)
    max_jump = float(pf.max_jump_px)
    up_divisor = float(getattr(pf, "up_divisor", 4.0))
    force_recompute = bool(getattr(getattr(cfg, "force", None), "pipeline", False))

    entries = [p for p, _, _ in iter_fly_distance_csvs(fly_dir, recursive=True)]
    # Forced, rms_copy_filter (later in the same run) rewrites the RMS copy of
    # every source it can read, from that source as gated here, so gating the
    # copy as well is wasted work: such copies go last and are skipped once
    # their source has been read. Everything else is gated as before: orphan
    # copies, a nested batch's copies, copies of copies, the copy of a source
    # that does not read, and every copy in a non-force run (rms_copy_filter
    # keeps a copy that is newer than its source).
    rewritten = _copies_rms_copy_filter_rewrites(fly_dir, entries) if force_recompute else {}
    read_stems: set[str] = set()

    files_modified = 0
    geo_total = 0
    vel_total = 0
    for csv_path in [p for p in entries if p not in rewritten] + list(rewritten):
        if rewritten.get(csv_path) in read_stems:
            continue
        try:
            df = read_table(csv_path)
        except Exception as exc:
            log.warning("[PROB-FILTER] Failed to read %s: %s", csv_path, exc)
            continue
        read_stems.add(csv_path.stem)

        cleaned, geo_count, vel_count = sanitize_proboscis_dataframe(
            df,
            max_distance_px=max_dist,
            max_jump_px=max_jump,
            up_divisor=up_divisor,
        )

        if geo_count or vel_count:
            write_table(cleaned, csv_path)
            files_modified += 1
            geo_total += geo_count
            vel_total += vel_count
            log.info(
                "[PROB-FILTER] %s: dropped %d out-of-radius, %d impossible-jump",
                csv_path.name,
                geo_count,
                vel_count,
            )
    return files_modified, geo_total, vel_total


def main(cfg: Settings) -> None:
    pf = getattr(cfg, "proboscis_filter", None)
    if pf is None or not getattr(pf, "enabled", False):
        log.info("[PROB-FILTER] Proboscis rejection is disabled in config")
        return

    max_dist = float(pf.max_eye_prob_distance_px)
    up_divisor = float(getattr(pf, "up_divisor", 4.0))
    max_jump = float(pf.max_jump_px)
    log.info(
        "[PROB-FILTER] Rejecting proboscis beyond eye gate (left/right/down %.1fpx, up %.1fpx), or jumping > %.1fpx/frame",
        max_dist,
        max_dist / up_divisor if up_divisor else max_dist,
        max_jump,
    )

    roots = get_main_directories(cfg)

    total_modified = 0
    total_geo = 0
    total_vel = 0
    for root in roots:
        if not root.is_dir():
            log.info("[PROB-FILTER] main_directories entry does not exist: %s", root)
            continue

        # Frozen experiment folders are skipped: no re-derivation, and their
        # existing per-trial CSVs stay put so build_wide_csv still emits them.
        fly_dirs = iter_live_batch_dirs(cfg, root)
        results = parallel_map(
            partial(_process_fly_dir, cfg=cfg),
            fly_dirs,
            enabled=cfg.parallel.enabled,
            n_jobs=cfg.parallel.n_jobs,
        )
        for files_modified, geo, vel in results:
            total_modified += files_modified
            total_geo += geo
            total_vel += vel

    log.info(
        "[PROB-FILTER] Done: %d CSVs modified, %d geometry + %d velocity points removed",
        total_modified,
        total_geo,
        total_vel,
    )
