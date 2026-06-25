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
from typing import Tuple

import pandas as pd

from ..config import Settings, get_main_directories
from ..utils.distance_sanity import sanitize_proboscis_dataframe
from ..utils.fly_files import iter_fly_distance_csvs
from ..utils.parallel import parallel_map
from ..utils.tables import read_table, write_table

log = logging.getLogger("fbpipe.reject_proboscis")


def _process_fly_dir(fly_dir: Path, cfg: Settings) -> Tuple[int, int, int]:
    """Process a single fly directory: sanitize all distance CSVs it contains.

    This is a module-level function so it is picklable by joblib's loky backend.
    All values it needs are derived from ``cfg`` (passed explicitly) or from
    ``fly_dir`` — no outer-scope closure variables are captured.

    Returns ``(files_modified, geometry_points_removed, velocity_points_removed)``
    so :func:`main` can report an aggregate summary across workers.
    """
    pf = cfg.proboscis_filter
    max_dist = float(pf.max_eye_prob_distance_px)
    max_jump = float(pf.max_jump_px)
    up_divisor = float(getattr(pf, "up_divisor", 4.0))

    files_modified = 0
    geo_total = 0
    vel_total = 0
    for csv_path, token, _ in iter_fly_distance_csvs(fly_dir, recursive=True):
        try:
            df = read_table(csv_path)
        except Exception as exc:
            log.warning("[PROB-FILTER] Failed to read %s: %s", csv_path, exc)
            continue

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

        fly_dirs = [p for p in root.iterdir() if p.is_dir()]
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
