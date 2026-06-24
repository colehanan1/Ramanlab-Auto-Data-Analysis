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

import pandas as pd

from ..config import Settings, get_main_directories
from ..utils.distance_sanity import sanitize_proboscis_dataframe
from ..utils.fly_files import iter_fly_distance_csvs

log = logging.getLogger("fbpipe.reject_proboscis")


def main(cfg: Settings) -> None:
    pf = getattr(cfg, "proboscis_filter", None)
    if pf is None or not getattr(pf, "enabled", False):
        log.info("[PROB-FILTER] Proboscis rejection is disabled in config")
        return

    max_dist = float(pf.max_eye_prob_distance_px)
    max_jump = float(pf.max_jump_px)
    up_divisor = float(getattr(pf, "up_divisor", 4.0))
    log.info(
        "[PROB-FILTER] Rejecting proboscis beyond eye gate (left/right/down %.1fpx, up %.1fpx), or jumping > %.1fpx/frame",
        max_dist,
        max_dist / up_divisor if up_divisor else max_dist,
        max_jump,
    )

    roots = get_main_directories(cfg)
    files_modified = 0
    total_geo = 0
    total_vel = 0

    for root in roots:
        if not root.is_dir():
            log.info("[PROB-FILTER] main_directories entry does not exist: %s", root)
            continue

        for fly_dir in [p for p in root.iterdir() if p.is_dir()]:
            for csv_path, token, _ in iter_fly_distance_csvs(fly_dir, recursive=True):
                try:
                    df = pd.read_csv(csv_path)
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
                    cleaned.to_csv(csv_path, index=False)
                    files_modified += 1
                    total_geo += geo_count
                    total_vel += vel_count
                    log.info(
                        "[PROB-FILTER] %s: dropped %d out-of-radius, %d impossible-jump",
                        csv_path.name,
                        geo_count,
                        vel_count,
                    )

    log.info(
        "[PROB-FILTER] Done: %d CSVs modified, %d geometry + %d velocity points removed",
        files_modified,
        total_geo,
        total_vel,
    )
