from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from ..config import Settings, get_main_directories
from ..utils.columns import (
    EYE_CLASS,
    find_proboscis_distance_percentage_column,
    find_proboscis_xy_columns,
)
from ..utils.fly_files import iter_fly_distance_csvs
from ..utils.parallel import parallel_map
from ..utils.tables import read_table, table_path, write_table


def _process_fly_dir(fly_dir: Path, cfg: Settings) -> None:
    """Process a single fly directory: copy and column-filter distance tables.

    Extracts the relevant coordinate and distance columns from every per-fly
    distance file found under *fly_dir* and writes filtered Parquet tables to
    ``<fly_dir>/RMS_calculations/``.  Files already up to date are skipped
    unless ``cfg.force.pipeline`` is True.

    This is a module-level function (not a closure) so it is picklable by
    joblib's loky backend for parallel execution.
    """
    force_recompute = bool(getattr(getattr(cfg, "force", None), "pipeline", False))
    dest = fly_dir / "RMS_calculations"
    dest.mkdir(exist_ok=True)
    for csv_path, _, _ in iter_fly_distance_csvs(fly_dir, recursive=True):
        if dest in csv_path.parents:
            continue

        try:
            df = read_table(csv_path)
        except Exception as exc:
            print(f"[RMS] Skip {csv_path}: {exc}")
            continue

        cols = [c for c in ("frame", "timestamp") if c in df.columns]
        eye_x_candidates = (
            f"x_class{EYE_CLASS}",
            f"x_class_{EYE_CLASS}",
            f"class{EYE_CLASS}_x",
            "x_class2",
            "x_class_2",
            "class2_x",
        )
        eye_y_candidates = (
            f"y_class{EYE_CLASS}",
            f"y_class_{EYE_CLASS}",
            f"class{EYE_CLASS}_y",
            "y_class2",
            "y_class_2",
            "class2_y",
        )
        for candidate in eye_x_candidates:
            if candidate in df.columns:
                cols.append(candidate)
                break
        for candidate in eye_y_candidates:
            if candidate in df.columns:
                cols.append(candidate)
                break
        x_prob, y_prob = find_proboscis_xy_columns(df)
        if x_prob and x_prob not in cols:
            cols.append(x_prob)
        if y_prob and y_prob not in cols:
            cols.append(y_prob)

        pct_col = find_proboscis_distance_percentage_column(df)
        if pct_col and pct_col not in cols:
            cols.append(pct_col)

        out_path = dest / ("updated_" + csv_path.name)
        out_parquet = table_path(out_path)
        if (
            not force_recompute
            and out_parquet.exists()
            and out_parquet.stat().st_mtime >= csv_path.stat().st_mtime
        ):
            print(f"[RMS] Skipping up-to-date output: {out_parquet.name}")
            continue
        try:
            written = write_table(df[cols], out_path)
            print(f"[RMS] {csv_path.name} → {written.name}")
        except Exception as exc:
            print(f"[RMS] Skip {csv_path}: {exc}")


def main(cfg: Settings) -> None:
    roots = get_main_directories(cfg)
    for root in roots:
        fly_dirs = [p for p in root.iterdir() if p.is_dir()]
        parallel_map(
            partial(_process_fly_dir, cfg=cfg),
            fly_dirs,
            enabled=cfg.parallel.enabled,
            n_jobs=cfg.parallel.n_jobs,
        )
