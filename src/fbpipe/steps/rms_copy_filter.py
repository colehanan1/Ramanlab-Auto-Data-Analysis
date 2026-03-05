from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Settings, get_main_directories
from ..utils.columns import (
    EYE_CLASS,
    find_proboscis_distance_percentage_column,
    find_proboscis_xy_columns,
)
from ..utils.fly_files import iter_fly_distance_csvs


def main(cfg: Settings) -> None:
    force_recompute = bool(getattr(getattr(cfg, "force", None), "pipeline", False))
    roots = get_main_directories(cfg)
    for root in roots:
        for fly_dir in [p for p in root.iterdir() if p.is_dir()]:
            dest = fly_dir / "RMS_calculations"
            dest.mkdir(exist_ok=True)
            for csv_path, _, _ in iter_fly_distance_csvs(fly_dir, recursive=True):
                if dest in csv_path.parents:
                    continue

                try:
                    df = pd.read_csv(csv_path)
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
                if (
                    not force_recompute
                    and out_path.exists()
                    and out_path.stat().st_mtime >= csv_path.stat().st_mtime
                ):
                    print(f"[RMS] Skipping up-to-date output: {out_path.name}")
                    continue
                try:
                    df[cols].to_csv(out_path, index=False)
                    print(f"[RMS] {csv_path.name} → {out_path.name}")
                except Exception as exc:
                    print(f"[RMS] Skip {csv_path}: {exc}")
