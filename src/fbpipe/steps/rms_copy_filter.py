from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Settings
from ..utils.columns import (
    find_proboscis_distance_percentage_column,
    find_proboscis_xy_columns,
)
from ..utils.fly_files import iter_fly_distance_csvs


def main(cfg: Settings) -> None:
    root = Path(cfg.main_directory).expanduser().resolve()
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

            cols = [c for c in ("frame", "timestamp", "x_class2", "y_class2") if c in df.columns]
            x_prob, y_prob = find_proboscis_xy_columns(df)
            if x_prob and x_prob not in cols:
                cols.append(x_prob)
            if y_prob and y_prob not in cols:
                cols.append(y_prob)

            pct_col = find_proboscis_distance_percentage_column(df)
            if pct_col and pct_col not in cols:
                cols.append(pct_col)

            out_path = dest / ("updated_" + csv_path.name)
            try:
                df[cols].to_csv(out_path, index=False)
                print(f"[RMS] {csv_path.name} → {out_path.name}")
            except Exception as exc:
                print(f"[RMS] Skip {csv_path}: {exc}")
