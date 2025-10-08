
from __future__ import annotations
from pathlib import Path
import json

import numpy as np, pandas as pd

from ..config import Settings
from ..utils.columns import (
    PROBOSCIS_DISTANCE_COL,
    PROBOSCIS_DISTANCE_PCT_COL,
    PROBOSCIS_MAX_DISTANCE_COL,
    PROBOSCIS_MIN_DISTANCE_COL,
    find_proboscis_distance_column,
)
from ..utils.csvs import gather_distance_csvs

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        stats_path = fly / "global_distance_stats_class_2.json"
        if not stats_path.exists():
            print(f"[NORM] Missing stats for {fly.name}; skipping")
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        gmin = stats["global_min"]; gmax = stats["global_max"]
        csvs = gather_distance_csvs(fly)
        for f in csvs:
            df = pd.read_csv(f)
            dist_col = find_proboscis_distance_column(df)
            if dist_col is None:
                continue
            d = pd.to_numeric(df[dist_col], errors="coerce").to_numpy()
            perc = np.empty_like(d, dtype=float)
            over = d > gmax; under = d < gmin; inr = ~(over | under)
            perc[over] = 101.0; perc[under] = -1.0
            if gmax != gmin:
                perc[inr] = 100.0 * (d[inr] - gmin) / (gmax - gmin)
            else:
                perc[inr] = 0.0
            df[dist_col] = d
            df[PROBOSCIS_DISTANCE_COL] = d
            if "distance_2_6" in df.columns:
                df["distance_2_6"] = d

            df[PROBOSCIS_DISTANCE_PCT_COL] = perc
            df["distance_percentage"] = perc
            for legacy_pct in ("distance_percentage_2_6", "distance_pct_2_6", "distance_percent", "distance_pct"):
                if legacy_pct in df.columns:
                    df[legacy_pct] = perc

            df[PROBOSCIS_MIN_DISTANCE_COL] = gmin
            df[PROBOSCIS_MAX_DISTANCE_COL] = gmax
            if "min_distance_2_6" in df.columns:
                df["min_distance_2_6"] = gmin
            if "max_distance_2_6" in df.columns:
                df["max_distance_2_6"] = gmax
            df.to_csv(f, index=False)
            print(f"[NORM] Updated {f}")
