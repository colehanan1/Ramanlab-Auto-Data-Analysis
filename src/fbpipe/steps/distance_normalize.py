from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..utils.columns import (
    PROBOSCIS_DISTANCE_COL,
    PROBOSCIS_DISTANCE_PCT_COL,
    PROBOSCIS_MAX_DISTANCE_COL,
    PROBOSCIS_MIN_DISTANCE_COL,
    find_proboscis_distance_column,
)
from ..utils.fly_files import iter_fly_distance_csvs


def main(cfg: Settings) -> None:
    root = Path(cfg.main_directory).expanduser().resolve()
    print(f"[NORM] Starting normalization in {root}")
    for fly_dir in [p for p in root.iterdir() if p.is_dir()]:
        print(f"[NORM] Processing fly directory: {fly_dir.name}")
        for csv_path, token, _ in iter_fly_distance_csvs(fly_dir, recursive=True):
            slot_label = token.replace("_distances", "")
            stats_path = fly_dir / f"{slot_label}_global_distance_stats_class_2.json"
            if not stats_path.exists():
                legacy_path = fly_dir / "global_distance_stats_class_2.json"
                if not legacy_path.exists():
                    print(
                        f"[NORM] Missing stats JSON for {fly_dir.name}/{slot_label}; "
                        f"expected {stats_path.name} or legacy file."
                    )
                    print(
                        f"[NORM] Skipping normalization for CSV {csv_path.name} due to missing stats."
                    )
                    continue
                stats = json.loads(legacy_path.read_text(encoding="utf-8"))
            else:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            print(
                f"[NORM] Loaded stats for {fly_dir.name}/{slot_label}: "
                f"min={stats['global_min']}, max={stats['global_max']}"
            )

            gmin = float(stats["global_min"])
            gmax = float(stats["global_max"])

            df = pd.read_csv(csv_path)
            dist_col = find_proboscis_distance_column(df)
            if dist_col is None:
                print(
                    f"[NORM] No proboscis distance column found in {csv_path.name};"
                    " expected aliases such as 'distance_2_8' or 'proboscis_distance'."
                )
                continue

            d = pd.to_numeric(df[dist_col], errors="coerce").to_numpy()
            perc = np.empty_like(d, dtype=float)
            over = d > gmax
            under = d < gmin
            inr = ~(over | under)
            perc[over] = 101.0
            perc[under] = -1.0
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
            for legacy_pct in (
                "distance_percentage_2_6",
                "distance_pct_2_6",
                "distance_percent",
                "distance_pct",
            ):
                if legacy_pct in df.columns:
                    df[legacy_pct] = perc

            df[PROBOSCIS_MIN_DISTANCE_COL] = gmin
            df[PROBOSCIS_MAX_DISTANCE_COL] = gmax
            if "min_distance_2_6" in df.columns:
                df["min_distance_2_6"] = gmin
            if "max_distance_2_6" in df.columns:
                df["max_distance_2_6"] = gmax

            df.to_csv(csv_path, index=False)
            print(
                f"[NORM] Normalized distances for {csv_path.name} with slot {slot_label}; "
                f"updated {len(df)} rows."
            )
