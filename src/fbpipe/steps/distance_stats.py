from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..config import Settings
from ..utils.columns import find_proboscis_distance_column
from ..utils.fly_files import iter_fly_distance_csvs


def main(cfg: Settings) -> None:
    root = Path(cfg.main_directory).expanduser().resolve()
    print(f"[DIST] Starting distance stats scan in {root}")
    for fly_dir in [p for p in root.iterdir() if p.is_dir()]:
        print(f"[DIST] Inspecting fly directory: {fly_dir.name}")
        slot_ranges: Dict[str, Tuple[float, float]] = {}
        for csv_path, token, _ in iter_fly_distance_csvs(fly_dir, recursive=True):
            print(
                f"[DIST] Reading {csv_path.name} for slot '{token}' in {fly_dir.name}"
            )
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(
                    f"[DIST] Failed to read {csv_path.name} (slot {token}): {exc}"
                )
                continue

            dist_col = find_proboscis_distance_column(df)
            if dist_col is None:
                print(
                    f"[DIST] No proboscis distance column found in {csv_path.name};"
                    " expected aliases such as 'distance_2_8' or 'proboscis_distance'."
                )
                continue

            distances = pd.to_numeric(df[dist_col], errors="coerce")
            mask = distances.between(cfg.class2_min, cfg.class2_max, inclusive="both")
            vals = distances[mask]
            if vals.empty:
                print(
                    f"[DIST] Slot {token} in {csv_path.name} had no values within "
                    f"[{cfg.class2_min}, {cfg.class2_max}]; skipping."
                )
                continue

            local_min = float(vals.min())
            local_max = float(vals.max())
            print(
                f"[DIST] Slot {token}: local min={local_min:.3f}, max={local_max:.3f}"
            )
            current = slot_ranges.get(token)
            if current is None:
                slot_ranges[token] = (local_min, local_max)
            else:
                slot_ranges[token] = (min(current[0], local_min), max(current[1], local_max))

        if not slot_ranges:
            print(f"[DIST] No in-range distances for {fly_dir.name}")
            continue

        for token, (gmin, gmax) in sorted(slot_ranges.items(), key=lambda item: item[0]):
            slot_label = token.replace("_distances", "")
            stats = {"global_min": gmin, "global_max": gmax}
            stats_path = fly_dir / f"{slot_label}_global_distance_stats_class_2.json"
            with open(stats_path, "w", encoding="utf-8") as fp:
                json.dump(stats, fp)
            print(f"[DIST] {fly_dir.name}/{slot_label}: min={gmin:.3f} max={gmax:.3f}")
            print(f"[DIST] Wrote stats JSON → {stats_path}")
