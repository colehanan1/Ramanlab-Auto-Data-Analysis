
from __future__ import annotations
from pathlib import Path
import json, glob
import pandas as pd
from ..config import Settings
from ..utils.columns import find_proboscis_distance_column

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        csvs = glob.glob(str(fly / "**" / "*merged.csv"), recursive=True)
        gmin, gmax = float("inf"), float("-inf")
        for f in csvs:
            df = pd.read_csv(f)
            dist_col = find_proboscis_distance_column(df)
            if dist_col is None:
                continue
            dist = pd.to_numeric(df[dist_col], errors="coerce")
            mask = dist.between(cfg.class2_min, cfg.class2_max, inclusive="both")
            vals = dist[mask]
            if vals.empty: continue
            gmin = min(gmin, float(vals.min()))
            gmax = max(gmax, float(vals.max()))
        if gmin == float("inf"): 
            print(f"[DIST] No in-range distances for {fly.name}")
            continue
        out = {"global_min": gmin, "global_max": gmax}
        with open(fly / "global_distance_stats_class_2.json", "w", encoding="utf-8") as fp:
            json.dump(out, fp)
        print(f"[DIST] {fly.name}: min={gmin:.3f} max={gmax:.3f}")
