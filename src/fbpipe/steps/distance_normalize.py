
from __future__ import annotations
from pathlib import Path
import glob, json
import numpy as np, pandas as pd
from ..config import Settings

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        stats_path = fly / "global_distance_stats_class_2.json"
        if not stats_path.exists():
            print(f"[NORM] Missing stats for {fly.name}; skipping")
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        gmin = stats["global_min"]; gmax = stats["global_max"]
        csvs = glob.glob(str(fly / "**" / "*merged.csv"), recursive=True)
        for f in csvs:
            df = pd.read_csv(f)
            if "distance_2_6" not in df.columns: continue
            d = pd.to_numeric(df["distance_2_6"], errors="coerce").to_numpy()
            perc = np.empty_like(d, dtype=float)
            over = d > gmax; under = d < gmin; inr = ~(over | under)
            perc[over] = 101.0; perc[under] = -1.0
            if gmax != gmin:
                perc[inr] = 100.0 * (d[inr] - gmin) / (gmax - gmin)
            else:
                perc[inr] = 0.0
            df["distance_percentage_2_6"] = perc
            df["min_distance_2_6"] = gmin; df["max_distance_2_6"] = gmax
            df.to_csv(f, index=False)
            print(f"[NORM] Updated {f}")
