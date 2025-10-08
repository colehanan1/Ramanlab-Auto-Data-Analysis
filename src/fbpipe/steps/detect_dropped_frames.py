
from __future__ import annotations
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..utils.columns import find_proboscis_distance_column
from ..utils.csvs import gather_distance_csvs

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        csvs = gather_distance_csvs(fly)
        for f in csvs:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            if "frame" not in df.columns:
                continue
            present = sorted(pd.to_numeric(df["frame"], errors="coerce").dropna().astype(int).unique())
            if not present: continue
            minf, maxf = present[0], present[-1]
            expected = set(range(minf, maxf+1))
            missing = sorted(expected - set(present))
            dropped = []
            dist_col = find_proboscis_distance_column(df)
            if dist_col:
                dropped = df[df[dist_col].isna()]["frame"].tolist()
            all_dropped = sorted(set(missing) | set(dropped))
            out = Path(f).with_suffix("").as_posix() + "_dropped_frames.txt"
            with open(out, "w", encoding="utf-8") as fp:
                if not all_dropped:
                    fp.write("No dropped frames found.\n")
                else:
                    fp.write("Dropped frames (missing or NaN distance):\n")
                    for fr in all_dropped: fp.write(f"{fr}\n")
                    fp.write(f"\nTotal dropped frames: {len(all_dropped)}\n")
            print(f"[DROP] {out}")
