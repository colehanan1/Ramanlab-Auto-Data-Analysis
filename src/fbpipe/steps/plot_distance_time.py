
from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ..config import Settings
from ..utils.timestamps import timestamp_to_seconds

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        csvs = glob.glob(str(fly / "**" / "*merged.csv"), recursive=True)
        for f in csvs:
            df = pd.read_csv(f)
            if {"timestamp", "distance_percentage_2_6"}.issubset(df.columns):
                ts = df["timestamp"].apply(timestamp_to_seconds).dropna()
                if ts.empty: 
                    print(f"[PLOT] skip {f}: invalid timestamps")
                    continue
                ts = ts - ts.iloc[0]
                plt.figure(figsize=(10,6))
                plt.plot(ts, df["distance_percentage_2_6"], marker="o", linestyle="-", markersize=2, label="Distance %")
                plt.xlabel("Time (s)"); plt.ylabel("Normalized Distance %"); plt.title(Path(f).name)
                plt.grid(True); plt.legend()
                out = Path(f).with_suffix("").as_posix() + "_time.png"
                plt.savefig(out); plt.close()
                print(f"[PLOT] {out}")
