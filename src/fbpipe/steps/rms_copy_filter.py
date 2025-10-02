
from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd
from ..config import Settings

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        dest = fly / "RMS_calculations"
        dest.mkdir(exist_ok=True)
        csvs = glob.glob(str(fly / "**" / "*merged.csv"), recursive=True)
        for f in csvs:
            p = Path(f)
            if str(dest) in str(p.parent): 
                continue
            try:
                df = pd.read_csv(p)
                cols = ["frame", "timestamp", "x_class2","y_class2","x_class6","y_class6","distance_percentage_2_6"]
                cols = [c for c in cols if c in df.columns]
                out = dest / ("updated_" + p.name)
                df[cols].to_csv(out, index=False)
                print(f"[RMS] {p.name} → {out.name}")
            except Exception as e:
                print(f"[RMS] Skip {p}: {e}")
