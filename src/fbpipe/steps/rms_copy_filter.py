
from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd
from ..config import Settings
from ..utils.columns import find_proboscis_distance_percentage_column, find_proboscis_xy_columns

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
                cols = [c for c in ("frame", "timestamp", "x_class2", "y_class2") if c in df.columns]
                x_prob, y_prob = find_proboscis_xy_columns(df)
                if x_prob and x_prob not in cols:
                    cols.append(x_prob)
                if y_prob and y_prob not in cols:
                    cols.append(y_prob)
                pct_col = find_proboscis_distance_percentage_column(df)
                if pct_col and pct_col not in cols:
                    cols.append(pct_col)
                out = dest / ("updated_" + p.name)
                df[cols].to_csv(out, index=False)
                print(f"[RMS] {p.name} → {out.name}")
            except Exception as e:
                print(f"[RMS] Skip {p}: {e}")
