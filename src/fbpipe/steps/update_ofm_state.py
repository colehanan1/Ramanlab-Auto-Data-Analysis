
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..config import Settings

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        rdir = fly / "RMS_calculations"
        if not rdir.is_dir(): continue
        outs = [p for p in fly.iterdir() if p.is_file() and p.suffix.lower()==".csv" and p.name.startswith("output_")]
        for f in rdir.iterdir():
            if not (f.is_file() and f.name.startswith("updated_") and f.suffix==".csv"):
                continue
            dfu = pd.read_csv(f)
            # Heuristic: try to find an "output" CSV with ActiveOFM column
            out_csv = None
            for o in outs:
                try:
                    dfo = pd.read_csv(o, nrows=5)
                    if "ActiveOFM" in dfo.columns:
                        out_csv = o; break
                except Exception:
                    continue
            if out_csv is None:
                # graceful skip
                continue
            dfo = pd.read_csv(out_csv)
            col = "ActiveOFM" if "ActiveOFM" in dfo.columns else None
            if col is None:
                continue
            on_idx = dfo.index[dfo[col].astype(str) != "off"].tolist()
            if not on_idx:
                continue
            first_on, last_on = min(on_idx), max(on_idx)
            if "Frame" in dfu.columns:
                dfu["OFM_State"] = dfu["Frame"].apply(lambda x: "before" if x < first_on else ("during" if first_on <= x <= last_on else "after"))
            elif "frame" in dfu.columns:
                dfu["OFM_State"] = dfu["frame"].apply(lambda x: "before" if x < first_on else ("during" if first_on <= x <= last_on else "after"))
            else:
                dfu["OFM_State"] = "unknown"
            dfu.to_csv(f, index=False)
            print(f"[OFM] updated {f.name} using {out_csv.name}")
