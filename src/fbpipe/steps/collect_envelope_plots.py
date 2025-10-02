
from __future__ import annotations
from pathlib import Path
import shutil
from ..config import Settings

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    dest = root / "all_envelope_plots"
    dest.mkdir(exist_ok=True, parents=True)
    count = 0
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        p = fly / "RMS_calculations" / "envelope_over_time_plots" / f"{fly.name}_envelope_over_time_subplots.png"
        if p.exists():
            shutil.copy2(p, dest / f"{fly.name}_envelope_over_time_subplots.png")
            count += 1
    print(f"[ENV-COLLECT] Collected {count} → {dest}")
