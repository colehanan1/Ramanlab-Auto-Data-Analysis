from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Settings
from ..utils.columns import find_proboscis_distance_column
from ..utils.fly_files import iter_fly_distance_csvs


def main(cfg: Settings) -> None:
    root = Path(cfg.main_directory).expanduser().resolve()
    print(f"[DROP] Scanning {root} for dropped frames")
    for fly_dir in [p for p in root.iterdir() if p.is_dir()]:
        print(f"[DROP] Checking fly directory: {fly_dir.name}")
        for csv_path, _, _ in iter_fly_distance_csvs(fly_dir, recursive=True):
            print(f"[DROP] Evaluating frames in {csv_path.name}")
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if "frame" not in df.columns:
                print(
                    f"[DROP] CSV {csv_path.name} is missing a 'frame' column; "
                    "cannot compute dropped frames."
                )
                continue

            present = (
                pd.to_numeric(df["frame"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            present = sorted(present.tolist())
            if not present:
                print(f"[DROP] CSV {csv_path.name} has no valid frame numbers; skipping.")
                continue

            min_frame, max_frame = present[0], present[-1]
            expected = set(range(min_frame, max_frame + 1))
            missing = sorted(expected - set(present))

            dropped = []
            dist_col = find_proboscis_distance_column(df)
            if dist_col:
                dropped = df[df[dist_col].isna()]["frame"].tolist()
            else:
                print(
                    f"[DROP] No proboscis distance column found in {csv_path.name}; "
                    "NaN distance-based drops cannot be identified."
                )

            all_dropped = sorted(set(missing) | set(dropped))
            out_path = csv_path.with_suffix("").as_posix() + "_dropped_frames.txt"
            with open(out_path, "w", encoding="utf-8") as fp:
                if not all_dropped:
                    fp.write("No dropped frames found.\n")
                else:
                    fp.write("Dropped frames (missing or NaN distance):\n")
                    for frame in all_dropped:
                        fp.write(f"{frame}\n")
                    fp.write(f"\nTotal dropped frames: {len(all_dropped)}\n")
            print(
                f"[DROP] Wrote dropped frame report for {csv_path.name} → {out_path}"
            )
