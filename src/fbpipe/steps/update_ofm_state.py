
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..config import Settings, get_main_directories
from ..utils.tables import read_table, write_table, table_path, resolve_existing, read_schema_columns

_OFM_COL_CANDIDATES = ("ActiveOFM", "Active OFM Pin")


def _find_ofm_col(df: pd.DataFrame) -> str | None:
    """Return the first OFM column name present in *df*, or None."""
    for candidate in _OFM_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def main(cfg: Settings):
    force_recompute = bool(getattr(getattr(cfg, "force", None), "pipeline", False))
    roots = get_main_directories(cfg)
    for root in roots:
        for fly in [p for p in root.iterdir() if p.is_dir()]:
            rdir = fly / "RMS_calculations"
            if not rdir.is_dir():
                continue
            outs = [p for p in fly.iterdir() if p.is_file() and p.suffix.lower()==".csv" and p.name.startswith("output_")]
            # Discover updated_* RMS tables, de-duplicated by stem so a stale
            # .csv left alongside its .parquet is not processed twice (.parquet
            # preferred).
            _rms_by_stem: dict[str, Path] = {}
            for p in sorted(rdir.iterdir()):
                if not (p.is_file() and p.name.startswith("updated_") and p.suffix in (".csv", ".parquet")):
                    continue
                if p.stem not in _rms_by_stem or p.suffix == ".parquet":
                    _rms_by_stem[p.stem] = p
            for f in _rms_by_stem.values():
                dfu = read_table(f)  # pipeline-produced RMS file (Rule 1)
                # Heuristic: try to find an "output" CSV with an OFM column
                out_csv = None
                for o in outs:
                    try:
                        dfo = read_table(o).head(5)  # external rig output_*.csv (Rule 2)
                        if _find_ofm_col(dfo) is not None:
                            out_csv = o; break
                    except Exception:
                        continue
                if out_csv is None:
                    # graceful skip
                    continue
                if not force_recompute:
                    try:
                        # Rule 3/5: use read_schema_columns for column-existence check;
                        # compare mtime against the actual parquet artifact (resolve_existing
                        # prefers .parquet so the skip fires only when the real output exists).
                        existing = resolve_existing(f)
                        if (
                            existing is not None
                            and "OFM_State" in read_schema_columns(existing)
                            and existing.stat().st_mtime >= out_csv.stat().st_mtime
                        ):
                            print(f"[OFM] Skipping up-to-date file: {f.name}")
                            continue
                    except Exception:
                        pass
                dfo = read_table(out_csv)  # external rig output_*.csv (Rule 2)
                col = _find_ofm_col(dfo)
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
                write_table(dfu, f)  # pipeline-produced RMS file -> parquet (Rule 1)
                print(f"[OFM] updated {f.name} using {out_csv.name}")
