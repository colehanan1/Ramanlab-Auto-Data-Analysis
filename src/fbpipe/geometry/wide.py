"""Wide tables of the geometry pipeline, in production's column order.

One row per (batch, fly slot, trial) and one file per measure. The metadata
block keeps ``dataset, fly, fly_number`` first (``wide_to_matrix`` relies on
``fly_number`` at index 2) and ``frozen`` last; the samples are frame-indexed
``dir_val_0 .. dir_val_{N-1}``, NaN-padded to the longest trial -- exactly what
``fbpipe.analysis.traces.read_wide_table`` and the production readers expect.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.frozen_folders import FROZEN_COLUMN
from . import columns as C
from .config import GeometrySettings
from .steps import load_trial_entries

WIDE_MEASURES: Dict[str, str] = {
    "per": C.PRO_PCT,
    "abdomen_angle": C.ABD_ANGLE_PCT,
    "abdomen_length": C.ABD_LEN_PCT,
    "abdomen_width": C.ABD_WID_PCT,
}
WIDE_META_COLUMNS: List[str] = [
    "dataset", "fly", "fly_number", "fly_type", "trial_index", "odor", "has_odor",
    "trial_odor_on_s", "trial_odor_off_s", "trial_duration_s", "trial_light_on_s", "trial_light_off_s",
    "trace_len", "tracked_fraction", "trial_type", "trial_label", "fps", FROZEN_COLUMN,
]


def batch_wide_rows(batch_dir: Path, dataset: str, frozen: bool, gcfg: GeometrySettings
                    ) -> Dict[str, List[Tuple[dict, np.ndarray]]]:
    rows: Dict[str, List[Tuple[dict, np.ndarray]]] = {m: [] for m in WIDE_MEASURES}
    for _, info, tables in load_trial_entries(batch_dir, gcfg):
        for slot, path in sorted(tables.items()):
            df = C.read_table_any(path)
            for measure, col in WIDE_MEASURES.items():
                trace = df[col].to_numpy(dtype=float) if col in df.columns else np.full(len(df), np.nan)
                meta = {
                    "dataset": dataset or info.dataset, "fly": Path(batch_dir).name, "fly_number": str(slot),
                    "fly_type": info.fly_type, "trial_index": int(info.trial_index), "odor": info.odor,
                    "has_odor": bool(info.has_odor), "trial_odor_on_s": info.odor_on_s,
                    "trial_odor_off_s": info.odor_off_s, "trial_duration_s": info.duration_s,
                    "trial_light_on_s": info.light_on_s, "trial_light_off_s": info.light_off_s,
                    "trace_len": int(trace.size),
                    "tracked_fraction": float(np.isfinite(trace).mean()) if trace.size else 0.0,
                    "trial_type": info.trial_type, "trial_label": info.label, "fps": float(info.fps),
                    FROZEN_COLUMN: bool(frozen),
                }
                rows[measure].append((meta, trace))
    return rows


def build_wide_tables(rows_by_measure: Dict[str, List[Tuple[dict, np.ndarray]]]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for measure, rows in rows_by_measure.items():
        max_len = max((t.size for _, t in rows), default=0)
        value_cols = [f"dir_val_{i}" for i in range(max_len)]
        records = []
        for meta, trace in rows:
            padded = np.full(max_len, np.nan)
            padded[: trace.size] = trace
            records.append({**{k: meta[k] for k in WIDE_META_COLUMNS}, **dict(zip(value_cols, padded))})
        tables[measure] = pd.DataFrame(records, columns=WIDE_META_COLUMNS + value_cols)
    return tables


def write_wide_tables(tables: Dict[str, pd.DataFrame], out_dir: Path, fmt: str) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {m: C.write_table_fmt(df, out_dir / f"geometry_rows_wide_{m}.{fmt}", fmt) for m, df in tables.items()}
