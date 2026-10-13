"""Per-fly baselines over every trial of a batch, and the percentage columns.

* PER: production's formula (min/max of the accepted proboscis distances over
  all trials, ``max(gmax, floor)`` as the denominator), with ``per_min_px``
  keeping a degenerate distance from becoming the minimum.
* Abdomen angle: 0 % at the fly's straightest observed angle, 100 % at its most
  curled, over every tip-paired frame of every trial (user spec 2026-09-08).
* Abdomen length/width: % of the resting median over the pre-odor windows.

Raw columns are never modified, so re-running is idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Settings
from . import columns as C
from . import metrics as M
from .config import GeometrySettings
from .steps import geometry_dir, load_trial_entries
from .trial_meta import TrialInfo


def pre_odor_mask(df: pd.DataFrame, info: TrialInfo, settings: Settings) -> np.ndarray:
    frame = df[C.FRAME].to_numpy(dtype=float)
    if info.has_odor and info.odor_on_frame is not None:
        return frame < int(info.odor_on_frame)
    return frame < int(round(settings.odor_on_s * max(info.fps, 1e-6)))


def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def _rest(values: Sequence[np.ndarray], fallback: Sequence[np.ndarray], min_samples: int) -> dict:
    pre = _finite(np.concatenate(values)) if values else np.array([])
    if pre.size >= min_samples:
        return {"rest_px": float(np.median(pre)), "n_samples": int(pre.size), "source": "pre_odor"}
    allv = _finite(np.concatenate(fallback)) if fallback else np.array([])
    if allv.size:
        return {"rest_px": float(np.median(allv)), "n_samples": int(allv.size), "source": "all"}
    return {"rest_px": None, "n_samples": 0, "source": "none"}


def _trim_resting(values: np.ndarray, trim_pct: float, *, high_end: bool) -> float:
    """The scale's RESTING endpoint, trimmed by ``trim_pct`` so one bad frame cannot set it.

    ``high_end`` picks which tail is the resting one: the abdomen angle rests near
    STRAIGHT (high degrees) while the proboscis rests RETRACTED (low pixels). The
    opposite tail is the signal and is never passed through here.
    """
    if values.size == 0:
        return float("nan")
    if trim_pct <= 0:
        return float(values.max() if high_end else values.min())
    return float(np.percentile(values, 100.0 - trim_pct if high_end else trim_pct))


def compute_slot_stats(slot: int, entries: Sequence[Tuple[TrialInfo, pd.DataFrame]], gcfg: GeometrySettings,
                       settings: Settings) -> dict:
    pro_all, ang_min, ang_max = [], (np.inf, None), (-np.inf, None)
    ang_all: List[np.ndarray] = []
    n_ang = 0
    len_pre, len_all, wid_pre, wid_all = [], [], [], []
    for info, df in entries:
        ok_pro = (df[C.PRO_STATUS] == C.PRO_OK).to_numpy()
        d = df[C.PRO_DIST].to_numpy(dtype=float)
        pro_all.append(d[ok_pro & np.isfinite(d) & (d >= gcfg.per_min_px)])
        ok_ang = (df[C.STATUS] == C.STATUS_OK).to_numpy() & np.isfinite(df[C.ABD_ANGLE].to_numpy(dtype=float))
        if ok_ang.any():
            ang = df[C.ABD_ANGLE].to_numpy(dtype=float)
            frames = df[C.FRAME].to_numpy()
            i_min = int(np.nanargmin(np.where(ok_ang, ang, np.inf)))
            i_max = int(np.nanargmax(np.where(ok_ang, ang, -np.inf)))
            if ang[i_min] < ang_min[0]:
                ang_min = (float(ang[i_min]), {"trial": info.label, "frame": int(frames[i_min])})
            if ang[i_max] > ang_max[0]:
                ang_max = (float(ang[i_max]), {"trial": info.label, "frame": int(frames[i_max])})
            ang_all.append(ang[ok_ang])
            n_ang += int(ok_ang.sum())
        has_abd = np.isfinite(df[C.ABD_LEN].to_numpy(dtype=float))
        pre = pre_odor_mask(df, info, settings) & has_abd
        len_pre.append(df[C.ABD_LEN].to_numpy(dtype=float)[pre])
        len_all.append(df[C.ABD_LEN].to_numpy(dtype=float)[has_abd])
        wid_pre.append(df[C.ABD_WID].to_numpy(dtype=float)[pre])
        wid_all.append(df[C.ABD_WID].to_numpy(dtype=float)[has_abd])
    pro = np.concatenate(pro_all) if pro_all else np.array([])
    # Resting endpoint trimmed, signal endpoint literal (see _trim_resting).
    gmin = _trim_resting(pro, gcfg.per_trim_pct, high_end=False) if pro.size else None
    gmax = float(pro.max()) if pro.size else None
    ang_pool = np.concatenate(ang_all) if ang_all else np.array([])
    ang_hi = _trim_resting(ang_pool, gcfg.angle_trim_pct, high_end=True) if ang_pool.size else None
    return {
        "slot": int(slot),
        "n_trials": len(entries),
        "proboscis": {"global_min": gmin, "global_max": gmax,
                      "effective_max": None if gmax is None else max(gmax, gcfg.per_effective_max_floor_px),
                      "raw_min": float(pro.min()) if pro.size else None,
                      "raw_max": float(pro.max()) if pro.size else None,
                      "trim_pct": gcfg.per_trim_pct, "trim_end": "low (resting)",
                      "n_frames": int(pro.size), "min_px": gcfg.per_min_px},
        "abdomen_angle": {"min_deg": None if ang_min[1] is None else ang_min[0],
                          "max_deg": ang_hi,
                          "raw_min": None if ang_min[1] is None else ang_min[0],
                          "raw_max": None if ang_max[1] is None else ang_max[0],
                          "trim_pct": gcfg.angle_trim_pct, "trim_end": "high (straight, resting)",
                          "min_at": ang_min[1], "max_at": ang_max[1], "n_frames": n_ang,
                          "min_range_deg": gcfg.angle_pct_min_range_deg},
        "abdomen_length": _rest(len_pre, len_all, gcfg.rest_min_samples),
        "abdomen_width": _rest(wid_pre, wid_all, gcfg.rest_min_samples),
    }


def apply_slot_stats(df: pd.DataFrame, stats: dict, gcfg: GeometrySettings) -> pd.DataFrame:
    out = df[C.PER_FRAME_COLUMNS].copy()
    p = stats["proboscis"]
    if p["global_min"] is not None:
        out[C.PRO_PCT] = M.proboscis_pct(out[C.PRO_DIST].to_numpy(dtype=float), p["global_min"], p["global_max"],
                                         gcfg.per_effective_max_floor_px)
    else:
        out[C.PRO_PCT] = np.nan
    a = stats["abdomen_angle"]
    if a["min_deg"] is not None and a["max_deg"] is not None:
        out[C.ABD_ANGLE_PCT] = M.angle_pct(out[C.ABD_ANGLE].to_numpy(dtype=float), a["min_deg"], a["max_deg"],
                                           gcfg.angle_pct_min_range_deg)
    else:
        out[C.ABD_ANGLE_PCT] = np.nan
    for col, src, key in ((C.ABD_LEN_PCT, C.ABD_LEN, "abdomen_length"), (C.ABD_WID_PCT, C.ABD_WID, "abdomen_width")):
        rest = stats[key]["rest_px"]
        out[col] = (100.0 * out[src].to_numpy(dtype=float) / rest) if rest else np.nan
    return out[C.PER_FRAME_COLUMNS + C.NORMALIZED_COLUMNS]


def normalize_batch(batch_dir: Path, settings: Settings, gcfg: GeometrySettings) -> dict:
    entries = load_trial_entries(batch_dir, gcfg)
    by_slot: Dict[int, List[Tuple[TrialInfo, pd.DataFrame, Path, Path]]] = {}
    for trial_dir, info, tables in entries:
        for slot, path in tables.items():
            by_slot.setdefault(slot, []).append((info, C.read_table_any(path), trial_dir, path))
    gdir = geometry_dir(batch_dir, gcfg)
    all_stats: Dict[int, dict] = {}
    for slot, items in sorted(by_slot.items()):
        stats = compute_slot_stats(slot, [(i, df) for i, df, _, _ in items], gcfg, settings)
        all_stats[slot] = stats
        gdir.mkdir(parents=True, exist_ok=True)
        C.stats_path(gdir, slot).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        for info, df, trial_dir, _ in items:
            out = apply_slot_stats(df, stats, gcfg)
            C.write_table_fmt(out, C.geometry_table_path(trial_dir, info.trial_name, slot, gcfg.table_format),
                              gcfg.table_format)
        print(f"[GEOMETRY] {Path(batch_dir).name} fly{slot}: PER {stats['proboscis']['global_min']}..{stats['proboscis']['global_max']} px, "
              f"angle {stats['abdomen_angle']['min_deg']}..{stats['abdomen_angle']['max_deg']} deg over {len(items)} trial(s)")
    return {"slots": sorted(all_stats), "n_trials": len(entries), "stats": all_stats}
