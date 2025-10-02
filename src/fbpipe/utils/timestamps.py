
from __future__ import annotations
import numpy as np
import pandas as pd

def timestamp_to_seconds(ts) -> float:
    if pd.isna(ts): return np.nan
    try:
        return float(ts)
    except Exception:
        s = str(ts).strip()
        parts = s.split(":")
        try:
            if len(parts)==4: hh,mm,ss,ms=parts; return int(hh)*3600+int(mm)*60+int(ss)+int(ms)/1000.0
            if len(parts)==3: hh,mm,ss=parts;   return int(hh)*3600+int(mm)*60+float(ss)
            if len(parts)==2: mm,ss=parts;      return int(mm)*60+float(ss)
            if len(parts)==1: return float(parts[0])
        except Exception:
            return np.nan
    return np.nan

TIMESTAMP_CANDIDATES = ["UTC_ISO", "Timestamp", "Number", "MonoNs", "timestamp", "time", "time_seconds", "relative_time"]
FRAME_CANDIDATES = ["Frame Number", "FrameNumber", "frame", "Frame"]

def pick_timestamp_column(df: pd.DataFrame):
    for c in TIMESTAMP_CANDIDATES:
        if c in df.columns: return c
    return None

def pick_frame_column(df: pd.DataFrame):
    for c in FRAME_CANDIDATES:
        if c in df.columns: return c
    return None

def to_seconds_series(df: pd.DataFrame, ts_col: str) -> pd.Series:
    s = df[ts_col]
    if ts_col in ("UTC_ISO", "Timestamp"):
        dt = pd.to_datetime(s, errors="coerce", utc=(ts_col == "UTC_ISO"))
        secs = dt.astype("int64") / 1e9
        t0 = np.nanmin(secs.values)
        return (secs - t0).astype(float)
    if ts_col == "Number":
        vals = pd.to_numeric(s, errors="coerce").astype(float)
        t0 = np.nanmin(vals.values)
        return vals - t0
    if ts_col == "MonoNs":
        vals = pd.to_numeric(s, errors="coerce").astype(float)
        secs = vals / 1e9
        t0 = np.nanmin(secs.values)
        return secs - t0
    # generic
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() >= max(3, int(0.8*len(vals))):
        base = vals.dropna().iloc[0]
        return vals - base
    # try parsing strings
    return s.apply(timestamp_to_seconds)
