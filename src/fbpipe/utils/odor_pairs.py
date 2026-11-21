"""IAA odor-pair trials: find each odor's valve window and score each odor alone.

A pair trial (``testing_<N>_<odor1>_then_<odor2>``, PiCode ``config_v2_pair*_A.yaml``,
2026-11-13) delivers odor 1 for 10 s, nothing for 1 s, then odor 2 for 10 s, in one
recording. The trial's stored odor window (``trial_metadata.parse_active_ofm``) is
odor 1 only.

User rule (2026-11-13): each odor gets its own model score, each against the
trial's pre-odor block, never the two odors combined. :func:`split_pair_rows`
turns every pair row of a wide table into two rows before scoring:

* **odor 1** -- the label unchanged and the stored window; the trace is cut (NaN,
  and ``trial_duration_s`` shortened) where odor 2 reaches the fly, i.e. odor 2's
  valve onset + :data:`ODOR_ARRIVAL_DELAY_S`, so odor 2 never enters its post window;
* **odor 2** -- ``<label>__odor2``; the trace is the pre-odor block ``[0, odor-1 onset)``
  followed by everything from odor 2's valve onset, timed like a normal trial
  (on = odor-1 onset, off = on + odor-2 length).

Caveat: odor 2 starts 1 s after odor 1 ends, so a fly still extended from odor 1
enters odor 2's window already extended. The pre-odor block baseline is the
user's choice, not a correction for that.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

#: The scorer's odor-arrival delay (FlyBehaviorScoring model attribute
#: ``fbr_odor_delay_s`` = 1.0 for the ordinal v2d1 models).
ODOR_ARRIVAL_DELAY_S = 1.0
ODOR2_SUFFIX = "__odor2"
_PAIR_RE = re.compile(r"_then_", re.IGNORECASE)
_CYCLE_RE = re.compile(r"^((?:pretest|training|testing)_\d+)_", re.IGNORECASE)


def is_pair_label(label: object) -> bool:
    s = str(label)
    return bool(_PAIR_RE.search(s)) and not s.endswith(ODOR2_SUFFIX)


def _runs(on: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous valve-open runs as 0-based ``[start, end)`` index pairs."""
    edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def pair_deliveries(states: pd.Series, on: np.ndarray) -> list[tuple[int, int]]:
    """Odor deliveries of a pair-trial log as ``[start, end)`` row-index pairs, in order.

    ``states`` is the lower-cased ``ActiveOFM`` column, ``on`` its valve-open mask.
    Paraffin is the master valve alone, logged as two ~2-frame ``OFM_P`` clicks
    (onset and offset): those two runs are one delivery. Any other channel is one
    contiguous run."""
    runs, out, i = _runs(on), [], 0
    while i < len(runs):
        a, b = runs[i]
        if (states.iloc[a] == "ofm_p" and i + 1 < len(runs)
                and states.iloc[runs[i + 1][0]] == "ofm_p"):
            out.append((a, runs[i + 1][1]))
            i += 2
        else:
            out.append((a, b))
            i += 1
    return out


def second_odor_window_s(frames_csv: Path | str, fps: float) -> Optional[tuple[float, float]]:
    """Odor 2's valve window in seconds, on the same ``FrameNumber / fps`` clock
    as ``trial_odor_on_s``; None when the log has no second delivery."""
    try:
        df = pd.read_csv(frames_csv, usecols=lambda c: c in {"FrameNumber", "ActiveOFM"})
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    if not {"FrameNumber", "ActiveOFM"} <= set(df.columns) or df.empty:
        return None
    states = df["ActiveOFM"].astype(str).str.strip().str.lower()
    on = (~states.isin({"off", "", "nan", "none"})).to_numpy()
    odors = pair_deliveries(states, on)
    if len(odors) < 2:
        return None
    a2, b2 = odors[1]
    frames = df["FrameNumber"].to_numpy()
    end_frame = frames[b2] if b2 < len(frames) else frames[-1] + 1
    return float(frames[a2]) / fps, float(end_frame) / fps


def find_frames_csv(row: pd.Series, roots: Iterable[Path]) -> Optional[Path]:
    """The rig per-frame log ``output_<batch>_<cycle>_<odors>_<ts>.csv`` of a wide row."""
    m = _CYCLE_RE.match(str(row["trial_label"]))
    if not m:
        return None
    cycle = m.group(1).lower()
    dataset, fly = str(row["dataset"]), str(row["fly"])
    pat = re.compile(rf"^output_.+_{re.escape(cycle)}_.+_then_.+_\d{{8}}_\d{{6}}\.csv$",
                     re.IGNORECASE)
    for root in roots:
        root = Path(root)
        batch = root / fly if root.name == dataset else root / dataset / fly
        if not batch.is_dir():
            continue
        hits = sorted(p for p in batch.glob("output_*.csv") if pat.match(p.name))
        if hits:
            return hits[0]
    return None


def _dir_cols(df: pd.DataFrame) -> list[str]:
    return sorted((c for c in df.columns if c.startswith("dir_val_")), key=lambda c: int(c[8:]))


def split_pair_rows(
    df: pd.DataFrame,
    odor2_window: Callable[[pd.Series], Optional[tuple[float, float]]],
    *,
    delay_s: float = ODOR_ARRIVAL_DELAY_S,
) -> tuple[pd.DataFrame, dict]:
    """Every pair row → an odor-1 row and an odor-2 row (see module docstring).

    ``odor2_window(row)`` returns odor 2's ``(on_s, off_s)`` or None; a pair row
    without one keeps its odor-1 row unchanged (reported in ``no_second_odor``)."""
    report = {"pairs": 0, "split": 0, "no_second_odor": []}
    if "trial_label" not in df.columns:
        return df, report
    mask = df["trial_label"].map(is_pair_label)
    report["pairs"] = int(mask.sum())
    if not mask.any():
        return df, report

    cols = _dir_cols(df)
    col_idx = {c: df.columns.get_loc(c) for c in cols}
    out_rows: list[pd.Series] = []
    for _, row in df.iterrows():
        if not is_pair_label(row["trial_label"]):
            out_rows.append(row)
            continue
        win = odor2_window(row)
        if win is None:
            report["no_second_odor"].append(
                (row["dataset"], row["fly"], row["fly_number"], row["trial_label"]))
            out_rows.append(row)
            continue
        fps = float(row.get("fps") or 40.0)
        trace = row[cols].to_numpy(dtype=float)
        dur = row.get("trial_duration_s")
        n_valid = len(trace)
        if pd.notna(dur) and float(dur) > 0:
            n_valid = min(n_valid, int(round(float(dur) * fps)))
        on1 = float(row["trial_odor_on_s"])
        on2, off2 = win
        on1_i = int(round(on1 * fps))
        on2_i = int(round(on2 * fps))

        cut = min(int(round((on2 + delay_s) * fps)), n_valid)
        t1 = trace.copy()
        t1[cut:] = np.nan
        r1 = row.copy()
        r1[cols] = t1
        r1["trial_duration_s"] = cut / fps

        tail = trace[on2_i:n_valid]
        n2 = on1_i + len(tail)
        t2 = np.full(len(trace), np.nan)
        t2[:on1_i] = trace[:on1_i]
        t2[on1_i:n2] = tail
        r2 = row.copy()
        r2[cols] = t2
        r2["trial_label"] = f"{row['trial_label']}{ODOR2_SUFFIX}"
        r2["trial_odor_on_s"] = on1_i / fps
        r2["trial_odor_off_s"] = (on1_i + int(round((off2 - on2) * fps))) / fps
        r2["trial_duration_s"] = n2 / fps
        out_rows += [r1, r2]
        report["split"] += 1
    out = pd.DataFrame(out_rows, columns=df.columns).reset_index(drop=True)
    return out.astype(df.dtypes.to_dict(), errors="ignore"), report
