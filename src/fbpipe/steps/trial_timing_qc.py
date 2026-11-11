"""Per-trial QC step: flag trials whose timing the pipeline would misread.

Reads each live trial's rig per-frame log (``output_<batch>_<cycle>_<odor>_<ts>.csv``:
``FrameNumber``, ``SensorNs`` camera clock, ``ActiveOFM`` valve state) and flags

* a **camera stall** -- any frame-to-frame gap over ``STALL_GAP_S`` on the camera
  clock. The sidecar's "achieved" fps then counts the stall too (36.27 fps for an
  8.4 s stall), the trial has fewer frames than its 90 s, and a stall inside the
  odor window leaves a hole in the response. The reason says how long, when, and
  whether it hit before, during or after the odor.
* a **camera clock that runs backwards** (a ``SensorNs`` reset): no time read
  from that log can be trusted, so its window is not compared.
* an **odor-window mismatch** -- the window the pipeline reads from the log
  (``trial_metadata.parse_active_ofm``) disagrees with the valve log itself
  (first valve-open frame -> the frame after the last) by more than
  ``WINDOW_TOLERANCE_S`` at either end. That is a valve pattern the parser does
  not understand, as the paraffin master valve's 50 ms clicks were on 2026-11-05.

Report-only -- nothing is excluded. Every trial is upserted into
``<report_dir>/trial_timing_qc.csv``; flagged trials are rewritten into
``trial_timing_qc_flagged.csv`` with a suggested fix, logged as
``[TRIAL-TIMING][FLAG]`` lines, and sent as one ntfy alert for trials *newly*
flagged (an unchanged trial never re-notifies). A log whose mtime is unchanged
since the last run is not re-read.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Settings, get_main_directories
from ..utils.frozen_folders import iter_live_batch_dirs
from ..utils.notify import ntfy_notify
from ..utils.trial_metadata import parse_active_ofm

DEFAULT_REPORT_DIR = Path(
    "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Trial-Timing-QC"
)
REPORT_FILENAME = "trial_timing_qc.csv"
FLAGGED_FILENAME = "trial_timing_qc_flagged.csv"
QC_VERSION = 1
#: A camera-clock gap longer than this between two frames is a stall (frames
#: are 25 ms apart at 40 fps; the paraffin valve clicks are 50 ms).
STALL_GAP_S = 0.5
#: How far the parsed odor window may sit from the valve log at either end.
WINDOW_TOLERANCE_S = 1.0
NOMINAL_FPS = 40.0
NTFY_MAX_LINES = 20

_CYCLE_RE = re.compile(r"((?:pretest|training|testing)_\d+_[^_]+)", re.IGNORECASE)

REPORT_COLUMNS = (
    "dataset", "batch", "trial", "frames_csv", "csv_mtime", "qc_version", "status",
    "n_frames", "clock_s", "frames_short", "max_gap_s", "stall_at_s", "stall_where",
    "valve_on_s", "valve_off_s", "pipeline_on_s", "pipeline_off_s",
    "flag", "reason", "suggested_fix",
)

FIX_STALL_DURING = ("The odor window has a hole: watch the video around the stall. If the "
                    "response is lost, exclude the trial (flagged-flys-truth.csv); otherwise keep it.")
FIX_STALL_OUTSIDE = ("The odor window is intact (the pipeline keeps the valve's own frames); the "
                     "trace after the stall is compressed in time. Usually keep; check the score.")
FIX_CLOCK = ("The camera timestamps in this log are unusable: check the video's length; no time "
             "read from this log can be trusted.")
FIX_WINDOW = ("A valve pattern fbpipe.utils.trial_metadata.parse_active_ofm does not understand: "
              "teach it the pattern, bump _CACHE_SCHEMA_VERSION there, and re-run the pipeline.")


def _trial_name(path: Path) -> str:
    m = _CYCLE_RE.search(path.name)
    return m.group(1) if m else path.stem


def _blank(path: Path) -> dict:
    row = {c: None for c in REPORT_COLUMNS}
    row.update(frames_csv=str(path), trial=_trial_name(path), qc_version=QC_VERSION,
               flag=False, reason="", suggested_fix="")
    try:
        row["csv_mtime"] = path.stat().st_mtime
    except OSError:
        pass
    return row


def assess_frames_csv(path: Path | str) -> dict:
    """One report row for one rig per-frame log."""
    path = Path(path)
    row = _blank(path)
    try:
        df = pd.read_csv(path, usecols=lambda c: c in {"FrameNumber", "SensorNs", "ActiveOFM"})
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        row.update(status="read_error", reason=f"{type(exc).__name__}: {exc}")
        return row
    if not {"FrameNumber", "SensorNs", "ActiveOFM"} <= set(df.columns) or len(df) < 2:
        row["status"] = "no_frame_log"
        return row

    t = (df["SensorNs"].to_numpy(dtype=float) - float(df["SensorNs"].iloc[0])) / 1e9
    gaps = np.diff(t)
    gi = int(np.argmax(gaps))
    row.update(n_frames=len(df), clock_s=round(float(t[-1]), 3),
               frames_short=int(round(t[-1] * NOMINAL_FPS)) + 1 - len(df),
               max_gap_s=round(float(gaps[gi]), 3), stall_at_s=round(float(t[gi]), 3))

    states = df["ActiveOFM"].astype(str).str.strip().str.lower()
    on = (~states.isin({"off", "", "nan", "none"})).to_numpy()
    reasons, fixes = [], []
    backwards = np.flatnonzero(gaps < 0)
    if backwards.size:
        b = int(backwards[0])
        reasons.append(f"camera clock went backwards {abs(gaps[b]):.1f} s at frame "
                       f"{int(df['FrameNumber'].iloc[b + 1])}")
        fixes.append(FIX_CLOCK)
    if not on.any():
        row.update(status="no_odor", stall_where="no odor")
        valve = None
    else:
        row["status"] = "assessed"
        first = int(np.argmax(on))
        last = len(on) - 1 - int(np.argmax(on[::-1]))
        valve = (float(t[first]), float(t[min(last + 1, len(t) - 1)]))
        row.update(valve_on_s=round(valve[0], 3), valve_off_s=round(valve[1], 3))
        frame_t = dict(zip(df["FrameNumber"].astype(int), t))
        on_f, off_f, _ = parse_active_ofm(path)
        p_on = frame_t.get(int(on_f)) if on_f is not None else None
        p_off = frame_t.get(int(off_f)) if off_f is not None else float(t[-1])
        row.update(pipeline_on_s=None if p_on is None else round(float(p_on), 3),
                   pipeline_off_s=None if p_off is None else round(float(p_off), 3))
        if backwards.size:
            pass  # the clock itself is broken; the reason above says so
        elif (p_on is None or p_off is None
                or abs(p_on - valve[0]) > WINDOW_TOLERANCE_S
                or abs(p_off - valve[1]) > WINDOW_TOLERANCE_S):
            shown = "none" if p_on is None else f"{p_on:.2f}-{p_off:.2f} s"
            reasons.append(f"pipeline odor window {shown} != valve log "
                           f"{valve[0]:.2f}-{valve[1]:.2f} s")
            fixes.append(FIX_WINDOW)

    if gaps[gi] > STALL_GAP_S:
        start, end = float(t[gi]), float(t[gi + 1])
        if valve is None:
            where = "no odor"
        elif end <= valve[0]:
            where = "before odor"
        elif start >= valve[1]:
            where = "after odor"
        else:
            where = "during odor"
        row["stall_where"] = where
        n_stalls = int((gaps > STALL_GAP_S).sum())
        more = f" (+{n_stalls - 1} more)" if n_stalls > 1 else ""
        reasons.append(f"camera stalled {gaps[gi]:.1f} s at {start:.1f} s ({where}){more}; "
                       f"{row['frames_short']} frames short")
        fixes.append(FIX_STALL_DURING if where == "during odor" else FIX_STALL_OUTSIDE)
    elif valve is not None:
        row["stall_where"] = ""

    if reasons:
        row.update(flag=True, reason="; ".join(reasons), suggested_fix=" ".join(fixes))
    return row


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load(path: Path) -> pd.DataFrame:
    try:
        old = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=list(REPORT_COLUMNS))
    for c in REPORT_COLUMNS:
        if c not in old.columns:
            old[c] = None
    return old[list(REPORT_COLUMNS)]


def _notify(newly: list[dict], flagged_path: Path) -> None:
    lines = [f"- {r['dataset']}/{r['batch']}/{r['trial']}: {r['reason']}"
             for r in newly[:NTFY_MAX_LINES]]
    extra = len(newly) - NTFY_MAX_LINES
    more = f"\n...and {extra} more" if extra > 0 else ""
    ntfy_notify(
        f"Trial timing QC: {len(newly)} trial(s) flagged",
        "Camera stall or an odor window the pipeline misreads:\n" + "\n".join(lines) + more
        + f"\n\nSuggested fixes: {flagged_path}",
        priority="high",
        tags="stopwatch,warning",
    )


def run(cfg: Settings, *, report_dir: Optional[Path] = None, notify: bool = True) -> pd.DataFrame:
    """Assess every live trial under the configured roots; return the new report."""
    report_dir = Path(report_dir) if report_dir is not None else DEFAULT_REPORT_DIR
    report_path = report_dir / REPORT_FILENAME
    flagged_path = report_dir / FLAGGED_FILENAME
    old = _load(report_path)
    prior = {str(r["frames_csv"]): r for r in old.to_dict("records")}
    prior_flagged = {k for k, r in prior.items()
                     if str(r.get("flag")).strip().lower() in {"true", "1"}}

    rows: list[dict] = []
    for root in get_main_directories(cfg):
        root = Path(root)
        if not root.is_dir():
            continue
        for batch in iter_live_batch_dirs(cfg, root):
            for csv_path in sorted(Path(batch).glob("output_*.csv")):
                key = str(csv_path)
                was = prior.get(key)
                try:
                    mtime = csv_path.stat().st_mtime
                except OSError:
                    continue
                if (was is not None and was.get("qc_version") == QC_VERSION
                        and was.get("csv_mtime") is not None
                        and abs(float(was["csv_mtime"]) - mtime) < 1e-6):
                    rows.append(dict(was, flag=str(was["flag"]).strip().lower() in {"true", "1"}))
                    continue
                row = assess_frames_csv(csv_path)
                row.update(dataset=root.name, batch=Path(batch).name)
                rows.append(row)

    new = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    keep_old = old[~old["frames_csv"].astype(str).isin(set(new["frames_csv"].astype(str)))]
    keep_old = keep_old[keep_old["frames_csv"].astype(str).map(lambda p: Path(p).exists())]
    report = pd.concat([keep_old, new], ignore_index=True) if not keep_old.empty else new
    report = report.sort_values(["dataset", "batch", "trial"], kind="stable").reset_index(drop=True)

    flagged = report[_truthy(report["flag"])]
    newly = [r for r in rows if r["flag"] and str(r["frames_csv"]) not in prior_flagged]
    for r in newly:
        print(f"[TRIAL-TIMING][FLAG] {r['dataset']}/{r['batch']}/{r['trial']}: {r['reason']}")
    _write_csv_atomic(report, report_path)
    _write_csv_atomic(flagged, flagged_path)
    n_stall = int(flagged["reason"].astype(str).str.contains("camera stalled").sum())
    n_window = int(flagged["reason"].astype(str).str.contains("odor window").sum())
    print(f"[TRIAL-TIMING] {len(new)} trials checked; {len(flagged)} flagged in the report "
          f"({n_stall} camera stall, {n_window} odor-window mismatch), {len(newly)} new "
          f"-> {flagged_path}")
    if newly and notify:
        _notify(newly, flagged_path)
    return report


def main(cfg: Settings) -> None:
    """Pipeline entry point. Never raises: a QC failure must not stop the run."""
    try:
        run(cfg)
    except Exception as exc:
        print(f"[TRIAL-TIMING] ERROR: trial timing QC failed ({type(exc).__name__}: {exc}); "
              "the pipeline continues without it.")
