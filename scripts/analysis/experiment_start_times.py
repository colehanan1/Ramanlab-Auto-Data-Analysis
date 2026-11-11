#!/usr/bin/env python3
"""When did each experiment run? One row per batch folder, plus one per batch.

The start time is the first pre-test trial, read from the raw trial file name
(``output_<batch>_pretest_<n>_<odor>_YYYYMMDD_HHMMSS.csv``, Pi local time).
The starvation start comes from the folder's ``session_metadata.txt``, so the
table also gives the hours a fly had actually been starved when it was tested.

Use it to pick batches that ran at about the same time of day, then compare
them with ``day_vs_day_report.py --by batch --units ...``.

Usage
-----
    python3 scripts/analysis/experiment_start_times.py --dataset IAA-Sensitivity-1-1
    python3 scripts/analysis/experiment_start_times.py --dataset IAA-Sensitivity-1-1 --dataset IAA-Sensitivity-12-1
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.analysis.per_fly_score_matrix import folder_key  # noqa: E402
from scripts.analysis.time_of_day_trend import first_pretest_time  # noqa: E402

DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys_New")
_STARVED_RE = re.compile(r"Starved Datetime \(ISO, local naive\):\s*(\S+)")
_TARGET_RE = re.compile(r"Target starvation duration \(hours\):\s*([\d.]+)")
_FOLDER_RE = re.compile(r"^[a-z]+_\d{2}_batch_\d+(_rig_\d+)?$")


def _metadata(folder: Path) -> str:
    p = Path(folder) / "session_metadata.txt"
    return p.read_text(errors="replace") if p.exists() else ""


def starved_at(folder: Path) -> dt.datetime | None:
    m = _STARVED_RE.search(_metadata(folder))
    try:
        return dt.datetime.fromisoformat(m.group(1)) if m else None
    except ValueError:
        return None


def target_hours(folder: Path) -> float | None:
    m = _TARGET_RE.search(_metadata(folder))
    return float(m.group(1)) if m else None


def start_table(data_root: Path, dataset: str) -> pd.DataFrame:
    """One row per batch folder: when it started, rig, starvation target and actual."""
    rows = []
    for f in sorted(Path(data_root, dataset).iterdir()):
        if not (f.is_dir() and _FOLDER_RE.match(f.name)):
            continue
        start = first_pretest_time(f)
        if start is None:
            continue
        day, batch, rig = folder_key(f.name)
        starved = starved_at(f)
        rows.append({
            "dataset": dataset, "folder": f.name, "date": day,
            "batch": f"{f.name.split('_rig_')[0]}", "batch_number": batch, "rig": rig,
            "first_pretest": start, "start_clock": start.strftime("%H:%M"),
            "starved_at": starved, "target_h": target_hours(f),
            "hours_starved_at_pretest": (start - starved).total_seconds() / 3600
            if starved else float("nan"),
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.sort_values(["date", "batch_number", "rig"]).reset_index(drop=True)


def batch_summary(t: pd.DataFrame) -> pd.DataFrame:
    """One row per batch: the earliest and latest rig start, and how many rigs ran."""
    g = t.groupby(["dataset", "batch"], sort=False)
    out = g.agg(date=("date", "first"), earliest=("first_pretest", "min"),
                latest=("first_pretest", "max"), n_rigs=("rig", "nunique"),
                hours_starved=("hours_starved_at_pretest", "median")).reset_index()
    out["earliest"] = out["earliest"].dt.strftime("%H:%M")
    out["latest"] = out["latest"].dt.strftime("%H:%M")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", action="append", required=True, help="repeatable")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--csv", type=Path, default=None, help="also write the per-folder table here")
    args = ap.parse_args(argv)

    tables = [start_table(args.data_root, d) for d in args.dataset]
    t = pd.concat([x for x in tables if not x.empty], ignore_index=True) if any(
        not x.empty for x in tables) else pd.DataFrame()
    if t.empty:
        raise SystemExit("no batch folders with a pre-test file found")
    show = t.assign(starved=t["starved_at"].map(lambda v: v.strftime("%m-%d %H:%M") if pd.notna(v) else "?"),
                    h_starved=t["hours_starved_at_pretest"].round(1))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print("Per rig folder (first pre-test = when the experiment started):\n")
        print(show[["dataset", "folder", "rig", "date", "start_clock", "starved", "target_h",
                    "h_starved"]].to_string(index=False))
        print("\nPer batch (all rigs of one session):\n")
        b = batch_summary(t)
        b["hours_starved"] = b["hours_starved"].round(1)
        print(b.to_string(index=False))
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"\n[OK] {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
