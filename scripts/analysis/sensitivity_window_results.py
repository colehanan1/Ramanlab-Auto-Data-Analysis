#!/usr/bin/env python3
"""One results folder for the sensitivity cohorts, flies started inside a clock window.

Run by the pipeline every time (``analysis.sensitivity_window_results`` in
config_new.yaml); a cohort's folders are redrawn only when its rows, the analysis
code, the config or the scoring model changed (or the folders did), the summaries
across cohorts always. First built by hand 2026-11-04 as
``Results/Sensitiviy-Results(11-15)`` (start 10:50-15:10, widened to 10:40-15:20 on 2026-11-05,
10:00-16:00 on 2026-11-06 and 09:40-16:20 on 2026-11-09). For each cohort:

    <cohort>/                   start_window_report.py: the windowed figure set,
                                FINDINGS.md, _inputs/ and _build_log.txt
    <cohort>/Per-Fly/           symlinks to the pipeline's per-fly figures for
                                exactly the flies in the window
    00_Primary-Endpoint/        (--control-cohort) the pre-specified primary endpoint
                                (``PRIMARY_ENDPOINT``): each exposed cohort's change on its
                                exposed odor minus the control flies', whole batches
                                shuffled; everything else here is exploratory
    Time-of-Day_<h>h-bins_whole-dataset/<cohort>/
                                time_of_day_trend.py over every fly of the cohort
    Cross-Cohort/               every odor of every cohort on one axis: change in %
                                responding and in mean score, and the pre- and
                                post-exposure levels themselves, window flies only
    Pre-Exposure-Baseline/      the naive pre-exposure response by cohort, by rig and
                                cohort x rig, window flies only (should all match)
    Before-vs-From-<date>/      (--split-date) pre-exposure baseline of window flies
                                started before vs from a date: all flies, plus each
                                --split-cohort
    Mean-Traces-By-Cohort/      every cohort's mean pre- and post-exposure trace per odor
                                on one axis, and whether the cohorts differ during the odor
    Trial-By-Trial/             (--trial-series-cohort) the sensitized odor at pre-exposure,
                                each exposure trial and post-exposure, same flies in
                                every bar, with every pair of trials tested
    00_Highlights/              symlinks to the headline figures
    Data/                       clean pre/post score + % responding tables (CSV
                                and one workbook), cohort x odor, fly x odor,
                                whole cohort by start bin
    README.md                   generated summary

A cohort with no folder on disk yet (a new cohort before its first batch) or no
batch started in the window is listed in the README and skipped. Only the
entries above are written; anything else in ``--out-dir`` is left alone.

Pre/post pairs come from the same code the score figures use
(``pretest_vs_test_comparison.load_paired_scores`` / ``per_odor_summary``), and
the windowed rows are checked fly by fly against the figure build, so the data
files and the figures cannot disagree.

Usage
-----
    python3 scripts/analysis/sensitivity_window_results.py \\
        --out-dir "/home/ramanlab/Documents/cole/Results/Sensitiviy-Results(11-15)" \\
        --start 10:40 --end 15:20 \\
        --dataset IAA-Sensitivity-1-1 --dataset Hex-Sensitivity-1-0.1
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2, friedmanchisquare
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scripts.analysis.pretest_vs_test_comparison as pvt  # noqa: E402
from scripts.analysis.phase_terms import EXPOSURE, POST_EXPOSURE, PRE_EXPOSURE  # noqa: E402
from scripts.analysis.randompanel_conc_comparison import holm_adjust  # noqa: E402
import scripts.analysis.start_window_report as swr  # noqa: E402
from scripts.analysis.day_vs_day_report import trained_odor  # noqa: E402
from scripts.analysis.experiment_start_times import DATA_ROOT, start_table  # noqa: E402
from fbpipe.analysis import fly_stats as fs  # noqa: E402
from fbpipe.utils import figure_cache as fc  # noqa: E402

FIGURES_ROOT = Path("/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures")
PY = sys.executable
THRESHOLD = pvt.BINARY_THRESHOLD

#: The pre-specified primary endpoint (user 2026-11-18): the one test this folder is read
#: for; every other figure in it is exploratory. Pinned by a test: a change to any of it
#: is a new registration with a new date, not an edit. The control cohort is
#: ``--control-cohort`` (config: ``control_cohort``).
PRIMARY_ENDPOINT = {
    "registered": "2026-11-18",
    "metric": "responded (score >= 2)",
    "window": "production scorer window (whole odor; model_predictions.csv)",
    "odor": "each exposed cohort's sensitized odor",
    "contrast": "exposed flies' post - pre change minus control flies' change on the same odor",
    "test": "two-sided permutation, whole batches (batch_folder) swapped between the two "
            "cohorts",
    "ci": "95 % percentile bootstrap over batches within each cohort",
}
PRIMARY_DIR = "00_Primary-Endpoint"
PRIMARY_PNG = "primary_endpoint.png"
PRIMARY_CSV = "primary_endpoint.csv"
#: ``fly_stats.controlled_change_test`` settings of the primary endpoint (fixed seed: a
#: rebuild of the same flies gives the same numbers).
PRIMARY_SEED, PRIMARY_N_PERM, PRIMARY_N_BOOT = 0, 20000, 10000

#: Exploratory multiplicity on every pre-vs-post figure set asked for here (2026-11-18):
#: Holm across the odors of a cohort.
P_ADJUST = ("--p-adjust", "holm")

OK = "built"
NO_DATA = "no data on disk yet"
NONE_IN_WINDOW = "no batch started in the window"
ALL_FLAGGED = "every fly in the window is flagged"
NOT_SCORED = "on disk, not scored yet (the next pipeline run scores it)"

#: pipeline figure folder -> Per-Fly/ sub-folder; files are <folder>_fly<N>_*
PER_FLY = {
    "Raw-Pre-Testing-PER-Traces": "1_Pre-exposure-Raw-Traces",
    "Raw-Training-PER-Traces": "2_Exposure-Raw-Traces",
    "Raw-Testing-PER-Traces": "3_Post-exposure-Raw-Traces",
}
ABDOMEN = ("Abdomen-Geometry-New", "4_Abdomen-Geometry")


# ---------------------------------------------------------------------------
# Which cohorts can be built
# ---------------------------------------------------------------------------


def dataset_status(data_root: Path, dataset: str, lo: dt.time,
                   hi: dt.time) -> tuple[str, list[str]]:
    """``(status, in-window folders)`` for one cohort."""
    if not (Path(data_root) / dataset).is_dir():
        return NO_DATA, []
    starts = start_table(Path(data_root), dataset)
    if starts.empty:
        return NO_DATA, []
    window = swr.folders_in_window(starts, lo, hi)
    if window.empty:
        return NONE_IN_WINDOW, []
    return OK, window["folder"].tolist()


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------


def scoring_status(predictions: pd.DataFrame, dataset: str, folders: Sequence[str]) -> str:
    """NOT_SCORED until the pipeline has scored at least one window folder."""
    hit = (predictions["dataset"] == dataset) & predictions["fly"].isin(folders)
    return OK if hit.any() else NOT_SCORED


def _swap_in(staging: Path, final: Path) -> None:
    """Replace ``final`` (a folder this script owns) with ``staging``."""
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)


def _staging(final: Path) -> Path:
    path = final.parent / f".building-{final.name}"
    if path.exists():
        shutil.rmtree(path)
    return path


BUILD_LOCK_TIMEOUT_S = 45 * 60


def acquire_build_lock(out: Path, timeout_s: float = BUILD_LOCK_TIMEOUT_S):
    """One build of this folder at a time; None if another keeps it past ``timeout_s``.

    Two builds share the staging folders and would delete each other's work. A pipeline
    run reaching this step during a hand-started build waits here, then finds nothing
    left to redraw. Hold the returned file open: the kernel drops the lock when it closes,
    however the process ends."""
    handle = open(out / ".build.lock", "a")
    deadline = time.monotonic() + timeout_s
    announced = False
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError:
            if time.monotonic() >= deadline:
                handle.close()
                return None
            if not announced:
                print("[sens-window] another build of this folder is running; waiting for it",
                      flush=True)
                announced = True
            time.sleep(0.5)


def replace_links(dst: Path, sources: Iterable[Path]) -> None:
    """Make ``dst`` hold one symlink per source. Only symlinks are removed, so a
    file someone dropped in by hand survives."""
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.iterdir():
        if old.is_symlink():
            old.unlink()
    for src in sources:
        (dst / src.name).symlink_to(src)


def per_fly_sources(figures_root: Path, dataset: str,
                    flies: Sequence[tuple[str, int]]) -> dict[str, list[Path]]:
    """Per-fly pipeline figures for exactly these (folder, fly) pairs.

    Matched on ``<folder>_fly<N>_``: the ``_fly`` right after the folder name is
    what keeps ``october_14_batch_1`` from pulling in ``october_14_batch_1_rig_2``.
    """
    out: dict[str, list[Path]] = {}
    for src_dir, name in PER_FLY.items():
        d = Path(figures_root) / src_dir / dataset
        out[name] = [h for folder, n in flies for h in sorted(d.glob(f"{folder}_fly{n}_*"))]
    abd_dir, name = ABDOMEN
    d = Path(figures_root) / abd_dir / dataset
    out[name] = [h for folder, n in flies
                 for h in sorted((d / folder).glob(f"{folder}_fly{n}_*.png"))]
    return {k: v for k, v in out.items() if v}


def kept_flies(inputs: Path) -> list[tuple[str, int]]:
    flies: set[tuple[str, int]] = set()
    for name in ("wide_pretest.parquet", "wide_testing.parquet", "wide_training.parquet"):
        path = inputs / name
        if path.exists():
            df = pd.read_parquet(path, columns=["fly", "fly_number"]).drop_duplicates()
            flies |= {(str(f), int(n)) for f, n in df.itertuples(index=False)}
    return sorted(flies)


# ---------------------------------------------------------------------------
# Data tables
# ---------------------------------------------------------------------------


def bin_label(t: dt.time, hours: float) -> str:
    width = int(round(hours * 60))
    lo = (t.hour * 60 + t.minute) // width * width
    hi = lo + width
    return f"{lo // 60:02d}:{lo % 60:02d}-{hi // 60:02d}:{hi % 60:02d}"


def tidy_pairs(paired: pd.DataFrame, dataset: str, starts: pd.DataFrame, sensitized: str,
               lo: dt.time, hi: dt.time, *, bin_hours: float) -> pd.DataFrame:
    """One row per fly x odor from ``load_paired_scores`` output."""
    s = starts.set_index("folder")
    start = pd.to_datetime(paired["fly"].map(s["first_pretest"]))
    clock = start.dt.time
    out = pd.DataFrame({
        "cohort": dataset,
        "batch_folder": paired["fly"].astype(str),
        "fly_number": paired["fly_number"].astype(int),
        "rig": paired["fly"].map(s["rig"]),
        "start_time": start.dt.strftime("%Y-%m-%d %H:%M"),
        "start_bin": [bin_label(c, bin_hours) for c in clock],
        "in_window": [lo <= c < hi for c in clock],
        "hours_starved_at_pretest": paired["fly"].map(s["hours_starved_at_pretest"]).round(1),
        "odor": paired["odor"].astype(str),
        "score_pre": paired["score_pre"].astype(int),
        "score_post": paired["score_post"].astype(int),
    })
    out.insert(3, "fly_id", out["batch_folder"] + "_fly" + out["fly_number"].astype(str))
    out.insert(out.columns.get_loc("odor") + 1, "sensitized_odor", out["odor"] == sensitized)
    out["score_change"] = out["score_post"] - out["score_pre"]
    out["responded_pre"] = out["score_pre"] >= THRESHOLD
    out["responded_post"] = out["score_post"] >= THRESHOLD
    return out.sort_values(["start_time", "batch_folder", "fly_number", "odor"],
                           ignore_index=True)


def summarise(long: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: flies pooled (one trial per fly per odor), never a mean of means.

    Pre and post are the same flies, so the rate test is the exact paired McNemar
    (``p_rate_mcnemar_paired``); there is no unpaired (Fisher) column."""
    rows = []
    for key, g in long.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        s = pvt.per_odor_summary(g[["odor", "score_pre", "score_post"]],
                                 binary_threshold=THRESHOLD, rate_test="mcnemar").iloc[0]
        rows.append({
            **dict(zip(by, key)),
            "sensitized_odor": bool(g["sensitized_odor"].iloc[0]),
            "n_flies": int(s["n_flies"]),
            "mean_score_pre": s["mean_pre"], "sem_score_pre": s["sem_pre"],
            "mean_score_post": s["mean_post"], "sem_score_post": s["sem_post"],
            "mean_score_change": float(g["score_change"].mean()),
            "n_responding_pre": int(g["responded_pre"].sum()),
            "n_responding_post": int(g["responded_post"].sum()),
            "pct_responding_pre": s["pct_pre"], "pct_responding_post": s["pct_post"],
            "pct_change": s["pct_post"] - s["pct_pre"],
            "p_score_wilcoxon": s["p_score"],
            "p_rate_mcnemar_paired": pvt.mcnemar_p(g["responded_pre"], g["responded_post"]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        num = df.select_dtypes("number").columns
        df[num] = df[num].astype(float).round(4)
        for col in ("n_flies", "n_responding_pre", "n_responding_post"):
            df[col] = df[col].astype(int)
    return df


def wide_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per fly: ``<Odor> pre`` / ``<Odor> post`` for every odor."""
    w = long.pivot_table(index=["cohort", "fly_id", "start_time", "in_window"],
                         columns="odor", values=["score_pre", "score_post"], aggfunc="first")
    w.columns = [f"{odor} {'pre' if m == 'score_pre' else 'post'}" for m, odor in w.columns]
    odors = sorted(long["odor"].unique())
    w = w[[f"{o} {ph}" for o in odors for ph in ("pre", "post") if f"{o} {ph}" in w.columns]]
    return w.reset_index().sort_values(["start_time", "fly_id"], ignore_index=True)


def cohort_pairs(dataset: str, predictions: pd.DataFrame, data_root: Path, lo: dt.time,
                 hi: dt.time, *, flagged: Path, bin_hours: float, scratch: Path) -> pd.DataFrame:
    """Every fly of the cohort (any start time), paired pre/post per odor."""
    starts = start_table(Path(data_root), dataset)
    rows = swr.subset_rows(predictions, dataset, starts["folder"].tolist(), flagged)
    tmp = scratch / f"_{dataset}_predictions.csv"
    rows.to_csv(tmp, index=False)
    try:
        paired = pvt.load_paired_scores(tmp, flagged_flies_csv=str(flagged))
    finally:
        tmp.unlink(missing_ok=True)
    paired = paired[paired["dataset_canon"] == dataset]
    return tidy_pairs(paired, dataset, starts, trained_odor(dataset), lo, hi,
                      bin_hours=bin_hours)


def check_matches_figures(long: pd.DataFrame, built_predictions: Path, flagged: Path) -> None:
    """The window rows must be exactly the pairs the figure build drew."""
    built = pvt.load_paired_scores(built_predictions, flagged_flies_csv=str(flagged))
    win = long[long["in_window"]]
    key = ["fly", "fly_number", "odor"]
    a = (win.rename(columns={"batch_folder": "fly"})[key + ["score_pre", "score_post"]]
         .sort_values(key, ignore_index=True))
    b = built[key + ["score_pre", "score_post"]].sort_values(key, ignore_index=True)
    b["fly_number"] = b["fly_number"].astype(int)
    if len(a) != len(b) or not (a.to_numpy() == b.to_numpy()).all():
        raise RuntimeError(f"data tables disagree with the figure build ({built_predictions}): "
                           f"{len(a)} vs {len(b)} pairs")


def write_data(out: Path, long: pd.DataFrame, cohorts: Sequence[str],
               bin_hours: float) -> dict[str, pd.DataFrame]:
    staging = _staging(out / "Data")
    staging.mkdir(parents=True)
    win = long[long["in_window"]]
    tag = f"{bin_hours:g}h"
    tables = {
        "window_summary_per_odor.csv": summarise(win, ["cohort", "odor"]),
        "window_per_fly_per_odor.csv": win,
        "window_per_fly_wide.csv": wide_table(win),
        "whole-cohort_summary_per_odor.csv": summarise(long, ["cohort", "odor"]),
        "whole-cohort_per_fly_per_odor.csv": long,
        f"whole-cohort_summary_per_odor_by_{tag}_start_bin.csv":
            summarise(long, ["cohort", "odor", "start_bin"]),
    }
    for name, df in tables.items():
        df.to_csv(staging / name, index=False)
    workbook = staging / "Sensitivity_pre_post_scores.xlsx"
    try:
        with pd.ExcelWriter(workbook) as xl:
            tables["window_summary_per_odor.csv"].to_excel(xl, sheet_name="window summary",
                                                           index=False)
            for cohort in cohorts:
                sub = win[win["cohort"] == cohort]
                if not sub.empty:
                    wide_table(sub).drop(columns="cohort").to_excel(
                        xl, sheet_name=f"window {cohort}"[:31], index=False)
            win.to_excel(xl, sheet_name="window per fly per odor", index=False)
            tables["whole-cohort_summary_per_odor.csv"].to_excel(xl, sheet_name="whole summary",
                                                                 index=False)
            tables[f"whole-cohort_summary_per_odor_by_{tag}_start_bin.csv"].to_excel(
                xl, sheet_name=f"whole by {tag} bin", index=False)
            long.to_excel(xl, sheet_name="whole per fly per odor", index=False)
    except ImportError as exc:
        # The workbook repeats the CSVs above; without an Excel engine, skip only it.
        workbook.unlink(missing_ok=True)
        print(f"[sens-window] WARNING: workbook skipped ({exc}); the CSVs are complete", flush=True)
    (staging / "README.md").write_text(DATA_README)
    _swap_in(staging, out / "Data")
    return tables


DATA_README = """# Pre- vs post-exposure scores: data files

Generated by the pipeline (`scripts/analysis/sensitivity_window_results.py`) on
every run; edits here are overwritten. Pairs and summaries come from the same
code as the score figures (`pretest_vs_test_comparison.py`), and the window rows
are checked fly by fly against the figure build. Flagged flies are dropped. One
pre and one post trial per fly per odor; the light-only probe has no pre-exposure
partner and is left out.

**Response** = model score >= 2. **% responding** = flies responding / flies.

| File | One row per |
|---|---|
| `window_summary_per_odor.csv` | cohort x odor, flies started in the window |
| `window_per_fly_per_odor.csv` | fly x odor (long), in the window |
| `window_per_fly_wide.csv` | fly: `<Odor> pre` / `<Odor> post` columns |
| `whole-cohort_per_fly_per_odor.csv` | fly x odor, every start time (`in_window`, `start_bin` to filter) |
| `whole-cohort_summary_per_odor.csv` | cohort x odor, every start time |
| `whole-cohort_summary_per_odor_by_*_start_bin.csv` | cohort x odor x start bin |
| `Sensitivity_pre_post_scores.xlsx` | all of the above, plus one per-fly sheet per cohort |

Columns: `sensitized_odor` marks the cohort's exposure odor. `start_time` is the
first pre-exposure trial (Pi clock). `score_change` = post - pre. `responded_*`
= score >= 2. In the summaries, `p_score_wilcoxon` is the paired signed-rank test
on scores, exact and correct with tied changes (blank below 6 changed flies).
`p_rate_mcnemar_paired` is the exact paired McNemar test the figures use (as in the
pipeline's Pre-vs-Post figures): pre and post are the same flies, so no unpaired
rate test is given. All p-values here are uncorrected. The pre-specified primary
endpoint (exposed minus control change, whole batches shuffled) is in
`../00_Primary-Endpoint/`; everything in these tables is exploratory.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _report_p_adjust() -> list[str]:
    """``--p-adjust holm`` for start_window_report.py, which hands it on to
    pretest_vs_test_comparison.py, once its parser takes the flag; nothing before that,
    since an unknown flag would fail the whole cohort build."""
    try:
        known = P_ADJUST[0] in swr.build_parser()._option_string_actions
    except Exception:            # a parser that cannot be built: the build itself says so
        known = False
    return list(P_ADJUST) if known else []


def _run(cmd: Sequence, log) -> int:
    cmd = [str(c) for c in cmd]
    print("  $", " ".join(cmd)[:200], flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    log.write(f"\n$ {' '.join(cmd)}\n[exit {r.returncode}]\n{r.stdout[-3000:]}\n"
              f"{r.stderr[-3000:]}\n")
    log.flush()
    return r.returncode


def build_cohort(out: Path, dataset: str, args, log) -> str:
    """The windowed figure set for one cohort, built aside and swapped in."""
    final = out / dataset
    staging = _staging(final)
    code = _run([PY, REPO_ROOT / "scripts/analysis/start_window_report.py",
                 "--dataset", dataset, "--start", args.start, "--end", args.end,
                 "--out-dir", staging, "--config", args.config,
                 "--data-root", args.data_root, "--csv-dir", args.csv_dir,
                 "--flagged-flies-csv", args.flagged_flies_csv,
                 "--rate-test", "mcnemar", *_report_p_adjust()], log)
    if not (staging / "FINDINGS.md").exists():
        shutil.rmtree(staging, ignore_errors=True)
        if code != swr.NOTHING_TO_PLOT:
            # A crash (Python exits 1), not a verdict on the flies: the last good set
            # stays, and the step fails.
            kept = "; previous figures kept" if final.exists() else ""
            return f"failed (exit {code}, see _build_log.txt){kept}"
        # Nothing to plot. Drop the old set too, or it would show flies that no
        # longer qualify.
        if final.exists():
            shutil.rmtree(final)
        return ALL_FLAGGED
    _swap_in(staging, final)
    link_per_fly(out, dataset, args)
    return OK if code == 0 else f"{OK}; some figure steps failed (exit {code})"


def link_per_fly(out: Path, dataset: str, args) -> None:
    final = out / dataset
    flies = kept_flies(final / "_inputs")
    for name, sources in per_fly_sources(args.figures_root, dataset, flies).items():
        replace_links(final / "Per-Fly" / name, sources)


def build_time_of_day(out: Path, dataset: str, args, log) -> int:
    final = out / f"Time-of-Day_{args.bin_hours:g}h-bins_whole-dataset" / dataset
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging(final)
    code = _run([PY, REPO_ROOT / "scripts/analysis/time_of_day_trend.py",
                 "--dataset", dataset, "--bin-hours", args.bin_hours,
                 "--title", f"{dataset}: all flies, by start time",
                 "--predictions-csv", Path(args.csv_dir) / "model_predictions.csv",
                 "--flagged-flies-csv", args.flagged_flies_csv,
                 "--data-root", args.data_root, "--out-dir", staging], log)
    if code == 0:
        _swap_in(staging, final)
    else:
        shutil.rmtree(staging, ignore_errors=True)
    return code


# ---------------------------------------------------------------------------
# Reuse a cohort's figures when nothing they are drawn from changed
# ---------------------------------------------------------------------------

CACHE_FILE = "_build_cache.json"
_ROW_KEY = ("fly", "fly_number", "trial_label")
#: What a kept cohort set must still hold besides FINDINGS.md: every table
#: start_window_report.write_inputs writes, which the data check, the Per-Fly links
#: and the cross-cohort traces read back.
INPUT_FILES = ("predictions.csv", "training_predictions.csv", "wide_pretest.parquet",
               "wide_testing.parquet", "wide_training.parquet")
#: The scorer the figure scripts import (an editable install), and the attribute of its
#: model JSON naming the guard ensemble (``flybehavior_response.xgb_ordinal.GUARD_DIR_ATTR``).
SCORER_PACKAGE = "flybehavior_response"
GUARD_ENSEMBLE_ATTR = "fbr_two_guard_ensemble"


def scorer_digest(package: str = SCORER_PACKAGE) -> str:
    """Digest of every ``.py`` of the scorer package, ``""`` when it is not installed.

    Its code changes without a byte of this repo changing. Found, not imported, so
    this step runs none of its code."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return ""
    roots = [Path(p) for p in (spec.submodule_search_locations or [])] if spec else []
    if not roots:
        return ""
    return fc.combine(*[(str(f.relative_to(root)), fc.file_digest(f))
                        for root in roots for f in sorted(root.rglob("*.py"))])


def model_digests(config: Path | str) -> tuple[str, str]:
    """Digests of the model ``reaction_prediction.model_path`` in ``config`` names and
    of every file of its guard ensemble, the folder beside it that the model JSON's
    ``learner.attributes.fbr_two_guard_ensemble`` names; ``""`` for a missing piece.

    The config names the model by path, so one retrained in place changes no byte of it."""
    try:
        cfg = yaml.safe_load(Path(config).read_text())
        model = Path((cfg.get("reaction_prediction") or {}).get("model_path") or "")
    except (OSError, ValueError, yaml.YAMLError, AttributeError, TypeError):
        return "", ""
    if not model.is_file():
        return "", ""
    try:
        name = json.loads(model.read_bytes())["learner"]["attributes"][GUARD_ENSEMBLE_ATTR]
    except (OSError, ValueError, KeyError, TypeError):   # not JSON, or no guard
        name = None
    # Beside the real file, as the scorer looks (load_xgb_model resolves the path first).
    ensemble = model.resolve().parent / name if isinstance(name, str) and name else None
    files = (sorted(f for f in ensemble.rglob("*") if f.is_file())
             if ensemble is not None and ensemble.is_dir() else [])
    members = (fc.combine(*[(str(f.relative_to(ensemble)), fc.file_digest(f)) for f in files])
               if files else "")
    return fc.file_digest(model), members


def settings_digest(args) -> str:
    """What every cohort's folders are drawn from besides its own rows: this code and
    what it runs, the config, the scorer and its model, and the window."""
    return fc.combine(fc.code_digest([Path(__file__)]), fc.file_digest(args.config),
                      scorer_digest(), *model_digests(args.config), args.start, args.end,
                      args.bin_hours, args.data_root)


def read_flagged(path: Path | str) -> pd.DataFrame:
    """The flagged-flies table, every cell as written (a blank in another cohort's row
    must not turn this cohort's ``1`` into ``1.0``); empty when there is none."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError):            # missing, unreadable or empty
        return pd.DataFrame(columns=["dataset"])


def table_text(df: pd.DataFrame, sort_by: Sequence[str]) -> str:
    """``df`` as CSV text, rows sorted: the same on every pandas version, where a hash of
    the values follows the dtypes each version infers (datetime units, strings)."""
    keys = [c for c in sort_by if c in df.columns]
    return (df.sort_values(keys, kind="stable") if keys else df).to_csv(index=False)


def _cohort_rows(table: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Rows of ``dataset``, matched as the build matches them: the name stripped."""
    if "dataset" not in table.columns:
        return table.iloc[0:0]
    return table[table["dataset"].astype(str).str.strip() == dataset]


def _drop_empty_traces(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` without the ``dir_val_*`` columns they leave empty: another cohort's
    longer trace adds those to the shared table without changing these rows."""
    cols = [c for c in rows.columns if str(c).startswith("dir_val_")]
    empty = rows[cols].isna().all()
    return rows.drop(columns=empty.index[empty.to_numpy(dtype=bool)])


def read_wide_for_fingerprints(csv_dir: Path, datasets: Sequence[str]) -> list[pd.DataFrame]:
    """Only these cohorts' rows of the three wide tables (seconds, not a full read)."""
    paths = [Path(csv_dir) / name for _key, name in swr._TABLES]
    return [pd.read_parquet(p, filters=[("dataset", "in", list(datasets))])
            for p in paths if p.exists()]


def cohort_fingerprint(dataset: str, folders: Sequence[str], *, predictions: pd.DataFrame,
                       training: pd.DataFrame, wide: Sequence[pd.DataFrame],
                       starts: pd.DataFrame, flagged: pd.DataFrame, settings: str) -> str:
    """Digest of what a cohort's folders are drawn from. Traces count only for the
    window's folders: the whole-cohort time-of-day set reads scores alone. Rows are
    picked as the build picks them, and nothing of another cohort counts: not its rows,
    its flagged flies (``flagged``: the ``read_flagged`` table) or its longer traces."""
    keep = set(folders)
    traces = [fc.frame_digest(_drop_empty_traces(
                  t[(t["dataset"].astype(str).str.strip() == dataset)
                    & t["fly"].astype(str).str.strip().isin(keep)]), _ROW_KEY)
              for t in wide]
    scores = [fc.frame_digest(_cohort_rows(t, dataset), _ROW_KEY)
              for t in (predictions, training)]
    mine = _cohort_rows(flagged, dataset)
    return fc.combine(settings, sorted(keep), table_text(starts, ("folder",)),
                      table_text(mine, list(mine.columns)), *scores, *traces)


def built_digest(out: Path, dataset: str, time_of_day: Path) -> str:
    """Every file of a cohort's set and of its time-of-day set, by path and size.
    ``Per-Fly/`` is left out: ``link_per_fly`` refreshes its links every run."""
    per_fly = out / dataset / "Per-Fly"
    files = []
    for top in (out / dataset, time_of_day):
        for root, dirs, names in os.walk(top):
            here = Path(root)
            dirs[:] = [d for d in dirs if here / d != per_fly]
            for name in names:
                path = here / name
                try:
                    files.append((str(path.relative_to(out)), path.lstat().st_size))
                except OSError:                  # removed while being listed
                    continue
    return fc.combine(*sorted(files))


def _intact(final: Path, time_of_day: Path) -> bool:
    return ((final / "FINDINGS.md").is_file() and time_of_day.is_dir()
            and all((final / "_inputs" / name).is_file() for name in INPUT_FILES))


def build_or_reuse(out: Path, dataset: str, args, log, cache: fc.FigureCache,
                   fingerprint: str) -> tuple[str, bool]:
    """Rebuild a cohort's folders unless nothing they are drawn from changed and they
    are still as the build left them. ``(status, reused)``: main checks a kept set."""
    final = out / dataset
    time_of_day = out / f"Time-of-Day_{args.bin_hours:g}h-bins_whole-dataset" / dataset
    if _intact(final, time_of_day) and cache.is_current(
            dataset, fc.combine(fingerprint, built_digest(out, dataset, time_of_day))):
        print(f"[sens-window] {dataset}: nothing changed, figures kept", flush=True)
        link_per_fly(out, dataset, args)
        return OK, True
    cache.forget(dataset)
    status = build_cohort(out, dataset, args, log)
    tod = build_time_of_day(out, dataset, args, log)
    if tod != 0:
        status += f"; time-of-day failed (exit {tod})"
    elif status == OK:
        # the folders as built, too: a file lost or changed later is a reason to rebuild
        cache.record(dataset, fc.combine(fingerprint, built_digest(out, dataset, time_of_day)))
    cache.save()
    return status, False


def check_built(out: Path, dataset: str, long: pd.DataFrame, status: str,
                flagged: Path) -> None:
    """``check_matches_figures`` on the set this run built or kept. A failed build leaves
    the last good set in place, drawn from older rows: it is not checked against today's."""
    if status.startswith(OK) and (out / dataset / "FINDINGS.md").exists():
        check_matches_figures(long, out / dataset / "_inputs" / "predictions.csv", flagged)


# ---------------------------------------------------------------------------
# The pre-specified primary endpoint (user 2026-11-18)
# ---------------------------------------------------------------------------

#: Columns of ``primary_endpoint_table`` (an empty table has them too).
PRIMARY_COLUMNS = (
    "cohort", "odor", "control", "estimate_pp", "ci_lo_pp", "ci_hi_pp", "p_batch", "p_fly",
    "perm_exact", "change_pp_exposed", "change_pp_control", "n_flies_exposed",
    "n_flies_control", "n_batches_exposed", "n_batches_control",
    "secondary_pct_pre_exposed", "secondary_pct_post_exposed", "secondary_p_mcnemar_exposed",
    "secondary_pct_pre_control", "secondary_pct_post_control", "secondary_p_mcnemar_control",
    "first_start_exposed", "last_start_exposed", "first_start_control", "last_start_control",
    "concurrent")


def _start_dates(rows: pd.DataFrame) -> tuple[str, str]:
    """First and last start date (YYYY-MM-DD) of these flies; ``""`` when unknown."""
    if "start_time" not in rows.columns:
        return "", ""
    t = pd.to_datetime(rows["start_time"], errors="coerce").dropna()
    if t.empty:
        return "", ""
    return t.min().strftime("%Y-%m-%d"), t.max().strftime("%Y-%m-%d")


def _endpoint_flies(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per fly on one odor: its batch and its post - pre response (-1, 0, +1)."""
    pre = rows["responded_pre"].astype(bool).to_numpy()
    post = rows["responded_post"].astype(bool).to_numpy()
    return pd.DataFrame({"batch_folder": rows["batch_folder"].astype(str).to_numpy(),
                         "change": post.astype(int) - pre.astype(int),
                         "responded_pre": pre, "responded_post": post})


def primary_endpoint_table(long: pd.DataFrame, cohorts: Sequence[str],
                           control: str | None) -> pd.DataFrame:
    """The pre-specified primary endpoint (``PRIMARY_ENDPOINT``), one row per exposed cohort.

    For every cohort but ``control`` whose sensitized odor the control flies were also
    tested on: the window flies (``in_window``, as on the folder's main figures), each
    fly's post - pre response to that odor (-1, 0, +1), compared with the control
    flies' on the same odor by ``fly_stats.controlled_change_test``: the exposed flies'
    change (pooled: summed over flies / flies) minus the control flies', its 95 %
    bootstrap CI over batches and the p of swapping whole batches between the two
    cohorts, in percentage points (``*_pp``). ``p_fly`` swaps single flies (reference).

    ``secondary_*``: each group's own % responding pre and post and its own exact
    McNemar p, not controlled for time or re-testing. ``first_start_*``/``last_start_*``:
    the first and last start date of each group's flies; ``concurrent``: whether the
    two date ranges overlap."""
    empty = pd.DataFrame(columns=list(PRIMARY_COLUMNS))
    if not control or long.empty:
        return empty
    rows = long[long["in_window"].astype(bool)] if "in_window" in long.columns else long
    ctrl = rows[rows["cohort"] == control]
    out = []
    for cohort in [c for c in dict.fromkeys(cohorts) if c != control]:
        mine = rows[rows["cohort"] == cohort]
        for odor in sorted(mine.loc[mine["sensitized_odor"].astype(bool), "odor"].unique()):
            xr, cr = mine[mine["odor"] == odor], ctrl[ctrl["odor"] == odor]
            if xr.empty or cr.empty:
                continue
            x, c = _endpoint_flies(xr), _endpoint_flies(cr)
            r = fs.controlled_change_test(x, c, value="change", batch="batch_folder",
                                          n_perm=PRIMARY_N_PERM, n_boot=PRIMARY_N_BOOT,
                                          seed=PRIMARY_SEED)
            row = {"cohort": cohort, "odor": odor, "control": control,
                   "estimate_pp": 100 * r["estimate"], "ci_lo_pp": 100 * r["ci_lo"],
                   "ci_hi_pp": 100 * r["ci_hi"], "p_batch": r["p_batch"], "p_fly": r["p_fly"],
                   "perm_exact": bool(r["perm_exact"]),
                   "change_pp_exposed": 100 * r["exposed_change"],
                   "change_pp_control": 100 * r["control_change"]}
            for group, g, dates in (("exposed", x, _start_dates(xr)),
                                    ("control", c, _start_dates(cr))):
                p = fs.mcnemar_exact_p(g["responded_pre"], g["responded_post"])
                row.update({
                    f"n_flies_{group}": int(r[f"n_flies_{group}"]),
                    f"n_batches_{group}": int(r[f"n_batches_{group}"]),
                    f"secondary_pct_pre_{group}": 100 * float(g["responded_pre"].mean()),
                    f"secondary_pct_post_{group}": 100 * float(g["responded_post"].mean()),
                    f"secondary_p_mcnemar_{group}": np.nan if p is None else p,
                    f"first_start_{group}": dates[0], f"last_start_{group}": dates[1],
                })
            known = all(row[f"{e}_start_{g}"] for e in ("first", "last")
                        for g in ("exposed", "control"))
            row["concurrent"] = (row["first_start_exposed"] <= row["last_start_control"]
                                 and row["first_start_control"] <= row["last_start_exposed"]
                                 ) if known else np.nan
            out.append(row)
    if not out:
        return empty
    t = pd.DataFrame(out, columns=list(PRIMARY_COLUMNS))
    pp = [c for c in t.columns if c.endswith("_pp") or c.startswith("secondary_pct")]
    t[pp] = t[pp].astype(float).round(4)
    ps = ["p_batch", "p_fly", "secondary_p_mcnemar_exposed", "secondary_p_mcnemar_control"]
    t[ps] = t[ps].astype(float).round(6)
    return t


def _signed(v, digits: int = 1) -> str:
    """'+6.6', '−14.7' (a true minus sign), '0.0' (never a signed zero)."""
    v = float(v)
    if not np.isfinite(v):
        return "–"
    text = f"{v:+.{digits}f}"
    return f"{0:.{digits}f}" if float(text) == 0 else text.replace("-", "−")


def _p_short(p) -> str:
    """'p = 0.61', 'p = 0.027', 'p < 0.001' (two significant digits)."""
    p = float(p) if p is not None else np.nan
    if not np.isfinite(p):
        return "p = –"
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def primary_effect_text(r) -> str:
    """'Δ = +6.6 pp [−14.7, +27.9] · p = 0.61 (whole batches shuffled)'."""
    return (f"Δ = {_signed(r['estimate_pp'])} pp [{_signed(r['ci_lo_pp'])}, "
            f"{_signed(r['ci_hi_pp'])}] · {_p_short(r['p_batch'])} (whole batches shuffled)")


def primary_n_text(r) -> str:
    """'54 vs 51 flies, 14 vs 14 batches' (exposed vs control)."""
    return (f"{int(r['n_flies_exposed'])} vs {int(r['n_flies_control'])} flies, "
            f"{int(r['n_batches_exposed'])} vs {int(r['n_batches_control'])} batches")


def primary_row_text(r) -> str:
    """One exposed cohort's primary result as one line."""
    return f"{primary_effect_text(r)} · {primary_n_text(r)}"


def primary_concurrency_notes(table: pd.DataFrame, *, bold: bool = False) -> list[str]:
    """A note per exposed cohort whose flies ran on days the control flies' do not overlap."""
    notes = []
    for _, r in table.iterrows():
        if pd.isna(r["concurrent"]) or bool(r["concurrent"]):
            continue
        lead = "**Not run concurrently:**" if bold else "Not run concurrently:"
        notes.append(f"{lead} {r['cohort']} flies started {r['first_start_exposed']} to "
                     f"{r['last_start_exposed']}, the {r['control']} (control) flies "
                     f"{r['first_start_control']} to {r['last_start_control']}; a drift between "
                     "those days is not controlled for.")
    return notes


def _typeset(text: str) -> str:
    return text.replace(">=", "≥").replace("post - pre", "post − pre")


def plot_primary_endpoint(table: pd.DataFrame, cohorts: Sequence[str], title: str, path: Path,
                          *, window: str = "") -> None:
    """The headline figure. A (largest): per exposed cohort, its flies' change in %
    responding to the exposed odor minus the control flies' change, with the 95 % CI,
    a zero line and the test. B (secondary: smaller, lighter): each group's own pre ->
    post on that odor with its own McNemar p, which time and re-testing also move.
    Footer: the pre-specification, the control and any non-concurrent control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from matplotlib.transforms import blended_transform_factory

    rows = table.reset_index(drop=True)
    n = len(rows)
    control = str(rows["control"].iloc[0])
    colours, order = cohort_colours(cohorts), list(cohorts)

    def style(cohort: str) -> tuple[str, str]:
        i = order.index(cohort) if cohort in order else len(order)
        return colours.get(cohort, INK_2), COHORT_MARKERS[i % len(COHORT_MARKERS)]

    spec = PRIMARY_ENDPOINT
    foot = [f"Pre-specified {spec['registered']} (PRIMARY_ENDPOINT in "
            f"sensitivity_window_results.py). Metric: {_typeset(spec['metric'])}; window: "
            f"{spec['window']}; odor: {spec['odor']}.",
            f"Contrast: {_typeset(spec['contrast'])}. Test: {spec['test']}. CI: {spec['ci']}.",
            f"Control cohort: {control}. Each group pools its flies (flies responding post − "
            "pre / flies, one trial per fly on that odor)"
            + (f"; flies started {window}" if window else "") + ". A test swapping single "
            "flies instead (p_fly, which ignores that flies of a batch share a day) is in "
            "primary_endpoint.csv."]
    foot = [w for line in foot for w in textwrap.wrap(line, 225)]
    notes = [w for line in primary_concurrency_notes(rows) for w in textwrap.wrap(line, 225)]
    line_in = 8.5 * 1.45 / 72

    ncol = min(n, 3)
    nrow_b = int(np.ceil(n / ncol))
    width = 14.0
    top, a_head, a_axes, a_below = 0.62, 0.45, 0.78 * n + 0.9, 0.95
    b_head, b_title, b_axes, b_below = 0.75, 0.34, 1.75, 0.45
    height = (top + a_head + a_axes + a_below + b_head + nrow_b * (b_title + b_axes + b_below)
              + line_in * (len(foot) + len(notes)) + 0.35)
    fig = plt.figure(figsize=(width, height))

    def fy(inches: float) -> float:          # inches from the top -> figure fraction
        return 1.0 - inches / height

    fig.suptitle(title, y=fy(0.18), va="top", fontsize=13.5, color=INK)

    # A: the controlled difference, the headline
    y_in = top
    fig.text(0.012, fy(y_in), "A  Exposed flies' change minus control flies' change on the "
             f"exposed odor (control: {control})", ha="left", va="top", fontsize=12.5,
             color=INK, fontweight="bold")
    y_in += a_head
    ax = fig.add_axes((0.205, fy(y_in + a_axes), 0.33, a_axes / height), label="primary_A")
    y_in += a_axes + a_below
    ys = np.arange(n)[::-1].astype(float)        # the first configured cohort on top
    vals = rows[["ci_lo_pp", "ci_hi_pp", "estimate_pp"]].to_numpy(float)
    reach = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 10.0
    lim = max(20.0, 10.0 * np.ceil(reach * 1.08 / 10.0))
    ax.axvline(0, color=INK_2, lw=1.0, zorder=1)
    on_left = blended_transform_factory(ax.transAxes, ax.transData)
    on_right = blended_transform_factory(fig.transFigure, ax.transData)
    for y, (_, r) in zip(ys, rows.iterrows()):
        colour, marker = style(r["cohort"])
        lo, hi = float(r["ci_lo_pp"]), float(r["ci_hi_pp"])
        ax.plot([lo, hi], [y, y], color=colour, lw=2.0, solid_capstyle="round", zorder=2)
        for end in (lo, hi):
            ax.plot([end, end], [y - 0.1, y + 0.1], color=colour, lw=1.4, zorder=2)
        ax.plot([float(r["estimate_pp"])], [y], ls="none", marker=marker, ms=10, color=colour,
                mec="white", mew=1.3, zorder=3)
        ax.text(-0.025, y + 0.14, r["cohort"], transform=on_left, ha="right", va="center",
                fontsize=10.5, color=INK)
        ax.text(-0.025, y - 0.2, f"exposed odor: {r['odor']}", transform=on_left, ha="right",
                va="center", fontsize=9, color=INK_2)
        ax.text(0.55, y + 0.14, primary_effect_text(r), transform=on_right, ha="left",
                va="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(0.55, y - 0.2, f"{primary_n_text(r)} (exposed vs control)", transform=on_right,
                ha="left", va="center", fontsize=9, color=INK_2)
    hint_y = n - 1 + 0.62
    ax.text(-0.97 * lim, hint_y, "← control flies changed more", ha="left", va="center",
            fontsize=8.5, color=INK_2)
    ax.text(0.97 * lim, hint_y, "exposed flies changed more →", ha="right", va="center",
            fontsize=8.5, color=INK_2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.75, n - 1 + 0.95)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MultipleLocator(10 if lim <= 40 else 20))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _signed(v, 0)))
    ax.set_xlabel("Exposed change − control change (percentage points)\n"
                  "in % responding to the exposed odor; bar: 95 % CI", fontsize=9.5, color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)

    # B: each group's own change, secondary
    fig.text(0.012, fy(y_in), "B  Each group's own change on that odor: secondary, not "
             "controlled for time or re-testing (exact McNemar p, pre vs post of the same "
             "flies)", ha="left", va="top", fontsize=10.5, color=INK_2)
    handles = [Line2D([], [], color=style(c)[0], marker=style(c)[1], ms=6, mec="white", lw=1.3,
                      label=f"{c} flies (exposed)") for c in rows["cohort"].unique()]
    handles.append(Line2D([], [], color=style(control)[0], marker=style(control)[1], ms=6,
                          mec="white", lw=1.3, label=f"{control} flies (control)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, fy(y_in + 0.3)),
               ncol=len(handles), frameon=False, fontsize=8.5, labelcolor=INK_2,
               handlelength=2.2, borderaxespad=0.0)
    y_in += b_head
    pcts = rows[[f"secondary_pct_{ph}_{g}" for ph in ("pre", "post")
                 for g in ("exposed", "control")]].to_numpy(float)
    ymax = float(min(100.0, max(40.0, 10.0 * np.ceil((np.nanmax(pcts) + 12.0) / 10.0))))
    left0, slot = 0.06, (0.99 - 0.06) / ncol
    for i, (_, r) in enumerate(rows.iterrows()):
        row_b, col_b = divmod(i, ncol)
        top_in = y_in + row_b * (b_title + b_axes + b_below) + b_title
        axb = fig.add_axes((left0 + col_b * slot, fy(top_in + b_axes), 0.4 * slot,
                            b_axes / height), label=f"primary_B_{i}")
        labels = []
        for group, who in (("exposed", r["cohort"]), ("control", control)):
            colour, marker = style(who)
            pre = float(r[f"secondary_pct_pre_{group}"])
            post = float(r[f"secondary_pct_post_{group}"])
            axb.plot([0, 1], [pre, post], color=colour, lw=1.3, alpha=0.85, marker=marker, ms=6,
                     mec="white", mew=0.8, zorder=3 if group == "exposed" else 2)
            labels.append([post, f"{group}: {pre:.0f}% → {post:.0f}% · "
                                 f"{_p_short(r[f'secondary_p_mcnemar_{group}'])}"])
        # the two end labels never overlap
        need = 0.12 * ymax
        (ya, _), (yb, _) = labels
        if abs(ya - yb) < need:
            mid, up = (ya + yb) / 2, 1.0 if ya >= yb else -1.0
            ya, yb = mid + up * need / 2, mid - up * need / 2
        shift = max(0.0, max(ya, yb) - 0.96 * ymax) - max(0.0, 0.04 * ymax - min(ya, yb))
        labels[0][0], labels[1][0] = ya - shift, yb - shift
        for yy, text in labels:
            axb.text(1.16, yy, text, ha="left", va="center", fontsize=8.5, color=INK)
        axb.set_xlim(-0.15, 1.15)
        axb.set_xticks([0, 1])
        axb.set_xticklabels([PRE_EXPOSURE, POST_EXPOSURE], fontsize=8.5, color=INK_2)
        axb.set_ylim(0, ymax)
        axb.yaxis.set_major_locator(MultipleLocator(25 if ymax > 60 else 10))
        axb.yaxis.grid(True, color=GRID, lw=0.7)
        axb.set_axisbelow(True)
        if col_b == 0:
            axb.set_ylabel("% of flies responding", fontsize=8.5, color=INK_2)
        else:
            axb.tick_params(labelleft=False)
        for side in ("top", "right"):
            axb.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axb.spines[side].set_color(INK_2)
            axb.spines[side].set_linewidth(0.8)
        axb.tick_params(colors=INK_2, labelsize=8)
        axb.set_title(f"{r['cohort']} · {r['odor']}", loc="left", fontsize=9, color=INK_2,
                      pad=6)

    y_in += nrow_b * (b_title + b_axes + b_below)
    fig.text(0.012, fy(y_in), "\n".join(foot), ha="left", va="top", fontsize=8.5, color=INK_2,
             linespacing=1.45)
    if notes:
        fig.text(0.012, fy(y_in + line_in * len(foot) + 0.04), "\n".join(notes), ha="left",
                 va="top", fontsize=8.5, color=INK, linespacing=1.45)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def primary_readme(table: pd.DataFrame, *, window: str = "") -> str:
    """``00_Primary-Endpoint/README.md``: the registration, the result, the method."""
    spec = PRIMARY_ENDPOINT
    control = str(table["control"].iloc[0])
    lines = [f"# Primary endpoint (pre-specified {spec['registered']})", "",
             "Generated by the pipeline (`scripts/analysis/sensitivity_window_results.py`, "
             "`PRIMARY_ENDPOINT`) on every run; edits here are overwritten. This is the one "
             "test the results folder is read for; everything else in it is exploratory."
             + (f" Flies started {window} (first pre-exposure trial, Pi clock)."
                if window else ""), "",
             "| | Pre-specified |", "|---|---|",
             *[f"| {k.upper() if k == 'ci' else k.capitalize()} | {v} |" for k, v in spec.items()],
             f"| Control cohort | {control} (`control_cohort`) |",
             "", "## Result", "",
             "| Exposed cohort | Odor | Δ, percentage points | 95 % CI | p, whole batches "
             "shuffled | Flies (exposed vs control) | Batches | Exposed flies started | "
             "Control flies started |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['cohort']} | {r['odor']} | {_signed(r['estimate_pp'])} | "
                     f"{_signed(r['ci_lo_pp'])} to {_signed(r['ci_hi_pp'])} | {_p(r['p_batch'])} "
                     f"| {int(r['n_flies_exposed'])} vs {int(r['n_flies_control'])} | "
                     f"{int(r['n_batches_exposed'])} vs {int(r['n_batches_control'])} | "
                     f"{r['first_start_exposed']} to {r['last_start_exposed']} | "
                     f"{r['first_start_control']} to {r['last_start_control']} |")
    lines += ["", "Δ = the exposed flies' change in % responding (post − pre) minus the control "
              "flies' change, on the exposed cohort's odor. Each group pools its flies (flies "
              "responding post − pre / flies; one trial per fly on that odor, every fly once). "
              "p: whole batches (`batch_folder`) swapped between the two cohorts and Δ "
              f"recomputed, over every split when there are at most {fs.MAX_EXACT_SPLITS:,} "
              f"(`perm_exact`), else {PRIMARY_N_PERM:,} random splits (seed {PRIMARY_SEED}). "
              f"CI: batches resampled with replacement within each cohort, {PRIMARY_N_BOOT:,} "
              f"draws (seed {PRIMARY_SEED}). `p_fly` in the CSV swaps single flies instead and "
              "ignores that flies of a batch share a day: reference only."]
    notes = primary_concurrency_notes(table, bold=True)
    if notes:
        lines += ["", *notes]
    lines += ["", "## Secondary: each group's own change (not controlled for time or "
              "re-testing)", "",
              "| Exposed cohort | Odor | Exposed flies, % responding pre → post | McNemar p | "
              f"{control} flies, pre → post | McNemar p |",
              "|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['cohort']} | {r['odor']} | "
                     f"{_pct(r['secondary_pct_pre_exposed'])} → "
                     f"{_pct(r['secondary_pct_post_exposed'])} | "
                     f"{_p(r['secondary_p_mcnemar_exposed'])} | "
                     f"{_pct(r['secondary_pct_pre_control'])} → "
                     f"{_pct(r['secondary_pct_post_control'])} | "
                     f"{_p(r['secondary_p_mcnemar_control'])} |")
    lines += ["", "Exact paired McNemar (pre and post are the same flies), uncorrected; – "
              "when no fly changed.", "",
              f"`{PRIMARY_CSV}`: one row per exposed cohort. `*_pp` in percentage points "
              "(`change_pp_*`: each group's own change); `secondary_*`: the per-group numbers "
              "above; `first_start_*`/`last_start_*`: the first and last start date of each "
              "group's flies; `concurrent`: whether those two date ranges overlap."]
    return "\n".join(lines) + "\n"


def write_primary_endpoint(out: Path, long: pd.DataFrame, cohorts: Sequence[str],
                           control: str | None, log, *, window: str = "") -> pd.DataFrame:
    """``00_Primary-Endpoint/``: the pre-specified primary endpoint as a figure,
    ``primary_endpoint.csv`` and a README, built aside and swapped in. Drawn on every
    run (no cache gates it). Returns the table; when it is empty (no control, or no
    exposed cohort's odor met by window control flies) the folder is removed, so no
    stale headline stays."""
    final = out / PRIMARY_DIR
    table = primary_endpoint_table(long, cohorts, control)
    if table.empty:
        shutil.rmtree(final, ignore_errors=True)
        msg = (f"[primary endpoint] not computed: no window flies of the control cohort "
               f"{control} on any exposed cohort's odor" if control
               else "[primary endpoint] not computed: no control cohort")
        print(f"[sens-window] {msg}", flush=True)
        if log is not None:
            log.write(f"\n{msg}\n")
            log.flush()
        return table
    staging = _staging(final)
    staging.mkdir(parents=True)
    table.to_csv(staging / PRIMARY_CSV, index=False)
    plot_primary_endpoint(table, cohorts,
                          f"Primary endpoint (pre-specified {PRIMARY_ENDPOINT['registered']})"
                          + (f": flies started {window}" if window else ""),
                          staging / PRIMARY_PNG, window=window)
    (staging / "README.md").write_text(primary_readme(table, window=window), encoding="utf-8")
    _swap_in(staging, final)
    for _, r in table.iterrows():
        line = f"[primary endpoint] {r['cohort']} vs {r['control']}: {primary_row_text(r)}"
        print(f"[sens-window] {line}", flush=True)
        if log is not None:
            log.write(f"\n{line}\n")
    if log is not None:
        log.flush()
    return table


def primary_readme_lines(primary: pd.DataFrame | None, control: str | None) -> list[str]:
    """The opening section of the folder README: the primary endpoint's numbers, then
    the one sentence that everything after it is exploratory."""
    lines = [f"## Primary endpoint (pre-specified {PRIMARY_ENDPOINT['registered']})", ""]
    if not control:
        lines.append("Not computed: no control cohort is set (`control_cohort` in "
                     "`analysis.sensitivity_window_results` of config_new.yaml).")
    elif primary is None or primary.empty:
        lines.append(f"Not computed this run: the control cohort {control} has no window "
                     "flies on any exposed cohort's odor yet.")
    else:
        lines += [f"Exposed flies' change in % responding (score ≥ {THRESHOLD}) to their "
                  f"cohort's exposed odor minus the control flies' ({control}) change on the "
                  "same odor, production scorer window, window flies. p: whole batches "
                  "shuffled between the two cohorts; 95 % CI: batches resampled within each "
                  f"cohort. Figure, table and method: `{PRIMARY_DIR}/`.", "",
                  "| Exposed cohort | Odor | Δ, percentage points | 95 % CI | p, whole batches "
                  "shuffled | Flies (exposed vs control) | Batches (exposed vs control) |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in primary.iterrows():
            lines.append(f"| {r['cohort']} | {r['odor']} | {_signed(r['estimate_pp'])} | "
                         f"{_signed(r['ci_lo_pp'])} to {_signed(r['ci_hi_pp'])} | "
                         f"{_p(r['p_batch'])} | {int(r['n_flies_exposed'])} vs "
                         f"{int(r['n_flies_control'])} | {int(r['n_batches_exposed'])} vs "
                         f"{int(r['n_batches_control'])} |")
        notes = primary_concurrency_notes(primary, bold=True)
        if notes:
            lines += ["", *notes]
    return lines + ["", "Everything after this section is exploratory, with p-values "
                    "Holm-adjusted within each figure."]


# ---------------------------------------------------------------------------
# Cross-cohort change figure
# ---------------------------------------------------------------------------

# Categorical slots, light surface (dataviz reference palette, adjacent pairs
# validated: CVD dE 9.1, normal-vision 22.9). Aqua and yellow sit under 3:1 on
# white, so every cohort also gets its own marker and the values are in the CSV.
COHORT_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948")
COHORT_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def cohort_colours(cohorts: Sequence[str]) -> dict[str, str]:
    """Slot by position in the configured list, so a cohort never changes colour."""
    return {c: COHORT_PALETTE[i % len(COHORT_PALETTE)] for i, c in enumerate(cohorts)}


def _sem(v: pd.Series) -> float:
    v = v.astype(float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


def change_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per cohort x odor: post - pre in % responding and in mean score.

    Flies pooled, one trial per fly per odor. The SEMs are paired: taken over each
    fly's own post - pre (responded: -1/0/+1, score: integer change)."""
    s = summarise(long, ["cohort", "odor"])
    if s.empty:
        return s
    flips = long["responded_post"].astype(int) - long["responded_pre"].astype(int)
    sem = (long.assign(_flip=flips * 100.0)
           .groupby(["cohort", "odor"])
           .agg(sem_pct_change=("_flip", _sem), sem_score_change=("score_change", _sem))
           .reset_index())
    keep = ["cohort", "odor", "sensitized_odor", "n_flies", "pct_responding_pre",
            "pct_responding_post", "pct_change", "mean_score_pre", "mean_score_post",
            "mean_score_change", "p_rate_mcnemar_paired", "p_score_wilcoxon"]
    out = s[keep].merge(sem, on=["cohort", "odor"])
    out[["sem_pct_change", "sem_score_change"]] = out[
        ["sem_pct_change", "sem_score_change"]].round(4)
    cols = keep[:7] + ["sem_pct_change"] + keep[7:10] + ["sem_score_change"] + keep[10:]
    return out[cols]


def odor_order(table: pd.DataFrame, cohorts: Sequence[str]) -> list[str]:
    """Each cohort's sensitized odor in cohort order, then the rest alphabetically."""
    first = []
    for c in cohorts:
        hit = table.loc[(table["cohort"] == c) & table["sensitized_odor"], "odor"]
        first += [o for o in hit if o not in first]
    return first + sorted(set(table["odor"]) - set(first))


def _stars(p) -> str:
    p = float(p) if p is not None else np.nan
    return "" if not np.isfinite(p) else "***" if p < 0.001 else "**" if p < 0.01 \
        else "*" if p < 0.05 else ""


#: Panels of the change figure: (value, its SEM, its p, y label, value format).
RESPONSE_PANELS = (("pct_change", "sem_pct_change", "p_rate_mcnemar_paired",
                    "Change in % of flies responding\n(post − pre, percentage points)", "{:+.0f}"),
                   ("mean_score_change", "sem_score_change", "p_score_wilcoxon",
                    "Change in mean PER score\n(post − pre)", "{:+.2f}"))
RESPONSE_NOTE = ("Black outline = the cohort's own sensitized odor (value shown); bars "
                 "in legend order within each odor. Whiskers ±1 paired SEM over flies. * p<0.05, "
                 "** p<0.01, *** p<0.001, uncorrected (top: McNemar on rates, bottom: Wilcoxon on "
                 "scores).")


def plot_cross_cohort(table: pd.DataFrame, cohorts: Sequence[str], colours: Mapping[str, str],
                      title: str, path: Path, *, panels: Sequence[tuple] = RESPONSE_PANELS,
                      note: str = RESPONSE_NOTE) -> None:
    """Two panels on one odor axis, a bar per cohort: by default the change in %
    responding and in mean score (``panels`` picks the columns)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    present = [c for c in cohorts if c in set(table["cohort"])]
    odors = odor_order(table, present)
    k = len(present)
    step = 0.84 / k
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 1.45 * len(odors) + 2.0), 8.6),
                             gridspec_kw={"hspace": 0.32})
    by = table.set_index(["cohort", "odor"])
    for ax, (col, sem_col, p_col, ylab, fmt) in zip(axes, panels):
        for i in range(len(odors)):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
        ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
        for j, c in enumerate(present):
            colour = colours[c]
            for i, odor in enumerate(odors):
                if (c, odor) not in by.index:
                    continue
                r = by.loc[(c, odor)]
                x = i + (j - (k - 1) / 2) * step
                y, e = float(r[col]), float(r[sem_col])
                sens = bool(r["sensitized_odor"])
                ax.bar(x, y, width=step * 0.86, color=colour, edgecolor=INK,
                       linewidth=1.6 if sens else 0, zorder=2)
                if np.isfinite(e):
                    ax.errorbar(x, y, yerr=e, fmt="none", ecolor=INK_2, elinewidth=1.0, zorder=3)
                top = y + (e if np.isfinite(e) else 0) if y >= 0 else y - (e if np.isfinite(e) else 0)
                label = (fmt.format(y) if sens else "") + _stars(r[p_col])
                if label:
                    ax.annotate(label, (x, top), xytext=(0, 5 if y >= 0 else -5),
                                textcoords="offset points", ha="center",
                                va="bottom" if y >= 0 else "top", fontsize=8.5,
                                color=INK, fontweight="bold" if sens else "normal")
        ax.set_xlim(-0.5, len(odors) - 0.5)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors, fontsize=10, color=INK)
        ax.set_ylabel(ylab, fontsize=10, color=INK)
        lo_, hi_ = ax.get_ylim()
        pad = 0.08 * (hi_ - lo_)
        ax.set_ylim(lo_ - pad, hi_ + pad)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2)
    n = table.groupby("cohort")["n_flies"].max()
    handles = [Patch(color=colours[c], label=f"{c}  (n={int(n[c])} flies, sensitized to "
                                             f"{trained_odor(c)})")
               for c in present]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=min(2, k), frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    fig.text(0.5, 0.955, note, ha="center", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(top=0.875 if k > 2 else 0.89, bottom=0.07, left=0.09, right=0.98)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def _perm_p(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p for mean(a) - mean(b): flies shuffled between the groups.

    Fixed seed, so a rebuild with the same flies gives the same p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return np.nan
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.tile(pooled, (n_perm, 1)), axis=1)
    diff = np.abs(perm[:, :a.size].mean(axis=1) - perm[:, a.size:].mean(axis=1))
    return float((np.sum(diff >= obs - 1e-12) + 1) / (n_perm + 1))


def control_subtracted_table(long: pd.DataFrame, control: str | None) -> pd.DataFrame:
    """Per cohort (not the control) x odor: its post - pre change minus the control
    cohort's change on the same odor, in % responding and in mean score.

    What exposure does to every odor whatever was paired with it (handling, time on the
    rig, the paraffin carrier) cancels; what is left is specific to the cohort's odor.
    Flies pooled as in ``change_table``; the two cohorts are different flies, so the
    paired SEMs add in quadrature and the p is a permutation test of the difference of
    mean per-fly changes (responded: -1/0/+1, score: integer change)."""
    if not control or long.empty or control not in set(long["cohort"]):
        return pd.DataFrame()
    flips = long["responded_post"].astype(int) - long["responded_pre"].astype(int)
    long = long.assign(_flip=flips * 100.0)
    change = change_table(long).set_index(["cohort", "odor"])
    ctrl = long[long["cohort"] == control]
    rows = []
    for (cohort, odor), g in long[long["cohort"] != control].groupby(["cohort", "odor"],
                                                                      sort=True):
        c = ctrl[ctrl["odor"] == odor]
        if c.empty:
            continue
        x, k = change.loc[(cohort, odor)], change.loc[(control, odor)]
        rows.append({
            "cohort": cohort, "odor": odor, "control": control,
            "sensitized_odor": bool(x["sensitized_odor"]),
            "n_flies": int(x["n_flies"]), "n_flies_control": int(k["n_flies"]),
            "pct_change": x["pct_change"], "pct_change_control": k["pct_change"],
            "pct_change_minus_control": x["pct_change"] - k["pct_change"],
            "sem_pct_change_minus_control": float(np.hypot(x["sem_pct_change"],
                                                           k["sem_pct_change"])),
            "mean_score_change": x["mean_score_change"],
            "mean_score_change_control": k["mean_score_change"],
            "mean_score_change_minus_control": x["mean_score_change"] - k["mean_score_change"],
            "sem_score_change_minus_control": float(np.hypot(x["sem_score_change"],
                                                             k["sem_score_change"])),
            "p_rate_perm": _perm_p(g["_flip"], c["_flip"]),
            "p_score_perm": _perm_p(g["score_change"], c["score_change"]),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        num = out.select_dtypes("float").columns
        out[num] = out[num].round(4)
    return out


#: Panels of the control-subtracted figure: (prefix, p, y label, value format).
CONTROL_PANELS = (("pct_change", "sem_pct_change_minus_control", "p_rate_perm",
                   "Change in % of flies responding\n(post − pre, percentage points)", "{:+.0f}"),
                  ("mean_score_change", "sem_score_change_minus_control", "p_score_perm",
                   "Change in mean PER score\n(post − pre)", "{:+.2f}"))


def control_subtracted_name(cohort: str) -> str:
    return f"change_minus_control_{cohort}_window.png"


def plot_control_subtracted(table: pd.DataFrame, cohort: str, colours: Mapping[str, str],
                            title: str, path: Path) -> None:
    """One cohort, every odor: its change, the control's change (tinted) and the
    difference (solid, ±1 SEM, the tested quantity), in % responding and mean score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = table[table["cohort"] == cohort]
    control = str(rows["control"].iloc[0])
    sens = rows.loc[rows["sensitized_odor"], "odor"].tolist()
    odors = sens + sorted(set(rows["odor"]) - set(sens))
    by = rows.set_index("odor")
    own, ctrl = colours.get(cohort, INK), colours.get(control, INK_2)
    step = 0.27
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 1.25 * len(odors) + 2.0), 8.6),
                             gridspec_kw={"hspace": 0.32})
    for ax, (col, sem_col, p_col, ylab, fmt) in zip(axes, CONTROL_PANELS):
        for i in range(len(odors)):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
        ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
        for i, odor in enumerate(odors):
            r = by.loc[odor]
            ax.bar(i - step, float(r[col]), width=step * 0.86, color=_tint(own, 0.55), zorder=2)
            ax.bar(i, float(r[f"{col}_control"]), width=step * 0.86, color=_tint(ctrl, 0.55),
                   zorder=2)
            y, e = float(r[f"{col}_minus_control"]), float(r[sem_col])
            is_sens = odor in sens
            ax.bar(i + step, y, width=step * 0.86, color=own, edgecolor=INK,
                   linewidth=1.6 if is_sens else 0, zorder=2)
            if np.isfinite(e):
                ax.errorbar(i + step, y, yerr=e, fmt="none", ecolor=INK_2, elinewidth=1.0,
                            zorder=3)
            top = y + (e if np.isfinite(e) else 0) * (1 if y >= 0 else -1)
            text = fmt.format(y)
            text = "0" if text.lstrip("+-").strip("0.") == "" else text   # never "-0"
            label = text + _stars(r[p_col])
            ax.annotate(label, (i + step, top), xytext=(0, 5 if y >= 0 else -5),
                        textcoords="offset points", ha="center",
                        va="bottom" if y >= 0 else "top", fontsize=8.5, color=INK,
                        fontweight="bold" if is_sens else "normal")
        ax.set_xlim(-0.5, len(odors) - 0.5)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors, fontsize=10, color=INK)
        ax.set_ylabel(ylab, fontsize=10, color=INK)
        lo_, hi_ = ax.get_ylim()
        pad = 0.1 * (hi_ - lo_)
        ax.set_ylim(lo_ - pad, hi_ + pad)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2)
    n, nc = int(rows["n_flies"].max()), int(rows["n_flies_control"].max())
    handles = [Patch(color=_tint(own, 0.55), label=f"{cohort} change (n={n} flies)"),
               Patch(color=_tint(ctrl, 0.55), label=f"{control} change (n={nc} flies)"),
               Patch(color=own, label="Difference (±1 SEM, tested)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=3,
               frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    note = (f"Sensitized to {trained_odor(cohort)} (black outline). Difference > 0: the odor "
            f"rose more (or fell less) after exposure than in the {control} flies; "
            "SEM = the two paired SEMs in quadrature.\n* p<0.05, ** p<0.01, *** p<0.001, "
            "uncorrected permutation test of the difference in per-fly change (20,000 "
            "shuffles of flies between the two cohorts).")
    fig.text(0.5, 0.955, note, ha="center", va="top", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(top=0.85, bottom=0.07, left=0.09, right=0.98)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def pre_post_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per cohort x odor: % responding (with 95 % Wilson CI) and mean score, pre and post."""
    s = summarise(long, ["cohort", "odor"])
    if s.empty:
        return s
    for phase in ("pre", "post"):
        ci = [wilson_ci(int(k), int(n)) for k, n in zip(s[f"n_responding_{phase}"], s["n_flies"])]
        s[f"ci95_low_{phase}"] = [lo_ for lo_, _ in ci]
        s[f"ci95_high_{phase}"] = [hi_ for _, hi_ in ci]
    return s[["cohort", "odor", "sensitized_odor", "n_flies",
              "n_responding_pre", "pct_responding_pre", "ci95_low_pre", "ci95_high_pre",
              "n_responding_post", "pct_responding_post", "ci95_low_post", "ci95_high_post",
              "mean_score_pre", "sem_score_pre", "mean_score_post", "sem_score_post",
              "p_rate_mcnemar_paired", "p_score_wilcoxon"]]


def _tint(colour: str, amount: float = 0.6) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb
    return tuple(v + (1 - v) * amount for v in to_rgb(colour))


def plot_cross_cohort_pre_post(table: pd.DataFrame, cohorts: Sequence[str],
                               colours: Mapping[str, str], title: str, path: Path) -> None:
    """The levels behind the change figure: per odor, each cohort's pre-exposure (light)
    and post-exposure (solid) % responding, and mean score below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    present = [c for c in cohorts if c in set(table["cohort"])]
    odors = odor_order(table, present)
    k = len(present)
    slot = 0.86 / k
    width = slot * 0.42
    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 1.9 * len(odors) + 2.0), 9.2),
                             gridspec_kw={"hspace": 0.32})
    panels = [("pct_responding", "p_rate_mcnemar_paired",
               f"% of flies responding\n(score ≥ {THRESHOLD})"),
              ("mean_score", "p_score_wilcoxon", "Mean PER score")]
    by = table.set_index(["cohort", "odor"])
    for ax, (value, p_col, ylab) in zip(axes, panels):
        for i in range(len(odors)):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
        tops = []
        for j, c in enumerate(present):
            for i, odor in enumerate(odors):
                if (c, odor) not in by.index:
                    continue
                r = by.loc[(c, odor)]
                centre = i + (j - (k - 1) / 2) * slot
                sens = bool(r["sensitized_odor"])
                top = -np.inf
                for phase, x, face in (("pre", centre - width / 2, _tint(colours[c])),
                                       ("post", centre + width / 2, colours[c])):
                    y = float(r[f"{value}_{phase}"])
                    if value == "pct_responding":
                        lo_, hi_ = float(r[f"ci95_low_{phase}"]), float(r[f"ci95_high_{phase}"])
                    else:
                        e = float(r[f"sem_score_{phase}"])
                        lo_, hi_ = (y - e, y + e) if np.isfinite(e) else (y, y)
                    ax.bar(x, y, width=width * 0.94, color=face, zorder=2,
                           edgecolor=INK if sens else colours[c], linewidth=1.5 if sens else 0.8)
                    ax.errorbar(x, y, yerr=[[y - lo_], [hi_ - y]], fmt="none", ecolor=INK_2,
                                elinewidth=0.9, zorder=3)
                    top = max(top, hi_)
                tops.append(top)
                star = _stars(r[p_col])
                if star:
                    ax.annotate(star, (centre, top), xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=9, color=INK)
        ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
        ax.set_xlim(-0.5, len(odors) - 0.5)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors, fontsize=10, color=INK)
        ax.set_ylabel(ylab, fontsize=10, color=INK)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(min(lo_, 0), max(hi_, max(tops) if tops else hi_) * 1.08)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2)
    n = table.groupby("cohort")["n_flies"].max()
    handles = [Patch(color=colours[c], label=f"{c}  (n={int(n[c])} flies, exposed to "
                                              f"{trained_odor(c)})") for c in present]
    handles += [Patch(facecolor=_tint(INK_2), edgecolor=INK_2, label=f"{PRE_EXPOSURE} (light bar)"),
                Patch(facecolor=INK_2, edgecolor=INK_2, label=f"{POST_EXPOSURE} (solid bar)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3,
               frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    fig.text(0.5, 0.962, "The same flies before and after exposure. Black outline = the cohort's "
             "own exposed odor. Whiskers: 95% Wilson CI (top), ±1 SEM over flies (bottom). "
             "* p<0.05, ** p<0.01, *** p<0.001, uncorrected (McNemar on rates, Wilcoxon on scores).",
             ha="center", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(top=0.85, bottom=0.06, left=0.08, right=0.985)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


PEAKZ_DIR = "PeakZ-Before-vs-During"
#: (name in the table, column stem in a cohort's peakz_per_fly_change.csv)
PEAKZ_RATIOS = (("signed_ratio", "zpeak_ratio_signed"),
                ("unsigned_ratio", "zpeak_ratio_unsigned"))
#: The odor window's own peak z, not divided by the before window's (user 2026-11-11).
PEAKZ_DURING = (("signed_during", "zpeak_during_signed"),
                ("unsigned_during", "zpeak_during_unsigned"))
PEAKZ_PANELS = (("signed_ratio_change", "sem_signed_ratio_change", "p_signed_ratio_wilcoxon",
                 "Change in signed peak-z ratio\n(during / |before|, post − pre)", "{:+.2f}"),
                ("unsigned_ratio_change", "sem_unsigned_ratio_change", "p_unsigned_ratio_wilcoxon",
                 "Change in unsigned peak-z ratio\n(|during| / |before|, post − pre)", "{:+.2f}"))
PEAKZ_NOTE = ("Bars: mean of each fly's own post − pre, as on each cohort's PeakZ figure; whiskers "
              "±1 SEM over flies. Black outline = the cohort's own sensitized odor (value shown). "
              "* p<0.05, ** p<0.01, *** p<0.001, Wilcoxon paired by fly, uncorrected.")
PEAKZ_DURING_PANELS = (
    ("signed_during_change", "sem_signed_during_change", "p_signed_during_wilcoxon",
     "Change in signed peak z during the odor\n(post − pre, SD of the before window)",
     "{:+.2f}"),
    ("unsigned_during_change", "sem_unsigned_during_change", "p_unsigned_during_wilcoxon",
     "Change in unsigned peak z during the odor\n(post − pre, SD of the before window)",
     "{:+.2f}"))


def peakz_change_table(out: Path, cohorts: Sequence[str],
                       measures: Sequence[tuple[str, str]] = PEAKZ_RATIOS) -> pd.DataFrame:
    """Per cohort x odor: post - pre of peak-z ``measures``, by default the during/before
    ratio, signed and unsigned (``PEAKZ_DURING``: the odor window's own peak z).

    Read from each cohort's own ``PeakZ-Before-vs-During/peakz_per_fly_change.csv``
    (window flies, one trial per fly per odor per phase), so these are the numbers
    that cohort's PeakZ figure draws: the mean of each fly's own change and the SEM over
    flies, with the shared exact, tie-aware Wilcoxon signed-rank p of those changes
    (``fly_stats.wilcoxon_signed_rank_p``), uncorrected. A fly missing a phase has no
    change and is left out, and so is a cohort without the table. The sensitized odor
    is the cohort's own (``trained_odor``): the table's ``trained`` column is blank on
    every row missing a phase."""
    rows = []
    for cohort in cohorts:
        path = out / cohort / PEAKZ_DIR / "peakz_per_fly_change.csv"
        if not path.is_file():
            continue
        t = pd.read_csv(path)
        if not {f"{stem}_{ph}" for _, stem in measures for ph in ("pre", "post")} <= set(t.columns):
            continue                     # a table from before these measures existed
        sensitized = trained_odor(cohort)
        for odor, g in t.groupby("odor", sort=True):
            vals = {(name, phase): pd.to_numeric(g[f"{stem}_{phase}"], errors="coerce")
                    .to_numpy(float) for name, stem in measures for phase in ("pre", "post")}
            ok = np.logical_and.reduce([np.isfinite(v) for v in vals.values()])
            if not ok.any():
                continue
            row = {"cohort": cohort, "odor": odor, "sensitized_odor": odor == sensitized,
                   "n_flies": int(ok.sum())}
            for name, _ in measures:
                pre, post = vals[(name, "pre")][ok], vals[(name, "post")][ok]
                change = post - pre
                p = fs.wilcoxon_signed_rank_p(change)       # None: no fly changed
                row.update({
                    f"{name}_pre": float(pre.mean()),
                    f"{name}_post": float(post.mean()),
                    f"{name}_change": float(change.mean()),
                    f"sem_{name}_change": _sem(pd.Series(change)),
                    f"median_{name}_change": float(np.median(change)),
                    f"p_{name}_wilcoxon": np.nan if p is None else p,
                })
            rows.append(row)
    return pd.DataFrame(rows)


ALL_FLIES = "All flies"
#: The odor of the first-vs-last-part figure (user 2026-11-11: the one every cohort
#: meets, and the one IAA-Sensitivity is exposed to).
HALVES_ODOR = "Isoamyl Acetate"
HALVES = (("first", "first 15 s"), ("second", "last 15 s"))
#: The responders' measures per half (user 2026-11-11: AUC as during / before, peak z as
#: during / before and during alone): (column stem in the flies CSV, y label, no-change).
HALF_PANELS = (
    ("auc_ratio", "Responders' AUC, during / before\n(raw area per second; 1 = no change)",
     1.0),
    ("zpeak_ratio", "Responders' peak z, during / |before|\n(signed; noise lands near ±1)",
     0.0),
    ("zpeak", "Responders' peak z during\n(signed, SD of the before window)", 0.0),
)


def odor_halves_flies(out: Path, cohorts: Sequence[str], odor: str = HALVES_ODOR
                      ) -> pd.DataFrame:
    """``odor``'s rows of each cohort's own cumulative-responders flies table (score,
    responding, start time, area in each half), plus the cohort and a pre/post key.
    A cohort without the table, or with one from before the halves, is left out."""
    import scripts.analysis.cumulative_responders as cr

    frames = []
    for cohort in cohorts:
        path = out / cohort / cr.FOLDER / f"{cr.TAG}_flies.csv"
        if not path.is_file():
            continue
        t = pd.read_csv(path)
        t = t[t["odor"] == odor]
        if t.empty or not set(cr.HALF_COLUMNS) <= set(t.columns):
            continue
        keys = {label: key for key, label in cr.phase_labels(cohort).items()}
        frames.append(t.assign(cohort=cohort, phase_key=t["phase"].map(keys)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _mann_whitney(a, b) -> float:
    from scipy.stats import mannwhitneyu
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(float)
    if a.size < 2 or b.size < 2:
        return np.nan
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def odor_halves_table(flies: pd.DataFrame, cohorts: Sequence[str]) -> pd.DataFrame:
    """Per cohort, then every fly pooled (``ALL_FLIES``: all of its flies, never a mean of
    the cohorts' %), pre and post: the responders (model score >= the score bars'
    threshold) that started in the first 15 s of the odor and in the rest (start: the
    first 0.5 s above the red line, as on the cumulative-responders figure), as % of
    every fly; responders with no clear start are counted apart. Then the responders'
    mean area in each half. p, uncorrected: McNemar on starting in a half (the same
    flies pre and post), Mann-Whitney on area and on start time (responders differ)."""
    import scripts.analysis.cumulative_responders as cr

    key = ["cohort", "fly", "fly_number"]
    present = set(flies["cohort"]) if "cohort" in flies.columns else set()
    groups = [c for c in cohorts if c in present]
    rows = []
    for group in groups + ([ALL_FLIES] if groups else []):
        g = flies if group == ALL_FLIES else flies[flies["cohort"] == group]
        by = {k: g[g["phase_key"] == k].sort_values(key) for k in ("pre", "post")}
        if by["pre"][key].values.tolist() != by["post"][key].values.tolist():
            raise RuntimeError(f"{group}: the two phases do not hold the same flies")
        row: dict = {"group": group, "n_flies": len(by["pre"])}
        started, onsets, resp = {}, {}, {}
        for k, d in by.items():
            resp[k] = d["responding"].astype(bool).to_numpy()
            onset = pd.to_numeric(d["onset_s"], errors="coerce").to_numpy(float)
            timed = resp[k] & np.isfinite(onset)
            onsets[k] = onset[timed]
            started[(k, "first")] = timed & (onset < cr.HALF_S)
            started[(k, "second")] = timed & (onset >= cr.HALF_S)
            row[f"n_responding_{k}"] = int(resp[k].sum())
            row[f"n_no_onset_{k}"] = int((resp[k] & ~np.isfinite(onset)).sum())
            for half, _ in HALVES:
                row[f"n_{half}_{k}"] = int(started[(k, half)].sum())
                row[f"pct_{half}_{k}"] = (100.0 * float(started[(k, half)].mean())
                                          if resp[k].size else np.nan)
                for measure, _, _ in HALF_PANELS:
                    v = pd.to_numeric(d.loc[resp[k], f"{measure}_{half}_half"],
                                      errors="coerce").dropna()
                    row[f"{measure}_{half}_{k}"] = float(v.mean()) if len(v) else np.nan
                    row[f"sem_{measure}_{half}_{k}"] = _sem(v)
                    row[f"n_{measure}_{half}_{k}"] = int(len(v))
            row[f"median_onset_{k}_s"] = (float(np.median(onsets[k])) if onsets[k].size
                                          else np.nan)
        for half, _ in HALVES:
            p = pvt.mcnemar_p(started[("pre", half)], started[("post", half)])
            row[f"p_{half}_mcnemar"] = np.nan if p is None else float(p)
            for measure, _, _ in HALF_PANELS:
                row[f"p_{measure}_{half}_mannwhitney"] = _mann_whitney(
                    by["pre"].loc[resp["pre"], f"{measure}_{half}_half"],
                    by["post"].loc[resp["post"], f"{measure}_{half}_half"])
        row["p_onset_mannwhitney"] = _mann_whitney(onsets["pre"], onsets["post"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_odor_halves(table: pd.DataFrame, flies: pd.DataFrame, *, odor: str, title: str,
                     path: Path) -> None:
    """Two panels, a group per cohort plus every fly: in each half of the odor, the %
    of flies that started responding there, then the responders' area there."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colour = {"pre": pvt.PRE_COLOR, "post": pvt.POST_COLOR}
    groups = table["group"].tolist()
    half_dx, phase_dx, width = 0.22, 0.085, 0.15
    rng = np.random.default_rng(0)
    panels = [("pct", "p_{}_mcnemar", "Flies that started responding\n(% of all flies)", None)]
    panels += [(m, f"p_{m}_{{}}_mannwhitney", ylab, null) for m, ylab, null in HALF_PANELS]
    height = 3.3 * len(panels) + 1.9            # inches: the panels, the header, the footer
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(max(9.0, 2.7 * len(groups) + 1.6), height),
                             gridspec_kw={"hspace": 0.62})
    for ax, (panel, p_col, ylab, null) in zip(axes, panels):
        for i in range(len(groups)):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
        ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
        if null not in (None, 0.0):
            ax.axhline(null, color=INK_2, lw=0.9, ls=(0, (3, 2)), zorder=1)
        for i, r in enumerate(table.itertuples(index=False)):
            g = flies if r.group == ALL_FLIES else flies[flies["cohort"] == r.group]
            for h, (half, _) in enumerate(HALVES):
                centre = i + (2 * h - 1) * half_dx
                tops = []
                for j, k in enumerate(("pre", "post")):
                    x = centre + (2 * j - 1) * phase_dx
                    if panel == "pct":
                        y, e = float(getattr(r, f"pct_{half}_{k}")), np.nan
                        dots = np.empty(0)
                    else:
                        y = float(getattr(r, f"{panel}_{half}_{k}"))
                        e = float(getattr(r, f"sem_{panel}_{half}_{k}"))
                        mine = g[(g["phase_key"] == k) & g["responding"].astype(bool)]
                        dots = pd.to_numeric(mine[f"{panel}_{half}_half"],
                                             errors="coerce").dropna().to_numpy(float)
                    if not np.isfinite(y):
                        continue
                    ax.bar(x, y, width=width, color=colour[k], zorder=2, lw=0)
                    if np.isfinite(e):
                        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=INK, elinewidth=1.0,
                                    zorder=4)
                    if dots.size:
                        ax.scatter(x + rng.uniform(-0.3, 0.3, dots.size) * width, dots, s=5,
                                   color=INK_2, alpha=0.45, lw=0, zorder=3)
                    tops.append(max([y + (e if np.isfinite(e) else 0.0),
                                     *(dots.tolist() or [y])]))
                star = _stars(getattr(r, p_col.format(half)))
                if star and tops:
                    ax.annotate(star, (centre, max(tops)), xytext=(0, 3),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=10, color=INK)
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_xticks([i + d for i in range(len(groups)) for d in (-half_dx, half_dx)])
        ax.set_xticklabels([label for _ in groups for _, label in HALVES], fontsize=8.5,
                           color=INK_2)
        for i, r in enumerate(table.itertuples(index=False)):
            sub = ("every cohort pooled" if r.group == ALL_FLIES
                   else f"exposed to {trained_odor(r.group)}")
            exposed = r.group != ALL_FLIES and trained_odor(r.group) == odor
            ax.text(i, -0.13, f"{r.group}\n{sub}, n={int(r.n_flies)} flies",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9,
                    color=INK, fontweight="bold" if exposed else "normal")
        ax.set_ylabel(ylab, fontsize=10, color=INK)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(min(lo_, 0), hi_ + 0.06 * (hi_ - min(lo_, 0)))
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2)
    fig.legend(handles=[Patch(color=colour["pre"], label=PRE_EXPOSURE),
                        Patch(color=colour["post"], label=POST_EXPOSURE)],
               loc="upper right", bbox_to_anchor=(0.985, 1 - 0.08 / height), ncol=2,
               frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle(title, fontsize=12.5, color=INK, x=0.07, ha="left", y=1 - 0.1 / height)
    fig.text(0.07, 1 - 0.45 / height,
             "Panel 1: flies the model scores as responding (as on the score bars), by when they "
             "started (their first 0.5 s above the red line, as on Cumulative-Responders), as % "
             "of every fly; responders with no clear\nstart are in neither part (they are in the "
             "CSV). Panels 2-4: each responder's measure in that part of the odor, as each "
             "cohort's AUC and PeakZ figures define it, both parts over the same before "
             "window;\nmean ± SEM, dots = flies. * p<0.05, ** p<0.01, *** p<0.001, uncorrected: "
             "McNemar (panel 1, the same flies pre and post), Mann-Whitney (panels 2-4, the "
             "responders differ). Bold: the cohort exposed to this odor.",
             fontsize=8.4, color=INK_2, va="top")
    fig.subplots_adjust(top=1 - 1.3 / height, bottom=0.85 / height, left=0.08, right=0.985)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def write_cross_cohort(out: Path, long: pd.DataFrame, cohorts: Sequence[str], *,
                       window: str, control: str | None = None) -> list[Path]:
    """``Cross-Cohort/``: every odor of every cohort on one axis, window flies only.

    With ``control`` set, also each cohort's change minus that cohort's change.
    The peak-z ratio figure reads each cohort's own PeakZ table from ``out``, so it is
    drawn after the cohorts are built, and only for those that have one."""
    staging = _staging(out / "Cross-Cohort")
    staging.mkdir(parents=True)
    rows = long[long["in_window"]]
    written = []
    if not rows.empty:
        table = change_table(rows)
        table.to_csv(staging / "cross_cohort_change_window.csv", index=False)
        png = staging / "cross_cohort_change_window.png"
        plot_cross_cohort(table, list(cohorts), cohort_colours(cohorts),
                          f"Pre- to post-exposure change, every odor: flies started {window}",
                          png)
        written.append(png)
        minus = control_subtracted_table(rows, control)
        if not minus.empty:
            minus.to_csv(staging / "change_minus_control_window.csv", index=False)
            for cohort in [c for c in cohorts if c in set(minus["cohort"])]:
                png = staging / control_subtracted_name(cohort)
                plot_control_subtracted(minus, cohort, cohort_colours(cohorts),
                                        f"{cohort}: pre- to post-exposure change minus "
                                        f"{control}'s, every odor: flies started {window}",
                                        png)
                written.append(png)
        levels = pre_post_table(rows)
        levels.to_csv(staging / "cross_cohort_pre_post_window.csv", index=False)
        png = staging / "cross_cohort_pre_post_window.png"
        plot_cross_cohort_pre_post(levels, list(cohorts), cohort_colours(cohorts),
                                   f"Pre- and post-exposure, every odor: flies started {window}",
                                   png)
        written.append(png)
        peakz = peakz_change_table(out, list(cohorts))
        if not peakz.empty:
            peakz.to_csv(staging / "cross_cohort_peakz_ratio_change_window.csv", index=False)
            png = staging / "cross_cohort_peakz_ratio_change_window.png"
            plot_cross_cohort(peakz, list(cohorts), cohort_colours(cohorts),
                              "Pre- to post-exposure change in the PER-trace peak z-score, "
                              f"during / before, every odor: flies started {window}", png,
                              panels=PEAKZ_PANELS, note=PEAKZ_NOTE)
            written.append(png)
        during = peakz_change_table(out, list(cohorts), measures=PEAKZ_DURING)
        if not during.empty:
            during.to_csv(staging / "cross_cohort_peakz_during_change_window.csv", index=False)
            png = staging / "cross_cohort_peakz_during_change_window.png"
            plot_cross_cohort(during, list(cohorts), cohort_colours(cohorts),
                              "Pre- to post-exposure change in the PER-trace peak z-score "
                              f"during the odor, every odor: flies started {window}", png,
                              panels=PEAKZ_DURING_PANELS, note=PEAKZ_NOTE)
            written.append(png)
        halves = odor_halves_flies(out, list(cohorts))
        if not halves.empty:
            halves_table = odor_halves_table(halves, list(cohorts))
            stem = f"cross_cohort_{HALVES_ODOR.lower().replace(' ', '_')}_halves_window"
            halves_table.to_csv(staging / f"{stem}.csv", index=False)
            png = staging / f"{stem}.png"
            plot_odor_halves(halves_table, halves, odor=HALVES_ODOR,
                             title=f"{HALVES_ODOR}, first vs last 15 s of the odor, pre- vs "
                                   f"post-exposure: flies started {window}", path=png)
            written.append(png)
    _swap_in(staging, out / "Cross-Cohort")
    return [out / "Cross-Cohort" / p.name for p in written]


# ---------------------------------------------------------------------------
# Pre-exposure baseline across cohorts and rigs
# ---------------------------------------------------------------------------

# Before exposure every cohort is the same naive fly on the same panel, so any
# cohort or rig difference here is a batch/rig effect, not biology.
ALL_ODORS = "All odors"
# Rig slots 5-7 of the same palette, never a cohort's colour (all-pairs validated:
# CVD dE 17.6 protan / 7.6 tritan, so the markers carry identity too).
RIG_PALETTE = {1: "#e87ba4", 2: "#008300", 3: "#6250d6"}
RIG_MARKERS = {1: "o", 2: "s", 3: "D"}


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson interval for k of n, in percent."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * max(0.0, centre - half), 2), round(100 * min(1.0, centre + half), 2))


def baseline_table(long: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Pre-exposure % responding per group x odor, plus an all-odor row per group.

    The all-odor row pools trials (responses / flies x odors), never a mean of the
    per-odor percentages, and says so in ``n_trials``."""
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in long.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        parts = [(o, go) for o, go in g.groupby("odor", sort=True)] + [(ALL_ODORS, g)]
        for odor, go in parts:
            k, n = int(go["responded_pre"].sum()), len(go)
            lo, hi = wilson_ci(k, n)
            s = go["score_pre"].astype(float).dropna()
            mean = float(s.mean()) if len(s) else np.nan
            # 95 % t interval on the trial scores (pooled trials, as the mean is)
            half = (float(student_t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s)))
                    if len(s) > 1 else np.nan)
            rows.append({**dict(zip(by, key)), "odor": odor,
                         "n_flies": int(go["fly_id"].nunique()), "n_trials": n,
                         "n_responding": k, "pct_responding": round(100 * k / n, 2),
                         "ci95_lo": lo, "ci95_hi": hi,
                         "mean_score_pre": round(mean, 4),
                         "score_ci95_lo": round(mean - half, 4),
                         "score_ci95_hi": round(mean + half, 4)})
    return pd.DataFrame(rows)


def baseline_stats(long: pd.DataFrame, by: str) -> pd.DataFrame:
    """Do the groups differ before exposure? One row per odor, all odors first.

    Per odor (one trial per fly): Fisher-Freeman-Halton on responded, Kruskal-
    Wallis on score. All odors: one value per fly (responded to any odor; its
    pre score averaged over the panel), so no fly counts seven times."""
    from scripts.analysis.starvation_pre_post_comparison import _ffh, _kw

    groups = sorted(long[by].dropna().unique())
    flies = (long.groupby([by, "fly_id"])
             .agg(any_resp=("responded_pre", "any"), score=("score_pre", "mean"))
             .reset_index())
    rows = [{"compared": by, "odor": ALL_ODORS, "n_groups": len(groups),
             "n_flies": len(flies),
             "p_rate_ffh": _ffh([flies.loc[flies[by] == g, "any_resp"].to_numpy()
                                 for g in groups]) if len(groups) > 1 else None,
             "p_score_kw": _kw([flies.loc[flies[by] == g, "score"].to_numpy(dtype=float)
                                for g in groups]) if len(groups) > 1 else None}]
    for odor, go in long.groupby("odor", sort=True):
        present = [g for g in groups if (go[by] == g).any()]
        rows.append({"compared": by, "odor": odor, "n_groups": len(present),
                     "n_flies": int(go["fly_id"].nunique()),
                     "p_rate_ffh": _ffh([go.loc[go[by] == g, "responded_pre"].to_numpy()
                                         for g in present]) if len(present) > 1 else None,
                     "p_score_kw": _kw([go.loc[go[by] == g, "score_pre"].to_numpy(dtype=float)
                                        for g in present]) if len(present) > 1 else None})
    return pd.DataFrame(rows)


def _p_text(p) -> str:
    p = float(p) if p is not None else np.nan
    return "" if not np.isfinite(p) else "p<0.001" if p < 0.001 else f"p={p:.2f}"


def _baseline_panel(ax, table: pd.DataFrame, key: str, groups: Sequence, odors: Sequence[str],
                    style: Mapping, stats: pd.DataFrame, overall: pd.DataFrame,
                    title: str, metric: str = "pct", phase: str = "Pre-exposure") -> None:
    """One baseline panel. ``metric="pct"``: % responding, Wilson CI, rate p.
    ``metric="score"``: mean model score, t CI, score p (same tables)."""
    value, lo_col, hi_col, p_col, ylabel = {
        "pct": ("pct_responding", "ci95_lo", "ci95_hi", "p_rate_ffh",
                f"{phase}\n% of flies responding"),
        "score": ("mean_score_pre", "score_ci95_lo", "score_ci95_hi", "p_score_kw",
                  f"{phase}\nmean score"),
    }[metric]
    k = len(groups)
    step = 0.78 / max(k, 1)
    by = table.set_index([key, "odor"])
    ptab = stats.set_index("odor")
    ov = overall.set_index("odor")
    hi_all = float(pd.concat([table[hi_col], table[value]]).max())
    lo_all = float(pd.concat([table[lo_col], table[value]]).min())
    if metric == "pct":
        bottom, top, pad = 0.0, hi_all + 14, 3.0
    else:
        bottom = min(0.0, lo_all - 0.1)
        span = max(hi_all - bottom, 1.0)
        top, pad = hi_all + 0.2 * span, 0.05 * span
    for i, odor in enumerate(odors):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
        ref = float(ov.loc[odor, value])
        ax.plot([i - 0.42, i + 0.42], [ref, ref], color=INK_2, lw=1.2, ls=(0, (3, 2)), zorder=1)
        p = ptab.loc[odor, p_col] if odor in ptab.index else None
        sig = p is not None and np.isfinite(float(p)) and float(p) < 0.05
        ax.text(i, top - pad, _p_text(p), ha="center", va="top", fontsize=9, color=INK,
                fontweight="bold" if sig else "normal")
        for j, g in enumerate(groups):
            if (g, odor) not in by.index:
                continue
            r = by.loc[(g, odor)]
            x = i + (j - (k - 1) / 2) * step
            colour, marker = style[g]
            ax.plot([x, x], [r[lo_col], r[hi_col]], color=colour, lw=1.4, alpha=0.6,
                    zorder=2, solid_capstyle="round")
            ax.plot(x, r[value], marker=marker, ms=8, color=colour, mec="white",
                    mew=1.0, zorder=3)
    if bottom < 0:
        ax.axhline(0, color=INK_2, lw=0.8, zorder=1)
    ax.axvline(len(odors) - 1.5, color=INK_2, lw=0.8)
    ax.set_xlim(-0.5, len(odors) - 0.5)
    ax.set_ylim(bottom, top)
    ax.set_xticks(range(len(odors)))
    ax.set_xticklabels([f"{o}\n(pooled trials)" if o == ALL_ODORS else o for o in odors],
                       fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=26)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def cohort_rig_cells(tables: Mapping[str, pd.DataFrame], present: Sequence[str],
                     rigs: Sequence[int], metric: str = "pct"):
    """Panel C's cells: per cohort, each rig then all rigs, all odors pooled over trials.
    ``metric="pct"``: % responding; ``"score"``: mean score. Returns (values, notes, columns)."""
    col, fmt = {"pct": ("pct_responding", "{:.0f}%"),
                "score": ("mean_score_pre", "{:.2f}")}[metric]
    grid = tables["pre_exposure_by_cohort_x_rig.csv"]
    grid = grid[grid["odor"] == ALL_ODORS].set_index(["cohort", "rig"])
    allr = tables["pre_exposure_by_cohort.csv"]
    allr = allr[allr["odor"] == ALL_ODORS].set_index("cohort")
    cols = [f"Rig {r}" for r in rigs] + ["All rigs"]
    vals = np.full((len(present), len(cols)), np.nan)
    notes = [[""] * len(cols) for _ in present]
    for i, c in enumerate(present):
        cells = [grid.loc[(c, r)] if (c, r) in grid.index else None for r in rigs]
        for j, g in enumerate(cells + [allr.loc[c]]):
            if g is None:
                notes[i][j] = "no flies"
                continue
            vals[i, j] = float(g[col])
            notes[i][j] = f"{fmt.format(g[col])}\n{int(g['n_flies'])} flies"
    return vals, notes, cols


def plot_pre_baseline(long: pd.DataFrame, cohorts: Sequence[str], tables: Mapping[str, pd.DataFrame],
                      title: str, path: Path, metric: str = "pct") -> None:
    """Panels A (by cohort), B (by rig), C (cohort x rig). ``metric="pct"``: % responding;
    ``"score"``: mean model score, t CI, Kruskal-Wallis (same tables)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    present = [c for c in cohorts if c in set(long["cohort"])]
    rigs = sorted(int(r) for r in long["rig"].dropna().unique())
    odors = sorted(set(long["odor"])) + [ALL_ODORS]
    overall = baseline_table(long.assign(_all=1), "_all")
    colours = cohort_colours(cohorts)
    cstyle = {c: (colours[c], COHORT_MARKERS[cohorts.index(c) % len(COHORT_MARKERS)])
              for c in present}
    rstyle = {r: (RIG_PALETTE.get(r, INK_2), RIG_MARKERS.get(r, "o")) for r in rigs}
    stats = tables["pre_exposure_stats.csv"]

    fig = plt.figure(figsize=(max(11.0, 1.5 * len(odors) + 2.5), 13.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.62], hspace=0.5)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    _baseline_panel(ax1, tables["pre_exposure_by_cohort.csv"], "cohort", present, odors,
                    cstyle, stats[stats["compared"] == "cohort"], overall, "A  By cohort",
                    metric=metric)
    nc = long.groupby("cohort")["fly_id"].nunique()
    ax1.legend(handles=[Line2D([], [], ls="none", marker=cstyle[c][1], color=cstyle[c][0],
                               mec="white", ms=8, label=f"{c} (n={nc[c]})") for c in present]
               + [Line2D([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)), label="all flies")],
               loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(present) + 1,
               frameon=False, fontsize=9, labelcolor=INK, borderaxespad=0.1)
    _baseline_panel(ax2, tables["pre_exposure_by_rig.csv"], "rig", rigs, odors, rstyle,
                    stats[stats["compared"] == "rig"], overall, "B  By rig", metric=metric)
    nr = long.groupby("rig")["fly_id"].nunique()
    ax2.legend(handles=[Line2D([], [], ls="none", marker=rstyle[r][1], color=rstyle[r][0],
                               mec="white", ms=8, label=f"Rig {r} (n={nr[r]})") for r in rigs]
               + [Line2D([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)), label="all flies")],
               loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(rigs) + 1,
               frameon=False, fontsize=9, labelcolor=INK, borderaxespad=0.1)

    # C: cohort x rig, all odors pooled, so a rig effect can't pass for a cohort one
    vals, notes, cols = cohort_rig_cells(tables, present, rigs, metric)
    floor = 50.0 if metric == "pct" else 1.5
    vmax = max(floor, float(np.nanmax(vals)) if np.isfinite(vals).any() else floor)
    cmap = plt.get_cmap("Blues")
    ax3.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=0, vmax=vmax * 1.25, aspect="auto")
    for i in range(len(present)):
        for j in range(len(cols)):
            v = vals[i, j]
            dark = np.isfinite(v) and v / (vmax * 1.25) > 0.55
            ax3.text(j, i, notes[i][j], ha="center", va="center", fontsize=9,
                     color="white" if dark else (INK if np.isfinite(v) else INK_2))
    ax3.axvline(len(rigs) - 0.5, color="white", lw=3)
    ax3.set_xticks(range(len(cols)))
    ax3.set_xticklabels(cols, fontsize=10, color=INK)
    ax3.set_yticks(range(len(present)))
    ax3.set_yticklabels(present, fontsize=10, color=INK)
    ax3.tick_params(length=0)
    for side in ax3.spines.values():
        side.set_visible(False)
    what = ("% of fly x odor trials responding" if metric == "pct"
            else "mean score over fly x odor trials")
    ax3.set_title(f"C  Cohort x rig, all odors pooled ({what})",
                  loc="left", fontsize=11.5, color=INK, pad=6)

    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    how = ("Whiskers: 95% Wilson CI. Dashed line: all flies together.\n"
           "p above each odor: Fisher-Freeman-Halton across groups, uncorrected; "
           "the all-odors p uses one value per fly (responded to any odor)."
           if metric == "pct" else
           "Whiskers: 95% t CI over trials. Dashed line: all flies together.\n"
           "p above each odor: Kruskal-Wallis across groups, uncorrected; "
           "the all-odors p uses one value per fly (its pre score over the panel).")
    fig.text(0.5, 0.975, "Before exposure every fly is naive and gets the same panel, so these "
             f"should match. {how}",
             ha="center", va="top", fontsize=9, color=INK_2, linespacing=1.4)
    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.17, right=0.98)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def write_pre_baseline(out: Path, long: pd.DataFrame, cohorts: Sequence[str], *,
                       window: str) -> list[Path]:
    """``Pre-Exposure-Baseline/``: are the naive pre-tests the same across cohorts and rigs?"""
    rows = long[long["in_window"]]
    if rows.empty:
        shutil.rmtree(out / "Pre-Exposure-Baseline", ignore_errors=True)
        return []
    staging = _staging(out / "Pre-Exposure-Baseline")
    staging.mkdir(parents=True)
    tables = {
        "pre_exposure_by_cohort.csv": baseline_table(rows, "cohort"),
        "pre_exposure_by_rig.csv": baseline_table(rows, "rig"),
        "pre_exposure_by_cohort_x_rig.csv": baseline_table(rows, ["cohort", "rig"]),
        "pre_exposure_stats.csv": pd.concat([baseline_stats(rows, "cohort"),
                                             baseline_stats(rows, "rig")], ignore_index=True),
    }
    for name, df in tables.items():
        df.to_csv(staging / name, index=False)
    pngs = [staging / "pre_exposure_baseline.png", staging / "pre_exposure_baseline_score.png"]
    plot_pre_baseline(rows, list(cohorts), tables,
                      f"Pre-exposure baseline, every cohort and rig: flies started {window}",
                      pngs[0])
    # the same flies as mean score (user 2026-11-17)
    plot_pre_baseline(rows, list(cohorts), tables,
                      f"Pre-exposure baseline, mean score, every cohort and rig: flies started "
                      f"{window}", pngs[1], metric="score")
    _swap_in(staging, out / "Pre-Exposure-Baseline")
    return [out / "Pre-Exposure-Baseline" / p.name for p in pngs]


# ---------------------------------------------------------------------------
# Before vs from a split date
# ---------------------------------------------------------------------------

ALL_COHORTS = "All cohorts"
# Earlier period recedes (neutral gray), the later one carries the hue; the
# markers differ too, so the pair never rests on colour alone.
PERIOD_STYLE = (("#7d7c78", "o"), ("#e34948", "s"))


def period_label(start_time, split: dt.date) -> str:
    day = f"{split:%b} {split.day}"
    return f"Before {day}" if pd.Timestamp(start_time).date() < split else f"{day} onward"


def plot_pre_vs_post_by_period(table: pd.DataFrame, cohort: str, periods: Sequence[str],
                               style: Mapping, title: str, note: str, path: Path) -> None:
    """Per odor, one pre -> post dumbbell per period (hollow = pre, filled = post)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    odors = sorted(set(table["odor"]))
    here = [p_ for p_ in periods if (table["period"] == p_).any()]
    by = table.set_index(["period", "odor"])
    k = len(here)
    step = 0.5 / max(k, 1)
    fig, ax = plt.subplots(figsize=(max(11.0, 1.5 * len(odors) + 2.5), 6.4))
    top = 100.0
    for i, odor in enumerate(odors):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
        for j, p_ in enumerate(here):
            if (p_, odor) not in by.index:
                continue
            r = by.loc[(p_, odor)]
            x = i + (j - (k - 1) / 2) * step * 1.6
            colour, marker = style[p_]
            y0, y1 = float(r["pct_responding_pre"]), float(r["pct_responding_post"])
            ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0,
                                        shrinkA=5, shrinkB=5, mutation_scale=12))
            ax.plot(x, y0, marker=marker, ms=8, mfc="white", mec=colour, mew=1.8, zorder=3)
            ax.plot(x, y1, marker=marker, ms=8, color=colour, mec="white", mew=1.0, zorder=3)
            pv = r["p_rate_mcnemar_paired"]
            sig = pv is not None and np.isfinite(float(pv)) and float(pv) < 0.05
            ax.text(x, max(y0, y1) + 4, _p_text(pv).replace("p=", ""), ha="center",
                    va="bottom", fontsize=8.5, color=INK,
                    fontweight="bold" if sig else "normal")
    ax.set_xlim(-0.5, len(odors) - 0.5)
    ax.set_ylim(0, top)
    ax.set_xticks(range(len(odors)))
    ax.set_xticklabels(odors, fontsize=10, color=INK)
    for t in ax.get_xticklabels():
        if t.get_text() == trained_odor(cohort):
            t.set_fontweight("bold")
    ax.set_ylabel("% of flies responding", fontsize=10, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    n = table.groupby("period")["n_flies"].max().to_dict()
    handles = [Line2D([], [], color=style[p_][0], marker=style[p_][1], ms=8, lw=2.0,
                      mec="white", label=f"{p_} (n={int(n.get(p_, 0))} flies)") for p_ in here]
    handles += [Line2D([], [], ls="none", marker="o", ms=8, mfc="white", mec=INK_2, mew=1.8,
                       label="pre-exposure"),
                Line2D([], [], ls="none", marker="o", ms=8, color=INK_2, mec="white",
                       label="post-exposure")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles),
              frameon=False, fontsize=9, labelcolor=INK, borderaxespad=0.1)
    fig.suptitle(title, fontsize=13, color=INK, y=0.985)
    fig.text(0.5, 0.935, note, ha="center", va="top", fontsize=9, color=INK_2,
             linespacing=1.4)
    fig.subplots_adjust(top=0.83, bottom=0.08, left=0.08, right=0.98)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def write_date_split(out: Path, long: pd.DataFrame, cohorts: Sequence[str], *,
                     split: dt.date, window: str, panels: Sequence[str] = ()) -> list[Path]:
    """``Before-vs-From-<date>/``: naive pre-exposure of window flies started before vs
    from ``split``: every fly pooled, then each cohort named in ``panels``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    name = f"Before-vs-From-{split.isoformat()}"
    for old in out.glob("Before-vs-From-*"):   # a moved split date leaves no stale set
        if old.name != name and old.is_dir():
            shutil.rmtree(old)
    rows = long[long["in_window"]].copy()
    if rows.empty:
        shutil.rmtree(out / name, ignore_errors=True)
        return []
    rows["period"] = [period_label(t, split) for t in rows["start_time"]]
    periods = [period_label(split - dt.timedelta(days=1), split), period_label(split, split)]
    style = dict(zip(periods, PERIOD_STYLE))
    odors = sorted(set(rows["odor"]))
    subsets = [(ALL_COHORTS, rows.assign(cohort=ALL_COHORTS))] + [
        (c, rows[rows["cohort"] == c]) for c in panels if (rows["cohort"] == c).any()]

    staging = _staging(out / name)
    staging.mkdir(parents=True)
    pre = pd.concat([baseline_table(sub, ["cohort", "period"]) for _, sub in subsets],
                    ignore_index=True)
    stats = pd.concat([baseline_stats(sub, "period").assign(cohort=c) for c, sub in subsets],
                      ignore_index=True)
    stats = stats[["cohort"] + [c for c in stats.columns if c != "cohort"]]
    pre.to_csv(staging / "pre_exposure_by_period.csv", index=False)
    stats.to_csv(staging / "pre_exposure_by_period_stats.csv", index=False)

    # user 2026-11-12: every fly as % responding, then the same flies as mean score
    # (the named cohorts keep their own pre-vs-post figures below)
    drawn = [(ALL_COHORTS, rows, "pct"), (ALL_COHORTS, rows, "score")]
    fig, axes = plt.subplots(len(drawn), 1, squeeze=False,
                             figsize=(max(11.0, 1.5 * (len(odors) + 1) + 2.5),
                                      4.2 * len(drawn) + 0.9))
    for ax, (c, sub, metric) in zip(axes[:, 0], drawn):
        sub = sub.assign(cohort=c)
        here = [p_ for p_ in periods if (sub["period"] == p_).any()]
        n = sub.groupby("period")["fly_id"].nunique().to_dict()
        title = c
        if c == ALL_COHORTS:
            mix = rows.drop_duplicates("fly_id").groupby(["period", "cohort"]).size()
            title = ("All flies, every cohort: "
                     + ("% responding" if metric == "pct" else "mean score") + "  (") + "; ".join(
                f"{p_}: " + ", ".join(f"{k.split('-')[0]} {v}" for k, v in mix[p_].items())
                for p_ in periods if p_ in mix.index.get_level_values(0)) + ")"
        _baseline_panel(ax, pre[pre["cohort"] == c], "period", here,
                        sorted(set(sub["odor"])) + [ALL_ODORS],
                        style, stats[stats["cohort"] == c],
                        baseline_table(sub.assign(_all=1), "_all"), title, metric=metric)
        handles = [Line2D([], [], ls="none", marker=style[p_][1], color=style[p_][0],
                          mec="white", ms=8, label=f"{p_} (n={n.get(p_, 0)} flies)")
                   for p_ in periods]
        handles.append(Line2D([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)),
                              label="both periods"))
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0),
                  ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK,
                  borderaxespad=0.1)
    h = 4.2 * len(drawn) + 0.9
    fig.suptitle(f"Pre-exposure baseline: before vs from {split:%b} {split.day}  "
                 f"(flies started {window})", fontsize=13, color=INK, y=1 - 0.12 / h)
    fig.text(0.5, 1 - 0.42 / h, "Period by batch start date. Top: % responding, 95% Wilson "
             "CI, Fisher exact p. Bottom: mean model score, 95% t CI, rank-test "
             "(Kruskal-Wallis) p. Uncorrected;\nthe all-odors p uses one value per fly "
             "(responded to any odor / its mean score over the panel).", ha="center",
             va="top", fontsize=9, color=INK_2, linespacing=1.4)
    fig.subplots_adjust(top=1 - 1.45 / h, bottom=0.45 / h, left=0.08, right=0.98,
                        hspace=0.55)
    fig.savefig(staging / "pre_exposure_by_period.png", dpi=200, facecolor="white")
    plt.close(fig)
    written = ["pre_exposure_by_period.png"]

    # each named cohort: pre vs post exposure, within each period
    for c, sub in subsets[1:]:
        table = summarise(sub, ["period", "odor"])
        stem = f"pre_vs_post_by_period_{c}"
        table.to_csv(staging / f"{stem}.csv", index=False)
        plot_pre_vs_post_by_period(
            table, c, periods, style,
            f"{c}: pre- vs post-exposure, before vs from {split:%b} {split.day}  "
            f"(flies started {window})",
            "Hollow marker = pre-exposure, filled = post-exposure, arrow = change. Number "
            "above each arrow: exact McNemar p (same fly pre vs post), uncorrected.\n"
            "Sensitized odor in bold. Period by batch start date.",
            staging / f"{stem}.png")
        written.append(f"{stem}.png")

    _swap_in(staging, out / name)
    return [out / name / w for w in written]


# ---------------------------------------------------------------------------
# One cohort split at a moment (e.g. a new batch of flies for that cohort)
# ---------------------------------------------------------------------------

COHORT_SPLIT_DIR = "Cohort-Split"


def parse_cohort_split(text: str) -> tuple[str, pd.Timestamp]:
    """``COHORT@YYYY-MM-DD[THH:MM]`` -> (cohort, moment)."""
    cohort, _, when = str(text).rpartition("@")
    if not cohort or not when:
        raise argparse.ArgumentTypeError(f"expected COHORT@YYYY-MM-DDTHH:MM, got {text!r}")
    return cohort, pd.Timestamp(when)


def _split_phases(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre as-is; post moved into the ``*_pre`` columns the baseline helpers read."""
    return {"Pre-exposure": rows,
            "Post-exposure": rows.assign(responded_pre=rows["responded_post"],
                                         score_pre=rows["score_post"])}


def write_cohort_split(out: Path, long: pd.DataFrame,
                       splits: Sequence[tuple[str, pd.Timestamp]]) -> list[Path]:
    """``Cohort-Split/``: one cohort's flies started before vs from a moment.

    Every fly of the cohort, window or not (user 2026-11-12: "all of the flies").
    Per split: pre- and post-exposure, % responding and mean score, the two groups
    side by side with a between-group p; the pre->post change within each group;
    and the same tests restricted to the rig(s) the later group ran on, since a
    new fly stock that lands on one rig is confounded with that rig."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = out / COHORT_SPLIT_DIR
    if not splits:
        shutil.rmtree(root, ignore_errors=True)
        return []
    staging = _staging(root)
    staging.mkdir(parents=True)
    written: list[str] = []
    for cohort, when in splits:
        rows = long[long["cohort"] == cohort].copy()
        if rows.empty:
            print(f"[sens-window] cohort split: no {cohort} flies; skipped", flush=True)
            continue
        stamp = f"{when:%b} {when.day} {when:%H:%M}"
        periods = [f"Before {stamp}", f"{stamp} onward"]
        later = pd.to_datetime(rows["start_time"]) >= when
        rows["period"] = np.where(later, periods[1], periods[0])
        here = [p_ for p_ in periods if (rows["period"] == p_).any()]
        style = dict(zip(periods, PERIOD_STYLE))
        stem = f"{cohort}_from_{when:%Y-%m-%d_%H%M}"
        odors = sorted(set(rows["odor"])) + [ALL_ODORS]
        later_rigs = sorted(set(rows.loc[later, "rig"].dropna().astype(str)))
        matched = rows[rows["rig"].astype(str).isin(later_rigs)]

        tables, stats = [], []
        for subset, sub in (("all flies", rows), ("same rig(s) as the later group", matched)):
            for ph, r in _split_phases(sub).items():
                if r.empty:
                    continue
                tables.append(baseline_table(r, "period").assign(comparison=subset, phase=ph))
                rigs = ", ".join(sorted(set(r["rig"].dropna().astype(str))))
                stats.append(baseline_stats(r, "period")
                             .assign(comparison=subset, phase=ph, rigs=rigs))
        table = pd.concat(tables, ignore_index=True)
        stat = pd.concat(stats, ignore_index=True)
        table.to_csv(staging / f"{stem}_by_group.csv", index=False)
        stat.to_csv(staging / f"{stem}_stats.csv", index=False)

        n = rows.groupby("period")["fly_id"].nunique().to_dict()
        def _rigs(r) -> str:
            rigs = sorted(set(r.dropna().astype(str).str.replace("rig_", "", regex=False)))
            return ("rig " if len(rigs) == 1 else "rigs ") + ", ".join(rigs) if rigs else "no rig"

        rig_of = {p_: _rigs(rows.loc[rows["period"] == p_, "rig"]) for p_ in here}
        panels = [(ph, m) for ph in ("Pre-exposure", "Post-exposure") for m in ("pct", "score")]
        h = 4.2 * len(panels) + 1.1
        fig, axes = plt.subplots(len(panels), 1, squeeze=False,
                                 figsize=(max(11.0, 1.5 * len(odors) + 2.5), h))
        allf = (table["comparison"] == "all flies")
        for ax, (ph, metric) in zip(axes[:, 0], panels):
            r = _split_phases(rows)[ph]
            sel = allf & (table["phase"] == ph)
            _baseline_panel(ax, table[sel], "period", here, odors, style,
                            stat[(stat["comparison"] == "all flies") & (stat["phase"] == ph)],
                            baseline_table(r.assign(_all=1), "_all"),
                            f"{ph}: " + ("% responding" if metric == "pct" else "mean score"),
                            metric=metric, phase=ph)
            handles = [Line2D([], [], ls="none", marker=style[p_][1], color=style[p_][0],
                              mec="white", ms=8,
                              label=f"{p_} (n={n.get(p_, 0)} flies; {rig_of.get(p_, '')})")
                       for p_ in here]
            handles.append(Line2D([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)),
                                  label="both groups"))
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0),
                      ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK,
                      borderaxespad=0.1)
        fig.suptitle(f"{cohort}: flies started before vs from {stamp}  (every fly, any "
                     "start time)", fontsize=13, color=INK, y=1 - 0.12 / h)
        fig.text(0.5, 1 - 0.42 / h,
                 "Group by batch start time. % responding: 95% Wilson CI, Fisher exact p. "
                 "Mean score: 95% t CI, rank-test (Kruskal-Wallis) p. Uncorrected; the "
                 "all-odors p uses one value per fly.\nThe later group ran on "
                 f"{_rigs(pd.Series(later_rigs))} only: {stem}_stats.csv repeats every "
                 "test on just those rig(s), so rig is not mistaken for the fly change.",
                 ha="center", va="top", fontsize=9, color=INK_2, linespacing=1.4)
        fig.subplots_adjust(top=1 - 1.45 / h, bottom=0.45 / h, left=0.08, right=0.98,
                            hspace=0.55)
        fig.savefig(staging / f"{stem}.png", dpi=200, facecolor="white")
        plt.close(fig)
        written.append(f"{stem}.png")

        change = summarise(rows, ["period", "odor"])
        change.to_csv(staging / f"{stem}_pre_vs_post.csv", index=False)
        plot_pre_vs_post_by_period(
            change, cohort, periods, style,
            f"{cohort}: pre- vs post-exposure, flies before vs from {stamp}",
            "Hollow marker = pre-exposure, filled = post-exposure, arrow = change. Number "
            "above each arrow: exact McNemar p (same fly pre vs post), uncorrected.\n"
            "Every fly of the cohort, any start time. Group by batch start time.",
            staging / f"{stem}_pre_vs_post.png")
        written.append(f"{stem}_pre_vs_post.png")
    _swap_in(staging, root)
    return [root / w for w in written]


def highlight_sources(out: Path, dataset: str, sensitized: str,
                      bin_hours: float) -> list[tuple[str, Path]]:
    tod = out / f"Time-of-Day_{bin_hours:g}h-bins_whole-dataset" / dataset / f"bins_{bin_hours:g}h"
    picks = [
        ("rate_bars", out / dataset / dataset / f"{dataset}_rate_bars.png"),
        ("score_bars", out / dataset / dataset / f"{dataset}_score_bars.png"),
        ("sensitized_odor_slopes", out / dataset / dataset / f"{dataset}_trained_odor.png"),
        ("sensitized_odor_mean_trace", out / dataset / "Mean-Traces_Pre-Test_vs_Post-Training"
         / f"{sensitized.replace(' ', '_')}_post_vs_pretest.png"),
        ("time-of-day_sensitized_odor", tod / "time_of_day_trained_odor.png"),
        ("time-of-day_overview", tod / "time_of_day_emphasis.png"),
    ]
    return [(f"{dataset}_{i}_{name}.png", p) for i, (name, p) in enumerate(picks, 1)
            if p.exists()]


def write_highlights(out: Path, built: Sequence[str], bin_hours: float,
                     cross: Sequence[Path] = (), *, primary: Sequence[Path] = ()) -> None:
    """``00_Highlights/``: symlinks, the pre-specified primary endpoint first
    (``00_00_*``, ahead of the cross-cohort ``00_*`` and the cohorts' own)."""
    dst = out / "00_Highlights"
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.iterdir():
        if old.is_symlink():
            old.unlink()
    for src in primary:
        (dst / f"00_00_{src.name}").symlink_to(Path("..") / src.relative_to(out))
    for ds in built:
        for name, src in highlight_sources(out, ds, trained_odor(ds), bin_hours):
            (dst / name).symlink_to(Path("..") / src.relative_to(out))
    for src in cross:
        (dst / f"00_{src.name}").symlink_to(Path("..") / src.relative_to(out))


# ---------------------------------------------------------------------------
# Mean traces of every cohort on one axis, one panel per odor
# ---------------------------------------------------------------------------

TRACE_FPS, TRACE_ON_S, TRACE_OFF_S, TRACE_END_S = 40.0, 30.0, 60.0, 90.0
TRACE_MIN_COVER = 0.9
TRACE_PHASES = (("pre", PRE_EXPOSURE, "wide_pretest", "pretest"),
                ("post", POST_EXPOSURE, "wide_testing", "testing"))


def _usable_trace(trace: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The first 90 s, and the tracked fractions of its baseline and of the odor window."""
    end, on, off = (int(s * TRACE_FPS) for s in (TRACE_END_S, TRACE_ON_S, TRACE_OFF_S))
    t = np.full(end, np.nan)
    t[: min(end, len(trace))] = trace[:end]
    ok = np.isfinite(t)
    return t, float(ok[:on].mean()), float(ok[on:off].mean())


def frame_mean_sem(arr: np.ndarray, min_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame mean and SEM over the flies with data in that frame; a frame where fewer
    than ``min_frac`` of the flies have data is left blank instead of drawn from a few."""
    finite = np.isfinite(arr)
    n = finite.sum(axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)
    thin = n < min_frac * arr.shape[0]
    mean[thin] = np.nan
    sem[thin] = np.nan
    return mean, sem


def frame_median_iqr(arr: np.ndarray, min_frac: float = 0.8
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame median and 25th / 75th percentiles over the flies with data in that
    frame, blanked where fewer than ``min_frac`` of the flies have data (as the mean)."""
    n = np.isfinite(arr).sum(axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q25, med, q75 = np.nanpercentile(arr, [25, 50, 75], axis=0)
    thin = n < min_frac * arr.shape[0]
    for a in (med, q25, q75):
        a[thin] = np.nan
    return med, q25, q75


def folder_permutation_p(values: Mapping[str, Mapping[str, float]], n_perm: int = 2000,
                         seed: int = 0) -> float:
    """Kruskal-Wallis p with whole batch folders, not flies, shuffled between cohorts.

    Flies recorded in one session share it, so they are not independent; a folder keeps
    its flies together and each cohort keeps its number of folders."""
    from scipy.stats import kruskal

    folders: dict[tuple[str, str], list[float]] = {}   # a folder name can recur in two datasets
    owner: dict[tuple[str, str], str] = {}
    for cohort, flies in values.items():
        for fly, v in flies.items():
            folder = (cohort, str(fly).rsplit("_fly", 1)[0])
            folders.setdefault(folder, []).append(float(v))
            owner[folder] = cohort
    names = list(folders)
    labels = [owner[f] for f in names]
    cohorts = sorted(set(labels))
    if len(cohorts) < 2:
        return np.nan

    def h(assign) -> float:
        groups = {c: [] for c in cohorts}
        for folder, c in zip(names, assign):
            groups[c].extend(folders[folder])
        return float(kruskal(*groups.values()).statistic)

    observed = h(labels)
    rng = np.random.default_rng(seed)
    hits = sum(h(rng.permutation(labels)) >= observed - 1e-9 for _ in range(n_perm))
    return (1 + hits) / (1 + n_perm)


def cohort_traces(out: Path, cohorts: Sequence[str], config: Path | None = None
                  ) -> tuple[dict, pd.DataFrame]:
    """``{(phase, odor, cohort): {fly: trace}}`` from each built cohort's ``_inputs``,
    made as the cohort's own mean-trace figures make them (baseline = first 30 s),
    plus the traces dropped for missing too much of the baseline or the odor."""
    import scripts.analysis.dataset_mean_traces_tvc as mt

    if config is not None:
        mt._apply_config_remap(str(config))
    data, excluded = {}, []
    for phase, _, table, trial_type in TRACE_PHASES:
        for cohort in cohorts:
            path = out / cohort / "_inputs" / f"{table}.parquet"
            if not path.exists():
                continue
            per_odor, _, _ = mt._prepare(pd.read_parquet(path), cohort, fps=TRACE_FPS,
                                         odor_on_s=TRACE_ON_S, trial_type=trial_type)
            for odor, flies in per_odor.items():
                if "Light Control" in odor:          # the light-only probe is no odor
                    continue
                for fly, trace in flies.items():
                    t, base, odor_frac = _usable_trace(trace)
                    if base >= TRACE_MIN_COVER and odor_frac >= TRACE_MIN_COVER:
                        data.setdefault((phase, odor, cohort), {})[fly] = t
                    else:
                        excluded.append({"phase": phase, "odor": odor, "cohort": cohort,
                                         "fly": fly, "baseline_finite_frac": round(base, 3),
                                         "odor_finite_frac": round(odor_frac, 3)})
    return data, pd.DataFrame(excluded, columns=["phase", "odor", "cohort", "fly",
                                                 "baseline_finite_frac", "odor_finite_frac"])


def trace_odor_order(data: Mapping, cohorts: Sequence[str]) -> list[str]:
    """Odors with at least 3 flies in at least 2 cohorts, each exposed odor first."""
    counts: dict[str, set] = {}
    for (phase, odor, cohort), flies in data.items():
        if len(flies) >= 3:
            counts.setdefault(odor, set()).add(cohort)
    shared = sorted(o for o, cs in counts.items() if len(cs) >= 2)
    exposed = [trained_odor(c) for c in cohorts]
    return [o for o in dict.fromkeys(exposed) if o in shared] + [o for o in shared if o not in exposed]


def _window_values(flies: Mapping[str, np.ndarray]) -> np.ndarray:
    """Each fly's mean during the odor."""
    on, off = int(TRACE_ON_S * TRACE_FPS), int(TRACE_OFF_S * TRACE_FPS)
    return np.nanmean(np.vstack(list(flies.values()))[:, on:off], axis=1)


def cohort_trace_stats(data: Mapping, cohorts: Sequence[str], odors: Sequence[str]
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per phase, odor and cohort: flies and their mean during the odor; Kruskal-Wallis
    across cohorts, also with whole folders shuffled (both Holm across odors); pairwise
    Mann-Whitney (Holm within the odor)."""
    from scipy.stats import kruskal, mannwhitneyu

    blocks, pairs = [], []
    for phase, *_ in TRACE_PHASES:
        rows = []
        for odor in odors:
            kept = [c for c in cohorts if len(data.get((phase, odor, c), {})) >= 3]
            groups = {c: _window_values(data[(phase, odor, c)]) for c in kept}
            h, p = kruskal(*groups.values()) if len(groups) >= 2 else (np.nan, np.nan)
            perm = folder_permutation_p({c: dict(zip(data[(phase, odor, c)], groups[c]))
                                         for c in kept}) if len(groups) >= 2 else np.nan
            means = {c: float(v.mean()) for c, v in groups.items()}
            spread = max(means.values()) - min(means.values()) if means else np.nan
            for c in cohorts:
                v = groups.get(c, np.array([]))
                q25, med, q75 = (np.percentile(v, [25, 50, 75]) if len(v)
                                 else (np.nan, np.nan, np.nan))
                rows.append({"phase": phase, "odor": odor, "cohort": c, "n_flies": len(v),
                             "odor_window_mean": float(v.mean()) if len(v) else np.nan,
                             "odor_window_sem": _sem(pd.Series(v)) if len(v) else np.nan,
                             "odor_window_median": float(med), "odor_window_q25": float(q25),
                             "odor_window_q75": float(q75),
                             "kw_H": float(h), "kw_p": float(p),
                             "kw_p_folder_permutation": float(perm),
                             "cohort_spread": float(spread)})
            for a, b in combinations(groups, 2):
                pairs.append({"phase": phase, "odor": odor, "cohort_a": a, "cohort_b": b,
                              "p_mwu": float(mannwhitneyu(groups[a], groups[b]).pvalue)})
        frame = pd.DataFrame(rows)
        if not frame.empty:
            for col in ("kw_p", "kw_p_folder_permutation"):
                per_odor = frame.drop_duplicates("odor").set_index("odor")[col]
                frame[f"{col}_holm"] = frame["odor"].map(
                    dict(zip(per_odor.index, holm_adjust(per_odor.to_numpy()))))
        blocks.append(frame)
    table = pd.concat(blocks, ignore_index=True)
    num = ["odor_window_mean", "odor_window_sem", "odor_window_median", "odor_window_q25",
           "odor_window_q75", "kw_H", "cohort_spread"]
    table[num] = table[num].round(4)
    pair = pd.DataFrame(pairs, columns=["phase", "odor", "cohort_a", "cohort_b", "p_mwu"])
    if not pair.empty:
        pair["p_mwu_holm_within_odor"] = pair.groupby(["phase", "odor"])["p_mwu"].transform(
            lambda s: holm_adjust(s.to_numpy()))
    return table, pair


def _trace_style(ax) -> None:
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


#: Trace figures per phase: stat -> (file stem, title word, caption of the line and band).
TRACE_STATS = {
    "mean": ("mean_traces", "mean",
             "Mean ± SEM over the flies with data at each moment (blank where under 80 % have "
             "data)"),
    "median": ("median_traces", "median",
               "Median, band = interquartile range (25th–75th percentile) over the flies with "
               "data at each moment (blank where under 80 % have data)"),
}


def _trace_centres(data: Mapping, cohorts: Sequence[str], odors: Sequence[str]
                   ) -> tuple[dict, dict]:
    """Per (phase, odor, cohort) with >= 3 flies: (mean, sem) and (median, q25, q75)."""
    means, medians = {}, {}
    for key, flies in data.items():
        if key[1] in odors and key[2] in cohorts and len(flies) >= 3:
            arr = np.vstack(list(flies.values()))
            means[key] = frame_mean_sem(arr)
            medians[key] = frame_median_iqr(arr)
    return means, medians


def plot_cohort_mean_vs_median(data: Mapping, cohorts: Sequence[str], odors: Sequence[str],
                               colours: Mapping[str, str], window: str,
                               out_dir: Path) -> list[Path]:
    """One figure per phase: a row per cohort, a column per odor, each panel that cohort's
    mean (solid) and median (dashed) over the same flies, so where they part shows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.analysis.dataset_means_specific_flies import Y_LABEL

    t = np.arange(int(TRACE_END_S * TRACE_FPS)) / TRACE_FPS
    means, medians = _trace_centres(data, cohorts, odors)
    lines = [m for m, _ in means.values()] + [m for m, *_ in medians.values()]
    lo, hi = min(np.nanmin(m) for m in lines), max(np.nanmax(m) for m in lines)
    pad = 0.1 * (hi - lo)
    written = []
    for phase, phase_name, *_ in TRACE_PHASES:
        rows, cols = len(cohorts), len(odors)
        fig, axes = plt.subplots(rows, cols, figsize=(2.55 * cols + 1.2, 2.25 * rows + 1.9),
                                 sharex=True, sharey=True, squeeze=False)
        for i, c in enumerate(cohorts):
            for j, odor in enumerate(odors):
                ax = axes[i, j]
                ax.axvspan(TRACE_ON_S, TRACE_OFF_S, color="#efeee9", lw=0, zorder=0)
                ax.axhline(0, color=INK_2, lw=0.7, zorder=1)
                key = (phase, odor, c)
                n = len(data.get(key, {}))
                if key in means:
                    ax.plot(t, means[key][0], color=colours[c], lw=1.3, zorder=3)
                    ax.plot(t, medians[key][0], color=INK, lw=1.1, ls=(0, (3, 1.5)), zorder=4)
                else:
                    ax.text(0.5, 0.5, "no flies" if not n else f"n={n}: too few",
                            transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
                            color=INK_2)
                ax.set_title(f"n={n}" if n else "", fontsize=8.5, color=INK_2, loc="right",
                             pad=2)
                if i == 0:
                    ax.annotate(odor, (0.5, 1.0), xycoords="axes fraction", xytext=(0, 16),
                                textcoords="offset points", ha="center", fontsize=10.5,
                                color=INK, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(f"{c.split('-')[0]}\n{Y_LABEL.replace('Mean ', '')}",
                                  color=INK, fontsize=9)
                if i == rows - 1:
                    ax.set_xlabel("Time (s)", color=INK, fontsize=9)
                _trace_style(ax)
                ax.tick_params(labelsize=8)
        axes[0, 0].set_xlim(0, TRACE_END_S)
        axes[0, 0].set_ylim(lo - pad, hi + pad)
        handles = [Line2D([], [], color=colours[c], lw=2.0,
                          label=f"{c} mean (exposed to {trained_odor(c)})") for c in cohorts]
        handles.append(Line2D([], [], color=INK, lw=1.4, ls=(0, (3, 1.5)),
                              label="median (dashed, every row)"))
        top_in = 1.75
        h = 2.25 * rows + 1.9
        fig.legend(handles=handles, loc="upper center", ncol=min(3, len(handles)),
                   bbox_to_anchor=(0.5, 1 - 0.62 / h), frameon=False, fontsize=9.5,
                   labelcolor=INK)
        fig.suptitle(f"{phase_name}: mean vs median trace, each cohort and odor: flies started "
                     f"{window}", fontsize=13.5, color=INK, y=1 - 0.08 / h)
        fig.text(0.5, 1 - 0.36 / h, "Both over the same flies (one trial per fly per odor), "
                 "blank where under 80 % have data. Where the solid line leaves the dashed one, "
                 "a few flies with large responses pull the mean away from the typical fly. "
                 "Grey band: 30–60 s odor.", ha="center", fontsize=8.8, color=INK_2)
        fig.subplots_adjust(top=1 - top_in / h, bottom=0.55 / h, left=0.06, right=0.99,
                            hspace=0.28, wspace=0.07)
        path = out_dir / f"{phase}_exposure_mean_vs_median_traces_by_cohort.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        written.append(path)
    return written


def plot_cohort_traces(data: Mapping, table: pd.DataFrame, cohorts: Sequence[str],
                       odors: Sequence[str], colours: Mapping[str, str], window: str,
                       out_dir: Path, stat: str = "mean") -> list[Path]:
    """One figure per phase: a panel per odor, every cohort on one axis, as the mean ±
    SEM or (``stat="median"``) the median with its interquartile range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.analysis.dataset_means_specific_flies import Y_LABEL

    stem, word, caption = TRACE_STATS[stat]
    y_label = Y_LABEL if stat == "mean" else Y_LABEL.replace("Mean", "Median", 1)
    t = np.arange(int(TRACE_END_S * TRACE_FPS)) / TRACE_FPS
    means, medians = _trace_centres(data, cohorts, odors)
    # From the centre lines, not the bands: one noisy 8-fly band must not flatten every panel.
    centres = ([m for m, _ in means.values()] if stat == "mean" else
               [m for m, *_ in medians.values()])
    lo = min(np.nanmin(m) for m in centres)
    hi = max(np.nanmax(m) for m in centres)
    pad = 0.1 * (hi - lo)
    cols = 4
    rows = -(-len(odors) // cols)
    written = []
    for phase, phase_name, *_ in TRACE_PHASES:
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.4 * rows), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, odor in zip(axes.flat, odors):
            ax.axvspan(TRACE_ON_S, TRACE_OFF_S, color="#efeee9", lw=0, zorder=0)
            ax.axhline(0, color=INK_2, lw=0.8, zorder=1)
            ns = []
            for c in cohorts:
                n = len(data.get((phase, odor, c), {}))
                ns.append(f"{c.split('-')[0]} {n if n else '–'}")
                if (phase, odor, c) not in means:
                    continue
                mean, sem = means[(phase, odor, c)]
                med, q25, q75 = medians[(phase, odor, c)]
                if stat == "mean":
                    ax.fill_between(t, mean - sem, mean + sem, color=colours[c], alpha=0.16,
                                    lw=0, zorder=2)
                    ax.plot(t, mean, color=colours[c], lw=1.8, zorder=3)
                else:
                    ax.fill_between(t, q25, q75, color=colours[c], alpha=0.16, lw=0, zorder=2)
                    ax.plot(t, med, color=colours[c], lw=1.8, zorder=3)
            r = table[(table["phase"] == phase) & (table["odor"] == odor)].iloc[0]
            p, q = r["kw_p"], r["kw_p_holm"]
            verdict = ("p n/a" if not np.isfinite(p) else
                       f"KW {_p_text(p)}, Holm {q:.3f}{_stars(q)}")
            ax.set_title(f"{odor}\n{verdict}\nn: {' · '.join(ns)}", fontsize=9.5, color=INK,
                         loc="left", fontweight="bold" if np.isfinite(q) and q < 0.05 else "normal")
            _trace_style(ax)
        for ax in axes.flat[len(odors):]:
            ax.axis("off")
        axes.flat[0].set_xlim(0, TRACE_END_S)
        axes.flat[0].set_ylim(lo - pad, hi + pad)
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)", color=INK)
        for ax in axes[:, 0]:
            ax.set_ylabel(y_label, color=INK, fontsize=9.5)
        handles = [Line2D([], [], color=colours[c], lw=2.4,
                          label=f"{c}  (exposed to {trained_odor(c)})") for c in cohorts]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2,
                   frameon=False, fontsize=10, labelcolor=INK)
        fig.suptitle(f"{phase_name} {word} traces by cohort: flies started {window}",
                     fontsize=14, color=INK, y=0.995)
        fig.text(0.5, 0.958, f"{caption}; one trial per fly per odor; baseline = mean of the 30 s "
                 "before the odor. Units: distance × angle, % of each fly's range.\nGrey band: 30–60 s (valves "
                 "opened at 29.3–30.3 s). p: Kruskal–Wallis across cohorts on each fly's mean during "
                 "30–60 s, Holm across odors; a test shuffling whole batch folders is in "
                 "cohort_trace_stats.csv. Paraffin Oil: batches from Nov 4 on only.",
                 ha="center", va="top", fontsize=8.8, color=INK_2)
        fig.subplots_adjust(top=0.82, bottom=0.07, left=0.07, right=0.99, hspace=0.55, wspace=0.08)
        path = out_dir / f"{phase}_exposure_{stem}_by_cohort.png"
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        written.append(path)
    return written


def plot_cohort_window_means(table: pd.DataFrame, cohorts: Sequence[str], odors: Sequence[str],
                             colours: Mapping[str, str], window: str, out_dir: Path,
                             stat: str = "mean") -> Path:
    """Each cohort's flies during the odor, pre and post side by side, a row per odor:
    their mean ± SEM, or (``stat="median"``) their median and interquartile range."""
    median = stat == "median"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    k = len(cohorts)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.62 * len(odors) + 2.4), sharey=True, sharex=True)
    for ax, (phase, phase_name, *_) in zip(axes, TRACE_PHASES):
        sub = table[table["phase"] == phase]
        for i, odor in enumerate(odors):
            if i % 2:
                ax.axhspan(i - 0.5, i + 0.5, color="#f4f3f0", lw=0, zorder=0)
            for j, c in enumerate(cohorts):
                r = sub[(sub["odor"] == odor) & (sub["cohort"] == c)]
                if r.empty or not np.isfinite(r["odor_window_mean"].iloc[0]):
                    continue
                y = i + (j - (k - 1) / 2) * 0.17
                if median:
                    m = float(r["odor_window_median"].iloc[0])
                    lo_, hi_ = float(r["odor_window_q25"].iloc[0]), float(r["odor_window_q75"].iloc[0])
                else:
                    m, e = float(r["odor_window_mean"].iloc[0]), float(r["odor_window_sem"].iloc[0])
                    lo_, hi_ = m - e, m + e
                ax.plot([lo_, hi_], [y, y], color=colours[c], lw=1.4, zorder=2)
                ax.plot(m, y, marker=COHORT_MARKERS[j % len(COHORT_MARKERS)], ms=7,
                        color=colours[c], mec="white", mew=0.8, zorder=3)
            q = sub[sub["odor"] == odor]["kw_p_holm"].iloc[0]
            if np.isfinite(q) and q < 0.05:
                ax.annotate(_stars(q), (1.0, i), xycoords=("axes fraction", "data"), ha="right",
                            va="center", fontsize=11, color=INK)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_title(phase_name, loc="left", fontsize=12, color=INK)
        ax.set_xlabel("Each fly's mean during the odor (30–60 s), baseline-subtracted",
                      color=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2)
    axes[0].set_yticks(range(len(odors)), odors, color=INK)
    axes[0].set_ylim(len(odors) - 0.5, -0.5)
    handles = [Line2D([], [], color=colours[c], marker=COHORT_MARKERS[j % len(COHORT_MARKERS)],
                      lw=1.4, ms=7, mec="white", label=c) for j, c in enumerate(cohorts)]
    fig.legend(handles=handles, loc="upper center", ncol=min(4, k), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.905), labelcolor=INK)
    fig.suptitle(f"Do the cohorts differ during the odor? Flies started {window}", fontsize=14,
                 color=INK, y=0.995)
    over = ("dot = median over flies, bar = interquartile range" if median
            else "dot = mean over flies, bar ± SEM")
    fig.text(0.5, 0.948, f"Each fly's mean during the odor; {over}. Stars: "
             "Kruskal–Wallis across cohorts, Holm across odors (* p<0.05, ** p<0.01, *** p<0.001).",
             ha="center", fontsize=9, color=INK_2)
    fig.subplots_adjust(top=0.81, bottom=0.12, left=0.13, right=0.98, wspace=0.06)
    path = out_dir / ("odor_window_medians_by_cohort.png" if median
                      else "odor_window_means_by_cohort.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def write_cohort_traces(out: Path, cohorts: Sequence[str], *, window: str,
                        config: Path | None = None) -> list[Path]:
    """``Mean-Traces-By-Cohort/``: every cohort's mean trace per odor, pre and post,
    window flies only, with whether the cohorts differ during the odor."""
    final = out / "Mean-Traces-By-Cohort"
    data, excluded = cohort_traces(out, cohorts, config)
    present = [c for c in cohorts if any(key[2] == c for key in data)]
    odors = trace_odor_order(data, present)
    if len(present) < 2 or not odors:
        shutil.rmtree(final, ignore_errors=True)
        return []
    staging = _staging(final)
    staging.mkdir(parents=True)
    table, pairs = cohort_trace_stats(data, present, odors)
    colours = cohort_colours(cohorts)
    written = []
    for stat in TRACE_STATS:      # mean and its median twin
        written += plot_cohort_traces(data, table, present, odors, colours, window, staging,
                                      stat=stat)
    written += plot_cohort_mean_vs_median(data, present, odors, colours, window, staging)
    for stat in ("mean", "median"):
        written.append(plot_cohort_window_means(table, present, odors, colours, window,
                                                staging, stat=stat))
    table.to_csv(staging / "cohort_trace_stats.csv", index=False)
    pairs.to_csv(staging / "cohort_trace_pairwise.csv", index=False)
    excluded.to_csv(staging / "excluded_traces_tracking_gaps.csv", index=False)
    _swap_in(staging, final)
    return [final / p.name for p in written]


# ---------------------------------------------------------------------------
# Window flies scored on odor + N s (user 2026-11-13)
# ---------------------------------------------------------------------------


def plus_after_dir(after_s: float) -> str:
    return f"Pre-vs-Post-Sensitivity-Odor-Plus-{after_s:g}s"


def read_rescored(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def rescored_pairs(rescored: pd.DataFrame, cohort: str, folders: Sequence[str],
                   flagged: Path, path: Path) -> pd.DataFrame:
    """The cohort's rescored pre/post pairs in exactly ``folders``; the rows are kept at
    ``path`` for the figure set drawn from them."""
    rows = swr.subset_rows(rescored, cohort, folders, flagged)
    rows.to_csv(path, index=False)
    if rows.empty:
        return pd.DataFrame(columns=["dataset_canon", "fly", "fly_number", "odor",
                                     "score_pre", "score_post"])
    paired = pvt.load_paired_scores(path, flagged_flies_csv=str(flagged))
    return paired[paired["dataset_canon"] == cohort]


def rescore_long(long: pd.DataFrame, paired: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The window rows of ``long`` with their scores replaced by the rescored pairs
    (``{cohort: pairs}``); a fly x odor the rescoring lacks drops out."""
    frames = []
    for cohort, p in paired.items():
        base = long[(long["cohort"] == cohort) & long["in_window"]].drop(
            columns=["score_pre", "score_post", "score_change", "responded_pre",
                     "responded_post"])
        if base.empty or p.empty:
            continue
        p = p.rename(columns={"fly": "batch_folder"})[
            ["batch_folder", "fly_number", "odor", "score_pre", "score_post"]]
        p = p.astype({"fly_number": int, "score_pre": int, "score_post": int})
        m = base.merge(p, on=["batch_folder", "fly_number", "odor"], how="inner")
        m["score_change"] = m["score_post"] - m["score_pre"]
        m["responded_pre"] = m["score_pre"] >= THRESHOLD
        m["responded_post"] = m["score_post"] >= THRESHOLD
        frames.append(m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def first_odor_dir(odor_s: float) -> str:
    return f"Pre-vs-Post-Sensitivity-Odor-First-{odor_s:g}s"


def write_plus_after(out: Path, long: pd.DataFrame, cohorts: Sequence[str], *,
                     rescored_csv: Path | None, after_s: float, window: str,
                     flagged: Path, control: str | None, log) -> list[Path]:
    """``Pre-vs-Post-Sensitivity-Odor-Plus-<N>s/``: the window flies of every cohort
    rescored on the odor + N s after (the pipeline's ``pretest_vs_test_plus_after``
    table): each cohort's standard pre-vs-post set, then the cohorts compared."""
    return write_rescored_window(out, long, cohorts, rescored_csv=rescored_csv,
                                 folder=plus_after_dir(after_s),
                                 tag=f"plus_{after_s:g}s_window",
                                 phrase=f"odor + {after_s:g} s after", window=window,
                                 flagged=flagged, control=control, log=log)


def write_first_odor(out: Path, long: pd.DataFrame, cohorts: Sequence[str], *,
                     rescored_csv: Path | None, odor_s: float, window: str,
                     flagged: Path, control: str | None, log) -> list[Path]:
    """``Pre-vs-Post-Sensitivity-Odor-First-<N>s/``: the same set with every trial
    scored on only the first N s of the odor (the pipeline's
    ``pretest_vs_test_first_odor`` table; user 2026-11-17)."""
    return write_rescored_window(out, long, cohorts, rescored_csv=rescored_csv,
                                 folder=first_odor_dir(odor_s),
                                 tag=f"first_{odor_s:g}s_odor_window",
                                 phrase=f"the first {odor_s:g} s of the odor",
                                 window=window, flagged=flagged, control=control, log=log)


def write_rescored_window(out: Path, long: pd.DataFrame, cohorts: Sequence[str], *,
                          rescored_csv: Path | None, folder: str, tag: str, phrase: str,
                          window: str, flagged: Path, control: str | None,
                          log) -> list[Path]:
    """``out/folder``: the window flies of every cohort with their scores from
    ``rescored_csv`` (a window_rescore.py table scored on ``phrase``): each cohort's
    standard pre-vs-post set, then the cohorts compared. No table, no folder."""
    final = out / folder
    if rescored_csv is None or not Path(rescored_csv).exists():
        shutil.rmtree(final, ignore_errors=True)
        return []
    rescored = read_rescored(Path(rescored_csv))
    staging = _staging(final)
    inputs = staging / "_inputs"
    inputs.mkdir(parents=True)
    note = f"scored on {phrase}; flies started {window}"
    paired = {}
    for cohort in cohorts:
        folders = sorted(long.loc[(long["cohort"] == cohort) & long["in_window"],
                                  "batch_folder"].unique())
        if not folders:
            continue
        path = inputs / f"{cohort}_predictions.csv"
        paired[cohort] = rescored_pairs(rescored, cohort, folders, flagged, path)
        if paired[cohort].empty:
            continue
        _run([PY, REPO_ROOT / "scripts/analysis/pretest_vs_test_comparison.py",
              "--predictions-csv", path, "--out-root", staging,
              "--binary-threshold", THRESHOLD, "--rate-test", "mcnemar", *P_ADJUST,
              "--window-note", note, "--flagged-flies-csv", flagged], log)
    plus = rescore_long(long, paired)
    written = []
    if not plus.empty:
        colours = cohort_colours(cohorts)
        table = change_table(plus)
        table.to_csv(staging / f"cross_cohort_change_{tag}.csv", index=False)
        png = staging / f"cross_cohort_change_{tag}.png"
        plot_cross_cohort(table, list(cohorts), colours,
                          f"Pre- to post-exposure change, every odor, scored on {phrase}: "
                          f"flies started {window}", png)
        written.append(png)
        levels = pre_post_table(plus)
        levels.to_csv(staging / f"cross_cohort_pre_post_{tag}.csv", index=False)
        png = staging / f"cross_cohort_pre_post_{tag}.png"
        plot_cross_cohort_pre_post(levels, list(cohorts), colours,
                                   f"Pre- and post-exposure, every odor, scored on {phrase}: "
                                   f"flies started {window}", png)
        written.append(png)
        minus = control_subtracted_table(plus, control)
        if not minus.empty:
            minus.to_csv(staging / f"change_minus_control_{tag}.csv", index=False)
            for cohort in [c for c in cohorts if c in set(minus["cohort"])]:
                png = staging / f"change_minus_control_{cohort}_{tag}.png"
                plot_control_subtracted(minus, cohort, colours,
                                        f"{cohort}: change minus {control}'s, scored on "
                                        f"{phrase}: flies started {window}", png)
                written.append(png)
        plus.to_csv(staging / f"per_fly_per_odor_{tag}.csv", index=False)
    _swap_in(staging, final)
    return [final / p.name for p in written]


# ---------------------------------------------------------------------------
# Post − pre traces by response transition (user 2026-11-13)
# ---------------------------------------------------------------------------

TRANSITION_DIR = "Response-Transitions"
#: (key, responded pre, responded post, label), in drawing order.
TRANSITIONS = (("new", False, True, "New responders: no → yes"),
               ("both", True, True, "Responded both times: yes → yes"),
               ("lost", True, False, "Stopped responding: yes → no"),
               ("never", False, False, "Never responded: no → no"))


def transition_groups(long: pd.DataFrame, data: Mapping, cohort: str, odor: str) -> dict:
    """``{transition: {fly: (pre, post, post - pre)}}`` for the cohort's window flies with a
    usable trace in both phases; a fly's group is its score (responded = score >= 2) pre and
    post on this odor."""
    rows = long[(long["cohort"] == cohort) & (long["odor"] == odor) & long["in_window"]]
    pre = data.get(("pre", odor, cohort), {})
    post = data.get(("post", odor, cohort), {})
    out: dict = {key: {} for key, *_ in TRANSITIONS}
    for r in rows.itertuples():
        fly = r.fly_id
        if fly not in pre or fly not in post:
            continue
        key = next(k for k, a, b, _ in TRANSITIONS
                   if bool(r.responded_pre) == a and bool(r.responded_post) == b)
        out[key][fly] = (pre[fly], post[fly], post[fly] - pre[fly])
    return out


def _window_mean(trace: np.ndarray, lo_s: float, hi_s: float) -> float:
    seg = trace[int(lo_s * TRACE_FPS): int(hi_s * TRACE_FPS)]
    return float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan


def transition_table(groups: Mapping, cohort: str, odor: str) -> pd.DataFrame:
    """Per transition: flies, and each fly's mean post − pre before, during and after the
    odor, as the mean and the median over flies; plus when the group's mean and median
    difference curves peak during the odor."""
    rows = []
    t = np.arange(int(TRACE_END_S * TRACE_FPS)) / TRACE_FPS
    on, off = int(TRACE_ON_S * TRACE_FPS), int(TRACE_OFF_S * TRACE_FPS)
    for key, a, b, label in TRANSITIONS:
        flies = groups.get(key, {})
        row = {"cohort": cohort, "odor": odor, "transition": key, "label": label,
               "responded_pre": a, "responded_post": b, "n_flies": len(flies)}
        if flies:
            diffs = np.vstack([d for _, _, d in flies.values()])
            for name, (lo_s, hi_s) in (("baseline", (0.0, TRACE_ON_S)),
                                       ("odor", (TRACE_ON_S, TRACE_OFF_S)),
                                       ("after", (TRACE_OFF_S, TRACE_END_S))):
                per_fly = np.array([_window_mean(d, lo_s, hi_s) for d in diffs])
                row[f"diff_{name}_mean"] = float(np.nanmean(per_fly))
                row[f"diff_{name}_median"] = float(np.nanmedian(per_fly))
            for kind, (centre, *_) in (("mean", frame_mean_sem(diffs, 0.5)),
                                       ("median", frame_median_iqr(diffs, 0.5))):
                seg = centre[on:off]
                row[f"diff_{kind}_peak_s"] = (float(t[on + int(np.nanargmax(seg))])
                                              if np.isfinite(seg).any() else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(4)
    return out


def plot_transitions(groups: Mapping, table: pd.DataFrame, cohort: str, odor: str,
                     window: str, path: Path) -> None:
    """A row per transition: pre vs post as the mean, pre vs post as the median, and the
    per-fly post − pre as the mean ± SEM with the median dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.analysis.dataset_means_specific_flies import Y_LABEL

    pre_c, post_c = TRIAL_PHASE_COLOURS[PRE_EXPOSURE], TRIAL_PHASE_COLOURS[POST_EXPOSURE]
    diff_c, med_c = INK, TRIAL_PHASE_COLOURS[EXPOSURE]
    t = np.arange(int(TRACE_END_S * TRACE_FPS)) / TRACE_FPS
    curves, lim_trace, lim_diff = {}, [], []
    for key, *_ in TRANSITIONS:
        flies = groups.get(key, {})
        if not flies:
            continue
        pre = np.vstack([p for p, _, _ in flies.values()])
        post = np.vstack([q for _, q, _ in flies.values()])
        diff = np.vstack([d for _, _, d in flies.values()])
        # half the flies, not 80 %: a 5-fly group would otherwise blank at one gap
        c = {"pre_mean": frame_mean_sem(pre, 0.5)[0], "post_mean": frame_mean_sem(post, 0.5)[0],
             "pre_med": frame_median_iqr(pre, 0.5)[0], "post_med": frame_median_iqr(post, 0.5)[0],
             "diff": frame_mean_sem(diff, 0.5), "diff_med": frame_median_iqr(diff, 0.5)[0]}
        curves[key] = c
        lim_trace += [c["pre_mean"], c["post_mean"], c["pre_med"], c["post_med"]]
        lim_diff += [c["diff"][0] - c["diff"][1], c["diff"][0] + c["diff"][1], c["diff_med"]]

    def _lim(arrs):
        lo = min(np.nanmin(a) for a in arrs if np.isfinite(a).any())
        hi = max(np.nanmax(a) for a in arrs if np.isfinite(a).any())
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    ylim_t, ylim_d = _lim(lim_trace), _lim(lim_diff)
    by = table.set_index("transition")
    rows = len(TRANSITIONS)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 2.75 * rows + 2.0), squeeze=False)
    h = fig.get_figheight()
    for i, (key, _, _, label) in enumerate(TRANSITIONS):
        n = int(by.loc[key, "n_flies"])
        for j, ax in enumerate(axes[i]):
            ax.axvspan(TRACE_ON_S, TRACE_OFF_S, color="#efeee9", lw=0, zorder=0)
            ax.axhline(0, color=INK_2, lw=0.8, zorder=1)
            _trace_style(ax)
            ax.set_xlim(0, TRACE_END_S)
            ax.set_ylim(*(ylim_d if j == 2 else ylim_t))
            if i == rows - 1:
                ax.set_xlabel("Time (s)", color=INK)
            if key not in curves:
                ax.text(0.5, 0.5, "no flies", transform=ax.transAxes, ha="center",
                        va="center", color=INK_2)
                continue
            c = curves[key]
            if j == 0:
                ax.plot(t, c["pre_mean"], color=pre_c, lw=1.5, zorder=2)
                ax.plot(t, c["post_mean"], color=post_c, lw=1.7, zorder=3)
            elif j == 1:
                ax.plot(t, c["pre_med"], color=pre_c, lw=1.5, ls="--", zorder=2)
                ax.plot(t, c["post_med"], color=post_c, lw=1.7, ls="--", zorder=3)
            else:
                m, e = c["diff"]
                ax.fill_between(t, m - e, m + e, color=diff_c, alpha=0.12, lw=0, zorder=2)
                ax.plot(t, m, color=diff_c, lw=1.6, zorder=3)
                ax.plot(t, c["diff_med"], color=med_c, lw=1.6, ls="--", zorder=4)
                r = by.loc[key]
                ax.text(0.99, 0.97, f"during odor: mean {r['diff_odor_mean']:+.2f}, "
                        f"median {r['diff_odor_median']:+.2f}\npeak at "
                        f"{r['diff_mean_peak_s']:.1f} s (mean), {r['diff_median_peak_s']:.1f} s "
                        "(median)", transform=ax.transAxes, ha="right", va="top", fontsize=8.3,
                        color=INK_2)
        axes[i, 0].set_ylabel(f"{label}\nn={n} flies\n{Y_LABEL.replace('Mean ', '')}",
                              color=INK, fontsize=9.5)
    for j, head in enumerate(("Mean trace: pre vs post", "Median trace: pre vs post",
                              "Post − pre, per fly: mean ± SEM, median dashed")):
        axes[0, j].set_title(head, loc="left", fontsize=11, color=INK, pad=8)
    handles = [Line2D([], [], color=pre_c, lw=2, label=PRE_EXPOSURE),
               Line2D([], [], color=post_c, lw=2, label=POST_EXPOSURE),
               Line2D([], [], color=diff_c, lw=2, label="post − pre, mean (± SEM)"),
               Line2D([], [], color=med_c, lw=2, ls="--", label="post − pre, median")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1 - 0.78 / h), labelcolor=INK)
    fig.suptitle(f"{cohort}, {odor}: traces by how each fly's response changed (flies "
                 f"started {window})", fontsize=13.5, color=INK, y=1 - 0.1 / h)
    fig.text(0.5, 1 - 0.42 / h, "Groups from each fly's model score on this odor (responded = "
             "score ≥ 2), pre- and post-exposure, so they differ by construction; the figure "
             "is for the shape and timing, not a test. Each trace baseline-subtracted (mean "
             "of the 30 s before the odor);\npost − pre is taken per fly, then the mean or "
             "median over flies. Blank where under half the group has data. Grey band: "
             "30–60 s odor. Each column has one y-scale.", ha="center", va="top", fontsize=8.8,
             color=INK_2)
    fig.subplots_adjust(top=1 - 1.45 / h, bottom=0.55 / h, left=0.08, right=0.99,
                        hspace=0.32, wspace=0.16)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_transitions(out: Path, long: pd.DataFrame, cohorts: Sequence[str], *,
                      window: str, config: Path | None = None) -> list[Path]:
    """``Response-Transitions/``: for each cohort in ``cohorts``, its sensitized odor's
    post − pre trace split by new responders, both, stopped and never (window flies)."""
    final = out / TRANSITION_DIR
    if not cohorts:
        shutil.rmtree(final, ignore_errors=True)
        return []
    data, _ = cohort_traces(out, list(cohorts), config)
    staging = _staging(final)
    staging.mkdir(parents=True)
    written = []
    for cohort in cohorts:
        odor = trained_odor(cohort)
        groups = transition_groups(long, data, cohort, odor)
        if not any(groups.values()):
            continue
        table = transition_table(groups, cohort, odor)
        stem = f"{cohort}_{odor.replace(' ', '_')}_post_minus_pre_by_transition"
        table.to_csv(staging / f"{stem}.csv", index=False)
        plot_transitions(groups, table, cohort, odor, window, staging / f"{stem}.png")
        written.append(staging / f"{stem}.png")
    _swap_in(staging, final)
    return [final / p.name for p in written]


# ---------------------------------------------------------------------------
# One odor across the session: pre-exposure, each exposure trial, post-exposure
# ---------------------------------------------------------------------------

TRIAL_PHASE_COLOURS = {PRE_EXPOSURE: "#7d7c78", EXPOSURE: "#eda100", POST_EXPOSURE: "#2a78d6"}
P_TIER_COLOURS = ("#f4f3f0", "#c5daf3", "#2a78d6")  # n.s. / uncorrected only / Holm


def trial_series(long: pd.DataFrame, training: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """One row per window fly, one column per trial of the cohort's sensitized odor.

    Only flies with every trial, so each bar is the same flies and every comparison
    is paired. A fly with a repeated exposure trial (aborted session) is left out."""
    rows = long[long["in_window"] & long["sensitized_odor"] & (long["cohort"] == cohort)]
    t = training[training["dataset"] == cohort]
    t = t.assign(_n=pd.to_numeric(t["trial_label"].str.extract(r"^training_(\d+)", expand=False)))
    t = t.dropna(subset=["_n"])
    if rows.empty or t.empty:
        return pd.DataFrame()
    t = t.assign(_n=t["_n"].astype(int),
                 fly_id=t["fly"].astype(str) + "_fly" + t["fly_number"].astype(int).astype(str))
    t = t[~t["fly_id"].isin(t.loc[t.duplicated(["fly_id", "_n"], keep=False), "fly_id"])]
    exposures = t.pivot(index="fly_id", columns="_n", values="score")
    numbers = sorted(exposures.columns)
    label = EXPOSURE.capitalize()
    exposures = exposures[numbers]
    exposures.columns = [f"{label} {n}" for n in numbers]
    panels = rows.set_index("fly_id")[["score_pre", "score_post"]].rename(
        columns={"score_pre": PRE_EXPOSURE, "score_post": POST_EXPOSURE})
    wide = panels.join(exposures, how="inner").dropna()
    order = [PRE_EXPOSURE, *exposures.columns, POST_EXPOSURE]
    return wide[order].astype(int)


def cochran_q(x: np.ndarray) -> tuple[float, float]:
    """Cochran's Q over a flies x trials boolean matrix: ``(Q, p)``."""
    k = x.shape[1]
    across, within = x.sum(axis=0), x.sum(axis=1)
    total = x.sum()
    denom = k * total - (within ** 2).sum()
    if denom == 0:
        return (np.nan, np.nan)
    q = (k - 1) * (k * (across ** 2).sum() - total ** 2) / denom
    return (float(q), float(chi2.sf(q, k - 1)))


def trial_series_summary(wide: pd.DataFrame) -> pd.DataFrame:
    """Per trial: flies pooled, one trial per fly, never a mean of means."""
    rows = []
    for trial in wide.columns:
        v = wide[trial]
        hits, n = int((v >= THRESHOLD).sum()), len(v)
        lo_, hi_ = wilson_ci(hits, n)
        rows.append({"trial": trial, "n_flies": n, "n_responding": hits,
                     "pct_responding": round(100.0 * hits / n, 2), "ci95_low": lo_,
                     "ci95_high": hi_, "mean_score": round(float(v.mean()), 4),
                     "sem_score": round(_sem(v), 4)})
    return pd.DataFrame(rows)


def trial_series_tests(wide: pd.DataFrame) -> pd.DataFrame:
    """Omnibus across all trials, then every pair, Holm-corrected within a measure."""
    binary = wide >= THRESHOLD
    n = len(wide)
    q_stat, q_p = cochran_q(binary.to_numpy())
    f_stat, f_p = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    omnibus = [("pct_responding", "Cochran Q", q_stat, q_p),
               ("score", "Friedman", float(f_stat), float(f_p))]
    rows = [{"measure": m, "test": test, "trial_a": "all", "trial_b": "all", "n_flies": n,
             "statistic": round(stat, 4), "p": p, "p_holm": p} for m, test, stat, p in omnibus]
    pairs = list(combinations(wide.columns, 2))
    for measure, test, fn, data in (("pct_responding", "McNemar exact", pvt.mcnemar_p, binary),
                                    ("score", "Wilcoxon signed-rank", pvt.wilcoxon_p, wide)):
        raw = [fn(data[a], data[b]) for a, b in pairs]
        raw = [np.nan if v is None else float(v) for v in raw]
        for (a, b), p, adj in zip(pairs, raw, holm_adjust(raw)):
            rows.append({"measure": measure, "test": test, "trial_a": a, "trial_b": b,
                         "n_flies": n, "statistic": np.nan, "p": p, "p_holm": float(adj)})
    out = pd.DataFrame(rows)
    out[["p", "p_holm"]] = out[["p", "p_holm"]].astype(float).round(6)
    return out


def _trial_short(trial: str) -> str:
    if trial == PRE_EXPOSURE:
        return "Pre"
    return "Post" if trial == POST_EXPOSURE else f"E{trial.split()[-1]}"


def _trial_colour(trial: str) -> str:
    return TRIAL_PHASE_COLOURS.get(trial, TRIAL_PHASE_COLOURS[EXPOSURE])


def _p_triangle(ax, trials: Sequence[str], tests: pd.DataFrame, title: str) -> None:
    """Lower-triangle matrix of every pair: text = Holm p, shade = which test passes."""
    from matplotlib.patches import Rectangle

    k = len(trials)
    lookup = {(r.trial_a, r.trial_b): r for r in tests.itertuples()}
    for i in range(1, k):
        for j in range(i):
            r = lookup[(trials[j], trials[i])]
            tier = 2 if r.p_holm < 0.05 else 1 if r.p < 0.05 else 0
            ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, lw=0,
                                   color=P_TIER_COLOURS[tier]))
            text = "–" if not np.isfinite(r.p_holm) else "<.001" if r.p_holm < 0.001 \
                else f"{r.p_holm:.3f}".lstrip("0") if r.p_holm < 1 else "1"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5,
                    color="white" if tier == 2 else INK)
    ax.set_xlim(-0.5, k - 1.5)
    ax.set_ylim(k - 0.5, 0.5)
    ax.set_xticks(range(k - 1), [_trial_short(t) for t in trials[:-1]], fontsize=9.5, color=INK)
    ax.set_yticks(range(1, k), [_trial_short(t) for t in trials[1:]], fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)


def plot_trial_series(summary: pd.DataFrame, tests: pd.DataFrame, title: str, path: Path) -> None:
    """Bars for % responding and mean score; beside each, every pair of trials tested."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    trials = summary["trial"].tolist()
    x = np.arange(len(trials))
    colours = [_trial_colour(t) for t in trials]
    fig = plt.figure(figsize=(14.5, 9.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.42, wspace=0.14)
    pct, score = summary["pct_responding"].to_numpy(), summary["mean_score"].to_numpy()
    lows, highs = summary["ci95_low"].to_numpy(), summary["ci95_high"].to_numpy()
    sem = summary["sem_score"].to_numpy()
    panels = [("pct_responding", pct, np.vstack([pct - lows, highs - pct]), "{:.0f}%",
               f"% of flies responding (score ≥ {THRESHOLD})", "McNemar exact, Holm p"),
              ("score", score, np.vstack([sem, sem]), "{:.2f}", "Mean PER score",
               "Wilcoxon signed-rank, Holm p")]
    for row, (measure, y, err, fmt, ylab, ptitle) in enumerate(panels):
        ax = fig.add_subplot(grid[row, 0])
        sub = tests[tests["measure"] == measure]
        vs_pre = {r.trial_b: r.p_holm for r in sub.itertuples() if r.trial_a == trials[0]}
        ax.bar(x, y, width=0.78, color=colours, zorder=2)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1.0, zorder=3)
        for i, trial in enumerate(trials):
            label = fmt.format(y[i]) + _stars(vs_pre.get(trial))
            ax.annotate(label, (i, y[i] + err[1, i]), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
        ax.set_xticks(x, [t.replace(" ", "\n", 1) if t[:3] == "Exp" else t.replace("-", "-\n", 1)
                          for t in trials], fontsize=9.5, color=INK)
        ax.set_xlim(-0.6, len(trials) - 0.4)
        ax.set_ylabel(ylab, fontsize=10.5, color=INK)
        ax.set_ylim(min(0.0, float((y - err[0]).min())), float((y + err[1]).max()) * 1.14)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2)
        omni = sub[sub["trial_a"] == "all"].iloc[0]
        ax.set_title(f"{omni['test']} across all {len(trials)} trials: {_p_text(omni['p'])}",
                     loc="left", fontsize=10.5, color=INK)
        _p_triangle(fig.add_subplot(grid[row, 1]), trials, sub[sub["trial_a"] != "all"], ptitle)
    fig.legend(handles=[Patch(color=P_TIER_COLOURS[2], label="p < 0.05 after Holm"),
                        Patch(color=P_TIER_COLOURS[1], label="p < 0.05 uncorrected only"),
                        Patch(color=P_TIER_COLOURS[0], label="not significant")],
               loc="upper right", bbox_to_anchor=(0.985, 0.925), ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK)
    fig.suptitle(title, fontsize=13.5, color=INK, y=0.985)
    fig.text(0.5, 0.945, "The same flies in every bar. Bars: % with 95% Wilson CI, score ± 1 SEM "
             "over flies. Stars: differs from pre-exposure after Holm correction for all "
             f"{len(trials) * (len(trials) - 1) // 2} pairs (* p<0.05, ** p<0.01, *** p<0.001). "
             "E1–E6 = exposure trials.", ha="center", fontsize=9, color=INK_2)
    fig.subplots_adjust(top=0.865, bottom=0.075, left=0.06, right=0.985)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def write_trial_series(out: Path, long: pd.DataFrame, training: pd.DataFrame,
                       cohorts: Sequence[str], *, window: str) -> list[Path]:
    """``Trial-By-Trial/``: the sensitized odor at every trial, window flies only."""
    staging = _staging(out / "Trial-By-Trial")
    staging.mkdir(parents=True)
    written = []
    for cohort in cohorts:
        wide = trial_series(long, training, cohort)
        if wide.empty:
            continue
        summary, tests = trial_series_summary(wide), trial_series_tests(wide)
        wide.reset_index().to_csv(staging / f"{cohort}_by_trial_per_fly.csv", index=False)
        summary.to_csv(staging / f"{cohort}_by_trial_summary.csv", index=False)
        tests.to_csv(staging / f"{cohort}_by_trial_stats.csv", index=False)
        png = staging / f"{cohort}_sensitized_odor_by_trial.png"
        plot_trial_series(summary, tests, f"{trained_odor(cohort)} across the session: {cohort}, "
                          f"flies started {window} (n={len(wide)} flies)", png)
        written.append(png)
    if not written:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(out / "Trial-By-Trial", ignore_errors=True)
        return []
    _swap_in(staging, out / "Trial-By-Trial")
    return [out / "Trial-By-Trial" / p.name for p in written]


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def _pct(v: float) -> str:
    return "–" if not np.isfinite(v) else f"{v:.0f}%"


def _p(v) -> str:
    v = float(v) if v is not None else np.nan
    return "–" if not np.isfinite(v) else ("<0.001" if v < 0.001 else f"{v:.3f}")


def write_readme(out: Path, args, statuses: Mapping[str, str],
                 tables: Mapping[str, pd.DataFrame] | None, *,
                 primary: pd.DataFrame | None = None) -> None:
    """The folder README. It opens with the pre-specified primary endpoint
    (``primary_endpoint_table``; why not, when it is missing), then says everything
    after it is exploratory."""
    lines = [f"# Sensitivity results: experiments started {args.start}–{args.end}", "",
             *primary_readme_lines(primary, getattr(args, "control_cohort", None)), "",
             f"Updated by the pipeline on every run ({dt.datetime.now():%Y-%m-%d %H:%M}), a cohort's "
             f"folders only when its data, the analysis code, the config or the scoring "
             f"model changed (delete "
             f"`{CACHE_FILE}` here to redraw everything), by "
             "`scripts/analysis/sensitivity_window_results.py`, settings in "
             "`analysis.sensitivity_window_results` of config_new.yaml. Window: first "
             f"pre-exposure trial at **{args.start}–{args.end}** (Pi clock, end exclusive). "
             "Flagged flies (`flagged-flys-truth.csv`) are dropped, as in every figure.", "",
             f"The cohort folders, `{PRIMARY_DIR}/`, `Data/`, `Cross-Cohort/`, "
             "`Pre-Exposure-Baseline/`, `Before-vs-From-*/`, "
             "`00_Highlights/` and "
             "`Time-of-Day_*` are "
             "regenerated each run, so don't save your own files inside them. Anything "
             "else in this folder is left alone.", "",
             "## Cohorts", "",
             "| Cohort | Status | Flies | Sensitized odor | % responding pre → post "
             "| p (rate, McNemar) | p (score, Wilcoxon) |",
             "|---|---|---|---|---|---|---|"]
    summary = None if tables is None else tables["window_summary_per_odor.csv"]
    for ds in args.dataset:
        odor = trained_odor(ds)
        row = None
        if summary is not None and not summary.empty:
            hit = summary[(summary["cohort"] == ds) & summary["sensitized_odor"]]
            row = hit.iloc[0] if not hit.empty else None
        if row is None:
            lines.append(f"| {ds} | {statuses[ds]} | – | {odor} | – | – | – |")
        else:
            lines.append(f"| {ds} | {statuses[ds]} | {row['n_flies']} | {odor} | "
                         f"{_pct(row['pct_responding_pre'])} → "
                         f"{_pct(row['pct_responding_post'])} | {_p(row['p_rate_mcnemar_paired'])} "
                         f"| {_p(row['p_score_wilcoxon'])} |")
    lines += ["", "Each cohort's own pre → post change, secondary to the primary endpoint "
              "above (not controlled for time or re-testing); the p-values in this table are "
              "uncorrected. Every odor of every cohort: `Data/window_summary_per_odor.csv`."]

    if tables is not None:
        bins = next(v for k, v in tables.items() if "by_" in k)
        lines += ["", f"## Sensitized odor by start time, whole cohort ({args.bin_hours:g} h bins)",
                  "", "| Cohort | Start bin | Flies | % responding pre → post | Mean score change |",
                  "|---|---|---|---|---|"]
        for _, r in bins[bins["sensitized_odor"]].iterrows():
            lines.append(f"| {r['cohort']} | {r['start_bin']} | {r['n_flies']} | "
                         f"{_pct(r['pct_responding_pre'])} → {_pct(r['pct_responding_post'])} "
                         f"| {r['mean_score_change']:+.2f} |")
        lines += ["", "Bin tests (Kruskal–Wallis, Fisher–Freeman–Halton, Spearman) are in "
                  "each cohort's `Time-of-Day_*/<cohort>/bins_*/stats_omnibus.csv`. A window "
                  "picked after looking at these numbers makes its own p-values optimistic."]

    lines += ["", "## Layout", "",
              f"- `{PRIMARY_DIR}/` (when `control_cohort` is set): the pre-specified primary "
              f"endpoint above as a figure (`{PRIMARY_PNG}`: A the controlled difference, B "
              f"each group's own change), `{PRIMARY_CSV}` and a README with the method.",
              "- `00_Highlights/`: symlinks to the headline figures, the primary endpoint "
              "first (`00_00_*`), then the cross-cohort ones (`00_*`) and each cohort's.",
              "- `Cross-Cohort/`: every odor of every cohort on one axis, change in % "
              "responding and in mean score (post − pre), and the pre- and post-exposure "
              "levels themselves (`cross_cohort_pre_post_window`), window flies only, with the "
              "plotted numbers as CSV. With `control_cohort` set, one figure per other cohort "
              "(`change_minus_control_<cohort>_window`): its change on every odor minus the "
              "control cohort's change on the same odor, permutation-tested with single flies "
              "shuffled (exploratory; the primary endpoint above shuffles whole batches).",
              "- `Pre-Exposure-Baseline/`: the naive pre-exposure response, window flies only, "
              "by cohort, by rig and cohort x rig, with tests of whether the groups differ "
              "(they should not: every fly is naive at that point); "
              "`pre_exposure_baseline.png` as % responding, `pre_exposure_baseline_score.png` "
              "as mean score.",
              "- `Before-vs-From-<date>/` (when `split_date` is set): window flies whose batch "
              "started before vs from that date: the pre-exposure baseline of every fly "
              "pooled, then each cohort in `split_cohorts`.",
              "- `Mean-Traces-By-Cohort/`: every cohort's mean pre- and post-exposure trace per "
              "odor on one axis (window flies), with a Kruskal–Wallis test across cohorts on each "
              "fly's mean during the odor, Holm across odors; each with a median twin "
              "(`*_median_traces_*`, band = interquartile range, `odor_window_medians_*`) and "
              "`*_mean_vs_median_traces_*`, a panel per cohort and odor with both lines.",
              "- `Pre-vs-Post-Sensitivity-Odor-Plus-<N>s/` (when the pipeline's "
              "`pretest_vs_test_plus_after` table exists): the window flies rescored on the "
              "odor + N s after: each cohort's pre-vs-post set, then the cohorts compared "
              "(change, pre/post levels, minus the control cohort), with the numbers as CSV.",
              "- `Pre-vs-Post-Sensitivity-Odor-First-<N>s/` (when the pipeline's "
              "`pretest_vs_test_first_odor` table exists): the same set with every trial "
              "scored on only the first N s of the odor. The rest of the odor is cut out of "
              "the trace, so the model's post-odor window is unchanged and never sees it; "
              "the model was trained on 30 s odor windows, so these scores are its reading "
              "of a shorter input than it learned on.",
              "- `Response-Transitions/` (when `transition_cohorts` is set): the cohort's "
              "sensitized odor, window flies split into new responders, responded both times, "
              "stopped and never; pre vs post as mean and as median, and each fly's post − pre "
              "(mean ± SEM, median), with the numbers as CSV.",
              "- `Cohort-Split/` (when `cohort_splits` is set): one cohort's flies started "
              "before vs from a moment (every fly, any start time), pre and post, % and "
              "score, plus the same tests on only the later group's rig(s).",
              "- `Trial-By-Trial/` (when `trial_series_cohorts` is set): the sensitized odor at "
              "pre-exposure, each exposure trial and post-exposure, window flies with every trial, "
              "with every pair of trials tested (McNemar and Wilcoxon, Holm-corrected).",
              "- `Data/`: pre/post score and % responding tables (CSV + one workbook); "
              "column guide in `Data/README.md`.",
              "- `<cohort>/`: the windowed figure set from `start_window_report.py`, with "
              "`FINDINGS.md` (every batch folder, key numbers), `_build_log.txt` and "
              "`_inputs/`. `Per-Fly/` links the pipeline's raw pre-exposure, exposure and "
              "post-exposure traces, plus abdomen geometry where it exists, for the same flies.",
              f"- `Time-of-Day_{args.bin_hours:g}h-bins_whole-dataset/<cohort>/`: every fly "
              "of the cohort at any start time, binned by start time. One cohort per run, so "
              "starvation level is held fixed."]
    (out / "README.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--start", required=True, help="window start, e.g. 10:40")
    p.add_argument("--end", required=True, help="window end (exclusive), e.g. 15:20")
    p.add_argument("--dataset", action="append", required=True, help="repeatable")
    p.add_argument("--bin-hours", type=float, default=2.0)
    p.add_argument("--config", type=Path, default=swr.CONFIG)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--csv-dir", type=Path, default=swr.CSV)
    p.add_argument("--flagged-flies-csv", type=Path, default=swr.FLAGGED)
    p.add_argument("--figures-root", type=Path, default=FIGURES_ROOT)
    p.add_argument("--split-date", type=dt.date.fromisoformat, default=None,
                   help="also compare window flies started before vs from this date "
                        "(YYYY-MM-DD)")
    p.add_argument("--split-cohort", action="append", default=[],
                   help="with --split-date, also show this cohort on its own (repeatable)")
    p.add_argument("--cohort-split", action="append", default=[], type=parse_cohort_split,
                   metavar="COHORT@YYYY-MM-DDTHH:MM",
                   help="also compare this cohort's flies started before vs from that "
                        "moment, every fly (repeatable)")
    p.add_argument("--trial-series-cohort", action="append", default=[],
                   help="cohort to plot at every trial of its sensitized odor (repeatable)")
    p.add_argument("--plus-after-csv", type=Path, default=None,
                   help="predictions rescored on odor + N s (window_rescore.py); also draw "
                        "the window flies' pre-vs-post set and cross-cohort figures on it")
    p.add_argument("--plus-after-s", type=float, default=5.0,
                   help="the N of --plus-after-csv, for folder name and titles")
    p.add_argument("--first-odor-csv", type=Path, default=None,
                   help="predictions rescored on only the first N s of the odor "
                        "(window_rescore.py --window odor_first); also draw the window "
                        "flies' pre-vs-post set and cross-cohort figures on it")
    p.add_argument("--first-odor-s", type=float, default=10.0,
                   help="the N of --first-odor-csv, for folder name and titles")
    p.add_argument("--transition-cohort", action="append", default=[],
                   help="cohort whose sensitized odor's post − pre trace is split by new "
                        "responders, both, stopped and never (repeatable)")
    p.add_argument("--control-cohort", default=None,
                   help="the control of the pre-specified primary endpoint "
                        "(00_Primary-Endpoint/); also plot each other cohort's change minus "
                        "this cohort's change (Cross-Cohort/change_minus_control_*)")
    return p


PATH_ARGS = ("out_dir", "config", "data_root", "csv_dir", "flagged_flies_csv", "figures_root",
             "plus_after_csv", "first_odor_csv")


def resolve_path_args(args):
    """Make every path absolute: the child scripts run with cwd=REPO_ROOT, so a relative
    path would name a different file for them than for this process."""
    for name in PATH_ARGS:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = resolve_path_args(build_parser().parse_args(argv))
    lo, hi = swr.parse_clock(args.start), swr.parse_clock(args.end)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = acquire_build_lock(out)  # held until this process exits
    if lock is None:
        print(f"[sens-window] ERROR: another build held {out} past the time limit; "
              "nothing rebuilt", flush=True)
        return 3

    statuses: dict[str, str] = {}
    built: list[str] = []
    longs: list[pd.DataFrame] = []
    predictions = pd.read_csv(Path(args.csv_dir) / "model_predictions.csv")
    training_csv = Path(args.csv_dir) / "model_predictions_training.csv"
    training = (pd.read_csv(training_csv) if training_csv.exists()
                else pd.DataFrame(columns=["dataset"]))
    wide = read_wide_for_fingerprints(args.csv_dir, args.dataset)
    flagged = read_flagged(args.flagged_flies_csv)
    settings = settings_digest(args)
    cache = fc.FigureCache(out / CACHE_FILE)
    with open(out / "_build_log.txt", "w") as log:
        for ds in args.dataset:
            status, folders = dataset_status(args.data_root, ds, lo, hi)
            if status == OK:
                status = scoring_status(predictions, ds, folders)
            print(f"[sens-window] {ds}: {status} ({len(folders)} folders)", flush=True)
            if status != OK:
                statuses[ds] = status
                for stale in (out / ds,):
                    if stale.exists():
                        shutil.rmtree(stale)
                continue
            fingerprint = cohort_fingerprint(ds, folders, predictions=predictions,
                                             training=training, wide=wide,
                                             starts=start_table(Path(args.data_root), ds),
                                             flagged=flagged, settings=settings)
            status, reused = build_or_reuse(out, ds, args, log, cache, fingerprint)
            long = cohort_pairs(ds, predictions, args.data_root, lo, hi,
                                flagged=args.flagged_flies_csv, bin_hours=args.bin_hours,
                                scratch=out)
            try:
                check_built(out, ds, long, status, args.flagged_flies_csv)
            except Exception as exc:
                if not reused:
                    raise        # a set built this run that disagrees is a real bug
                # Kept, yet out of step with the data: something the fingerprint does
                # not cover changed. Redraw it and check that instead.
                print(f"[sens-window] {ds}: kept figures fail the data check ({exc}); "
                      "rebuilding", flush=True)
                cache.forget(ds)
                status, _ = build_or_reuse(out, ds, args, log, cache, fingerprint)
                check_built(out, ds, long, status, args.flagged_flies_csv)
            statuses[ds] = status
            if (out / ds / "FINDINGS.md").exists():
                built.append(ds)
            longs.append(long)

    tables, cross, primary = None, [], None
    if longs and any(not d.empty for d in longs):
        long = pd.concat(longs, ignore_index=True)
        tables = write_data(out, long, args.dataset, args.bin_hours)
        window = f"{args.start}–{args.end}"
        if args.control_cohort:
            # the pre-specified headline before every exploratory figure, redrawn every
            # run like the other cross-cohort folders (the cohort cache never gates it)
            with open(out / "_build_log.txt", "a") as log:
                primary = write_primary_endpoint(out, long, args.dataset, args.control_cohort,
                                                 log, window=window)
        cross = write_cross_cohort(out, long, args.dataset, window=window,
                                   control=args.control_cohort)
        cross += write_pre_baseline(out, long, args.dataset, window=window)
        if args.split_date is not None:
            cross += write_date_split(out, long, args.dataset, split=args.split_date,
                                      window=window, panels=args.split_cohort)
        cross += write_cohort_split(out, long, args.cohort_split)
        if args.trial_series_cohort:
            cross += write_trial_series(out, long, training, args.trial_series_cohort,
                                        window=window)
        cross += write_cohort_traces(out, args.dataset, window=window, config=args.config)
        cross += write_transitions(out, long, args.transition_cohort, window=window,
                                   config=args.config)
        with open(out / "_build_log.txt", "a") as log:
            cross += write_plus_after(out, long, args.dataset,
                                      rescored_csv=args.plus_after_csv,
                                      after_s=args.plus_after_s, window=window,
                                      flagged=args.flagged_flies_csv,
                                      control=args.control_cohort, log=log)
            cross += write_first_odor(out, long, args.dataset,
                                      rescored_csv=args.first_odor_csv,
                                      odor_s=args.first_odor_s, window=window,
                                      flagged=args.flagged_flies_csv,
                                      control=args.control_cohort, log=log)
    if primary is None:          # not computed this run: no stale headline stays
        shutil.rmtree(out / PRIMARY_DIR, ignore_errors=True)
    headline = out / PRIMARY_DIR / PRIMARY_PNG
    write_highlights(out, built, args.bin_hours, cross,
                     primary=[headline] if headline.is_file() else [])
    write_readme(out, args, statuses, tables, primary=primary)
    for ds, s in statuses.items():
        print(f"[sens-window] {ds}: {s}")
    failed = [ds for ds, s in statuses.items() if "fail" in s]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
