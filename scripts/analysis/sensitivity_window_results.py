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
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2, friedmanchisquare

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
from fbpipe.utils import figure_cache as fc  # noqa: E402

FIGURES_ROOT = Path("/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures")
PY = sys.executable
THRESHOLD = pvt.BINARY_THRESHOLD

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
    """Per group: flies pooled (one trial per fly per odor), never a mean of means."""
    rows = []
    for key, g in long.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        s = pvt.per_odor_summary(g[["odor", "score_pre", "score_post"]],
                                 binary_threshold=THRESHOLD, rate_test="fisher").iloc[0]
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
            "p_score_wilcoxon": s["p_score"], "p_rate_fisher": s["p_rate"],
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
on scores (blank below 6 changed flies). `p_rate_mcnemar_paired` is the exact
paired test the figures show (as in the pipeline's Pre-vs-Post figures);
`p_rate_fisher` is the unpaired test, kept for reference. All p-values are
uncorrected.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


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
                 "--rate-test", "mcnemar"], log)
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
            "mean_score_change", "p_rate_fisher", "p_rate_mcnemar_paired", "p_score_wilcoxon"]
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
    that cohort's PeakZ figure draws: the mean of each fly's own change, the SEM over
    flies and the same Wilcoxon p, post vs pre paired by fly, uncorrected. A fly
    missing a phase has no change and is left out, and so is a cohort without the
    table. The sensitized odor is the cohort's own (``trained_odor``): the table's
    ``trained`` column is blank on every row missing a phase."""
    from scripts.analysis.auc_before_during import _wilcoxon

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
                row.update({
                    f"{name}_pre": float(pre.mean()),
                    f"{name}_post": float(post.mean()),
                    f"{name}_change": float(change.mean()),
                    f"sem_{name}_change": _sem(pd.Series(change)),
                    f"median_{name}_change": float(np.median(change)),
                    f"p_{name}_wilcoxon": _wilcoxon(post, pre),
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
                       window: str) -> list[Path]:
    """``Cross-Cohort/``: every odor of every cohort on one axis, window flies only.

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
            rows.append({**dict(zip(by, key)), "odor": odor,
                         "n_flies": int(go["fly_id"].nunique()), "n_trials": n,
                         "n_responding": k, "pct_responding": round(100 * k / n, 2),
                         "ci95_lo": lo, "ci95_hi": hi,
                         "mean_score_pre": round(float(go["score_pre"].mean()), 4)})
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
                    title: str) -> None:
    k = len(groups)
    step = 0.78 / max(k, 1)
    by = table.set_index([key, "odor"])
    ptab = stats.set_index("odor")
    ov = overall.set_index("odor")
    top = float(table["ci95_hi"].max()) + 14
    for i, odor in enumerate(odors):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
        ref = float(ov.loc[odor, "pct_responding"])
        ax.plot([i - 0.42, i + 0.42], [ref, ref], color=INK_2, lw=1.2, ls=(0, (3, 2)), zorder=1)
        p = ptab.loc[odor, "p_rate_ffh"] if odor in ptab.index else None
        sig = p is not None and np.isfinite(float(p)) and float(p) < 0.05
        ax.text(i, top - 3, _p_text(p), ha="center", va="top", fontsize=9, color=INK,
                fontweight="bold" if sig else "normal")
        for j, g in enumerate(groups):
            if (g, odor) not in by.index:
                continue
            r = by.loc[(g, odor)]
            x = i + (j - (k - 1) / 2) * step
            colour, marker = style[g]
            ax.plot([x, x], [r["ci95_lo"], r["ci95_hi"]], color=colour, lw=1.4, alpha=0.6,
                    zorder=2, solid_capstyle="round")
            ax.plot(x, r["pct_responding"], marker=marker, ms=8, color=colour, mec="white",
                    mew=1.0, zorder=3)
    ax.axvline(len(odors) - 1.5, color=INK_2, lw=0.8)
    ax.set_xlim(-0.5, len(odors) - 0.5)
    ax.set_ylim(0, top)
    ax.set_xticks(range(len(odors)))
    ax.set_xticklabels([f"{o}\n(pooled trials)" if o == ALL_ODORS else o for o in odors],
                       fontsize=10, color=INK)
    ax.set_ylabel("Pre-exposure\n% of flies responding", fontsize=10, color=INK)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=26)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def plot_pre_baseline(long: pd.DataFrame, cohorts: Sequence[str], tables: Mapping[str, pd.DataFrame],
                      title: str, path: Path) -> None:
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
                    cstyle, stats[stats["compared"] == "cohort"], overall, "A  By cohort")
    nc = long.groupby("cohort")["fly_id"].nunique()
    ax1.legend(handles=[Line2D([], [], ls="none", marker=cstyle[c][1], color=cstyle[c][0],
                               mec="white", ms=8, label=f"{c} (n={nc[c]})") for c in present]
               + [Line2D([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)), label="all flies")],
               loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(present) + 1,
               frameon=False, fontsize=9, labelcolor=INK, borderaxespad=0.1)
    _baseline_panel(ax2, tables["pre_exposure_by_rig.csv"], "rig", rigs, odors, rstyle,
                    stats[stats["compared"] == "rig"], overall, "B  By rig")
    nr = long.groupby("rig")["fly_id"].nunique()
    ax2.legend(handles=[Line2D([], [], ls="none", marker=rstyle[r][1], color=rstyle[r][0],
                               mec="white", ms=8, label=f"Rig {r} (n={nr[r]})") for r in rigs]
               + [Line2D([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)), label="all flies")],
               loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(rigs) + 1,
               frameon=False, fontsize=9, labelcolor=INK, borderaxespad=0.1)

    # C: cohort x rig, all odors pooled, so a rig effect can't pass for a cohort one
    grid = tables["pre_exposure_by_cohort_x_rig.csv"]
    grid = grid[grid["odor"] == ALL_ODORS].set_index(["cohort", "rig"])
    allr = tables["pre_exposure_by_cohort.csv"]
    allr = allr[allr["odor"] == ALL_ODORS].set_index("cohort")
    cols = [f"Rig {r}" for r in rigs] + ["All rigs"]
    vals = np.full((len(present), len(cols)), np.nan)
    notes = [[""] * len(cols) for _ in present]
    for i, c in enumerate(present):
        for j, r in enumerate(rigs):
            if (c, r) in grid.index:
                g = grid.loc[(c, r)]
                vals[i, j] = g["pct_responding"]
                notes[i][j] = f"{g['pct_responding']:.0f}%\n{int(g['n_flies'])} flies"
            else:
                notes[i][j] = "no flies"
        vals[i, -1] = allr.loc[c, "pct_responding"]
        notes[i][-1] = f"{allr.loc[c, 'pct_responding']:.0f}%\n{int(allr.loc[c, 'n_flies'])} flies"
    vmax = max(50.0, float(np.nanmax(vals)) if np.isfinite(vals).any() else 50.0)
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
    ax3.set_title("C  Cohort x rig, all odors pooled (% of fly x odor trials responding)",
                  loc="left", fontsize=11.5, color=INK, pad=6)

    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    fig.text(0.5, 0.975, "Before exposure every fly is naive and gets the same panel, so these "
             "should match. Whiskers: 95% Wilson CI. Dashed line: all flies together.\n"
             "p above each odor: Fisher-Freeman-Halton across groups, uncorrected; "
             "the all-odors p uses one value per fly (responded to any odor).",
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
    png = staging / "pre_exposure_baseline.png"
    plot_pre_baseline(rows, list(cohorts), tables,
                      f"Pre-exposure baseline, every cohort and rig: flies started {window}", png)
    _swap_in(staging, out / "Pre-Exposure-Baseline")
    return [out / "Pre-Exposure-Baseline" / png.name]


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

    fig, axes = plt.subplots(len(subsets), 1, squeeze=False,
                             figsize=(max(11.0, 1.5 * (len(odors) + 1) + 2.5),
                                      4.2 * len(subsets) + 0.9))
    for ax, (c, sub) in zip(axes[:, 0], subsets):
        here = [p_ for p_ in periods if (sub["period"] == p_).any()]
        n = sub.groupby("period")["fly_id"].nunique().to_dict()
        title = c
        if c == ALL_COHORTS:
            mix = rows.drop_duplicates("fly_id").groupby(["period", "cohort"]).size()
            title = "All flies, every cohort  (" + "; ".join(
                f"{p_}: " + ", ".join(f"{k.split('-')[0]} {v}" for k, v in mix[p_].items())
                for p_ in periods if p_ in mix.index.get_level_values(0)) + ")"
        _baseline_panel(ax, pre[pre["cohort"] == c], "period", here,
                        sorted(set(sub["odor"])) + [ALL_ODORS],
                        style, stats[stats["cohort"] == c],
                        baseline_table(sub.assign(_all=1), "_all"), title)
        handles = [Line2D([], [], ls="none", marker=style[p_][1], color=style[p_][0],
                          mec="white", ms=8, label=f"{p_} (n={n.get(p_, 0)} flies)")
                   for p_ in periods]
        handles.append(Line2D([], [], color=INK_2, lw=1.2, ls=(0, (3, 2)),
                              label="both periods"))
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0),
                  ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK,
                  borderaxespad=0.1)
    h = 4.2 * len(subsets) + 0.9
    fig.suptitle(f"Pre-exposure baseline: before vs from {split:%b} {split.day}  "
                 f"(flies started {window})", fontsize=13, color=INK, y=1 - 0.12 / h)
    fig.text(0.5, 1 - 0.42 / h, "Period by batch start date. Whiskers: 95% Wilson CI. "
             "p: Fisher exact between periods, uncorrected; the all-odors p uses one value "
             "per fly (responded to any odor).", ha="center", va="top", fontsize=9,
             color=INK_2)
    fig.subplots_adjust(top=1 - 1.35 / h, bottom=0.45 / h, left=0.08, right=0.98,
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
                     cross: Sequence[Path] = ()) -> None:
    dst = out / "00_Highlights"
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.iterdir():
        if old.is_symlink():
            old.unlink()
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
                rows.append({"phase": phase, "odor": odor, "cohort": c, "n_flies": len(v),
                             "odor_window_mean": float(v.mean()) if len(v) else np.nan,
                             "odor_window_sem": _sem(pd.Series(v)) if len(v) else np.nan,
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
    num = ["odor_window_mean", "odor_window_sem", "kw_H", "cohort_spread"]
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


def plot_cohort_traces(data: Mapping, table: pd.DataFrame, cohorts: Sequence[str],
                       odors: Sequence[str], colours: Mapping[str, str], window: str,
                       out_dir: Path) -> list[Path]:
    """One figure per phase: a panel per odor, every cohort's mean ± SEM on one axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.analysis.dataset_means_specific_flies import Y_LABEL

    t = np.arange(int(TRACE_END_S * TRACE_FPS)) / TRACE_FPS
    means = {}
    for key, flies in data.items():
        if key[1] in odors and key[2] in cohorts and len(flies) >= 3:
            means[key] = frame_mean_sem(np.vstack(list(flies.values())))
    # From the means, not the bands: one noisy 8-fly band must not flatten every panel.
    lo = min(np.nanmin(m) for m, _ in means.values())
    hi = max(np.nanmax(m) for m, _ in means.values())
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
                ax.fill_between(t, mean - sem, mean + sem, color=colours[c], alpha=0.16, lw=0,
                                zorder=2)
                ax.plot(t, mean, color=colours[c], lw=1.8, zorder=3)
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
            ax.set_ylabel(Y_LABEL, color=INK, fontsize=9.5)
        handles = [Line2D([], [], color=colours[c], lw=2.4,
                          label=f"{c}  (exposed to {trained_odor(c)})") for c in cohorts]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2,
                   frameon=False, fontsize=10, labelcolor=INK)
        fig.suptitle(f"{phase_name} mean traces by cohort: flies started {window}", fontsize=14,
                     color=INK, y=0.995)
        fig.text(0.5, 0.958, "Mean ± SEM over the flies with data at each moment (blank where under "
                 "80 % have data); one trial per fly per odor; baseline = mean of the 30 s before the "
                 "odor. Units: distance × angle, % of each fly's range.\nGrey band: 30–60 s (valves "
                 "opened at 29.3–30.3 s). p: Kruskal–Wallis across cohorts on each fly's mean during "
                 "30–60 s, Holm across odors; a test shuffling whole batch folders is in "
                 "cohort_trace_stats.csv. Paraffin Oil: batches from Nov 4 on only.",
                 ha="center", va="top", fontsize=8.8, color=INK_2)
        fig.subplots_adjust(top=0.82, bottom=0.07, left=0.07, right=0.99, hspace=0.55, wspace=0.08)
        path = out_dir / f"{phase}_exposure_mean_traces_by_cohort.png"
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        written.append(path)
    return written


def plot_cohort_window_means(table: pd.DataFrame, cohorts: Sequence[str], odors: Sequence[str],
                             colours: Mapping[str, str], window: str, out_dir: Path) -> Path:
    """Each cohort's mean during the odor, pre and post side by side, a row per odor."""
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
                m, e = float(r["odor_window_mean"].iloc[0]), float(r["odor_window_sem"].iloc[0])
                ax.plot([m - e, m + e], [y, y], color=colours[c], lw=1.4, zorder=2)
                ax.plot(m, y, marker=COHORT_MARKERS[j % len(COHORT_MARKERS)], ms=7,
                        color=colours[c], mec="white", mew=0.8, zorder=3)
            q = sub[sub["odor"] == odor]["kw_p_holm"].iloc[0]
            if np.isfinite(q) and q < 0.05:
                ax.annotate(_stars(q), (1.0, i), xycoords=("axes fraction", "data"), ha="right",
                            va="center", fontsize=11, color=INK)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_title(phase_name, loc="left", fontsize=12, color=INK)
        ax.set_xlabel("Mean during the odor (30–60 s), baseline-subtracted", color=INK)
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
    fig.text(0.5, 0.948, "Each fly's mean during the odor; dot = mean over flies, bar ± SEM. Stars: "
             "Kruskal–Wallis across cohorts, Holm across odors (* p<0.05, ** p<0.01, *** p<0.001).",
             ha="center", fontsize=9, color=INK_2)
    fig.subplots_adjust(top=0.81, bottom=0.12, left=0.13, right=0.98, wspace=0.06)
    path = out_dir / "odor_window_means_by_cohort.png"
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
    written = plot_cohort_traces(data, table, present, odors, colours, window, staging)
    written.append(plot_cohort_window_means(table, present, odors, colours, window, staging))
    table.to_csv(staging / "cohort_trace_stats.csv", index=False)
    pairs.to_csv(staging / "cohort_trace_pairwise.csv", index=False)
    excluded.to_csv(staging / "excluded_traces_tracking_gaps.csv", index=False)
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
                 tables: Mapping[str, pd.DataFrame] | None) -> None:
    lines = [f"# Sensitivity results: experiments started {args.start}–{args.end}", "",
             f"Updated by the pipeline on every run ({dt.datetime.now():%Y-%m-%d %H:%M}), a cohort's "
             f"folders only when its data, the analysis code, the config or the scoring "
             f"model changed (delete "
             f"`{CACHE_FILE}` here to redraw everything), by "
             "`scripts/analysis/sensitivity_window_results.py`, settings in "
             "`analysis.sensitivity_window_results` of config_new.yaml. Window: first "
             f"pre-exposure trial at **{args.start}–{args.end}** (Pi clock, end exclusive). "
             "Flagged flies (`flagged-flys-truth.csv`) are dropped, as in every figure.", "",
             "The cohort folders, `Data/`, `Cross-Cohort/`, `Pre-Exposure-Baseline/`, "
             "`Before-vs-From-*/`, "
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
    lines += ["", "All p-values are uncorrected. Every odor of every cohort: "
              "`Data/window_summary_per_odor.csv`."]

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
              "- `00_Highlights/`: symlinks to the headline figures of each cohort.",
              "- `Cross-Cohort/`: every odor of every cohort on one axis, change in % "
              "responding and in mean score (post − pre), and the pre- and post-exposure "
              "levels themselves (`cross_cohort_pre_post_window`), window flies only, with the "
              "plotted numbers as CSV.",
              "- `Pre-Exposure-Baseline/`: the naive pre-exposure response, window flies only, "
              "by cohort, by rig and cohort x rig, with tests of whether the groups differ "
              "(they should not: every fly is naive at that point).",
              "- `Before-vs-From-<date>/` (when `split_date` is set): window flies whose batch "
              "started before vs from that date: the pre-exposure baseline of every fly "
              "pooled, then each cohort in `split_cohorts`.",
              "- `Mean-Traces-By-Cohort/`: every cohort's mean pre- and post-exposure trace per "
              "odor on one axis (window flies), with a Kruskal–Wallis test across cohorts on each "
              "fly's mean during the odor, Holm across odors.",
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
    p.add_argument("--trial-series-cohort", action="append", default=[],
                   help="cohort to plot at every trial of its sensitized odor (repeatable)")
    return p


PATH_ARGS = ("out_dir", "config", "data_root", "csv_dir", "flagged_flies_csv", "figures_root")


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

    tables, cross = None, []
    if longs and any(not d.empty for d in longs):
        long = pd.concat(longs, ignore_index=True)
        tables = write_data(out, long, args.dataset, args.bin_hours)
        window = f"{args.start}–{args.end}"
        cross = write_cross_cohort(out, long, args.dataset, window=window)
        cross += write_pre_baseline(out, long, args.dataset, window=window)
        if args.split_date is not None:
            cross += write_date_split(out, long, args.dataset, split=args.split_date,
                                      window=window, panels=args.split_cohort)
        if args.trial_series_cohort:
            cross += write_trial_series(out, long, training, args.trial_series_cohort,
                                        window=window)
        cross += write_cohort_traces(out, args.dataset, window=window, config=args.config)
    write_highlights(out, built, args.bin_hours, cross)
    write_readme(out, args, statuses, tables)
    for ds, s in statuses.items():
        print(f"[sens-window] {ds}: {s}")
    failed = [ds for ds, s in statuses.items() if "fail" in s]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
