"""Per-trial QC step: confirm the LED light did NOT appear where it wasn't commanded.

``check_light_stimulus`` confirms the light was physically ON for its
sensors-commanded window, and skips every trial without a light event. This
step is its mirror image: it decodes every trial video and looks for the
light where the sensors log says there should be none:

* a trial whose sensors log has no light event at all (pretest, testing,
  control training) must show no light anywhere in the video;
* a trial with commanded light must show none outside its commanded span
  (first light event -> last light event, plus a small margin).

Two failure shapes are caught: an on/off step inside the trial
(``leak_reason == "step"``), and a trial lit for its whole length, which has
no step and only shows up as a baseline that differs from the rest of its
batch (``leak_reason == "baseline_shift"``).

Trials without a sensors log are recorded as ``no_sensors_log`` and never
flagged: without the log there is no expectation to check against.

Rows go to ``logs/light_leak_flags.csv`` (override with ``LIGHT_LEAK_CSV``);
any *newly* leaking trial triggers one batched ntfy alert per run.
"""

from __future__ import annotations

import os
import re
from functools import partial
from pathlib import Path

from ..config import Settings, get_main_directories
from ..utils.frozen_folders import iter_live_batch_dirs
from ..utils.light_stimulus import (
    LEAK_MIN_DEV,
    check_trial_light_leak,
    flag_baseline_shifts,
    update_light_check_csv,
)
from ..utils.notify import ntfy_notify
from ..utils.parallel import parallel_map
from ..utils.trial_metadata import find_sensors_csv, load_trial_metadata, parse_light_span_seconds

TRIAL_DIR_RE = re.compile(r"(pretest|training|testing)_(\d+)$", re.IGNORECASE)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = REPO_ROOT / "logs" / "light_leak_flags.csv"
DEFAULT_PLOTS_DIR = REPO_ROOT / "figures" / "light_leak_check"
DEFAULT_STRIDE = 2


def _process_fly_dir(fly_dir: Path, cfg: Settings) -> list[dict]:
    """Check every trial subfolder in one batch directory. Picklable for joblib."""
    force_recompute = bool(getattr(getattr(cfg, "force", None), "pipeline", False))
    known_datasets = getattr(cfg, "datasets", ())

    rows: list[dict] = []
    try:
        trial_dirs = sorted(p for p in fly_dir.iterdir() if p.is_dir())
    except OSError as exc:
        print(f"[LIGHT-LEAK] cannot list {fly_dir}: {exc}")
        return rows
    for trial_dir in trial_dirs:
        m = TRIAL_DIR_RE.search(trial_dir.name)
        if not m:
            continue
        try:
            meta = load_trial_metadata(trial_dir, known_datasets=known_datasets)
        except FileNotFoundError:
            continue
        except Exception as exc:  # report-only QC: never take the run down
            print(f"[LIGHT-LEAK] {trial_dir.name}: metadata error, skipping ({exc})")
            rows.append({"trial_dir": str(trial_dir), "status": "metadata_error",
                         "passed": None, "leaked": None})
            continue
        extra = {
            "dataset": meta.dataset,
            "batch": str(meta.batch),
            "trial_type": meta.trial_type,
            "trial_index": meta.trial_index,
            "odor": meta.odor,
        }

        # Same lookup load_trial_metadata uses, but we need to know whether
        # the log exists at all: "no log" must never read as "no light".
        sensors_csv = find_sensors_csv(
            sidecar_csv=meta.sidecar_csv,
            batch_dir=meta.batch,
            trial_type=meta.raw_trial_type or m.group(1).lower(),
            trial_index=meta.trial_index,
        )
        if sensors_csv is None:
            rows.append({**extra, "trial_dir": str(trial_dir), "status": "no_sensors_log",
                         "passed": None, "leaked": None})
            continue

        on_s, last_s = parse_light_span_seconds(sensors_csv)
        allowed = [(on_s, last_s)] if on_s is not None else []

        row = check_trial_light_leak(
            trial_dir,
            allowed,
            stride=DEFAULT_STRIDE,
            force=force_recompute,
            plots_dir=DEFAULT_PLOTS_DIR,
            extra_columns=extra,
        )
        rows.append(row)
        if row["status"] == "checked" and row["leaked"]:
            print(f"[LIGHT-LEAK] {trial_dir.name}: LIGHT WHERE NONE COMMANDED "
                  f"(peak {row['leak_peak_dev']:+.1f}, intervals {row['leak_intervals']})")

    # A light stuck on for a whole trial has no step: compare baselines
    # across the batch. Recomputed every run (cheap) from cached baselines.
    flag_baseline_shifts(rows, min_dev=LEAK_MIN_DEV)
    for row in rows:
        if "baseline_shift" in (row.get("leak_reason") or ""):
            print(f"[LIGHT-LEAK] {Path(row['trial_dir']).name}: baseline "
                  f"{row['baseline_shift']:+.1f} vs batch (light on all trial?)")
    return rows


def main(cfg: Settings) -> None:
    csv_path = Path(os.getenv("LIGHT_LEAK_CSV", str(DEFAULT_CSV_PATH)))
    roots = get_main_directories(cfg)

    all_rows: list[dict] = []
    for root in roots:
        if not root.is_dir():
            print(f"[LIGHT-LEAK] main_directories entry does not exist: {root}")
            continue
        print(f"[LIGHT-LEAK] Scanning {root} for light outside commanded windows")
        fly_dirs = iter_live_batch_dirs(cfg, root)
        for result in parallel_map(
            partial(_process_fly_dir, cfg=cfg),
            fly_dirs,
            enabled=cfg.parallel.enabled,
            n_jobs=cfg.parallel.n_jobs,
        ):
            all_rows.extend(result)

    if not all_rows:
        print("[LIGHT-LEAK] No trials found.")
        return

    newly_flagged = update_light_check_csv(csv_path, all_rows)
    n_checked = sum(1 for r in all_rows if r["status"] == "checked")
    n_leaked = sum(1 for r in all_rows if r.get("leaked") is True)
    n_nolog = sum(1 for r in all_rows if r["status"] == "no_sensors_log")
    print(f"[LIGHT-LEAK] Checked {n_checked} trials, {n_leaked} with unexpected light, "
          f"{len(newly_flagged)} newly flagged this run, {n_nolog} without a sensors log. "
          f"CSV: {csv_path}")

    if newly_flagged:
        def _line(r: dict) -> str:
            where = ("whole trial vs batch" if r.get("leak_reason") == "baseline_shift"
                     else f"from {r['first_leak_s']:.1f}s, {r['leak_total_s']:.1f}s total")
            kind = "light trial, outside window" if r.get("light_commanded") else "no light commanded"
            return f"- {r['dataset']}/{Path(r['trial_dir']).name}: {kind}; {where}"

        lines = [_line(r) for r in newly_flagged[:20]]
        more = f"\n...and {len(newly_flagged) - 20} more" if len(newly_flagged) > 20 else ""
        ntfy_notify(
            f"Light leak check: {len(newly_flagged)} trial(s) flagged",
            "Light seen on video where the sensors log commanded none:\n"
            + "\n".join(lines) + more
            + f"\n\nFull details: {csv_path}",
            priority="high",
            tags="bulb,rotating_light",
        )
