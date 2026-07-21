"""Per-trial QC step: confirm the red LED light stimulus actually appeared on screen.

The sensors log tells us when the LED driver was *commanded* on/off
(``TrialMetadata.light_on_s`` / ``light_off_s``); this step decodes each
trial's video and checks the red channel to confirm the light was *physically*
on for that whole window. A trial where the driver was commanded correctly but
the LED dropped out early (bad wiring, a flaky LDD-L driver, etc.) is flagged
here — the sensors log alone can't catch that.

Flagged/failed trials are appended to a repo-wide CSV
(``logs/light_stimulus_flags.csv`` by default, override with
``LIGHT_CHECK_CSV``) and trigger one batched ntfy alert per run for any
*newly* failing trial (repeat runs over unchanged videos never re-notify).
"""

from __future__ import annotations

import os
import re
from functools import partial
from pathlib import Path

from ..config import Settings, get_main_directories
from ..utils.light_stimulus import check_trial_light_stimulus, update_light_check_csv
from ..utils.notify import ntfy_notify
from ..utils.parallel import parallel_map
from ..utils.trial_metadata import load_trial_metadata

TRIAL_DIR_RE = re.compile(r"(training|testing)_(\d+)$", re.IGNORECASE)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = REPO_ROOT / "logs" / "light_stimulus_flags.csv"
DEFAULT_PLOTS_DIR = REPO_ROOT / "figures" / "light_stimulus_check"
DEFAULT_STRIDE = 2


def _resolve_expected_window(meta) -> tuple[tuple[float, float] | None, str]:
    """Prefer the sensors-log light window; fall back to the odor window.

    Returns (None, "none") when neither a light-on event nor an odor window
    is available — those trials are skipped, not flagged (we can't fabricate
    an expectation).
    """
    if meta.light_on_s is not None and meta.light_off_s is not None:
        return (meta.light_on_s, meta.light_off_s), "sensors"
    if meta.light_on_s is not None and meta.odor_off_s is not None:
        # Light-on event logged but no explicit off event (rig cut short) —
        # the light tracks the odor window in this paradigm.
        return (meta.light_on_s, meta.odor_off_s), "sensors_on+odor_off"
    return None, "none"


def _process_fly_dir(fly_dir: Path, cfg: Settings) -> list[dict]:
    """Check every trial subfolder in one batch directory. Picklable for joblib."""
    force_recompute = bool(getattr(getattr(cfg, "force", None), "pipeline", False))
    known_datasets = getattr(cfg, "datasets", ())

    rows: list[dict] = []
    for trial_dir in sorted(p for p in fly_dir.iterdir() if p.is_dir()):
        if not TRIAL_DIR_RE.search(trial_dir.name):
            continue
        try:
            meta = load_trial_metadata(trial_dir, known_datasets=known_datasets)
        except FileNotFoundError:
            continue

        expected_on, window_source = _resolve_expected_window(meta)
        if expected_on is None:
            continue

        row = check_trial_light_stimulus(
            trial_dir,
            expected_on,
            window_source=window_source,
            stride=DEFAULT_STRIDE,
            force=force_recompute,
            plots_dir=DEFAULT_PLOTS_DIR,
            extra_columns={
                "dataset": meta.dataset,
                "batch": str(meta.batch),
                "trial_type": meta.trial_type,
                "trial_index": meta.trial_index,
                "odor": meta.odor,
            },
        )
        rows.append(row)
        if row["status"] == "checked":
            verdict = "PASS" if row["passed"] else "FAIL"
            print(f"[LIGHT] {trial_dir.name}: {verdict} "
                  f"(window={window_source}, on={row['fraction_on_in_window']*100:.1f}% of expected)")
        elif row["status"] == "no_video":
            print(f"[LIGHT] {trial_dir.name}: no video found, skipping")
    return rows


def main(cfg: Settings) -> None:
    csv_path = Path(os.getenv("LIGHT_CHECK_CSV", str(DEFAULT_CSV_PATH)))
    roots = get_main_directories(cfg)

    all_rows: list[dict] = []
    for root in roots:
        if not root.is_dir():
            print(f"[LIGHT] main_directories entry does not exist: {root}")
            continue
        print(f"[LIGHT] Scanning {root} for light-stimulus trials")
        fly_dirs = [p for p in root.iterdir() if p.is_dir()]
        for result in parallel_map(
            partial(_process_fly_dir, cfg=cfg),
            fly_dirs,
            enabled=cfg.parallel.enabled,
            n_jobs=cfg.parallel.n_jobs,
        ):
            all_rows.extend(result)

    if not all_rows:
        print("[LIGHT] No trials with a resolvable light-on window found.")
        return

    newly_flagged = update_light_check_csv(csv_path, all_rows)
    n_checked = sum(1 for r in all_rows if r["status"] == "checked")
    n_failed = sum(1 for r in all_rows if r.get("passed") is False)
    print(f"[LIGHT] Checked {n_checked} trials, {n_failed} failing, "
          f"{len(newly_flagged)} newly flagged this run. CSV: {csv_path}")

    if newly_flagged:
        lines = [
            f"- {r['dataset']}/{Path(r['trial_dir']).name}: "
            f"{r['fraction_on_in_window']*100:.0f}% on"
            + (f", dropout @ {r['dropout_time_s']:.1f}s" if r.get("dropout_time_s") is not None else "")
            for r in newly_flagged[:20]
        ]
        more = f"\n...and {len(newly_flagged) - 20} more" if len(newly_flagged) > 20 else ""
        ntfy_notify(
            f"Light stimulus check: {len(newly_flagged)} trial(s) flagged",
            "Light did not stay on for its full sensor-commanded window:\n"
            + "\n".join(lines) + more
            + f"\n\nFull details: {csv_path}",
            priority="high",
            tags="bulb,warning",
        )
