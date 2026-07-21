"""Run the light-stimulus QC check across every trial in every configured dataset.

Unlike ``fbpipe.pipeline``'s ``check_light_stimulus`` step (which only scans the
local staging directories in ``main_directories`` — useful for freshly-landed,
not-yet-synced batches), this script scans the full archive: for each dataset
in the config's ``datasets:`` list it prefers the secured-storage copy
(``dataset_bases.secured``) and falls back to the local copy
(``dataset_bases.data``) when a dataset hasn't been synced yet. That covers the
entire historical archive, not just new trials.

Usage:
    python scripts/check_all_light_stimulus.py --config config/config_new.yaml --thaw
    python scripts/check_all_light_stimulus.py --config config/config_new.yaml --dataset EB-Training-24-1 --thaw
    python scripts/check_all_light_stimulus.py --config config/config_new.yaml --limit 5   # smoke test

``--thaw`` forces every trial to be recomputed, ignoring each trial's on-disk
``_light_check.json`` cache (this is a local cache for this checker only — it
is unrelated to the dataset freeze/wide-CSV cache used elsewhere in the
pipeline).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fbpipe.config import load_raw_config, load_settings  # noqa: E402
from fbpipe.utils.light_stimulus import check_trial_light_stimulus, update_light_check_csv  # noqa: E402
from fbpipe.utils.notify import ntfy_notify  # noqa: E402
from fbpipe.utils.parallel import parallel_map  # noqa: E402
from fbpipe.utils.trial_metadata import load_trial_metadata  # noqa: E402

TRIAL_DIR_RE = re.compile(r"(training|testing)_(\d+)$", re.IGNORECASE)

DEFAULT_CSV_PATH = REPO_ROOT / "logs" / "light_stimulus_flags.csv"
DEFAULT_PLOTS_DIR = REPO_ROOT / "figures" / "light_stimulus_check"


def _resolve_expected_window(meta):
    if meta.light_on_s is not None and meta.light_off_s is not None:
        return (meta.light_on_s, meta.light_off_s), "sensors"
    if meta.light_on_s is not None and meta.odor_off_s is not None:
        return (meta.light_on_s, meta.odor_off_s), "sensors_on+odor_off"
    return None, "none"


def resolve_dataset_roots(config_path: str, dataset_filter: str | None) -> list[tuple[str, Path]]:
    """Return [(dataset_name, root_path), ...], preferring secured over local."""
    raw = load_raw_config(config_path)
    bases = raw.get("dataset_bases", {})
    data_base = Path(bases.get("data", "/home/ramanlab/Documents/cole/Data/flys_New"))
    secured_base = Path(bases.get("secured", "/securedstorage/DATAsec/cole/Data-secured-New"))
    datasets = raw.get("datasets") or []

    roots: list[tuple[str, Path]] = []
    for ds in datasets:
        if dataset_filter and ds != dataset_filter:
            continue
        secured_dir = secured_base / ds
        data_dir = data_base / ds
        if secured_dir.is_dir():
            roots.append((ds, secured_dir))
        elif data_dir.is_dir():
            roots.append((ds, data_dir))
        else:
            print(f"[LIGHT-ALL] Dataset {ds!r} not found under secured or local base; skipping.")
    return roots


def _process_batch_dir(batch_dir: Path, *, dataset: str, known_datasets: tuple, stride: int, force: bool) -> list[dict]:
    rows: list[dict] = []
    for trial_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
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
            stride=stride,
            force=force,
            plots_dir=DEFAULT_PLOTS_DIR,
            extra_columns={
                "dataset": dataset,
                "batch": str(meta.batch),
                "trial_type": meta.trial_type,
                "trial_index": meta.trial_index,
                "odor": meta.odor,
            },
        )
        rows.append(row)
        if row["status"] == "checked":
            verdict = "PASS" if row["passed"] else "FAIL"
            print(f"[LIGHT-ALL] {dataset}/{trial_dir.name}: {verdict} "
                  f"(window={window_source}, on={row['fraction_on_in_window']*100:.1f}%)")
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "config_new.yaml"))
    p.add_argument("--dataset", default=None, help="Only scan this one dataset (by name from the config's datasets: list)")
    p.add_argument("--stride", type=int, default=2, help="Sample every Nth frame (default: 2)")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many batch folders (smoke testing)")
    p.add_argument("--thaw", action="store_true",
                    help="Force recompute for every trial, ignoring each trial's on-disk check cache")
    p.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    p.add_argument("--jobs", type=int, default=1,
                    help="Parallel workers across batch folders (default: 1, serial). "
                         "Overrides the config's parallel.enabled/n_jobs for this run only.")
    args = p.parse_args()

    settings = load_settings(args.config)
    if args.jobs and args.jobs > 1:
        settings = replace(settings, parallel=replace(settings.parallel, enabled=True, n_jobs=args.jobs))
    known_datasets = tuple(settings.datasets)
    roots = resolve_dataset_roots(args.config, args.dataset)
    if not roots:
        print("[LIGHT-ALL] No dataset roots resolved; nothing to do.")
        return 0

    all_rows: list[dict] = []
    n_batches = 0
    for dataset, root in roots:
        print(f"[LIGHT-ALL] Scanning dataset {dataset!r} at {root}")
        batch_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if args.limit is not None:
            remaining = args.limit - n_batches
            if remaining <= 0:
                break
            batch_dirs = batch_dirs[:remaining]

        results = parallel_map(
            partial(
                _process_batch_dir,
                dataset=dataset,
                known_datasets=known_datasets,
                stride=args.stride,
                force=args.thaw,
            ),
            batch_dirs,
            enabled=settings.parallel.enabled,
            n_jobs=settings.parallel.n_jobs,
        )
        for r in results:
            all_rows.extend(r)
        n_batches += len(batch_dirs)

    if not all_rows:
        print("[LIGHT-ALL] No trials with a resolvable light-on window found.")
        return 0

    csv_path = Path(args.csv)
    newly_flagged = update_light_check_csv(csv_path, all_rows)
    n_checked = sum(1 for r in all_rows if r["status"] == "checked")
    n_failed = sum(1 for r in all_rows if r.get("passed") is False)
    n_no_video = sum(1 for r in all_rows if r["status"] == "no_video")
    print(f"[LIGHT-ALL] Checked {n_checked} trials ({n_no_video} had no video), "
          f"{n_failed} failing, {len(newly_flagged)} newly flagged this run. CSV: {csv_path}")

    if newly_flagged:
        lines = [
            f"- {r['dataset']}/{Path(r['trial_dir']).name}: {r['fraction_on_in_window']*100:.0f}% on"
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
