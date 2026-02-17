"""Generate dataset-level mean +/- SEM plots of Distance % for testing odors.

For each dataset root, produces one figure with one panel per odor showing the
across-fly mean +/- SEM trace over time, with shaded odor presentation windows.
A JSON sidecar with metadata is saved alongside each figure.

Example
-------
    python scripts/analysis/dataset_means.py \
        --config config/config.yaml \
        --outdir artifacts/dataset_means
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Repo root setup (so we can import fbpipe and sibling scripts)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
_src_str = str(ROOT / "src")
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from fbpipe.config import load_settings, resolve_config_path  # noqa: E402
from fbpipe.utils.columns import (  # noqa: E402
    find_proboscis_distance_percentage_column,
    find_proboscis_distance_column,
    find_proboscis_min_distance_column,
    find_proboscis_max_distance_column,
)
from fbpipe.utils.nanstats import (  # noqa: E402
    count_finite_contributors,
    nan_pad_stack,
    nanmean_sem,
)
from scripts.analysis.envelope_combined import (  # noqa: E402
    DIST_COLS,
    _canon_dataset,
    _display_odor,
    _infer_category,
    _locate_trials,
    _pick_column,
    _trial_label,
)

LOGGER = logging.getLogger("dataset_means")

TRIAL_NUM_RE = re.compile(r"(\d+)")

# ---------------------------------------------------------------------------
# Plotting style (consistent with existing repo conventions)
# ---------------------------------------------------------------------------
DPI = 300
FIGWIDTH = 8


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    png = base_path.with_suffix(".png")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", png)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _read_distance_pct(
    csv_path: Path,
    min_floor_px: float,
) -> Optional[np.ndarray]:
    """Read a trial CSV and return the Distance % trace as a 1-D array.

    If a ``distance_percentage`` column exists it is used directly.
    Otherwise the raw distance is normalised using min/max columns
    with an effective-minimum floor.  Returns *None* on failure.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        LOGGER.warning("Failed to read %s: %s", csv_path, exc)
        return None

    pct_col = find_proboscis_distance_percentage_column(df)
    if pct_col is not None:
        vals = pd.to_numeric(df[pct_col], errors="coerce").to_numpy(dtype=np.float64)
        return vals

    # Fallback: normalise raw distance with min/max columns
    dist_col_name = _pick_column(df, DIST_COLS)
    if dist_col_name is None:
        dist_col_name_alt = find_proboscis_distance_column(df)
        if dist_col_name_alt is None:
            LOGGER.debug("No distance column in %s", csv_path)
            return None
        dist_col_name = dist_col_name_alt

    raw = pd.to_numeric(df[dist_col_name], errors="coerce").to_numpy(dtype=np.float64)
    min_col = find_proboscis_min_distance_column(df)
    max_col = find_proboscis_max_distance_column(df)
    if min_col and max_col:
        raw_min = pd.to_numeric(df[min_col], errors="coerce").iloc[0]
        raw_max = pd.to_numeric(df[max_col], errors="coerce").iloc[0]
    else:
        raw_min = np.nanmin(raw)
        raw_max = np.nanmax(raw)

    eff_min = max(float(raw_min), min_floor_px)
    span = float(raw_max) - eff_min
    if span <= 0:
        LOGGER.debug("Zero span in %s (raw_max=%.2f, eff_min=%.2f)", csv_path, raw_max, eff_min)
        return None

    vals = 100.0 * (raw - eff_min) / span
    vals[raw < min_floor_px] = np.nan
    return vals


def _trial_number(label: str) -> int:
    m = TRIAL_NUM_RE.search(label)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def compute_dataset_means(
    dataset_root: Path,
    *,
    trial_type: str = "testing",
    min_floor_px: float = 9.5,
    fps: float = 40.0,
) -> dict[str, dict]:
    """Discover flies under *dataset_root* and aggregate per-odor mean/SEM.

    Returns a dict keyed by odor label with values::

        {
            "mean": np.ndarray,
            "sem": np.ndarray,
            "n_flies": int,
            "fly_names": list[str],
            "source_csvs": list[str],
        }

    Returns an empty dict if no usable data is found.
    """
    dataset_name = dataset_root.name
    dataset_canon = _canon_dataset(dataset_name)
    LOGGER.info("Dataset: %s (canon=%s)", dataset_name, dataset_canon)

    # odor_label -> list of per-fly mean traces
    odor_fly_traces: dict[str, list[np.ndarray]] = defaultdict(list)
    odor_fly_names: dict[str, list[str]] = defaultdict(list)
    odor_source_csvs: dict[str, list[str]] = defaultdict(list)

    fly_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if not fly_dirs:
        LOGGER.warning("No fly directories found in %s", dataset_root)
        return {}

    for fly_dir in fly_dirs:
        fly_name = fly_dir.name
        LOGGER.debug("  Fly: %s", fly_name)

        entries = _locate_trials(
            fly_dir,
            ("*fly*_distances.csv", "*_distances.csv"),
            DIST_COLS,
        )
        # Filter to requested trial type
        type_entries = [
            (label, path, cat)
            for label, path, cat in entries
            if cat == trial_type
        ]
        if not type_entries:
            LOGGER.debug("  No %s trials for %s", trial_type, fly_name)
            continue

        # Group trials by odor label
        odor_trials: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for label, path, _cat in type_entries:
            odor = _display_odor(dataset_canon, label)
            odor_trials[odor].append((label, path))

        for odor, trials in odor_trials.items():
            traces = []
            csv_paths = []
            for label, path in trials:
                trace = _read_distance_pct(path, min_floor_px)
                if trace is not None and len(trace) > 0 and np.any(np.isfinite(trace)):
                    traces.append(trace)
                    csv_paths.append(str(path))

            if not traces:
                continue

            # Per-fly mean across this fly's trials for the odor
            stacked = nan_pad_stack(traces)
            with np.errstate(all="ignore"):
                fly_mean = np.nanmean(stacked, axis=0)

            if np.any(np.isfinite(fly_mean)):
                odor_fly_traces[odor].append(fly_mean)
                odor_fly_names[odor].append(fly_name)
                odor_source_csvs[odor].extend(csv_paths)

    if not odor_fly_traces:
        LOGGER.warning("No usable data found for dataset %s", dataset_name)
        return {}

    results: dict[str, dict] = {}
    for odor in sorted(odor_fly_traces.keys()):
        fly_traces = odor_fly_traces[odor]
        stacked = nan_pad_stack(fly_traces)
        mean, sem = nanmean_sem(stacked)
        n_flies = count_finite_contributors(stacked)
        results[odor] = {
            "mean": mean,
            "sem": sem,
            "n_flies": n_flies,
            "fly_names": odor_fly_names[odor],
            "source_csvs": odor_source_csvs[odor],
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_dataset_means(
    results: dict[str, dict],
    *,
    dataset_name: str,
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
) -> Optional[plt.Figure]:
    """Create a single plot with all odors overlaid in different colours."""
    if not results:
        return None

    odors = list(results.keys())
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 5))

    for idx, odor in enumerate(odors):
        data = results[odor]
        mean = data["mean"]
        n_flies = data["n_flies"]
        n_frames = len(mean)
        time = np.arange(n_frames) / fps
        colour = f"C{idx % 10}"

        ax.plot(time, mean, linewidth=1.3, color=colour, label=f"{odor} (n={n_flies})")

    # Odor window
    ax.axvline(odor_on_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(odor_off_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.12, color="grey")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance %")
    ax.set_title(f"{dataset_name} \u2014 Testing Odors Mean", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# JSON sidecar
# ---------------------------------------------------------------------------


def write_sidecar(
    path: Path,
    *,
    dataset_name: str,
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    min_floor_px: float,
    trial_type: str,
    results: dict[str, dict],
) -> None:
    info: dict = {
        "dataset": dataset_name,
        "fps": fps,
        "odor_on_s": odor_on_s,
        "odor_off_s": odor_off_s,
        "min_floor_px": min_floor_px,
        "trial_type": trial_type,
        "n_definition": "count of flies with at least one finite value in the trace",
        "odors": {},
    }
    for odor, data in results.items():
        info["odors"][odor] = {
            "n_flies": data["n_flies"],
            "fly_names": data["fly_names"],
            "n_source_csvs": len(data["source_csvs"]),
            "n_timepoints": len(data["mean"]),
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    LOGGER.info("Saved sidecar %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/home/ramanlab/Documents/cole/Results/Opto/Dataset_means"),
        help="Output directory for plots and sidecars",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override frames per second")
    parser.add_argument("--odor-on-s", type=float, default=None, help="Odor onset in seconds")
    parser.add_argument("--odor-off-s", type=float, default=None, help="Odor offset in seconds")
    parser.add_argument(
        "--trial-type",
        default="testing",
        help="Trial type to aggregate (default: testing)",
    )
    parser.add_argument(
        "--min-floor-px",
        type=float,
        default=9.5,
        help="Minimum distance floor in pixels for normalisation fallback",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit dataset roots (overrides config)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args(argv)


def _discover_roots(args: argparse.Namespace) -> list[Path]:
    """Return dataset root directories from CLI args or config."""
    if args.roots:
        return [Path(r).expanduser().resolve() for r in args.roots]

    import yaml

    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        LOGGER.error("Config not found: %s", config_path)
        return []

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # Primary: analysis.combined.combine.roots
    combine = data.get("analysis", {}).get("combined", {}).get("combine", {})
    roots = combine.get("roots", [])
    if roots:
        return [Path(r).expanduser().resolve() for r in roots]

    # Fallback: main_directories / main_directory
    md = data.get("main_directories", data.get("main_directory", []))
    if isinstance(md, str):
        md = [md]
    return [Path(r).expanduser().resolve() for r in md]


def _read_timing(args: argparse.Namespace) -> tuple[float, float, float]:
    """Return (fps, odor_on_s, odor_off_s) from CLI overrides or config."""
    import yaml

    fps = args.fps
    odor_on = args.odor_on_s
    odor_off = args.odor_off_s

    if fps is not None and odor_on is not None and odor_off is not None:
        return fps, odor_on, odor_off

    config_path = resolve_config_path(args.config)
    cfg_data: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg_data = yaml.safe_load(fh) or {}

    combine = cfg_data.get("analysis", {}).get("combined", {}).get("combine", {})

    if fps is None:
        fps = combine.get("fps_default", cfg_data.get("fps_default", 40.0))
    if odor_on is None:
        odor_on = combine.get("odor_on", cfg_data.get("odor_on_s", 30.0))
    if odor_off is None:
        odor_off = combine.get("odor_off", cfg_data.get("odor_off_s", 60.0))

    return float(fps), float(odor_on), float(odor_off)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser(argv)
    _configure_logging(args.verbose)

    roots = _discover_roots(args)
    if not roots:
        LOGGER.error("No dataset roots found. Check --roots or config.")
        sys.exit(1)

    fps, odor_on_s, odor_off_s = _read_timing(args)
    LOGGER.info("FPS=%.1f  odor_on=%.1fs  odor_off=%.1fs", fps, odor_on_s, odor_off_s)

    outdir = args.outdir.expanduser().resolve()
    _ensure_directory(outdir)

    processed = 0
    for root in roots:
        if not root.is_dir():
            LOGGER.warning("Skipping non-existent root: %s", root)
            continue

        dataset_name = root.name
        LOGGER.info("=== Dataset: %s ===", dataset_name)

        results = compute_dataset_means(
            root,
            trial_type=args.trial_type,
            min_floor_px=args.min_floor_px,
            fps=fps,
        )

        if not results:
            LOGGER.warning("No data for %s — skipping.", dataset_name)
            continue

        base = outdir / f"{dataset_name}_{args.trial_type}_odors_mean_sem"
        fig = plot_dataset_means(
            results,
            dataset_name=dataset_name,
            fps=fps,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
        )
        if fig is not None:
            _save_figure(fig, base)

        write_sidecar(
            base.with_suffix(".json"),
            dataset_name=dataset_name,
            fps=fps,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
            min_floor_px=args.min_floor_px,
            trial_type=args.trial_type,
            results=results,
        )
        processed += 1

    LOGGER.info("Done. Processed %d dataset(s).", processed)


if __name__ == "__main__":
    main()
