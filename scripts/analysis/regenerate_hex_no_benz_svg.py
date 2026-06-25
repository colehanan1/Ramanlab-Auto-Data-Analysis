"""Regenerate the Hex_plots_no_benz figures as editable SVG + PNG.

Reproduces *exactly* the figure set currently living under
``Opto-Fly-Figures-OctNov/Matrix-PER-Reactions-Model/Hex_plots_no_benz`` —
Benzaldehyde excluded everywhere — but writes each figure twice:

* ``<name>.png`` (300 dpi, raster, unchanged appearance)
* ``<name>.svg`` (``svg.fonttype="none"`` so every text label, line and patch
  stays an editable vector object in Illustrator / Inkscape)

It reuses the plotting helpers from ``dataset_means_specific_flies`` so the
output is identical to the originals; only the file formats change.

Binary-reactions CSVs (the fly lists) are read from the *parent*
``Matrix-PER-Reactions-Model/<dataset>/`` folder, where they actually live.

Run::

    /home/ramanlab/anaconda3/bin/python \
        scripts/analysis/regenerate_hex_no_benz_svg.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fbpipe.utils.tables import read_table
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.config import resolve_config_path  # noqa: E402
from scripts.analysis.dataset_means import (  # noqa: E402
    compute_dataset_means,
    plot_dataset_means,
    write_sidecar,
)
from scripts.analysis.dataset_means_specific_flies import (  # noqa: E402
    MAX_TIME_S,
    _canon_dataset,
    _collect_per_fly_first_presentation,
    _display_name,
    _filter_to_flies,
    _load_specific_flies,
    _mean_trace,
    _odor_first_presentation_trial,
    _plot_all_odors_means,
    _plot_odor_avg_with_individuals,
    _plot_training_vs_control_for_odor,
    _safe_odor_filename,
    _select_wide_csv,
    _shared_ylim_from_means,
    _shared_ylim_from_per_fly,
)

LOGGER = logging.getLogger("regenerate_hex_no_benz_svg")

DPI = 300
EXCLUDE = {"Benzaldehyde"}

# Figures live here; binary-reactions CSVs (fly lists) come from the parent.
PARENT_DIR = Path(
    "/home/ramanlab/Documents/cole/Results/Opto-Fly-Figures-OctNov/Matrix-PER-Reactions-Model"
)
DEFAULT_OUT_DIR = PARENT_DIR / "Hex_plots_no_benz"
DATASETS = ("Hex-Control", "Hex-Training")


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _savefig(fig: plt.Figure, base: Path) -> None:
    """Write ``base.png`` (300 dpi) and an editable ``base.svg``."""
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s (+ .svg)", png.name)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--parent-dir", type=Path, default=PARENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--fps", type=float, default=40.0)
    parser.add_argument("--odor-on-s", type=float, default=30.0)
    parser.add_argument("--odor-off-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    _configure_logging()

    # Keep text editable in the SVGs and stay on Arial.
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cfg = _load_yaml(resolve_config_path(args.config))
    wide_csv = _select_wide_csv(cfg)
    if not wide_csv.exists():
        LOGGER.error("Wide CSV not found: %s", wide_csv)
        sys.exit(1)
    LOGGER.info("Wide CSV: %s", wide_csv)
    wide_df = read_table(wide_csv)

    fps = args.fps
    odor_on_s = args.odor_on_s
    odor_off_s = args.odor_off_s

    # ------------------------------------------------------------------
    # Pass 1: per dataset — overlay figure + collect per-fly traces.
    # ------------------------------------------------------------------
    prepared_all: dict[str, dict] = {}
    for ds in args.datasets:
        binary_csv = args.parent_dir / ds / f"binary_reactions_{ds}_unordered.csv"
        if not binary_csv.exists():
            LOGGER.error("Missing %s — skipping %s.", binary_csv, ds)
            continue
        out_ds = args.out_dir / ds
        out_ds.mkdir(parents=True, exist_ok=True)

        flies = _load_specific_flies(binary_csv)
        LOGGER.info("=== %s : %d flies ===", ds, len(flies))

        ds_df = wide_df[wide_df["dataset"] == ds].copy()
        ds_df = ds_df[ds_df["trial_type"] == "testing"]
        ds_df = _filter_to_flies(ds_df, flies)
        if ds_df.empty:
            LOGGER.warning("No rows for %s — skipping.", ds)
            continue

        results = compute_dataset_means(
            ds_df, ds, excluded_flies=set(), fps=fps,
            odor_on_s=odor_on_s, subtract_baseline=True,
        )

        # Overlay mean plot (Benzaldehyde excluded).
        overlay_results = {o: d for o, d in results.items() if o not in EXCLUDE}
        overlay_base = out_ds / f"{ds}_testing_odors_mean_specific_flies"
        fig = plot_dataset_means(
            overlay_results,
            dataset_name=f"{_display_name(ds)} (specific flies)",
            fps=fps, odor_on_s=odor_on_s, odor_off_s=odor_off_s,
            baseline_subtracted=True, trial_type="testing",
        )
        if fig is not None:
            _savefig(fig, overlay_base)
        write_sidecar(
            overlay_base.with_suffix(".json"),
            dataset_name=ds, fps=fps, odor_on_s=odor_on_s, odor_off_s=odor_off_s,
            trial_type="testing", results=results, baseline_subtracted=True,
        )

        dataset_canon = _canon_dataset(ds)
        baseline_frames = max(1, int(round(float(odor_on_s) * float(fps))))
        dir_cols = sorted(
            [c for c in ds_df.columns if c.startswith("dir_val_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        per_odor_per_fly, odor_trials_by_fly = _collect_per_fly_first_presentation(
            ds_df, dataset_canon, baseline_frames=baseline_frames, dir_cols=dir_cols,
        )
        prepared_all[ds] = {
            "dataset_name": ds,
            "binary_csv": binary_csv,
            "outdir": out_ds,
            "per_odor_per_fly": per_odor_per_fly,
            "odor_trials_by_fly": odor_trials_by_fly,
            "n_selected_flies": len(flies),
        }

    if not prepared_all:
        LOGGER.error("Nothing prepared — aborting.")
        sys.exit(1)

    # Shared y-limits computed over ALL odors (incl. Benzaldehyde) so the
    # regenerated axes match the originals exactly.
    shared_ylim = _shared_ylim_from_per_fly(
        {ds: p["per_odor_per_fly"] for ds, p in prepared_all.items()}, fps=fps
    )
    LOGGER.info("Shared per-odor ylim: %s", shared_ylim)

    max_frames = int(MAX_TIME_S * fps)
    all_mean_traces = [
        _mean_trace(pf, max_frames)
        for p in prepared_all.values()
        for pf in p["per_odor_per_fly"].values()
    ]
    mean_ylim = _shared_ylim_from_means(all_mean_traces, fps=fps)
    LOGGER.info("Shared mean-only ylim: %s", mean_ylim)

    # ------------------------------------------------------------------
    # Pass 2: per-odor "avg + individuals" + all-odor-means overlay.
    # ------------------------------------------------------------------
    for ds, prep in prepared_all.items():
        out_ds: Path = prep["outdir"]
        per_odor_per_fly = prep["per_odor_per_fly"]
        odor_trials_by_fly = prep["odor_trials_by_fly"]

        odor_dir = out_ds / "per_odor_first_presentation"
        per_odor_meta: dict[str, dict] = {}
        for odor, per_fly in per_odor_per_fly.items():
            if odor in EXCLUDE:
                continue
            target_trial = _odor_first_presentation_trial(odor, odor_trials_by_fly)
            assert target_trial is not None
            base = odor_dir / (
                f"{ds}_{_safe_odor_filename(odor)}_trial{target_trial}_avg_individuals"
            )
            fig = _plot_odor_avg_with_individuals(
                odor=odor, per_fly=per_fly, fps=fps,
                odor_on_s=odor_on_s, odor_off_s=odor_off_s,
                dataset_name=ds, trial_num=target_trial, ylim=shared_ylim,
            )
            _savefig(fig, base)
            per_odor_meta[odor] = {
                "trial_num": target_trial,
                "n_flies": len(per_fly),
                "fly_ids": sorted(per_fly.keys()),
            }
        odor_dir.mkdir(parents=True, exist_ok=True)
        with open(odor_dir / f"{ds}_per_odor_first_presentation.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "dataset": ds,
                    "binary_reactions_csv": str(prep["binary_csv"]),
                    "fps": fps, "odor_on_s": odor_on_s, "odor_off_s": odor_off_s,
                    "n_selected_flies": prep["n_selected_flies"],
                    "shared_ylim": list(shared_ylim),
                    "excluded_odors": sorted(EXCLUDE),
                    "per_odor": per_odor_meta,
                },
                fh, indent=2,
            )

        # All-odor means overlay (Benzaldehyde already dropped inside the plot).
        fig = _plot_all_odors_means(
            per_odor_per_fly=per_odor_per_fly, fps=fps,
            odor_on_s=odor_on_s, odor_off_s=odor_off_s,
            dataset_name=ds, ylim=mean_ylim,
        )
        _savefig(fig, out_ds / f"{ds}_all_odor_means_first_presentation")

    # ------------------------------------------------------------------
    # Pass 3: Training vs Control per-odor comparisons (Benz excluded).
    # ------------------------------------------------------------------
    def _role(name: str) -> str | None:
        low = name.lower()
        if "training" in low:
            return "training"
        if "control" in low:
            return "control"
        return None

    by_role: dict[str, dict] = {}
    for ds, prep in prepared_all.items():
        role = _role(ds)
        if role is not None:
            by_role.setdefault(role, {})[ds] = prep

    if "training" in by_role and "control" in by_role:
        train_ds = next(iter(by_role["training"]))
        ctrl_ds = next(iter(by_role["control"]))
        train_prep = by_role["training"][train_ds]
        ctrl_prep = by_role["control"][ctrl_ds]
        cmp_dir = args.out_dir / "Training_vs_Control"
        cmp_dir.mkdir(parents=True, exist_ok=True)

        common_odors = sorted(
            (set(train_prep["per_odor_per_fly"]) & set(ctrl_prep["per_odor_per_fly"]))
            - EXCLUDE
        )
        comparison_meta: dict[str, dict] = {}
        for odor in common_odors:
            fig = _plot_training_vs_control_for_odor(
                odor=odor,
                train_per_fly=train_prep["per_odor_per_fly"][odor],
                ctrl_per_fly=ctrl_prep["per_odor_per_fly"][odor],
                fps=fps, odor_on_s=odor_on_s, odor_off_s=odor_off_s, ylim=mean_ylim,
            )
            _savefig(fig, cmp_dir / f"{_safe_odor_filename(odor)}_training_vs_control")
            comparison_meta[odor] = {
                "n_training_flies": len(train_prep["per_odor_per_fly"][odor]),
                "n_control_flies": len(ctrl_prep["per_odor_per_fly"][odor]),
            }
        with open(cmp_dir / "training_vs_control.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "training_dataset": train_ds, "control_dataset": ctrl_ds,
                    "fps": fps, "odor_on_s": odor_on_s, "odor_off_s": odor_off_s,
                    "shared_mean_ylim": list(mean_ylim),
                    "excluded_odors": sorted(EXCLUDE),
                    "per_odor": comparison_meta,
                },
                fh, indent=2,
            )

    LOGGER.info("Done. Output: %s", args.out_dir)


if __name__ == "__main__":
    main()
