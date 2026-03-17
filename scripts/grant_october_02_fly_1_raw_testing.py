#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import EnvelopePlotConfig, generate_envelope_plots

# Grant-focused single-fly renderer. Edit these constants directly if you want
# to keep iterating on the same figure without touching pipeline defaults.
TARGET_FLY = "october_02_fly_1"
TARGET_FLY_NUMBER = "1"
MATRIX_NPY = Path("/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/matrix/raw/envelope_matrix_float16.npy")
CODES_JSON = Path("/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/matrix/raw/code_maps.json")
DEFAULT_OUT_DIR = Path("/tmp/grant-raw-testing-october_02_fly_1")

STYLE = {
    "style_scale": 2.0,
    "trace_linewidth_scale": 2.0,
    "panel_title_scale": 0.9,
    "figure_title_scale": 0.72,
    "figure_subtitle_scale": 0.5,
    "legend_scale": 0.675,
    "plot_size_scale": 1.2,
    "single_ylabel_trial_num": 5,
    "fixed_y_max": 105.0,
    "y_label_override": "Max Distance x Angle %",
    "show_legend": False,
    "show_figure_title": False,
    "show_figure_subtitle": False,
    "legend_anchor_y": 0.895,
    "figure_title_x": 0.5,
    "figure_title_y": 0.925,
    "figure_title_ha": "center",
    "figure_subtitle_x": 0.5,
    "figure_subtitle_y": 0.905,
    "figure_subtitle_ha": "center",
    "panel_title_x": 0.01,
    "panel_title_y": 105.0,
    "panel_title_va": "top",
    "panel_title_use_data_y": True,
    "odor_on_label_trial_num": 1,
    "odor_on_label_text": "ODOR ON",
    "odor_on_label_scale": 1.8,
    "odor_on_label_y": 85.0,
    "tight_h_pad": 0.12,
}


def build_config(out_dir: Path, *, overwrite: bool) -> EnvelopePlotConfig:
    return EnvelopePlotConfig(
        matrix_npy=MATRIX_NPY,
        codes_json=CODES_JSON,
        out_dir=out_dir,
        latency_sec=0.0,
        fps_default=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
        odor_latency_s=2.15,
        after_show_sec=30.0,
        threshold_std_mult=3.0,
        trial_type="testing",
        fly_filter=TARGET_FLY,
        fly_number_filter=TARGET_FLY_NUMBER,
        overwrite=overwrite,
        **STYLE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the combined-base Raw-Testing-PER-Traces figure for october_02_fly_1 fly1."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Destination directory for the rendered figure.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the figure even if the target PNG already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    cfg = build_config(out_dir, overwrite=args.overwrite)
    generate_envelope_plots(cfg)
    output_png = (
        out_dir
        / "Hex-Training"
        / f"{TARGET_FLY}_fly{TARGET_FLY_NUMBER}_testing_envelope_trials_by_odor_30_shifted.png"
    )
    print(output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
