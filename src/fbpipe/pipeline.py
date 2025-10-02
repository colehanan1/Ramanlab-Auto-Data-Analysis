
from __future__ import annotations
import argparse, sys
from pathlib import Path
from .config import load_settings
from .steps import (
    yolo_infer, distance_stats, distance_normalize, plot_distance_time,
    detect_dropped_frames, rms_copy_filter, update_ofm_state, histograms,
    envelope_over_time, collect_envelope_plots, angle_compute_and_plots,
    angle_heatmaps, move_videos, compose_videos_rms
)

STEPS_IN_ORDER = [
    yolo_infer.main,
    distance_stats.main,
    distance_normalize.main,
    plot_distance_time.main,
    detect_dropped_frames.main,
    rms_copy_filter.main,
    update_ofm_state.main,
    histograms.main,
    envelope_over_time.main,
    collect_envelope_plots.main,
    angle_compute_and_plots.main,
    angle_heatmaps.main,
    move_videos.main,
    compose_videos_rms.main,
]

def run_all(config_path: Path):
    cfg = load_settings(config_path)
    for step in STEPS_IN_ORDER:
        step(cfg)

def main():
    p = argparse.ArgumentParser(description="Fly Behavior Pipeline")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("cmd", nargs="*", help="all or a list of steps")
    args = p.parse_args()

    if not args.cmd or args.cmd == ["all"]:
        run_all(args.config)
        return

    # selective execution
    cfg = load_settings(args.config)
    name2step = {
        "yolo": yolo_infer.main,
        "distance_stats": distance_stats.main,
        "distance_normalize": distance_normalize.main,
        "plot_distance_time": plot_distance_time.main,
        "detect_dropped_frames": detect_dropped_frames.main,
        "rms_copy_filter": rms_copy_filter.main,
        "update_ofm_state": update_ofm_state.main,
        "histograms": histograms.main,
        "envelope": envelope_over_time.main,
        "collect_envelope": collect_envelope_plots.main,
        "angle": angle_compute_and_plots.main,
        "heatmaps": angle_heatmaps.main,
        "move_videos": move_videos.main,
        "compose_videos_rms": compose_videos_rms.main,
    }
    for cmd in args.cmd:
        f = name2step.get(cmd)
        if f is None:
            print(f"Unknown step: {cmd}", file=sys.stderr)
            continue
        f(cfg)

if __name__ == "__main__":
    main()
