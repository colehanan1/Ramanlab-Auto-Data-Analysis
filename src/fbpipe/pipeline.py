from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable

from .config import Settings, load_settings

# GPU-accelerated imports: Auto-detect CUDA and use GPU versions when available
import torch

USE_GPU = torch.cuda.is_available()

if USE_GPU:
    print("[PIPELINE] 🚀 GPU acceleration ENABLED (CUDA detected)")
    print("[PIPELINE] ⚡ Using BATCH processing (12-15x speedup)")
    from .steps import (
        calculate_acceleration_gpu as calculate_acceleration,
        compose_videos_rms,
        detect_dropped_frames,
        distance_normalize_ultra as distance_normalize,  # ULTRA: Batch processing
        distance_stats,
        move_videos,
        rms_copy_filter,
        update_ofm_state,
        yolo_infer,
    )
else:
    print("[PIPELINE] ⚠️  GPU acceleration DISABLED (CUDA not available, using CPU)")
    from .steps import (
        calculate_acceleration,
        compose_videos_rms,
        detect_dropped_frames,
        distance_normalize,
        distance_stats,
        move_videos,
        rms_copy_filter,
        update_ofm_state,
        yolo_infer,
    )


@dataclass(frozen=True)
class Step:
    """Executable unit within the pipeline."""

    name: str
    runner: Callable[[Settings], None]
    description: str


# The order emphasises dependencies: normalization relies on the distance
# stats, OFM state annotations require the RMS copies, and the video overlay
# comes last so that all metadata is already embedded in the CSVs.
# Modification #4: Added calculate_acceleration step after compose_videos_rms
ORDERED_STEPS: tuple[Step, ...] = (
    Step("yolo", yolo_infer.main, "Run Ultralytics YOLO inference and export merged CSVs"),
    Step("distance_stats", distance_stats.main, "Derive global class-2 distance bounds per fly"),
    Step("distance_normalize", distance_normalize.main, "Convert distances into percentage scores"),
    Step("detect_dropped_frames", detect_dropped_frames.main, "Report missing frames and NaN segments"),
    Step("rms_copy_filter", rms_copy_filter.main, "Copy curated columns into RMS_calculations"),
    Step("update_ofm_state", update_ofm_state.main, "Annotate RMS tables with OFM state transitions"),
    Step("move_videos", move_videos.main, "Stage annotated videos into the delivery directory"),
    Step("compose_videos_rms", compose_videos_rms.main, "Render RMS overlays onto exported videos"),
    Step("calculate_acceleration", calculate_acceleration.main, "Calculate frame-to-frame acceleration metrics"),
)

STEP_REGISTRY = {step.name: step for step in ORDERED_STEPS}


def run_steps(step_names: Iterable[str], config_path: str | Path) -> None:
    cfg = load_settings(config_path)
    for name in step_names:
        step = STEP_REGISTRY[name]
        step.runner(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fly Behavior Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to pipeline configuration")
    parser.add_argument(
        "steps",
        nargs="*",
        metavar="STEP",
        help="Subset of steps to run. Use 'list' to display available steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.steps or args.steps == ["all"]:
        run_steps((step.name for step in ORDERED_STEPS), args.config)
        return

    if args.steps == ["list"]:
        for step in ORDERED_STEPS:
            print(f"{step.name:>20}  - {step.description}")
        return

    unknown = [name for name in args.steps if name not in STEP_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown step(s): {', '.join(unknown)}")

    run_steps(args.steps, args.config)


if __name__ == "__main__":
    main()
