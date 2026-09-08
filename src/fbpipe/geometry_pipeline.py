"""CLI of the abdomen geometry pipeline (adjacent to ``fbpipe.pipeline``).

    conda run -n yolo-env python -m fbpipe.geometry_pipeline --config config/config_abdomen.yaml [STEP ...]

Steps run in order: track -> normalize -> wide -> figures. ``list`` prints them.
"""
from __future__ import annotations

import argparse
import inspect
from typing import Iterable, Optional, Sequence

from .geometry import steps

STEP_NAMES = ("track", "normalize", "wide", "figures")
DESCRIPTIONS = {
    "track": "Run the 4-class model + fly_geometry rules per trial video; per-fly geometry tables + annotated mp4",
    "normalize": "Per-fly stats over the batch (PER min/max, angle min/max, resting size) and the % columns",
    "wide": "Wide tables (production column order, dir_val_*) per measure",
    "figures": "Per-fly PER + abdomen traces, size traces, QC figure and the trial summary CSV",
}


def run_steps(names: Iterable[str], settings, gcfg, **options) -> None:
    for name in names:
        runner = getattr(steps, f"run_{name}")
        accepted = inspect.signature(runner).parameters
        kwargs = {k: v for k, v in options.items() if k in accepted}
        runner(settings, gcfg, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Abdomen geometry pipeline")
    p.add_argument("--config", required=True, help="e.g. config/config_abdomen.yaml")
    p.add_argument("--force", action="store_true", help="re-track trials that already have outputs")
    p.add_argument("--only-trial", default=None, help="substring of the trial folder name, e.g. pretest_1")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames per trial (smoke tests)")
    p.add_argument("steps", nargs="*", metavar="STEP", help=f"subset of {STEP_NAMES}; 'list' shows them")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps == ["list"]:
        for name in STEP_NAMES:
            print(f"{name:>10}  - {DESCRIPTIONS[name]}")
        return 0
    unknown = [s for s in args.steps if s not in STEP_NAMES]
    if unknown:
        raise SystemExit(f"Unknown step(s): {', '.join(unknown)}; choose from {STEP_NAMES}")
    from .geometry.config import load_geometry_config
    settings, gcfg = load_geometry_config(args.config)
    names = args.steps or list(STEP_NAMES)
    run_steps(names, settings, gcfg, force=args.force, only_trial=args.only_trial, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
