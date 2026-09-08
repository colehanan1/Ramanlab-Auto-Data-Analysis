"""CLI of the abdomen geometry pipeline (adjacent to ``fbpipe.pipeline``).

    conda run -n yolo-env python -m fbpipe.geometry_pipeline --config config/config_abdomen.yaml [STEP ...]

Steps run in order: track -> normalize -> wide -> figures. ``list`` prints them.
"""
from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .geometry import steps

REPO_ROOT = Path(__file__).resolve().parents[2]

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


def worker_commands(config_path: Path, n_workers: int, *, cores: Optional[int] = None,
                    options: Optional[Dict[str, object]] = None) -> List[Tuple[List[str], Dict[str, str]]]:
    """One ``(command, env)`` per worker for the track step: each worker takes a disjoint
    1/N slice of the video list (``NUM_WORKERS``/``WORKER_INDEX``, read by ``steps.worker_shard``)
    and a share of the cores, exactly like ``scripts/pipeline/parallel_yolo.py``."""
    options = dict(options or {})
    n = max(1, int(n_workers))
    cores = int(cores or os.cpu_count() or n)
    threads = max(1, cores // n)
    base_env = dict(os.environ)
    base_env.pop("MAIN_DIRECTORY", None)          # every worker must see ALL dirs to partition
    base_env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + base_env.get("PYTHONPATH", "")
    base_env["PYTHONUNBUFFERED"] = "1"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        base_env[var] = str(threads)
    cmd = [sys.executable, "-m", "fbpipe.geometry_pipeline", "--config", str(config_path), "--no-fan-out"]
    if options.get("force"):
        cmd.append("--force")
    if options.get("only_trial"):
        cmd += ["--only-trial", str(options["only_trial"])]
    if options.get("max_frames") is not None:
        cmd += ["--max-frames", str(int(options["max_frames"]))]
    cmd.append("track")
    return [(list(cmd), dict(base_env, NUM_WORKERS=str(n), WORKER_INDEX=str(k))) for k in range(n)]


def fan_out_track(config_path: Path, n_workers: int, *, settings, options: Dict[str, object]) -> int:
    """Run the track step in ``n_workers`` processes; returns the number of failed workers."""
    logdir = Path(getattr(settings, "cache_dir", "") or (REPO_ROOT / "cache")) / "geometry_workers"
    logdir.mkdir(parents=True, exist_ok=True)
    procs = []
    print(f"[GEOMETRY] track: fanning out to {n_workers} worker(s); logs in {logdir}")
    for k, (cmd, env) in enumerate(worker_commands(config_path, n_workers, options=options)):
        log_path = logdir / f"worker_{k}.log"
        logf = open(log_path, "w")
        procs.append((k, subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT), logf, log_path))
        print(f"[GEOMETRY] worker {k + 1}/{n_workers} pid={procs[-1][1].pid} -> {log_path}")
    failed = 0
    for k, proc, logf, log_path in procs:
        rc = proc.wait()
        logf.close()
        text = log_path.read_text(errors="ignore")
        tracked = text.count(" slot(s), ")
        print(f"[GEOMETRY] worker {k + 1}/{n_workers}: {'OK' if rc == 0 else f'FAILED rc={rc}'} | tracked~{tracked}")
        if rc != 0:
            failed += 1
            print("\n".join(text.splitlines()[-15:]))
    return failed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Abdomen geometry pipeline")
    p.add_argument("--config", required=True, help="e.g. config/config_abdomen.yaml")
    p.add_argument("--force", action="store_true", help="re-track trials that already have outputs")
    p.add_argument("--only-trial", default=None, help="substring of the trial folder name, e.g. pretest_1")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames per trial (smoke tests)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes for the track step (default: yolo.num_workers in the config)")
    p.add_argument("--no-fan-out", action="store_true", help=argparse.SUPPRESS)   # set on the workers themselves
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
    from .geometry import config as geometry_config
    settings, gcfg = geometry_config.load_geometry_config(args.config)
    names = list(args.steps or STEP_NAMES)
    options = {"force": args.force, "only_trial": args.only_trial, "max_frames": args.max_frames}
    n_workers = args.workers if args.workers is not None else int(getattr(settings, "yolo_num_workers", 1) or 1)
    if "track" in names and n_workers > 1 and not args.no_fan_out:
        failed = fan_out_track(Path(args.config), n_workers, settings=settings, options=options)
        if failed:
            raise SystemExit(f"[GEOMETRY] {failed} track worker(s) failed; see the worker logs")
        names = [n for n in names if n != "track"]
    run_steps(names, settings, gcfg, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
