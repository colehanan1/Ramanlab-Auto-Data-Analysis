#!/usr/bin/env python3
"""Run the YOLO inference step across N parallel worker processes.

The fly pipeline is CPU-bound (decode + tracking + H.264 encode run serially per
frame) while the GPU sits at ~22% util, so batching alone barely helps wall-clock.
This driver fills the idle CPU cores / GPU headroom by running N copies of the
YOLO step concurrently. Each worker runs::

    python -m fbpipe.pipeline --config <cfg> yolo

with ``NUM_WORKERS``/``WORKER_INDEX`` set so it processes a disjoint 1/N slice of
the deterministically-ordered video list (see yolo_infer.main). Combined with a
small-batch dynamic engine (so N TensorRT contexts fit in VRAM) and NVENC encode
(so encode leaves the CPU), this scales throughput until the GPU or CPU saturates.

The per-video output-folder skip is the safety net: re-running is idempotent, and
workers never collide because their video slices are disjoint.

Usage:
    python scripts/pipeline/parallel_yolo.py --config config/config_new.yaml --workers 5

Run inside the project's conda env, e.g.:
    conda run -n yolo-env python scripts/pipeline/parallel_yolo.py --config config/config_new.yaml --workers 5

After this finishes, run the full workflow to do the downstream analysis steps;
it will skip the already-processed videos:
    python scripts/pipeline/run_workflows.py --config config/config_new.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the YOLO step across N parallel workers.")
    ap.add_argument("--config", required=True, help="Pipeline config YAML.")
    # 3 is the measured sweet spot on a 16-core / RTX-3090 box: 4+ workers fall off
    # an oversubscription cliff (decode-thread contention -> ~1M ctx-switches/s,
    # throughput COLLAPSES). 3 workers ~= 3x serial at ~120k ctx-switches/s, ~18GB
    # VRAM. Re-measure (vmstat sy%, throughput) if cores/GPU differ.
    ap.add_argument("--workers", type=int, default=3, help="Number of parallel worker processes.")
    ap.add_argument(
        "--logdir",
        default=str(REPO_ROOT / ".parallel_yolo_logs"),
        help="Directory for per-worker logs (default: <repo>/.parallel_yolo_logs).",
    )
    args = ap.parse_args()

    n = max(1, int(args.workers))
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # src/ on PYTHONPATH so `-m fbpipe.pipeline` resolves regardless of install mode.
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + base_env.get("PYTHONPATH", "")
    base_env["PYTHONUNBUFFERED"] = "1"  # flush worker prints so logs are live

    # Cap each worker's CPU-library thread pools to cores/N. Without this, every
    # worker's OpenCV/BLAS/torch each grab ALL cores, so N workers oversubscribe
    # (e.g. 4 workers spawned ~200 threads on 16 cores -> 60% of CPU lost to kernel
    # context-switching). Sharing the cores cleanly keeps useful work high.
    cores = os.cpu_count() or 16
    threads_per_worker = max(1, cores // n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        base_env[var] = str(threads_per_worker)
    print(f"[parallel-yolo] {cores} cores / {n} workers -> {threads_per_worker} CPU threads per worker", flush=True)

    procs = []
    t0 = time.time()
    print(f"[parallel-yolo] launching {n} workers over config {args.config}", flush=True)
    for k in range(n):
        env = dict(base_env, NUM_WORKERS=str(n), WORKER_INDEX=str(k))
        env.pop("MAIN_DIRECTORY", None)  # each worker must see ALL dirs to partition
        logpath = logdir / f"worker_{k}.log"
        logf = open(logpath, "w")
        cmd = [sys.executable, "-m", "fbpipe.pipeline", "--config", args.config, "yolo"]
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((k, p, logf, logpath))
        print(f"[parallel-yolo] worker {k+1}/{n} pid={p.pid} -> {logpath}", flush=True)

    failures = []
    for k, p, logf, logpath in procs:
        rc = p.wait()
        logf.close()
        text = logpath.read_text(errors="ignore")
        processed = _count(r"→ .* in [0-9].*s|-> .* in [0-9].*s", text) or _count(r" in [0-9.]+s \(per-fly", text)
        skipped = _count(r"Skipping ", text)
        status = "OK" if rc == 0 else f"FAILED(rc={rc})"
        print(f"[parallel-yolo] worker {k}: {status} | processed≈{processed} skipped≈{skipped}", flush=True)
        if rc != 0:
            failures.append(k)
            # surface the tail of a failed worker's log for quick diagnosis
            tail = "\n".join(text.splitlines()[-15:])
            print(f"[parallel-yolo] --- worker {k} log tail ---\n{tail}\n--- end ---", flush=True)

    dt = time.time() - t0
    total_proc = sum(
        _count(r" in [0-9.]+s \(per-fly", (logdir / f"worker_{k}.log").read_text(errors="ignore"))
        for k in range(n)
    )
    print(
        f"[parallel-yolo] DONE: {n} workers in {dt:.1f}s ({dt/60:.1f} min); "
        f"~{total_proc} videos processed; failures: {failures or 'none'}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
