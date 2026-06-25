#!/usr/bin/env python3
"""Benchmark harness for the Parquet/parallel optimizations.

Measures three things, on real data when available:

1. I/O format: read/write/size of CSV vs Parquet for a per-fly distance file.
2. parallel_map speedup on a CPU-bound workload.
3. End-to-end stage chain (distance_stats -> distance_normalize) serial vs
   parallel, on a fixture replicated from a real per-fly distance file.

Usage:
    python scripts/bench_pipeline.py [--sample CSV] [--flies N]

This script only READS your data (a sample CSV) and writes throwaway fixtures
to a temp dir; it never touches your dataset.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = (
    REPO / "engine_test_output" / "april_13_batch_1_training_1"
    / "april_13_batch_1_training_1_fly1_distances.csv"
)


def _best(fn, n=5):
    best = math.inf
    out = None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def bench_io(sample: Path) -> None:
    print("\n=== 1. I/O format (best of 5) ===")
    if not sample.is_file():
        print(f"  (skipped: sample not found at {sample})")
        return
    df = pd.read_csv(sample)
    tmp = Path(tempfile.mkdtemp())
    csv_o, pq_o = tmp / "o.csv", tmp / "o.parquet"
    t_rcsv, _ = _best(lambda: pd.read_csv(sample))
    t_wcsv, _ = _best(lambda: df.to_csv(csv_o, index=False))
    t_wpq, _ = _best(lambda: df.to_parquet(pq_o, engine="pyarrow", index=False))
    t_rpq, _ = _best(lambda: pd.read_parquet(pq_o, engine="pyarrow"))
    sz_csv, sz_pq = csv_o.stat().st_size, pq_o.stat().st_size
    print(f"  shape={df.shape}")
    print(f"  read : csv {t_rcsv*1e3:7.2f}ms | parquet {t_rpq*1e3:7.2f}ms ({t_rcsv/t_rpq:4.1f}x)")
    print(f"  write: csv {t_wcsv*1e3:7.2f}ms | parquet {t_wpq*1e3:7.2f}ms ({t_wcsv/t_wpq:4.1f}x)")
    print(f"  size : csv {sz_csv/1e6:6.3f}MB | parquet {sz_pq/1e6:6.3f}MB ({sz_csv/sz_pq:4.1f}x smaller)")
    shutil.rmtree(tmp, ignore_errors=True)


def _cpu_work(x: int) -> float:
    s = 0.0
    for i in range(200_000):
        s += math.sin(x + i) * math.cos(i)
    return s


def bench_parallel() -> None:
    print("\n=== 2. parallel_map speedup (CPU-bound) ===")
    from fbpipe.utils.parallel import parallel_map, resolve_n_jobs
    items = list(range(max(4, (os.cpu_count() or 4))))
    t_serial, _ = _best(lambda: parallel_map(_cpu_work, items, enabled=False), n=2)
    t_par, _ = _best(lambda: parallel_map(_cpu_work, items, enabled=True, n_jobs=0), n=2)
    print(f"  n_items={len(items)} workers={resolve_n_jobs(0)}")
    print(f"  serial {t_serial*1e3:7.1f}ms | parallel {t_par*1e3:7.1f}ms ({t_serial/t_par:4.1f}x)")


def bench_stage_chain(sample: Path, flies: int) -> None:
    print(f"\n=== 3. distance_stats -> distance_normalize, {flies} flies, serial vs parallel ===")
    if not sample.is_file():
        print(f"  (skipped: sample not found at {sample})")
        return
    from fbpipe.config import ForceSettings, ParallelSettings, Settings
    from fbpipe.steps import distance_normalize, distance_stats
    base_df = pd.read_csv(sample)

    def build(root: Path) -> None:
        for k in range(flies):
            trial = root / f"fly{k:03d}" / "trial1"
            trial.mkdir(parents=True)
            base_df.to_csv(trial / f"prefix_fly1_distances.csv", index=False)

    def run(parallel: bool) -> float:
        root = Path(tempfile.mkdtemp())
        build(root)
        cfg = Settings(
            model_path="", main_directories=[str(root)],
            force=ForceSettings(pipeline=False),
            parallel=ParallelSettings(enabled=parallel, n_jobs=0),
        )
        t = time.perf_counter()
        distance_stats.main(cfg)
        distance_normalize.main(cfg)
        dt = time.perf_counter() - t
        shutil.rmtree(root, ignore_errors=True)
        return dt

    t_serial = run(False)
    t_par = run(True)
    print(f"  serial {t_serial*1e3:7.1f}ms | parallel {t_par*1e3:7.1f}ms ({t_serial/max(t_par,1e-9):4.1f}x)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--flies", type=int, default=24)
    args = ap.parse_args()
    print(f"cpu_count={os.cpu_count()}  sample={args.sample}")
    bench_io(args.sample)
    bench_parallel()
    bench_stage_chain(args.sample, args.flies)


if __name__ == "__main__":
    main()
