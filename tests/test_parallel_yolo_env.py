"""scripts/pipeline/parallel_yolo.py: per-worker CPU thread-pool environment.

numpy's bundled OpenBLAS keeps its pool spinning at 100% CPU after the
tracker's tiny per-frame ``np.linalg.inv`` calls. Measured 2026-10-05: an
OPENBLAS_NUM_THREADS cap of 16, 5 or 4 still left 3 spinning threads per
worker, 2 left 1, and only 1 stopped it (2-video run: 205.6 -> 61.1 CPU-s,
outputs md5-identical). So the BLAS pools get exactly one thread, while the
OpenMP pool (torch/OpenCV) keeps its cores/N share.

MKL_NUM_THREADS is the exception for a lone worker: torch sizes its default
intra-op pool from it (it wins over OMP_NUM_THREADS; measured with torch 2.4:
8 threads under MKL=16, 1 under MKL=1) and yolo_infer re-pins torch to cores/N
only when NUM_WORKERS > 1, so ``--workers 1`` keeps the old cores//1 value.
"""
from __future__ import annotations

import sys

from scripts.pipeline import parallel_yolo

BLAS_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class _FakeProc:
    pid = 0

    def wait(self) -> int:
        return 0


def _launch(tmp_path, monkeypatch, workers: int, cores: int):
    launched = []

    def _popen(cmd, cwd=None, env=None, stdout=None, stderr=None):
        launched.append((cmd, env))
        return _FakeProc()

    monkeypatch.setattr(parallel_yolo.subprocess, "Popen", _popen)
    monkeypatch.setattr(parallel_yolo.os, "cpu_count", lambda: cores)
    monkeypatch.setenv("MAIN_DIRECTORY", "/data/must-not-leak")
    # A parent shell's own caps must not leak into the workers.
    for var in BLAS_VARS + ("OMP_NUM_THREADS",):
        monkeypatch.setenv(var, "16")
    monkeypatch.setattr(
        sys, "argv",
        ["parallel_yolo.py", "--config", "c.yaml", "--workers", str(workers), "--logdir", str(tmp_path)],
    )
    assert parallel_yolo.main() == 0
    return launched


def test_workers_get_one_blas_thread_and_a_share_of_openmp(tmp_path, monkeypatch):
    launched = _launch(tmp_path, monkeypatch, workers=3, cores=16)

    assert len(launched) == 3
    for k, (cmd, env) in enumerate(launched):
        assert cmd[-1] == "yolo"
        for var in BLAS_VARS:
            assert env[var] == "1", (var, env[var])
        assert env["OMP_NUM_THREADS"] == "5"  # 16 cores // 3 workers
        assert env["NUM_WORKERS"] == "3"
        assert env["WORKER_INDEX"] == str(k)
        assert "MAIN_DIRECTORY" not in env  # every worker must see all dirs


def test_single_worker_caps_openblas_but_keeps_torchs_mkl_share(tmp_path, monkeypatch):
    (cmd, env), = _launch(tmp_path, monkeypatch, workers=1, cores=16)
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "16"
    # NUM_WORKERS=1: yolo_infer leaves torch's pool alone, so MKL keeps cores//1.
    assert env["MKL_NUM_THREADS"] == "16"
