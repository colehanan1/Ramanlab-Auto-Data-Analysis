"""The YOLO fan-out caps each worker's BLAS pool at exactly one thread.

numpy's bundled OpenBLAS sizes its pool from OPENBLAS_NUM_THREADS, falling back
to OMP_NUM_THREADS; at its default 16 threads the pool spins at 100% CPU
between the tiny per-frame ``np.linalg.inv`` calls in the Kalman tracker. The
nightly workers escaped that only because importing ultralytics in the driver
sets OMP_NUM_THREADS=1 when it is unset -- a side effect that a preset
OMP_NUM_THREADS or another ultralytics release removes. The cap is now
explicit. Only 1 stops the spin (2 leaves a spinner), and outputs are
identical either way. It is scoped to the YOLO worker env: the per-trial steps
run through the single-process path and keep their threads, and OMP/torch
threads are left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline import run_workflows as rw  # noqa: E402

BLAS_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class _FakeProc:
    def wait(self) -> int:
        return 0


def _launch_workers(tmp_path, monkeypatch, n=3):
    envs = []

    def _popen(cmd, env=None, **k):
        envs.append(env)
        return _FakeProc()

    monkeypatch.setattr(rw.subprocess, "Popen", _popen)
    monkeypatch.setattr(
        rw.subprocess, "run", lambda *a, **k: pytest.fail("run() is the n<=1 path")
    )
    rw._run_pipeline(
        tmp_path / "c.yaml", main_directory="/data/x", steps=("yolo",), num_workers=n
    )
    return envs


def test_every_yolo_worker_gets_blas_pools_capped_at_one(tmp_path, monkeypatch):
    for var in BLAS_VARS:
        monkeypatch.delenv(var, raising=False)
    envs = _launch_workers(tmp_path, monkeypatch)
    assert len(envs) == 3
    for env in envs:
        for var in BLAS_VARS:
            assert env[var] == "1", var


def test_cap_is_exactly_one_even_when_the_parent_env_sets_more(tmp_path, monkeypatch):
    # 16 / 5 / 4 / 2 all left spinners in the measurement; only 1 stops them.
    for var in BLAS_VARS:
        monkeypatch.setenv(var, "16")
    for env in _launch_workers(tmp_path, monkeypatch):
        for var in BLAS_VARS:
            assert env[var] == "1", var


def test_omp_threads_are_left_as_the_parent_has_them(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    for env in _launch_workers(tmp_path, monkeypatch):
        assert env["OMP_NUM_THREADS"] == "7"

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    for env in _launch_workers(tmp_path, monkeypatch):
        assert "OMP_NUM_THREADS" not in env


def test_worker_slicing_env_is_unchanged(tmp_path, monkeypatch):
    envs = _launch_workers(tmp_path, monkeypatch)
    assert all(e["NUM_WORKERS"] == "3" for e in envs)
    assert sorted(e["WORKER_INDEX"] for e in envs) == ["0", "1", "2"]
    assert all(e["MAIN_DIRECTORY"] == "/data/x" for e in envs)


def test_single_process_path_keeps_the_parent_blas_settings(tmp_path, monkeypatch):
    """The per-trial steps (and a 1-worker YOLO run) are not capped."""
    for var in BLAS_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    seen = []

    def _run(cmd, check=False, env=None, **k):
        seen.append(env)

    monkeypatch.setattr(rw.subprocess, "run", _run)
    monkeypatch.setattr(
        rw.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen() is the fan-out path")
    )
    rw._run_pipeline(tmp_path / "c.yaml", main_directory="/data/x", steps=["distance_stats"])
    rw._run_pipeline(
        tmp_path / "c.yaml", main_directory="/data/x", steps=("yolo",), num_workers=1
    )
    assert len(seen) == 2
    for env in seen:
        assert "OPENBLAS_NUM_THREADS" not in env
        assert "NUMEXPR_NUM_THREADS" not in env
        assert env["MKL_NUM_THREADS"] == "4"
