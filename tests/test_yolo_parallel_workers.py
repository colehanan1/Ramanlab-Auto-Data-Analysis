"""yolo.num_workers: run_workflows fans the YOLO step across N parallel workers.

Config knob parsing + the _run_pipeline fan-out (each worker gets a disjoint 1/N
video slice via NUM_WORKERS / WORKER_INDEX, the same mechanism scripts/pipeline/
parallel_yolo.py uses). num_workers>1 is only ever passed for the yolo step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbpipe.config import Settings, load_settings  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402


# --- config parsing --------------------------------------------------------

def test_default_yolo_num_workers_is_one():
    assert Settings(model_path="m.pt", main_directories="/tmp").yolo_num_workers == 1


def test_yolo_num_workers_from_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("YOLO_NUM_WORKERS", raising=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("yolo:\n  num_workers: 3\n")
    assert load_settings(str(cfg)).yolo_num_workers == 3


def test_yolo_num_workers_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("yolo:\n  num_workers: 3\n")
    monkeypatch.setenv("YOLO_NUM_WORKERS", "5")
    assert load_settings(str(cfg)).yolo_num_workers == 5


# --- _run_pipeline fan-out -------------------------------------------------

class _FakeProc:
    def __init__(self, rc: int = 0):
        self._rc = rc

    def wait(self) -> int:
        return self._rc


def test_run_pipeline_single_worker_uses_subprocess_run(tmp_path, monkeypatch):
    calls = {"run": 0, "popen": 0}

    def _run(*a, **k):
        calls["run"] += 1

    def _popen(*a, **k):
        calls["popen"] += 1
        return _FakeProc(0)

    monkeypatch.setattr(rw.subprocess, "run", _run)
    monkeypatch.setattr(rw.subprocess, "Popen", _popen)
    rw._run_pipeline(tmp_path / "c.yaml", main_directory="/data/x", steps=("yolo",), num_workers=1)
    assert calls == {"run": 1, "popen": 0}


def test_run_pipeline_parallel_launches_n_disjoint_workers(tmp_path, monkeypatch):
    launched_envs = []

    def _popen(cmd, env=None, **k):
        launched_envs.append(env)
        return _FakeProc(0)

    monkeypatch.setattr(rw.subprocess, "Popen", _popen)
    monkeypatch.setattr(
        rw.subprocess, "run", lambda *a, **k: pytest.fail("run() must not be used in parallel mode")
    )
    rw._run_pipeline(tmp_path / "c.yaml", main_directory="/data/x", steps=("yolo",), num_workers=3)

    assert len(launched_envs) == 3
    assert all(e["NUM_WORKERS"] == "3" for e in launched_envs)
    assert sorted(e["WORKER_INDEX"] for e in launched_envs) == ["0", "1", "2"]
    # MAIN_DIRECTORY still propagated to every worker.
    assert all(e["MAIN_DIRECTORY"] == "/data/x" for e in launched_envs)


def test_run_pipeline_parallel_raises_when_a_worker_fails(tmp_path, monkeypatch):
    rcs = iter([0, 1, 0])
    monkeypatch.setattr(rw.subprocess, "Popen", lambda *a, **k: _FakeProc(next(rcs)))
    with pytest.raises(rw.subprocess.CalledProcessError):
        rw._run_pipeline(tmp_path / "c.yaml", main_directory="/data/x", steps=("yolo",), num_workers=3)
