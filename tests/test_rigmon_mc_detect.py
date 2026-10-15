import json
import sys
from pathlib import Path

import pytest

from rigmon.mountcheck.detect import WorkerError, dets_from_arrays, run_worker

GOLDEN = Path(__file__).parent / "golden"
ENGINE = Path("/home/ramanlab/Documents/cole/model/YOLOProjectProboscisLegs/runs/obb/train12/weights/best.engine")


def test_dets_from_arrays_names_and_rounds():
    out = dets_from_arrays({0: "eye", 1: "proboscis"}, [1.0], [[10.04, 20.06, 5.0, 6.0, 0.12345]], [0.9876])
    assert out == [{"cls": "proboscis", "x": 10.0, "y": 20.1, "w": 5.0, "h": 6.0, "r": 0.1235, "conf": 0.988}]


def test_run_worker_reports_failure_with_stderr(tmp_path):
    (tmp_path / "f").mkdir()
    with pytest.raises(WorkerError) as e:
        run_worker(tmp_path / "missing.engine", tmp_path / "f", tmp_path / "o.json", timeout=60)
    assert "missing.engine" in str(e.value) or "No such file" in str(e.value) or "not found" in str(e.value).lower()


def test_run_worker_timeout_is_a_worker_error(tmp_path):
    (tmp_path / "f").mkdir()
    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(30)")
    with pytest.raises(WorkerError, match="timed out"):
        run_worker(tmp_path / "m", tmp_path / "f", tmp_path / "o.json", timeout=1,
                   python_argv=[sys.executable, str(slow)])


@pytest.mark.skipif(not ENGINE.exists(), reason="detector engine not on this machine")
def test_real_engine_finds_four_eyes_on_the_golden_frame(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    fdir = tmp_path / "frames"
    fdir.mkdir()
    (fdir / "000.jpg").write_bytes((GOLDEN / "mount_qc_pi2_20261014.jpg").read_bytes())
    res = run_worker(ENGINE, fdir, tmp_path / "out.json", timeout=300)
    dets = res["frames"][0]["dets"]
    assert (res["width"], res["height"]) == (1080, 1080)
    assert sum(d["cls"] == "eye" for d in dets) == 4
