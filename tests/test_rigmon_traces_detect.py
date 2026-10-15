import sys
from pathlib import Path

import cv2
import pytest

from rigmon.mountcheck.detect import WorkerError
from rigmon.traces import detect_video
from rigmon.traces.detect_video import run_video_worker

GOLDEN = Path(__file__).parent / "golden" / "mount_qc_pi2_20261014.jpg"
ENGINE = Path("/home/ramanlab/Documents/cole/model/YOLOProjectProboscisLegs/runs/obb/train12/weights/best.engine")


def make_mp4(path, n=10, fps=40.0):
    img = cv2.imread(str(GOLDEN))[:, :1080]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1080, 1080))
    for _ in range(n):
        vw.write(img)
    vw.release()
    return path


def test_missing_model_is_a_worker_error(tmp_path):
    with pytest.raises(WorkerError):
        run_video_worker(tmp_path / "nope.engine", make_mp4(tmp_path / "v.mp4"), tmp_path / "o.json", timeout=60)


def test_timeout_is_a_worker_error(tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(30)")
    with pytest.raises(WorkerError, match="timed out"):
        run_video_worker(tmp_path / "m", tmp_path / "v.mp4", tmp_path / "o.json", timeout=1,
                         python_argv=[sys.executable, str(slow)])


@pytest.mark.skipif(not ENGINE.exists(), reason="engine not on this machine")
def test_real_engine_on_a_short_clip(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    res = run_video_worker(ENGINE, make_mp4(tmp_path / "v.mp4", n=6), tmp_path / "o.json", timeout=300, stride=2)
    assert [f["i"] for f in res["frames"]] == [0, 2, 4]
    assert res["fps"] == pytest.approx(40.0, rel=0.05) and (res["width"], res["height"]) == (1080, 1080)
    assert all(sum(d["cls"] == "eye" for d in f["dets"]) == 4 for f in res["frames"])


def _garbage(tmp_path):
    g = tmp_path / "x.mp4"
    g.write_bytes(np_bytes := bytes(range(256)) * 64)
    return g


def test_garbage_video_is_a_worker_error(tmp_path):
    model = tmp_path / "m.engine"
    model.write_bytes(b"")
    with pytest.raises(WorkerError, match="video|x.mp4"):
        run_video_worker(model, _garbage(tmp_path), tmp_path / "o.json", timeout=60)


def test_main_rejects_undecodable_video_before_loading_model(tmp_path):
    model = tmp_path / "m.engine"
    model.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        detect_video.main(["--model", str(model), "--video", str(_garbage(tmp_path)),
                           "--out", str(tmp_path / "o.json")])
    assert "video" in str(e.value)


def test_stale_output_is_never_returned(tmp_path):
    out = tmp_path / "o.json"
    out.write_text('{"frames": []}')
    noop = tmp_path / "noop.py"
    noop.write_text("pass")
    with pytest.raises(WorkerError):
        run_video_worker(tmp_path / "m", tmp_path / "v.mp4", out, timeout=60,
                         python_argv=[sys.executable, str(noop)])
