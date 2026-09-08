"""Production mode of the geometry pipeline: TensorRT engine preference, engine batch
clamp, and the N-worker fan-out of the track step."""
import os
import sys
from pathlib import Path

import pytest

from fbpipe import geometry_pipeline as GP
from fbpipe.config import Settings
from fbpipe.geometry import detect as D
from fbpipe.geometry import steps as S
from fbpipe.geometry.config import GeometrySettings, parse_geometry_block


def test_resolve_model_path_prefers_the_sibling_engine(tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"x")
    assert D.resolve_model_path(pt, prefer_engine=True) == pt
    eng = tmp_path / "best.engine"
    eng.write_bytes(b"x")
    assert D.resolve_model_path(pt, prefer_engine=True) == eng
    assert D.resolve_model_path(pt, prefer_engine=False) == pt
    assert D.resolve_model_path(eng, prefer_engine=False) == eng
    assert D.is_engine_path(eng) and not D.is_engine_path(pt)


def test_engine_models_are_never_moved_with_to():
    pytest.importorskip("ultralytics")
    calls = []

    class FakeEngine:
        def to(self, device):
            calls.append(("to", device))

        def predict(self, images, **kw):
            calls.append(("predict", kw.get("device")))
            return ["r"] * len(images)

    fn = D.make_predict_fn(FakeEngine(), imgsz=1088, iou=0.7, allow_cpu=True, is_engine=True)
    assert fn([1], 0.25) == ["r"]
    assert not any(c[0] == "to" for c in calls), "TensorRT engines reject .to(); device goes per call"


def test_effective_batch_size_clamps_static_engines():
    s = Settings(model_path="x", main_directories=[], inference_batch_size=16, engine_supports_batch=False)
    assert S.effective_batch_size(s, is_engine=False) == 16
    assert S.effective_batch_size(s, is_engine=True) == 1
    s2 = Settings(model_path="x", main_directories=[], inference_batch_size=16, engine_supports_batch=True)
    assert S.effective_batch_size(s2, is_engine=True) == 16


def test_prefer_engine_is_a_geometry_setting():
    assert parse_geometry_block(None).prefer_engine is True
    assert parse_geometry_block({"prefer_engine": False}).prefer_engine is False


def test_worker_commands_shard_and_cap_threads(tmp_path):
    cmds = GP.worker_commands(Path("config/config_abdomen.yaml"), 3, cores=16,
                              options={"force": True, "only_trial": "pretest_1", "max_frames": 200})
    assert len(cmds) == 3
    for k, (cmd, env) in enumerate(cmds):
        assert cmd[:2] == [sys.executable, "-m"] and "fbpipe.geometry_pipeline" in cmd
        assert "track" in cmd and "--no-fan-out" in cmd and "--force" in cmd
        assert cmd[cmd.index("--only-trial") + 1] == "pretest_1" and cmd[cmd.index("--max-frames") + 1] == "200"
        assert env["NUM_WORKERS"] == "3" and env["WORKER_INDEX"] == str(k)
        assert env["OMP_NUM_THREADS"] == "5" and env["MKL_NUM_THREADS"] == "5"
        assert env["PYTHONUNBUFFERED"] == "1" and "src" in env["PYTHONPATH"]
        assert "MAIN_DIRECTORY" not in env


def test_main_fans_out_track_when_workers_configured(monkeypatch, tmp_path):
    settings = Settings(model_path="x", main_directories=[], yolo_num_workers=3, cache_dir=str(tmp_path))
    gcfg = GeometrySettings()
    monkeypatch.setattr("fbpipe.geometry.config.load_geometry_config", lambda p: (settings, gcfg))
    seen = {}
    monkeypatch.setattr(GP, "fan_out_track", lambda cfg, n, *, settings, options: seen.update(n=n, opts=dict(options)) or 0)
    ran = []
    monkeypatch.setattr(S, "run_track", lambda *a, **k: ran.append("track"))
    monkeypatch.setattr(S, "run_normalize", lambda *a, **k: ran.append("normalize"))
    assert GP.main(["--config", "c.yaml", "--only-trial", "x", "track", "normalize"]) == 0
    assert seen["n"] == 3 and seen["opts"]["only_trial"] == "x"
    assert ran == ["normalize"], "track ran in the workers, the rest in-process"
    # --workers overrides the config; 1 worker or --no-fan-out runs in-process
    seen.clear(); ran.clear()
    assert GP.main(["--config", "c.yaml", "--workers", "1", "track"]) == 0
    assert not seen and ran == ["track"]
    ran.clear()
    assert GP.main(["--config", "c.yaml", "--no-fan-out", "track"]) == 0
    assert ran == ["track"]
