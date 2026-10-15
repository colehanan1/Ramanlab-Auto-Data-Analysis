import csv
import json
from pathlib import Path
from types import SimpleNamespace

from rigmon.config import HubConfig, RigConfig
from rigmon.mountcheck.detect import WorkerError
from rigmon.store import Store
from rigmon.traces.runner import TraceRunner
from rigmon.transfer import TrialTransfer

PI2 = RigConfig("Flybehavior2", "Pi 2", "100.93.23.70")
SETTINGS = SimpleNamespace(model_path="/m.engine", rig_gate_overrides=(),
                           proboscis_filter=SimpleNamespace(max_eye_prob_distance_px=280.0, up_divisor=4.0))
TRIAL = {"index": 3, "label": "Testing 3/7", "phase_tag": "testing_3", "video": "t3.h264",
         "frames_csv": "t3.csv", "ended_at": 5.0}


def det(cls, x, y):
    return {"cls": cls, "x": x, "y": y, "w": 40, "h": 40, "r": 0.0, "conf": 0.9}


def fake_worker(model, video, out_json, *, timeout, stride=2, conf=0.25):
    frames = [{"i": i, "dets": [d for y in (150, 400, 650, 850)
                                 for d in (det("eye", 700, y), det("proboscis", 700, y + 80))]}
              for i in range(0, 20, stride)]
    return {"fps": 40.0, "frame_count": 20, "width": 1080, "height": 1080, "frames": frames}


def setup(tmp_path, worker=fake_worker):
    cfg = HubConfig(rigs=(PI2,), staging_root=tmp_path / "s", data_root=tmp_path / "Data",
                    db_path=tmp_path / "db.sqlite", qc_root=tmp_path / "qc")
    store = Store(cfg.db_path)
    transfer = TrialTransfer(cfg, store)
    st = transfer.staging_dir("Flybehavior2", "b2")
    st.mkdir(parents=True)
    (st / "t3.mp4").write_bytes(b"mp4")
    (st / "t3.csv").write_text("FrameNumber,SensorNs,UTC_ISO,ActiveOFM\n"
                               + "".join(f"{k},{k * 25_000_000},x,{'OFM_I' if 5 <= k < 12 else 'off'}\n"
                                         for k in range(20)))
    store.upsert_trial("Flybehavior2", "b2", TRIAL)
    return TraceRunner(cfg, store, SETTINGS, transfer, worker=worker), store, cfg


def test_trace_job_writes_csv_png_and_marks_row(tmp_path):
    runner, store, cfg = setup(tmp_path)
    assert runner.run(PI2, "b2", TRIAL) == "t3"
    out = cfg.qc_root / "Flybehavior2" / "b2" / "traces"
    rows = list(csv.DictReader((out / "t3.csv").open()))
    assert list(rows[0].keys()) == ["t_s", "odor", "fly1_px", "fly2_px", "fly3_px", "fly4_px"]
    assert float(rows[0]["fly1_px"]) == 80.0 and len(rows) == 10
    assert any(r["odor"] == "OFM_I" for r in rows) and rows[0]["odor"] == ""
    assert (out / "t3.png").is_file()
    assert store.get_trial("Flybehavior2", "b2", 3)["trace"] == "t3"


def test_worker_failure_sets_trace_error_and_never_raises(tmp_path):
    def worker(*a, **k):
        raise WorkerError("trace detector timed out after 600 s")
    runner, store, _ = setup(tmp_path, worker)
    assert runner.run(PI2, "b2", TRIAL) is None
    row = store.get_trial("Flybehavior2", "b2", 3)
    assert row["trace"] == "" and "timed out" in row["trace_error"]


def test_missing_mp4_is_a_trace_error(tmp_path):
    runner, store, cfg = setup(tmp_path)
    (cfg.staging_root / "Flybehavior2" / "b2" / "t3.mp4").unlink()
    assert runner.run(PI2, "b2", TRIAL) is None
    assert "mp4" in store.get_trial("Flybehavior2", "b2", 3)["trace_error"]


def test_trace_job_holds_the_gpu_semaphore(tmp_path):
    import threading
    sem = threading.Semaphore(1)
    seen = {}

    def worker(*a, **k):
        seen["locked"] = not sem.acquire(blocking=False)
        return fake_worker(*a, **k)
    runner, _, _ = setup(tmp_path, worker)
    runner.gpu = sem
    runner.run(PI2, "b2", TRIAL)
    assert seen["locked"] is True


def test_mountcheck_runner_uses_the_shared_gpu_lock():
    import rigmon.gpu
    import rigmon.mountcheck.runner as mcr
    assert mcr._GPU is rigmon.gpu.GPU


def test_series_uses_mount_check_thresholds(tmp_path):
    from dataclasses import replace

    from rigmon.config import MountCheckConfig
    runner, store, cfg = setup(tmp_path)
    runner.cfg = replace(cfg, mount_check=MountCheckConfig(thresholds={"expected_flies": 2}))
    assert runner.run(PI2, "b2", TRIAL) == "t3"
    out = cfg.qc_root / "Flybehavior2" / "b2" / "traces"
    header = next(csv.reader((out / "t3.csv").open()))
    assert header == ["t_s", "odor", "fly1_px", "fly2_px"]


def test_no_flies_detected_is_a_trace_error(tmp_path):
    def worker(*a, **k):
        return {"fps": 40.0, "frame_count": 20, "frames": [{"i": i, "dets": []} for i in range(0, 20, 2)]}
    runner, store, cfg = setup(tmp_path, worker)
    assert runner.run(PI2, "b2", TRIAL) is None
    row = store.get_trial("Flybehavior2", "b2", 3)
    assert row["trace"] == "" and "no flies detected" in row["trace_error"]
    out = cfg.qc_root / "Flybehavior2" / "b2" / "traces"
    assert not (out / "t3.png").exists() and not (out / "t3.csv").exists()
