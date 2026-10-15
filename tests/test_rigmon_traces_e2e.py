"""Copy (local rig, real rsync, recording gate) → staged mp4 → trace job (fake worker) → PNG + row."""

from pathlib import Path
from types import SimpleNamespace

from rigmon.config import HubConfig, RigConfig
from rigmon.store import Store
from rigmon.traces.runner import TraceRunner
from rigmon.transfer import TrialTransfer

LOCAL = RigConfig("fakepi", "Fake", "local", 1)
SETTINGS = SimpleNamespace(model_path="/m.engine", rig_gate_overrides=(),
                           proboscis_filter=SimpleNamespace(max_eye_prob_distance_px=160.0, up_divisor=4.0))


def det(cls, x, y):
    return {"cls": cls, "x": x, "y": y, "w": 40, "h": 40, "r": 0.0, "conf": 0.9}


def worker(model, video, out_json, *, timeout, stride=2, conf=0.25):
    assert Path(video).is_file()
    return {"fps": 40.0, "frame_count": 40, "width": 1080, "height": 1080,
            "frames": [{"i": i, "dets": [d for y in (150, 400, 650, 850)
                                         for d in (det("eye", 700, y), det("proboscis", 700, y + 80 + (60 if 10 <= i < 20 else 0)))]}
                       for i in range(0, 40, stride)]}


def test_trial_copy_then_trace(tmp_path):
    pi = tmp_path / "pi" / "b1"
    pi.mkdir(parents=True)
    (pi / "t1.h264").write_bytes(b"h" * 64)
    (pi / "t1.csv").write_text("FrameNumber,SensorNs,UTC_ISO,ActiveOFM\n"
                               + "".join(f"{k},{k * 25_000_000},x,{'OFM_I' if 10 <= k < 20 else 'off'}\n" for k in range(40)))
    cfg = HubConfig(rigs=(LOCAL,), staging_root=tmp_path / "staging", data_root=tmp_path / "Data",
                    db_path=tmp_path / "db.sqlite", qc_root=tmp_path / "qc")
    store = Store(cfg.db_path)
    recording = iter([True, False] + [False] * 50)
    sleeps = []

    def remux_runner(argv, timeout, env=None, abort=None):
        if argv[0] == "rsync":
            from rigmon.transfer import subprocess_runner
            return subprocess_runner(argv, timeout, env)
        h264 = Path(argv[argv.index("--input-dir") + 1]) / argv[argv.index("--only") + 1]
        h264.with_suffix(".mp4").write_bytes(b"mp4" * 100)
        h264.unlink()
        return 0, "ok"
    transfer = TrialTransfer(cfg, store, runner=remux_runner, sleep=sleeps.append,
                             is_recording=lambda n: next(recording))
    trial = {"index": 1, "label": "Testing 1/7", "video": "t1.h264", "frames_csv": "t1.csv",
             "video_bytes": 64, "csv_bytes": (pi / "t1.csv").stat().st_size, "ended_at": 1.0, "closed": True}
    assert transfer.process(LOCAL, "b1", str(pi), trial) == "done"
    stem = TraceRunner(cfg, store, SETTINGS, transfer, worker=worker).run(LOCAL, "b1", trial)
    assert sleeps == [2.0], "recording gate must wait exactly once (poll_s) before the first pull"
    assert next(recording) is False
    assert stem == "t1"
    out = cfg.qc_root / "fakepi" / "b1" / "traces"
    assert (out / "t1.png").is_file() and (out / "t1.csv").read_text().count("\n") == 21
    assert store.get_trial("fakepi", "b1", 1)["trace"] == "t1"
    assert store.get_trial("fakepi", "b1", 1)["trace_error"] == ""
