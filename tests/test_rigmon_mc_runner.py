import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fbpipe.utils.mount_qc import Detection, analyze
from rigmon.config import HubConfig, MountCheckConfig, RigConfig
from rigmon.mountcheck.detect import WorkerError
from rigmon.mountcheck.frames import decode
from rigmon.mountcheck.runner import MountCheckRunner
from rigmon.mountcheck.snapshot import render
from rigmon.store import Store

GOLDEN = Path(__file__).parent / "golden"
GJPG = (GOLDEN / "mount_qc_pi2_20261014.jpg").read_bytes()
GDETS = json.loads((GOLDEN / "mount_qc_pi2_20261014.json").read_text())
PI2 = RigConfig("Flybehavior2", "Pi 2", "100.93.23.70")
SETTINGS = SimpleNamespace(model_path="/models/best.engine", rig_gate_overrides=(),
                           proboscis_filter=SimpleNamespace(max_eye_prob_distance_px=280.0, up_divisor=4.0))


class FakeNotifier:
    def __init__(self):
        self.calls = []

    def mount_check(self, label, state, messages):
        self.calls.append((label, state, messages))


def fake_worker(model, frames_dir, out_json, *, timeout, conf=0.25):
    names = sorted(p.name for p in Path(frames_dir).glob("*.jpg"))
    return {"width": 1080, "height": 1080, "frames": [{"name": n, "dets": GDETS["detections"]} for n in names]}


def make(tmp_path, sampler, worker=fake_worker):
    cfg = HubConfig(rigs=(PI2,), staging_root=tmp_path / "s", data_root=tmp_path / "Data",
                    db_path=tmp_path / "db.sqlite", qc_root=tmp_path / "qc")
    store, notifier = Store(cfg.db_path), FakeNotifier()
    runner = MountCheckRunner(cfg, store, notifier, SETTINGS, sampler=sampler, worker=worker,
                              spawn=lambda f: f())
    return runner, store, notifier, cfg


def test_render_marks_flagged_items_in_red():
    img = decode(GJPG)
    dets = [Detection.from_dict(d) for d in GDETS["detections"]]
    rep = analyze([dets], 1080, 1080, gate_px=280.0)
    out = render(img, dets, rep)
    assert out.shape == img.shape and not np.array_equal(out, img)
    red = (out[:, :, 2] > 200) & (out[:, :, 1] < 60) & (out[:, :, 0] < 60)
    assert red.sum() > 50


def test_golden_run_reports_issues_and_writes_files(tmp_path):
    runner, store, notifier, cfg = make(tmp_path, lambda *a, **k: ([GJPG] * 3, True))
    cid = runner.start(PI2, "october_14_batch_2", "auto")
    row = store.latest_mount_check("Flybehavior2")
    assert row["id"] == cid and row["state"] == "issues" and row["frames"] == 3
    assert [f["code"] for f in row["flags"]] == ["edge_margin", "bottom_room"]
    assert Path(row["json_path"]).parent == cfg.qc_root / "Flybehavior2" / "october_14_batch_2"
    assert Path(row["jpg_path"]).is_file() and json.loads(Path(row["json_path"]).read_text())["gate_px"] == 280.0
    assert notifier.calls[0][1] == "issues" and "fly 4: proboscis" in notifier.calls[0][2][1]


def test_camera_unreachable_is_flagged_without_running_the_worker(tmp_path):
    def worker(*a, **k):
        raise AssertionError("worker must not run")
    runner, store, notifier, _ = make(tmp_path, lambda *a, **k: ([], False), worker)
    runner.start(PI2, "b", "manual")
    row = store.latest_mount_check("Flybehavior2")
    assert row["state"] == "issues" and row["flags"][0]["code"] == "camera"
    assert notifier.calls


def test_worker_failure_is_an_error_row_and_notifies(tmp_path):
    def worker(*a, **k):
        raise WorkerError("detector timed out after 600 s")
    runner, store, notifier, _ = make(tmp_path, lambda *a, **k: ([GJPG], True), worker)
    runner.start(PI2, "b", "manual")
    row = store.latest_mount_check("Flybehavior2")
    assert row["state"] == "error" and "timed out" in row["error"]
    assert notifier.calls[0][1] == "error"


def test_clean_check_is_ok_and_silent(tmp_path):
    clean = [d for d in GDETS["detections"] if 300 < d["y"] < 800]   # flies 2 and 3: well placed

    def worker(model, frames_dir, out_json, *, timeout, conf=0.25):
        return {"width": 1080, "height": 1080, "frames": [{"name": "0.jpg", "dets": clean}]}
    runner, store, notifier, _ = make(tmp_path, lambda *a, **k: ([GJPG], True), worker)
    runner.start(PI2, "b", "manual")
    row = store.latest_mount_check("Flybehavior2")
    # only 2 flies in this synthetic frame → flies_visible; prove the "ok" path with expected_flies=2
    assert row["state"] == "issues"
    runner.cfg = replace(runner.cfg, mount_check=MountCheckConfig(thresholds={"expected_flies": 2}))
    runner.start(PI2, "b", "manual")
    row = store.latest_mount_check("Flybehavior2")
    assert row["state"] == "ok" and row["sharpness"] is not None
    assert len(notifier.calls) == 1


def test_short_sample_is_incomplete_when_clean(tmp_path):
    clean = [d for d in GDETS["detections"] if 300 < d["y"] < 800]   # flies 2 and 3: well placed

    def worker(model, frames_dir, out_json, *, timeout, conf=0.25):
        return {"width": 1080, "height": 1080, "frames": [{"name": "0.jpg", "dets": clean}]}
    runner, store, _, cfg = make(tmp_path, lambda *a, **k: ([GJPG], False), worker)
    runner.cfg = replace(cfg, mount_check=MountCheckConfig(thresholds={"expected_flies": 2}))
    runner.start(PI2, "b", "manual")
    assert store.latest_mount_check("Flybehavior2")["state"] == "incomplete"


def test_second_start_while_running_is_refused(tmp_path):
    runner, store, _, _ = make(tmp_path, lambda *a, **k: ([GJPG], True))
    runner.spawn = lambda f: None          # leave the first one "running"
    assert runner.start(PI2, "b", "manual") is not None
    assert runner.start(PI2, "b", "manual") is None
