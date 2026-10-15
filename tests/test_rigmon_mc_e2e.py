"""Fake rig serves the golden preview frame; the real runner (fake worker unless
CUDA + engine are available) produces the golden flags and files."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from rigmon.config import HubConfig, RigConfig
from rigmon.mountcheck.runner import MountCheckRunner
from rigmon.store import Store

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"
_spec = importlib.util.spec_from_file_location("fake_rig", REPO / "scripts" / "rigmon" / "fake_rig.py")
fake_rig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake_rig)


def test_mount_check_end_to_end(tmp_path):
    rig_srv = fake_rig.FakeRig({}, preview_jpegs=[(GOLDEN / "mount_qc_pi2_20261014.jpg").read_bytes()],
                               preview_interval=0.05)
    try:
        rig = RigConfig("Flybehavior2", "Pi 2", "local", rig_srv.port)
        from rigmon.config import MountCheckConfig
        cfg = HubConfig(rigs=(rig,), staging_root=tmp_path / "s", data_root=tmp_path / "Data",
                        db_path=tmp_path / "db.sqlite", qc_root=tmp_path / "qc",
                        mount_check=MountCheckConfig(sample_s=1.0, sample_fps=4.0))
        golden = json.loads((GOLDEN / "mount_qc_pi2_20261014.json").read_text())

        def worker(model, frames_dir, out_json, *, timeout, conf=0.25):
            names = sorted(p.name for p in Path(frames_dir).glob("*.jpg"))
            return {"width": 1080, "height": 1080, "frames": [{"name": n, "dets": golden["detections"]} for n in names]}
        settings = SimpleNamespace(model_path="m.engine", rig_gate_overrides=(),
                                   proboscis_filter=SimpleNamespace(max_eye_prob_distance_px=280.0, up_divisor=4.0))
        calls = []
        notifier = SimpleNamespace(mount_check=lambda *a: calls.append(a))
        store = Store(cfg.db_path)
        MountCheckRunner(cfg, store, notifier, settings, worker=worker, spawn=lambda f: f()).start(rig, "b2", "manual")
        row = store.latest_mount_check("Flybehavior2")
        assert row["state"] == "issues" and 2 <= row["frames"] <= 6
        assert [f["code"] for f in row["flags"]] == ["edge_margin", "bottom_room"]
        assert Path(row["jpg_path"]).is_file() and calls
    finally:
        rig_srv.stop()
