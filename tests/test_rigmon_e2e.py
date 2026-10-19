"""End to end: a fake Pi on localhost → hub polls → trials pulled + remuxed
(stub ffmpeg) → Pi hands off → session lands in a temp flys_New."""

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from rigmon.app import create_app
from rigmon.config import HubConfig, RigConfig
from rigmon.finalize import Finalizer
from rigmon.hub import Hub
from rigmon.notify import Notifier
from rigmon.store import Store
from rigmon.transfer import TransferWorkers, TrialTransfer
from test_remux_from_csv import STUB_SOURCE, write_frame_csv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PiCode"))
import rig_session  # noqa: E402

_spec = importlib.util.spec_from_file_location("fake_rig", REPO / "scripts" / "rigmon" / "fake_rig.py")
fake_rig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake_rig)


def test_session_lands_end_to_end(tmp_path, monkeypatch):
    stub = tmp_path / "stubbin"
    stub.mkdir()
    (stub / "ffmpeg").write_text(STUB_SOURCE.replace("__PYTHON__", sys.executable))
    (stub / "ffmpeg").chmod(0o755)
    monkeypatch.setenv("STUB_FFMPEG_LOG", str(tmp_path / "ffmpeg.log"))
    monkeypatch.setenv("STUB_FFMPEG_SLEEP", "0")
    monkeypatch.setenv("REMUX_LOCK_DIR", str(tmp_path / "locks"))

    batch = "october_14_batch_3"
    pi = tmp_path / "pi" / batch
    pi.mkdir(parents=True)
    trials = []
    for i in (1, 2):
        (pi / f"t{i}.h264").write_bytes(b"\x00\x00\x00\x01 fake")
        write_frame_csv(pi / f"t{i}.csv")
        trials.append({"index": i, "label": f"Training {i}/2", "odor": "ACV",
                       "video": f"t{i}.h264", "frames_csv": f"t{i}.csv",
                       "video_bytes": (pi / f"t{i}.h264").stat().st_size,
                       "csv_bytes": (pi / f"t{i}.csv").stat().st_size,
                       "ended_at": float(i), "closed": True})
    (pi / "session_header.txt").write_text("fly 3")

    rig = fake_rig.FakeRig({"phase": "GAP", "server_time": 0.0, "run": {
        "state": "complete", "batch_dir": batch, "output_dir": str(pi),
        "push_mode": "hub", "trials": trials}})
    try:
        cfg = HubConfig(rigs=(RigConfig("fakepi", "Fake", "local", rig.port),),
                        staging_root=tmp_path / "staging", data_root=tmp_path / "Data",
                        db_path=tmp_path / "db.sqlite", remux_python=sys.executable,
                        remux_path_env=f"{stub}:/usr/bin:/bin")
        store = Store(cfg.db_path)
        notifier = Notifier(cfg, store, runner=lambda *a, **k: (0, ""), spawn=lambda f: f())
        transfer = TrialTransfer(cfg, store, sleep=lambda s: None)
        workers = TransferWorkers(threaded=False)
        hub = Hub(cfg, store, notifier, transfer, Finalizer(cfg, store, transfer), workers)

        async def poll():
            async with httpx.AsyncClient() as c:
                await hub.poll_once(c)
        asyncio.run(poll())
        workers.drain()
        assert [t["state"] for t in store.trials()] == ["done", "done"]

        client = TestClient(create_app(hub, start_polling=False), base_url="http://localhost",
                        client=("127.0.0.1", 50000))
        url = f"/api/sessions/fakepi/{batch}"
        r = client.post(f"{url}/complete", json={
            "output_dir": str(pi), "manifest": rig_session.build_manifest(pi),
            "push_to": f"ramanlab@100.94.160.60:{tmp_path}/Data/flys_New/Cohort/"})
        assert r.status_code == 202
        workers.drain()
        assert client.get(f"{url}/status").json()["state"] == "landed"
        dest = tmp_path / "Data" / "flys_New" / "Cohort" / batch
        assert sorted(p.name for p in dest.iterdir()) == \
            ["session_header.txt", "t1.csv", "t1.mp4", "t2.csv", "t2.mp4"]
        assert client.get(f"/media/fakepi/{batch}/t1.mp4").status_code == 200
    finally:
        rig.stop()
