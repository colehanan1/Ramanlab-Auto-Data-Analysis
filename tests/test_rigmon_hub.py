import asyncio

import httpx

from rigmon.config import HubConfig, RigConfig
from rigmon.hub import Hub, fetch_status
from rigmon.store import Store
from rigmon.transfer import TransferWorkers

PI2 = RigConfig("flybehavior2", "Pi 2", "100.93.23.70")


class Spy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a)) or "done"


def _hub(tmp_path, transfer_enabled=True):
    cfg = HubConfig(rigs=(PI2,), staging_root=tmp_path / "s", data_root=tmp_path / "d",
                    db_path=tmp_path / "db.sqlite", transfer_enabled=transfer_enabled)
    store, notifier, transfer, finalizer = Store(cfg.db_path), Spy(), Spy(), Spy()
    workers = TransferWorkers(threaded=False)
    hub = Hub(cfg, store, notifier, transfer, finalizer, workers, clock=lambda: 100.0)
    return hub, store, notifier, transfer, finalizer, workers


def snap(trials=(1,), state="running"):
    return {"phase": "GAP", "trial_label": "", "server_time": 99.0,
            "run": {"state": state, "batch_dir": "b", "output_dir": "/pi/b",
                    "trials": [{"index": i, "label": f"T{i}", "video": f"t{i}.h264",
                                "frames_csv": f"t{i}.csv", "ended_at": float(i), "closed": True}
                               for i in trials]}}


def test_ingest_records_notifies_and_queues_transfer(tmp_path):
    hub, store, notifier, transfer, _, workers = _hub(tmp_path)
    hub.ingest("flybehavior2", snap(), 100.0)
    assert [t["idx"] for t in store.trials()] == [1]
    assert [e["kind"] for e in store.recent_events()] == ["trial_finished", "session_started"]
    assert [c[0] for c in notifier.calls] == ["handle", "handle"]
    workers.drain()
    assert transfer.calls[0][0] == "process"
    assert transfer.calls[0][1][1:3] == ("b", "/pi/b")


def test_transfer_disabled_queues_nothing(tmp_path):
    hub, _, _, transfer, _, workers = _hub(tmp_path, transfer_enabled=False)
    hub.ingest("flybehavior2", snap(), 100.0)
    workers.drain()
    assert transfer.calls == []


def test_phase_changes_are_not_stored(tmp_path):
    hub, store, *_ = _hub(tmp_path)
    hub.ingest("flybehavior2", snap(), 100.0)
    n = len(store.recent_events())
    hub.ingest("flybehavior2", {**snap(), "phase": "BASELINE"}, 101.0)
    assert len(store.recent_events()) == n


def test_failed_transfer_raises_an_alert(tmp_path):
    hub, store, notifier, transfer, _, workers = _hub(tmp_path)
    transfer.process = lambda *a: (store.upsert_trial("flybehavior2", "b", a[3]),
                                   store.set_trial_state("flybehavior2", "b", 1, "failed",
                                                         error="rsync boom"), "failed")[-1]
    hub.ingest("flybehavior2", snap(), 100.0)
    workers.drain()
    assert ("alert", ("Pi 2 transfer failed", "T1: rsync boom")) in notifier.calls


def test_request_finalize_sets_pending_and_runs_after_trials(tmp_path):
    hub, store, _, transfer, finalizer, workers = _hub(tmp_path)
    hub.ingest("flybehavior2", snap(), 100.0)
    hub.request_finalize(PI2, "b", "/pi/b", "u@h:/d/", [])
    assert store.get_session("flybehavior2", "b")["state"] == "pending"
    workers.drain()
    assert transfer.calls[0][0] == "process" and finalizer.calls[0][0] == "run"


def test_resolve_rig_by_name_or_ip(tmp_path):
    hub, *_ = _hub(tmp_path)
    assert hub.resolve_rig("flybehavior2", "") is PI2
    assert hub.resolve_rig("raspberrypi", "100.93.23.70") is PI2
    assert hub.resolve_rig("nope", "1.1.1.1") is None


def test_state_shape(tmp_path):
    hub, store, *_ = _hub(tmp_path)
    hub.ingest("flybehavior2", snap(), 100.0)
    st = hub.state()
    rig = st["rigs"]["flybehavior2"]
    assert rig["reachable"] and rig["status"]["run"]["batch_dir"] == "b"
    assert rig["live_url"] == "http://100.93.23.70:8008/" and rig["muted"] is False
    assert st["server_time"] == 100.0 and st["trials"][0]["idx"] == 1


def test_fetch_status_handles_errors():
    def handler(request):
        if "100.93.23.70" in str(request.url):
            return httpx.Response(200, json={"phase": "GAP"})
        return httpx.Response(500)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            ok = await fetch_status(c, PI2)
            bad = await fetch_status(c, RigConfig("x", "X", "10.0.0.1"))
            return ok, bad

    assert asyncio.run(go()) == ({"phase": "GAP"}, None)


def test_notifier_exception_does_not_drop_trial(tmp_path):
    hub, store, notifier, transfer, _, workers = _hub(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("x")
    notifier.handle = boom
    hub.ingest("flybehavior2", snap(), 100.0)
    workers.drain()
    assert [t["idx"] for t in store.trials()] == [1]
    assert transfer.calls and transfer.calls[0][0] == "process"


def test_request_finalize_skips_landed_session(tmp_path):
    hub, store, _, _, finalizer, workers = _hub(tmp_path)
    store.set_session("flybehavior2", "b", "landed")
    hub.request_finalize(PI2, "b", "/pi/b", "u@h:/d/", [])
    workers.drain()
    assert store.get_session("flybehavior2", "b")["state"] == "landed"
    assert finalizer.calls == []
