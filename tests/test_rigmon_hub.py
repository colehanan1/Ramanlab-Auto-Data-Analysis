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
    hub.ingest("flybehavior2", snap(), 100.0)
    store.set_session("flybehavior2", "b", "landed")
    hub.request_finalize(PI2, "b", "/pi/b", "u@h:/d/", [])
    workers.drain()
    assert store.get_session("flybehavior2", "b")["state"] == "landed"
    assert finalizer.calls == []


def test_request_finalize_refuses_without_or_mismatched_snapshot(tmp_path):
    import pytest
    hub, store, _, _, finalizer, workers = _hub(tmp_path)
    with pytest.raises(ValueError):
        hub.request_finalize(PI2, "b", "/pi/b", "u@h:/d/", [])
    hub.ingest("flybehavior2", snap(), 100.0)
    for batch, out in (("other", "/pi/b"), ("b", "/etc")):
        with pytest.raises(ValueError):
            hub.request_finalize(PI2, batch, out, "u@h:/d/", [])
    workers.drain()
    assert store.get_session("flybehavior2", "b") is None
    assert store.get_session("flybehavior2", "other") is None
    assert finalizer.calls == []


def test_replayed_trials_of_a_landed_session_are_not_requeued(tmp_path):
    hub, store, _, transfer, _, workers = _hub(tmp_path)
    store.set_session("flybehavior2", "b", "landed")
    hub.ingest("flybehavior2", snap(), 100.0)
    workers.drain()
    assert transfer.calls == []


def test_replayed_done_trial_is_not_requeued(tmp_path):
    hub, store, _, transfer, _, workers = _hub(tmp_path)
    store.upsert_trial("flybehavior2", "b", {"index": 1, "video": "t1.h264"})
    store.set_trial_state("flybehavior2", "b", 1, "done", mp4="t1.mp4")
    hub.ingest("flybehavior2", snap(trials=(1, 2)), 100.0)
    workers.drain()
    assert [c[1][3]["index"] for c in transfer.calls] == [2]


def test_state_hides_push_to_and_output_dir_without_mutating(tmp_path):
    hub, *_ = _hub(tmp_path)
    s = snap()
    s["run"]["push_to"] = "u@h:/secret/"
    hub.ingest("flybehavior2", s, 100.0)
    run = hub.state()["rigs"]["flybehavior2"]["status"]["run"]
    assert "push_to" not in run and "output_dir" not in run
    assert run["batch_dir"] == "b" and run["trials"][0]["index"] == 1
    assert hub.latest["flybehavior2"]["run"]["push_to"] == "u@h:/secret/"
    assert hub.latest["flybehavior2"]["run"]["output_dir"] == "/pi/b"


class FakeMC:
    def __init__(self):
        self.started = []

    def start(self, rig, batch_dir, trigger, stop=lambda: False):
        self.started.append((rig.name, batch_dir, trigger))
        return len(self.started)


def wait_snap(started_at, server_time, ends_at, batch="b2", trials_done=0, phase="WAITING", state="running"):
    return {"phase": phase, "trial_label": "", "label": "Training 1 starts in", "server_time": server_time,
            "started_at": started_at, "ends_at": ends_at,
            "run": {"state": state, "batch_dir": batch, "output_dir": "/pi/" + batch, "trials": [],
                    "trials_done": trials_done}}


def _mc_hub(tmp_path):
    hub, store, *_ = _hub(tmp_path)
    hub.mountcheck = FakeMC()
    return hub, store


def test_auto_mount_check_fires_once_after_30_min(tmp_path):
    hub, store = _mc_hub(tmp_path)
    t0 = 1_000_000.0
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 1799, t0 + 5400), 50.0)
    assert hub.mountcheck.started == []
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 1800, t0 + 5400), 51.0)
    assert hub.mountcheck.started == [("flybehavior2", "b2", "auto")]
    store.add_mount_check("flybehavior2", "b2", "auto")   # what the real runner records
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 1900, t0 + 5400), 52.0)
    assert len(hub.mountcheck.started) == 1


def test_trigger_uses_pi_clock(tmp_path):
    hub, _ = _mc_hub(tmp_path)
    t0 = 1_000_000.0          # Pi clock; the hub clock (100.0) is wildly different
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 2000, t0 + 5400), 100.0)
    assert hub.mountcheck.started


def test_no_check_too_close_to_first_trial_or_mid_run(tmp_path):
    hub, _ = _mc_hub(tmp_path)
    t0 = 1_000_000.0
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 4300, t0 + 5400), 1.0)          # 1100 s left < 1200
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 2000, t0 + 5400, trials_done=3), 2.0)
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 2000, t0 + 5400, phase="BASELINE"), 3.0)
    assert hub.mountcheck.started == []


def test_new_batch_gets_its_own_check(tmp_path):
    hub, store = _mc_hub(tmp_path)
    t0 = 1_000_000.0
    store.add_mount_check("flybehavior2", "b1", "auto")
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 2000, t0 + 5400, batch="b2"), 1.0)
    assert hub.mountcheck.started == [("flybehavior2", "b2", "auto")]


def test_disabled_mount_check_never_fires(tmp_path):
    from dataclasses import replace
    from rigmon.config import MountCheckConfig
    hub, _ = _mc_hub(tmp_path)
    hub.cfg = replace(hub.cfg, mount_check=MountCheckConfig(enabled=False))
    t0 = 1_000_000.0
    hub.ingest("flybehavior2", wait_snap(t0, t0 + 2000, t0 + 5400), 1.0)
    assert hub.mountcheck.started == []


def test_state_includes_latest_mount_check(tmp_path):
    hub, store = _mc_hub(tmp_path)
    cid = store.add_mount_check("flybehavior2", "b2", "manual")
    store.update_mount_check(cid, state="ok")
    assert hub.state()["rigs"]["flybehavior2"]["mount_check"]["state"] == "ok"


def test_build_mountcheck_disabled_and_missing_config(tmp_path):
    from dataclasses import replace
    from rigmon.__main__ import build_mountcheck
    from rigmon.config import MountCheckConfig
    hub, store, notifier, *_ = _hub(tmp_path)
    off = replace(hub.cfg, mount_check=MountCheckConfig(enabled=False), pipeline_config=tmp_path / "nope.yaml")
    assert build_mountcheck(off, store, notifier) == (None, None)
    on = replace(hub.cfg, mount_check=MountCheckConfig(enabled=True), pipeline_config=tmp_path / "nope.yaml")
    runner, reason = build_mountcheck(on, store, notifier)
    assert runner is None and "pipeline_config not found" in reason


def test_build_mountcheck_degrades_when_settings_fail_to_load(tmp_path):
    from dataclasses import replace
    from rigmon.__main__ import build_mountcheck
    from rigmon.config import MountCheckConfig
    hub, store, notifier, *_ = _hub(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [valid")
    on = replace(hub.cfg, mount_check=MountCheckConfig(enabled=True), pipeline_config=bad)
    runner, reason = build_mountcheck(on, store, notifier)
    assert runner is None and reason


def test_state_reports_mount_check_error_and_hides_paths(tmp_path):
    hub, store = _mc_hub(tmp_path)
    hub.mountcheck = None
    hub.mount_check_error = "pipeline_config not found: /x"
    cid = store.add_mount_check("flybehavior2", "b2", "manual")
    store.update_mount_check(cid, state="ok", json_path="/secret/dir/a.json", jpg_path="/secret/dir/a.jpg")
    st = hub.state()
    assert st["mount_check_error"] == "pipeline_config not found: /x" and st["mount_check_enabled"] is False
    mc = st["rigs"]["flybehavior2"]["mount_check"]
    assert "json_path" not in mc and "jpg_path" not in mc and mc["jpg_name"] == "a.jpg"
    assert store.latest_mount_check("flybehavior2")["jpg_path"] == "/secret/dir/a.jpg"   # store copy untouched


def test_is_recording_by_phase(tmp_path):
    hub, *_ = _hub(tmp_path)
    for phase, rec in (("BASELINE", True), ("STIMULUS", True), ("POST", True), ("GAP", False),
                       ("WAITING", False), ("COMPLETE", False)):
        hub.ingest("flybehavior2", {**snap(), "phase": phase}, 100.0)
        assert hub.is_recording("flybehavior2") is rec
    hub.ingest("flybehavior2", {**snap(), "phase": "STIMULUS"}, 150.0)
    hub.ingest("flybehavior2", None, 200.0)
    assert hub.is_recording("flybehavior2") is True   # last known recording state is kept when the rig drops off


class FakeTraces:
    def __init__(self):
        self.runs = []

    def run(self, rig, batch_dir, trial):
        self.runs.append((rig.name, batch_dir, int(trial["index"])))
        return "x"


def test_trace_job_runs_after_a_successful_copy(tmp_path):
    hub, store, notifier, transfer, finalizer, workers = _hub(tmp_path)
    hub.traces = FakeTraces()
    hub.ingest("flybehavior2", snap(), 100.0)
    workers.drain()
    assert hub.traces.runs == [("flybehavior2", "b", 1)]


def test_no_trace_job_after_a_failed_copy(tmp_path):
    hub, store, notifier, transfer, finalizer, workers = _hub(tmp_path)
    hub.traces = FakeTraces()
    transfer.process = lambda *a: "failed"
    hub.ingest("flybehavior2", snap(), 100.0)
    workers.drain()
    assert hub.traces.runs == []


def test_replay_requeues_missing_trace_only(tmp_path):
    hub, store, notifier, transfer, finalizer, workers = _hub(tmp_path)
    hub.traces = FakeTraces()
    store.upsert_trial("flybehavior2", "b", {"index": 1, "video": "t1.h264", "frames_csv": "t1.csv", "ended_at": 1.0})
    store.set_trial_state("flybehavior2", "b", 1, "done", mp4="t1.mp4")
    st = tmp_path / "s" / "flybehavior2" / "b"
    st.mkdir(parents=True)
    (st / "t1.mp4").write_bytes(b"x")
    transfer.staging_dir = lambda rig, batch: tmp_path / "s" / rig / batch
    hub.ingest("flybehavior2", snap(), 100.0)
    workers.drain()
    assert hub.traces.runs == [("flybehavior2", "b", 1)]
    assert not [c for c in transfer.calls if c[0] == "process"]
