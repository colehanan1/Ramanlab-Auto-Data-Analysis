from rigmon.events import Event
from rigmon.store import Store

T1 = {"index": 1, "phase_tag": "training_1", "label": "Training 1/6", "odor": "ACV",
      "video": "a.h264", "frames_csv": "a.csv", "video_bytes": 10, "csv_bytes": 2,
      "ended_at": 50.0}


def test_events_round_trip_newest_first(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_event(Event("trial_finished", "pi2", 1.0, {"x": 1}))
    s.add_event(Event("session_complete", "pi2", 2.0, {}))
    ev = s.recent_events()
    assert [e["kind"] for e in ev] == ["session_complete", "trial_finished"]
    assert ev[1]["data"] == {"x": 1}


def test_trial_upsert_is_idempotent_and_state_updates(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.upsert_trial("pi2", "b", T1)
    s.set_trial_state("pi2", "b", 1, "done", mp4="a.mp4")
    s.upsert_trial("pi2", "b", T1)              # replay after restart
    t = s.get_trial("pi2", "b", 1)
    assert t["state"] == "done" and t["mp4"] == "a.mp4" and t["label"] == "Training 1/6"
    s.set_trial_state("pi2", "b", 1, "failed", error="x")
    assert s.get_trial("pi2", "b", 1)["mp4"] == "a.mp4"


def test_trials_newest_first(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.upsert_trial("pi2", "b", T1)
    s.upsert_trial("pi2", "b", {**T1, "index": 2, "ended_at": 60.0})
    assert [t["idx"] for t in s.trials()] == [2, 1]
    assert s.trials()[0]["state"] == "queued"


def test_sessions_and_mutes(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    assert s.get_session("pi2", "b") is None
    s.set_session("pi2", "b", "landed", dest="/d/b")
    assert s.get_session("pi2", "b")["dest"] == "/d/b"
    assert s.is_muted("pi2") is False
    s.set_muted("pi2", True)
    assert s.is_muted("pi2") and s.mutes() == {"pi2": True}


def test_interrupted_sessions_are_failed_on_start(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    for i, state in enumerate(["pending", "pulling", "remuxing", "landed", "aborted"]):
        s.set_session("pi2", f"b{i}", state)
    s2 = Store(tmp_path / "db.sqlite")
    assert s2.fail_interrupted_sessions() == 3
    assert [s2.get_session("pi2", f"b{i}")["state"] for i in range(5)] == \
        ["failed", "failed", "failed", "landed", "aborted"]
    assert s2.get_session("pi2", "b0")["error"] == "hub restarted during finalize"


def test_set_session_keeps_dest_and_replaces_error(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.set_session("pi2", "b", "landed", dest="/d/b")
    s.set_session("pi2", "b", "failed", error="boom")
    got = s.get_session("pi2", "b")
    assert got["state"] == "failed" and got["dest"] == "/d/b" and got["error"] == "boom"
    s.set_session("pi2", "b", "landed")
    got = s.get_session("pi2", "b")
    assert got["dest"] == "/d/b" and got["error"] == ""


def test_interrupted_trials_are_failed_too(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    for i, state in enumerate(["pulling", "remuxing", "done", "queued"], start=1):
        s.upsert_trial("pi2", "b", {**T1, "index": i})
        s.set_trial_state("pi2", "b", i, state)
    s.set_session("pi2", "b", "pulling")
    assert s.fail_interrupted_sessions() == 1
    states = [s.get_trial("pi2", "b", i) for i in range(1, 5)]
    assert [t["state"] for t in states] == ["failed", "failed", "done", "queued"]
    assert states[0]["error"] == "hub restarted mid-transfer" and states[2]["error"] == ""


def test_mount_check_lifecycle(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    cid = s.add_mount_check("pi2", "b1", "auto")
    assert s.mount_check_running("pi2") and s.mount_check_done("pi2", "b1", "auto")
    assert not s.mount_check_done("pi2", "b2", "auto") and not s.mount_check_done("pi2", "b1", "manual")
    s.update_mount_check(cid, state="issues", flags=[{"code": "wobble"}], slots=[{"slot": 1}],
                         sharpness=42.0, frames=600, json_path="/q/a.json", jpg_path="/q/a.jpg")
    row = s.latest_mount_check("pi2")
    assert row["state"] == "issues" and row["flags"] == [{"code": "wobble"}] and row["frames"] == 600
    assert not s.mount_check_running("pi2")
    assert s.latest_mount_check("pi3") is None


def test_sharpness_reference_uses_ok_checks_only(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    assert s.sharpness_reference("pi2") is None
    for val, state in ((40.0, "ok"), (60.0, "ok"), (5.0, "issues"), (50.0, "ok")):
        s.update_mount_check(s.add_mount_check("pi2", "b", "manual"), state=state, sharpness=val)
    assert s.sharpness_reference("pi2") == 50.0


def test_running_checks_fail_on_restart(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_mount_check("pi2", "b1", "auto")
    s2 = Store(tmp_path / "db.sqlite")
    assert s2.fail_interrupted_mount_checks() == 1
    row = s2.latest_mount_check("pi2")
    assert row["state"] == "error" and row["error"] == "hub restarted during the check"
    assert not s2.mount_check_running("pi2")
