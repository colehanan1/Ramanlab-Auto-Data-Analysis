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
