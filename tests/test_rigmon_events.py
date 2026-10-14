from rigmon.events import RigTracker


def snap(phase="GAP", trial_label="", state="running", trials=(), batch="oct_14_batch_3"):
    return {"phase": phase, "trial_label": trial_label, "label": "",
            "run": {"state": state, "batch_dir": batch, "output_dir": "/pi/" + batch,
                    "expected_end_at": 5000.0, "error": "boom" if state == "failed" else "",
                    "trials": [{"index": i, "label": f"T{i}"} for i in trials]}}


def kinds(events):
    return [e.kind for e in events]


def test_first_snapshot_replays_trials_and_start():
    tr = RigTracker("pi2")
    ev = tr.update(snap(trials=(1, 2)), 100.0)
    assert kinds(ev) == ["session_started", "trial_finished", "trial_finished"]
    assert all(e.data["replay"] for e in ev)


def test_new_trial_and_phase_change():
    tr = RigTracker("pi2")
    tr.update(snap(trials=(1,)), 100.0)
    ev = tr.update(snap(phase="BASELINE", trial_label="Training 2", trials=(1, 2)), 101.0)
    assert kinds(ev) == ["phase_changed", "trial_finished"]
    assert ev[1].data == {"batch_dir": "oct_14_batch_3", "output_dir": "/pi/oct_14_batch_3",
                          "trial": {"index": 2, "label": "T2"}, "replay": False}


def test_same_snapshot_twice_is_silent():
    tr = RigTracker("pi2")
    tr.update(snap(trials=(1,)), 100.0)
    assert tr.update(snap(trials=(1,)), 101.0) == []


def test_complete_and_failed_fire_once():
    tr = RigTracker("pi2")
    tr.update(snap(), 100.0)
    assert kinds(tr.update(snap(state="complete"), 101.0)) == ["session_complete"]
    assert tr.update(snap(state="complete"), 102.0) == []
    tr2 = RigTracker("pi2")
    tr2.update(snap(), 100.0)
    ev = tr2.update(snap(state="failed"), 101.0)
    assert kinds(ev) == ["session_failed"] and ev[0].data["error"] == "boom"


def test_new_batch_is_a_new_session_with_fresh_trial_numbers():
    tr = RigTracker("pi2")
    tr.update(snap(trials=(1,)), 100.0)
    ev = tr.update(snap(trials=(1,), batch="oct_14_batch_4"), 101.0)
    assert kinds(ev) == ["session_started", "trial_finished"]


def test_offline_after_threshold_once_then_online():
    tr = RigTracker("pi2", offline_after_s=60)
    tr.update(snap(), 100.0)
    assert tr.update(None, 159.0) == []
    ev = tr.update(None, 160.0)
    assert kinds(ev) == ["rig_offline"] and ev[0].data == {"last_seen": 100.0, "was_running": True}
    assert tr.update(None, 300.0) == []
    assert kinds(tr.update(snap(), 301.0)) == ["rig_online"]


def test_never_seen_rig_is_never_offline():
    assert RigTracker("pi2").update(None, 1e6) == []


def test_malformed_trial_is_skipped_and_tracking_continues():
    tr = RigTracker("r")
    good = {"index": 2, "label": "x"}
    snap = {"phase": "GAP", "run": {"state": "running", "batch_dir": "b",
                                    "trials": [{"label": "no index"}, {"index": "nan"}, good]}}
    evs = tr.update(snap, 1.0)
    assert [e.data["trial"] for e in evs if e.kind == "trial_finished"] == [good]
    assert tr.last is snap and tr.last_seen == 1.0
