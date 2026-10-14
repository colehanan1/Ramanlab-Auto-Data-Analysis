"""Unit tests for PiCode/rig_session.py — run ETA, finished trials, phone
reminders and the hub hand-off. Stdlib-only module, so it imports off-rig."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "PiCode"))

import phase_status as ps  # noqa: E402
import rig_session as rs  # noqa: E402

# cycle 1: (0 + 30 + 0 + 240) = 270 per rep × 2 = 540, + 600 delay_after = 1140
# cycle 2: 0 + 30 + (10 + 2 post_delay) + 5 = 47 ; cycle 3 has repeat 0 → skipped
CYCLES = [
    {"cycle": 2, "repeat": 1, "delay_after": 0, "steps": [
        {"name": "Start Recording", "duration": 0},
        {"name": "Baseline Period", "duration": 30},
        {"name": "Odor", "duration": 10, "post_delay": 2},
        {"name": "Stop Recording", "duration": 5}]},
    {"cycle": 1, "repeat": 2, "delay_after": 600, "steps": [
        {"name": "Start Recording", "duration": 0},
        {"name": "Odor", "duration": 30},
        {"name": "Stop Recording", "duration": 0},
        {"name": "Baseline Period", "duration": 240}]},
    {"cycle": 3, "repeat": 0, "steps": [{"name": "Start Recording", "duration": 999}]},
]


def test_plan_totals():
    plan = rs.RunPlan(CYCLES)
    assert plan.cycles_total_s == 1187
    assert plan.trials_total == 3


def test_remaining_from_first_step_is_whole_run():
    assert rs.RunPlan(CYCLES).remaining_from(1, 0, 0) == 1187


def test_remaining_counts_current_step_onward():
    assert rs.RunPlan(CYCLES).remaining_from(1, 1, 3) == 240 + 600 + 47


def test_delay_after_is_its_own_position():
    assert rs.RunPlan(CYCLES).remaining_from(1, 1, rs.DELAY_STEP) == 600 + 47


def test_cycles_run_in_numeric_order():
    assert rs.RunPlan(CYCLES).remaining_from(2, 0, 0) == 47


def test_unknown_position_raises():
    with pytest.raises(KeyError):
        rs.RunPlan(CYCLES).remaining_from(9, 0, 0)


def test_step_seconds_tolerates_missing_and_negative():
    assert rs.step_seconds({}) == 0
    assert rs.step_seconds({"duration": -5, "post_delay": 3}) == 3


# ── TrialLog / manifest ──────────────────────────────────────────────────────

def test_trial_log_records_sizes_and_basenames(tmp_path):
    (tmp_path / "t1.h264").write_bytes(b"x" * 10)
    (tmp_path / "t1.csv").write_bytes(b"y" * 4)
    log = rs.TrialLog(tmp_path)
    e = log.add(phase_tag="training_1", label="Training 1/6", odor="ACV",
                video=str(tmp_path / "t1.h264"), frames_csv=str(tmp_path / "t1.csv"),
                started_at=1.0, ended_at=2.0, closed=True)
    assert e == {"index": 1, "phase_tag": "training_1", "label": "Training 1/6",
                 "odor": "ACV", "video": "t1.h264", "frames_csv": "t1.csv",
                 "video_bytes": 10, "csv_bytes": 4, "started_at": 1.0,
                 "ended_at": 2.0, "closed": True}
    assert len(log) == 1


def test_trial_log_missing_file_is_zero_bytes_and_index_increments(tmp_path):
    log = rs.TrialLog(tmp_path)
    for _ in range(2):
        e = log.add(phase_tag="t", label="", odor="", video="nope.h264",
                    frames_csv="nope.csv", started_at=0, ended_at=0, closed=False)
    assert e["index"] == 2 and e["video_bytes"] == 0 and e["csv_bytes"] == 0


def test_trial_log_to_list_is_a_copy(tmp_path):
    log = rs.TrialLog(tmp_path)
    log.add(phase_tag="t", label="", odor="", video="a", frames_csv="b",
            started_at=0, ended_at=0, closed=True)
    log.to_list()[0]["index"] = 99
    assert log.to_list()[0]["index"] == 1


def test_manifest_is_relative_sorted_and_skips_hidden(tmp_path):
    (tmp_path / "b.csv").write_bytes(b"12")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_bytes(b"1")
    (tmp_path / ".hidden").write_bytes(b"1")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"1")
    assert rs.build_manifest(tmp_path) == [
        {"path": "b.csv", "bytes": 2}, {"path": "sub/a.txt", "bytes": 1}]


# ── reminders ────────────────────────────────────────────────────────────────

def test_reminders_fire_once_each_in_order():
    sched = rs.ReminderSchedule()
    sched.arm(0, 10800)                      # 3 h run
    assert sched.due(10800 - 5401, 10800) == []
    assert sched.due(5400, 10800) == [5400]
    assert sched.due(5401, 10800) == []
    assert sched.due(7200, 10800) == [3600]
    assert sched.due(9000, 10800) == []


def test_eta_slipping_later_does_not_refire():
    sched = rs.ReminderSchedule()
    sched.arm(0, 10800)
    assert sched.due(5400, 10800) == [5400]
    assert sched.due(5500, 12000) == []      # overran: 6500 s left again


def test_short_run_skips_the_90_minute_reminder():
    sched = rs.ReminderSchedule()
    sched.arm(0, 4500)                        # 75 min
    assert sched.due(900, 4500) == [3600]


def test_past_reminders_are_never_sent_late():
    sched = rs.ReminderSchedule()
    sched.arm(0, 4500)
    assert sched.due(4000, 4500) == [3600]    # never 5400


def test_run_under_an_hour_gets_no_reminders():
    sched = rs.ReminderSchedule()
    sched.arm(0, 3000)
    assert sched.due(2999, 3000) == []


def test_unarmed_schedule_is_silent():
    assert rs.ReminderSchedule().due(0, 10) == []


@pytest.mark.parametrize("sec, text", [(5400, "1 h 30 min"), (3600, "1 h"), (2700, "45 min")])
def test_fmt_left(sec, text):
    assert rs.fmt_left(sec) == text


def test_messages_are_ascii_titles_with_times():
    from datetime import datetime
    t = datetime(2026, 10, 14, 14, 5).timestamp()
    title, body = rs.start_message("Pi2 Fly 3 ACV Training", t, t + 3600)
    assert title == "Pi2 Fly 3 ACV Training - started" and title.isascii()
    assert "14:05" in body and "15:05" in body
    title, body = rs.reminder_message("Pi2 Fly 3", 5400, t)
    assert title == "Pi2 Fly 3 - 1 h 30 min left" and "14:05" in body


# ── RunTracker ───────────────────────────────────────────────────────────────

class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _tracker(tmp_path, clock):
    phase = ps.PhaseState(clock=clock)
    run = rs.RunTracker(phase, rs.RunPlan(CYCLES), host="flybehavior2",
                        batch_dir="october_14_batch_3", output_dir=tmp_path,
                        push_to="u@h:/data/flys_New/X/", push_mode="hub",
                        prefix="Pi2 Fly 3", clock=clock)
    return phase, run


def test_tracker_publishes_idle_run_metadata(tmp_path):
    phase, _ = _tracker(tmp_path, Clock())
    run = phase.snapshot()["run"]
    assert run["state"] == "idle" and run["batch_dir"] == "october_14_batch_3"
    assert run["push_mode"] == "hub" and run["trials_total"] == 3 and run["trials"] == []


def test_begin_sets_eta_from_startup_plus_plan(tmp_path):
    clock = Clock(1000.0)
    phase, run = _tracker(tmp_path, clock)
    training_at, end = run.begin(5400)
    assert training_at == 6400.0 and end == 6400.0 + 1187
    snap = phase.snapshot()["run"]
    assert snap["state"] == "running" and snap["expected_end_at"] == end
    assert snap["planned_total_s"] == 5400 + 1187 and snap["progress"] == 0.0


def test_mark_recomputes_eta_and_progress(tmp_path):
    clock = Clock(1000.0)
    phase, run = _tracker(tmp_path, clock)
    run.begin(0)
    clock.t = 1500.0
    run.mark(1, 1, rs.DELAY_STEP)              # 647 s left
    snap = phase.snapshot()["run"]
    assert snap["expected_end_at"] == 1500.0 + 647
    assert snap["progress"] == pytest.approx(1 - 647 / 1187)


def test_mark_unknown_position_keeps_eta(tmp_path):
    clock = Clock(1000.0)
    phase, run = _tracker(tmp_path, clock)
    run.begin(0)
    run.mark(99, 0, 0)
    assert phase.snapshot()["run"]["expected_end_at"] == 1000.0 + 1187


def test_trial_closed_publishes_the_list(tmp_path):
    phase, run = _tracker(tmp_path, Clock())
    run.trial_closed(phase_tag="training_1", label="Training 1/2", odor="ACV",
                     video="a.h264", frames_csv="a.csv", started_at=1, ended_at=2,
                     closed=True)
    snap = phase.snapshot()["run"]
    assert snap["trials_done"] == 1 and snap["trials"][0]["phase_tag"] == "training_1"


def test_finish_complete_and_failed(tmp_path):
    clock = Clock(1000.0)
    phase, run = _tracker(tmp_path, clock)
    run.begin(0)
    clock.t = 1200.0
    run.finish("complete")
    snap = phase.snapshot()["run"]
    assert snap["state"] == "complete" and snap["progress"] == 1.0
    assert snap["expected_end_at"] == 1200.0
    run.finish("failed", error="boom")
    snap = phase.snapshot()["run"]
    assert snap["state"] == "failed" and snap["error"] == "boom"


def test_due_messages_use_the_tracker_eta(tmp_path):
    clock = Clock(0.0)
    _, run = _tracker(tmp_path, clock)
    run.begin(10800 - 1187)                    # ends at 10800
    clock.t = 5400.0
    (title, body), = run.due_messages()
    assert title == "Pi2 Fly 3 - 1 h 30 min left"
    assert run.due_messages() == []


# ── hub hand-off ─────────────────────────────────────────────────────────────

class FakeHub:
    def __init__(self, states, fail_complete=False):
        self.states = list(states)
        self.fail_complete = fail_complete
        self.posts = []

    def post(self, url, payload, timeout=10.0):
        self.posts.append((url, payload))
        if self.fail_complete and url.endswith("/complete"):
            raise OSError("connection refused")
        return {}

    def get(self, url, timeout=10.0):
        state = self.states.pop(0)
        if isinstance(state, Exception):
            raise state
        return {"state": state}


def _handoff(hub, timeout_s=100.0):
    t = [0.0]
    return rs.hub_handoff(
        "http://hub:8765/", "flybehavior2", "october_14_batch_3",
        output_dir="/pi/october_14_batch_3", push_to="u@h:/d/", manifest=[{"path": "a", "bytes": 1}],
        timeout_s=timeout_s, poll_s=10.0, post=hub.post, get=hub.get,
        sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0], log=lambda m: None)


def test_session_url_quotes_parts():
    assert rs.session_url("http://h:1/", "pi 2", "a/b", "status") == \
        "http://h:1/api/sessions/pi%202/a%2Fb/status"


def test_handoff_landed_after_polls_and_sends_manifest():
    hub = FakeHub(["pending", "remuxing", "landed"])
    assert _handoff(hub) == "landed"
    url, payload = hub.posts[0]
    assert url == "http://hub:8765/api/sessions/flybehavior2/october_14_batch_3/complete"
    assert payload == {"output_dir": "/pi/october_14_batch_3", "push_to": "u@h:/d/",
                       "manifest": [{"path": "a", "bytes": 1}]}
    assert not any(u.endswith("/abort") for u, _ in hub.posts)


def test_handoff_failed_posts_abort():
    hub = FakeHub(["pending", "failed"])
    assert _handoff(hub) == "failed"
    assert hub.posts[-1][0].endswith("/abort")


def test_handoff_timeout_posts_abort():
    hub = FakeHub(["pending"] * 20)
    assert _handoff(hub, timeout_s=30.0) == "timeout"
    assert hub.posts[-1][0].endswith("/abort")


def test_handoff_transient_poll_error_keeps_polling():
    hub = FakeHub([OSError("blip"), "landed"])
    assert _handoff(hub) == "landed"


def test_handoff_unreachable_still_tries_abort():
    hub = FakeHub([], fail_complete=True)
    assert _handoff(hub) == "unreachable"
    assert hub.posts[-1][0].endswith("/abort")


def test_titles_are_ascii_even_with_non_ascii_prefix():
    t = 1000.0
    for title, _ in (rs.start_message("Pi2 Fly 3 \u00b5L-ACV", t, t + 60),
                     rs.reminder_message("Pi2 Fly 3 \u00b5L-ACV", 5400, t)):
        assert title.isascii() and "Pi2 Fly 3" in title


# ── best-effort guarantee ────────────────────────────────────────────────────

class _BrokenPhase:
    def set_run(self, **fields):
        if fields.get("state") == "idle":
            return  # let __init__ publish
        raise RuntimeError("boom")


def _broken_tracker(tmp_path):
    return rs.RunTracker(_BrokenPhase(), rs.RunPlan(CYCLES), host="h",
                         batch_dir="b", output_dir=tmp_path, push_to="",
                         push_mode="hub", prefix="p", clock=Clock())


def test_best_effort_methods_never_raise_and_warn_once(tmp_path, capsys):
    run = _broken_tracker(tmp_path)
    kw = dict(phase_tag="t", label="", odor="", video="a", frames_csv="b",
              started_at=0, ended_at=0, closed=True)
    for _ in range(2):
        run.mark(1, 0, 0)
        run.trial_closed(**kw)
        run.finish("complete")
    err = capsys.readouterr().err
    for name in ("mark", "trial_closed", "finish"):
        assert err.count(f"[RUN] {name} failed: RuntimeError('boom')") == 1


def test_unknown_mark_position_warns_once(tmp_path, capsys):
    _, run = _tracker(tmp_path, Clock())
    run.mark(99, 0, 0)
    run.mark(99, 0, 0)
    err = capsys.readouterr().err
    assert err.count("[RUN] mark: unknown step position (99, 0, 0)") == 1
