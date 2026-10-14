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
