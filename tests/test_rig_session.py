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
