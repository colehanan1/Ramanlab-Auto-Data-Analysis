"""Tests for PiCode/config_v2_bench_remote.yaml, the short no-fly bench protocol used
to compare rig recording with the live stream OFF and ON.

The protocol is written in the expanded cycle/step form (the v2 format cannot shorten
the 30 s baseline/post), so expand_config must hand it back untouched, and the rig's
RunPlan must see exactly the timeline the bench analysis expects: 3 recorded trials of
10 s baseline / 5 s odor with light inside the odor window / 10 s post, 20 s ITI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))

from expand_config import expand_config   # noqa: E402
from rig_session import RunPlan, step_seconds   # noqa: E402

CONFIG = PICODE / "config_v2_bench_remote.yaml"


@pytest.fixture(scope="module")
def cfg():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return expand_config(raw)


def all_steps(cfg):
    return [(c["cycle"], s) for c in cfg["cycles"] for s in c["steps"]]


def trial_blocks(cfg):
    """Steps of each recorded trial: Start Recording .. Stop Recording inclusive."""
    blocks, cur = [], None
    for _, s in all_steps(cfg):
        if s["name"] == "Start Recording":
            cur = [s]
        elif cur is not None:
            cur.append(s)
            if s["name"] == "Stop Recording":
                blocks.append(cur)
                cur = None
    return blocks


def test_expands_cleanly_and_passes_through(cfg):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw.get("format") != "v2"
    assert expand_config(raw) is raw             # no code change needed on any Pi
    assert cfg["cycles"]


def test_exactly_three_recorded_trials(cfg):
    assert RunPlan(cfg["cycles"]).trials_total == 3
    assert len(trial_blocks(cfg)) == 3
    starts = [s for _, s in all_steps(cfg) if s["name"] == "Start Recording"]
    stops = [s for _, s in all_steps(cfg) if s["name"] == "Stop Recording"]
    assert len(starts) == len(stops) == 3


def test_total_planned_duration_is_short(cfg):
    total = RunPlan(cfg["cycles"]).cycles_total_s
    assert total < 10 * 60
    assert total == 3 * 25 + 2 * 20              # trials + the two ITIs, none trailing


def test_each_trial_is_10s_baseline_5s_odor_10s_post(cfg):
    for block in trial_blocks(cfg):
        timed = [s for s in block if s["name"] not in ("Start Recording", "Stop Recording")]
        assert [s["name"] for s in timed] == [
            "Baseline Period", "Odor + Light Pattern", "Post-Stimulus Period"]
        assert [step_seconds(s) for s in timed] == [10, 5, 10]


def test_odor_window_is_5s_with_an_ofm_on_action(cfg):
    for block in trial_blocks(cfg):
        odor = next(s for s in block if s["name"] == "Odor + Light Pattern")
        assert odor["duration"] == 5
        on = [a for a in odor["actions"] if a["device"] == "ofm"]
        assert len(on) == 1 and on[0]["state"] == "on" and on[0]["pin"].startswith("OFM_")


def test_light_is_inside_the_odor_window_and_solid(cfg):
    for block in trial_blocks(cfg):
        odor = next(s for s in block if s["name"] == "Odor + Light Pattern")
        sched = odor["light_schedule"]
        assert sched, "light must be scheduled"
        for seg in sched:
            assert 0 <= seg["start"] < seg["end"] <= odor["duration"]
        # solid: the rig holds the pin high at 100 % duty (no pulse edges in the data)
        assert odor["light_duty"] == 100.0


def test_no_light_outside_the_odor_step(cfg):
    for _, s in all_steps(cfg):
        if s["name"] != "Odor + Light Pattern":
            assert "light_schedule" not in s
            assert not any(a.get("device") == "light" for a in s.get("actions", []))


def test_iti_is_20s_between_trials_only(cfg):
    steps = [s for _, s in all_steps(cfg)]
    stops = [i for i, s in enumerate(steps) if s["name"] == "Stop Recording"]
    for i in stops[:-1]:
        nxt = steps[i + 1]
        assert nxt["name"] == "Baseline Period" and nxt["duration"] == 20
        assert steps[i + 2]["name"] == "Start Recording"
    assert stops[-1] == len(steps) - 1           # nothing after the last trial
    assert cfg["cycles"][-1].get("delay_after", 0) == 0


def test_cycle_numbering_gives_testing_tags(cfg):
    # rig phase tags: cycle 2 -> testing_1..3 (get_cycle_name in combinedv2_1*.py)
    assert [c["cycle"] for c in cfg["cycles"]] == [2]
    assert cfg["cycles"][0]["repeat"] == 1


def test_no_other_config_family_globs_pick_it_up():
    # tests/test_training_iti.py and test_norest_configs.py glob these patterns
    assert not CONFIG.match("config_v2_?.yaml")
    assert not CONFIG.match("config_v2_*_norest.yaml")
    assert not CONFIG.match("config_v2_*_pretest.yaml")
