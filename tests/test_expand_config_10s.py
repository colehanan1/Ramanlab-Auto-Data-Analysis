#!/usr/bin/env python3
"""config_v2_A_10s.yaml: 10 s IAA training/control session.

  training  - 6 x IAA 10 s, light SOLID t=3 -> t=10 of the odor (training only)
  post-test - IAA, then the 6 other vials + the paraffin click shuffled, then IAA
              (9 trials, 10 s odor)
  probe     - straight after test 9, no gap: 10 s + 15 s solid light + 10 s,
              in BOTH modes, tagged testing_10_LightOnly (inside cycle 2)
"""
import sys
from pathlib import Path

import pytest
import yaml

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))

import expand_config as ec
from expand_config import expand_config

VIALS = ["OFM_A", "OFM_B", "OFM_C", "OFM_E", "OFM_H", "OFM_L", "OFM_O"]
CFG_PATH = PICODE / "config_v2_A_10s.yaml"


@pytest.fixture(autouse=True)
def _default_pins():
    saved = list(ec._ALL_ODOR_PINS)
    ec._ALL_ODOR_PINS[:] = VIALS
    yield
    ec._ALL_ODOR_PINS[:] = saved


def _load():
    return yaml.safe_load(CFG_PATH.read_text())


def _recordings(steps):
    """Split a cycle's steps into recordings (Start .. Stop) and the gaps between."""
    recs, cur, gaps, gap = [], None, [], []
    for s in steps:
        if s["name"] == "Start Recording":
            gaps.append(gap)
            gap = []
            cur = [s]
        elif cur is not None:
            cur.append(s)
            if s["name"] == "Stop Recording":
                recs.append(cur)
                cur = None
        else:
            gap.append(s)
    return recs, gaps


def _odor_pin(rec):
    for s in rec:
        for a in s.get("actions", []):
            if a.get("device") == "ofm" and a.get("state") == "on":
                return a["pin"]
    return None


def _cycles(control):
    out = expand_config(_load(), control=control)
    return {c["cycle"]: c for c in out["cycles"]}


@pytest.mark.parametrize("control", [False, True])
def test_training_is_six_10s_iaa_with_light_3_to_10_only_when_training(control):
    c1 = _cycles(control)[1]
    assert c1["repeat"] == 6
    odor = [s for s in c1["steps"] if s.get("actions")]
    assert len(odor) == 1 and odor[0]["duration"] == 10
    assert odor[0]["actions"][0]["pin"] == "OFM_A"
    if control:
        assert "light_schedule" not in odor[0]
    else:
        assert odor[0]["light_schedule"] == [{"start": 3, "end": 10}]
        assert odor[0]["light_duty"] == 100.0


@pytest.mark.parametrize("control", [False, True])
def test_post_test_iaa_first_and_last_with_paraffin_in_shuffled_middle(control):
    cyc = _cycles(control)
    recs, _ = _recordings(cyc[2]["steps"])
    odor_recs = [r for r in recs if r[0]["odor_label"] != "LightOnly"]
    pins = [_odor_pin(r) for r in odor_recs]
    assert len(pins) == 9
    assert pins[0] == "OFM_A" and pins[-1] == "OFM_A"
    assert sorted(pins[1:-1]) == sorted(["OFM_B", "OFM_C", "OFM_E", "OFM_H",
                                         "OFM_L", "OFM_O", "OFM_P"])
    for r in odor_recs:
        odor = [s for s in r if s.get("actions")]
        assert odor[0]["duration"] == 10
        assert "light_schedule" not in odor[0]


def test_paraffin_position_varies_between_sessions():
    seen = set()
    for _ in range(60):
        recs, _ = _recordings(_cycles(False)[2]["steps"])
        pins = [_odor_pin(r) for r in recs]
        seen.add(pins.index("OFM_P"))
    assert seen <= set(range(1, 8)) and len(seen) > 3


@pytest.mark.parametrize("control", [False, True])
def test_light_probe_10_15_10_straight_after_test_9_as_testing_10(control):
    cyc = _cycles(control)
    assert 3 not in cyc and 4 not in cyc     # probe lives inside cycle 2
    assert cyc[2]["delay_after"] == 0
    recs, gaps = _recordings(cyc[2]["steps"])
    assert len(recs) == 10                  # 10th recording in cycle 2 -> testing_10
    probe = recs[-1]
    assert probe[0]["odor_label"] == "LightOnly"
    assert _odor_pin(probe) is None
    assert [s["duration"] for s in probe] == [0, 10, 15, 10, 0]
    light = probe[2]
    assert light["light_schedule"] == [{"start": 0, "end": 15}]
    assert light["light_duty"] == 100.0
    # 0 delay: nothing timed between test 9 stopping and the probe starting
    assert sum(s.get("duration", 0) for s in gaps[-1]) == 0


def test_rig_tag_for_probe_is_testing_10():
    """The rigs name a cycle-2 recording testing_<n>; mirror get_cycle_name."""
    src = (PICODE / "combinedv2_1_pi3.py").read_text()
    assert 'elif cycle_num == 2:\n        tag = f"testing_{step_count + 1}"' in src


# ── defaults elsewhere unchanged ─────────────────────────────────────────
def test_plain_config_keeps_cycle3_probe_30_25_30():
    out = expand_config({"format": "v2", "experiment": {"trained_odor": "OFM_A"}})
    cyc = {c["cycle"]: c for c in out["cycles"]}
    assert set(cyc) == {1, 2, 3}
    recs, _ = _recordings(cyc[2]["steps"])
    assert len(recs) == 8
    assert [s["duration"] for s in cyc[3]["steps"]] == [0, 30, 25, 30, 0]


def test_light_only_baseline_post_override_on_cycle3():
    out = expand_config({"format": "v2", "experiment": {
        "trained_odor": "OFM_A",
        "light_only": {"duration": 15, "baseline": 10, "post": 10}}})
    c3 = [c for c in out["cycles"] if c["cycle"] == 3][0]
    assert [s["duration"] for s in c3["steps"]] == [0, 10, 15, 10, 0]


def test_testing_paraffin_rejects_non_bool():
    with pytest.raises(ValueError):
        expand_config({"format": "v2", "experiment": {
            "trained_odor": "OFM_A", "testing_paraffin": "yes"}})
