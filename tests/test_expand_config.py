#!/usr/bin/env python3
"""Tests for expand_config.py preprocessor."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PiCode"))

from expand_config import expand_config, _testing_odor_sequence, _ALL_ODOR_PINS

def _make_v2(trained="OFM_B"):
    return {
        "format": "v2",
        "experiment": {
            "trained_odor": trained,
            "light": {"hertz": 5, "duty_cycle": 50},
            "light_schedule": [
                {"start": 5, "end": 10},
                {"start": 11, "end": 15},
                {"start": 16, "end": 20},
                {"start": 22, "end": 25},
                {"start": 26, "end": 30},
            ],
        },
    }

def _count_recordings(cycle):
    return sum(1 for s in cycle["steps"] if s["name"] == "Start Recording")

# ── Tests ─────────────────────────────────────────────────────────────

def test_passthrough():
    old = {"cycles": [{"cycle": 1, "repeat": 1, "steps": []}]}
    assert expand_config(old) is old
    print("PASS: test_passthrough")

def test_3_cycles():
    result = expand_config(_make_v2("OFM_B"))
    assert len(result["cycles"]) == 3
    assert [c["cycle"] for c in result["cycles"]] == [1, 2, 3]
    print("PASS: test_3_cycles")

def test_cycle1_training():
    result = expand_config(_make_v2("OFM_B"))
    c1 = result["cycles"][0]
    assert c1["repeat"] == 6
    assert c1["delay_after"] == 1680
    odor_steps = [s for s in c1["steps"] if "light_schedule" in s]
    assert len(odor_steps) == 1
    assert odor_steps[0]["actions"][0]["pin"] == "OFM_B"
    print("PASS: test_cycle1_training")

def test_cycle1_control_no_light():
    result = expand_config(_make_v2("OFM_B"), control=True)
    c1 = result["cycles"][0]
    for s in c1["steps"]:
        assert "light_schedule" not in s
    print("PASS: test_cycle1_control_no_light")

def test_testing_sequence_structure():
    """Testing 1=trained, 2-7=remaining (randomized), 8=trained again."""
    seq = _testing_odor_sequence("OFM_H")
    assert seq[0] == "OFM_H", "First testing trial must be trained odor"
    assert seq[-1] == "OFM_H", "Last testing trial must be trained odor"
    assert len(seq) == 8, "Must have 8 testing odors"
    middle = seq[1:-1]
    assert "OFM_H" not in middle, "Trained odor must not appear in middle"
    assert set(middle) == set(p for p in _ALL_ODOR_PINS if p != "OFM_H"), \
        "Middle must contain all remaining odors exactly once"
    print("PASS: test_testing_sequence_structure")

def test_testing_sequence_B():
    """Testing sequence for OFM_B: trained first/last, 6 remaining in middle."""
    seq = _testing_odor_sequence("OFM_B")
    assert seq[0] == "OFM_B"
    assert seq[-1] == "OFM_B"
    assert len(seq) == 8
    middle = set(seq[1:-1])
    expected_middle = set(_ALL_ODOR_PINS) - {"OFM_B"}
    assert middle == expected_middle
    print("PASS: test_testing_sequence_B")

def test_testing_sequence_randomized():
    """Verify testing 2-7 are randomized (not always sorted)."""
    # Run multiple times; if always sorted, randomization is broken
    seen_orders = set()
    for _ in range(20):
        seq = _testing_odor_sequence("OFM_H")
        middle = tuple(seq[1:-1])
        seen_orders.add(middle)
    assert len(seen_orders) > 1, \
        "Testing 2-7 should be randomized but got same order 20 times"
    print("PASS: test_testing_sequence_randomized")

def test_cycle2_no_light():
    """Cycle 2: 8 testing trials, NO light on any."""
    result = expand_config(_make_v2("OFM_H"))
    c2 = result["cycles"][1]
    assert c2["repeat"] == 1
    assert _count_recordings(c2) == 8
    for s in c2["steps"]:
        assert "light_schedule" not in s, f"Testing should have no light: {s['name']}"
    print("PASS: test_cycle2_no_light")

def test_cycle2_odor_order():
    """Verify cycle 2 odor order: trained first/last, 6 remaining in middle."""
    result = expand_config(_make_v2("OFM_H"))
    c2 = result["cycles"][1]
    odor_steps = [s for s in c2["steps"]
                  if any(a.get("device") == "ofm" for a in s.get("actions", []))]
    pins = [s["actions"][0]["pin"] for s in odor_steps]
    assert pins[0] == "OFM_H"
    assert pins[-1] == "OFM_H"
    assert set(pins[1:-1]) == set(p for p in _ALL_ODOR_PINS if p != "OFM_H")
    print("PASS: test_cycle2_odor_order")

def test_cycle3_light_only_training():
    """Cycle 3: light flash, no odor, in training mode."""
    result = expand_config(_make_v2("OFM_B"))
    c3 = result["cycles"][2]
    assert c3["cycle"] == 3
    assert c3["repeat"] == 1
    assert _count_recordings(c3) == 1
    light_step = [s for s in c3["steps"] if "Light Only" in s.get("name", "")]
    assert len(light_step) == 1
    assert light_step[0]["actions"] == []
    assert "light_schedule" in light_step[0]
    print("PASS: test_cycle3_light_only_training")

def test_cycle3_light_only_control():
    """Cycle 3 in control: no light, no odor."""
    result = expand_config(_make_v2("OFM_B"), control=True)
    c3 = result["cycles"][2]
    light_step = [s for s in c3["steps"] if "Light Only" in s.get("name", "")]
    assert "light_schedule" not in light_step[0]
    print("PASS: test_cycle3_light_only_control")

def test_control_no_light_anywhere():
    result = expand_config(_make_v2("OFM_B"), control=True)
    for c in result["cycles"]:
        for s in c["steps"]:
            assert "light_schedule" not in s
    print("PASS: test_control_no_light_anywhere")

def test_iti_timing():
    result = expand_config(_make_v2("OFM_B"))
    c1 = result["cycles"][0]
    total = sum(s["duration"] for s in c1["steps"])
    assert total == 330  # 0+30+30+90+0+180
    print("PASS: test_iti_timing")

def test_test_valves():
    cfg = {"format": "v2", "experiment": {"mode": "test_valves"}}
    result = expand_config(cfg)
    assert len(result["cycles"]) == 1
    odor_steps = [s for s in result["cycles"][0]["steps"] if s.get("name", "").startswith("Test OFM_")]
    assert len(odor_steps) == 7
    print("PASS: test_test_valves")

def test_all_trained_odors():
    for odor in ["OFM_B", "OFM_H", "OFM_A", "OFM_E", "OFM_O", "OFM_X"]:
        result = expand_config(_make_v2(odor))
        assert len(result["cycles"]) == 3, f"Failed for {odor}"
    print("PASS: test_all_trained_odors")

def test_odor_label_on_start_recording():
    """All Start Recording steps must have an odor_label field."""
    result = expand_config(_make_v2("OFM_H"))
    for c in result["cycles"]:
        for s in c["steps"]:
            if s["name"] == "Start Recording":
                assert "odor_label" in s, f"Start Recording missing odor_label: {s}"
                assert isinstance(s["odor_label"], str) and s["odor_label"], \
                    f"odor_label must be a non-empty string: {s}"
    print("PASS: test_odor_label_on_start_recording")

def test_odor_label_on_odor_steps():
    """Odor action steps must have an odor_label field."""
    result = expand_config(_make_v2("OFM_H"))
    for c in result["cycles"]:
        for s in c["steps"]:
            if any(a.get("device") == "ofm" for a in s.get("actions", [])):
                assert "odor_label" in s, f"Odor step missing odor_label: {s}"
    print("PASS: test_odor_label_on_odor_steps")

def test_odor_label_light_only():
    """Cycle 3 light-only step should have odor_label='LightOnly'."""
    result = expand_config(_make_v2("OFM_B"))
    c3 = result["cycles"][2]
    light_step = [s for s in c3["steps"] if "Light Only" in s.get("name", "")]
    assert light_step[0].get("odor_label") == "LightOnly"
    print("PASS: test_odor_label_light_only")

def test_odor_label_values():
    """Check that training odor_label matches the trained odor display name."""
    result = expand_config(_make_v2("OFM_H"))
    c1 = result["cycles"][0]
    start_steps = [s for s in c1["steps"] if s["name"] == "Start Recording"]
    assert start_steps[0]["odor_label"] == "Hexanol"
    print("PASS: test_odor_label_values")

if __name__ == "__main__":
    test_passthrough()
    test_3_cycles()
    test_cycle1_training()
    test_cycle1_control_no_light()
    test_testing_sequence_structure()
    test_testing_sequence_B()
    test_testing_sequence_randomized()
    test_cycle2_no_light()
    test_cycle2_odor_order()
    test_cycle3_light_only_training()
    test_cycle3_light_only_control()
    test_control_no_light_anywhere()
    test_iti_timing()
    test_test_valves()
    test_all_trained_odors()
    test_odor_label_on_start_recording()
    test_odor_label_on_odor_steps()
    test_odor_label_light_only()
    test_odor_label_values()
    print("\n=== ALL TESTS PASSED ===")
