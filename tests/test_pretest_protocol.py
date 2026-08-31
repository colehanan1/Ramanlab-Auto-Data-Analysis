"""The pre-test protocol: test panel -> 30 min rest -> training -> test panel.

Cycle numbers double as both execution order (the rig scripts run
``sorted(cycle_dict)``) and the phase tag (``get_cycle_name``), so this asserts
the numbers themselves, not just the shape:

    5 = pre-training panel   -> pretest_1..7
    6 = training             -> training_1..6
    7 = post-training panel  -> testing_1..7
    8 = light-only probe     -> testing_9
"""

import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PiCode"))
from expand_config import expand_config, _ALL_ODOR_PINS, _PIN_DISPLAY, _DEFAULTS

PRETEST, TRAINING, POSTTEST, LIGHT = 5, 6, 7, 8


def _v2(**exp):
    base = {"trained_odor": "OFM_H", "pretest": True,
            "testing_sequence": "all_random", "wait_before_testing": 0}
    base.update(exp)
    return {"format": "v2", "experiment": base}


def _cycles(result):
    return {c["cycle"]: c for c in result["cycles"]}


def _odors(cycle):
    return [s["odor_label"] for s in cycle["steps"] if s["name"] == "Start Recording"]


def test_four_cycles_numbered_for_execution_order():
    cyc = _cycles(expand_config(_v2()))
    assert sorted(cyc) == [PRETEST, TRAINING, POSTTEST, LIGHT]


def test_pretest_runs_before_training():
    """The rigs sort by cycle number, so the pre-test number must be lowest."""
    cyc = _cycles(expand_config(_v2()))
    assert min(cyc) == PRETEST


def test_pretest_is_all_seven_odors_once():
    cyc = _cycles(expand_config(_v2()))
    assert sorted(_odors(cyc[PRETEST])) == sorted(_PIN_DISPLAY[p] for p in _ALL_ODOR_PINS)


def test_pretest_has_no_light_even_in_training_mode():
    """A pre-training panel is a naive measurement: never paired with light."""
    cyc = _cycles(expand_config(_v2(), control=False))
    for step in cyc[PRETEST]["steps"]:
        assert "light_schedule" not in step


def test_rest_after_pretest_is_24_minutes():
    """1440 s, set 2026-08-28; was 1800."""
    cyc = _cycles(expand_config(_v2()))
    assert cyc[PRETEST]["delay_after"] == 1440
    assert _DEFAULTS["pretest_wait"] == 1440


def _puff_to_puff(from_cycle, to_cycle):
    """Seconds from the last odor ONSET in one cycle to the first in the next."""
    def is_odor(s):
        return (s.get("odor_label") and s.get("duration") == 30
                and "Baseline" not in s["name"])
    steps = from_cycle["steps"]
    last = max(i for i, s in enumerate(steps) if is_odor(s))
    tail = sum(s.get("duration", 0) for s in steps[last:])
    nxt = to_cycle["steps"]
    first = min(i for i, s in enumerate(nxt) if is_odor(s))
    head = sum(s.get("duration", 0) for s in nxt[:first])
    return tail + from_cycle["delay_after"] + head


def test_pretest_to_training_gap_is_built_like_the_standard_protocol():
    """Trailing ITI + delay_after + next baseline, exactly as training->testing.

    Without the trailing inter-trial gap the rest is 240 s short, so the same
    1440 s wait yields 25.5 min here and 29.5 min in the standard config.
    """
    cyc = _cycles(expand_config(_v2()))
    assert _puff_to_puff(cyc[PRETEST], cyc[TRAINING]) == 30 + 30 + 240 + 1440 + 30


def test_pretest_to_training_matches_standard_training_to_testing():
    plain = {"format": "v2", "experiment": {"trained_odor": "OFM_H"}}
    std = {c["cycle"]: c for c in expand_config(plain)["cycles"]}
    pre = _cycles(expand_config(_v2()))
    assert _puff_to_puff(pre[PRETEST], pre[TRAINING]) == _puff_to_puff(std[1], std[2])


def test_pretest_panel_still_has_only_six_internal_gaps():
    """The trailing gap is extra, not one of the between-trial gaps."""
    gaps = [s for s in _cycles(expand_config(_v2()))[PRETEST]["steps"]
            if s["name"] == "Baseline Period" and s["duration"] == 240]
    assert len(gaps) == 7          # 6 between 7 trials, + 1 trailing


def test_posttest_panel_has_no_trailing_gap():
    """Nothing follows it, so a trailing gap would just idle the rig."""
    steps = _cycles(expand_config(_v2()))[POSTTEST]["steps"]
    assert steps[-1]["name"] == "Stop Recording"


def test_pretest_wait_is_configurable():
    cyc = _cycles(expand_config(_v2(pretest_wait=600)))
    assert cyc[PRETEST]["delay_after"] == 600


def test_training_is_six_trials_of_the_trained_odor():
    cyc = _cycles(expand_config(_v2()))
    train = cyc[TRAINING]
    assert train["repeat"] == 6
    assert _odors(train) == ["Hexanol"]          # one per rep
    odor_steps = [s for s in train["steps"] if s.get("duration") == 30
                  and s.get("odor_label") == "Hexanol"]
    assert odor_steps, "no 30 s hexanol odor step"


def test_training_keeps_the_standard_four_minute_gap():
    cyc = _cycles(expand_config(_v2()))
    gaps = [s for s in cyc[TRAINING]["steps"]
            if s["name"] == "Baseline Period"
            and s["duration"] == _DEFAULTS["inter_trial_baseline"]]
    assert len(gaps) == 1                        # one per rep, x6 reps
    assert gaps[0]["duration"] == 240


def test_training_light_present_in_training_mode():
    cyc = _cycles(expand_config(_v2(), control=False))
    assert any("light_schedule" in s for s in cyc[TRAINING]["steps"])


def test_training_light_absent_in_control_mode():
    cyc = _cycles(expand_config(_v2(), control=True))
    assert not any("light_schedule" in s for s in cyc[TRAINING]["steps"])


def test_no_wait_between_training_and_posttest():
    cyc = _cycles(expand_config(_v2()))
    assert cyc[TRAINING]["delay_after"] == 0


def test_posttest_is_all_seven_odors_once():
    cyc = _cycles(expand_config(_v2()))
    assert sorted(_odors(cyc[POSTTEST])) == sorted(_PIN_DISPLAY[p] for p in _ALL_ODOR_PINS)


def test_pretest_and_posttest_are_shuffled_independently():
    """Two draws, not one order reused for both panels.

    Considered pairing them 2026-08-28 and deliberately kept independent.
    """
    pairs = set()
    for _ in range(30):
        cyc = _cycles(expand_config(_v2()))
        pairs.add((tuple(_odors(cyc[PRETEST])), tuple(_odors(cyc[POSTTEST]))))
    assert any(pre != post for pre, post in pairs)


def test_light_only_probe_last():
    cyc = _cycles(expand_config(_v2()))
    assert max(cyc) == LIGHT
    assert any(s.get("odor_label") == "LightOnly" for s in cyc[LIGHT]["steps"])


def test_pretest_off_by_default_keeps_the_standard_three_cycles():
    plain = {"format": "v2", "experiment": {"trained_odor": "OFM_H"}}
    assert sorted(_cycles(expand_config(plain))) == [1, 2, 3]


def test_pretest_honours_trained_first_sequence():
    """The panel style is still whatever testing_sequence says."""
    cyc = _cycles(expand_config(_v2(testing_sequence="trained_first")))
    assert len(_odors(cyc[PRETEST])) == 8
    assert _odors(cyc[PRETEST])[0] == "Hexanol"
