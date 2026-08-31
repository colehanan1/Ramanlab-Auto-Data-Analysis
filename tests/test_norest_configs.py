"""The seven ``config_v2_*_norest.yaml`` rig configs.

One per odor: training identical to the standard protocol, zero training->testing
rest period, and a testing panel of all 7 odors in a fresh random order each run.
These are the files handed to ``combinedv2_1.py --config``, so they are checked
by expanding them exactly as the rig scripts do.
"""

import sys, os
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PiCode"))
from expand_config import expand_config, _ALL_ODOR_PINS, _PIN_DISPLAY, _DEFAULTS

PICODE = os.path.join(os.path.dirname(__file__), "..", "PiCode")
LETTERS = [p.split("_")[1] for p in _ALL_ODOR_PINS]


def _load(letter):
    with open(os.path.join(PICODE, f"config_v2_{letter}_norest.yaml")) as fh:
        return yaml.safe_load(fh)


def _cycle(result, n):
    return next(c for c in result["cycles"] if c["cycle"] == n)


def _odors(cycle):
    return [s["odor_label"] for s in cycle["steps"] if s["name"] == "Start Recording"]


@pytest.mark.parametrize("letter", LETTERS)
def test_config_exists_and_trains_its_own_odor(letter):
    cfg = _load(letter)
    assert cfg["format"] == "v2"
    assert cfg["experiment"]["trained_odor"] == f"OFM_{letter}"


@pytest.mark.parametrize("letter", LETTERS)
def test_no_rest_before_testing(letter):
    assert _cycle(expand_config(_load(letter)), 1)["delay_after"] == 0


@pytest.mark.parametrize("letter", LETTERS)
def test_training_identical_to_standard_protocol(letter):
    """Same 6 trials and same step timings as a config without the overrides."""
    plain = dict(_load(letter))
    plain["experiment"] = {k: v for k, v in plain["experiment"].items()
                           if k not in ("wait_before_testing", "testing_sequence")}
    base = _cycle(expand_config(plain), 1)
    norest = _cycle(expand_config(_load(letter)), 1)
    assert norest["repeat"] == base["repeat"] == _DEFAULTS["classical_trials"]
    assert norest["steps"] == base["steps"]


@pytest.mark.parametrize("letter", LETTERS)
def test_testing_is_all_seven_odors_once(letter):
    odors = _odors(_cycle(expand_config(_load(letter)), 2))
    assert sorted(odors) == sorted(_PIN_DISPLAY[p] for p in _ALL_ODOR_PINS)


@pytest.mark.parametrize("letter", LETTERS)
def test_testing_order_is_redrawn_each_run(letter):
    cfg = _load(letter)
    seen = {tuple(_odors(_cycle(expand_config(cfg), 2))) for _ in range(30)}
    assert len(seen) > 1


@pytest.mark.parametrize("letter", LETTERS)
def test_control_mode_has_no_light_in_training_or_testing(letter):
    result = expand_config(_load(letter), control=True)
    for n in (1, 2):
        for step in _cycle(result, n)["steps"]:
            assert "light_schedule" not in step


@pytest.mark.parametrize("letter", LETTERS)
def test_light_only_probe_still_present(letter):
    """Cycle 3 is the light-response probe and must survive in both modes."""
    for control in (False, True):
        c3 = _cycle(expand_config(_load(letter), control=control), 3)
        assert any(s.get("odor_label") == "LightOnly" for s in c3["steps"])


# ── config_v2_*_pretest.yaml: pre-test / train / post-test ────────────

def _load_pretest(letter):
    with open(os.path.join(PICODE, f"config_v2_{letter}_pretest.yaml")) as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("letter", LETTERS)
def test_pretest_config_exists_per_odor(letter):
    cfg = _load_pretest(letter)
    assert cfg["experiment"]["trained_odor"] == f"OFM_{letter}"
    assert cfg["experiment"]["pretest"] is True


@pytest.mark.parametrize("letter", LETTERS)
def test_pretest_config_cycle_order(letter):
    result = expand_config(_load_pretest(letter))
    assert [c["cycle"] for c in result["cycles"]] == [5, 6, 7, 8]


@pytest.mark.parametrize("letter", LETTERS)
def test_pretest_config_timings(letter):
    cyc = {c["cycle"]: c for c in expand_config(_load_pretest(letter))["cycles"]}
    assert cyc[5]["delay_after"] == 1440      # 24 min rest after the naive panel
    assert cyc[6]["repeat"] == 6              # 6 training trials
    assert cyc[6]["delay_after"] == 0         # post-test starts right after


@pytest.mark.parametrize("letter", LETTERS)
def test_pretest_config_both_panels_are_full_random(letter):
    cyc = {c["cycle"]: c for c in expand_config(_load_pretest(letter))["cycles"]}
    every = sorted(_PIN_DISPLAY[p] for p in _ALL_ODOR_PINS)
    for n in (5, 7):
        assert sorted(_odors(cyc[n])) == every


@pytest.mark.parametrize("letter", LETTERS)
def test_pretest_config_trains_its_own_odor_for_30s(letter):
    cyc = {c["cycle"]: c for c in expand_config(_load_pretest(letter))["cycles"]}
    label = _PIN_DISPLAY[f"OFM_{letter}"]
    assert _odors(cyc[6]) == [label]
    assert any(s.get("odor_label") == label and s.get("duration") == 30
               for s in cyc[6]["steps"])
