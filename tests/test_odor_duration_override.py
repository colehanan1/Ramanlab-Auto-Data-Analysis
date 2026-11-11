"""experiment.odor_duration: a per-config odor-on time for every odor trial.

Default stays 30 s for every protocol. A config that sets the key (only
config_v2_A_pretest.yaml, 5 s since 2026-10-14) shortens the odor step of the
pre-test, training and post-test trials alike; baseline, post-stimulus and the
light-only probe keep their lengths.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))
from expand_config import expand_config, _DEFAULTS

PRETEST, TRAINING, POSTTEST, LIGHT = 5, 6, 7, 8


def _v2(pretest=True, **exp):
    base = {"trained_odor": "OFM_A", "pretest": pretest,
            "testing_sequence": "all_random", "wait_before_testing": 0}
    base.update(exp)
    return {"format": "v2", "experiment": base}


def _odor_steps(cycle):
    return [s for s in cycle["steps"]
            if any(a.get("device") == "ofm" and a.get("state") == "on"
                   for a in s.get("actions", []))]


def _durations(cycle, name):
    return [s["duration"] for s in cycle["steps"] if s["name"] == name]


def _by_num(result):
    return {c["cycle"]: c for c in result["cycles"]}


def test_default_odor_is_30_s_everywhere():
    cyc = _by_num(expand_config(_v2(), control=True))
    for num in (PRETEST, TRAINING, POSTTEST):
        durs = [s["duration"] for s in _odor_steps(cyc[num])]
        assert durs and set(durs) == {30}, (num, durs)


@pytest.mark.parametrize("control", [True, False])
def test_override_shortens_every_odor_trial(control):
    cfg = _v2(odor_duration=5, light_schedule=[{"start": 1, "end": 5}])
    cyc = _by_num(expand_config(cfg, control=control))
    for num in (PRETEST, TRAINING, POSTTEST):
        durs = [s["duration"] for s in _odor_steps(cyc[num])]
        assert durs and set(durs) == {5}, (num, durs)


def test_light_past_a_short_odor_is_refused_in_training_mode():
    """The rig runs the light only inside the odor step: the default t=5->30
    inside a 5 s odor would train with no light at all."""
    with pytest.raises(ValueError, match="runs past"):
        expand_config(_v2(odor_duration=5), control=False)


def test_control_mode_has_no_light_to_check():
    cyc = _by_num(expand_config(_v2(odor_duration=5), control=True))
    assert all("light_schedule" not in s for s in cyc[TRAINING]["steps"])


def test_override_leaves_baseline_post_and_light_probe_alone():
    plain = _by_num(expand_config(_v2(), control=True))
    short = _by_num(expand_config(_v2(odor_duration=5), control=True))
    for num in (PRETEST, TRAINING, POSTTEST):
        for name in ("Post-Stimulus Period", "Baseline Period"):
            assert _durations(short[num], name) == _durations(plain[num], name)
    assert short[LIGHT] == plain[LIGHT]
    for num in plain:
        assert short[num]["delay_after"] == plain[num]["delay_after"]


def test_override_applies_to_the_standard_protocol_too():
    cyc = _by_num(expand_config(_v2(pretest=False, odor_duration=5), control=True))
    for num in (1, 2):
        durs = [s["duration"] for s in _odor_steps(cyc[num])]
        assert durs and set(durs) == {5}, (num, durs)


@pytest.mark.parametrize("bad", [0, -5, "five", None])
def test_bad_value_is_refused(bad):
    with pytest.raises(ValueError):
        expand_config(_v2(odor_duration=bad), control=True)


def test_defaults_untouched():
    assert _DEFAULTS["odor_duration"] == 30
    assert _DEFAULTS["testing_odor_duration"] == 30


# ── the real rig configs ──────────────────────────────────────────────

def _load(name):
    return yaml.safe_load((PICODE / name).read_text())


needs_picode = pytest.mark.skipif(not (PICODE / "config_v2_A_pretest.yaml").exists(),
                                  reason="PiCode/ is gitignored; configs absent")


# Configs that deliberately shorten the odor; every other one keeps 30 s.
SHORT_ODOR_CONFIGS = {"config_v2_A_pretest.yaml", "config_v2_A_10s.yaml"}


@needs_picode
def test_a_pretest_config_runs_5_s_odor():
    cyc = _by_num(expand_config(_load("config_v2_A_pretest.yaml"), control=True))
    for num in (PRETEST, TRAINING, POSTTEST):
        durs = [s["duration"] for s in _odor_steps(cyc[num])]
        assert durs and set(durs) == {5}, (num, durs)


@needs_picode
@pytest.mark.parametrize("name", sorted(p.name for p in PICODE.glob("config_v2_*.yaml")
                                        if p.name not in SHORT_ODOR_CONFIGS))
def test_every_other_config_keeps_its_odor_duration(name):
    cfg = _load(name)
    if (cfg or {}).get("format") != "v2" or (cfg.get("experiment") or {}).get("mode"):
        pytest.skip("not a classical v2 protocol")
    assert "odor_duration" not in cfg["experiment"]


@needs_picode
@pytest.mark.parametrize("control", [True, False])
def test_a_pretest_light_probe_is_10_s(control):
    """config_v2_A_pretest.yaml shortens the light-only probe 25 -> 10 s too."""
    cfg = _load("config_v2_A_pretest.yaml")
    if not control:
        cfg["experiment"]["light_schedule"] = [{"start": 0, "end": 5}]
    light = _by_num(expand_config(cfg, control=control))[LIGHT]
    step = next(s for s in light["steps"] if "light_schedule" in s)
    assert step["duration"] == 10
    assert step["light_schedule"] == [{"start": 0, "end": 10}]
    assert _durations(light, "Baseline Period") == [30]
    assert _durations(light, "Post-Stimulus Period") == [30]


# ── the --sensitivity launch route ────────────────────────────────────
# The rigs run `--odors I --control --sensitivity`: the scheduler loads the
# PLAIN config_v2_A.yaml and apply_sensitivity_overrides turns it into the
# pre-test protocol. The October 14 batches ran 30 s odors because that route
# never read config_v2_A_pretest.yaml. It must give the same session.

import json
import random

from expand_config import apply_sensitivity_overrides


def _expanded(cfg, seed=0):
    random.seed(seed)
    return json.dumps(expand_config(cfg, control=True)["cycles"], sort_keys=True)


@needs_picode
def test_sensitivity_flag_on_plain_iaa_config_runs_5_s_odor_and_10_s_probe():
    cyc = _by_num(expand_config(apply_sensitivity_overrides(_load("config_v2_A.yaml")),
                                control=True))
    for num in (PRETEST, TRAINING, POSTTEST):
        durs = [s["duration"] for s in _odor_steps(cyc[num])]
        assert durs and set(durs) == {5}, (num, durs)
    probe = next(s for s in cyc[LIGHT]["steps"] if "light_schedule" in s)
    assert probe["duration"] == 10


@needs_picode
@pytest.mark.parametrize("letter", list("ABCEHLO"))
def test_sensitivity_flag_matches_the_pretest_config(letter):
    via_flag = apply_sensitivity_overrides(_load(f"config_v2_{letter}.yaml"))
    via_file = _load(f"config_v2_{letter}_pretest.yaml")
    assert _expanded(via_flag) == _expanded(via_file)


def test_sensitivity_flag_keeps_keys_the_config_sets():
    cfg = {"format": "v2", "experiment": {"trained_odor": "OFM_A", "odor_duration": 7,
                                          "inter_trial_baseline": 240}}
    exp = apply_sensitivity_overrides(cfg)["experiment"]
    assert exp["odor_duration"] == 7
    assert exp["inter_trial_baseline"] == 240


@needs_picode
def test_5_s_iaa_runs_push_to_their_own_cohort_folder():
    """5 s sessions must never pool with the 30 s IAA-Sensitivity-* data."""
    from experiment_scheduler import get_push_folder
    folder = lambda name: get_push_folder(
        "OFM_A" if "_A" in name else "OFM_H", "Control", "B/", starvation_hours=1,
        odor_vial_conc="1", sensitivity=True,
        cohort=apply_sensitivity_overrides(_load(name))["experiment"].get("cohort"))
    assert folder("config_v2_A.yaml") == "B/IAA-Sensitivity5s-1-1/"
    assert folder("config_v2_H.yaml") == "B/Hex-Sensitivity-1-1/"


def test_pretest_odor_duration_only_changes_the_pretest():
    cyc = _by_num(expand_config(_v2(pretest_odor_duration=10), control=True))
    assert {s["duration"] for s in _odor_steps(cyc[PRETEST])} == {10}
    for num in (TRAINING, POSTTEST):
        assert {s["duration"] for s in _odor_steps(cyc[num])} == {30}, num


def test_pretest_odor_duration_beats_odor_duration():
    cyc = _by_num(expand_config(_v2(odor_duration=5, pretest_odor_duration=10),
                                control=True))
    assert {s["duration"] for s in _odor_steps(cyc[PRETEST])} == {10}
    assert {s["duration"] for s in _odor_steps(cyc[POSTTEST])} == {5}


@needs_picode
def test_a_10s_config_light_on_t4_to_odor_off():
    """config_v2_A_10s.yaml: 10 s odor in training + post-test, light t=4->10."""
    cyc = _by_num(expand_config(_load("config_v2_A_10s.yaml"), control=False))
    train = [s for s in cyc[1]["steps"] if s["name"] == "Odor + Light Pattern"]
    assert train and all(s["duration"] == 10 for s in train)
    assert all(s["light_schedule"] == [{"start": 4, "end": 10}] for s in train)
    test = _odor_steps(cyc[2])
    assert test and {s["duration"] for s in test} == {10}
    assert not any("light_schedule" in s for s in test)
