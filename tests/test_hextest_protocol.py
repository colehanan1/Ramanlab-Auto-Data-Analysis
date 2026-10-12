"""The hex test protocol (config_v2_hextest_H.yaml / config_v2_hextest_L.yaml).

As specified 2026-10-01, pre-test changed 2026-10-02 and again 2026-10-03:

    1. pre-test   ONE trial of ethyl butyrate 10 ("eb 10"), found by odor + conc
                  in ~/odor_pins.txt, whichever valve holds it; no light
                  (was: every non-hex vial; 240 s apart -> 0 s on 2026-10-02)
    2. rest       the normal 4 min gap: 240 s camera-off = 5 min odor-off -> on
                  (was 30 min until 2026-10-03)
    3. training   6 x the trained hex vial, 240 s between trials
    4. rest       30 min (odor-off -> odor-on)
    5. post-test  the trained hex first, then every other vial in random order
    6. light-only probe (testing_9), as in every protocol

Every odor recording is 30 s pre + 30 s odor + 30 s post = 90 s. ``_H`` trains
the hex 0.1 vial, ``_L`` the hex 1 vial (per ~/odor_pins.txt).

It rides on the pre-test cycle numbers (5/6/7/8), so the rig scripts' phase
tags (pretest_N / training_N / testing_N / testing_9) need no change.
"""

import copy
import sys
from pathlib import Path

import pytest
import yaml

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))
import expand_config as ec  # noqa: E402
import experiment_scheduler as sched  # noqa: E402
from expand_config import expand_config, apply_sensitivity_overrides  # noqa: E402
from experiment_scheduler import get_push_folder, get_odor_from_config  # noqa: E402
from odor_pins import apply_pin_map, parse_pin_text  # noqa: E402

PRETEST, TRAINING, POSTTEST, LIGHT = 5, 6, 7, 8

PINS = """\
H  hex   0.1
L  hex   1
B  benz  1
E  eb    1
C  eb    10
A  iaa   1
O  iaa   0.1
"""
NON_HEX = {"Benzaldehyde-1", "EthylButyrate-1", "EthylButyrate-10",
           "IsoamylAcetate-1", "IsoamylAcetate-0.1"}
ALL_SEVEN = NON_HEX | {"Hexanol-0.1", "Hexanol-1"}

CONFIGS = {
    "config_v2_hextest_H.yaml": ("OFM_H", "Hexanol-0.1", "Hexanol-1"),
    "config_v2_hextest_L.yaml": ("OFM_L", "Hexanol-1", "Hexanol-0.1"),
}


@pytest.fixture(autouse=True)
def _pins():
    display = dict(ec._PIN_DISPLAY)
    pins = list(ec._ALL_ODOR_PINS)
    cmap = copy.deepcopy(sched.ODOR_CONFIG_MAP)
    applied = ec._PINS_APPLIED
    apply_pin_map(parse_pin_text(PINS))
    yield
    ec._PIN_DISPLAY.clear()
    ec._PIN_DISPLAY.update(display)
    ec._ALL_ODOR_PINS[:] = pins
    sched.ODOR_CONFIG_MAP.clear()
    sched.ODOR_CONFIG_MAP.update(cmap)
    ec._PINS_APPLIED = applied


def _load(name):
    with open(PICODE / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _cycles(name, control=False):
    return {c["cycle"]: c for c in expand_config(_load(name), control=control)["cycles"]}


def _odors(cycle):
    return [s["odor_label"] for s in cycle["steps"] if s["name"] == "Start Recording"]


def _recordings(cycle):
    """Seconds between each Start Recording and its Stop Recording."""
    out, t0, t = [], None, 0
    for s in cycle["steps"]:
        if s["name"] == "Start Recording":
            t0 = t
        t += s.get("duration", 0)
        if s["name"] == "Stop Recording":
            out.append(t - t0)
    return out


def _gaps(cycle):
    """Dark seconds between each Stop Recording and the next Start Recording."""
    out, t_stop, t = [], None, 0
    for s in cycle["steps"]:
        if s["name"] == "Start Recording" and t_stop is not None:
            out.append(t - t_stop)
        t += s.get("duration", 0)
        if s["name"] == "Stop Recording":
            t_stop = t
    return out


def _trailing_dark(cycle):
    i = max(i for i, s in enumerate(cycle["steps"]) if s["name"] == "Stop Recording")
    return sum(s.get("duration", 0) for s in cycle["steps"][i + 1:]) + cycle["delay_after"]


def _odor_off_to_odor_on(a, b):
    """Last odor OFF in cycle a -> first odor ON in cycle b (post + dark + pre)."""
    return 30 + _trailing_dark(a) + 30


# ── the files themselves ──────────────────────────────────────────────

@pytest.mark.parametrize("name", CONFIGS)
def test_config_file_names_its_trained_valve(name):
    """The rigs take the trained pin from the filename's last letter."""
    assert get_odor_from_config(PICODE / name) == CONFIGS[name][0]
    assert _load(name)["experiment"]["trained_odor"] == CONFIGS[name][0]


@pytest.mark.parametrize("name", CONFIGS)
def test_cycles_are_the_pretest_numbers(name):
    assert sorted(_cycles(name)) == [PRETEST, TRAINING, POSTTEST, LIGHT]


# ── 1. pre-test ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", CONFIGS)
def test_pretest_is_one_trial_of_eb_10(name):
    cyc = _cycles(name)[PRETEST]
    assert _odors(cyc) == ["EthylButyrate-10"]
    acts = [a for s in cyc["steps"] for a in s.get("actions", [])]
    assert acts == [{"device": "ofm", "pin": "OFM_C", "state": "on"}]


@pytest.mark.parametrize("name", CONFIGS)
def test_pretest_follows_eb_10_to_whatever_valve_holds_it(name):
    """Selected by odor + conc, not valve letter: a vial swap must not change it."""
    apply_pin_map(parse_pin_text(PINS.replace("C  eb    10", "C  iaa   10")
                                     .replace("O  iaa   0.1", "O  eb    10")))
    cyc = _cycles(name)[PRETEST]
    assert _odors(cyc) == ["EthylButyrate-10"]
    acts = [a for s in cyc["steps"] for a in s.get("actions", [])]
    assert acts == [{"device": "ofm", "pin": "OFM_O", "state": "on"}]


@pytest.mark.parametrize("name", CONFIGS)
def test_pretest_refuses_when_no_valve_holds_eb_10(name):
    apply_pin_map(parse_pin_text(PINS.replace("C  eb    10", "C  eb    0.1")))
    with pytest.raises(ValueError, match="eb 10"):
        expand_config(_load(name))


@pytest.mark.parametrize("name", CONFIGS)
def test_pretest_never_has_light(name):
    for step in _cycles(name, control=False)[PRETEST]["steps"]:
        assert "light_schedule" not in step


# ── 2. training ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", CONFIGS)
def test_training_is_six_trials_of_the_trained_hex_vial(name):
    train = _cycles(name)[TRAINING]
    assert train["repeat"] == 6
    assert _odors(train) == [CONFIGS[name][1]]
    pin = CONFIGS[name][0]
    acts = [a for s in train["steps"] for a in s.get("actions", [])]
    assert acts == [{"device": "ofm", "pin": pin, "state": "on"}]


@pytest.mark.parametrize("name", CONFIGS)
def test_training_has_240_s_between_trials(name):
    """Unlike the sensitivity protocol (0 s), every trial here is 240 s apart."""
    steps = _cycles(name)[TRAINING]["steps"]
    assert steps[-1] == {"name": "Baseline Period", "duration": 240}


@pytest.mark.parametrize("name", CONFIGS)
def test_training_light_follows_the_mode_flag(name):
    assert any("light_schedule" in s for s in _cycles(name, control=False)[TRAINING]["steps"])
    assert not any("light_schedule" in s for s in _cycles(name, control=True)[TRAINING]["steps"])


# ── 3. post-test ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name", CONFIGS)
def test_posttest_is_trained_hex_first_then_all_others_once(name):
    odors = _odors(_cycles(name)[POSTTEST])
    assert len(odors) == 7
    assert odors[0] == CONFIGS[name][1]
    assert set(odors[1:]) == ALL_SEVEN - {CONFIGS[name][1]}
    assert CONFIGS[name][2] in odors[1:]       # the other hex vial is tested


def test_posttest_order_after_the_first_is_random():
    orders = {tuple(_odors(_cycles("config_v2_hextest_L.yaml")[POSTTEST])[1:])
              for _ in range(40)}
    assert len(orders) > 1


@pytest.mark.parametrize("name", CONFIGS)
def test_posttest_never_has_light(name):
    for step in _cycles(name, control=False)[POSTTEST]["steps"]:
        assert "light_schedule" not in step


# ── timing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", CONFIGS)
def test_posttest_recordings_are_90_s_and_240_s_apart(name):
    cyc = _cycles(name)[POSTTEST]
    assert set(_recordings(cyc)) == {90}
    assert set(_gaps(cyc)) == {240}


@pytest.mark.parametrize("name", CONFIGS)
def test_pretest_is_a_single_90_s_recording(name):
    cyc = _cycles(name)[PRETEST]
    assert _recordings(cyc) == [90]
    assert _gaps(cyc) == []


@pytest.mark.parametrize("name", CONFIGS)
def test_zero_gap_drops_the_gap_steps(name):
    """No 0-duration filler steps: the pre-test ends on its last Stop Recording,
    so the whole rest is delay_after (the rig's countdown covers all of it)."""
    steps = _cycles(name)[PRETEST]["steps"]
    # Each trial still has its own 30 s pre-odor "Baseline Period"; what must
    # be gone is any step between one Stop Recording and the next Start.
    after_stop = [steps[i + 1]["name"] for i, s in enumerate(steps[:-1])
                  if s["name"] == "Stop Recording"]
    assert after_stop == ["Start Recording"] * (len(_recordings(_cycles(name)[PRETEST])) - 1)
    assert steps[-1]["name"] == "Stop Recording"


@pytest.mark.parametrize("name", CONFIGS)
def test_pretest_to_training_is_the_normal_4_min_gap(name):
    """User 2026-10-03: "wait only 4 minutes between trials from pre test to
    start of training" -- camera off 240 s, like every other inter-trial gap."""
    cyc = _cycles(name)
    assert cyc[PRETEST]["delay_after"] == 240
    assert _trailing_dark(cyc[PRETEST]) == 240


def test_sensitivity_pretest_keeps_240_s_gaps():
    """Only the hex test changed; sensitivity panels must stay poolable."""
    cyc = _cycles("config_v2_H_pretest.yaml")
    assert set(_gaps(cyc[PRETEST])) == {240}


@pytest.mark.parametrize("name", CONFIGS)
def test_training_recordings_are_90_s(name):
    assert _recordings(_cycles(name)[TRAINING]) == [90]


@pytest.mark.parametrize("name", CONFIGS)
def test_pretest_to_training_is_5_min_odor_off_to_on(name):
    cyc = _cycles(name)
    assert _odor_off_to_odor_on(cyc[PRETEST], cyc[TRAINING]) == 300


@pytest.mark.parametrize("name", CONFIGS)
def test_training_to_posttest_rest_is_30_min_odor_off_to_on(name):
    """Unchanged by the 2026-10-03 pre-test rework."""
    cyc = _cycles(name)
    assert _odor_off_to_odor_on(cyc[TRAINING], cyc[POSTTEST]) == 1800


@pytest.mark.parametrize("name", CONFIGS)
def test_light_probe_closes_the_session(name):
    light = _cycles(name)[LIGHT]
    assert _odors(light) == ["LightOnly"]


def _session_seconds(name):
    cyc = _cycles(name)
    total = 0
    for n in sorted(cyc):
        c = cyc[n]
        total += sum(s.get("duration", 0) for s in c["steps"]) * c["repeat"]
        total += c["delay_after"]
    return total


@pytest.mark.parametrize("name", CONFIGS)
def test_session_length(name):
    # pre-test 1x90 + 240; training 6x(90+240) + 1500;
    # post-test 7x90 + 6x240; light probe 85
    assert _session_seconds(name) == 330 + 3480 + 2070 + 85 == 5965


# ── safety checks ─────────────────────────────────────────────────────

def test_refuses_to_train_a_valve_that_is_not_hexanol():
    """With the built-in labels valve L is linalool; a hex test must not run."""
    ec._PIN_DISPLAY.update({"OFM_L": "Linalool"})
    sched.ODOR_CONFIG_MAP["OFM_L"]["folder"] = "Lin"
    with pytest.raises(ValueError, match="Hex"):
        expand_config(_load("config_v2_hextest_L.yaml"))


def test_sensitivity_flag_does_not_change_the_hex_test():
    """--sensitivity only fills keys the config leaves unset."""
    cfg = _load("config_v2_hextest_H.yaml")
    assert apply_sensitivity_overrides(cfg)["experiment"] == cfg["experiment"]


def test_existing_pretest_configs_keep_their_panels():
    """Adding pretest_sequence must not change the sensitivity protocol."""
    cyc = _cycles("config_v2_H_pretest.yaml")
    assert len(_odors(cyc[PRETEST])) == 7 and len(_odors(cyc[POSTTEST])) == 7
    assert cyc[TRAINING]["steps"][-1]["name"] == "Stop Recording"   # 0 s ITI


# ── cohort folder ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name,mode,expected", [
    ("config_v2_hextest_L.yaml", "Training", "Hex-HexTest-Training-24-1"),
    ("config_v2_hextest_L.yaml", "Control", "Hex-HexTest-24-1"),
    ("config_v2_hextest_H.yaml", "Training", "Hex-HexTest-Training-24-0.1"),
    ("config_v2_hextest_H.yaml", "Control", "Hex-HexTest-24-0.1"),
])
def test_push_folder(name, mode, expected):
    exp = _load(name)["experiment"]
    conc = parse_pin_text(PINS)[exp["trained_odor"]].conc
    got = get_push_folder(exp["trained_odor"], mode, "u@h:/base", starvation_hours=24,
                          odor_vial_conc=conc, sensitivity=bool(exp["pretest"]),
                          cohort=exp["cohort"])
    assert got == f"u@h:/base/{expected}/"


def test_sensitivity_folder_unchanged_without_a_cohort():
    got = get_push_folder("OFM_H", "Training", "/b", starvation_hours=24,
                          odor_vial_conc="0.1", sensitivity=True)
    assert got == "/b/Hex-Sensitivity-Training-24-0.1/"


@pytest.mark.parametrize("bad", ["Hex-Test", "Hex Test", "Hex_Test", ""])
def test_cohort_must_be_one_folder_segment(bad):
    with pytest.raises(ValueError):
        get_push_folder("OFM_H", "Training", "/b", cohort=bad, sensitivity=True)


# ── review fixes 2026-10-01 ───────────────────────────────────────────

@pytest.mark.parametrize("name", CONFIGS)
def test_refuses_to_run_without_a_pins_file(name):
    """With the built-in labels valve L reads 'Linalool' in the pre-test."""
    ec._PINS_APPLIED = False
    with pytest.raises(ValueError, match="odor_pins"):
        expand_config(_load(name))


def _ntfy(cycle):
    return [s for s in cycle["steps"] if s["name"] == "Ntfy Notification"]


@pytest.mark.parametrize("name", CONFIGS)
def test_mount_next_flies_only_in_the_post_test(name):
    """Pre-test trial 4 is ~2 h before the end: far too early to mount."""
    cyc = _cycles(name)
    assert _ntfy(cyc[PRETEST]) == []
    assert len(_ntfy(cyc[POSTTEST])) == 1
