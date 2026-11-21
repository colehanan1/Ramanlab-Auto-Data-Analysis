#!/usr/bin/env python3
"""Tests for expand_config ``mode: odor_pair`` (IAA-pair protocols)."""

import random
import sys, os
from pathlib import Path

import pytest
import yaml

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))

import expand_config as ec
from expand_config import expand_config, _pair_order, _odor_label, _ALL_ODOR_PINS

ANCHOR = "OFM_A"
VIALS = ["OFM_A", "OFM_B", "OFM_C", "OFM_E", "OFM_H", "OFM_L", "OFM_O"]


@pytest.fixture(autouse=True)
def _default_pins():
    saved = list(ec._ALL_ODOR_PINS)
    ec._ALL_ODOR_PINS[:] = VIALS
    yield
    ec._ALL_ODOR_PINS[:] = saved


def _cfg(order="iaa_first", **exp):
    e = {"trained_odor": ANCHOR, "mode": "odor_pair", "pair_order": order,
         "odor_duration": 10, "cohort": "PairFirst"}
    e.update(exp)
    return {"format": "v2", "experiment": e}


def _trials(cycle, light=False):
    """Split a cycle's steps into trials: list of dicts with start index, steps,
    and the gap step that follows (or None). The trailing light-only probe
    (odor_label LightOnly) is left out unless ``light=True``."""
    steps = cycle["steps"]
    out, i = [], 0
    while i < len(steps):
        assert steps[i]["name"] == "Start Recording", steps[i]
        j = i
        while steps[j]["name"] != "Stop Recording":
            j += 1
        gap = None
        if j + 1 < len(steps) and steps[j + 1]["name"] != "Start Recording":
            gap = steps[j + 1]
            assert gap["name"] == "Inter-Trial Gap"
        out.append({"steps": steps[i:j + 1], "gap": gap, "start": i})
        i = j + (2 if gap else 1)
    if not light:
        out = [t for t in out if t["steps"][0].get("odor_label") != "LightOnly"]
    return out


def _probe(cycle):
    """The light-only trial(s) of a cycle."""
    return [t for t in _trials(cycle, light=True)
            if t["steps"][0].get("odor_label") == "LightOnly"]


def _odor_steps(trial):
    return [s for s in trial["steps"]
            if any(a.get("device") == "ofm" for a in s.get("actions", []))]


def _clip_len(trial):
    return sum(s["duration"] for s in trial["steps"])


def _onset_offset(trial):
    """Seconds from Start Recording to the first odor step."""
    t = 0
    for s in trial["steps"]:
        if any(a.get("device") == "ofm" for a in s.get("actions", [])):
            return t
        t += s["duration"]
    raise AssertionError("no odor step")


def _expand(order="iaa_first", **exp):
    r = expand_config(_cfg(order, **exp), control=True)
    cyc = {c["cycle"]: c for c in r["cycles"]}
    return r, cyc[5], cyc[7]


def _pins(trial):
    return [s["actions"][0]["pin"] for s in _odor_steps(trial)]


# ── structure ─────────────────────────────────────────────────────────

def test_cycle_numbers_and_delay_after_zero():
    r, pre, pair = _expand(seed=1)
    assert [c["cycle"] for c in r["cycles"]] == [5, 7]
    assert all(c["delay_after"] == 0 and c["repeat"] == 1 for c in r["cycles"])


def test_counts_and_paraffin_first():
    _, pre, pair = _expand(seed=1)
    pre_t, pair_t = _trials(pre), _trials(pair)
    assert len(pre_t) == 8 and len(pair_t) == 16
    assert _pins(pre_t[0]) == ["OFM_P"]
    assert sorted(_pins(t)[0] for t in pre_t[1:]) == sorted(VIALS)
    assert all(len(_pins(t)) == 1 for t in pre_t)


def test_pre_block_clip_is_70s_with_10s_odor_no_light():
    _, pre, _ = _expand(seed=2)
    for t in _trials(pre):
        names = [s["name"] for s in t["steps"]]
        assert names[0] == "Start Recording" and names[-1] == "Stop Recording"
        assert _clip_len(t) == 70
        assert _onset_offset(t) == 30
        assert _odor_steps(t)[0]["duration"] == 10


def test_pair_clip_is_81s_with_1s_empty_gap():
    _, _, pair = _expand(seed=2)
    for t in _trials(pair):
        assert _clip_len(t) == 81
        assert _onset_offset(t) == 30
        odor = _odor_steps(t)
        assert len(odor) == 2 and all(s["duration"] == 10 for s in odor)
        a, b = t["steps"].index(odor[0]), t["steps"].index(odor[1])
        assert b == a + 2
        mid = t["steps"][a + 1]
        assert mid["duration"] == 1 and not mid.get("actions")
        assert t["steps"][1]["duration"] == 30 and t["steps"][-2]["duration"] == 30


def test_gap_durations():
    _, pre, pair = _expand(seed=3)
    pre_t, pair_t = _trials(pre), _trials(pair)
    assert [t["gap"]["duration"] for t in pre_t[:-1]] == [230] * 7
    assert pre_t[-1]["gap"]["duration"] == 1730
    assert [t["gap"]["duration"] for t in pair_t[:-1]] == [219] * 15


def test_no_trailing_gap_and_session_ends_with_stop():
    _, _, pair = _expand(seed=3)
    assert pair["steps"][-1]["name"] == "Stop Recording"
    # user 2026-11-17: no wait before the light probe -- test 16 runs straight into it
    assert _trials(pair)[-1]["gap"] is None
    assert _trials(pair, light=True)[-1]["gap"] is None
    assert _trials(pair, light=True)[-1]["steps"][0]["odor_label"] == "LightOnly"


def test_every_gap_flagged_continuous_gap_and_only_those():
    r, _, _ = _expand(seed=3)
    for c in r["cycles"]:
        for s in c["steps"]:
            if s["name"] == "Inter-Trial Gap":
                assert s["continuous_gap"] is True
            else:
                assert not s.get("continuous_gap")
        # camera-off time lives only in gap steps: every non-gap step sits in
        # a Start..Stop window
        rec = False
        for s in c["steps"]:
            if s["name"] == "Start Recording":
                rec = True
            elif s["name"] == "Stop Recording":
                rec = False
            elif s["name"] != "Inter-Trial Gap":
                assert rec, s


def test_onset_to_onset_300_and_1800_by_summing():
    _, pre, pair = _expand(seed=4)
    allsteps = pre["steps"] + pair["steps"]
    onsets = []
    # first odor of each trial
    t, in_trial_first = 0.0, True
    for s in allsteps:
        if s["name"] == "Start Recording":
            in_trial_first = True
        if s.get("actions") and in_trial_first:
            onsets.append(t)
            in_trial_first = False
        t += s["duration"]
    assert len(onsets) == 24
    d = [b - a for a, b in zip(onsets, onsets[1:])]
    assert d[:7] == [300] * 7
    assert d[7] == 1800
    assert d[8:] == [300] * 15


def test_light_probe_starts_right_after_test_16():
    _, pre, pair = _expand(seed=4)
    last = _trials(pair)[-1]
    stop = last["start"] + len(last["steps"]) - 1
    assert pair["steps"][stop]["name"] == "Stop Recording"
    assert pair["steps"][stop + 1] == {"name": "Start Recording", "duration": 0,
                                       "odor_label": "LightOnly"}
    t, last_odor_onset, light_onset, first = 0.0, None, None, True
    for s in pre["steps"] + pair["steps"]:
        if s["name"] == "Start Recording":
            first = True
        if s.get("actions") and first:
            last_odor_onset, first = t, False
        if s.get("light_schedule") is not None:
            light_onset = t
        t += s["duration"]
    # rest of test 16 (81 - 30 s) + the probe's 10 s baseline
    assert light_onset - last_odor_onset == 61


def test_session_span():
    _, pre, pair = _expand(seed=4)
    total = sum(s["duration"] for s in pre["steps"] + pair["steps"])
    # 30 s before the first onset ... last pair onset at 8400 s, + 51 s, then
    # the 30 s light probe (10 s + 10 s light + 10 s) with no gap
    assert total == 30 + 8400 + 51 + 30


# ── ordering ──────────────────────────────────────────────────────────

def test_pair_order_rounds_rule_200_seeds():
    partners = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for seed in range(200):
        order = _pair_order(partners, 2, random.Random(seed))
        assert len(order) == 16
        assert sorted(order[:8]) == sorted(partners)
        assert sorted(order[8:]) == sorted(partners)
        assert order[8] != order[7]


def test_pair_order_deterministic_and_not_mutating():
    partners = list("abcdefgh")
    a = _pair_order(partners, 2, random.Random(5))
    b = _pair_order(partners, 2, random.Random(5))
    assert a == b and partners == list("abcdefgh")


def test_partner_set_has_p_and_iaa_twice():
    _, _, pair = _expand(seed=6)
    other = []
    for t in _trials(pair):
        p = _pins(t)
        other.append(p[1] if p[0] == ANCHOR else p[0])
    assert sorted(other) == sorted([x for x in VIALS if x != ANCHOR] * 2
                                   + ["OFM_P"] * 2 + [ANCHOR] * 2)


def test_iaa_first_slot_placement_and_aa_present():
    _, _, pair = _expand("iaa_first", seed=7)
    ps = [_pins(t) for t in _trials(pair)]
    assert all(p[0] == ANCHOR for p in ps)
    assert [p for p in ps if p[1] == ANCHOR] == [[ANCHOR, ANCHOR]] * 2


def test_iaa_second_slot_placement_and_aa_present():
    _, _, pair = _expand("iaa_second", seed=7)
    ps = [_pins(t) for t in _trials(pair)]
    assert all(p[1] == ANCHOR for p in ps)
    assert [p for p in ps if p[0] == ANCHOR] == [[ANCHOR, ANCHOR]] * 2


def test_bad_pair_order_rejected():
    with pytest.raises(ValueError):
        expand_config(_cfg("sideways"), control=True)


def test_missing_vial_fails_loudly():
    ec._ALL_ODOR_PINS[:] = [p for p in VIALS if p != "OFM_C"]
    with pytest.raises(ValueError):
        expand_config(_cfg(seed=1), control=True)


# ── labels, light, tags ───────────────────────────────────────────────

def test_labels_in_delivery_order():
    for order in ("iaa_first", "iaa_second"):
        _, _, pair = _expand(order, seed=8)
        for t in _trials(pair):
            p = _pins(t)
            want = f"{_odor_label(p[0])}_then_{_odor_label(p[1])}"
            assert t["steps"][0]["odor_label"] == want


def test_pre_labels_single_odor():
    _, pre, _ = _expand(seed=8)
    for t in _trials(pre):
        assert t["steps"][0]["odor_label"] == _odor_label(_pins(t)[0])
    assert _trials(pre)[0]["steps"][0]["odor_label"] == "ParaffinOil"


def _has_light(s):
    return (any(k.startswith("light") for k in s)
            or any(a.get("device") == "light" for a in s.get("actions", [])))


def test_light_only_in_the_final_probe():
    r, pre, pair = _expand(seed=9)
    for c in r["cycles"]:
        for t in _trials(c):
            assert not any(_has_light(s) for s in t["steps"]), t
    assert not any(_has_light(s) for s in pre["steps"])
    lit = [s for s in pair["steps"] if _has_light(s)]
    assert len(lit) == 1 and lit[0] in _probe(pair)[0]["steps"]


def test_light_probe_is_last_recording_of_cycle_7():
    # one recording in the pair cycle -> the rig tags it testing_<n_pairs+1>_LightOnly
    # (cycle 8 would tag it testing_9 and collide with pair 9)
    r, pre, pair = _expand(seed=9)
    assert [c["cycle"] for c in r["cycles"]] == [5, 7]
    allt = _trials(pair, light=True)
    assert len(allt) == 17 and len(_probe(pair)) == 1
    assert allt[-1] is not None and allt[-1]["steps"][0]["odor_label"] == "LightOnly"
    assert not _probe(pre)


def test_light_probe_shape_10_10_10_solid_no_odor():
    _, _, pair = _expand(seed=9)
    t = _probe(pair)[0]
    assert [s["name"] for s in t["steps"]] == [
        "Start Recording", "Baseline Period", "Testing Light Only (No Odor)",
        "Post-Stimulus Period", "Stop Recording"]
    # user 2026-11-17: "10s pre 10s light 10 post"
    assert _clip_len(t) == 30
    light = t["steps"][2]
    assert light["duration"] == 10 and light["actions"] == []
    assert light["light_schedule"] == [{"start": 0, "end": 10}]
    assert light["light_hertz"] == 1.0 and light["light_duty"] == 100.0
    assert t["steps"][1]["duration"] == 10 and t["steps"][3]["duration"] == 10
    assert not _odor_steps(t)


def test_light_probe_duration_override_and_opt_out():
    _, _, pair = _expand(seed=9, light_only={"duration": 25})
    assert _probe(pair)[0]["steps"][2]["duration"] == 25
    _, _, pair = _expand(seed=9, light_only=False)
    assert not _probe(pair)
    assert pair["steps"][-1]["name"] == "Stop Recording"
    assert _trials(pair)[-1]["gap"] is None
    assert not any(_has_light(s) for s in pair["steps"])


def test_light_probe_fires_in_control_mode():
    r = expand_config(_cfg(seed=9), control=True)
    pair = {c["cycle"]: c for c in r["cycles"]}[7]
    assert _probe(pair)


def test_tags_pretest_testing():
    # the rig derives tags from the cycle number: 5 -> pretest_N, 7 -> testing_N
    r, pre, pair = _expand(seed=9)
    assert pre["cycle"] == ec._PRETEST_CYCLE and pair["cycle"] == ec._PRETEST_POST_CYCLE


# ── continuous passthrough, overrides ─────────────────────────────────

def test_continuous_absent_by_default():
    r, _, _ = _expand(seed=1)
    assert "continuous" not in r


def test_continuous_defaults_filled():
    r, _, _ = _expand(seed=1, continuous={"fps": 20})
    assert r["continuous"] == {"fps": 20, "bitrate": 1000000,
                               "guard_s": 2, "min_free_gb": 2}
    r, _, _ = _expand(seed=1, continuous={"fps": 25, "bitrate": 2000000,
                                          "guard_s": 3, "min_free_gb": 5})
    assert r["continuous"] == {"fps": 25, "bitrate": 2000000,
                               "guard_s": 3, "min_free_gb": 5}


def test_smoke_overrides():
    r, pre, pair = _expand(seed=1, max_pretest=2, max_pairs=2,
                           onset_spacing_s=120, rest_onset_to_onset_s=240)
    pre_t, pair_t = _trials(pre), _trials(pair)
    assert len(pre_t) == 2 and len(pair_t) == 2
    assert _pins(pre_t[0]) == ["OFM_P"]
    assert pre_t[0]["gap"]["duration"] == 120 - 70
    assert pre_t[1]["gap"]["duration"] == 240 - 70
    assert pair_t[0]["gap"]["duration"] == 120 - 81
    assert pair_t[1]["gap"] is None                      # straight into the probe
    assert len(_probe(pair)) == 1


def test_spacing_shorter_than_clip_rejected():
    with pytest.raises(ValueError):
        expand_config(_cfg(seed=1, onset_spacing_s=60), control=True)


# ── the real yaml files ───────────────────────────────────────────────

CONFIGS = {
    "config_v2_pairfirst_A.yaml": ("PairFirst", "iaa_first"),
    "config_v2_pairsecond_A.yaml": ("PairSecond", "iaa_second"),
    "config_v2_pairfirst_test_A.yaml": ("PairFirstTest", "iaa_first"),
    "config_v2_pairsecond_test_A.yaml": ("PairSecondTest", "iaa_second"),
    "config_v2_pair_smoke_A.yaml": ("PairSmoke", "iaa_first"),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_real_configs(name):
    from experiment_scheduler import get_odor_from_config, get_push_folder
    cohort, order = CONFIGS[name]
    path = PICODE / name
    cfg = yaml.safe_load(path.read_text())
    exp = cfg["experiment"]
    assert cfg["format"] == "v2" and exp["mode"] == "odor_pair"
    assert exp["trained_odor"] == "OFM_A" and exp["odor_duration"] == 10
    assert exp["cohort"] == cohort and exp["pair_order"] == order
    assert exp["continuous"]["fps"] == 20
    # the rig derives the odor pin from the filename; it must come out as A
    assert get_odor_from_config(path) == "OFM_A"
    folder = get_push_folder("OFM_A", "Control", "host:/base", starvation_hours=24,
                             odor_vial_conc="1", sensitivity=False, cohort=cohort)
    assert f"IAA-{cohort}-Control-24-1" in folder
    r = expand_config(cfg, control=True)
    assert r["continuous"]["bitrate"] == 1000000
    cyc = {c["cycle"]: c for c in r["cycles"]}
    n_pre, n_pair = len(_trials(cyc[5])), len(_trials(cyc[7]))
    probe = _probe(cyc[7])
    assert len(probe) == 1 and probe[0]["steps"][2]["duration"] == 10
    if name == "config_v2_pair_smoke_A.yaml":
        assert (n_pre, n_pair) == (2, 2)
        assert _trials(cyc[5])[0]["gap"]["duration"] == 50
        assert _trials(cyc[5])[1]["gap"]["duration"] == 170
        assert _trials(cyc[7])[0]["gap"]["duration"] == 39
    else:
        assert (n_pre, n_pair) == (8, 16)
        assert _trials(cyc[5])[-1]["gap"]["duration"] == 1730
