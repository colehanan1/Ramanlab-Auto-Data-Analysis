"""P = paraffin oil: the master valve (OFM_P) alone, no odor vial ever opens.

User spec 2026-11-02: a pre-test / train / post-test session like H and O, but
the 6 training trials deliver paraffin oil.

Polarity (user 2026-11-10, "FOR PAR OIL OFF IS WHEN VALVE IS OPEN"): the master
OFF (de-energized) IS the paraffin-oil line; energizing it switches the air to
the odor manifold. So a P step leaves the master OFF -- paraffin keeps flowing
-- and only clicks it ON for 50 ms at onset and again at offset, so the valve
sound mimics a real odor trial. The 2026-11-02 code held it ON for the whole
30 s (ON/OFF/ON ... OFF/ON/OFF), i.e. it shut paraffin off for the window.
"""

import ast
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))

import experiment_scheduler as sched  # noqa: E402
from expand_config import (  # noqa: E402
    _ALL_ODOR_PINS, apply_sensitivity_overrides, expand_config)
import odor_pins  # noqa: E402

SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]
# Point the GPIO checks at another copy of the rig scripts (e.g. files pulled
# from a Pi before pushing a patch) with PARAFFIN_RIG_DIR=<dir>.
RIG_DIR = Path(os.environ.get("PARAFFIN_RIG_DIR", PICODE))
BASE = "ramanlab@host:/data/flys_New"


def _load(name):
    return yaml.safe_load((PICODE / name).read_text(encoding="utf-8"))


# ── scheduler / push folder ──────────────────────────────────────────

@pytest.mark.parametrize("key", ["P", "p", "OFM_P", "PARAFFIN"])
def test_letter_resolves_to_master_pin(key):
    assert sched.resolve_odor_pin(key) == "OFM_P"


def test_paraffin_stays_out_of_the_unfiltered_round_robin():
    assert "OFM_P" not in sched.ALL_ODORS
    assert len(sched.ALL_CONDITIONS) == 14


def test_odors_filter_can_pick_paraffin():
    assert sched.get_config_path("OFM_P", PICODE).name == "config_v2_P.yaml"


def test_paraffin_is_always_100_percent():
    # user 2026-11-04: "P odor conc is always 100%" -- whatever --odor-vial-conc says
    for arg in ("E=1,H=0.1,O=0.1,I=1", "10", None, "P=5,H=0.1"):
        assert sched.resolve_odor_vial_conc(arg, "OFM_P") == "100"
    folder = sched.get_push_folder("OFM_P", "Control", BASE, starvation_hours=1,
                                   odor_vial_conc="100", sensitivity=True)
    assert folder == f"{BASE}/Paraffin-Sensitivity-1-100/"


def test_config_names_its_own_pin():
    assert sched.get_odor_from_config(PICODE / "config_v2_P.yaml") == "OFM_P"
    assert sched.get_odor_from_config(PICODE / "config_v2_P_pretest.yaml") == "OFM_P"


# ── odor_pins.txt ────────────────────────────────────────────────────

def test_pins_file_does_not_need_a_paraffin_line():
    pin_map = odor_pins.parse_pin_text("H hex 0.1\nO 3oct 0.1\nE eb 1\n")
    conc_arg, warning = odor_pins.vial_conc_arg(pin_map, "E=1,H=0.1", "OFM_P")
    assert warning is None
    assert sched.resolve_odor_vial_conc(conc_arg, "OFM_P") == "100"


def test_pins_file_labels_paraffin_with_its_100_percent(monkeypatch):
    import expand_config
    for pin, label in dict(expand_config._PIN_DISPLAY).items():
        monkeypatch.setitem(expand_config._PIN_DISPLAY, pin, label)
    monkeypatch.setattr(expand_config, "_ALL_ODOR_PINS", list(expand_config._ALL_ODOR_PINS))
    monkeypatch.setattr(expand_config, "_PINS_APPLIED", False)
    folders = {p: dict(v) for p, v in sched.ODOR_CONFIG_MAP.items()}
    for p, v in folders.items():
        monkeypatch.setitem(sched.ODOR_CONFIG_MAP, p, v)
    odor_pins.apply_pin_map(odor_pins.parse_pin_text("H hex 0.1\nO 3oct 0.1\n"))
    assert expand_config._PIN_DISPLAY["OFM_H"] == "Hexanol-0.1"
    assert expand_config._PIN_DISPLAY["OFM_P"] == "ParaffinOil-100"
    assert "OFM_P" not in expand_config._ALL_ODOR_PINS


def test_pins_file_rejects_master_as_a_vial():
    with pytest.raises(ValueError, match="Unknown valve"):
        odor_pins.parse_pin_text("P hex 0.1\n")


# ── protocol ─────────────────────────────────────────────────────────

def _expanded(odor_letter):
    """Replay the rig launch route: --odors X --control --sensitivity."""
    cfg = apply_sensitivity_overrides(_load(f"config_v2_{odor_letter}.yaml"))
    return {c["cycle"]: c for c in expand_config(cfg, control=True)["cycles"]}


def _odor_steps(cycle):
    return [s for s in cycle["steps"] if s.get("actions")]


def test_pretest_file_matches_hexanol_except_the_trained_odor():
    p, h = _load("config_v2_P_pretest.yaml"), _load("config_v2_H_pretest.yaml")
    assert p["experiment"].pop("trained_odor") == "OFM_P"
    h["experiment"].pop("trained_odor")
    assert p == h


def test_training_is_six_paraffin_trials_no_vial():
    cyc = _expanded("P")
    train = cyc[6]
    assert train["repeat"] == 6
    (step,) = _odor_steps(train)
    assert step["actions"] == [{"device": "ofm", "pin": "OFM_P", "state": "on"}]
    assert step["odor_label"] == "ParaffinOil"
    assert "light_schedule" not in step          # --control: no light


def _panel(cycle):
    return [s["actions"][0]["pin"] for s in _odor_steps(cycle)]


# user 2026-11-04: every sensitivity session gets a paraffin trial (the master
# click) in both panels -- FIRST in the pre-test, LAST in the post-test -- so the
# 7 vial trials keep the order/timing of the data already collected.
@pytest.mark.parametrize("letter", list("ABCEHLOP"))
def test_panels_bookend_with_paraffin(letter):
    cyc = _expanded(letter)
    pre, post = _panel(cyc[5]), _panel(cyc[7])
    assert pre[0] == "OFM_P" and post[-1] == "OFM_P"
    for vials in (pre[1:], post[:-1]):
        assert sorted(vials) == sorted(_ALL_ODOR_PINS)      # the 7 vials, once each
    assert pre.count("OFM_P") == post.count("OFM_P") == 1
    assert _load(f"config_v2_{letter}_pretest.yaml")["experiment"]["paraffin_bookends"] is True


@pytest.mark.parametrize("letter", list("AHO"))
def test_bookend_trial_runs_like_the_panel_trials(letter):
    """Same baseline/odor/post as the vial trials (A's 5 s odor included), and
    the 7 vial trials keep the exact step timings they had without it."""
    cyc = _expanded(letter)
    for n in (5, 7):
        trials = [s for s in cyc[n]["steps"] if s["name"] == "Testing Odor Only"]
        assert len(trials) == 8
        assert len({s["duration"] for s in trials}) == 1
        p = [s for s in trials if s["actions"][0]["pin"] == "OFM_P"][0]
        assert p["odor_label"] == "ParaffinOil"
        assert "light_schedule" not in p
    base = apply_sensitivity_overrides(_load(f"config_v2_{letter}.yaml"))
    base["experiment"]["paraffin_bookends"] = False
    plain = {c["cycle"]: c for c in expand_config(base, control=True)["cycles"]}

    def per_trial(steps):
        """[(name, duration), ...] per recording, Start -> Stop; the gaps
        between recordings are checked separately below."""
        out, cur, gaps = [], None, []
        for s in steps:
            if s["name"] == "Start Recording":
                cur = []
            if cur is None:
                gaps.append(s["duration"])
                continue
            cur.append((s["name"], s["duration"]))
            if s["name"] == "Stop Recording":
                out.append(cur)
                cur = None
        return out, gaps

    pre_b, gaps_b = per_trial(cyc[5]["steps"])
    pre_p, gaps_p = per_trial(plain[5]["steps"])
    assert len(pre_b) == 8 and len(pre_p) == 7
    assert pre_b[1:] == pre_p      # P is trial 1, the vial trials are unchanged
    assert gaps_b == [240] + gaps_p   # one more standard 240 s gap, after P
    assert cyc[5]["delay_after"] == plain[5]["delay_after"]   # same 30 min rest
    # the post-test vial trials run exactly as before; P only follows them
    post_b, gaps_b = per_trial(cyc[7]["steps"])
    post_p, gaps_p = per_trial(plain[7]["steps"])
    assert post_b[:-1] == post_p
    assert [g for g in gaps_b if g] == [g for g in gaps_p if g] + [240]


def test_bookends_are_opt_in(tmp_path):
    """Protocols without the key (hex test, norest, benzpre) keep their panels."""
    for name in ("config_v2_hextest_H.yaml", "config_v2_A_benzpre.yaml"):
        cfg = _load(name)
        assert "paraffin_bookends" not in cfg["experiment"]
    cfg = {"format": "v2", "experiment": {"trained_odor": "OFM_H", "pretest": True,
                                          "testing_sequence": "all_random"}}
    cyc = {c["cycle"]: c for c in expand_config(cfg, control=True)["cycles"]}
    assert "OFM_P" not in _panel(cyc[5]) + _panel(cyc[7])


def test_timings_match_hexanol_session():
    def shape(cyc):
        return {n: (c["repeat"], c["delay_after"],
                    [(s["name"], s["duration"]) for s in c["steps"]])
                for n, c in cyc.items()}
    p, h = shape(_expanded("P")), shape(_expanded("H"))
    assert sorted(p) == sorted(h) == [5, 6, 7, 8]
    assert p[6] == h[6] and p[8] == h[8]
    # panels shuffle their order, so compare the multiset of step timings
    for n in (5, 7):
        assert p[n][:2] == h[n][:2]
        assert sorted(p[n][2]) == sorted(h[n][2])


# ── rig scripts: the GPIO sequence itself ────────────────────────────

class _FakeGPIO:
    def __init__(self):
        self.state, self.writes = {}, []

    def gpio_write(self, _h, pin, level):
        self.state[pin] = level
        self.writes.append((pin, level))

    def gpio_read(self, _h, pin):
        return self.state.get(pin, 0)


def _rig_namespace(script):
    """Module-level pin constants + the valve functions, run against a fake lgpio."""
    src = (RIG_DIR / script).read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep_assign = {"OFM_H", "OFM_B", "OFM_C", "OFM_L", "OFM_P", "OFM_A", "OFM_O",
                   "OFM_E", "OFM_PIN_MAPPING", "PARAFFIN_CLICK_S"}
    keep_def = {"_set_odor", "_set_paraffin", "_any_ofm_branch_on",
                "get_active_ofm_pin", "_normalize_pin_name"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in keep_assign for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep_def:
            body.append(node)
    fake = _FakeGPIO()
    logged = []
    ns = {"lgpio": fake, "gpio_handle": object(), "time": time,
          "log_sensor_row": logged.append, "ODOR_ALIAS_TO_PIN_NAME": {}}
    exec(compile(ast.Module(body=body, type_ignores=[]), script, "exec"), ns)
    return ns, fake, logged


@pytest.mark.parametrize("script", SCRIPTS)
def test_paraffin_only_clicks_the_master_and_leaves_paraffin_flowing(script):
    ns, fake, logged = _rig_namespace(script)
    master = ns["OFM_P"]
    assert ns["PARAFFIN_CLICK_S"] == 0.05
    assert ns["_normalize_pin_name"]("P") == "OFM_P"
    assert "OFM_P" in ns["OFM_PIN_MAPPING"]

    class _Clock:  # record the sleeps in line with the GPIO writes
        @staticmethod
        def sleep(s):
            fake.writes.append(("sleep", s))
    ns["time"] = _Clock()

    click = [(master, 1), ("sleep", 0.05), (master, 0)]
    ns["_set_odor"]("OFM_P", True)
    assert fake.writes == click
    # master OFF = paraffin line open: it must stay OFF for the whole window
    assert fake.state[master] == 0
    assert ns["get_active_ofm_pin"]() == "off"
    fake.writes.clear()
    ns["_set_odor"]("OFM_P", False)
    assert fake.writes == click
    assert fake.state[master] == 0
    assert ns["get_active_ofm_pin"]() == "off"
    assert logged == ["OFM_P_ON", "OFM_P_OFF"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_real_odor_sequence_unchanged(script):
    ns, fake, logged = _rig_namespace(script)
    master, hexv = ns["OFM_P"], ns["OFM_H"]
    ns["_set_odor"]("OFM_H", True)
    assert fake.writes == [(hexv, 1), (master, 1)]
    assert ns["get_active_ofm_pin"]() == "OFM_H"
    fake.writes.clear()
    ns["_set_odor"]("OFM_H", False)
    assert fake.writes == [(master, 0), (hexv, 0)]
    assert logged == ["OFM_H_ON", "OFM_H_OFF"]


# ── --odors H,O,P: the round-robin must be able to pick P ────────────

def test_filtered_round_robin_includes_paraffin(tmp_path, monkeypatch):
    monkeypatch.setattr(sched, "STATE_FILE", tmp_path / "state.json")
    flt = [sched.resolve_odor_pin(x) for x in "H,O,P".split(",")]
    pool = sched.get_remaining_conditions(odor_filter=flt)
    assert sorted(c for c in pool if c[1] == "Control") == [
        ("OFM_H", "Control"), ("OFM_O", "Control"), ("OFM_P", "Control")]
    # after H and O, P is the one left -- each Pi rotates through all three
    sched.mark_condition_done("OFM_H", "Control")
    sched.mark_condition_done("OFM_O", "Control")
    left = [c for c in sched.get_remaining_conditions(odor_filter=flt) if c[1] == "Control"]
    assert left == [("OFM_P", "Control")]


def test_unfiltered_round_robin_never_picks_paraffin(tmp_path, monkeypatch):
    monkeypatch.setattr(sched, "STATE_FILE", tmp_path / "state.json")
    pool = sched.get_remaining_conditions()
    assert len(pool) == 14 and all(p != "OFM_P" for p, _ in pool)
