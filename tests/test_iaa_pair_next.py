"""PiCode/iaa_pair_next.py: per-rig balanced IAA-first / IAA-second picker + launcher."""
import importlib.util
import random
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "PiCode" / "iaa_pair_next.py"
_spec = importlib.util.spec_from_file_location("iaa_pair_next", _P)
ipn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ipn)

FULL = ["--fly", "2", "--n-flies", "4", "--fly-type", "GR5a-Old", "--born-date", "11/09/2026",
        "--starved-date", "11/13/2026", "--starved-time", "09:34", "--starvation-hours", "1",
        "--odor-vial-conc", "E=1,H=0.1,O=0.1,I=1,C=1,B=0.1,L=1",
        "--odor-vial-created-date", "11/11/2026", "--retinal-vial-activate-date", "11/05/2026"]


# ── choice rule ───────────────────────────────────────────────────────────
def test_half_done_block_forces_the_other():
    assert ipn.choose(["first"], random.Random(0)) == "second"
    assert ipn.choose(["second"], random.Random(0)) == "first"
    assert ipn.choose(["first", "second", "second"], random.Random(0)) == "first"


def test_complete_block_is_random_and_both_come_up():
    seen = {ipn.choose(["first", "second"], random.Random(s)) for s in range(40)}
    assert seen == {"first", "second"}
    seen = {ipn.choose([], random.Random(s)) for s in range(40)}
    assert seen == {"first", "second"}


def test_no_version_ever_three_in_a_row_or_ahead_by_two():
    rng = random.Random(1)
    hist = []
    for _ in range(200):
        hist.append(ipn.choose(hist, rng))
        assert abs(hist.count("first") - hist.count("second")) <= 1
    for i in range(0, 200, 2):
        assert sorted(hist[i:i + 2]) == ["first", "second"]


# ── history file ──────────────────────────────────────────────────────────
def test_history_roundtrip_and_undo(tmp_path):
    h = tmp_path / "hist.txt"
    assert ipn.read_history(h) == []
    ipn.append_history(h, "first", "fly=1")
    ipn.append_history(h, "second", "fly=2")
    assert ipn.read_history(h) == ["first", "second"]
    assert ipn.undo_last(h) == "second"
    assert ipn.read_history(h) == ["first"]


def test_history_ignores_comments_and_junk(tmp_path):
    h = tmp_path / "hist.txt"
    h.write_text("# header\n\n2026-11-13 11:48  first  fly=1 seeded\nnonsense line\n")
    assert ipn.read_history(h) == ["first"]


# ── argv building / validation ────────────────────────────────────────────
def test_missing_required_flag_is_reported():
    argv = [a for a in FULL]
    i = argv.index("--retinal-vial-activate-date")
    del argv[i:i + 2]
    assert ipn.missing_flags(argv) == ["--retinal-vial-activate-date"]
    assert ipn.missing_flags(FULL) == []


def test_flag_given_with_equals_counts():
    argv = [a for a in FULL if a not in ("--fly", "2")] + ["--fly=2"]
    assert "--fly" not in ipn.missing_flags(argv)


def test_user_may_not_pass_config_or_mode():
    with pytest.raises(SystemExit):
        ipn.build_argv("combinedv2_1.py", "first", FULL + ["--config", "x.yaml"])
    with pytest.raises(SystemExit):
        ipn.build_argv("combinedv2_1.py", "first", FULL + ["--sensitivity"])


def test_build_argv_uses_right_config():
    a = ipn.build_argv("combinedv2_1_pi3.py", "second", FULL)
    assert a[:6] == ["combinedv2_1_pi3.py", "--config", "config_v2_pairsecond_A.yaml",
                     "--control", "--push-mode", "hub"]
    assert a[6:] == FULL
    b = ipn.build_argv("combinedv2_1_pi1.py", "first", FULL)
    assert b[2] == "config_v2_pairfirst_A.yaml"


def test_script_for_host():
    assert ipn.script_for_host("BehaviorLocust") == "combinedv2_1_pi1.py"
    assert ipn.script_for_host("Flybehavior2") == "combinedv2_1.py"
    assert ipn.script_for_host("Flybehavior3") == "combinedv2_1_pi3.py"
    with pytest.raises(SystemExit):
        ipn.script_for_host("behavior4080")


# ── main(): dry run never writes history, a real launch records then execs ──
def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    h = tmp_path / "hist.txt"
    monkeypatch.setattr(ipn, "rig_running", lambda: False)
    monkeypatch.setattr(ipn.os, "execv", lambda *a: pytest.fail("must not exec"))
    ipn.main(["--dry-run", "--history", str(h), "--host", "Flybehavior2", *FULL])
    assert not h.exists()


def test_launch_records_then_execs(tmp_path, monkeypatch):
    h = tmp_path / "hist.txt"
    h.write_text("2026-11-13 11:48  second  fly=1 seeded\n")
    calls = []
    monkeypatch.setattr(ipn, "rig_running", lambda: False)
    monkeypatch.setattr(ipn.os, "execv", lambda exe, argv: calls.append(argv))
    ipn.main(["--history", str(h), "--host", "Flybehavior2", *FULL])
    assert ipn.read_history(h) == ["second", "first"]
    assert calls and "config_v2_pairfirst_A.yaml" in calls[0]


def test_missing_flag_aborts_before_recording(tmp_path, monkeypatch):
    h = tmp_path / "hist.txt"
    monkeypatch.setattr(ipn, "rig_running", lambda: False)
    monkeypatch.setattr(ipn.os, "execv", lambda *a: pytest.fail("must not exec"))
    with pytest.raises(SystemExit):
        ipn.main(["--history", str(h), "--host", "Flybehavior2", *FULL[:-2]])
    assert not h.exists()


def test_refuses_when_rig_already_running(tmp_path, monkeypatch):
    h = tmp_path / "hist.txt"
    monkeypatch.setattr(ipn, "rig_running", lambda: True)
    monkeypatch.setattr(ipn.os, "execv", lambda *a: pytest.fail("must not exec"))
    with pytest.raises(SystemExit):
        ipn.main(["--history", str(h), "--host", "Flybehavior2", *FULL])
    assert not h.exists()
