"""PiCode/iaa_10s_next.py: per-rig balanced training / control launcher for
config_v2_A_10s.yaml."""
import importlib.util
import random
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[1] / "PiCode"
sys.path.insert(0, str(_DIR))
_spec = importlib.util.spec_from_file_location("iaa_10s_next", _DIR / "iaa_10s_next.py")
nxt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nxt)

FULL = ["--fly", "2", "--n-flies", "4", "--fly-type", "GR5a-Old", "--born-date", "11/09/2026",
        "--starved-date", "11/13/2026", "--starved-time", "09:34", "--starvation-hours", "24",
        "--odor-vial-conc", "E=1,H=0.1,O=0.1,I=1,C=1,B=0.1,L=1",
        "--odor-vial-created-date", "11/11/2026", "--retinal-vial-activate-date", "11/05/2026"]


def test_half_done_block_forces_the_other():
    assert nxt.choose(["training"], random.Random(0)) == "control"
    assert nxt.choose(["control"], random.Random(0)) == "training"


def test_blocks_of_two_always_balanced():
    rng = random.Random(3)
    hist = []
    for _ in range(100):
        hist.append(nxt.choose(hist, rng))
    for i in range(0, 100, 2):
        assert sorted(hist[i:i + 2]) == ["control", "training"]
    seen = {nxt.choose([], random.Random(s)) for s in range(40)}
    assert seen == {"training", "control"}


@pytest.mark.parametrize("version,flag", [("training", "--training"), ("control", "--control")])
def test_argv_sets_10s_config_mode_and_hub(version, flag):
    argv = nxt.build_argv("combinedv2_1_pi3.py", version, FULL)
    assert argv[:6] == ["combinedv2_1_pi3.py", "--config", "config_v2_A_10s.yaml",
                        flag, "--push-mode", "hub"]
    assert argv[6:] == FULL
    other = "--control" if flag == "--training" else "--training"
    assert other not in argv


@pytest.mark.parametrize("bad", ["--config", "--control", "--training", "--sensitivity",
                                 "--odors", "--push-mode"])
def test_forbidden_flags(bad):
    with pytest.raises(SystemExit):
        nxt.build_argv("combinedv2_1.py", "training", FULL + [bad, "x"])


def test_history_is_separate_from_pair_history(tmp_path):
    assert nxt.DEFAULT_HISTORY.name == "iaa_10s_history.txt"
    h = tmp_path / "h.txt"
    nxt.append_history(h, "training", "fly=2")
    nxt.append_history(h, "control", "fly=3")
    # pair-launcher lines are ignored
    with open(h, "a") as f:
        f.write("2026-11-20 10:00  first  fly=1\n")
    assert nxt.read_history(h) == ["training", "control"]
    assert nxt.undo_last(h) == "control"
    assert nxt.read_history(h) == ["training"]


def test_missing_flags_abort_and_record_nothing(tmp_path):
    h = tmp_path / "h.txt"
    with pytest.raises(SystemExit):
        nxt.main(["--history", str(h), "--host", "flybehavior3", "--fly", "1"])
    assert not h.exists()


def test_dry_run_records_nothing(tmp_path, capsys):
    h = tmp_path / "h.txt"
    nxt.main(["--history", str(h), "--host", "flybehavior2", "--dry-run", *FULL])
    out = capsys.readouterr().out
    assert "config_v2_A_10s.yaml" in out and "combinedv2_1.py" in out
    assert not h.exists()


def test_unknown_host():
    with pytest.raises(SystemExit):
        nxt.main(["--host", "behavior4080", "--dry-run", *FULL])
