"""``--sensitivity``: run the pre-test protocol from any standard config.

The rigs are usually launched with ``--odors H`` and no ``--config`` (the
scheduler picks ``config_v2_H.yaml``), so the protocol has to be selectable from
the command line as well as from a ``_pretest`` config file. The override itself
lives in ``expand_config.apply_sensitivity_overrides`` so it is testable off-Pi;
the flag wiring is asserted against each script's AST.
"""

import ast
import sys, os
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))
from expand_config import apply_sensitivity_overrides, expand_config

SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


def _plain():
    return {"format": "v2", "experiment": {"trained_odor": "OFM_H"}}


# ── the override itself ───────────────────────────────────────────────

def test_turns_a_plain_config_into_the_pretest_protocol():
    cfg = apply_sensitivity_overrides(_plain())
    assert [c["cycle"] for c in expand_config(cfg)["cycles"]] == [5, 6, 7, 8]


def test_sets_the_three_protocol_keys():
    exp = apply_sensitivity_overrides(_plain())["experiment"]
    assert exp["pretest"] is True
    assert exp["wait_before_testing"] == 0
    assert exp["testing_sequence"] == "all_random"


def test_does_not_mutate_the_caller_config():
    cfg = _plain()
    apply_sensitivity_overrides(cfg)
    assert "pretest" not in cfg["experiment"]


def test_keeps_an_explicit_config_value():
    """A config that already tunes the protocol wins over the flag's defaults."""
    cfg = _plain()
    cfg["experiment"]["pretest_wait"] = 600
    cfg["experiment"]["testing_sequence"] = "trained_first"
    exp = apply_sensitivity_overrides(cfg)["experiment"]
    assert exp["pretest_wait"] == 600
    assert exp["testing_sequence"] == "trained_first"


def test_is_a_no_op_on_a_pretest_config():
    import yaml
    with open(PICODE / "config_v2_H_pretest.yaml") as fh:
        cfg = yaml.safe_load(fh)
    assert apply_sensitivity_overrides(cfg)["experiment"] == cfg["experiment"]


def test_leaves_the_trained_odor_alone():
    assert apply_sensitivity_overrides(_plain())["experiment"]["trained_odor"] == "OFM_H"


# ── flag wiring on each rig ───────────────────────────────────────────

@pytest.mark.parametrize("script", SCRIPTS)
def test_flag_exists_as_a_store_true(script):
    tree = ast.parse((PICODE / script).read_text(encoding="utf-8"))
    call = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
         and any(isinstance(a, ast.Constant) and a.value == "--sensitivity" for a in n.args)),
        None,
    )
    assert call is not None, f"{script} has no --sensitivity"
    action = next((k.value for k in call.keywords if k.arg == "action"), None)
    assert getattr(action, "value", None) == "store_true"


@pytest.mark.parametrize("script", SCRIPTS)
def test_flag_applies_the_overrides(script):
    src = (PICODE / script).read_text(encoding="utf-8")
    assert "apply_sensitivity_overrides" in src
    # and the cohort-folder flag must see it too
    assert "args.sensitivity" in src
    # the helper is used at module scope, so it must actually be imported --
    # a missing import is a NameError on the rig, not a test failure here
    tree = ast.parse(src)
    imported = {alias.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module == "expand_config"
                for alias in n.names}
    assert "apply_sensitivity_overrides" in imported
