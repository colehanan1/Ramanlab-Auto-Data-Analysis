"""Rig wiring for the pre-test ("Sensitivity") cohort.

The scripts can't be imported off the Pi, so the wiring is asserted against the
AST: the push folder must be derived from the config's ``pretest`` flag on every
rig, and both odor panels must reach the session metadata.
"""

import ast
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


def _src(script):
    return (PICODE / script).read_text(encoding="utf-8")


def _tree(script):
    return ast.parse(_src(script))


@pytest.mark.parametrize("script", SCRIPTS)
def test_sensitivity_flag_read_from_config(script):
    """_is_sensitivity comes from experiment.pretest, not a CLI flag."""
    tree = _tree(script)
    assign = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.Assign)
         and any(getattr(t, "id", None) == "_is_sensitivity" for t in n.targets)),
        None,
    )
    assert assign is not None, f"{script} has no _is_sensitivity"
    assert "pretest" in ast.unparse(assign.value)


@pytest.mark.parametrize("script", SCRIPTS)
def test_sensitivity_forwarded_to_push_folder(script):
    calls = [n for n in ast.walk(_tree(script))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "get_push_folder"]
    assert calls, f"{script} never calls get_push_folder"
    for call in calls:
        kw = next((k for k in call.keywords if k.arg == "sensitivity"), None)
        assert kw is not None, f"{script}: get_push_folder missing sensitivity="
        assert getattr(kw.value, "id", None) == "_is_sensitivity"


@pytest.mark.parametrize("script", SCRIPTS)
def test_sensitivity_read_before_it_is_used(script):
    """cfg must already be loaded, and the flag set before the push call."""
    src = _src(script)
    assert src.index("cfg = yaml.safe_load") < src.index("_is_sensitivity =")
    assert src.index("_is_sensitivity =") < src.index("get_push_folder(")


@pytest.mark.parametrize("script", SCRIPTS)
def test_both_panels_recorded_in_metadata(script):
    src = _src(script)
    assert "pretest_odor_order" in src
    # the post-training panel is cycle 7 under the pre-test protocol
    assert "_panel_order(7)" in src
    assert "_panel_order(5)" in src
