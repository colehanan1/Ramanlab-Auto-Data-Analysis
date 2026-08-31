"""``get_cycle_name`` must tag the pre-test protocol's four cycles.

The rig scripts can't be imported off the Pi, but ``get_cycle_name`` is a pure
function, so it is lifted out of the AST and exec'd on its own. Cycle numbers
here must match the ``_PRETEST_*`` constants in expand_config.py.
"""

import ast
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


def _lift(script, func_name):
    tree = ast.parse((PICODE / script).read_text(encoding="utf-8"))
    node = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    assert node is not None, f"{script} has no top-level {func_name}"
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), script, "exec"), ns)
    return ns[func_name]


@pytest.fixture(params=SCRIPTS)
def get_cycle_name(request):
    return _lift(request.param, "get_cycle_name")


def test_standard_protocol_tags_unchanged(get_cycle_name):
    assert get_cycle_name(1, 0) == "training_1"
    assert get_cycle_name(2, 0) == "testing_1"
    assert get_cycle_name(3, 0) == "testing_9"
    assert get_cycle_name(4, 0) == "testing_10"


def test_pretest_cycle_tags(get_cycle_name):
    assert get_cycle_name(5, 0) == "pretest_1"
    assert get_cycle_name(5, 6) == "pretest_7"


def test_pretest_training_cycle_tags(get_cycle_name):
    assert get_cycle_name(6, 0) == "training_1"
    assert get_cycle_name(6, 5) == "training_6"


def test_pretest_posttest_cycle_tags(get_cycle_name):
    assert get_cycle_name(7, 0) == "testing_1"
    assert get_cycle_name(7, 6) == "testing_7"


def test_pretest_light_probe_tag(get_cycle_name):
    assert get_cycle_name(8, 0) == "testing_9"


def test_odor_label_appended(get_cycle_name):
    assert get_cycle_name(5, 0, odor_label="Citral") == "pretest_1_Citral"


def test_unknown_cycle_still_unknown(get_cycle_name):
    assert get_cycle_name(99, 0) == "unknown"


@pytest.mark.parametrize("script", SCRIPTS)
def test_split_phase_index_understands_pretest(script):
    split = _lift(script, "_split_phase_index")
    assert split("pretest_3_Hexanol") == ("pretest", 3)
    assert split("training_5_Hexanol") == ("training", 5)
    assert split("testing_2_ACV") == ("testing", 2)


@pytest.mark.parametrize("script", SCRIPTS)
def test_cycle_numbers_match_expand_config(script):
    """The tag map and expand_config's cycle numbers must not drift apart."""
    import sys, os
    sys.path.insert(0, str(PICODE))
    import expand_config as ec
    get = _lift(script, "get_cycle_name")
    assert get(ec._PRETEST_CYCLE, 0).startswith("pretest_")
    assert get(ec._PRETEST_TRAINING_CYCLE, 0).startswith("training_")
    assert get(ec._PRETEST_POST_CYCLE, 0).startswith("testing_")
    assert get(ec._PRETEST_LIGHT_CYCLE, 0) == "testing_9"
