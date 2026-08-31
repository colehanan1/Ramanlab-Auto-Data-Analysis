"""Wiring checks for the ``--wait-before-testing`` flag on all three rigs.

The ``combinedv2_1*.py`` scripts cannot be imported off the Pi (camera, GPIO and
preview server all come up at module scope), so the flag is asserted against the
AST: it must exist on every rig, default to ``None`` (meaning "not specified" so
the config/default still wins), and be forwarded into ``expand_config``. The
override *behaviour* is covered by ``test_expand_config.py``.
"""

import ast
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


def _tree(name: str) -> ast.Module:
    return ast.parse((PICODE / name).read_text(encoding="utf-8"))


def _add_argument_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument"
    ]


def _flag(tree: ast.Module, flag: str) -> ast.Call:
    for call in _add_argument_calls(tree):
        if any(isinstance(a, ast.Constant) and a.value == flag for a in call.args):
            return call
    return None


def _kwarg(call: ast.Call, key: str):
    return next((kw.value for kw in call.keywords if kw.arg == key), None)


@pytest.mark.parametrize("script", SCRIPTS)
def test_flag_exists_with_int_type(script):
    call = _flag(_tree(script), "--wait-before-testing")
    assert call is not None, f"{script} lost --wait-before-testing"
    assert getattr(_kwarg(call, "type"), "id", None) == "int"


@pytest.mark.parametrize("script", SCRIPTS)
def test_flag_defaults_to_none(script):
    """None, not 1680: the config value must still apply when the flag is absent."""
    call = _flag(_tree(script), "--wait-before-testing")
    default = _kwarg(call, "default")
    assert isinstance(default, ast.Constant) and default.value is None


@pytest.mark.parametrize("script", SCRIPTS)
def test_no_wait_shorthand_is_zero(script):
    call = _flag(_tree(script), "--no-wait-before-testing")
    assert call is not None, f"{script} lost --no-wait-before-testing"
    assert getattr(_kwarg(call, "dest"), "value", None) == "wait_before_testing"
    const = _kwarg(call, "const")
    assert isinstance(const, ast.Constant) and const.value == 0


@pytest.mark.parametrize("script", SCRIPTS)
def test_forwarded_to_expand_config(script):
    tree = _tree(script)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "expand_config"
    ]
    assert calls, f"{script} never calls expand_config"
    forwarded = [c for c in calls if _kwarg(c, "wait_before_testing") is not None]
    assert forwarded, f"{script} does not forward wait_before_testing"
    for call in forwarded:
        val = _kwarg(call, "wait_before_testing")
        assert isinstance(val, ast.Attribute) and val.attr == "wait_before_testing"
        assert getattr(val.value, "id", None) == "args"
