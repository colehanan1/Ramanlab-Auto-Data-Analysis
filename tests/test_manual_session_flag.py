"""Static wiring checks for the rig-2 ``--manual`` flag.

``PiCode/combinedv2_1.py`` cannot be imported here: it opens the camera, claims
GPIO and starts the preview server at module scope, and depends on picamera2 /
lgpio / board, none of which exist off the Pi. The folder-naming *logic* is
covered by ``test_push_folder_naming.py``; what remains is the wiring inside the
script, which is asserted here against its AST so a refactor that drops the flag,
flips a default, or stops forwarding ``manual=`` fails loudly.
"""

import ast
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "PiCode" / "combinedv2_1.py"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _calls(tree: ast.Module, func_name: str) -> list[ast.Call]:
    """Every call to a bare name or attribute ending in *func_name*."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == func_name:
            out.append(node)
    return out


def _kwarg(call: ast.Call, key: str):
    return next((kw.value for kw in call.keywords if kw.arg == key), None)


def _assigned_value(tree: ast.Module, target: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == target for t in node.targets
        ):
            return node.value
    return None


# ── the flag itself ──────────────────────────────────────────────────────────

def test_manual_flag_is_declared_as_an_off_by_default_switch(tree):
    manual = [
        c for c in _calls(tree, "add_argument")
        if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "--manual"
    ]
    assert len(manual) == 1, "expected exactly one --manual argument declaration"
    call = manual[0]
    action = _kwarg(call, "action")
    default = _kwarg(call, "default")
    assert isinstance(action, ast.Constant) and action.value == "store_true"
    # Opto is the default session kind; manual must be opt-in.
    assert isinstance(default, ast.Constant) and default.value is False


# ── local recording root ─────────────────────────────────────────────────────

def test_root_dirs_point_at_separate_opto_and_manual_trees(tree):
    roots = {}
    for name in ("OPTO_ROOT_DIR", "MANUAL_ROOT_DIR"):
        value = _assigned_value(tree, name)
        assert value is not None, f"{name} is not assigned"
        # Path("...") → grab the literal
        assert isinstance(value, ast.Call) and value.args
        roots[name] = value.args[0].value
    assert roots["OPTO_ROOT_DIR"].endswith("/Opto")
    assert roots["MANUAL_ROOT_DIR"].endswith("/Manual")
    assert roots["OPTO_ROOT_DIR"] != roots["MANUAL_ROOT_DIR"]


def test_root_dir_switches_on_the_manual_flag(tree):
    value = _assigned_value(tree, "ROOT_DIR")
    assert isinstance(value, ast.IfExp), "ROOT_DIR must branch on args.manual"
    test = value.test
    assert isinstance(test, ast.Attribute) and test.attr == "manual"
    assert isinstance(value.body, ast.Name) and value.body.id == "MANUAL_ROOT_DIR"
    assert isinstance(value.orelse, ast.Name) and value.orelse.id == "OPTO_ROOT_DIR"


# ── conditioning light ───────────────────────────────────────────────────────

def test_expand_config_is_told_about_manual_mode(tree):
    """--manual must suppress the training light: the expand_config call has to
    forward the flag, otherwise cycles are built with light_schedule and the
    LED fires during manual training trials."""
    calls = [
        c for c in _calls(tree, "expand_config")
        if not isinstance(c.func, ast.Attribute)  # skip the import statement
    ]
    assert calls, "expand_config call site not found"
    for call in calls:
        manual = _kwarg(call, "manual")
        assert manual is not None, "expand_config must receive manual="
        assert isinstance(manual, ast.Attribute) and manual.attr == "manual"


# ── push destination ─────────────────────────────────────────────────────────

def test_push_folder_call_forwards_the_manual_flag(tree):
    calls = _calls(tree, "get_push_folder")
    assert len(calls) == 1, "expected a single get_push_folder call site"
    manual = _kwarg(calls[0], "manual")
    assert manual is not None, "get_push_folder must be told whether this is a manual run"
    assert isinstance(manual, ast.Attribute) and manual.attr == "manual"


def test_manual_suffix_is_imported_rather_than_hardcoded(tree):
    """The warning path must reuse the scheduler's suffix constant."""
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "experiment_scheduler"
        for alias in node.names
    }
    assert "MANUAL_SUFFIX" in imported


# ── session provenance ───────────────────────────────────────────────────────

def test_session_header_records_manual_provenance(tree):
    """A manual run must be identifiable from its metadata alone."""
    keys = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "md"
        and isinstance(node.slice, ast.Constant)
    }
    assert {"session_kind", "manual_session", "output_root"} <= keys
