"""AST wiring checks for the live status page inside ``PiCode/combinedv2_1.py``.

The script cannot be imported off-Pi (camera/GPIO claimed at module scope), so
these tests parse it and assert the wiring that ``test_phase_status.py`` cannot
see: HTTP routes, the threading server, phase publishing at step boundaries,
and the T-5:00 ntfy warning in the startup countdown.
"""

import ast
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "PiCode" / "combinedv2_1.py"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _calls_to(node, func_name):
    """Every Call whose target is a bare name or attribute named *func_name*."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == func_name:
            out.append(sub)
    return out


def _constants(node):
    return {c.value for c in ast.walk(node)
            if isinstance(c, ast.Constant) and isinstance(c.value, (str, int, float))}


# ── HTTP server ──────────────────────────────────────────────────────────────

def test_http_server_is_threading(tree):
    """/preview holds its connection open forever; on a single-threaded
    HTTPServer, /status.json would never get a turn. The server must be a
    ThreadingHTTPServer."""
    cls = _find_class(tree, "ReusableHTTPServer")
    bases = {getattr(b, "attr", getattr(b, "id", None)) for b in cls.bases}
    assert "ThreadingHTTPServer" in bases


def test_handler_serves_all_three_routes(tree):
    handler = _find_class(tree, "_MJPEGHandler")
    do_get = _find_function(handler, "do_GET")
    consts = _constants(do_get)
    for route in ("/", "/status", "/status.json", "/preview"):
        assert route in consts, f"do_GET does not route {route}"


def test_status_json_route_serializes_phase_state(tree):
    handler = _find_class(tree, "_MJPEGHandler")
    do_get = _find_function(handler, "do_GET")
    calls = _calls_to(do_get, "to_json")
    assert any(
        isinstance(c.func, ast.Attribute)
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "PHASE"
        for c in calls
    ), "do_GET must serve PHASE.to_json()"


def test_status_page_html_served(tree):
    handler = _find_class(tree, "_MJPEGHandler")
    do_get = _find_function(handler, "do_GET")
    names = {n.id for n in ast.walk(do_get) if isinstance(n, ast.Name)}
    assert "STATUS_PAGE_HTML" in names


# ── module wiring ────────────────────────────────────────────────────────────

def test_imports_phase_status_module(tree):
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "phase_status"
        for alias in node.names
    }
    assert {"PhaseState", "STATUS_PAGE_HTML", "classify_step"} <= imported


def test_phase_state_instantiated_at_module_level(tree):
    assigns = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PHASE" for t in node.targets)
    ]
    assert assigns, "PHASE = PhaseState() must exist at module level"
    value = assigns[0].value
    assert isinstance(value, ast.Call) and getattr(value.func, "id", None) == "PhaseState"


# ── phase publishing ─────────────────────────────────────────────────────────

def test_record_cycle_publishes_step_phases(tree):
    record = _find_function(tree, "record_cycle")
    assert _calls_to(record, "_publish_step_phase"), \
        "record_cycle must publish phase state at step boundaries"


def test_publish_helper_classifies_and_enters(tree):
    helper = _find_function(tree, "_publish_step_phase")
    assert _calls_to(helper, "classify_step")
    enters = _calls_to(helper, "enter")
    assert any(
        isinstance(c.func, ast.Attribute)
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "PHASE"
        for c in enters
    )


def test_inter_cycle_wait_publishes_waiting(tree):
    """The 28-min delay_after gap must show a countdown, not a stale phase."""
    record = _find_function(tree, "record_cycle")
    names = {n.id for n in ast.walk(record) if isinstance(n, ast.Name)}
    assert "PHASE_WAITING" in names


def test_experiment_completion_is_published(tree):
    execute = _find_function(tree, "execute_cycles_in_order")
    names = {n.id for n in ast.walk(execute) if isinstance(n, ast.Name)}
    assert "PHASE_COMPLETE" in names


# ── startup countdown: T-5:00 ntfy + WAITING phase ──────────────────────────

def test_startup_countdown_publishes_waiting(tree):
    countdown = _find_function(tree, "_startup_countdown")
    enters = _calls_to(countdown, "enter")
    assert any(
        isinstance(c.func, ast.Attribute)
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "PHASE"
        for c in enters
    )


def test_startup_countdown_sends_5_minute_warning(tree):
    countdown = _find_function(tree, "_startup_countdown")
    assert _calls_to(countdown, "ntfy_notify"), \
        "_startup_countdown must send the pre-training ntfy warning"
    assert 300 in _constants(countdown), \
        "warning threshold must be 300 s (5 minutes)"


def test_5_minute_warning_fires_once(tree):
    """The countdown loop passes rem<=300 every second after the threshold —
    a sent-flag must prevent 300 duplicate notifications."""
    countdown = _find_function(tree, "_startup_countdown")
    src = ast.unparse(countdown)
    assert "notified" in src
