"""Source-level checks that every rig script wires the startup-wait controls."""
import ast
from pathlib import Path
import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


def _src(name):
    return (PICODE / name).read_text()


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(name)


@pytest.mark.parametrize("script", SCRIPTS)
def test_handler_has_control_post(script):
    tree = ast.parse(_src(script))
    handler = _func(tree, "_MJPEGHandler")
    methods = {n.name for n in handler.body if isinstance(n, ast.FunctionDef)}
    assert "do_POST" in methods
    seg = ast.get_source_segment(_src(script), handler)
    assert "/control/wait" in seg and "handle_wait_control" in seg and "X-Rig-Key" in seg


@pytest.mark.parametrize("script", SCRIPTS)
def test_countdown_uses_wait_control(script):
    src = _src(script)
    fn = _func(ast.parse(src), "_startup_countdown")
    seg = ast.get_source_segment(src, fn)
    assert "WaitControl(" in seg and ".due()" in seg and ".close()" in seg
    assert "RUN.shift" in seg
    assert "range(int(seconds)" not in seg          # fixed-count loop is gone
    for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "enter"][1:]:
        assert any(k.arg == "started_at" for k in call.keywords)


@pytest.mark.parametrize("script", SCRIPTS)
def test_metadata_records_wait(script):
    src = _src(script)
    for key in ("startup_wait_planned_s", "startup_wait_actual_s", "startup_wait_log"):
        assert key in src
    assert "load_control_key" in src and ".rigmon_key" in src


@pytest.mark.parametrize("script", SCRIPTS)
def test_countdown_holds_reminders_and_label_follows_paused(script):
    src = _src(script)
    seg = ast.get_source_segment(src, _func(ast.parse(src), "_startup_countdown"))
    assert "RUN.hold(True)" in seg and "RUN.hold(False)" in seg
    assert 'snap["paused"]' in seg
    assert 'e["action"] == "pause":\n                        PHASE.enter' not in seg


@pytest.mark.parametrize("script", SCRIPTS)
def test_do_post_sets_socket_timeout(script):
    src = _src(script)
    seg = ast.get_source_segment(src, _func(ast.parse(src), "_MJPEGHandler"))
    assert seg.index("settimeout(5)") < seg.index("rfile.read")
