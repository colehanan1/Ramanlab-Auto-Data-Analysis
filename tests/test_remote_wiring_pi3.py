"""AST wiring checks for the opt-in --remote stream/telemetry hooks in the Pi 3 and Pi 1 scripts.

Parses each script instead of importing it (it claims camera/GPIO at import).
Set PI3_SCRIPT=<path> / PI1_SCRIPT=<path> to validate a draft copy; defaults are
PiCode/combinedv2_1_pi3.py and PiCode/combinedv2_1_pi1.py.
The checks pin the experiment-timing rules: lores only when --remote, the trial
encoder (not all encoders) is stopped, and no telemetry in the capture loop or
the light pulse loop.
"""

import ast
import os
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
SCRIPTS = {
    "pi3": Path(os.environ.get("PI3_SCRIPT", PICODE / "combinedv2_1_pi3.py")),
    "pi1": Path(os.environ.get("PI1_SCRIPT", PICODE / "combinedv2_1_pi1.py")),
}


@pytest.fixture(scope="module", params=sorted(SCRIPTS))
def tree(request) -> ast.Module:
    return ast.parse(SCRIPTS[request.param].read_text(encoding="utf-8"))


def _find(tree, name, cls=(ast.FunctionDef, ast.AsyncFunctionDef)):
    for node in ast.walk(tree):
        if isinstance(node, cls) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _refs_name(node, name):
    return name in _names(node)


def _refs_args_remote(test):
    return any(isinstance(n, ast.Attribute) and n.attr == "remote"
               and isinstance(n.value, ast.Name) and n.value.id == "args"
               for n in ast.walk(test))


def _emit_calls(tree, event=None):
    out = []
    for c in ast.walk(tree):
        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "emit" and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "TELEMETRY" and c.args
                and isinstance(c.args[0], ast.Constant)
                and (event is None or c.args[0].value == event)):
            out.append(c)
    return out


def test_remote_flag_defaults_off(tree):
    for c in ast.walk(tree):
        if (isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "add_argument"
                and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "--remote"):
            kw = {k.arg: ast.literal_eval(k.value) for k in c.keywords
                  if k.arg in ("action", "default")}
            assert kw == {"action": "store_true", "default": False}
            return
    raise AssertionError("--remote not defined")


def test_lores_only_when_remote(tree):
    fn = _find(tree, "initialize_camera_safe")
    guarded = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.If, ast.IfExp)) and _refs_args_remote(n.test):
            for sub in ast.walk(n):
                guarded.add(id(sub))
    hits = [n for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and n.value == "lores"]
    assert hits, "lores config missing"
    assert all(id(h) in guarded for h in hits), "lores configured outside an args.remote guard"


def test_stop_encoder_takes_the_trial_encoder_only_when_streaming(tree):
    """With --remote, stop only the trial's encoder so the lores tap survives.
    Without it, the original bare stop_encoder() must stay (user decision:
    the remote-off path stays textually identical to the live script)."""
    rec = _find(tree, "video_recorder")
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "stop_encoder"]
    assert calls and all(c in list(ast.walk(rec)) for c in calls), \
        "stop_encoder used outside video_recorder"
    ifexps = [n for n in ast.walk(rec) if isinstance(n, ast.IfExp)
              and isinstance(n.test, ast.Name) and n.test.id == "STREAMER"]
    assert len(ifexps) == 1, "expected `... if STREAMER else ...` around stop_encoder"
    body, orelse = ifexps[0].body, ifexps[0].orelse
    assert getattr(body.func, "attr", None) == "stop_encoder"
    assert [getattr(a, "id", None) for a in body.args] == ["encoder"]
    assert getattr(orelse.func, "attr", None) == "stop_encoder"
    assert not orelse.args and not orelse.keywords, "remote-off path must keep the bare call"
    assert len(calls) == 2


def test_no_remote_calls_in_capture_or_pulse_loops(tree):
    rec = _find(tree, "video_recorder")
    loops = [n for n in ast.walk(rec) if isinstance(n, ast.While)]
    assert loops
    for loop in loops:
        assert not ({"TELEMETRY", "STREAMER", "_telemetry_light"} & _names(loop))
    # The LightController pulse loop; Pi 1 (TTL, digital only) has none.
    for run in [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_run"]:
        assert not ({"TELEMETRY", "STREAMER", "_telemetry_light"} & _names(run))


@pytest.mark.parametrize("event", ["session_start", "trial_start", "odor_on", "odor_off",
                                   "opto_on", "opto_off", "trial_end", "session_end", "error"])
def test_event_is_emitted(tree, event):
    emitted = {c.args[0].value for c in _emit_calls(tree)}
    # opto_* go through the single _telemetry_light choke point as a conditional expression
    for c in ast.walk(_find(tree, "_telemetry_light")):
        if isinstance(c, ast.Constant) and isinstance(c.value, str):
            emitted.add(c.value)
    assert event in emitted


def test_session_end_has_complete_and_aborted_sites(tree):
    statuses = {kw.value.value for c in _emit_calls(tree, "session_end")
                for kw in c.keywords if kw.arg == "status"}
    assert {"complete", "aborted"} <= statuses
    assert len(_emit_calls(tree, "session_end")) >= 2


def test_odor_emits_follow_the_gpio_log_row(tree):
    """Each branch: GPIO writes, 50 ms sleep, log_sensor_row, THEN the emit."""
    fn = _find(tree, "_set_odor")
    branch_if = next(n for n in fn.body if isinstance(n, ast.If) and _refs_name(n.test, "on"))
    for body, ev in ((branch_if.body, "odor_on"), (branch_if.orelse, "odor_off")):
        stmts = [ast.Module(body=[s], type_ignores=[]) for s in body]
        log_i = [i for i, m in enumerate(stmts) if any(
            isinstance(c, ast.Call) and getattr(c.func, "id", None) == "log_sensor_row"
            for c in ast.walk(m))]
        emit_i = [i for i, m in enumerate(stmts) if _emit_calls(m, ev)]
        sleep_i = [i for i, m in enumerate(stmts) if any(
            isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "sleep"
            for c in ast.walk(m))]
        assert log_i and len(emit_i) == 1
        assert emit_i[0] > log_i[-1] > sleep_i[-1]


def test_no_researcher_field(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            assert "researcher" not in n.value.lower()
            assert "experimenter" not in n.value.lower()
        if isinstance(n, ast.keyword):
            assert n.arg not in ("researcher", "experimenter")


def test_defaults_are_inert(tree):
    assigns = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name)}
    assert getattr(assigns["TELEMETRY"].func, "id", None) == "_NoopTelemetry"
    assert isinstance(assigns["STREAMER"], ast.Constant) and assigns["STREAMER"].value is None


def test_remote_modules_imported_only_under_flag(tree):
    # No module-level import of the remote modules: with --remote off the script
    # must not even import them.
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    mods = {getattr(n, "module", None) for n in top}
    assert not ({"remote_config", "telemetry", "remote_stream", "stream_worker"} & mods)
