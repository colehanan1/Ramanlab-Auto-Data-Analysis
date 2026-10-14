"""AST wiring checks for the rig-monitor hooks in the three rig scripts.

The scripts claim the camera/GPIO at import, so (like
test_status_page_wiring.py) these parse them instead of importing.
"""

import ast
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


@pytest.fixture(scope="module", params=SCRIPTS)
def tree(request) -> ast.Module:
    return ast.parse((PICODE / request.param).read_text(encoding="utf-8"))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _attr_calls(node, obj, attr):
    return [c for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == attr and isinstance(c.func.value, ast.Name)
            and c.func.value.id == obj]


def _name_calls(node, name):
    return [c for c in ast.walk(node)
            if isinstance(c, ast.Call) and getattr(c.func, "id", None) == name]


def _argparse_option(tree, flag):
    for c in ast.walk(tree):
        if (isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "add_argument"
                and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == flag):
            return {k.arg: ast.literal_eval(k.value) for k in c.keywords
                    if k.arg in ("default", "choices")}
    raise AssertionError(f"{flag} not defined")


def test_imports_rig_session(tree):
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module == "rig_session" for a in n.names}
    assert {"RunPlan", "RunTracker", "DELAY_STEP", "build_manifest", "hub_handoff"} <= imported


def test_push_mode_defaults_to_legacy(tree):
    opt = _argparse_option(tree, "--push-mode")
    assert opt["default"] == "legacy" and set(opt["choices"]) == {"legacy", "hub"}


def test_hub_flags_exist(tree):
    assert _argparse_option(tree, "--hub-url")["default"] == "http://100.94.160.60:8765"
    assert _argparse_option(tree, "--hub-timeout")["default"] == 1200


def test_run_tracker_at_module_level(tree):
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "RUN" for t in n.targets)]
    assert assigns and getattr(assigns[0].value.func, "id", None) == "RunTracker"


def test_record_cycle_marks_steps_and_delay(tree):
    record = _find_function(tree, "record_cycle")
    assert len(_attr_calls(record, "RUN", "mark")) >= 2
    assert "DELAY_STEP" in {n.id for n in ast.walk(record) if isinstance(n, ast.Name)}


def test_stop_recording_publishes_closed_trial(tree):
    record = _find_function(tree, "record_cycle")
    assert _name_calls(record, "_record_trial_closed")
    helper = _find_function(tree, "_record_trial_closed")
    assert _attr_calls(helper, "RUN", "trial_closed")
    assert "is_alive" in ast.unparse(helper)


def test_startup_countdown_begins_run_and_reminders(tree):
    countdown = _find_function(tree, "_startup_countdown")
    assert _attr_calls(countdown, "RUN", "begin")
    assert _attr_calls(countdown, "RUN", "start_message")
    assert "_run_reminder_loop" in ast.unparse(countdown)


def test_reminder_loop_sends_due_messages(tree):
    loop = _find_function(tree, "_run_reminder_loop")
    assert _attr_calls(loop, "RUN", "due_messages")
    assert _name_calls(loop, "ntfy_notify")


def test_main_hands_off_and_finishes(tree):
    src = ast.unparse(tree)
    assert _name_calls(tree, "hub_handoff")
    assert 'RUN.finish(\'complete\')' in src or 'RUN.finish("complete")' in src
    assert "RUN.finish('failed'" in src or 'RUN.finish("failed"' in src
    assert "args.push_mode == 'hub'" in src or 'args.push_mode == "hub"' in src


def test_hub_handoff_guarded_by_try_exception(tree):
    for t in ast.walk(tree):
        if isinstance(t, ast.Try) and any(
                isinstance(c, ast.Call) and getattr(c.func, "id", None) == "hub_handoff"
                for stmt in t.body for c in ast.walk(stmt)):
            if any(getattr(h.type, "id", None) == "Exception" for h in t.handlers):
                return
    raise AssertionError("hub_handoff not inside try/except Exception")


def test_start_notice_sent_from_thread(tree):
    countdown = _find_function(tree, "_startup_countdown")
    targets = [k.value.id for c in ast.walk(countdown)
               if isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "Thread"
               for k in c.keywords if k.arg == "target" and isinstance(k.value, ast.Name)]
    assert "ntfy_notify" in targets


def test_reminder_stop_set_after_finish(tree):
    assert len(_attr_calls(tree, "_run_reminder_stop", "set")) >= 2
