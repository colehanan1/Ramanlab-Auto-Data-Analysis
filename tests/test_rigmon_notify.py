from pathlib import Path

from rigmon.config import HubConfig, RigConfig
from rigmon.events import Event
from rigmon.notify import Notifier, describe, notify_send_argv
from rigmon.store import Store


def _notifier(tmp_path, topic="t"):
    cfg = HubConfig(rigs=(RigConfig("pi2", "Pi 2", "1.2.3.4"),), staging_root=tmp_path / "s",
                    data_root=tmp_path / "d", db_path=tmp_path / "db.sqlite", ntfy_topic=topic)
    store = Store(cfg.db_path)
    popups, pushes = [], []
    n = Notifier(cfg, store, runner=lambda argv, timeout, env=None: popups.append(argv) or (0, ""),
                 ntfy=lambda topic, title, body: pushes.append((topic, title)), spawn=lambda f: f())
    return n, store, popups, pushes


def trial_ev(replay=False):
    return Event("trial_finished", "pi2", 1.0, {"batch_dir": "b", "replay": replay,
                 "trial": {"label": "Training 3/6", "odor": "ACV"}})


def test_notify_send_argv():
    assert notify_send_argv("T", "B", "critical") == \
        ["notify-send", "-a", "Rig Monitor", "-u", "critical", "T", "B"]


def test_trial_popup(tmp_path):
    n, _, popups, pushes = _notifier(tmp_path)
    n.handle(trial_ev(), "Pi 2")
    assert popups == [notify_send_argv("Pi 2 · Training 3/6 done", "ACV", "normal")]
    assert pushes == []


def test_mute_silences_trials_but_not_failures(tmp_path):
    n, store, popups, _ = _notifier(tmp_path)
    store.set_muted("pi2", True)
    n.handle(trial_ev(), "Pi 2")
    assert popups == []
    n.handle(Event("session_failed", "pi2", 1.0, {"batch_dir": "b", "error": "boom"}), "Pi 2")
    assert popups[0][4] == "critical" and popups[0][-1] == "boom"


def test_replay_events_are_silent(tmp_path):
    n, _, popups, _ = _notifier(tmp_path)
    n.handle(trial_ev(replay=True), "Pi 2")
    assert popups == []


def test_offline_only_matters_mid_run(tmp_path):
    n, _, popups, pushes = _notifier(tmp_path)
    n.handle(Event("rig_offline", "pi2", 1.0, {"was_running": False}), "Pi 2")
    assert popups == [] and pushes == []
    n.handle(Event("rig_offline", "pi2", 1.0, {"was_running": True}), "Pi 2")
    assert len(popups) == 1 and pushes == [("t", "Pi 2 · offline")]


def test_alert_goes_to_desktop_and_ntfy_even_when_muted(tmp_path):
    n, store, popups, pushes = _notifier(tmp_path)
    store.set_muted("pi2", True)
    n.alert("Pi 2 finalize failed", "b: destination exists")
    assert len(popups) == 1 and pushes == [("t", "Pi 2 finalize failed")]


def test_no_topic_no_ntfy(tmp_path):
    n, _, _, pushes = _notifier(tmp_path, topic=None)
    n.alert("x", "y")
    assert pushes == []


def test_phase_changes_are_not_announced():
    assert describe(Event("phase_changed", "pi2", 1.0, {}), "Pi 2") is None


def test_describe_tolerates_bad_expected_end():
    t, b, _ = describe(Event("session_started", "pi2", 1.0,
                             {"batch_dir": "b", "expected_end_at": "garbage"}), "Pi 2")
    assert t == "Pi 2 · session started" and b == "b"


def test_mount_check_issues_is_critical_ntfy_and_ignores_mute(tmp_path):
    n, store, popups, pushes = _notifier(tmp_path)
    store.set_muted("pi2", True)
    n.mount_check("Pi 2", "issues", [f"m{i}" for i in range(9)])
    assert popups[0][4] == "critical" and popups[0][5] == "Pi 2 mount check: 9 issue(s)"
    assert popups[0][6].split("\n") == [f"m{i}" for i in range(6)]      # truncated to 6
    assert pushes == [("t", "Pi 2 mount check: 9 issue(s)")]


def test_mount_check_error_title(tmp_path):
    n, _, popups, pushes = _notifier(tmp_path)
    n.mount_check("Pi 2", "error", ["detector timed out"])
    assert popups[0][5] == "Pi 2 mount check failed" and popups[0][6] == "detector timed out"
    assert popups[0][4] == "critical" and pushes == [("t", "Pi 2 mount check failed")]
