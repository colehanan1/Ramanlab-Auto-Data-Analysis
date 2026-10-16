"""--n-flies: the run's fly count reaches the mount check and the trace plots."""
import pytest

from rigmon.traces.plot import legend_title
from test_rigmon_wait import hub, notifier  # noqa: F401  (fixtures)


class FakeMC:
    def __init__(self):
        self.calls = []

    def start(self, rig, batch, trigger, stop=None, expected_flies=None):
        self.calls.append((trigger, expected_flies))
        return 1


@pytest.mark.parametrize("session,expected", [
    ({"n_flies": 3}, 3), ({"n_flies": 1}, 1), ({}, 4), ({"n_flies": 0}, 4),
    ({"n_flies": 9}, 4), ({"n_flies": "x"}, 4), (None, 4)])
def test_expected_flies_from_status(hub, session, expected):
    hub.latest["RigA"] = None if session is None else {"session": session}
    assert hub.expected_flies("RigA") == expected


def test_manual_and_auto_checks_use_the_run_fly_count(hub):
    hub.mountcheck = FakeMC()
    hub.latest["RigA"] = {"session": {"n_flies": 2}, "run": {"batch_dir": "b"}}
    hub.start_mount_check("RigA")
    snap = {"phase": "WAITING", "started_at": 0.0, "ends_at": 5400.0, "server_time": 1900.0,
            "session": {"n_flies": 2}, "run": {"state": "running", "trials_done": 0, "batch_dir": "b"}}
    hub.latest["RigA"] = snap
    hub._maybe_mount_check(hub.cfg.rig("RigA"), snap)
    assert hub.mountcheck.calls == [("manual", 2), ("auto", 2)]


def test_trace_legend_counts_against_the_run():
    assert legend_title(3, 3) is None
    assert legend_title(2, 3) == "2 of 3 flies found"
    assert legend_title(4, 4) is None
