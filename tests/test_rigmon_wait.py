import json

import httpx
import pytest
from fastapi.testclient import TestClient

from rigmon.app import create_app
from rigmon.config import HubConfig, RigConfig
from rigmon.hub import Hub
from rigmon.store import Store
from rigmon.transfer import TransferWorkers

A = RigConfig("RigA", "Pi A", "100.1.1.1", control_key="secret")
B = RigConfig("RigB", "Pi B", "100.1.1.2")


class Notifier:
    def __init__(self):
        self.alerts = []

    def alert(self, t, b):
        self.alerts.append((t, b))

    def handle(self, *a):
        pass


@pytest.fixture
def notifier():
    return Notifier()


@pytest.fixture
def hub(tmp_path, notifier):
    cfg = HubConfig(rigs=(A, B), staging_root=tmp_path / "s", data_root=tmp_path / "d",
                    db_path=tmp_path / "db.sqlite")
    return Hub(cfg, Store(cfg.db_path), notifier, None, None, TransferWorkers(threaded=False),
               clock=lambda: 100.0)


def test_forward_add_ok_notifies(hub, notifier):
    calls = []

    def post(url, json, headers, timeout):
        calls.append((url, json, headers))
        return 200, {"ends_at": 1_700_003_600.0, "paused": False}
    code, payload = hub.wait_control("RigA", "add", 900, post=post)
    assert code == 200
    url, body, headers = calls[0]
    assert url.endswith(":8008/control/wait") and body == {"action": "add", "seconds": 900}
    assert headers == {"X-Rig-Key": "secret"}
    assert notifier.alerts[-1][0] == "Pi A +15 min"
    assert "Training 1 at" in notifier.alerts[-1][1]


def test_finish_time_shifts_with_ends_at(hub, notifier):
    hub.latest["RigA"] = {"ends_at": 1_700_000_000.0, "run": {"expected_end_at": 1_700_010_000.0}}
    hub.wait_control("RigA", "add", 300, post=lambda *a, **k: (200, {"ends_at": 1_700_000_300.0}))
    assert "finish ~" in notifier.alerts[-1][1]


def test_pause_resume_titles(hub, notifier):
    hub.wait_control("RigA", "pause", post=lambda *a, **k: (200, {}))
    hub.wait_control("RigA", "resume", post=lambda *a, **k: (200, {"ends_at": 1_700_000_300.0}))
    assert [a[0] for a in notifier.alerts] == ["Pi A wait paused", "Pi A wait resumed"]
    assert notifier.alerts[0][1] == "Reminders every 5 min until resumed"


def test_forward_409_no_notify(hub, notifier):
    code, payload = hub.wait_control("RigA", "pause", post=lambda *a, **k: (409, {"error": "wait over"}))
    assert (code, payload) == (409, {"error": "wait over"}) and notifier.alerts == []


def test_unreachable_502(hub):
    def post(*a, **k):
        raise httpx.ConnectError("x")
    assert hub.wait_control("RigA", "pause", post=post)[0] == 502


def test_no_key_409(hub):
    assert hub.wait_control("RigB", "pause", post=lambda *a, **k: (200, {}))[0] == 409


def test_pause_reminder_every_300(hub, notifier):
    snap = {"phase": "WAITING", "run": {"state": "running", "trials_done": 0,
                                        "wait": {"paused": True, "paused_since": 1000.0}}}
    for now in (1100, 1299):
        hub._maybe_pause_reminder(A, snap, now)
    assert notifier.alerts == []
    for now in (1300, 1400, 1600):
        hub._maybe_pause_reminder(A, snap, now)
    assert [a[0] for a in notifier.alerts] == ["Pi A still paused", "Pi A still paused"]
    assert "10 min" in notifier.alerts[-1][1]


def test_pause_reminder_uses_pi_paused_since(hub, notifier):
    snap = {"phase": "WAITING", "run": {"wait": {"paused": True, "paused_since": 0.0}}}
    hub._maybe_pause_reminder(A, snap, 1020)
    assert len(notifier.alerts) == 1 and "17 min" in notifier.alerts[0][1]


def test_new_pause_starts_over(hub, notifier):
    mk = lambda s: {"run": {"wait": {"paused": True, "paused_since": s}}}
    hub._maybe_pause_reminder(A, mk(0.0), 300)
    hub._maybe_pause_reminder(A, mk(1000.0), 1300)
    assert len(notifier.alerts) == 2


def test_no_reminder_when_not_paused(hub, notifier):
    hub._maybe_pause_reminder(A, {"run": {"wait": None}}, 99999)
    assert notifier.alerts == []


def test_ingest_triggers_reminder(hub, notifier):
    hub.ingest("RigA", {"phase": "WAITING", "run": {"wait": {"paused": True, "paused_since": 0.0}}}, 400.0)
    assert len(notifier.alerts) == 1


def test_state_exposes_control_flag_not_key(hub):
    st = hub.state()
    assert st["rigs"]["RigA"]["control"] is True and st["rigs"]["RigB"]["control"] is False
    assert "secret" not in json.dumps(st, default=str) and "secret" not in repr(A)


@pytest.fixture
def client(hub):
    return TestClient(create_app(hub, start_polling=False))


def test_api_validation(client):
    assert client.post("/api/rig/RigA/wait", json={"action": "add", "seconds": 120}).status_code == 422
    assert client.post("/api/rig/RigA/wait", json={"action": "add"}).status_code == 422
    assert client.post("/api/rig/RigA/wait", json={"action": "nope"}).status_code == 422
    assert client.post("/api/rig/nope/wait", json={"action": "pause"}).status_code == 404


def test_api_relays_status_and_payload(hub, client):
    hub.wait_control = lambda *a, **k: (409, {"error": "wait over"})
    r = client.post("/api/rig/RigA/wait", json={"action": "pause"})
    assert r.status_code == 409 and r.json()["detail"] == "wait over"
    hub.wait_control = lambda *a, **k: (200, {"paused": True})
    r = client.post("/api/rig/RigA/wait", json={"action": "add", "seconds": 300})
    assert r.status_code == 200 and r.json() == {"paused": True}
