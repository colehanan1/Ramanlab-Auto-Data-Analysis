"""Public (Tailscale Funnel) entrance: a separate localhost port where every
request needs HTTP basic auth and the Pi hand-off API does not exist."""
import base64
import dataclasses

import pytest
import yaml
from fastapi.testclient import TestClient

from rigmon.app import create_app
from rigmon.config import PublicConfig, load_config

from test_rigmon_wait import hub, notifier  # noqa: F401  (fixtures)

PUB = PublicConfig(port=8766, user="u", password="p w!", hosts=("box.example.ts.net",))


def _auth(user, pw):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


@pytest.fixture
def pub(hub):
    hub.cfg = dataclasses.replace(hub.cfg, public=PUB)
    hub.wait_control = lambda *a, **k: (200, {})
    app = create_app(hub, start_polling=False)
    return (TestClient(app, base_url="https://box.example.ts.net:8766"),
            TestClient(app, base_url="http://localhost:8765"))


def test_public_port_views_without_login(pub):
    c, _ = pub
    assert c.get("/").status_code == 200
    assert c.get("/api/state").status_code == 200
    assert c.get("/api/whoami").json() == {"public": True, "can_edit": False}


def test_public_port_writes_need_admin(pub):
    c, _ = pub
    o = {"Origin": "https://box.example.ts.net:8766"}
    for path, js in (("/api/rig/RigA/wait", {"action": "pause"}),
                     ("/api/mute/RigA", {"muted": True}), ("/api/mountcheck/RigA", None)):
        r = c.post(path, json=js, headers=o)
        assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic"), path
        assert c.post(path, json=js, headers=o | _auth("u", "wrong")).status_code == 401, path
    h = o | _auth("u", "p w!")
    assert c.post("/api/rig/RigA/wait", json={"action": "pause"}, headers=h).status_code == 200


def test_login_challenges_then_redirects(pub):
    c, _ = pub
    r = c.get("/login", follow_redirects=False)
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic")
    r = c.get("/login", headers=_auth("u", "p w!"), follow_redirects=False)
    assert r.status_code in (302, 303, 307) and r.headers["location"] == "/"
    assert c.get("/api/whoami", headers=_auth("u", "p w!")).json() == {"public": True, "can_edit": True}


def test_public_port_hides_pi_handoff(pub):
    c, _ = pub
    for m, path in (("post", "/api/sessions/RigA/b/abort"), ("get", "/api/sessions/RigA/b/status"),
                    ("post", "/api/sessions/RigA/b/complete")):
        assert getattr(c, m)(path, headers=_auth("u", "p w!")).status_code == 404, path


def test_tailnet_port_unchanged(pub):
    _, t = pub
    assert t.get("/api/whoami").json() == {"public": False, "can_edit": True}
    assert t.post("/api/rig/RigA/wait", json={"action": "pause"}).status_code == 200


def test_malformed_auth_header(pub):
    c, _ = pub
    for v in ("Basic !!!", "Bearer x", "Basic " + base64.b64encode(b"nocolon").decode(),
              "Basic " + base64.b64encode("u:p w!\u00e9".encode()).decode()):
        r = c.post("/api/mute/RigA", json={"muted": True},
                   headers={"Authorization": v, "Origin": "https://box.example.ts.net:8766"})
        assert r.status_code == 401


def test_public_config_loads_and_hides_password(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({
        "rigs": [{"name": "A", "label": "Pi 1", "ip": "local"}],
        "staging_root": str(tmp_path / "s"), "data_root": str(tmp_path / "d"),
        "db_path": str(tmp_path / "x.db"),
        "public": {"port": 8766, "user": "u", "password": "sekrit", "hosts": ["h.ts.net"]}}))
    cfg = load_config(p)
    assert cfg.public.port == 8766 and cfg.public.hosts == ("h.ts.net",)
    assert "sekrit" not in repr(cfg)


def test_public_config_requires_password(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({
        "rigs": [{"name": "A", "label": "Pi 1", "ip": "local"}],
        "staging_root": str(tmp_path / "s"), "data_root": str(tmp_path / "d"),
        "db_path": str(tmp_path / "x.db"), "public": {"port": 8766, "user": "u", "password": ""}}))
    with pytest.raises(ValueError):
        load_config(p)


def test_public_entrance_listens_on_loopback_only(hub, monkeypatch):
    import rigmon.__main__ as m
    seen = []
    monkeypatch.setattr(m, "bind_socket", lambda h, p: seen.append((h, p)) or (h, p))
    cfg = dataclasses.replace(hub.cfg, bind=("127.0.0.1", "100.9.9.9"), port=8765, public=PUB)
    m.listen_sockets(cfg)
    assert seen == [("127.0.0.1", 8765), ("100.9.9.9", 8765), ("127.0.0.1", 8766)]
    seen.clear()
    m.listen_sockets(dataclasses.replace(cfg, public=None))
    assert ("127.0.0.1", 8766) not in seen
