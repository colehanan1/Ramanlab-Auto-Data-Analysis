import sys, threading
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "PiCode"))
import phase_status as ps  # noqa: E402


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


def make(seconds=600):
    mono, wall = Clock(50.0), Clock(1_700_000_000.0)
    def tick(dt):
        mono.t += dt; wall.t += dt
    return ps.WaitControl(seconds, clock=mono, wall=wall), tick, wall


def test_counts_down_and_becomes_due():
    wc, tick, wall = make(600)
    assert wc.remaining() == 600 and not wc.due()
    tick(600)
    assert wc.due()
    assert wc.snapshot()["ends_at"] == pytest.approx(wall.t)


def test_add_extends_and_logs():
    wc, tick, wall = make(600)
    tick(100)
    snap = wc.add(300)
    assert wc.remaining() == 800
    assert snap["log"][-1] == {"at": wall.t, "action": "add", "seconds": 300.0}


@pytest.mark.parametrize("bad", [0, -5, 3601, float("nan")])
def test_add_rejects_bad_values(bad):
    wc, _, _ = make()
    with pytest.raises(ValueError):
        wc.add(bad)


def test_pause_freezes_remaining():
    wc, tick, _ = make(600)
    tick(100); wc.pause(); tick(10_000)
    assert wc.remaining() == 500 and not wc.due()
    snap = wc.snapshot()
    assert snap["paused"] and snap["ends_at"] is None and snap["paused_since"] is not None


def test_resume_restores_remaining():
    wc, tick, wall = make(600)
    tick(595); wc.pause(); tick(1200); snap = wc.resume()
    assert wc.remaining() == 5 and not wc.due()
    assert snap["log"][-1]["action"] == "resume" and snap["log"][-1]["seconds"] == 1200
    assert snap["ends_at"] == pytest.approx(wall.t + 5)
    tick(5)
    assert wc.due()


def test_pause_resume_idempotent():
    wc, tick, _ = make()
    wc.resume(); wc.pause(); wc.pause()
    assert [e["action"] for e in wc.snapshot()["log"]] == ["pause"]


def test_add_while_paused_counts():
    wc, tick, _ = make(600)
    wc.pause(); wc.add(300); tick(50); wc.resume()
    assert wc.remaining() == 900


def test_closed_raises():
    wc, _, _ = make()
    wc.close()
    assert not wc.is_open
    for call in (lambda: wc.add(60), wc.pause, wc.resume):
        with pytest.raises(ps.WaitClosed):
            call()


def test_planned_and_elapsed():
    wc, tick, _ = make(600)
    wc.add(300); tick(100); wc.pause(); tick(50); wc.resume(); tick(850)
    snap = wc.snapshot()
    assert snap["planned_s"] == 600 and snap["elapsed_s"] == pytest.approx(1000)
    assert wc.due()


def test_thread_safe_adds():
    wc, _, _ = make(0)
    ts = [threading.Thread(target=wc.add, args=(1,)) for _ in range(200)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert wc.remaining() == 200


import json


def call(wc, body, key="k", expected="k"):
    return ps.handle_wait_control(wc, expected, key, json.dumps(body).encode())


def test_no_key_503():
    wc, _, _ = make()
    assert call(wc, {"action": "pause"}, expected=None)[0] == 503


def test_bad_key_403():
    wc, _, _ = make()
    assert call(wc, {"action": "pause"}, key="nope")[0] == 403
    assert call(wc, {"action": "pause"}, key=None)[0] == 403


@pytest.mark.parametrize("body", [b"not json", b"[1]", b'{"action":"skip"}',
                                  b'{"action":"add"}', b'{"action":"add","seconds":"x"}',
                                  b'{"action":"add","seconds":-60}'])
def test_bad_body_400(body):
    wc, _, _ = make()
    assert ps.handle_wait_control(wc, "k", "k", body)[0] == 400


def test_closed_or_missing_409():
    assert call(None, {"action": "pause"})[0] == 409
    wc, _, _ = make(); wc.close()
    assert call(wc, {"action": "add", "seconds": 300})[0] == 409


def test_ok_paths():
    wc, _, _ = make(600)
    code, snap = call(wc, {"action": "add", "seconds": 300})
    assert code == 200 and snap["remaining_s"] == 900
    assert call(wc, {"action": "pause"})[1]["paused"] is True
    assert call(wc, {"action": "resume"})[1]["paused"] is False


def test_load_control_key(tmp_path):
    p = tmp_path / "k"
    assert ps.load_control_key(p) is None
    p.write_text("  \n"); assert ps.load_control_key(p) is None
    p.write_text("abc123\n"); assert ps.load_control_key(p) == "abc123"


def test_non_ascii_key_is_403_not_typeerror():
    wc, _, _ = make()
    assert call(wc, {"action": "pause"}, key="kéy")[0] == 403
    assert call(wc, {"action": "pause"}, key="k", expected="ké")[0] == 403


def test_load_control_key_bad_utf8(tmp_path):
    p = tmp_path / "key"
    p.write_bytes(b"\xff\xfe\x80")
    assert ps.load_control_key(p) is None
