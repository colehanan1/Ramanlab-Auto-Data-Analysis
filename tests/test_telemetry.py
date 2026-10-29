"""Unit tests for PiCode/telemetry.py — non-blocking MQTT telemetry for the rig
scripts. paho-mqtt is never imported here: every test injects a FakeClient, and
the "paho missing" cases poison sys.modules instead."""

import json
import logging
import sys
import threading
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "PiCode"))

import remote_config as rc  # noqa: E402
import telemetry as tm  # noqa: E402

ALLOWED = {"online", "session_id", "trial", "phase", "odor", "opto",
           "experiment_name"}
ENVELOPE = {"rig_id", "ts", "monotonic_ns", "session_id", "seq"}


def make_cfg(**kw):
    kw.setdefault("rig_id", "rig2")
    kw.setdefault("server_ip", "100.1.2.3")
    return rc.RemoteConfig(**kw)


class FakeClient:
    """Records every call; publish result and failures are configurable."""

    def __init__(self, rc_value=0, publish_raises=None, loop_stop_block=None):
        self.rc_value = rc_value
        self.publish_raises = publish_raises
        self.loop_stop_block = loop_stop_block
        self.published = []  # (topic, payload dict, qos, retain)
        self.calls = []  # (method, args, kwargs)
        self.will = None
        self.on_connect = None
        self.on_disconnect = None
        self.cond = threading.Condition()

    def _rec(self, name, *a, **kw):
        self.calls.append((name, a, kw))

    def username_pw_set(self, *a, **kw):
        self._rec("username_pw_set", *a, **kw)

    def will_set(self, topic, payload=None, qos=0, retain=False):
        self.will = (topic, json.loads(payload), qos, retain)
        self._rec("will_set", topic, payload, qos=qos, retain=retain)

    def reconnect_delay_set(self, **kw):
        self._rec("reconnect_delay_set", **kw)

    def max_queued_messages_set(self, n):
        self._rec("max_queued_messages_set", n)

    def connect_async(self, *a, **kw):
        self._rec("connect_async", *a, **kw)

    def loop_start(self):
        self._rec("loop_start")

    def loop_stop(self):
        self._rec("loop_stop")
        if self.loop_stop_block is not None:
            self.loop_stop_block.wait(10)

    def disconnect(self):
        self._rec("disconnect")

    def publish(self, topic, payload, qos=0, retain=False):
        with self.cond:
            self.calls.append(("publish", (topic,), {}))
            if self.publish_raises is not None:
                self.cond.notify_all()
                raise self.publish_raises
            self.published.append((topic, json.loads(payload), qos, retain))
            self.cond.notify_all()
        return types.SimpleNamespace(rc=self.rc_value)

    def called(self, name):
        return [c for c in self.calls if c[0] == name]

    def on(self, suffix):
        return [p for p in self.published if p[0].endswith(suffix)]


def wait_until(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.002)
    return pred()


def started(client=None, cfg=None, **kw):
    """A started Telemetry on a FakeClient; sensors effectively off by default."""
    client = client or FakeClient()
    kw.setdefault("sensor_period_s", 1000.0)
    t = tm.Telemetry(cfg or make_cfg(), client_factory=lambda: client, **kw)
    t.start()
    return t, client


# ------------------------------------------------------------------ emit
def test_emit_on_full_queue_never_blocks_and_counts_dropped():
    t = tm.Telemetry(make_cfg(), queue_size=5)  # never started -> nobody drains
    n = 5000
    t0 = time.perf_counter()
    for i in range(n):
        t.emit("trial_start", trial=i)
    per_call = (time.perf_counter() - t0) / n
    assert per_call < 0.001
    st = t.stats()
    assert st["queued"] == 5
    assert st["dropped"] == n - 5


def test_emit_drop_warning_is_rate_limited(caplog):
    t = tm.Telemetry(make_cfg(), queue_size=1, clock=lambda: 5_000_000_000)
    with caplog.at_level(logging.WARNING, logger="telemetry"):
        for _ in range(50):
            t.emit("x")
    drops = [r for r in caplog.records if "drop" in r.getMessage().lower()]
    assert len(drops) == 1


def test_emit_never_raises_on_odd_fields():
    t, client = started()
    t.emit("odd", obj=object(), nan=float("nan"), inf=float("inf"),
           nested={"a": {1, 2}})
    assert wait_until(lambda: client.on("/events"))
    msg = client.on("/events")[0][1]
    assert isinstance(msg["obj"], str)
    assert msg["nan"] is None and msg["inf"] is None
    t.close()


def test_emit_fields_cannot_clobber_envelope():
    t, client = started()
    t.emit("e", seq=-5, rig_id="evil", ts="x")
    assert wait_until(lambda: client.on("/events"))
    msg = client.on("/events")[0][1]
    assert msg["seq"] >= 0 and msg["rig_id"] == "rig2" and msg["ts"] != "x"
    t.close()


def test_envelope_fields_and_utc_iso_ts():
    t, client = started()
    t.start_session("sess-1")
    t.emit("trial_start", trial=3)
    assert wait_until(lambda: client.on("/events"))
    topic, msg, qos, retain = client.on("/events")[0]
    assert topic == "lab/rig2/events" and qos == 1 and retain is False
    assert msg["event"] == "trial_start" and msg["trial"] == 3
    assert ENVELOPE <= set(msg)
    assert msg["rig_id"] == "rig2" and msg["session_id"] == "sess-1"
    assert isinstance(msg["monotonic_ns"], int)
    ts = datetime.fromisoformat(msg["ts"])
    assert ts.utcoffset() == timedelta(0) and msg["ts"].endswith("+00:00")
    t.close()


# ------------------------------------------------------------------- seq
def test_seq_strictly_increasing_across_events_and_status():
    t, client = started()
    t.emit("a")
    t.update_status(trial=1)
    t.emit("b")
    assert wait_until(lambda: len(client.on("/events")) == 2 and client.on("/status"))
    seqs = [p[1]["seq"] for p in client.published if not p[0].endswith("/sensors")]
    assert len(seqs) == len(set(seqs))
    assert seqs == sorted(seqs)
    t.close()


def test_seq_shared_with_sensors():
    t, client = started(sensor_period_s=0.02)
    assert wait_until(lambda: len(client.on("/sensors")) >= 3)
    t.emit("a")
    assert wait_until(lambda: client.on("/events"))
    t.close()
    seqs = [p[1]["seq"] for p in client.published]
    assert len(seqs) == len(set(seqs))
    sensor_seqs = [p[1]["seq"] for p in client.on("/sensors")]
    assert sensor_seqs == sorted(sensor_seqs)
    assert len(sensor_seqs) >= 3


# ---------------------------------------------------------------- status
def test_will_is_retained_offline_status():
    t, client = started()
    topic, payload, qos, retain = client.will
    assert topic == "lab/rig2/status" and qos == 1 and retain is True
    assert payload["online"] is False and payload["rig_id"] == "rig2"
    t.close()


def test_on_connect_publishes_retained_online_status():
    t, client = started()
    assert client.on_connect is not None
    client.on_connect(client, None, {}, 0, None)  # VERSION2 signature
    topic, msg, qos, retain = client.on("/status")[-1]
    assert topic == "lab/rig2/status" and retain is True and qos == 1
    assert msg["online"] is True
    t.close()


def test_on_connect_failure_publishes_nothing():
    t, client = started()
    failed = types.SimpleNamespace(is_failure=True)
    client.on_connect(client, None, {}, failed, None)
    assert client.on("/status") == []
    t.close()


def test_status_has_exactly_allowed_keys_and_no_researcher():
    t, client = started()
    t.start_session("s1", experiment_name="exp", researcher="bob")
    t.update_status(trial=2, phase="odor", odor="IAA", opto=True,
                    researcher="alice", bogus=1)
    assert wait_until(lambda: any(p[1].get("trial") == 2 for p in client.on("/status")))
    t.close()
    for topic, msg, qos, retain in client.on("/status"):
        assert retain is True
        assert set(msg) == ALLOWED | ENVELOPE
        assert "researcher" not in msg
    last = [p for p in client.on("/status") if p[1].get("trial") == 2][-1][1]
    assert last["phase"] == "odor" and last["odor"] == "IAA"
    assert last["opto"] is True and last["experiment_name"] == "exp"
    assert last["session_id"] == "s1"
    for _, msg, _, _ in client.published:
        assert "researcher" not in json.dumps(msg)


def test_status_updates_coalesce_to_latest():
    client = FakeClient()
    t = tm.Telemetry(make_cfg(), client_factory=lambda: client, sensor_period_s=1000)
    # Not started: nothing drains, so these pile up as one pending status.
    for i in range(100):
        t.update_status(trial=i)
    assert t.stats()["queued"] <= 1
    t.start()
    assert wait_until(lambda: client.on("/status"))
    assert wait_until(lambda: client.on("/status")[-1][1]["trial"] == 99)
    t.close()


# --------------------------------------------------------------- sensors
def test_sensor_ticker_publishes_cache_with_bme_age():
    now = 10_000_000_000
    t, client = started(sensor_period_s=0.02, clock=lambda: now)
    t.update_sensors(bme={"temperature_c": 22.5, "humidity_pct": 40.0,
                          "pressure_hpa": 1000.0, "gas_ohms": 1234,
                          "read_monotonic_ns": now - 3_000_000_000},
                     light=True, valves={"A": 1})
    assert wait_until(lambda: any("bme" in p[1] for p in client.on("/sensors")))
    msg = [p for p in client.on("/sensors") if "bme" in p[1]][0][1]
    assert msg["bme"]["temperature_c"] == 22.5
    assert msg["bme_age_s"] == pytest.approx(3.0)
    assert msg["light"] is True and msg["valves"] == {"A": 1}
    assert ENVELOPE <= set(msg)
    t.close()


def test_sensor_age_none_when_never_read():
    t, client = started(sensor_period_s=0.02)
    assert wait_until(lambda: client.on("/sensors"))
    assert client.on("/sensors")[0][1]["bme_age_s"] is None
    t.close()


def test_update_sensors_merges_and_snapshots():
    t, client = started(sensor_period_s=0.02)
    valves = {"A": 1}
    t.update_sensors(valves=valves)
    valves["A"] = 99  # caller mutating afterwards must not leak into a tick
    t.update_sensors(light=False)
    assert wait_until(lambda: any("light" in p[1] and "valves" in p[1]
                                  for p in client.on("/sensors")))
    msg = [p for p in client.on("/sensors") if "valves" in p[1]][0][1]
    assert msg["valves"] == {"A": 1} and msg["light"] is False
    t.close()


# ---------------------------------------------------------- failure paths
@pytest.mark.parametrize("client", [
    FakeClient(rc_value=4),
    FakeClient(publish_raises=RuntimeError("boom")),
])
def test_publish_failure_is_counted_not_raised(client):
    t, _ = started(client)
    t.emit("a")
    t.emit("b")
    assert wait_until(lambda: t.stats()["dropped"] >= 2)
    assert t.stats()["sent"] == 0
    t.close()


def test_publish_success_counts_sent():
    t, client = started()
    t.emit("a")
    assert wait_until(lambda: t.stats()["sent"] >= 1)
    t.close()


def test_client_factory_raising_is_a_no_op(caplog):
    def boom():
        raise RuntimeError("no broker lib")

    t = tm.Telemetry(make_cfg(rtsp_pass="SECRETRTSP", mqtt_pass="SECRETMQTT"),
                     client_factory=boom)
    with caplog.at_level(logging.WARNING, logger="telemetry"):
        t.start()
        t.start()  # a second start must not warn again
        for i in range(20):
            t.emit("a", i=i)
            t.update_status(trial=i)
            t.update_sensors(light=True)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "SECRET" not in caplog.text
    assert t.stats() == {"sent": 0, "dropped": 0, "queued": 0}
    t.close()


def test_client_factory_importerror_is_a_no_op():
    def missing():
        raise ImportError("No module named paho")

    t = tm.Telemetry(make_cfg(), client_factory=missing)
    t.start()
    t.emit("a")
    assert t.stats()["queued"] == 0
    t.close()


def test_default_factory_without_paho_is_a_no_op(monkeypatch):
    for name in ("paho", "paho.mqtt", "paho.mqtt.client"):
        monkeypatch.setitem(sys.modules, name, None)  # import -> ImportError
    t = tm.Telemetry(make_cfg())
    t.start()
    t.emit("a")
    t.update_status(trial=1)
    assert t.stats()["queued"] == 0
    t.close()


def test_client_setup_raising_midway_is_a_no_op():
    class Bad(FakeClient):
        def connect_async(self, *a, **kw):
            raise OSError("nope")

    t, _ = started(Bad())
    t.emit("a")
    assert t.stats()["queued"] == 0
    t.close()


def test_default_factory_builds_paho_v2_client(monkeypatch):
    made = {}

    class CallbackAPIVersion:
        VERSION2 = "v2"

    class Client(FakeClient):
        def __init__(self, **kw):
            super().__init__()
            made.update(kw)

    pkg = types.ModuleType("paho")
    sub = types.ModuleType("paho.mqtt")
    mod = types.ModuleType("paho.mqtt.client")
    mod.Client, mod.CallbackAPIVersion = Client, CallbackAPIVersion
    pkg.mqtt, sub.client = sub, mod
    for name, m in (("paho", pkg), ("paho.mqtt", sub), ("paho.mqtt.client", mod)):
        monkeypatch.setitem(sys.modules, name, m)
    t = tm.Telemetry(make_cfg(), sensor_period_s=1000)
    t.start()
    t.close()
    assert made == {"callback_api_version": "v2", "client_id": "rig2-rig",
                    "clean_session": True}


# ------------------------------------------------------------ client setup
def test_client_setup_calls():
    cfg = make_cfg(mqtt_user="rig", mqtt_pass="pw", mqtt_port=1884)
    t, client = started(cfg=cfg)
    assert client.called("username_pw_set")[0][1] == ("rig", "pw")
    assert client.called("reconnect_delay_set")[0][2] == {"min_delay": 1, "max_delay": 60}
    assert client.called("max_queued_messages_set")[0][1] == (1000,)
    ca = client.called("connect_async")[0]
    assert ca[1] == ("100.1.2.3", 1884) and ca[2] == {"keepalive": 30}
    assert client.called("loop_start")
    t.close()


def test_no_username_when_user_unset():
    t, client = started()
    assert not client.called("username_pw_set")
    t.close()


# --------------------------------------------------------------- factory
def test_make_telemetry_null_when_disabled_or_incomplete():
    for cfg in (None, make_cfg(telemetry_enabled=False), make_cfg(server_ip="")):
        t = tm.make_telemetry(cfg, client_factory=lambda: FakeClient())
        assert isinstance(t, tm.NullTelemetry) and t.enabled is False


def test_make_telemetry_returns_started_telemetry():
    client = FakeClient()
    t = tm.make_telemetry(make_cfg(), client_factory=lambda: client,
                          sensor_period_s=1000)
    assert isinstance(t, tm.Telemetry) and t.enabled is True
    assert client.called("loop_start")
    t.close()


def test_make_telemetry_never_raises():
    t = tm.make_telemetry(make_cfg(), client_factory=lambda: 1 / 0)
    t.emit("a")
    t.close()


def test_null_telemetry_is_inert():
    n = tm.NullTelemetry()
    assert n.enabled is False
    n.start()
    n.emit("a", x=1)
    n.start_session("s", trial=1)
    n.update_status(trial=1)
    n.update_sensors(light=True)
    assert n.stats() == {"sent": 0, "dropped": 0, "queued": 0}
    n.close()
    n.close()


# ----------------------------------------------------------------- close
def test_close_idempotent_bounded_and_publishes_offline():
    t, client = started()
    t.emit("a")
    t0 = time.monotonic()
    t.close()
    t.close()
    assert time.monotonic() - t0 < 3.0
    assert len(client.called("loop_stop")) == 1
    assert len(client.called("disconnect")) == 1
    last_status = client.on("/status")[-1]
    assert last_status[1]["online"] is False and last_status[3] is True
    # publisher drained the queued event before shutting down
    assert client.on("/events")
    # disconnect precedes loop_stop so the offline publish gets flushed
    names = [c[0] for c in client.calls]
    assert names.index("disconnect") < names.index("loop_stop")


def test_close_is_bounded_even_if_loop_stop_hangs():
    release = threading.Event()
    t, client = started(FakeClient(loop_stop_block=release))
    t0 = time.monotonic()
    t.close(timeout=0.5)
    elapsed = time.monotonic() - t0
    release.set()
    assert elapsed < 3.0


def test_emit_after_close_is_ignored():
    t, client = started()
    t.close()
    before = len(client.published)
    t.emit("late")
    t.update_status(trial=9)
    time.sleep(0.05)
    assert len(client.published) == before
    assert t.stats()["dropped"] == 0


def test_close_without_start_is_safe():
    t = tm.Telemetry(make_cfg())
    t.emit("a")
    t.close()
    t.close()
