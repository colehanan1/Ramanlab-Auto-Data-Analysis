"""Tests for scripts/tools/remote_connectivity_check.py.

The script is the desktop-side gate before any Pi is touched: it publishes a test
stream to the remote RTSP server and reads it back with ffprobe, and round-trips a
retained message and a last-will through the MQTT broker.  No test here touches the
network: command builders and the ffprobe verdict are pure, ffmpeg/ffprobe are stub
shell scripts, and paho-mqtt is replaced by an in-memory broker.

Passwords must never reach stdout/stderr; that is asserted on everything printed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tools" / "remote_connectivity_check.py"

spec = importlib.util.spec_from_file_location("remote_connectivity_check", SCRIPT)
rcc = importlib.util.module_from_spec(spec)
sys.modules["remote_connectivity_check"] = rcc
spec.loader.exec_module(rcc)

PW = "s3cret:p@ss/w0rd!"            # hostile characters: ':' '@' '/' must stay hidden
MQTT_PW = "mq-secret-99"
ENV_TEXT = f"""\
# remote server
SERVER_TAILNET_IP=100.64.0.9
RIG_ID=rigtest
RTSP_USER=rigpub
RIG_PASS="{PW}"
export MQTT_USER='telem'
MQTT_PASS={MQTT_PW}
"""


def make_cfg(**kw) -> "rcc.Config":
    env = rcc.parse_env_text(ENV_TEXT)
    env.update(kw)
    return rcc.config_from_env(env)


# ── env parsing ──────────────────────────────────────────────────────────────

def test_parse_env_text_handles_comments_quotes_export():
    env = rcc.parse_env_text(ENV_TEXT)
    assert env["SERVER_TAILNET_IP"] == "100.64.0.9"
    assert env["RIG_PASS"] == PW                    # quotes stripped
    assert env["MQTT_USER"] == "telem"              # 'export' and single quotes
    assert "# remote server" not in env
    assert rcc.parse_env_text("A=b=c\nbroken line\n\n")["A"] == "b=c"


def test_config_defaults():
    cfg = rcc.config_from_env({"SERVER_TAILNET_IP": "1.2.3.4", "RIG_PASS": "x"})
    assert (cfg.rig_id, cfg.rtsp_port, cfg.rtsp_user, cfg.mqtt_port) == \
        ("rigtest", 8554, "rigpub", 1883)
    assert (cfg.read_user, cfg.read_pass) == ("rigpub", "x")   # default = publish creds


def test_config_read_credentials_override():
    cfg = make_cfg(RTSP_READ_USER="viewer", RTSP_READ_PASS="v-pass")
    assert (cfg.read_user, cfg.read_pass) == ("viewer", "v-pass")
    assert cfg.rtsp_user == "rigpub" and cfg.rig_pass == PW


def test_config_requires_server_ip():
    with pytest.raises(rcc.SetupError):
        rcc.config_from_env({"RIG_PASS": "x"})


def test_config_bad_port():
    with pytest.raises(rcc.SetupError):
        rcc.config_from_env({"SERVER_TAILNET_IP": "1.2.3.4", "RTSP_PORT": "eighty"})


# ── command builders ─────────────────────────────────────────────────────────

def test_publish_command_has_the_exact_rig_encoder_flags():
    cfg = make_cfg(RIG_PASS="pw")
    cmd = rcc.build_publish_cmd(cfg, "/usr/bin/ffmpeg", 12)
    assert cmd == [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
        "-f", "lavfi", "-i", "testsrc2=size=540x540:rate=40",
        "-t", "12",
        "-c:v", "libx264", "-profile:v", "main", "-bf", "0",
        "-g", "40", "-keyint_min", "40", "-sc_threshold", "0",
        "-b:v", "1500k", "-maxrate", "1500k", "-bufsize", "3000k",
        "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-tune", "zerolatency",
        "-an", "-f", "rtsp", "-rtsp_transport", "tcp",
        "rtsp://rigpub:pw@100.64.0.9:8554/rigtest-selftest",
    ]


def test_probe_command_has_the_exact_flags_and_read_credentials():
    cfg = make_cfg(RIG_PASS="pw", RTSP_READ_USER="viewer", RTSP_READ_PASS="vp")
    assert rcc.build_probe_cmd(cfg, "/usr/bin/ffprobe") == [
        "/usr/bin/ffprobe", "-rtsp_transport", "tcp", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,profile,width,height,pix_fmt,has_b_frames,avg_frame_rate",
        "-show_entries", "frame=key_frame,pts_time",
        "-read_intervals", "%+4", "-of", "json",
        "rtsp://viewer:vp@100.64.0.9:8554/rigtest-selftest",
    ]


def test_url_percent_encodes_hostile_password():
    url = rcc.rtsp_url(make_cfg(), "rigpub", PW)
    assert url.count("@") == 1                      # the password's own '@' is encoded
    assert url.startswith("rtsp://rigpub:s3cret%3Ap%40ss%2Fw0rd%21@100.64.0.9:8554/")


def test_custom_port_and_rig_id_reach_the_url():
    cfg = make_cfg(RTSP_PORT="9554", RIG_ID="rig3")
    assert rcc.build_publish_cmd(cfg, "ffmpeg")[-1].endswith("@100.64.0.9:9554/rig3-selftest")


def test_builders_return_lists_never_shell_strings():
    cfg = make_cfg()
    assert isinstance(rcc.build_publish_cmd(cfg, "ffmpeg"), list)
    assert isinstance(rcc.build_probe_cmd(cfg, "ffprobe"), list)


# ── redaction ────────────────────────────────────────────────────────────────

def test_redact_hides_password_in_commands():
    cfg = make_cfg()
    for cmd in (rcc.build_publish_cmd(cfg, "ffmpeg"), rcc.build_probe_cmd(cfg, "ffprobe")):
        text = rcc.format_cmd(cmd, cfg.secrets)
        assert PW not in text
        assert "s3cret" not in text and "w0rd" not in text      # neither raw nor encoded
        assert "rtsp://rigpub:***@100.64.0.9:8554/rigtest-selftest" in text
        red = rcc.redact(cmd, cfg.secrets)
        assert isinstance(red, list)
        assert all(PW not in a and "s3cret" not in a for a in red)


def test_redact_masks_any_url_password_even_when_not_in_secrets():
    assert rcc.redact("rtsp://u:unknown@h:1/x") == "rtsp://u:***@h:1/x"


def test_redact_replaces_literal_secret_in_free_text():
    out = rcc.redact(f"auth failed for password {MQTT_PW} on broker", [MQTT_PW])
    assert MQTT_PW not in out and "***" in out


def test_redact_leaves_clean_text_alone():
    assert rcc.redact("rtsp://100.64.0.9:8554/x", ["pw"]) == "rtsp://100.64.0.9:8554/x"


# ── ffprobe verdict ──────────────────────────────────────────────────────────

def probe_json(*, codec="h264", profile="Main", w=540, h=540, pix="yuv420p", bf=0,
               fps="40/1", gop=40, n_frames=160):
    frames = [{"key_frame": 1 if i % gop == 0 else 0, "pts_time": f"{i / 40:.6f}"}
              for i in range(n_frames)]
    return {"streams": [{"codec_name": codec, "profile": profile, "width": w,
                         "height": h, "pix_fmt": pix, "has_b_frames": bf,
                         "avg_frame_rate": fps}],
            "frames": frames}


def failed(checks):
    return sorted(c.name for c in checks if not c.ok)


def test_good_stream_passes_every_check():
    checks = rcc.evaluate_probe(probe_json())
    assert failed(checks) == []
    assert len(checks) >= 7


@pytest.mark.parametrize("kwargs, expected", [
    ({"profile": "High"}, ["profile Main"]),
    ({"bf": 2}, ["no B-frames (has_b_frames 0)"]),
    ({"w": 640, "h": 480}, ["size 540x540"]),
    ({"gop": 60, "n_frames": 200}, ["keyframe interval 40 frames"]),
    ({"codec": "hevc"}, ["codec h264"]),
    ({"pix": "yuv444p"}, ["pix_fmt yuv420p"]),
    ({"fps": "30/1"}, ["frame rate 40/1"]),
    ({"fps": "0/0"}, ["frame rate 40/1"]),
])
def test_bad_streams_fail_the_right_check(kwargs, expected):
    assert failed(rcc.evaluate_probe(probe_json(**kwargs))) == expected


def test_fps_written_as_fraction_of_80_over_2_is_still_40():
    assert failed(rcc.evaluate_probe(probe_json(fps="80/2"))) == []


def test_irregular_keyframes_fail():
    data = probe_json()
    data["frames"][40]["key_frame"] = 0      # keyframes at 0, 80, 120: intervals 80 and 40
    assert failed(rcc.evaluate_probe(data)) == ["keyframe interval 40 frames"]


def test_too_few_keyframes_to_judge_fails():
    data = probe_json(gop=1000, n_frames=100)          # only frame 0 is a keyframe
    assert failed(rcc.evaluate_probe(data)) == ["keyframe interval 40 frames"]


def test_missing_stream_fails():
    assert failed(rcc.evaluate_probe({"streams": []})) == ["video stream present"]


def test_has_b_frames_may_arrive_as_string():
    assert failed(rcc.evaluate_probe(probe_json(bf="0"))) == []


# ── RTSP end to end with stub ffmpeg / ffprobe ───────────────────────────────

def write_exe(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def env_file(tmp_path):
    p = tmp_path / "remote.env"
    p.write_text(ENV_TEXT)
    return p


def stubs(tmp_path, probe=None, ffmpeg_body=None):
    log = tmp_path / "argv.log"
    ffmpeg = write_exe(tmp_path / "ffmpeg", ffmpeg_body or
                       f'echo "$@" >> {log}\necho $$ > {tmp_path}/ffmpeg.pid\nexec sleep 30\n')
    canned = tmp_path / "probe.json"
    canned.write_text(json.dumps(probe if probe is not None else probe_json()))
    ffprobe = write_exe(tmp_path / "ffprobe", f'echo "$@" >> {log}\ncat {canned}\n')
    return ffmpeg, ffprobe, log


def run_main(capsys, *argv):
    code = rcc.main(list(map(str, argv)))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_rtsp_pass_with_stubs_and_no_password_printed(tmp_path, env_file, capsys):
    ffmpeg, ffprobe, log = stubs(tmp_path)
    code, out, err = run_main(capsys, "--env", env_file, "--rtsp", "--ffmpeg", ffmpeg,
                              "--ffprobe", ffprobe, "--settle", "0.3")
    assert code == 0, out + err
    assert "RESULT: PASS" in out
    assert out.count("PASS  ") >= 8
    for text in (out, err):
        assert PW not in text and "s3cret" not in text and "w0rd" not in text
    # the stubs really received the credentialed URL (encoded) and the flags
    argv = log.read_text()
    assert "s3cret%3Ap%40ss%2Fw0rd%21@100.64.0.9:8554/rigtest-selftest" in argv
    assert "-profile:v main" in argv and "-read_intervals %+4" in argv


def test_rtsp_fail_on_high_profile(tmp_path, env_file, capsys):
    ffmpeg, ffprobe, _ = stubs(tmp_path, probe=probe_json(profile="High"))
    code, out, _ = run_main(capsys, "--env", env_file, "--rtsp", "--ffmpeg", ffmpeg,
                            "--ffprobe", ffprobe, "--settle", "0.3")
    assert code == 1
    assert "FAIL  profile Main" in out and "RESULT: FAIL" in out


def test_rtsp_publisher_dying_early_is_reported_redacted(tmp_path, env_file, capsys):
    ffmpeg, ffprobe, _ = stubs(
        tmp_path,
        ffmpeg_body=f'echo "rtsp://rigpub:{PW}@100.64.0.9:8554/x: 401 Unauthorized '
                    f'(password {PW})" >&2\nexit 1\n')
    code, out, err = run_main(capsys, "--env", env_file, "--rtsp", "--ffmpeg", ffmpeg,
                              "--ffprobe", ffprobe, "--settle", "0.5")
    assert code == 1
    assert "FAIL  ffmpeg publish" in out and "401 Unauthorized" in out
    assert PW not in out + err and "s3cret" not in out + err


def test_rtsp_ffprobe_error_output_is_redacted(tmp_path, env_file, capsys):
    ffmpeg, _, log = stubs(tmp_path)
    ffprobe = write_exe(tmp_path / "ffprobe2",
                        f'echo "Connection refused rtsp://rigpub:{PW}@h/x" >&2\nexit 1\n')
    code, out, err = run_main(capsys, "--env", env_file, "--rtsp", "--ffmpeg", ffmpeg,
                              "--ffprobe", ffprobe, "--settle", "0.3")
    assert code == 1
    assert "FAIL  ffprobe read" in out
    assert PW not in out + err and "s3cret" not in out + err


def test_no_leftover_publisher_process(tmp_path, env_file, capsys):
    ffmpeg, ffprobe, log = stubs(tmp_path)
    run_main(capsys, "--env", env_file, "--rtsp", "--ffmpeg", ffmpeg,
             "--ffprobe", ffprobe, "--settle", "0.3")
    pid = int((tmp_path / "ffmpeg.pid").read_text())     # the stub exec()s sleep: same pid
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# ── setup errors (exit 2) ────────────────────────────────────────────────────

def test_missing_env_file_exit_2(tmp_path, capsys):
    code, _, err = run_main(capsys, "--env", tmp_path / "nope.env", "--rtsp")
    assert code == 2 and "env file" in err


def test_missing_server_ip_exit_2(tmp_path, capsys):
    p = tmp_path / "e"
    p.write_text("RIG_PASS=x\n")
    code, _, err = run_main(capsys, "--env", p, "--rtsp")
    assert code == 2 and "SERVER_TAILNET_IP" in err


def test_missing_rig_pass_exit_2(tmp_path, capsys):
    p = tmp_path / "e"
    p.write_text("SERVER_TAILNET_IP=1.2.3.4\n")
    code, _, err = run_main(capsys, "--env", p, "--rtsp")
    assert code == 2 and "RIG_PASS" in err


def test_missing_ffmpeg_exit_2(tmp_path, env_file, capsys, monkeypatch):
    monkeypatch.setattr(rcc.shutil, "which", lambda name: None)
    code, _, err = run_main(capsys, "--env", env_file, "--rtsp")
    assert code == 2 and "ffmpeg" in err


def test_missing_paho_gives_venv_instructions_and_exit_2(env_file, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "paho", None)
    monkeypatch.setitem(sys.modules, "paho.mqtt", None)
    monkeypatch.setitem(sys.modules, "paho.mqtt.client", None)
    code, _, err = run_main(capsys, "--env", env_file, "--mqtt")
    assert code == 2
    assert "venv" in err and "pip install paho-mqtt" in err


# ── MQTT against an in-memory broker ─────────────────────────────────────────

class FakeBroker:
    def __init__(self, refuse=False, deliver_will=True):
        self.retained: dict[str, bytes] = {}
        self.clients: list = []
        self.refuse = refuse
        self.deliver_will = deliver_will
        self.log: list[str] = []

    def deliver(self, topic, payload, retain):
        if isinstance(payload, str):
            payload = payload.encode()
        if retain:
            if payload:
                self.retained[topic] = payload
            else:
                self.retained.pop(topic, None)
        for c in list(self.clients):
            if c.connected and c.matches(topic) and payload:
                c.on_message(c, None, types.SimpleNamespace(topic=topic, payload=payload))


class FakeSocket:
    def __init__(self, client):
        self.client = client

    def close(self):
        self.client.broker.drop_unclean(self.client)


def _drop_unclean(self, client):
    client.connected = False
    if client.will and self.deliver_will:
        topic, payload, retain = client.will
        self.deliver(topic, payload, retain)


FakeBroker.drop_unclean = _drop_unclean


class FakeClient:
    broker: FakeBroker = None

    def __init__(self, *args, client_id=None):
        self.client_id = client_id
        self.connected = False
        self.will = None
        self.subs: list[str] = []
        self.creds = None
        self.on_connect = self.on_subscribe = self.on_message = None

    def username_pw_set(self, user, password):
        self.creds = (user, password)

    def will_set(self, topic, payload, qos=0, retain=False):
        self.will = (topic, payload, retain)

    def connect(self, host, port, keepalive=60):
        self.host, self.port = host, port

    def loop_start(self):
        self.connected = not self.broker.refuse
        self.broker.clients.append(self)
        self.on_connect(self, None, {}, 5 if self.broker.refuse else 0)

    def loop_stop(self):
        pass

    def disconnect(self):
        self.connected = False

    def subscribe(self, topic, qos=0):
        self.subs.append(topic)
        self.on_subscribe(self, None, 1, (qos,))
        for t, p in list(self.broker.retained.items()):
            if self.matches(t):
                self.on_message(self, None, types.SimpleNamespace(topic=t, payload=p))

    def matches(self, topic):
        return any(topic == s or (s.endswith("/#") and topic.startswith(s[:-1]))
                   for s in self.subs)

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.broker.deliver(topic, payload if payload is not None else b"", retain)
        return types.SimpleNamespace(wait_for_publish=lambda timeout=None: True)

    def socket(self):
        return FakeSocket(self)


def fake_paho(broker, v2=True):
    class Client(FakeClient):
        pass
    Client.broker = broker
    mod = types.ModuleType("paho.mqtt.client")
    mod.Client = Client
    if v2:
        mod.CallbackAPIVersion = types.SimpleNamespace(VERSION2=2)
    mod.created = Client
    return mod


def test_mqtt_happy_path_checks_retained_will_and_cleanup():
    broker = FakeBroker()
    checks = rcc.run_mqtt_check(make_cfg(), fake_paho(broker), timeout_s=2)
    assert [(c.name, c.ok) for c in checks] == [
        ("mqtt connect", True), ("mqtt subscribe", True),
        ("retained status round-trip", True), ("last-will delivery", True),
        ("retained status cleared", True)]
    assert broker.retained == {}                     # retained selftest message cleared


def test_mqtt_works_with_paho_1x_client_signature():
    broker = FakeBroker()
    checks = rcc.run_mqtt_check(make_cfg(), fake_paho(broker, v2=False), timeout_s=2)
    assert all(c.ok for c in checks)


def test_mqtt_uses_credentials_and_selftest_topic():
    broker = FakeBroker()
    rcc.run_mqtt_check(make_cfg(), fake_paho(broker), timeout_s=2)
    assert broker.clients and all(c.creds == ("telem", MQTT_PW) for c in broker.clients)
    assert broker.clients[0].subs == ["lab/rigtest-selftest/#"]
    assert broker.clients[1].will[0] == "lab/rigtest-selftest/status"


def test_mqtt_anonymous_when_no_user():
    broker = FakeBroker()
    rcc.run_mqtt_check(make_cfg(MQTT_USER="", MQTT_PASS=""), fake_paho(broker), timeout_s=2)
    assert all(c.creds is None for c in broker.clients)


def test_mqtt_refused_connection_fails_cleanly():
    broker = FakeBroker(refuse=True)
    checks = rcc.run_mqtt_check(make_cfg(), fake_paho(broker), timeout_s=1)
    assert [(c.name, c.ok) for c in checks] == [("mqtt connect", False)]
    assert "refused" in checks[0].detail


def test_mqtt_missing_will_fails_but_still_cleans_up():
    broker = FakeBroker(deliver_will=False)
    checks = rcc.run_mqtt_check(make_cfg(), fake_paho(broker), timeout_s=0.5)
    by = {c.name: c.ok for c in checks}
    assert by["last-will delivery"] is False
    assert by["retained status round-trip"] is True
    assert broker.retained == {}


def test_mqtt_connect_exception_is_redacted():
    broker = FakeBroker()
    mod = fake_paho(broker)

    def boom(self, host, port, keepalive=60):
        raise OSError(f"auth with {MQTT_PW} failed")
    mod.Client.connect = boom
    checks = rcc.run_mqtt_check(make_cfg(), mod, timeout_s=1)
    assert checks[0].ok is False and MQTT_PW not in checks[0].detail


def test_mqtt_main_prints_table_without_passwords(env_file, capsys, monkeypatch):
    broker = FakeBroker()
    mod = fake_paho(broker)
    paho = types.ModuleType("paho")
    paho_mqtt = types.ModuleType("paho.mqtt")
    paho.mqtt, paho_mqtt.client = paho_mqtt, mod
    monkeypatch.setitem(sys.modules, "paho", paho)
    monkeypatch.setitem(sys.modules, "paho.mqtt", paho_mqtt)
    monkeypatch.setitem(sys.modules, "paho.mqtt.client", mod)
    code, out, err = run_main(capsys, "--env", env_file, "--mqtt")
    assert code == 0, out + err
    assert "PASS  last-will delivery" in out and "RESULT: PASS" in out
    assert MQTT_PW not in out + err and PW not in out + err
