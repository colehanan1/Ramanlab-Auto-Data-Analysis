#!/usr/bin/env python3
"""One-shot desktop-side check of the remote stream/telemetry server, run BEFORE any
Pi is touched.

    python3 scripts/tools/remote_connectivity_check.py [--env config/remote.env]
        [--rtsp] [--mqtt] [--ffmpeg PATH] [--ffprobe PATH] [--secs 12]

With neither --rtsp nor --mqtt both checks run.

RTSP: publishes an ffmpeg ``testsrc2`` test stream with exactly the encoder settings
the rigs will use (H.264 Main, no B-frames, GOP 40, 540x540 yuv420p, 40 fps) to
``rtsp://.../<RIG_ID>-selftest``, reads it back with ffprobe and checks codec,
profile, size, pixel format, has_b_frames, frame rate and the keyframe interval.

MQTT: connects with the telemetry credentials, subscribes ``lab/<RIG_ID>-selftest/#``,
publishes a retained JSON status and sees it come back, then drops a second client
that carries a will without a clean DISCONNECT and sees the will arrive; finally the
retained message is cleared.  Needs paho-mqtt, which this script never installs: make
a throw-away venv in a scratch dir and ``pip install paho-mqtt`` there.

Env file (KEY=VALUE, ``#`` comments): SERVER_TAILNET_IP, RIG_ID (rigtest), RTSP_PORT
(8554), RTSP_USER (rigpub), RIG_PASS, RTSP_READ_USER / RTSP_READ_PASS (default: the
publish credentials), MQTT_PORT (1883), MQTT_USER, MQTT_PASS.  config/* is gitignored.

Passwords are never printed: every command and every line of subprocess output goes
through redact().  Commands are built as argument lists and never run through a shell.

Exit status: 0 every check passed, 1 a check failed, 2 usage / environment error.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

MQTT_TIMEOUT_S = 10.0
EXPECT_SIZE = (540, 540)
EXPECT_FPS = 40
EXPECT_GOP = 40
REDACTED = "***"


class SetupError(Exception):
    """Missing env key / binary / library (exit status 2)."""


# ── env file ─────────────────────────────────────────────────────────────────

def parse_env_text(text: str) -> dict[str, str]:
    """Small KEY=VALUE parser: ``#`` comments, blank lines, optional ``export``,
    one layer of matching single or double quotes around the value."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key.strip()] = value
    return out


@dataclass
class Config:
    ip: str
    rig_id: str = "rigtest"
    rtsp_port: int = 8554
    rtsp_user: str = "rigpub"
    rig_pass: str = ""
    read_user: str = ""
    read_pass: str = ""
    mqtt_port: int = 1883
    mqtt_user: str = ""
    mqtt_pass: str = ""

    @property
    def secrets(self) -> list[str]:
        return [s for s in (self.rig_pass, self.read_pass, self.mqtt_pass) if s]


def config_from_env(env: dict[str, str]) -> Config:
    ip = env.get("SERVER_TAILNET_IP", "").strip()
    if not ip:
        raise SetupError("SERVER_TAILNET_IP is not set in the env file")
    user = env.get("RTSP_USER", "rigpub") or "rigpub"
    rig_pass = env.get("RIG_PASS", "")
    try:
        rtsp_port = int(env.get("RTSP_PORT", "8554") or 8554)
        mqtt_port = int(env.get("MQTT_PORT", "1883") or 1883)
    except ValueError as exc:
        raise SetupError(f"bad port in env file: {exc}") from exc
    return Config(
        ip=ip,
        rig_id=env.get("RIG_ID", "rigtest") or "rigtest",
        rtsp_port=rtsp_port,
        rtsp_user=user,
        rig_pass=rig_pass,
        read_user=env.get("RTSP_READ_USER") or user,
        read_pass=env.get("RTSP_READ_PASS") or rig_pass,
        mqtt_port=mqtt_port,
        mqtt_user=env.get("MQTT_USER", ""),
        mqtt_pass=env.get("MQTT_PASS", ""),
    )


# ── redaction ────────────────────────────────────────────────────────────────

_URL_CREDS = re.compile(r"(://[^:/@\s]*):[^@\s]*@")


def redact(obj, secrets=()):
    """Return ``obj`` (a string, or a list of arguments) with passwords hidden.

    Any ``://user:password@`` is masked whatever the password is, and every literal
    secret (raw and URL-quoted) is replaced wherever it appears.  A list stays a list.
    """
    if isinstance(obj, (list, tuple)):
        return [redact(x, secrets) for x in obj]
    text = _URL_CREDS.sub(rf"\1:{REDACTED}@", str(obj))
    for s in secrets:
        if not s:
            continue
        for variant in {s, quote(s, safe="")}:
            text = text.replace(variant, REDACTED)
    return text


def format_cmd(args: list[str], secrets=()) -> str:
    """Shell-quoted, redacted rendering of an argument list, for logs only."""
    return shlex.join(redact(list(args), secrets))


# ── command builders (return argument lists) ─────────────────────────────────

def selftest_path(cfg: Config) -> str:
    return f"{cfg.rig_id}-selftest"


def rtsp_url(cfg: Config, user: str, password: str) -> str:
    return (f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}"
            f"@{cfg.ip}:{cfg.rtsp_port}/{selftest_path(cfg)}")


def build_publish_cmd(cfg: Config, ffmpeg: str, secs: float = 12) -> list[str]:
    secs_arg = str(int(secs)) if float(secs).is_integer() else str(secs)
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-re",
        "-f", "lavfi", "-i", "testsrc2=size=540x540:rate=40",
        "-t", secs_arg,
        "-c:v", "libx264", "-profile:v", "main", "-bf", "0",
        "-g", "40", "-keyint_min", "40", "-sc_threshold", "0",
        "-b:v", "1500k", "-maxrate", "1500k", "-bufsize", "3000k",
        "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-tune", "zerolatency",
        "-an", "-f", "rtsp", "-rtsp_transport", "tcp",
        rtsp_url(cfg, cfg.rtsp_user, cfg.rig_pass),
    ]


def build_probe_cmd(cfg: Config, ffprobe: str) -> list[str]:
    return [
        ffprobe, "-rtsp_transport", "tcp", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,profile,width,height,pix_fmt,"
                         "has_b_frames,avg_frame_rate",
        "-show_entries", "frame=key_frame,pts_time",
        "-read_intervals", "%+4", "-of", "json",
        rtsp_url(cfg, cfg.read_user, cfg.read_pass),
    ]


# ── ffprobe verdict (pure) ───────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _fps(text) -> Fraction | None:
    try:
        return Fraction(str(text))
    except (ValueError, ZeroDivisionError):
        return None


def evaluate_probe(data: dict) -> list[Check]:
    """Verdict on ffprobe's JSON for the self-test stream."""
    streams = data.get("streams") or []
    if not streams:
        return [Check("video stream present", False, "ffprobe reported no video stream")]
    s = streams[0]
    checks = [Check("video stream present", True)]

    codec = s.get("codec_name")
    checks.append(Check("codec h264", codec == "h264", f"got {codec!r}"))
    profile = s.get("profile")
    checks.append(Check("profile Main", profile == "Main", f"got {profile!r}"))
    size = (s.get("width"), s.get("height"))
    checks.append(Check("size 540x540", size == EXPECT_SIZE, f"got {size[0]}x{size[1]}"))
    pix = s.get("pix_fmt")
    checks.append(Check("pix_fmt yuv420p", pix == "yuv420p", f"got {pix!r}"))
    try:
        bf = int(s.get("has_b_frames"))
    except (TypeError, ValueError):
        bf = None
    checks.append(Check("no B-frames (has_b_frames 0)", bf == 0, f"got {bf!r}"))
    fps = _fps(s.get("avg_frame_rate"))
    checks.append(Check("frame rate 40/1", fps == EXPECT_FPS,
                        f"got {s.get('avg_frame_rate')!r}"))

    frames = data.get("frames") or []
    keys = [i for i, f in enumerate(frames) if str(f.get("key_frame")) == "1"]
    if len(keys) < 2:
        checks.append(Check(f"keyframe interval {EXPECT_GOP} frames", False,
                            f"only {len(keys)} keyframe(s) in {len(frames)} probed frame(s)"))
    else:
        gaps = sorted({b - a for a, b in zip(keys, keys[1:])})
        checks.append(Check(f"keyframe interval {EXPECT_GOP} frames",
                            gaps == [EXPECT_GOP], f"intervals seen: {gaps}"))
    return checks


# ── RTSP check ───────────────────────────────────────────────────────────────

def run_rtsp_check(cfg: Config, ffmpeg: str, ffprobe: str, secs: float = 12,
                   settle_s: float = 3.0) -> list[Check]:
    secrets = cfg.secrets
    pub = build_publish_cmd(cfg, ffmpeg, secs)
    probe = build_probe_cmd(cfg, ffprobe)
    print(f"[rtsp] publish: {format_cmd(pub, secrets)}")
    print(f"[rtsp] probe:   {format_cmd(probe, secrets)}")
    proc = subprocess.Popen(pub, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(settle_s)
        if proc.poll() is not None:
            err = redact(proc.stderr.read() if proc.stderr else "", secrets).strip()
            return [Check("ffmpeg publish", False,
                          f"exited {proc.returncode} before the probe: {err[-400:]}")]
        checks = [Check("ffmpeg publish", True, "running while probed")]
        try:
            res = subprocess.run(probe, capture_output=True, text=True, timeout=25)
        except subprocess.TimeoutExpired:
            return checks + [Check("ffprobe read", False, "timed out after 25 s")]
        if res.returncode != 0:
            err = redact(res.stderr, secrets).strip()
            return checks + [Check("ffprobe read", False,
                                   f"exit {res.returncode}: {err[-400:]}")]
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            return checks + [Check("ffprobe read", False, "output was not JSON")]
        return checks + evaluate_probe(data)
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()


# ── MQTT check ───────────────────────────────────────────────────────────────

def _rc_ok(rc) -> bool:
    """paho 1.x passes an int, 2.x a ReasonCode."""
    if isinstance(rc, int):
        return rc == 0
    return not getattr(rc, "is_failure", True)


def _make_client(mqtt, client_id: str):
    """paho-mqtt 2.x needs the callback API version, 1.x does not know it."""
    if hasattr(mqtt, "CallbackAPIVersion"):
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    return mqtt.Client(client_id=client_id)


def run_mqtt_check(cfg: Config, mqtt, timeout_s: float = MQTT_TIMEOUT_S) -> list[Check]:
    """``mqtt`` is the paho.mqtt.client module (injected so tests can fake it)."""
    import threading

    deadline = time.monotonic() + timeout_s
    base = f"lab/{selftest_path(cfg)}"
    topic = f"{base}/status"
    marker = f"selftest-{int(time.time())}"
    retained_payload = json.dumps({"state": "online", "selftest": marker})
    will_payload = json.dumps({"state": "offline", "selftest": marker})

    got_retained = threading.Event()
    got_will = threading.Event()
    subscribed = threading.Event()
    connected_a = threading.Event()
    connected_b = threading.Event()
    connect_error: list[str] = []
    b_error: list[str] = []

    def left() -> float:
        return max(0.0, deadline - time.monotonic())

    def creds(client) -> None:
        if cfg.mqtt_user:
            client.username_pw_set(cfg.mqtt_user, cfg.mqtt_pass)

    a = _make_client(mqtt, f"{selftest_path(cfg)}-a-{marker}")
    creds(a)

    def a_connect(client, userdata, flags, rc, *extra):
        if _rc_ok(rc):
            connected_a.set()
            client.subscribe(f"{base}/#", qos=1)
        else:
            connect_error.append(f"CONNACK refused: {rc}")
            connected_a.set()

    def a_subscribe(client, userdata, mid, *extra):
        subscribed.set()

    def a_message(client, userdata, msg):
        payload = msg.payload.decode("utf-8", "replace")
        if payload == retained_payload and msg.topic == topic:
            got_retained.set()
        elif payload == will_payload and msg.topic == topic:
            got_will.set()

    a.on_connect, a.on_subscribe, a.on_message = a_connect, a_subscribe, a_message

    state: dict = {"b": None}

    def body() -> list[Check]:
        checks: list[Check] = []
        try:
            a.connect(cfg.ip, cfg.mqtt_port, keepalive=30)
        except Exception as exc:                       # noqa: BLE001 - network errors vary
            return [Check("mqtt connect", False, redact(str(exc), cfg.secrets))]
        a.loop_start()
        if not connected_a.wait(left()) or connect_error:
            why = connect_error[0] if connect_error else "no CONNACK before the timeout"
            return [Check("mqtt connect", False, why)]
        checks.append(Check("mqtt connect", True))
        if not subscribed.wait(left()):
            return checks + [Check("mqtt subscribe", False, "no SUBACK before the timeout")]
        checks.append(Check("mqtt subscribe", True, f"{base}/#"))

        a.publish(topic, retained_payload, qos=1, retain=True)
        ok = got_retained.wait(left())
        checks.append(Check("retained status round-trip", ok,
                            "" if ok else "own retained message never came back"))

        b = state["b"] = _make_client(mqtt, f"{selftest_path(cfg)}-b-{marker}")
        creds(b)
        b.will_set(topic, will_payload, qos=1, retain=False)

        def b_connect(client, userdata, flags, rc, *extra):
            if _rc_ok(rc):
                connected_b.set()
            else:
                b_error.append(f"will client CONNACK refused: {rc}")

        b.on_connect = b_connect
        try:
            b.connect(cfg.ip, cfg.mqtt_port, keepalive=30)
        except Exception as exc:                       # noqa: BLE001
            return checks + [Check("last-will delivery", False,
                                   "will client could not connect: "
                                   + redact(str(exc), cfg.secrets))]
        b.loop_start()
        if not connected_b.wait(left()):
            return checks + [Check("last-will delivery", False,
                                   b_error[0] if b_error
                                   else "will client got no CONNACK before the timeout")]
        # Unclean death: close the socket without sending DISCONNECT, then stop the
        # network thread so it does not reconnect and cancel the experiment.
        try:
            b.socket().close()
        finally:
            b.loop_stop()
        ok = got_will.wait(left())
        checks.append(Check("last-will delivery", ok,
                            "" if ok else "will not received (broker down or ACL blocks it)"))
        return checks

    try:
        checks = body()
    finally:
        cleared = False
        if connected_a.is_set() and not connect_error:   # only if we ever got in
            try:                                         # always clear the retained message
                info = a.publish(topic, b"", qos=1, retain=True)
                info.wait_for_publish(2.0)
                cleared = True
            except Exception:                            # noqa: BLE001
                cleared = False
        for c in (a, state["b"]):
            if c is not None:
                try:
                    c.loop_stop()
                    c.disconnect()
                except Exception:                        # noqa: BLE001
                    pass
    if connected_a.is_set() and not connect_error:
        checks.append(Check("retained status cleared", cleared,
                            "empty retained publish sent" if cleared
                            else "could not clear lab/<rig>-selftest/status; delete it by hand"))
    return checks


# ── CLI ──────────────────────────────────────────────────────────────────────

def format_table(title: str, checks: list[Check]) -> str:
    width = max([len(c.name) for c in checks] + [4])
    lines = [f"== {title} =="]
    for c in checks:
        lines.append(f"{'PASS' if c.ok else 'FAIL'}  {c.name.ljust(width)}  {c.detail}".rstrip())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check the remote RTSP/MQTT server before touching any Pi.")
    ap.add_argument("--env", default="config/remote.env")
    ap.add_argument("--rtsp", action="store_true", help="run the RTSP check")
    ap.add_argument("--mqtt", action="store_true", help="run the MQTT check")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--ffprobe", default=None)
    ap.add_argument("--secs", type=float, default=12.0, help="test stream length")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds to let the publisher run before probing (default 3)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    do_rtsp = args.rtsp or not args.mqtt
    do_mqtt = args.mqtt or not args.rtsp
    try:
        env_path = Path(args.env).expanduser()
        try:
            cfg = config_from_env(parse_env_text(env_path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise SetupError(f"cannot read env file {env_path}: {exc.strerror}") from exc
        if do_rtsp and not cfg.rig_pass:
            raise SetupError("RIG_PASS is not set in the env file (needed for --rtsp)")
        ffmpeg = ffprobe = None
        if do_rtsp:
            ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
            ffprobe = args.ffprobe or shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise SetupError("ffmpeg and ffprobe are required for --rtsp "
                                 "(pass --ffmpeg / --ffprobe)")
        mqtt = None
        if do_mqtt:
            try:
                import paho.mqtt.client as mqtt                # noqa: PLC0415
            except ImportError as exc:
                raise SetupError(
                    "paho-mqtt is not installed, and this script does not install it. "
                    "Create a throw-away venv in a scratch dir and install it there:\n"
                    "  python3 -m venv /tmp/mqttcheck-venv && "
                    "/tmp/mqttcheck-venv/bin/pip install paho-mqtt\n"
                    "  /tmp/mqttcheck-venv/bin/python scripts/tools/remote_connectivity_check.py "
                    "--mqtt") from exc
    except SetupError as exc:
        print(f"remote_connectivity_check: {exc}", file=sys.stderr)
        return 2

    all_checks: list[Check] = []
    print(f"server {cfg.ip}, rig id {cfg.rig_id}\n")
    if do_rtsp:
        checks = run_rtsp_check(cfg, ffmpeg, ffprobe, args.secs, args.settle)
        print(format_table("RTSP", checks) + "\n")
        all_checks += checks
    if do_mqtt:
        checks = run_mqtt_check(cfg, mqtt)
        print(format_table("MQTT", checks) + "\n")
        all_checks += checks
    ok = all(c.ok for c in all_checks)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
