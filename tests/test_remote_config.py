"""Unit tests for PiCode/remote_config.py — .env parsing and the RemoteConfig
the rig scripts use for the remote stream + MQTT telemetry. Stdlib-only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "PiCode"))

import remote_config as rc  # noqa: E402


# --------------------------------------------------------------- load_env
def test_load_env_missing_file_is_empty(tmp_path):
    assert rc.load_env(tmp_path / "nope.env") == {}


def test_load_env_parses_comments_quotes_export(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "\n"
        "PLAIN=abc\n"
        'DQ="with space"\n'
        "SQ='single q'\n"
        "export EXPORTED=yes\n"
        "  SPACED  =  padded  \n"
        "EMPTY=\n"
        "EQ_IN_VALUE=a=b=c\n"
    )
    env = rc.load_env(p)
    assert env["PLAIN"] == "abc"
    assert env["DQ"] == "with space"
    assert env["SQ"] == "single q"
    assert env["EXPORTED"] == "yes"
    assert env["SPACED"] == "padded"
    assert env["EMPTY"] == ""
    assert env["EQ_IN_VALUE"] == "a=b=c"


def test_load_env_skips_malformed_lines(tmp_path):
    p = tmp_path / ".env"
    p.write_text("no equals sign\n=novalue_key\nGOOD=1\n   \n#X=1\n")
    assert rc.load_env(p) == {"GOOD": "1"}


def test_load_env_never_raises_on_binary_or_directory(tmp_path):
    p = tmp_path / ".env"
    p.write_bytes(b"\xff\xfe\x00GOOD=1\n")
    assert isinstance(rc.load_env(p), dict)
    assert rc.load_env(tmp_path) == {}  # a directory, not a file


# ------------------------------------------------------------ rig_id_for_host
def test_rig_id_for_host_known_prefixes_case_insensitive():
    assert rc.rig_id_for_host("behaviorlocust") == "rig1"
    assert rc.rig_id_for_host("BehaviorLocust") == "rig1"
    assert rc.rig_id_for_host("flybehavior2") == "rig2"
    assert rc.rig_id_for_host("FlyBehavior3.local") == "rig3"


def test_rig_id_for_host_unknown_is_sanitized_lowercase():
    assert rc.rig_id_for_host("My Host/01") == "my-host-01"
    assert rc.rig_id_for_host("") != ""


# ------------------------------------------------------------- RemoteConfig
def test_defaults():
    cfg = rc.RemoteConfig(rig_id="rig2", server_ip="")
    assert (cfg.width, cfg.height, cfg.fps) == (540, 540, 40)
    assert cfg.gop == 40
    assert cfg.bitrate == 1_500_000
    assert cfg.profile == "main"
    assert cfg.rtsp_port == 8554 and cfg.rtsp_user == "rigpub"
    assert cfg.mqtt_port == 1883
    assert cfg.stream_enabled is True and cfg.telemetry_enabled is True
    assert cfg.is_complete() is False


def test_from_env_maps_every_key():
    env = {
        "SERVER_TAILNET_IP": "100.1.2.3", "RIG_ID": "custom",
        "RTSP_PORT": "9000", "RTSP_USER": "u", "RIG_PASS": "rp",
        "MQTT_PORT": "1884", "MQTT_USER": "mu", "MQTT_PASS": "mp",
        "STREAM_WIDTH": "640", "STREAM_HEIGHT": "480", "STREAM_FPS": "30",
        "STREAM_BITRATE": "2000000", "STREAM_GOP": "60",
        "REMOTE_STREAM": "0", "REMOTE_TELEMETRY": "0",
    }
    cfg = rc.RemoteConfig.from_env(env, "flybehavior2")
    assert cfg.server_ip == "100.1.2.3" and cfg.rig_id == "custom"
    assert (cfg.rtsp_port, cfg.rtsp_user, cfg.rtsp_pass) == (9000, "u", "rp")
    assert (cfg.mqtt_port, cfg.mqtt_user, cfg.mqtt_pass) == (1884, "mu", "mp")
    assert (cfg.width, cfg.height, cfg.fps) == (640, 480, 30)
    assert (cfg.bitrate, cfg.gop) == (2_000_000, 60)
    assert cfg.stream_enabled is False and cfg.telemetry_enabled is False
    assert cfg.is_complete() is True


def test_from_env_rig_id_from_hostname_and_defaults():
    cfg = rc.RemoteConfig.from_env({"SERVER_TAILNET_IP": "100.1.2.3"}, "flybehavior3")
    assert cfg.rig_id == "rig3"
    assert cfg.stream_enabled is True and cfg.telemetry_enabled is True
    assert cfg.width == 540


def test_from_env_bad_ints_keep_defaults():
    env = {"SERVER_TAILNET_IP": "x", "RTSP_PORT": "abc", "STREAM_FPS": "",
           "STREAM_GOP": "4.5", "MQTT_PORT": None}
    cfg = rc.RemoteConfig.from_env(env, "behaviorlocust")
    assert cfg.rtsp_port == 8554 and cfg.fps == 40
    assert cfg.gop == 40 and cfg.mqtt_port == 1883


def test_rtsp_url_quotes_credentials():
    cfg = rc.RemoteConfig(rig_id="rig1", server_ip="100.1.2.3",
                          rtsp_user="us er", rtsp_pass="p@ss:/w#rd")
    assert cfg.rtsp_url() == "rtsp://us%20er:p%40ss%3A%2Fw%23rd@100.1.2.3:8554/rig1"


def test_redaction_never_leaks_passwords():
    cfg = rc.RemoteConfig(rig_id="rig1", server_ip="100.1.2.3",
                          rtsp_pass="SECRETRTSP", mqtt_pass="SECRETMQTT")
    red = cfg.rtsp_url_redacted()
    assert "SECRETRTSP" not in red and ":***@" in red
    assert red.endswith("@100.1.2.3:8554/rig1")
    for text in (repr(cfg), str(cfg), f"{cfg}"):
        assert "SECRETRTSP" not in text and "SECRETMQTT" not in text
    assert "***" in repr(cfg)


def test_config_is_frozen():
    import dataclasses
    import pytest
    cfg = rc.RemoteConfig(rig_id="r", server_ip="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.fps = 1  # type: ignore[misc]


def test_default_env_path_sits_beside_module():
    p = rc.default_env_path()
    assert p.name == ".env" and p.parent == Path(rc.__file__).parent


def test_stream_profile_from_env():
    import remote_config as rc
    base = {"SERVER_TAILNET_IP": "100.104.46.34"}
    assert rc.RemoteConfig.from_env(base, "flybehavior3").profile == "main"
    for raw, want in (("baseline", "baseline"), (" Baseline ", "baseline"),
                      ("MAIN", "main"), ("high", "main"), ("", "main")):
        cfg = rc.RemoteConfig.from_env(dict(base, STREAM_PROFILE=raw), "flybehavior3")
        assert cfg.profile == want, raw
