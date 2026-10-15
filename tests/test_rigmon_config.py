from pathlib import Path

import pytest

from rigmon.config import HubConfig, RigConfig, load_config

YAML = """
rigs:
  - {name: behaviorlocust, label: "Pi 1", ip: 100.121.64.36}
  - {name: flybehavior2, label: "Pi 2", ip: 100.93.23.70, port: 8010}
staging_root: {root}/staging
data_root: {root}/Data
db_path: {root}/staging/rigmon.sqlite
bind: [127.0.0.1, 100.94.160.60]
ntfy_topic: secret-topic
"""


def _write(tmp_path, text):
    p = tmp_path / "rigmon.yaml"
    p.write_text(text.replace("{root}", str(tmp_path)))
    return p


def test_load_config_parses_rigs_and_paths(tmp_path):
    cfg = load_config(_write(tmp_path, YAML))
    assert [r.name for r in cfg.rigs] == ["behaviorlocust", "flybehavior2"]
    assert cfg.rig("flybehavior2").status_url == "http://100.93.23.70:8010/status.json"
    assert cfg.rig("behaviorlocust").live_url == "http://100.121.64.36:8008/"
    assert cfg.staging_root == tmp_path / "staging" and cfg.bind == ("127.0.0.1", "100.94.160.60")
    assert cfg.port == 8765 and cfg.ntfy_topic == "secret-topic"
    assert cfg.remux_args == ("--crf", "22", "--preset", "slow", "--faststart",
                              "--scale", "1.0", "--fps-cap", "50.0")
    assert cfg.remux_python == "/usr/bin/python3" and cfg.remux_path_env == "/usr/bin:/bin"


def test_local_rig_polls_loopback():
    assert RigConfig("x", "X", "local", 1234).status_url == "http://127.0.0.1:1234/status.json"


def test_unknown_rig_raises(tmp_path):
    with pytest.raises(KeyError):
        load_config(_write(tmp_path, YAML)).rig("nope")


def test_no_rigs_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "rigs: []\nstaging_root: /a\ndata_root: /b\ndb_path: /c\n"))


def test_duplicate_rig_names_are_an_error(tmp_path):
    text = YAML.replace("name: flybehavior2", "name: behaviorlocust")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_staging_inside_flys_new_is_refused(tmp_path):
    text = YAML.replace("{root}/staging", "{root}/Data/flys_New/staging")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_example_config_loads():
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "config" / "example.rigmon.yaml")
    assert {r.ip for r in cfg.rigs} == {"100.121.64.36", "100.93.23.70", "100.97.155.65"}
    assert "0.0.0.0" not in cfg.bind


@pytest.mark.parametrize("addr", ["0.0.0.0", "::"])
def test_bind_all_interfaces_is_refused(tmp_path, addr):
    text = YAML.replace("100.94.160.60", f'"{addr}"')
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


from rigmon.config import MountCheckConfig


def test_mount_check_defaults_and_qc_root(tmp_path):
    cfg = load_config(_write(tmp_path, YAML))
    assert cfg.qc_root == tmp_path / "Data" / "rigmon_qc"
    assert cfg.pipeline_config.name == "config_new.yaml"
    mc = cfg.mount_check
    assert (mc.enabled, mc.delay_after_wait_start_s, mc.sample_s, mc.sample_fps) == (True, 1800.0, 300.0, 2.0)
    assert (mc.min_remaining_s, mc.worker_timeout_s) == (1200.0, 600.0)
    th = mc.th()
    assert (th.expected_flies, th.edge_margin_px, th.bottom_room_px) == (4, 40.0, 120.0)


def test_mount_check_overrides(tmp_path):
    text = YAML + "mount_check:\n  sample_s: 120\n  thresholds: {bottom_room_px: 150}\nqc_root: {root}/qc\n"
    cfg = load_config(_write(tmp_path, text))
    assert cfg.mount_check.sample_s == 120 and cfg.mount_check.th().bottom_room_px == 150
    assert cfg.qc_root == tmp_path / "qc"


def test_unknown_threshold_key_is_an_error(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        load_config(_write(tmp_path, YAML + "mount_check:\n  thresholds: {bogus: 1}\n"))


def test_qc_root_inside_flys_new_is_refused(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, YAML + "qc_root: {root}/Data/flys_New/qc\n"))


from rigmon.config import TracesConfig


def test_traces_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, YAML))
    assert cfg.traces == TracesConfig()
    assert (cfg.traces.enabled, cfg.traces.frame_stride, cfg.traces.conf) == (True, 2, 0.25)
    assert cfg.transfer_bwlimit_kbps == 40000 and cfg.cleanup_time == "00:05"


def test_traces_overrides_and_bad_cleanup_time(tmp_path):
    cfg = load_config(_write(tmp_path, YAML + "traces: {frame_stride: 3}\ncleanup_time: '01:30'\n"))
    assert cfg.traces.frame_stride == 3 and cfg.cleanup_time == "01:30"
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, YAML + "cleanup_time: '25:00'\n"))


def test_control_key_and_pause_reminder_load(tmp_path):
    text = YAML.replace("ip: 100.93.23.70, port: 8010}", "ip: 100.93.23.70, port: 8010, control_key: k1}") \
        + "pause_reminder_s: 120\n"
    cfg = load_config(_write(tmp_path, text))
    assert cfg.rig("flybehavior2").control_key == "k1" and cfg.pause_reminder_s == 120
    assert cfg.rig("behaviorlocust").control_key == "" and "k1" not in repr(cfg)
    assert load_config(_write(tmp_path, YAML)).pause_reminder_s == 300.0


@pytest.mark.parametrize("v", ["0", "-5"])
def test_pause_reminder_must_be_positive(tmp_path, v):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, YAML + f"pause_reminder_s: {v}\n"))
