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
    text = YAML.replace("100.94.160.60", addr)
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))
