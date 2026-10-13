"""An extends-overlay may only ADD abdomen/mounting; PER readers always get the base."""
from pathlib import Path
import pytest, yaml
from fbpipe.config import load_raw_config, load_settings, resolve_config_path
from fbpipe.config_layers import LayeredConfigError, load_layered, resolve_layers

BASE = {"protocol": "v2", "yolo": {"conf_thres": 0.25}, "force": {"pipeline": True}, "datasets": []}

def _w(p: Path, data) -> Path:
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8"); return p

def test_plain_file_is_its_own_base_with_the_identical_path(tmp_path):
    p = _w(tmp_path / "config_new.yaml", BASE)
    layers = resolve_layers(p)
    assert layers.base_path == p and str(layers.base_path) == str(p) and not layers.is_layered
    assert dict(layers.overlay) == {}
    assert load_layered(p) == yaml.safe_load(p.read_text())

def test_overlay_resolves_extends_against_its_own_directory(tmp_path, monkeypatch):
    sub = tmp_path / "cfg"; sub.mkdir()
    base = _w(sub / "config_new.yaml", BASE)
    o = _w(sub / "config_new_abdomen.yaml", {"extends": "config_new.yaml", "abdomen": {"enabled": True}})
    monkeypatch.chdir(tmp_path)                                  # cwd must not matter
    layers = resolve_layers(o)
    assert layers.base_path == resolve_config_path(base) and layers.is_layered
    assert dict(layers.overlay) == {"abdomen": {"enabled": True}}
    merged = load_layered(o)
    assert {k: v for k, v in merged.items() if k != "abdomen"} == BASE

@pytest.mark.parametrize("bad", [{"yolo": {"conf_thres": 0.5}}, {"force": {"pipeline": False}}, {"datasets": ["X"]}])
def test_overlay_may_not_touch_a_base_key(tmp_path, bad):
    _w(tmp_path / "config_new.yaml", BASE)
    o = _w(tmp_path / "o.yaml", {"extends": "config_new.yaml", "abdomen": {}, **bad})
    with pytest.raises(LayeredConfigError, match="may only add"):
        resolve_layers(o)

@pytest.mark.parametrize("base_extra,match", [({"abdomen": {}}, "already defines"), ({"extends": "x.yaml"}, "chained")])
def test_bad_bases_are_refused(tmp_path, base_extra, match):
    _w(tmp_path / "config_new.yaml", dict(BASE, **base_extra))
    o = _w(tmp_path / "o.yaml", {"extends": "config_new.yaml", "mounting": {}})
    with pytest.raises(LayeredConfigError, match=match):
        resolve_layers(o)

def test_self_reference_and_missing_base_are_refused(tmp_path):
    o = _w(tmp_path / "o.yaml", {"extends": "o.yaml", "abdomen": {}})
    with pytest.raises(LayeredConfigError, match="itself"):
        resolve_layers(o)
    o2 = _w(tmp_path / "o2.yaml", {"extends": "nope.yaml", "abdomen": {}})
    with pytest.raises(LayeredConfigError, match="not found"):
        resolve_layers(o2)

def test_plain_config_with_inline_blocks_exposes_them_as_overlay(tmp_path):
    p = _w(tmp_path / "c.yaml", dict(BASE, abdomen={"enabled": True}))
    assert dict(resolve_layers(p).overlay) == {"abdomen": {"enabled": True}}

def test_loaders_read_the_base_for_an_overlay(tmp_path):
    base = _w(tmp_path / "config_new.yaml", BASE)
    o = _w(tmp_path / "o.yaml", {"extends": "config_new.yaml", "abdomen": {"enabled": True}})
    assert load_settings(o) == load_settings(base)
    assert load_raw_config(o) == load_raw_config(base)

def test_live_config_new_passes_through_unchanged():
    live = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"
    if not live.is_file():
        pytest.skip("config/config_new.yaml is gitignored and absent")
    assert resolve_layers(resolve_config_path(live)).base_path == resolve_config_path(live)
