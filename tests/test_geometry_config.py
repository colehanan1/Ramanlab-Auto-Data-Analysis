"""The geometry: config block and the hard validation of load_geometry_config."""
from pathlib import Path

import pytest
import yaml

from fbpipe.geometry.config import GeometrySettings, load_geometry_config, parse_geometry_block

REPO = Path(__file__).resolve().parents[1]


def test_defaults_and_overrides():
    g = parse_geometry_block(None)
    assert g.table_format == "csv" and g.measure_in == "native" and g.imgsz == 1088
    assert g.warmup_frames == 25 and g.per_effective_max_floor_px == 95.0
    assert g.angle_pct_min_range_deg == 0.0
    assert g.limits.d_eye_abd_max == 7.4
    g2 = parse_geometry_block({"table_format": "parquet", "limits": {"d_eye_abd_max": 8.0}, "imgsz": 1088})
    assert g2.table_format == "parquet" and g2.limits.d_eye_abd_max == 8.0
    assert g2.limits.d_eye_abd_min == 3.4, "untouched limits keep their defaults"


@pytest.mark.parametrize("bad", [
    {"table_format": "xlsx"},
    {"measure_in": "sideways"},
    {"warmup_frames": 0},
    {"limits": {"not_a_limit": 1.0}},
    {"proboscis_max_jump_px": -1},
])
def test_invalid_blocks_raise(bad):
    with pytest.raises(ValueError):
        parse_geometry_block(bad)


def _write_config(tmp_path, data_base, model, datasets=("Hex-Sensitivity-24-0.1",)):
    cfg = {
        "protocol": "v2",
        "model_path": str(model),
        "dataset_bases": {"data": str(data_base), "secured": str(tmp_path / "secured")},
        "datasets": list(datasets),
        "cache_dir": str(tmp_path / "cache"),
        "yolo": {"conf_thres": 0.25, "max_flies": 5},
        "geometry": {"wide_out_dir": str(tmp_path / "wide"), "figures_out_dir": str(tmp_path / "figs")},
    }
    p = tmp_path / "config_abdomen.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_geometry_config_resolves_main_directories(tmp_path):
    base = tmp_path / "flys_Abdomen"
    (base / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3_copy").mkdir(parents=True)
    model = tmp_path / "best.pt"
    model.write_bytes(b"x")
    settings, g = load_geometry_config(_write_config(tmp_path, base, model))
    assert settings.model_path == str(model)
    assert [Path(d).name for d in settings.main_directories] == ["Hex-Sensitivity-24-0.1"]
    assert settings.max_flies == 5 and settings.conf_thres == 0.25
    assert isinstance(g, GeometrySettings) and g.wide_out_dir.endswith("wide")


def test_load_geometry_config_refuses_a_missing_data_base(tmp_path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        load_geometry_config(_write_config(tmp_path, tmp_path / "missing_base", model))


def test_load_geometry_config_refuses_an_empty_dataset_list(tmp_path):
    base = tmp_path / "flys_Abdomen"
    base.mkdir()                       # exists, but holds no dataset dir
    model = tmp_path / "best.pt"
    model.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        load_geometry_config(_write_config(tmp_path, base, model))


def test_load_geometry_config_refuses_a_missing_model(tmp_path):
    base = tmp_path / "flys_Abdomen"
    (base / "Hex-Sensitivity-24-0.1" / "b").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_geometry_config(_write_config(tmp_path, base, tmp_path / "nope.pt"))


def test_example_config_parses():
    raw = yaml.safe_load((REPO / "config" / "example_abdomen.yaml").read_text())
    g = parse_geometry_block(raw["geometry"])
    assert g.table_format == "csv"
    assert raw["model_path"].endswith("runs/obb/train/weights/best.pt")
    assert "train4" not in raw["model_path"]
    for key in ("dataset_bases", "datasets", "cache_dir", "yolo"):
        assert key in raw
    assert not any(k.startswith("freeze") for k in raw), "the demo config carries no freeze rules"
