"""eye_live_measure: measure from the per-frame eye, held while the light is on.

Only for the datasets listed in ``eye_measure.datasets`` (mode
``live_light_hold``); a dataset taken off the list is put back on the fixed
eye. Runs after YOLO and before reject_bad_proboscis.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe import pipeline
from fbpipe.config import EyeMeasureSettings, Settings
from fbpipe.steps import eye_live_measure as elm


def _trial(batch: Path, name: str, live_x, light=None, n=400):
    t = batch / name
    t.mkdir(parents=True)
    meta = {"fps": 40.0}
    if light:
        meta["light_on_s"], meta["light_off_s"] = light
    (t / "_trial_meta.json").write_text(json.dumps(meta))
    df = pd.DataFrame({
        "frame": np.arange(n),
        "x_class0": 500.0, "y_class0": 300.0,
        "x_class1": 560.0, "y_class1": 300.0,
        "x_anchor": 0.0, "y_anchor": 540.0,
        "eye_live_x": live_x, "eye_live_y": 300.0,
        "distance_0_1": 60.0,
    })
    p = t / f"{name}_fly1_distances.parquet"
    df.to_parquet(p, index=False)
    return p


def _cfg(tmp_path, datasets=("ds",)):
    cfg = Settings(model_path="unused", main_directories=[str(tmp_path / "ds"), str(tmp_path / "other")])
    cfg.eye_measure = EyeMeasureSettings(mode="live_light_hold", datasets=tuple(datasets))
    cfg.parallel.enabled = False
    return cfg


def _setup(tmp_path):
    live = np.full(400, 503.0)
    live[200:240] = 512.0                         # light on 5-6 s: YOLO misplaces the eye
    p = _trial(tmp_path / "ds" / "oct_01_batch_1", "oct_01_batch_1_testing_9", live, light=(5.0, 6.0))
    q = _trial(tmp_path / "other" / "oct_01_batch_1", "oct_01_batch_1_testing_9", live, light=(5.0, 6.0))
    return p, q


def test_listed_dataset_measures_from_the_per_frame_eye_held_under_the_light(tmp_path):
    p, q = _setup(tmp_path)
    elm.main(_cfg(tmp_path))
    d = pd.read_parquet(p)
    assert d.distance_0_1.iloc[50] == pytest.approx(57.0)          # per-frame eye 503
    assert d.distance_0_1.iloc[220] == pytest.approx(57.0)         # held at 503, not 512
    assert d.x_class0_anchor.iloc[0] == 500.0                      # fixed eye kept
    assert pd.read_parquet(q).distance_0_1.iloc[50] == 60.0        # other dataset untouched


def test_running_twice_changes_nothing_more(tmp_path):
    p, _ = _setup(tmp_path)
    elm.main(_cfg(tmp_path))
    once = pd.read_parquet(p)
    elm.main(_cfg(tmp_path))
    pd.testing.assert_frame_equal(once, pd.read_parquet(p))


def test_dataset_taken_off_the_list_goes_back_to_the_fixed_eye(tmp_path):
    p, _ = _setup(tmp_path)
    original = pd.read_parquet(p)
    elm.main(_cfg(tmp_path))
    elm.main(_cfg(tmp_path, datasets=()))
    back = pd.read_parquet(p)
    assert "x_class0_anchor" not in back.columns
    np.testing.assert_allclose(back.distance_0_1, original.distance_0_1)
    np.testing.assert_allclose(back.x_class0, original.x_class0)


def test_tables_without_per_frame_eye_are_left_alone(tmp_path):
    p, _ = _setup(tmp_path)
    df = pd.read_parquet(p).drop(columns=["eye_live_x", "eye_live_y"])
    df.to_parquet(p, index=False)
    elm.main(_cfg(tmp_path))
    pd.testing.assert_frame_equal(df, pd.read_parquet(p))


def test_mode_frozen_does_nothing(tmp_path):
    p, _ = _setup(tmp_path)
    before = pd.read_parquet(p)
    cfg = _cfg(tmp_path)
    cfg.eye_measure = EyeMeasureSettings(mode="frozen", datasets=("ds",))
    elm.main(cfg)
    pd.testing.assert_frame_equal(before, pd.read_parquet(p))


def test_registered_before_reject_bad_proboscis_in_both_import_branches():
    names = [s.name for s in pipeline.ORDERED_STEPS]
    assert names.index("yolo") < names.index("eye_live_measure") < names.index("reject_bad_proboscis")
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "steps"]
    assert all("eye_live_measure" in {a.name for a in n.names} for n in imports)


def test_settings_read_from_yaml(tmp_path):
    import yaml
    from fbpipe.config import load_settings
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "config" / "config_new.yaml").read_text())
    data["eye_measure"] = {"mode": "live_light_hold", "datasets": ["IAA-Sensitivity-1-1"]}
    data.setdefault("distance_limits", {})["light_exclusion_datasets"] = ["IAA-Sensitivity-1-1"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    s = load_settings(p)
    assert s.eye_measure.mode == "live_light_hold" and s.eye_measure.datasets == ("IAA-Sensitivity-1-1",)
    assert s.minmax_light_datasets == ("IAA-Sensitivity-1-1",)


def test_min_exclusion_only_for_listed_datasets(tmp_path):
    from fbpipe.utils.light_frames import light_exclusion_applies
    cfg = _cfg(tmp_path)
    cfg.minmax_exclude_light = True
    cfg.minmax_light_datasets = ("ds",)
    assert light_exclusion_applies(cfg, tmp_path / "ds" / "oct_01_batch_1")
    assert not light_exclusion_applies(cfg, tmp_path / "other" / "oct_01_batch_1")
    cfg.minmax_light_datasets = ()
    assert light_exclusion_applies(cfg, tmp_path / "other" / "oct_01_batch_1")   # empty list = every dataset
    cfg.minmax_exclude_light = False
    assert not light_exclusion_applies(cfg, tmp_path / "ds" / "oct_01_batch_1")


def test_off_with_no_datasets_returns_without_reading_anything(tmp_path, monkeypatch):
    _setup(tmp_path)
    cfg = _cfg(tmp_path)
    cfg.eye_measure = EyeMeasureSettings()                  # the default: frozen, nothing listed
    monkeypatch.setattr(elm, "iter_live_batch_dirs", lambda *a, **k: pytest.fail("walked the data"))
    elm.main(cfg)
