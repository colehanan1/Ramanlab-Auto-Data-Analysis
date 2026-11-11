"""Leaving stimulus-light frames out of each fly's global min/max.

Off by default (production unchanged). On: distance_stats ignores frames from
light-on - margin to light-off + margin; distance_normalize then caps (not
blanks) the in-gate frames that fall outside the narrower range.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import Settings
from fbpipe.steps import distance_normalize, distance_stats
from fbpipe.utils.light_frames import light_frame_mask


def _trial(batch: Path, name: str, dist, light=None, fps=40.0):
    t = batch / name
    t.mkdir(parents=True)
    meta = {"fps": fps, "trial_type": name.split("_")[-2]}
    if light:
        meta["light_on_s"], meta["light_off_s"] = light
    (t / "_trial_meta.json").write_text(json.dumps(meta))
    n = len(dist)
    df = pd.DataFrame({
        "frame": np.arange(n),
        "x_class0": 500.0, "y_class0": 300.0,
        "x_class1": 500.0 + np.asarray(dist, float), "y_class1": 300.0,
        "distance_0_1": np.asarray(dist, float),
    })
    p = t / f"{name}_fly1_distances.parquet"
    df.to_parquet(p, index=False)
    return p


def _cfg(tmp_path, on: bool) -> Settings:
    cfg = Settings(model_path="unused", main_directories=[str(tmp_path)])
    cfg.class2_min, cfg.class2_max = 10.0, 160.0
    cfg.minmax_exclude_light = on
    cfg.parallel.enabled = False
    return cfg


def _build(tmp_path):
    batch = tmp_path / "ds" / "oct_01_batch_1"
    odor = np.full(400, 60.0)
    odor[100:120] = 110.0                      # biggest odor response: 110 px
    lit = np.full(400, 60.0)
    lit[200:240] = 150.0                       # light-evoked PER: 150 px, light on 4.5-6.5 s
    lit[205] = 20.0                            # junk under the light: 20 px
    _trial(batch, "oct_01_batch_1_testing_1", odor)
    p_light = _trial(batch, "oct_01_batch_1_testing_9", lit, light=(4.5, 6.5))
    return batch, p_light


def test_mask_covers_light_with_margins():
    meta = {"fps": 40.0, "light_on_s": 30.0, "light_off_s": 55.0}
    m = light_frame_mask(np.arange(3600), meta, before_s=0.5, after_s=1.0)
    assert m[1180] and m[2240] and not m[1179] and not m[2241]
    assert not light_frame_mask(np.arange(10), {"fps": 40.0}, 0.5, 1.0).any()


def test_off_keeps_light_frames_in_the_range(tmp_path):
    batch, _ = _build(tmp_path)
    distance_stats._process_fly_dir(batch, _cfg(tmp_path, on=False))
    s = json.loads((batch / "fly1_global_distance_stats_class_0.json").read_text())
    assert s["global_min"] == 20.0 and s["global_max"] == 150.0


def test_on_leaves_light_frames_out_of_the_range(tmp_path):
    batch, _ = _build(tmp_path)
    distance_stats._process_fly_dir(batch, _cfg(tmp_path, on=True))
    s = json.loads((batch / "fly1_global_distance_stats_class_0.json").read_text())
    assert s["global_min"] == 60.0 and s["global_max"] == 110.0


def test_on_caps_light_frames_outside_the_range_instead_of_blanking(tmp_path):
    batch, p_light = _build(tmp_path)
    cfg = _cfg(tmp_path, on=True)
    distance_stats._process_fly_dir(batch, cfg)
    pct = distance_normalize.cap_outside_range(
        pd.read_parquet(p_light)["distance_0_1"].to_numpy(), np.full(400, np.nan),
        gmin=60.0, effective_max=110.0, class2_min=10.0, class2_max=160.0)
    assert pct[220] == 100.0                   # 150 px > 110: capped, not NaN
    assert pct[205] == 0.0                     # 20 px < 60: floored, not NaN


def test_cap_leaves_out_of_gate_frames_blank():
    d = np.array([5.0, 200.0, 80.0])
    pct = distance_normalize.cap_outside_range(d, np.array([np.nan, np.nan, 40.0]), gmin=60.0,
                                               effective_max=110.0, class2_min=10.0, class2_max=160.0)
    assert np.isnan(pct[0]) and np.isnan(pct[1]) and pct[2] == 40.0


def test_rms_copy_finds_its_trial_light_window(tmp_path):
    batch, _ = _build(tmp_path)
    rms = batch / "RMS_calculations"
    rms.mkdir()
    p = rms / "updated_oct_01_batch_1_testing_9_fly1_distances.parquet"
    from fbpipe.utils.light_frames import trial_meta_for_table
    assert trial_meta_for_table(p, batch)["light_on_s"] == 4.5


# ---------------------------------------------------------------- envelope (scored trace) bounds


def _ec():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ec_light", root / "scripts" / "analysis" / "envelope_combined.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["ec_light"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_envelope_bound_mask_follows_the_option(tmp_path):
    ec = _ec()
    batch, p_light = _build(tmp_path)
    df = pd.read_parquet(p_light)
    ec.set_runtime_settings(_cfg(tmp_path, on=False))
    assert ec._light_bound_mask(df, p_light, batch) is None
    ec.set_runtime_settings(_cfg(tmp_path, on=True))
    m = ec._light_bound_mask(df, p_light, batch)
    # light 4.5-6.5 s at 40 fps, margins 0.5 s before / 1 s after -> frames 160..300
    assert m[160] and m[300] and not m[159] and not m[301]
    ec.set_runtime_settings(None)


def test_envelope_bounds_skip_light_frames():
    ec = _ec()
    v = np.array([80.0, 300.0, 40.0, 90.0])
    lit = np.array([False, True, True, False])
    assert ec._bounds_outside_light(v, lit) == (80.0, 90.0)
    assert ec._bounds_outside_light(v, None) == (40.0, 300.0)


# ---------------------------------------------------------------- min only (keep the light PER as 100 %)


def test_min_only_keeps_the_light_per_as_the_max(tmp_path):
    batch, _ = _build(tmp_path)
    cfg = _cfg(tmp_path, on=True)
    cfg.minmax_light_scope = "min"
    distance_stats._process_fly_dir(batch, cfg)
    s = json.loads((batch / "fly1_global_distance_stats_class_0.json").read_text())
    assert s["global_min"] == 60.0             # the 20 px junk frame under the light is left out
    assert s["global_max"] == 150.0            # the light-evoked PER still sets 100 %


def test_envelope_bounds_min_only():
    ec = _ec()
    v = np.array([80.0, 300.0, 40.0, 90.0])
    lit = np.array([False, True, True, False])
    assert ec._bounds_outside_light(v, lit, scope="min") == (80.0, 300.0)


def test_scope_is_read_from_the_config(tmp_path):
    import yaml
    from fbpipe.config import load_settings
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "config" / "config_new.yaml").read_text())
    data["distance_limits"].update({"exclude_light_frames_from_minmax": True, "light_frames_left_out_of": "min"})
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    s = load_settings(p)
    assert s.minmax_exclude_light is True and s.minmax_light_scope == "min"
