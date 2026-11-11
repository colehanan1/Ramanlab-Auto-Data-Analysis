"""Per-frame eye everywhere except while the odor (or light) is on.

Outside the hold the eye is the one detected in that frame (anchor where
none); inside it the eye stays where it sat over the second before the hold.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("les", ROOT / "scripts" / "analysis" / "live_eye_shadow.py")
les = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(les)


def _table(n=400, eye=(500.0, 300.0), prob=(560.0, 300.0), live=None):
    lx, ly = live if live is not None else (np.full(n, eye[0]), np.full(n, eye[1]))
    return pd.DataFrame({
        "frame": np.arange(n),
        "x_class0": eye[0], "y_class0": eye[1],
        "x_class1": prob[0], "y_class1": prob[1],
        "x_anchor": 0.0, "y_anchor": 540.0,
        "eye_live_x": lx, "eye_live_y": ly,
        "distance_0_1": np.hypot(prob[0] - eye[0], prob[1] - eye[1]),
    })


def test_outside_the_hold_the_eye_follows_every_frame():
    lx = np.full(400, 500.0)
    lx[50] = 503.0
    out = les.apply_live_eye_held(_table(live=(lx, np.full(400, 300.0))), hold=(200, 300), pre_frames=40)
    assert out["x_class0"].iloc[50] == 503.0
    assert out["distance_0_1"].iloc[50] == pytest.approx(57.0)
    assert not out["eye_held"].iloc[50]


def test_inside_the_hold_the_eye_stays_where_it_sat_before():
    lx = np.full(400, 504.0)
    lx[200:300] = 512.0                       # detection shifts while the odor is on
    out = les.apply_live_eye_held(_table(live=(lx, np.full(400, 300.0))), hold=(200, 300), pre_frames=40)
    held = out.iloc[200:301]
    assert held["eye_held"].all()
    assert np.allclose(held["x_class0"], 504.0)
    assert np.allclose(held["distance_0_1"], 56.0)
    assert out["x_class0"].iloc[350] == 504.0  # per-frame again after it


def test_no_detection_before_the_hold_falls_back_to_the_anchor():
    lx = np.full(400, np.nan)
    lx[250:] = 508.0
    ly = np.where(np.isnan(lx), np.nan, 300.0)
    out = les.apply_live_eye_held(_table(live=(lx, ly)), hold=(200, 300), pre_frames=40)
    assert np.allclose(out["x_class0"].iloc[200:301], 500.0)


def test_no_hold_is_the_plain_per_frame_eye():
    lx = 500.0 + np.random.default_rng(0).normal(0, 0.5, 400)
    t = _table(live=(lx, np.full(400, 300.0)))
    a = les.apply_live_eye_held(t, hold=None, pre_frames=40)
    b = les.apply_live_eye(t)
    np.testing.assert_allclose(a["distance_0_1"], b["distance_0_1"])


def test_twice_equals_once_and_input_untouched():
    lx = np.full(400, 504.0)
    lx[200:300] = 512.0
    t = _table(live=(lx, np.full(400, 300.0)))
    before = t.copy()
    once = les.apply_live_eye_held(t, hold=(200, 300), pre_frames=40)
    twice = les.apply_live_eye_held(once, hold=(200, 300), pre_frames=40)
    pd.testing.assert_frame_equal(t, before)
    np.testing.assert_allclose(twice["distance_0_1"], once["distance_0_1"])
    assert np.allclose(twice["x_class0_anchor"], 500.0)


def test_stimulus_hold_spans_odor_and_light():
    meta = {"odor_on_frame": 1200, "odor_off_frame": 2400, "fps": 40.0,
            "light_on_s": 30.0, "light_off_s": 62.0}
    assert les.stimulus_hold(meta) == (1180, 2520)        # light-off + 1 s is later
    assert les.stimulus_hold({"odor_on_frame": 1200, "odor_off_frame": 2400, "fps": 40.0}) == (1180, 2440)
    assert les.stimulus_hold({}) is None


def test_light_only_hold_ignores_the_odor():
    meta = {"odor_on_frame": 1200, "odor_off_frame": 2400, "fps": 40.0,
            "light_on_s": 30.0, "light_off_s": 55.0}
    assert les.stimulus_hold(meta, odor=False) == (1180, 2240)
    assert les.stimulus_hold({"odor_on_frame": 1200, "odor_off_frame": 2400, "fps": 40.0}, odor=False) is None
