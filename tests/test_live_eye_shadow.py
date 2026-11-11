"""Pin the live-eye transform used to build the flys_LiveEye comparison tree.

Production measures every proboscis distance from a frozen eye anchor; the
shadow tree measures it from the eye detected in that frame (``eye_live_x/y``),
falling back to the anchor on frames where the eye was not found.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "live_eye_shadow", ROOT / "scripts" / "analysis" / "live_eye_shadow.py"
)
les = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(les)

from fbpipe.steps.yolo_infer import _angle_deg_between  # noqa: E402


def _frame(**over):
    base = dict(
        frame=[0, 1, 2, 3],
        x_class0=[100.0] * 4,
        y_class0=[200.0] * 4,
        x_class1=[100.0, 110.0, np.nan, 100.0],
        y_class1=[280.0, 290.0, np.nan, 300.0],
        x_anchor=[0.0] * 4,
        y_anchor=[540.0] * 4,
        eye_live_x=[100.0, 103.0, 104.0, np.nan],
        eye_live_y=[200.0, 196.0, 195.0, np.nan],
    )
    base.update(over)
    df = pd.DataFrame(base)
    ex, ey = df.x_class0, df.y_class0
    df["distance_0_1"] = np.hypot(df.x_class1 - ex, df.y_class1 - ey)
    df["distance_0_anchor"] = np.hypot(ex - df.x_anchor, ey - df.y_anchor)
    df["angle_deg_c0_c1_vs_anchor"] = [
        _angle_deg_between((px - x, py - y), (ax - x, ay - y)) if np.isfinite(px) else np.nan
        for px, py, x, y, ax, ay in zip(df.x_class1, df.y_class1, ex, ey, df.x_anchor, df.y_anchor)
    ]
    return df


def test_live_eye_equal_to_anchor_changes_nothing():
    df = _frame(eye_live_x=[100.0] * 4, eye_live_y=[200.0] * 4)
    out = les.apply_live_eye(df)
    for col in ("distance_0_1", "distance_0_anchor", "angle_deg_c0_c1_vs_anchor", "x_class0", "y_class0"):
        pd.testing.assert_series_equal(out[col], df[col], check_names=False)


def test_distance_and_angle_measured_from_the_live_eye():
    out = les.apply_live_eye(_frame())
    # frame 1: live eye (103,196), proboscis (110,290)
    assert out.loc[1, "distance_0_1"] == pytest.approx(np.hypot(7.0, 94.0))
    assert out.loc[1, "angle_deg_c0_c1_vs_anchor"] == pytest.approx(
        _angle_deg_between((7.0, 94.0), (-103.0, 344.0))
    )
    assert out.loc[1, "distance_0_anchor"] == pytest.approx(np.hypot(103.0, 344.0))
    assert (out.loc[1, "x_class0"], out.loc[1, "y_class0"]) == (103.0, 196.0)


def test_missing_live_eye_falls_back_to_the_anchor():
    df = _frame()
    out = les.apply_live_eye(df)
    assert out.loc[3, "distance_0_1"] == pytest.approx(df.loc[3, "distance_0_1"])
    assert (out.loc[3, "x_class0"], out.loc[3, "y_class0"]) == (100.0, 200.0)
    assert bool(out.loc[3, "eye_from_live"]) is False
    assert out["eye_from_live"].tolist() == [True, True, True, False]


def test_missing_proboscis_stays_missing():
    out = les.apply_live_eye(_frame())
    assert np.isnan(out.loc[2, "distance_0_1"])
    assert np.isnan(out.loc[2, "angle_deg_c0_c1_vs_anchor"])


def test_frozen_anchor_is_kept_for_traceability_and_input_untouched():
    df = _frame()
    before = df.copy()
    out = les.apply_live_eye(df)
    pd.testing.assert_frame_equal(df, before)
    assert out["x_class0_anchor"].tolist() == [100.0] * 4
    assert out["y_class0_anchor"].tolist() == [200.0] * 4


def test_applying_twice_is_the_same_as_applying_once():
    once = les.apply_live_eye(_frame())
    twice = les.apply_live_eye(once)
    pd.testing.assert_frame_equal(once, twice)


def test_tree_writer_refuses_paths_outside_the_comparison_tree(tmp_path):
    with pytest.raises(SystemExit):
        les.apply_tree(tmp_path / "flys_New" / "Hex-Sensitivity-1-0.1")


def test_tree_writer_rewrites_trial_tables_only(tmp_path):
    batch = tmp_path / "flys_LiveEye" / "DS" / "b1"
    trial = batch / "b1_testing_1"
    rms = batch / "RMS_calculations"
    trial.mkdir(parents=True)
    rms.mkdir()
    _frame().to_parquet(trial / "b1_testing_1_fly1_distances.parquet")
    _frame().to_parquet(rms / "updated_b1_testing_1_fly1_distances.parquet")
    summary = les.apply_tree(tmp_path / "flys_LiveEye" / "DS")
    assert len(summary) == 1
    assert "eye_from_live" in pd.read_parquet(trial / "b1_testing_1_fly1_distances.parquet")
    assert "eye_from_live" not in pd.read_parquet(rms / "updated_b1_testing_1_fly1_distances.parquet")


def test_file_without_live_eye_columns_is_returned_unchanged_but_flagged():
    df = _frame().drop(columns=["eye_live_x", "eye_live_y"])
    out = les.apply_live_eye(df)
    pd.testing.assert_series_equal(out["distance_0_1"], df["distance_0_1"])
    assert not out["eye_from_live"].any()
