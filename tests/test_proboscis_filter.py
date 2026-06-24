import numpy as np
import pandas as pd

from fbpipe.utils.columns import (
    PROBOSCIS_DISTANCE_COL,
    PROBOSCIS_X_COL,
    PROBOSCIS_Y_COL,
)
from fbpipe.utils.distance_sanity import (
    sanitize_eye_prob_geometry_dataframe,
    sanitize_proboscis_dataframe,
    sanitize_proboscis_velocity_dataframe,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_geometry_gate_blanks_points_beyond_radius():
    df = _df({
        "frame": [0, 1, 2],
        PROBOSCIS_X_COL: [100.0, 101.0, 102.0],
        PROBOSCIS_Y_COL: [100.0, 100.0, 100.0],
        PROBOSCIS_DISTANCE_COL: [50.0, 200.0, 40.0],  # row 1 is out of radius
    })
    cleaned, count = sanitize_eye_prob_geometry_dataframe(df, max_distance_px=150.0)
    assert count == 1
    assert pd.isna(cleaned.loc[1, PROBOSCIS_X_COL])
    assert pd.isna(cleaned.loc[1, PROBOSCIS_DISTANCE_COL])
    # In-radius rows are untouched
    assert cleaned.loc[0, PROBOSCIS_X_COL] == 100.0
    assert cleaned.loc[2, PROBOSCIS_DISTANCE_COL] == 40.0


def test_anisotropic_gate_tight_right_up_generous_left_down():
    # Eye at (500,500). divisor=4, max=175 -> right/up limit 43.75, left/down 175.
    rows = {
        "frame": [0, 1, 2, 3, 4],
        "x_class0": [500.0, 500.0, 500.0, 500.0, 500.0],
        "y_class0": [500.0, 500.0, 500.0, 500.0, 500.0],
        # max=150, up_divisor=4 -> up limit 37.5; left/right/down unrestricted (150)
        # 0: 100px down  (within 150) KEEP
        # 1: 100px left  (within 150) KEEP
        # 2: 100px right (within 150) KEEP
        # 3: 100px up    (beyond 37.5) REJECT
        # 4: 40px right  (within 150) KEEP
        "x_class1": [500.0, 400.0, 600.0, 500.0, 540.0],
        "y_class1": [600.0, 500.0, 500.0, 400.0, 500.0],
        "distance_0_1": [100.0, 100.0, 100.0, 100.0, 40.0],
    }
    df = _df(rows)
    cleaned, count = sanitize_eye_prob_geometry_dataframe(
        df, max_distance_px=150.0, up_divisor=4.0
    )
    assert count == 1
    assert not pd.isna(cleaned.loc[0, "x_class1"])  # down kept
    assert not pd.isna(cleaned.loc[1, "x_class1"])  # left kept
    assert not pd.isna(cleaned.loc[2, "x_class1"])  # right kept (no restriction)
    assert pd.isna(cleaned.loc[3, "x_class1"])      # up rejected
    assert not pd.isna(cleaned.loc[4, "x_class1"])  # small right kept


def test_velocity_gate_blanks_impossible_jump_only():
    # Steady creep then a single impossible jump back to near the start.
    df = _df({
        "frame": [0, 1, 2, 3],
        PROBOSCIS_X_COL: [100.0, 110.0, 400.0, 120.0],  # row 2 jumps ~290px
        PROBOSCIS_Y_COL: [100.0, 100.0, 100.0, 100.0],
        PROBOSCIS_DISTANCE_COL: [10.0, 20.0, 30.0, 40.0],
    })
    cleaned, count = sanitize_proboscis_velocity_dataframe(df, max_jump_px=80.0)
    assert count == 1
    assert pd.isna(cleaned.loc[2, PROBOSCIS_X_COL])
    # row 3 is compared against the last accepted point (row 1), so it survives
    assert cleaned.loc[3, PROBOSCIS_X_COL] == 120.0


def test_velocity_gate_skips_existing_nans():
    df = _df({
        "frame": [0, 1, 2],
        PROBOSCIS_X_COL: [100.0, np.nan, 130.0],
        PROBOSCIS_Y_COL: [100.0, np.nan, 100.0],
        PROBOSCIS_DISTANCE_COL: [10.0, np.nan, 20.0],
    })
    cleaned, count = sanitize_proboscis_velocity_dataframe(df, max_jump_px=80.0)
    assert count == 0
    assert cleaned.loc[2, PROBOSCIS_X_COL] == 130.0


def test_combined_is_idempotent():
    df = _df({
        "frame": [0, 1, 2, 3],
        PROBOSCIS_X_COL: [100.0, 500.0, 130.0, 600.0],
        PROBOSCIS_Y_COL: [100.0, 100.0, 100.0, 100.0],
        PROBOSCIS_DISTANCE_COL: [10.0, 20.0, 300.0, 30.0],  # row 2 out of radius
    })
    once, geo1, vel1 = sanitize_proboscis_dataframe(df, max_distance_px=150.0, max_jump_px=80.0)
    twice, geo2, vel2 = sanitize_proboscis_dataframe(once, max_distance_px=150.0, max_jump_px=80.0)
    assert geo2 == 0 and vel2 == 0  # second pass finds nothing new
    pd.testing.assert_frame_equal(once, twice)


def test_zero_thresholds_disable_gates():
    df = _df({
        "frame": [0, 1],
        PROBOSCIS_X_COL: [100.0, 900.0],
        PROBOSCIS_Y_COL: [100.0, 900.0],
        PROBOSCIS_DISTANCE_COL: [10.0, 9999.0],
    })
    _, geo, vel = sanitize_proboscis_dataframe(df, max_distance_px=0.0, max_jump_px=0.0)
    assert geo == 0 and vel == 0
