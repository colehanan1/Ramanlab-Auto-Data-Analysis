"""Pure-geometry helpers of the abdomen pipeline."""
import math

import numpy as np
import pytest

from fbpipe.geometry import metrics as M


def rect(cx, cy, w, h, angle_deg=0.0):
    """Corners of a w x h box centred on (cx, cy), rotated, in cyclic order."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    half = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rot = half @ np.array([[c, s], [-s, c]])
    return rot + np.array([cx, cy])


def test_map_corners_scales_x_and_y_independently():
    pts = rect(100, 200, 40, 20)
    out = M.map_corners(pts, 1152 / 1080, 1.0)
    assert np.allclose(out[:, 0], pts[:, 0] * 1152 / 1080)
    assert np.allclose(out[:, 1], pts[:, 1])
    assert out.shape == (4, 2)


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 137.0])
def test_obb_length_width_is_rotation_invariant(angle):
    length, width = M.obb_length_width(rect(300, 300, 160, 70, angle))
    assert length == pytest.approx(160.0, abs=1e-6)
    assert width == pytest.approx(70.0, abs=1e-6)


def test_obb_length_width_after_non_uniform_mapping_uses_mapped_corners():
    """Under x-scaling a 45-degree box changes length by less than the x factor."""
    pts = M.map_corners(rect(0, 0, 100, 40, 45.0), 1152 / 1080, 1.0)
    length, _ = M.obb_length_width(pts)
    assert 100.0 < length < 100.0 * 1152 / 1080


def test_interior_angle_is_180_for_a_straight_fly_and_90_for_a_right_angle():
    eye, abd = (0.0, 0.0), (100.0, 0.0)
    assert M.interior_angle(eye, abd, (200.0, 0.0)) == pytest.approx(180.0)
    assert M.interior_angle(eye, abd, (100.0, 60.0)) == pytest.approx(90.0)
    assert M.interior_angle(eye, abd, (50.0, 0.0)) == pytest.approx(0.0)
    assert math.isnan(M.interior_angle(eye, abd, abd))


def test_signed_bend_is_positive_toward_the_proboscis_side_and_mirror_invariant():
    eye, abd = np.array([0.0, 0.0]), np.array([100.0, 0.0])
    pro = np.array([20.0, 40.0])          # proboscis below the axis (image y down)
    tip_same = np.array([150.0, 50.0])    # tip curls to the same side
    tip_other = np.array([150.0, -50.0])
    side = M.ventral_sign(eye, abd, pro)
    assert side in (1.0, -1.0)
    assert M.signed_bend(eye, abd, tip_same, side) > 0
    assert M.signed_bend(eye, abd, tip_other, side) < 0
    assert M.signed_bend(eye, abd, tip_same, side) == pytest.approx(
        180.0 - M.interior_angle(eye, abd, tip_same))
    # mirror the whole frame in x: the sign must not change
    flip = np.array([-1.0, 1.0])
    side_m = M.ventral_sign(eye * flip, abd * flip, pro * flip)
    assert M.signed_bend(eye * flip, abd * flip, tip_same * flip, side_m) > 0
    # no side information -> unsigned bend
    assert M.signed_bend(eye, abd, tip_other, 0.0) > 0
    assert M.signed_bend(eye, abd, np.array([200.0, 0.0]), side) == pytest.approx(0.0)


def test_velocity_gate_blanks_a_jump_but_keeps_the_reference():
    xy = np.array([[0, 0], [5, 0], [300, 0], [10, 0], [np.nan, np.nan], [15, 0]], float)
    reject = M.velocity_gate(xy, max_jump_px=80.0)
    assert reject.tolist() == [False, False, True, False, False, False]
    gate = M.OnlineVelocityGate(80.0)
    assert [gate.rejects(x, y) for x, y in xy] == reject.tolist()
    assert not M.velocity_gate(xy, max_jump_px=0.0).any(), "0 disables the gate"


def test_proboscis_pct_matches_the_production_formula():
    d = np.array([36.7, 50.0, 58.3, 100.0, 20.0, np.nan])
    out = M.proboscis_pct(d, gmin=36.7, gmax=58.3, floor=95.0)
    # effective max is max(gmax, floor) = 95; out-of-range values are NaN
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(100.0 * (50.0 - 36.7) / (95.0 - 36.7))
    assert out[2] == pytest.approx(100.0 * (58.3 - 36.7) / (95.0 - 36.7))
    assert np.isnan(out[3]) and np.isnan(out[4]) and np.isnan(out[5])


def test_proboscis_pct_agrees_with_distance_normalize_numbers():
    """Same inputs as fbpipe.steps.distance_normalize: gmin 36.71, gmax 102.64 (> floor)."""
    d = np.array([36.71, 70.0, 102.64])
    out = M.proboscis_pct(d, gmin=36.71, gmax=102.64, floor=95.0)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(100.0 * (70.0 - 36.71) / (102.64 - 36.71))
    assert out[2] == pytest.approx(100.0)


def test_angle_pct_maps_straightest_to_zero_and_most_curled_to_hundred():
    a = np.array([175.0, 150.0, 120.0, np.nan])
    out = M.angle_pct(a, angle_min=120.0, angle_max=175.0)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(100.0 * 25.0 / 55.0)
    assert out[2] == pytest.approx(100.0)
    assert np.isnan(out[3])


def test_angle_pct_range_floor_keeps_a_static_fly_below_hundred():
    a = np.array([178.0, 176.0])
    assert M.angle_pct(a, 176.0, 178.0)[1] == pytest.approx(100.0)
    assert M.angle_pct(a, 176.0, 178.0, min_range_deg=10.0)[1] == pytest.approx(20.0)
    assert np.isnan(M.angle_pct(a, 178.0, 178.0)).all()


def test_anchor_angle_matches_yolo_infer_helper():
    from fbpipe.steps.yolo_infer import _angle_deg_between
    eye, pro, anchor = (700.0, 80.0), (740.0, 120.0), (0.0, 540.0)
    assert M.anchor_angle_deg(eye, pro, anchor) == pytest.approx(
        _angle_deg_between((pro[0] - eye[0], pro[1] - eye[1]),
                           (anchor[0] - eye[0], anchor[1] - eye[1])))
