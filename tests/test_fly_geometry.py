"""Tests for utils/fly_geometry -- the combined eye/proboscis/abdomen/tip rules.

The thresholds encode measurements from 900 correctly-paired labelled flies, so
they are pinned here: a silent edit to a limit changes which detections the
pipeline keeps, and that must not happen without a test failing.

Synthetic flies are built from the measured medians (eye_scale 48 px, proboscis
1.29 eye-widths out, abdomen 4.96, tip 6.37) so each test reads as "a normal
fly, with one thing wrong".
"""

import math

import numpy as np
import pytest

from fbpipe.utils.fly_geometry import (
    ABDOMEN,
    ABDOMEN_TIP,
    DEFAULT_LIMITS,
    EYE,
    PROBOSCIS,
    FlyGeometryLimits,
    Part,
    angle_between,
    assemble,
    check_eye,
    check_proboscis,
    estimate_chirality,
    estimate_posterior_axis,
    eye_scale,
    summarise,
    touches_border,
)

EYE_SCALE = 48.0
AXIS = np.array([1.0, 0.0])          # posterior direction: to the right


def box(center, side_x, side_y=None):
    """Axis-aligned quad of the given full width/height, centred on ``center``."""
    cx, cy = center
    hx = side_x / 2.0
    hy = (side_y if side_y is not None else side_x) / 2.0
    return np.array([[cx - hx, cy - hy], [cx + hx, cy - hy],
                     [cx + hx, cy + hy], [cx - hx, cy + hy]], dtype=float)


def eye_part(center=(300.0, 300.0), scale=EYE_SCALE):
    return Part(EYE, box(center, scale))


def normal_fly(eye_c=(300.0, 300.0), scale=EYE_SCALE, axis=AXIS, chirality=-1.0):
    """A fly sitting at every measured median. ``chirality`` places the proboscis."""
    eye_c = np.asarray(eye_c, dtype=float)
    axis = np.asarray(axis, dtype=float)
    perp = np.array([-axis[1], axis[0]])
    # median angle(proboscis, abdomen) is 79 deg, with the sign set by the rig
    ang = math.radians(79.0)
    pro_dir = math.cos(-ang) * axis + math.sin(-ang) * perp * chirality
    parts = [
        Part(EYE, box(eye_c, scale)),
        Part(PROBOSCIS, box(eye_c + pro_dir * 1.29 * scale, 0.58 * scale)),
        Part(ABDOMEN, box(eye_c + axis * 4.96 * scale, 3.4 * scale, 2.0 * scale)),
        Part(ABDOMEN_TIP, box(eye_c + axis * 6.37 * scale, 0.93 * scale)),
    ]
    return parts


# --------------------------------------------------------------------------- #
# the measured limits are load-bearing constants
# --------------------------------------------------------------------------- #
def test_limits_match_the_measured_label_distribution():
    lim = DEFAULT_LIMITS
    assert (lim.eye_scale_min, lim.eye_scale_max) == (30.0, 75.0)
    assert (lim.d_eye_pro_min, lim.d_eye_pro_max) == (0.6, 3.2)
    assert (lim.d_eye_abd_min, lim.d_eye_abd_max) == (3.4, 7.4)
    assert (lim.d_eye_tip_min, lim.d_eye_tip_max) == (4.4, 10.0)
    assert (lim.tip_ratio_min, lim.tip_ratio_max) == (1.03, 1.55)
    assert lim.posterior_reach == 6.37
    # The abdomen is NOT opposite the proboscis: the labels put it at 79 deg.
    assert lim.ang_pro_abd_min < 79.0 < lim.ang_pro_abd_max
    assert lim.ang_pro_abd_max < 180.0


def test_every_measured_bound_brackets_its_observed_range():
    """Each gate must admit the full observed range, or it clips real flies."""
    lim = DEFAULT_LIMITS
    observed = [
        (lim.d_eye_pro_min, lim.d_eye_pro_max, 0.79, 2.82),
        (lim.d_eye_abd_min, lim.d_eye_abd_max, 3.56, 7.19),
        (lim.d_eye_tip_min, lim.d_eye_tip_max, 4.59, 9.71),
        (lim.d_abd_tip_min, lim.d_abd_tip_max, 0.94, 2.64),
        (lim.tip_ratio_min, lim.tip_ratio_max, 1.08, 1.50),
        (lim.abd_size_min, lim.abd_size_max, 1.82, 4.23),
        (lim.ang_pro_abd_min, lim.ang_pro_abd_max, 43.3, 125.9),
    ]
    for lo, hi, obs_lo, obs_hi in observed:
        assert lo <= obs_lo, f"gate min {lo} clips observed {obs_lo}"
        assert hi >= obs_hi, f"gate max {hi} clips observed {obs_hi}"


# --------------------------------------------------------------------------- #
# eye
# --------------------------------------------------------------------------- #
def test_a_neighbours_tip_is_not_borrowed_when_a_fly_lacks_its_own():
    """Why d_eye_tip_min stays at 4.4.

    Measuring "nearest tip to the eye" on real video appears to show tips at
    3.76 eye-widths, suggesting the floor clips short flies. It does not: those
    are the fly ABOVE's tip, and they fail ratio (0.88 -- nearer than the
    abdomen), abdomen->tip distance (6.85 vs 0.9-2.8) and the projection test
    (negative) all at once. Loosening the floor would admit exactly these.
    """
    lower = normal_fly(eye_c=(300.0, 500.0))
    orphan_eye = Part(EYE, box((300.0, 700.0), EYE_SCALE))
    flies = assemble(lower + [orphan_eye], 1080, 1080)
    orphan = next(f for f in flies if f.eye is orphan_eye)
    assert orphan.abdomen is None, "borrowed the neighbouring fly's abdomen/tip"
    assert not orphan.trackable


def test_bare_eye_is_not_a_fly():
    """The fly1 case: a false-positive eye on frame-edge clutter.

    It had 56.5% presence, confidence 0.38, and its nearest proboscis 636 px
    away -- 15 eye-widths. Nothing attaches to it, so no slot should be frozen.
    """
    lone = Part(EYE, box((1065.0, 120.0), EYE_SCALE))
    flies = assemble(normal_fly() + [lone], 1080, 1080)
    real = next(f for f in flies if f.eye is not lone)
    fake = next(f for f in flies if f.eye is lone)
    assert real.is_fly and real.trackable
    assert not fake.is_fly
    assert any("not a fly" in r for r in fake.rejected)


def test_a_fly_with_only_a_proboscis_still_counts():
    """A momentary abdomen miss must not demote a real fly to a false positive."""
    parts = [p for p in normal_fly() if p.label in (EYE, PROBOSCIS)]
    fly = assemble(parts, 1080, 1080)[0]
    assert fly.is_fly
    assert not any("not a fly" in r for r in fly.rejected)


def test_eye_scale_is_sqrt_area():
    assert eye_scale(eye_part(scale=48.0)) == pytest.approx(48.0)


def test_check_eye_rejects_a_zoomed_crop_eye():
    """The 129 px eyes in the head-crop images are a different imaging domain."""
    verdict = check_eye(eye_part(scale=129.0))
    assert not verdict
    assert "eye_scale" in verdict.reason


def test_check_eye_rejects_an_implausibly_elongated_eye():
    assert not check_eye(Part(EYE, box((300, 300), 90.0, 25.0)))


# --------------------------------------------------------------------------- #
# proboscis -- the eye-relative gate
# --------------------------------------------------------------------------- #
def test_proboscis_accepted_at_the_measured_median():
    eye = eye_part()
    pro = Part(PROBOSCIS, box((300 + 1.29 * EYE_SCALE, 300.0), 0.58 * EYE_SCALE))
    assert check_proboscis(eye, pro)


def test_proboscis_rejected_when_it_belongs_to_another_fly():
    eye = eye_part()
    far = Part(PROBOSCIS, box((300 + 8.0 * EYE_SCALE, 300.0), 0.58 * EYE_SCALE))
    verdict = check_proboscis(eye, far)
    assert not verdict
    assert "eye->proboscis" in verdict.reason


def test_eye_relative_gate_keeps_a_long_extension_that_a_fixed_radius_clips():
    """The point of scaling by the eye.

    A large fly (eye_scale 66 px, the observed max) extending 2.7 eye-widths is
    real, but lands at 178 px -- past the 160 px absolute gate in config_new.yaml
    and the 150 px dataclass default. The eye-relative gate keeps it.
    """
    big = 66.0
    eye = eye_part(scale=big)
    d_px = 2.7 * big
    assert d_px > 160.0
    pro = Part(PROBOSCIS, box((300 + d_px, 300.0), 0.58 * big))
    assert check_proboscis(eye, pro)


def test_same_pixel_distance_is_rejected_for_a_small_fly():
    """The converse: 178 px from a small eye is another fly's proboscis."""
    small = 34.0
    eye = eye_part(scale=small)
    pro = Part(PROBOSCIS, box((300 + 178.0, 300.0), 0.58 * small))
    assert not check_proboscis(eye, pro)


def test_proboscis_falls_back_to_an_absolute_cap_without_a_usable_eye():
    eye = Part(EYE, box((300, 300), 0.0))          # degenerate: zero area
    near = Part(PROBOSCIS, box((300 + 100.0, 300.0), 20.0))
    far = Part(PROBOSCIS, box((300 + 400.0, 300.0), 20.0))
    assert check_proboscis(eye, near, scale=0.0)
    assert not check_proboscis(eye, far, scale=0.0)


# --------------------------------------------------------------------------- #
# rig-level quantities: estimated, never hardcoded
# --------------------------------------------------------------------------- #
def test_posterior_axis_recovers_a_known_direction():
    pairs = [(np.array([0.0, 0.0]), np.array([100.0, 40.0])),
             (np.array([0.0, 200.0]), np.array([100.0, 235.0])),
             (np.array([0.0, 400.0]), np.array([100.0, 445.0]))]
    axis = estimate_posterior_axis(pairs)
    assert angle_between(axis, np.array([100.0, 40.0])) < 5.0


def test_posterior_axis_trims_a_mispaired_outlier():
    good = [(np.array([0.0, float(i)]), np.array([100.0, float(i) + 35.0]))
            for i in range(0, 400, 100)]
    outlier = (np.array([0.0, 0.0]), np.array([0.0, -100.0]))     # points straight up
    axis = estimate_posterior_axis(good + [outlier])
    assert angle_between(axis, np.array([100.0, 35.0])) < 10.0


def test_chirality_separates_the_mirrored_rig():
    """Negative in all 23 standard sessions, positive in the mirrored one."""
    eye = np.array([0.0, 0.0])
    abd = np.array([100.0, 0.0])
    assert estimate_chirality([(eye, np.array([20.0, 40.0]), abd)]) == -1.0
    assert estimate_chirality([(eye, np.array([20.0, -40.0]), abd)]) == 1.0
    assert estimate_chirality([]) == 0.0


# --------------------------------------------------------------------------- #
# whole-frame assembly
# --------------------------------------------------------------------------- #
def test_assemble_builds_a_complete_fly():
    flies = assemble(normal_fly(), 1080, 1080)
    assert len(flies) == 1
    fly = flies[0]
    assert fly.proboscis is not None
    assert fly.abdomen is not None
    assert fly.tip is not None
    assert fly.trackable
    assert fly.rejected == []


def test_assemble_handles_the_two_class_pipeline():
    """Given only eyes and proboscises it gates those and stays quiet about the rest."""
    parts = [p for p in normal_fly() if p.label in (EYE, PROBOSCIS)]
    flies = assemble(parts, 1080, 1080)
    assert len(flies) == 1
    assert flies[0].proboscis is not None
    assert flies[0].abdomen is None
    assert not flies[0].trackable


def test_missing_tip_withholds_the_abdomen():
    """All 970 labelled abdomens had a tip, so a tipless abdomen is not trackable."""
    parts = [p for p in normal_fly() if p.label != ABDOMEN_TIP]
    flies = assemble(parts, 1080, 1080)
    assert flies[0].abdomen is None
    assert not flies[0].trackable
    assert flies[0].rejected


def test_tip_on_the_near_side_is_rejected():
    """The tip must continue AWAY from the eye -- true in 100% of labels."""
    parts = normal_fly()
    eye_c = np.array([300.0, 300.0])
    # put the tip between the eye and the abdomen instead of beyond it
    parts[3] = Part(ABDOMEN_TIP, box(eye_c + AXIS * 3.9 * EYE_SCALE, 0.93 * EYE_SCALE))
    flies = assemble(parts, 1080, 1080)
    assert flies[0].abdomen is None


def test_stacked_flies_are_not_cross_paired():
    """Two flies in a column: each abdomen must go to its own head.

    Nearest-neighbour matching links the upper fly's abdomen to the lower fly's
    eye whenever the bodies overlap; the tip-side rule is what prevents it.
    """
    upper = normal_fly(eye_c=(300.0, 300.0))
    lower = normal_fly(eye_c=(360.0, 480.0))
    flies = assemble(upper + lower, 1080, 1080)
    assert len(flies) == 2
    for fly in flies:
        assert fly.abdomen is not None
        # the abdomen must sit downstream of ITS OWN eye
        offset = fly.abdomen.centroid - fly.eye.centroid
        assert abs(offset[1]) < 2.0 * EYE_SCALE, "paired with the wrong fly's abdomen"


def test_abdomen_withheld_when_the_tip_cannot_be_in_frame():
    """A fly at the right edge is a half-fly: its tip is off-image, so no tracking."""
    complete = normal_fly(eye_c=(300.0, 300.0))
    # a second eye jammed against the right border, with no body parts of its own
    edge_eye = Part(EYE, box((1065.0, 120.0), EYE_SCALE))
    flies = assemble(complete + [edge_eye], 1080, 1080)
    edge = next(f for f in flies if f.eye is edge_eye)
    assert edge.abdomen is None
    assert edge.tip_margin is not None and edge.tip_margin < 0
    assert any("outside frame" in r for r in edge.rejected)


def test_clipped_abdomen_is_diagnosed_as_such():
    """The fly3 case: a real fly at the bottom edge whose tip is off-frame.

    Its abdomen passes distance (4.38) and size (2.59) but its box runs past the
    frame bottom, and no tip is detected because the tip is outside the image.
    The predicted-tip margin stays positive (+1.7) because this fly is angled
    more steeply than the frame-average axis, so the margin alone would give the
    wrong reason -- the clipped box is what identifies it.
    """
    H = W = 1080
    scale = 49.0
    eye_c = np.array([634.0, 878.0])
    # this fly runs at ~41 deg, steeper than the 22 deg frame axis -- which is
    # exactly why the predicted-tip margin stays positive while the fly is cut off
    offset = np.array([163.0, 140.0])      # |offset| = 4.38 eye-widths
    parts = [
        Part(EYE, box(eye_c, scale)),
        Part(PROBOSCIS, box(eye_c + np.array([-40.0, 46.0]), 0.58 * scale)),
        # abdomen in range but running past the bottom border, and no tip at all
        Part(ABDOMEN, box(eye_c + offset, 3.2 * scale, 2.8 * scale)),
    ]
    abdomen = parts[-1]
    assert abdomen.pts[:, 1].max() > H, "test fixture should overrun the frame"
    fly = assemble(parts, W, H)[0]
    assert fly.is_fly                      # it IS a fly: eye + proboscis
    assert not fly.trackable
    assert any("clipped by the frame edge" in r for r in fly.rejected)


def test_touches_border_detects_clipping_on_every_side():
    W = H = 1080
    assert touches_border(Part(ABDOMEN, box((540.0, 1075.0), 40.0)), W, H)
    assert touches_border(Part(ABDOMEN, box((3.0, 540.0), 40.0)), W, H)
    assert not touches_border(Part(ABDOMEN, box((540.0, 540.0), 40.0)), W, H)


def test_reject_reasons_are_recorded_for_audit():
    parts = [p for p in normal_fly() if p.label != ABDOMEN_TIP]
    fly = assemble(parts, 1080, 1080)[0]
    assert fly.rejected and isinstance(fly.rejected[0], str)


def test_summarise_counts_each_part():
    flies = assemble(normal_fly(), 1080, 1080)
    s = summarise(flies)
    assert s == {"flies": 1, "eye_ok": 1, "with_proboscis": 1,
                 "with_abdomen": 1, "trackable": 1}


def test_limits_are_overridable_without_touching_the_module():
    strict = FlyGeometryLimits(d_eye_abd_max=4.0)
    flies = assemble(normal_fly(), 1080, 1080, limits=strict)
    assert flies[0].abdomen is None          # median 4.96 now exceeds the cap


def test_empty_input_is_safe():
    assert assemble([], 1080, 1080) == []


# --------------------------------------------------------------------------- #
# geometry-pipeline extensions: chirality prior and tip-less fallback
# --------------------------------------------------------------------------- #
def test_prior_chirality_gates_a_single_fly_frame():
    """One fly alone cannot vote on handedness; the frozen prior must decide."""
    mirrored = normal_fly(chirality=+1.0)
    assert assemble(mirrored, 1080, 1080)[0].abdomen is not None
    assert assemble(mirrored, 1080, 1080, prior_chirality=+1.0)[0].abdomen is not None
    assert assemble(mirrored, 1080, 1080, prior_chirality=-1.0)[0].abdomen is None


def test_abdomen_only_fallback_records_the_box_without_a_tip():
    parts = [p for p in normal_fly() if p.label != ABDOMEN_TIP]
    strict = assemble(parts, 1080, 1080)[0]
    assert strict.abdomen is None and not strict.abdomen_only
    fly = assemble(parts, 1080, 1080, abdomen_only_fallback=True)[0]
    assert fly.abdomen is not None
    assert fly.abdomen_only
    assert fly.tip is None
    assert not fly.trackable
    assert fly.rejected, "the reason for the missing pair must still be recorded"


def test_abdomen_only_fallback_skips_clipped_and_claimed_abdomens():
    # clipped: the fly3 fixture -- the box runs past the frame bottom
    scale = 49.0
    eye_c = np.array([634.0, 878.0])
    clipped = [
        Part(EYE, box(eye_c, scale)),
        Part(PROBOSCIS, box(eye_c + np.array([-40.0, 46.0]), 0.58 * scale)),
        Part(ABDOMEN, box(eye_c + np.array([163.0, 140.0]), 3.2 * scale, 2.8 * scale)),
    ]
    fly = assemble(clipped, 1080, 1080, abdomen_only_fallback=True)[0]
    assert fly.abdomen is None and not fly.abdomen_only

    # claimed: the upper fly's tip-paired abdomen must not be handed to the lower fly
    upper = normal_fly(eye_c=(300.0, 300.0))
    lower = [p for p in normal_fly(eye_c=(360.0, 480.0)) if p.label != ABDOMEN_TIP]
    flies = assemble(upper + lower, 1080, 1080, abdomen_only_fallback=True)
    assert flies[0].abdomen is not None and not flies[0].abdomen_only
    assert flies[1].abdomen is not None and flies[1].abdomen_only
    assert flies[0].abdomen is not flies[1].abdomen
