"""StablePairing must respect the eye->proboscis gate, not just raw distance.

Regression for october_07_batch_1_pretest_1 (Flybehavior2, 2026-10-07): a
sticky binding kept a proboscis on the wrong eye because nothing re-checked the
gate once the pair was formed, so the owning eye could never claim it.
"""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from fbpipe.config import ProboscisFilterSettings, Settings
from fbpipe.utils.multi_fly import StablePairing


def _track(track_id, cx, cy):
    return SimpleNamespace(
        id=track_id,
        box_xyxy=np.array([cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0], dtype=np.float32),
    )


def _below_only(eye, prob):
    """Toy gate: the proboscis must sit at or below its eye."""
    return prob[1] >= eye[1]


def test_step_without_accept_keeps_legacy_nearest_binding():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=220.0)
    pairer.step([1, 2], [(0.0, 0.0), (0.0, 300.0)], [_track(7, 0.0, 250.0)])
    # Eye 1 is visited first and 250 > 220, so it stays empty; eye 2 is 50 px away.
    assert pairer.eye_to_cls8 == {1: None, 2: 7}


def test_step_never_binds_a_track_the_gate_rejects():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=220.0)
    # Track is 50 px ABOVE eye 2 (nearest) and 250 px below eye 1 (too far).
    pairer.step([1, 2], [(0.0, 0.0), (0.0, 300.0)], [_track(7, 0.0, 250.0)], accept=_below_only)
    assert pairer.eye_to_cls8 == {1: None, 2: None}


def test_step_releases_sticky_binding_that_leaves_gate_and_owner_claims_it_same_frame():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=220.0)
    pairer.eye_to_cls8 = {1: None, 2: 7}
    # Track 7 is 80 px below eye 1 and 220 px above eye 2.
    pairer.step([1, 2], [(0.0, 0.0), (0.0, 300.0)], [_track(7, 0.0, 80.0)], accept=_below_only)
    assert pairer.eye_to_cls8 == {1: 7, 2: None}


def test_step_keeps_sticky_binding_the_gate_accepts():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=220.0)
    pairer.eye_to_cls8 = {1: None, 2: 7}
    # Track 7 is now nearer eye 1 but still legal for eye 2: stickiness wins.
    pairer.step([1, 2], [(0.0, 0.0), (0.0, 120.0)], [_track(7, 0.0, 130.0)], accept=_below_only)
    assert pairer.eye_to_cls8 == {1: None, 2: 7}


# --- cost mode: joint assignment by depth inside each eye's gate -------------
# Regression for IAA-Sensitivity-24-1/october_07_batch_2_pretest_5 (Flybehavior2,
# 2026-10-07): eyes 1 and 2 sit only 178 px apart, so their 280 px gates overlap.
# Eye 1, visited first, grabbed the proboscis hanging 55 px under eye 2 and kept
# it (sticky) in 2,395 of 3,598 frames, while its own proboscis went unpaired.


def _gate_cost(eye, prob, max_px=280.0, up=70.0):
    """Toy anisotropic gate: 0 at the eye, 1 at the edge, inf outside."""
    dx, dy = prob[0] - eye[0], prob[1] - eye[1]
    b = up if dy < 0 else max_px
    rho = ((dx / max_px) ** 2 + (dy / b) ** 2) ** 0.5
    return rho if rho <= 1.0 else float("inf")


# Real slot geometry from that trial (eye centres, px).
_EYES = [(754.0, 116.0), (688.0, 423.0), (675.0, 601.0), (625.0, 875.0)]


def test_cost_mode_gives_each_close_fly_its_own_proboscis():
    pairer = StablePairing(max_pairs=4, rebind_max_dist_px=305.0)
    # Frame 1400: track 18 hangs 55 px under eye 2 but eye 1 holds it; track 19
    # is eye 1's own proboscis, 73 px under it, unpaired.
    pairer.eye_to_cls8 = {1: 10, 2: 18, 3: None, 4: 8}
    tracks = [_track(10, 753.0, 170.0), _track(18, 699.0, 656.0),
              _track(19, 649.0, 496.0), _track(8, 615.0, 953.0)]
    pairer.step([1, 2, 3, 4], _EYES, tracks, cost=_gate_cost)
    assert pairer.eye_to_cls8 == {1: 10, 2: 19, 3: 18, 4: 8}


def test_cost_mode_lower_fly_wins_a_lone_proboscis_hanging_just_under_its_eye():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    pairer.step([1, 2], _EYES[1:3], [_track(18, 699.0, 656.0)], cost=_gate_cost)
    assert pairer.eye_to_cls8 == {1: None, 2: 18}


def test_cost_mode_stickiness_breaks_near_ties_only():
    # 145 px under the upper eye, 33 px above the lower one (near the boundary,
    # ~142 px down at this spacing): the lower eye's cost is only a little lower,
    # so a track the upper eye already holds stays; an unheld one goes down.
    prob = (682.0, 568.0)
    up, low = _gate_cost(_EYES[1], prob), _gate_cost(_EYES[2], prob)
    assert 0.0 < up - low < 0.05  # a literal margin: never the constant under test
    held = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    held.eye_to_cls8 = {1: 7, 2: None}
    held.step([1, 2], _EYES[1:3], [_track(7, *prob)], cost=_gate_cost)
    assert held.eye_to_cls8 == {1: 7, 2: None}
    fresh = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    fresh.step([1, 2], _EYES[1:3], [_track(7, *prob)], cost=_gate_cost)
    assert fresh.eye_to_cls8 == {1: None, 2: 7}


def test_cost_mode_assigns_jointly_not_greedily():
    # Greedy-by-cheapest would hand track 5 to eye 1 (0.30 < 0.32) and strand
    # eye 2; the joint optimum gives eye 1 its second choice.
    costs = {(1, 5): 0.30, (1, 6): 0.40, (2, 5): 0.32}
    eyes = [(0.0, 0.0), (0.0, 1000.0)]
    pos = {5: (10.0, 10.0), 6: (60.0, 60.0)}   # far enough apart to be two proboscises
    def cost(eye, prob):
        e = 1 if eye == eyes[0] else 2
        t = 5 if prob == pos[5] else 6
        return costs.get((e, t), float("inf"))
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=5000.0)
    pairer.step([1, 2], eyes, [_track(5, *pos[5]), _track(6, *pos[6])], cost=cost)
    assert pairer.eye_to_cls8 == {1: 6, 2: 5}


def test_cost_mode_drops_bindings_outside_the_gate_and_dead_tracks():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    pairer.eye_to_cls8 = {1: 99, 2: 7}   # 99 is gone; 7 sits above eye 2's tight top
    pairer.step([1, 2], _EYES[1:3], [_track(7, 675.0, 450.0)], cost=_gate_cost)
    assert pairer.eye_to_cls8 == {1: 7, 2: None}


def test_cost_mode_new_binding_still_respects_the_rebind_radius():
    pairer = StablePairing(max_pairs=1, rebind_max_dist_px=100.0)
    pairer.step([1], [_EYES[1]], [_track(7, 688.0, 573.0)], cost=_gate_cost)  # 150 px, in gate
    assert pairer.eye_to_cls8 == {1: None}


def test_cost_mode_exact_tie_goes_to_the_upper_slot():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    pairer.step([1, 2], [(0.0, 0.0), (400.0, 0.0)], [_track(7, 200.0, 0.0)], cost=_gate_cost)
    assert pairer.eye_to_cls8 == {1: 7, 2: None}


def test_cost_mode_one_proboscis_seen_as_two_tracks_goes_to_one_eye_only():
    # Same trial, frames 1053-1058 and 1209-1218: the tracker briefly held two
    # tracks on fly 2's single proboscis (1 px apart) while fly 1's own proboscis
    # was undetected, and eye 1 took the duplicate.
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    pairer.eye_to_cls8 = {1: 5, 2: 2}
    pairer.step([1, 2], _EYES[1:3], [_track(2, 702.0, 671.0), _track(5, 702.0, 670.0)], cost=_gate_cost)
    assert pairer.eye_to_cls8 == {1: None, 2: 2}


def test_cost_mode_keeps_two_genuinely_separate_proboscises():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    pairer.step([1, 2], _EYES[1:3], [_track(19, 649.0, 496.0), _track(18, 699.0, 656.0)], cost=_gate_cost)
    assert pairer.eye_to_cls8 == {1: 19, 2: 18}


def test_cost_mode_keeps_two_proboscises_25_px_apart():
    # Two flies' tips can meet between close mounts; 25 px apart they are two.
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    pairer.step([1, 2], [(0.0, 0.0), (400.0, 0.0)], [_track(5, 190.0, 50.0), _track(6, 215.0, 50.0)], cost=_gate_cost)
    assert pairer.eye_to_cls8 == {1: 5, 2: 6}


def test_cost_mode_takes_tracks_12_px_apart_for_one_proboscis():
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    pairer.step([1, 2], [(0.0, 0.0), (400.0, 0.0)], [_track(5, 195.0, 50.0), _track(6, 207.0, 50.0)], cost=_gate_cost)
    assert len([t for t in pairer.eye_to_cls8.values() if t is not None]) == 1


# --- cost mode with the production gate depth (280 px gate, up_divisor 4) ----
# How much a held binding is worth. Too little, and a genuine deep PER is
# handed to -- or abandoned for -- a neighbour's shallow claim; too much, and a
# binding as wrong as pretest_5's is never handed over.


def _real_gate_cost():
    yolo_infer = pytest.importorskip("fbpipe.steps.yolo_infer")
    settings = Settings(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=2,
        proboscis_filter=ProboscisFilterSettings(max_eye_prob_distance_px=280.0, up_divisor=4.0),
    )
    return lambda eye, prob: yolo_infer._eye_prob_gate_depth(eye, prob, settings, 2)


def test_cost_mode_never_abandons_a_held_track_only_its_eye_can_take():
    # Both flies extended ~258 px (one odor pulse reaches every fly). Track 2 lies
    # in eye 1002's gate only; the upper fly's tip (track 1) hangs 42 px above eye
    # 1002, inside its 70 px upward reach. Handing track 1 down would cost the
    # upper fly its PER and leave the lower fly's own PER unpaired.
    cost = _real_gate_cost()
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    pairer.eye_to_cls8 = {1001: 1, 1002: 2}
    tracks = [_track(1, 543.0, 558.0), _track(2, 537.0, 858.0)]
    pairer.step([1001, 1002], [(540.0, 300.0), (540.0, 600.0)], tracks, cost=cost)
    assert pairer.eye_to_cls8 == {1001: 1, 1002: 2}


@pytest.mark.parametrize("peak", [150.0, 160.0])
def test_cost_mode_upper_fly_keeps_its_per_as_it_nears_a_trackless_lower_eye(peak):
    # Close mount (october_07_batch_2 spacing: eyes 178 px apart). The upper fly's
    # proboscis extends 60 px -> peak -> 60 px; at its deepest it sits 28 / 18 px
    # above the lower eye, inside that eye's upward reach. The lower eye has no
    # track. The upper eye must keep its own PER in every frame.
    cost = _real_gate_cost()
    upper, lower = (540.0, 300.0), (540.0, 478.0)
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.0)
    reach = list(np.linspace(60.0, peak, 40)) + [peak] * 10 + list(np.linspace(peak, 60.0, 40))
    for d in reach:
        pairer.step([1001, 1002], [upper, lower], [_track(1, 543.0, 300.0 + d)], cost=cost)
        assert pairer.eye_to_cls8 == {1001: 1, 1002: None}, f"lost the PER {d:.0f} px down"


def test_cost_mode_hands_over_a_held_proboscis_far_deeper_in_its_holders_gate():
    # pretest_5 with fly 1's own proboscis undetected: eye 1 holds track 18, fly
    # 2's proboscis 49 px under eye 2 -- 0.82 deep in eye 1's gate, 0.24 in eye
    # 2's. Hysteresis must not keep a binding this clearly wrong.
    cost = _real_gate_cost()
    upper, lower, prob = (688.0, 423.0), (675.0, 601.0), (720.0, 650.0)
    assert 0.55 < cost(upper, prob) - cost(lower, prob) < 0.6  # a literal margin
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=305.5)
    pairer.eye_to_cls8 = {1: 18, 2: None}
    pairer.step([1, 2], [upper, lower], [_track(18, *prob)], cost=cost)
    assert pairer.eye_to_cls8 == {1: None, 2: 18}
