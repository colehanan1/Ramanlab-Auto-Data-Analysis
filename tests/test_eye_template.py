"""Batch-level eye anchor template.

The 5-frame warm-up scan in ``yolo_infer`` decides how many fly slots a trial
gets from that one video alone. A fly whose eye sits at the frame edge is
detected in some trials and not others, so the slot count -- and with it the
fly numbering -- changes from trial to trial inside one batch
(``IAA-Sensitivity-6-1/october_09_batch_3``: 4 flies in 3 of 21 trials, 3 in
the other 18, so ``fly1`` names two different physical flies).

The template fixes the slot set once per batch, from every trial's first
frames, and lends a missing slot its anchor. A trial's own detections always
win where it has them; only what it missed is borrowed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbpipe.utils import eye_template as et  # noqa: E402


def _box(cx: float, cy: float, w: float = 60.0, h: float = 52.0) -> np.ndarray:
    """An eye box centred on ``(cx, cy)`` -- roughly the size YOLO emits."""
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _boxes(*centres) -> np.ndarray:
    if not centres:
        return np.zeros((0, 4), np.float32)
    return np.stack([_box(cx, cy) for cx, cy in centres], axis=0)


# ── Real geometry from october_09_batch_3 ────────────────────────────────────
# The top fly's eye centre sits 4 px from the frame edge; the three below it
# are detected in every trial. Positions drift ~10 px over the session.
EDGE_FLY = (721.0, 4.0)
ALWAYS_SEEN = [(635.0, 280.0), (495.0, 570.0), (467.0, 830.0)]


def _batch_october_09_batch_3() -> dict:
    """21 trials: 3 see the edge fly, 18 do not. Anchors drift as the session runs."""
    trials = {}
    for i in range(21):
        drift = -0.5 * i  # the mount settles ~10 px over 21 trials
        centres = [(x, y + drift) for x, y in ALWAYS_SEEN]
        if i in (1, 2, 6):  # pretest_2, pretest_3, pretest_7
            centres = [EDGE_FLY] + centres
        trials[f"trial_{i:02d}"] = _boxes(*centres)
    return trials


# ── Building the template ────────────────────────────────────────────────────


def test_slot_seen_in_only_one_trial_is_excluded_at_min_trials_two():
    trials = {
        "a": _boxes((500.0, 200.0), (500.0, 600.0)),
        "b": _boxes((500.0, 200.0), (500.0, 600.0)),
        "c": _boxes((500.0, 200.0), (500.0, 600.0), (500.0, 950.0)),
    }
    slots = et.build_template(trials, min_trials=2)
    assert [(round(s.cx), round(s.cy)) for s in slots] == [(500, 200), (500, 600)]


def test_slot_seen_in_two_trials_is_included_at_min_trials_two():
    trials = {
        "a": _boxes((500.0, 200.0), (500.0, 600.0)),
        "b": _boxes((500.0, 200.0), (500.0, 600.0), (500.0, 950.0)),
        "c": _boxes((500.0, 200.0), (500.0, 600.0), (500.0, 950.0)),
    }
    slots = et.build_template(trials, min_trials=2)
    assert [(round(s.cx), round(s.cy)) for s in slots] == [
        (500, 200),
        (500, 600),
        (500, 950),
    ]


def test_slots_are_ordered_top_to_bottom_with_x_as_tie_break():
    """Same convention as EyeAnchorManager's ``np.lexsort((xs, ys))``."""
    trials = {
        "a": _boxes((800.0, 900.0), (300.0, 200.0), (700.0, 200.0)),
        "b": _boxes((800.0, 900.0), (300.0, 200.0), (700.0, 200.0)),
    }
    slots = et.build_template(trials, min_trials=2)
    assert [(round(s.cx), round(s.cy)) for s in slots] == [
        (300, 200),
        (700, 200),
        (800, 900),
    ]


def test_one_eye_drifting_across_trials_is_a_single_slot():
    """Anchors settle ~10 px over a session; that is one fly, not twenty."""
    trials = {
        f"t{i:02d}": _boxes((635.0, 280.0 - 0.5 * i), (495.0, 570.0 - 0.5 * i))
        for i in range(21)
    }
    slots = et.build_template(trials, min_trials=2)
    assert len(slots) == 2
    assert all(s.n_trials_seen == 21 for s in slots)


def test_slot_centre_is_the_median_of_its_contributing_detections():
    trials = {
        "a": _boxes((500.0, 100.0)),
        "b": _boxes((500.0, 110.0)),
        "c": _boxes((500.0, 120.0)),
    }
    slots = et.build_template(trials, min_trials=2)
    assert len(slots) == 1
    assert slots[0].cy == pytest.approx(110.0, abs=0.5)


def test_two_flies_mounted_176_px_apart_stay_separate_slots():
    """The closest mounts seen are 176-178 px apart -- never merge those."""
    trials = {
        "a": _boxes((500.0, 400.0), (500.0, 576.0)),
        "b": _boxes((502.0, 402.0), (498.0, 578.0)),
    }
    slots = et.build_template(trials, min_trials=2)
    assert len(slots) == 2


def test_template_is_independent_of_trial_order():
    """Workers see a batch's trials sliced by ``_job_idx % num_workers``; the
    template must not depend on which worker built it."""
    trials = _batch_october_09_batch_3()
    forward = et.build_template(trials, min_trials=2)
    reverse = et.build_template(dict(reversed(list(trials.items()))), min_trials=2)
    assert [(s.cx, s.cy, s.n_trials_seen) for s in forward] == [
        (s.cx, s.cy, s.n_trials_seen) for s in reverse
    ]


def test_october_09_batch_3_gets_four_slots_with_the_edge_fly_first():
    """The regression this whole module exists for."""
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    assert len(slots) == 4
    assert (round(slots[0].cx), round(slots[0].cy)) == (721, 4)
    assert slots[0].n_trials_seen == 3
    assert [s.n_trials_seen for s in slots[1:]] == [21, 21, 21]


def test_a_fly_seen_in_a_single_trial_leaves_the_batch_at_three_slots():
    """june_03_batch_2: 4 flies in 1 of 15 trials. min_trials=2 drops it, so
    every trial in the batch numbers its flies the same way."""
    trials = {f"t{i:02d}": _boxes(*ALWAYS_SEEN) for i in range(15)}
    trials["t07"] = _boxes(EDGE_FLY, *ALWAYS_SEEN)
    slots = et.build_template(trials, min_trials=2)
    assert len(slots) == 3


def test_an_empty_batch_yields_no_slots():
    assert et.build_template({}, min_trials=2) == []


def test_a_batch_where_nothing_was_detected_yields_no_slots():
    trials = {"a": _boxes(), "b": _boxes()}
    assert et.build_template(trials, min_trials=2) == []


# ── Applying the template to one trial ───────────────────────────────────────


def test_matched_slots_keep_the_trials_own_box_not_the_templates():
    """Anchors drift over a session. Freezing every trial at the template
    position would read as a standing eye_drift_px on every fly."""
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    own = _boxes((634.0, 272.0), (493.0, 562.0), (466.0, 822.0))
    seeded, borrowed = et.seed_anchor_boxes(slots, own, match_radius_px=120.0)

    assert len(seeded) == 4
    assert borrowed == [True, False, False, False]
    for slot_box, own_box in zip(seeded[1:], own):
        assert np.allclose(slot_box, own_box)


def test_a_missing_slot_borrows_the_template_box():
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    own = _boxes((634.0, 272.0), (493.0, 562.0), (466.0, 822.0))
    seeded, borrowed = et.seed_anchor_boxes(slots, own, match_radius_px=120.0)

    cx, cy = (seeded[0][0] + seeded[0][2]) / 2, (seeded[0][1] + seeded[0][3]) / 2
    assert (round(float(cx)), round(float(cy))) == (721, 4)
    assert borrowed[0] is True


def test_a_trial_that_detects_every_slot_borrows_nothing():
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    own = _boxes(EDGE_FLY, (634.0, 272.0), (493.0, 562.0), (466.0, 822.0))
    seeded, borrowed = et.seed_anchor_boxes(slots, own, match_radius_px=120.0)
    assert borrowed == [False, False, False, False]
    for slot_box, own_box in zip(seeded, own):
        assert np.allclose(slot_box, own_box)


def test_slot_order_is_the_templates_regardless_of_detection_order():
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    shuffled = _boxes((466.0, 822.0), (634.0, 272.0), (493.0, 562.0))
    seeded, _ = et.seed_anchor_boxes(slots, shuffled, match_radius_px=120.0)
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in seeded]
    assert [round(float(cy)) for _, cy in centres] == [4, 272, 562, 822]


def test_a_detection_matching_no_slot_is_ignored():
    """The template is authoritative: a stray eye cannot add a fly mid-batch."""
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    own = _boxes((634.0, 272.0), (493.0, 562.0), (466.0, 822.0), (100.0, 1000.0))
    seeded, borrowed = et.seed_anchor_boxes(slots, own, match_radius_px=120.0)
    assert len(seeded) == 4
    assert borrowed == [True, False, False, False]


def test_a_trial_with_no_detections_borrows_every_slot():
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    seeded, borrowed = et.seed_anchor_boxes(slots, _boxes(), match_radius_px=120.0)
    assert len(seeded) == 4
    assert borrowed == [True, True, True, True]


def test_no_two_slots_take_the_same_detection():
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    own = _boxes((500.0, 566.0))  # nearest to slot 2, in range of slot 1 too
    seeded, borrowed = et.seed_anchor_boxes(slots, own, match_radius_px=400.0)
    assert borrowed.count(False) == 1


# ── Cache file ───────────────────────────────────────────────────────────────


def test_template_round_trips_through_its_json_cache(tmp_path):
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    path = et.template_path(tmp_path)
    et.save_template(path, slots, min_trials=2)
    loaded = et.load_template(path)
    assert [(s.cx, s.cy, s.n_trials_seen, s.trials) for s in loaded] == [
        (s.cx, s.cy, s.n_trials_seen, s.trials) for s in slots
    ]


def test_the_cache_names_the_trials_that_saw_each_slot(tmp_path):
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    path = et.template_path(tmp_path)
    et.save_template(path, slots, min_trials=2)
    payload = json.loads(path.read_text())
    assert payload["slots"][0]["trials"] == ["trial_01", "trial_02", "trial_06"]
    assert payload["min_trials"] == 2


def test_loading_a_missing_cache_returns_none(tmp_path):
    assert et.load_template(et.template_path(tmp_path)) is None


def test_loading_a_corrupt_cache_returns_none_rather_than_raising(tmp_path):
    path = et.template_path(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    assert et.load_template(path) is None


def test_a_cache_built_under_a_different_min_trials_is_not_reused(tmp_path):
    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    path = et.template_path(tmp_path)
    et.save_template(path, slots, min_trials=2)
    assert et.load_template(path, min_trials=3) is None
    assert et.load_template(path, min_trials=2) is not None


# ── Seeding EyeAnchorManager ─────────────────────────────────────────────────


def test_a_seeded_manager_is_confirmed_immediately():
    """Unseeded, the manager only confirms once it has gathered max_eyes eyes.
    A trial that never shows the edge fly would otherwise keep accreting
    anchors for the whole video."""
    from fbpipe.utils.multi_fly import EyeAnchorManager

    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    seeded, _ = et.seed_anchor_boxes(slots, _boxes(), match_radius_px=120.0)

    mgr = EyeAnchorManager(max_eyes=len(slots))
    mgr.seed_from_boxes(seeded)
    assert mgr.confirmed is True
    assert len(mgr.anchor_ids) == 4


def test_a_seeded_manager_keeps_the_template_slot_order():
    from fbpipe.utils.multi_fly import EyeAnchorManager

    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    own = _boxes((634.0, 272.0), (493.0, 562.0), (466.0, 822.0))
    seeded, _ = et.seed_anchor_boxes(slots, own, match_radius_px=120.0)

    mgr = EyeAnchorManager(max_eyes=len(slots))
    mgr.seed_from_boxes(seeded)
    assert [round(cy) for _, cy in mgr.get_centers()] == [4, 272, 562, 822]


def test_a_seeded_manager_ignores_later_detections():
    """The anchors are frozen the moment the template is applied; nothing a
    later frame detects may move or add one."""
    from fbpipe.utils.multi_fly import EyeAnchorManager

    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    seeded, _ = et.seed_anchor_boxes(slots, _boxes(), match_radius_px=120.0)

    mgr = EyeAnchorManager(max_eyes=len(slots))
    mgr.seed_from_boxes(seeded)
    before = mgr.get_centers()
    mgr.try_update_from_dets(_boxes((50.0, 50.0)), np.array([0.9], np.float32))
    assert mgr.get_centers() == before


def test_seeded_anchor_ids_are_unique_and_run_top_to_bottom():
    from fbpipe.utils.multi_fly import EyeAnchorManager

    slots = et.build_template(_batch_october_09_batch_3(), min_trials=2)
    seeded, _ = et.seed_anchor_boxes(slots, _boxes(), match_radius_px=120.0)

    mgr = EyeAnchorManager(max_eyes=len(slots))
    mgr.seed_from_boxes(seeded)
    assert mgr.anchor_ids == sorted(set(mgr.anchor_ids))
    assert len(mgr.anchor_ids) == 4


# ── A slot must be a fly we can actually measure ─────────────────────────────
#
# june_03_batch_2's 4th eye sits 21 px from the BOTTOM edge: two trials of
# fifteen detect it, so it cleared min_trials=2, but its proboscis is entirely
# out of frame -- 0 % paired frames across all 15 trials. An eye alone is not a
# measurable fly, so the template asks for a proboscis inside the eye's gate
# too. october_09_batch_3's edge fly pairs in 100 % of frames and stays.


def test_a_slot_whose_eye_never_pairs_a_proboscis_is_rejected():
    trials = {
        "a": _boxes((500.0, 200.0), (500.0, 1059.0)),
        "b": _boxes((500.0, 200.0), (500.0, 1059.0)),
    }
    paired = {"a": [True, False], "b": [True, False]}
    slots = et.build_template(
        trials, trial_paired=paired, require_proboscis=True, min_trials=2
    )
    assert [(round(s.cx), round(s.cy)) for s in slots] == [(500, 200)]


def test_a_slot_that_pairs_in_even_one_trial_is_kept():
    """The edge fly is hard to detect, not hard to measure once detected."""
    trials = {
        "a": _boxes((500.0, 200.0), (721.0, 4.0)),
        "b": _boxes((500.0, 200.0), (721.0, 4.0)),
        "c": _boxes((500.0, 200.0), (721.0, 4.0)),
    }
    paired = {"a": [True, False], "b": [True, False], "c": [True, True]}
    slots = et.build_template(
        trials, trial_paired=paired, require_proboscis=True, min_trials=2
    )
    assert len(slots) == 2
    assert (round(slots[0].cx), round(slots[0].cy)) == (721, 4)


def test_the_requirement_can_be_turned_off():
    trials = {
        "a": _boxes((500.0, 200.0), (500.0, 1059.0)),
        "b": _boxes((500.0, 200.0), (500.0, 1059.0)),
    }
    paired = {"a": [True, False], "b": [True, False]}
    slots = et.build_template(
        trials, trial_paired=paired, require_proboscis=False, min_trials=2
    )
    assert len(slots) == 2


def test_without_pairing_information_every_slot_is_kept():
    """A caller that cannot say (an old cache, a test stand-in) must not have
    all of its flies silently rejected."""
    trials = {
        "a": _boxes((500.0, 200.0), (500.0, 1059.0)),
        "b": _boxes((500.0, 200.0), (500.0, 1059.0)),
    }
    slots = et.build_template(trials, require_proboscis=True, min_trials=2)
    assert len(slots) == 2


def test_a_slot_records_whether_it_ever_paired():
    trials = {
        "a": _boxes((500.0, 200.0)),
        "b": _boxes((500.0, 200.0)),
    }
    slots = et.build_template(
        trials, trial_paired={"a": [False], "b": [True]}, require_proboscis=True,
        min_trials=2,
    )
    assert slots[0].n_trials_paired == 1
