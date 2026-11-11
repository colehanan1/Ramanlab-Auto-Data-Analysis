"""Fly identity across the anchor/live comparison arms is decided by eye position.

The live arm's november_07_batch_1 gained a fly at the top of the frame, so
every fly number below it shifted by one; matching by number would compare
different flies.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "live_eye_compare", ROOT / "scripts" / "analysis" / "live_eye_compare.py"
)
lec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lec)


def test_same_slots_map_one_to_one():
    eyes = {1: (594.0, 297.0), 2: (506.0, 595.0), 3: (454.0, 807.0)}
    assert lec.match_flies_by_eye(eyes, dict(eyes)) == {1: 1, 2: 2, 3: 3}


def test_added_top_fly_shifts_numbers_and_has_no_partner():
    anchor = {1: (594.0, 297.0), 2: (506.0, 595.0), 3: (454.0, 807.0)}
    live = {1: (656.0, 13.0), 2: (595.0, 295.0), 3: (507.0, 595.0), 4: (455.0, 807.0)}
    assert lec.match_flies_by_eye(anchor, live) == {2: 1, 3: 2, 4: 3, 1: None}


def test_far_eyes_are_not_matched():
    assert lec.match_flies_by_eye({1: (100.0, 100.0)}, {1: (400.0, 400.0)}) == {1: None}
