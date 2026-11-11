"""Dataset-wide fly numbers (fbpipe.utils.fly_registry).

Fly 1 is the earliest fly of the dataset; later flies count up. A number,
once given, never changes: not when a re-run finds a new fly above fly 1 of
an old batch, not when a fly is flagged, not when a fly disappears.
"""
import pandas as pd
import pytest

from fbpipe.utils import fly_registry as fr


def _flies(rows):
    return pd.DataFrame(rows, columns=["batch", "fly_number", "eye_x", "eye_y", "first_recorded", "flagged"])


BASE = [
    ("oct_02_batch_1", 1, 600.0, 300.0, "20261002_101500", False),
    ("oct_02_batch_1", 2, 590.0, 600.0, "20261002_101500", True),
    ("oct_01_batch_1", 1, 610.0, 200.0, "20261001_093000", False),
    ("oct_01_batch_1", 2, 605.0, 500.0, "20261001_093000", False),
    ("oct_01_batch_1", 3, 600.0, 800.0, "20261001_093000", False),
]


def _number(reg, batch, eye_y):
    row = reg[(reg.batch == batch) & (reg.eye_y == eye_y)]
    assert len(row) == 1
    return int(row.dataset_fly.iloc[0])


def test_first_numbering_is_by_date_then_top_to_bottom():
    reg = fr.update_registry(_flies(BASE), None)
    assert _number(reg, "oct_01_batch_1", 200.0) == 1
    assert _number(reg, "oct_01_batch_1", 500.0) == 2
    assert _number(reg, "oct_01_batch_1", 800.0) == 3
    assert _number(reg, "oct_02_batch_1", 300.0) == 4
    assert _number(reg, "oct_02_batch_1", 600.0) == 5
    assert reg.present.all()


def test_new_batch_is_numbered_on_top():
    reg = fr.update_registry(_flies(BASE), None)
    more = BASE + [("oct_03_batch_1", 1, 600.0, 400.0, "20261003_100000", False)]
    reg2 = fr.update_registry(_flies(more), reg)
    assert _number(reg2, "oct_03_batch_1", 400.0) == 6
    for b, y in [("oct_01_batch_1", 200.0), ("oct_02_batch_1", 600.0)]:
        assert _number(reg2, b, y) == _number(reg, b, y)


def test_fly_found_above_fly_1_of_an_old_batch_goes_on_top_and_renumbers_nobody():
    reg = fr.update_registry(_flies(BASE), None)
    # a re-run finds a fly at the top of oct_01_batch_1: every slot below shifts by one
    rerun = [
        ("oct_02_batch_1", 1, 600.0, 300.0, "20261002_101500", False),
        ("oct_02_batch_1", 2, 590.0, 600.0, "20261002_101500", True),
        ("oct_01_batch_1", 1, 640.0, 15.0, "20261001_093000", False),
        ("oct_01_batch_1", 2, 611.0, 201.0, "20261001_093000", False),
        ("oct_01_batch_1", 3, 606.0, 499.0, "20261001_093000", False),
        ("oct_01_batch_1", 4, 600.0, 801.0, "20261001_093000", False),
    ]
    reg2 = fr.update_registry(_flies(rerun), reg)
    assert _number(reg2, "oct_01_batch_1", 15.0) == 6           # added on top
    old = reg2[reg2.dataset_fly <= 5].sort_values("dataset_fly")
    assert list(old.fly_number) == [2, 3, 4, 1, 2]               # slot numbers follow the re-run
    assert list(old.dataset_fly) == [1, 2, 3, 4, 5]               # dataset numbers do not move


def test_flagged_fly_keeps_its_number():
    reg = fr.update_registry(_flies(BASE), None)
    flipped = [r[:5] + (not r[5],) for r in BASE]
    reg2 = fr.update_registry(_flies(flipped), reg)
    assert list(reg2.sort_values("dataset_fly").dataset_fly) == [1, 2, 3, 4, 5]
    assert bool(reg2.loc[reg2.dataset_fly == 4, "flagged"].iloc[0]) is True
    assert bool(reg2.loc[reg2.dataset_fly == 5, "flagged"].iloc[0]) is False


def test_missing_fly_keeps_its_number_and_it_is_never_reused():
    reg = fr.update_registry(_flies(BASE), None)
    without = [r for r in BASE if not (r[0] == "oct_01_batch_1" and r[1] == 2)]
    reg2 = fr.update_registry(_flies(without), reg)
    gone = reg2[reg2.dataset_fly == 2]
    assert len(gone) == 1 and not bool(gone.present.iloc[0])
    reg3 = fr.update_registry(_flies(without + [("oct_09_batch_1", 1, 600.0, 400.0, "20261009_100000", False)]), reg2)
    assert _number(reg3, "oct_09_batch_1", 400.0) == 6


def test_same_input_twice_is_stable():
    reg = fr.update_registry(_flies(BASE), None)
    again = fr.update_registry(_flies(BASE), reg)
    pd.testing.assert_frame_equal(
        reg.sort_values("dataset_fly").reset_index(drop=True),
        again.sort_values("dataset_fly").reset_index(drop=True),
    )


def test_eye_beyond_match_radius_is_a_new_fly():
    reg = fr.update_registry(_flies(BASE[:1]), None)
    moved = [("oct_02_batch_1", 1, 600.0, 420.0, "20261002_101500", False)]
    reg2 = fr.update_registry(_flies(moved), reg, match_px=60.0)
    assert sorted(reg2.dataset_fly) == [1, 2]
    assert not bool(reg2.loc[reg2.dataset_fly == 1, "present"].iloc[0])


def test_label():
    assert fr.label(7) == "Fly 7"
