"""scripts/analysis/b2b_stats.py: batch-level tests for the b2b figures (user 2026-11-18)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analysis import b2b_stats as bs  # noqa: E402


def test_session_is_the_folder_without_its_rig_suffix():
    assert bs.session_of("november_13_batch_1_rig_3") == "november_13_batch_1"
    assert bs.session_of("november_13_batch_1") == "november_13_batch_1"


def test_fly_sign_flip_is_exact_on_the_half_grid():
    assert bs.sign_flip_p([1, 1, 1, 1]) == pytest.approx(2 / 16)
    assert bs.sign_flip_p([1, 1, 1, 0]) == pytest.approx(2 / 8)    # a zero carries no sign
    assert bs.sign_flip_p([0.5, 1.5, -0.5]) == pytest.approx(6 / 8)


def test_batch_sign_flip_flips_every_fly_of_a_batch_together():
    # 3 batches, all positive: 2 of the 8 patterns reach |sum| -> 2/8, however many flies
    vals = [1, 1, 2, 0.5, 0.5, 3]
    batches = ["a", "a", "b", "b", "b", "c"]
    r = bs.batch_sign_flip(vals, batches)
    assert r["p"] == pytest.approx(2 / 8) and r["exact"] and r["n_batches"] == 3
    assert r["estimate"] == pytest.approx(np.mean(vals))      # per-fly mean, flies pooled
    # one batch carrying everything: nothing to flip against
    assert bs.batch_sign_flip([1, 2, 3], ["a"] * 3)["p"] == pytest.approx(1.0)


def test_batch_ci_inverts_the_same_test():
    rng = np.random.default_rng(3)
    for shift in (0.0, 0.4, 1.0):
        vals = rng.normal(shift, 1.0, 40)
        batches = np.repeat(np.arange(10), 4)
        r = bs.batch_sign_flip(vals, batches, ci=True)
        assert r["ci_lo"] <= r["estimate"] <= r["ci_hi"]
        assert (r["ci_lo"] > 0 or r["ci_hi"] < 0) == (r["p"] < 0.05)


def test_too_few_batches_leave_the_ci_open():
    r = bs.batch_sign_flip([1, 1, 1, 1], ["a", "b", "c", "d"], ci=True)   # min p = 2/16
    assert r["p"] == pytest.approx(0.125)
    assert np.isnan(r["ci_lo"]) and np.isnan(r["ci_hi"])


def test_label_permutation_keeps_each_session_s_counts():
    # 2 sessions x (1 A + 2 B batches): 3 x 3 = 9 assignments, enumerated
    vals = [5, 5, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0]
    batches = ["s1_a", "s1_a", "s1_b", "s1_b", "s1_c", "s1_c",
               "s2_a", "s2_a", "s2_b", "s2_b", "s2_c", "s2_c"]
    labels = {"s1_a": "A", "s1_b": "B", "s1_c": "B", "s2_a": "A", "s2_b": "B", "s2_c": "B"}
    strata = {b: b[:2] for b in labels}
    r = bs.batch_label_perm(vals, batches, labels, strata)
    assert r["exact"] and r["n_perm"] == 9
    assert r["p"] == pytest.approx(1 / 9)             # only the observed split is that extreme
    assert r["means"] == {"A": 5.0, "B": 0.0}


def test_label_permutation_without_strata_mixes_every_batch():
    vals = list(range(8))
    batches = [f"b{i}" for i in range(8)]
    labels = {b: ("x" if i < 4 else "y") for i, b in enumerate(batches)}
    r = bs.batch_label_perm(vals, batches, labels)
    assert r["exact"] and r["n_perm"] == 70           # C(8, 4)
    assert r["p"] == pytest.approx(2 / 70)


def test_batch_trend_is_one_value_per_batch():
    starts = {"a": 9.0, "b": 11.0, "c": 13.0, "d": 15.0, "e": 17.0}
    vals = [1, 1, 1, 2, 2, 3, 4, 5, 5, 5]               # rising with the batch's start
    batches = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    r = bs.batch_trend(vals, batches, starts)
    assert r["rho"] == pytest.approx(1.0) and r["n_batches"] == 5
    assert r["p"] == pytest.approx(2 / 120)           # exact over 5! orderings, two-sided
