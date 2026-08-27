"""Ratio-based row sorting, and one fixed odor order across both arms.

Two changes that travel together, because they exist to make a control raster
and its trained partner readable side by side.

**Rows** are ranked by AUC-During / AUC-Before -- a fold-change of extension over
that trial's own baseline, rather than a bare magnitude. The AUCs are areas under
the *envelope*, not the wide table's threshold-relative ``AUC-*`` columns: those
measure area ABOVE theta, which is exactly 0 for a fly that held still before
odor onset, so their ratio is undefined for 40-74% of training trials. A zero
denominator becomes 1 so the division still has an answer; with envelope areas
that never triggers on the current cohorts (smallest measured baseline area is
22.9), so it is a guard rather than something that shapes the figures.

Controls average the ratio over all conditioning trials. Trained arms use the
FIRST trained-odor exposure only: by trial 6 the fly has been conditioned, so a
mean across trials measures the training rather than the animal.

**Panels** run trained-odor presentation 1, presentation 2, then every other odor
alphanumerically -- a fixed order that does not depend on the trial indices a
particular cohort happened to draw, so column N is the same odor in the control
and the trained figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import math

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from scripts.analysis import binarized_per_rasters as mod  # noqa: E402
from test_binarized_per_rasters import (  # noqa: E402
    DATASET, _training_frame, _testing_frame,
)


@pytest.fixture()
def train():
    return mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")


@pytest.fixture()
def test_trials():
    return mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")


# ── the per-trial ratio ───────────────────────────────────────────────────


def test_trials_carry_raw_window_areas(train):
    for col in ("auc_before_raw", "auc_during_raw", "auc_ratio"):
        assert col in train.columns, col


def test_ratio_is_during_over_before(train):
    r = train.iloc[0]
    assert r["auc_before_raw"] > 0
    assert float(r["auc_ratio"]) == pytest.approx(
        r["auc_during_raw"] / r["auc_before_raw"]
    )


def test_zero_before_becomes_one():
    """A 0 denominator is replaced by 1 so the division still has an answer
    (user's call, 2026-08-26). Does not arise in the current cohorts -- the
    smallest baseline envelope area measured is 22.9 -- so this is a guard
    against a degenerate trace, not something that shapes today's figures."""
    assert mod.auc_ratio(10.0, 0.0) == pytest.approx(10.0)
    assert mod.auc_ratio(0.0, 0.0) == pytest.approx(0.0)


def test_an_unmeasurable_before_stays_undefined():
    """NaN is not 0: it means the window could not be computed at all, so there
    is nothing to substitute FOR. Only an actual zero area is replaced."""
    assert math.isnan(mod.auc_ratio(10.0, float("nan")))
    assert math.isnan(mod.auc_ratio(float("nan"), 5.0))


def test_only_exact_zero_is_substituted():
    """A small-but-real baseline keeps its own value; it is not floored up to
    1, which would silently shrink that fly's ratio."""
    assert mod.auc_ratio(10.0, 0.5) == pytest.approx(20.0)
    assert mod.auc_ratio(10.0, 4.0) == pytest.approx(2.5)


def test_a_nan_ratio_sorts_last(train):
    """Guards the silent-leader failure: NaN must not float to the top."""
    t = train.copy()
    victim = t.iloc[0]["fly_id"]
    t.loc[t["fly_id"] == victim, "auc_ratio"] = float("nan")
    order = mod.fly_order(t, by="ratio")
    assert order[-1] == victim


def test_raw_areas_are_defined_for_every_trial(train):
    """The whole point of using envelope area over threshold-relative AUC."""
    assert np.isfinite(train["auc_before_raw"].to_numpy(float)).all()
    assert np.isfinite(train["auc_ratio"].to_numpy(float)).all()


# ── the two new sort keys ─────────────────────────────────────────────────


def test_ratio_sort_keys_exist():
    assert "ratio" in mod.SORT_KEYS
    assert "ratio_first" in mod.SORT_KEYS


def test_ratio_scores_average_over_all_training_trials(train):
    scores = mod.fly_scores(train, by="ratio")
    fid = next(iter(scores))
    expected = train[train["fly_id"] == fid]["auc_ratio"].mean()
    assert scores[fid] == pytest.approx(expected)


def test_ratio_first_uses_only_the_first_exposure(train):
    scores = mod.fly_scores(train, by="ratio_first")
    fid = next(iter(scores))
    g = train[train["fly_id"] == fid].sort_values("trial_index")
    assert scores[fid] == pytest.approx(g.iloc[0]["auc_ratio"])


def test_ratio_first_differs_from_the_mean_when_trials_differ(train):
    mean = mod.fly_scores(train, by="ratio")
    first = mod.fly_scores(train, by="ratio_first")
    assert set(mean) == set(first)
    assert any(mean[f] != pytest.approx(first[f]) for f in mean)


def test_highest_ratio_sorts_to_the_top(train):
    order = mod.fly_order(train, by="ratio")
    scores = mod.fly_scores(train, by="ratio")
    assert scores[order[0]] >= scores[order[-1]]
    assert order == sorted(order, key=lambda f: -scores[f])


# ── panel order ───────────────────────────────────────────────────────────


def test_trained_odor_leads_the_panels(test_trials):
    cs = mod.trained_odor(mod.build_trials(
        _training_frame(), dataset=DATASET, trial_type="training"
    ))
    panels = mod.testing_panels(test_trials, dataset=DATASET, trained=cs)
    assert panels[0].odor == cs


def test_repeat_presentations_of_the_cs_come_first_and_in_order(test_trials):
    cs = mod.trained_odor(mod.build_trials(
        _training_frame(), dataset=DATASET, trial_type="training"
    ))
    panels = mod.testing_panels(test_trials, dataset=DATASET, trained=cs)
    cs_panels = [p for p in panels if p.odor == cs]
    assert [p.rank for p in cs_panels] == sorted(p.rank for p in cs_panels)
    assert all(p.odor == cs for p in panels[:len(cs_panels)])


def test_the_rest_are_alphanumeric(test_trials):
    cs = mod.trained_odor(mod.build_trials(
        _training_frame(), dataset=DATASET, trial_type="training"
    ))
    panels = mod.testing_panels(test_trials, dataset=DATASET, trained=cs)
    rest = [p.label for p in panels if p.odor != cs]
    assert rest == sorted(rest)


def test_panel_order_ignores_trial_index(test_trials):
    """The old order keyed on median trial index, which is randomised per fly,
    so two cohorts running the same panel could disagree about column N."""
    cs = mod.trained_odor(mod.build_trials(
        _training_frame(), dataset=DATASET, trial_type="training"
    ))
    a = [p.label for p in mod.testing_panels(test_trials, dataset=DATASET, trained=cs)]
    shuffled = test_trials.copy()
    shuffled["trial_index"] = shuffled["trial_index"].max() + 1 - shuffled["trial_index"]
    b = [p.label for p in mod.testing_panels(shuffled, dataset=DATASET, trained=cs)]
    assert a == b


def test_no_trained_odor_still_produces_alphanumeric_panels(test_trials):
    panels = mod.testing_panels(test_trials, dataset=DATASET, trained=None)
    assert [p.label for p in panels] == sorted(p.label for p in panels)


def test_trained_odor_is_the_modal_training_odor(train):
    assert mod.trained_odor(train) == train["odor"].mode().iloc[0]


def test_trained_odor_of_empty_frame_is_none():
    assert mod.trained_odor(pd.DataFrame()) is None
