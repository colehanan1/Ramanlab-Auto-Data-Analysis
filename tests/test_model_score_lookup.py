"""Regression tests for the per-trial "Score: N" annotation on envelope traces.

``run_workflows._load_model_scores_for_envelopes`` keys ``_MODEL_SCORES`` by the
``dataset`` column of ``model_predictions.csv`` — the dataset FOLDER name. The
trace plots look scores up by ``dataset_canon``. For every cohort but one those
strings are identical, so the mismatch stayed invisible; the 3Oct cohorts
canonicalise "3Oct-Training-24-0.1" -> "3OCT-Training-24-0.1", so every lookup
missed and the Raw-Testing-PER-Traces panels rendered with no score at all.

Registration canonicalises the dataset component so either spelling resolves.
"""

from __future__ import annotations

import pytest

from fbpipe.odor_constants import canon_dataset
from scripts.analysis import envelope_visuals as ev

RAW_DATASET = "3Oct-Training-24-0.1"
CANON_DATASET = "3OCT-Training-24-0.1"
FLY = "july_20_batch_1_rig_2"
FLY_NUMBER = "1"
TRIAL = "testing_1_3-octonol"


@pytest.fixture()
def restore_scores():
    saved = dict(ev._MODEL_SCORES)
    try:
        yield
    finally:
        ev.set_model_scores(saved)


def test_precondition_dataset_name_differs_from_canon() -> None:
    assert canon_dataset(RAW_DATASET) == CANON_DATASET != RAW_DATASET


def test_score_resolves_when_registered_under_the_csv_dataset_name(restore_scores) -> None:
    """Registered with the CSV's folder name; looked up with the canon name."""
    ev.set_model_scores({(RAW_DATASET, FLY, FLY_NUMBER, TRIAL): 4})
    assert ev._lookup_model_score(CANON_DATASET, FLY, FLY_NUMBER, TRIAL) == 4
    # The as-registered spelling must keep working too.
    assert ev._lookup_model_score(RAW_DATASET, FLY, FLY_NUMBER, TRIAL) == 4


def test_unrelated_keys_still_miss(restore_scores) -> None:
    """Canonicalising keys must not make unrelated lookups start matching."""
    ev.set_model_scores({(RAW_DATASET, FLY, FLY_NUMBER, TRIAL): 4})
    assert ev._lookup_model_score(CANON_DATASET, FLY, "2", TRIAL) is None
    assert ev._lookup_model_score(CANON_DATASET, FLY, FLY_NUMBER, "testing_2_acv") is None
    assert ev._lookup_model_score("EB-Training-24-1", FLY, FLY_NUMBER, TRIAL) is None


def test_explicit_canon_key_wins_over_the_alias(restore_scores) -> None:
    """A real canon-keyed entry must not be overwritten by an alias of another."""
    ev.set_model_scores(
        {
            (CANON_DATASET, FLY, FLY_NUMBER, TRIAL): 5,
            (RAW_DATASET, FLY, FLY_NUMBER, TRIAL): 4,
        }
    )
    assert ev._lookup_model_score(CANON_DATASET, FLY, FLY_NUMBER, TRIAL) == 5


def test_scores_of_zero_survive_registration(restore_scores) -> None:
    """0 is a real score, not a missing one — falsy values must round-trip."""
    ev.set_model_scores({(RAW_DATASET, FLY, FLY_NUMBER, TRIAL): 0})
    assert ev._lookup_model_score(CANON_DATASET, FLY, FLY_NUMBER, TRIAL) == 0
