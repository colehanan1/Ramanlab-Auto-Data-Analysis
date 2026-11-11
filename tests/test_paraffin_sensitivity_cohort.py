"""Paraffin-Sensitivity-1-100: the paraffin-oil (master valve only) cohort.

The rigs push P sessions to ``Paraffin-Sensitivity-<h>-100`` from 2026-11-04 and
every sensitivity panel now opens and closes with a ``ParaffinOil`` trial. Before
this, the analysis side knew neither: the trial labels rendered as the raw token
``ParaffinOil`` and the cohort's sensitized odor fell through to the dataset
NAME, so no figure could pick out its paraffin trials.
"""

from pathlib import Path

import pytest
import yaml

from scripts.analysis.day_vs_day_report import trained_odor
from scripts.analysis.envelope_visuals import _display_odor_v2, _trained_label

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"
PARAFFIN = "Paraffin-Sensitivity-1-100"


@pytest.mark.parametrize("label", ["pretest_1_ParaffinOil", "testing_8_paraffinoil",
                                   "training_3_ParaffinOil"])
def test_paraffin_trials_get_a_display_name(label):
    assert _display_odor_v2(PARAFFIN, label) == "Paraffin Oil"


def test_paraffin_bookends_in_other_cohorts_get_the_same_name():
    assert _display_odor_v2("IAA-Sensitivity-1-1", "pretest_1_paraffinoil") == "Paraffin Oil"


def test_paraffin_cohort_sensitized_odor_matches_its_trials():
    assert _trained_label(PARAFFIN) == "Paraffin Oil"
    assert trained_odor(PARAFFIN) == "Paraffin Oil"


def test_existing_cohorts_keep_their_sensitized_odor():
    assert trained_odor("IAA-Sensitivity-1-1") == "Isoamyl Acetate"
    assert trained_odor("Hex-Sensitivity-1-0.1") == "Hexanol"
    assert trained_odor("3Oct-Sensitivity-1-0.1") == "3-Octanol"


def test_paraffin_is_a_registered_pretest_cohort():
    import scripts.analysis.naive_vs_trial_score_bars as nvt

    spec = nvt.PRETEST_COHORTS[PARAFFIN]
    assert spec.odor == "Paraffin Oil"
    assert spec.pretest_baseline is True


# ── config/config_new.yaml (gitignored: pins the on-disk half) ────────────


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.exists():
        pytest.skip("config/config_new.yaml not present (gitignored)")
    return yaml.safe_load(CONFIG.read_text())


def test_paraffin_is_listed_so_the_pipeline_picks_it_up(cfg):
    assert PARAFFIN in cfg["datasets"]


def test_paraffin_is_not_frozen(cfg):
    freeze = ((cfg.get("dataset_overrides") or {}).get(PARAFFIN) or {}).get("freeze") or {}
    assert freeze.get("data") is not True and freeze.get("figures") is not True


def test_paraffin_lands_in_the_sensitivity_window_results(cfg):
    block = cfg["analysis"]["sensitivity_window_results"]
    assert PARAFFIN in block["datasets"]
