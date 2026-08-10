"""Cohort selection: the fly list in the binary-reactions CSV is authoritative.

The Hex-Control cohort behind the OctNov Training_vs_Control figures is 15
flies. Since those figures were first drawn, 11 of them were re-assigned to the
``Hex-Control-flagged`` dataset in the wide table, so a plain
``dataset == "Hex-Control"`` filter now silently yields 4 flies and the figure
regenerates with a different n and nobody notices.

Two guards: an opt-in that also accepts the ``-flagged`` sibling dataset for
flies the cohort CSV names explicitly, and a warning whenever selected flies
have no rows at all.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from scripts.analysis.dataset_means_specific_flies import (
    _flagged_dataset_name,
    _missing_flies,
    _select_dataset_rows,
)

DS = "Hex-Control"
FLAGGED = "Hex-Control-flagged"


def _wide() -> pd.DataFrame:
    rows = [
        (DS, "october_10_batch_1", 1, "testing"),
        (DS, "october_10_batch_1", 2, "testing"),
        (DS, "october_10_batch_1", 1, "training"),
        (FLAGGED, "november_03_batch_1_rig_2", 1, "testing"),
        (FLAGGED, "november_03_batch_1_rig_2", 2, "testing"),
        (FLAGGED, "never_selected_batch", 9, "testing"),
        ("Hex-Training", "october_11_batch_1", 1, "testing"),
    ]
    return pd.DataFrame(
        rows, columns=["dataset", "fly", "fly_number", "trial_type"]
    )


COHORT = {
    ("october_10_batch_1", 1),
    ("october_10_batch_1", 2),
    ("november_03_batch_1_rig_2", 1),
    ("november_03_batch_1_rig_2", 2),
}


def test_flagged_dataset_name():
    assert _flagged_dataset_name(DS) == FLAGGED
    assert _flagged_dataset_name("Hex-Training") == "Hex-Training-flagged"


def test_default_selection_ignores_the_flagged_sibling():
    got = _select_dataset_rows(_wide(), DS, flies=COHORT, include_flagged=False)
    assert set(got["fly"]) == {"october_10_batch_1"}
    assert set(got["trial_type"]) == {"testing"}


def test_include_flagged_recovers_the_reassigned_flies():
    got = _select_dataset_rows(_wide(), DS, flies=COHORT, include_flagged=True)
    assert set(zip(got["fly"], got["fly_number"])) == COHORT


def test_include_flagged_does_not_widen_the_cohort():
    """A flagged fly the cohort CSV never listed stays out."""
    got = _select_dataset_rows(_wide(), DS, flies=COHORT, include_flagged=True)
    assert "never_selected_batch" not in set(got["fly"])


def test_include_flagged_does_not_pull_in_another_dataset():
    got = _select_dataset_rows(
        _wide(),
        DS,
        flies=COHORT | {("october_11_batch_1", 1)},
        include_flagged=True,
    )
    assert set(got["dataset"]) <= {DS, FLAGGED}


def test_training_rows_are_excluded_either_way():
    for include in (False, True):
        got = _select_dataset_rows(
            _wide(), DS, flies=COHORT, include_flagged=include
        )
        assert "training" not in set(got["trial_type"])


# --------------------------------------------------------------------------
# Missing-fly reporting
# --------------------------------------------------------------------------


def test_missing_flies_lists_cohort_members_with_no_rows():
    got = _select_dataset_rows(_wide(), DS, flies=COHORT, include_flagged=False)
    assert _missing_flies(got, COHORT) == [
        ("november_03_batch_1_rig_2", 1),
        ("november_03_batch_1_rig_2", 2),
    ]


def test_missing_flies_empty_when_every_fly_resolved():
    got = _select_dataset_rows(_wide(), DS, flies=COHORT, include_flagged=True)
    assert _missing_flies(got, COHORT) == []


def test_missing_flies_handles_an_empty_frame():
    empty = _wide().iloc[0:0]
    assert _missing_flies(empty, COHORT) == sorted(COHORT)


def test_prepare_dataset_warns_when_flies_are_missing(caplog):
    """A shrunken cohort has to be loud — it silently changes every figure's n."""
    from scripts.analysis import dataset_means_specific_flies as mod

    with caplog.at_level(logging.WARNING, logger=mod.LOGGER.name):
        mod.LOGGER.propagate = True
        try:
            got = _select_dataset_rows(
                _wide(), DS, flies=COHORT, include_flagged=False
            )
            mod._warn_missing_flies(DS, got, COHORT)
        finally:
            mod.LOGGER.propagate = False
    assert any(
        "2 of 4" in rec.message and "november_03_batch_1_rig_2" in rec.message
        for rec in caplog.records
    ), caplog.text


def test_no_warning_when_the_cohort_is_complete(caplog):
    from scripts.analysis import dataset_means_specific_flies as mod

    with caplog.at_level(logging.WARNING, logger=mod.LOGGER.name):
        mod.LOGGER.propagate = True
        try:
            got = _select_dataset_rows(
                _wide(), DS, flies=COHORT, include_flagged=True
            )
            mod._warn_missing_flies(DS, got, COHORT)
        finally:
            mod.LOGGER.propagate = False
    assert not caplog.records


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_cli_defaults_to_excluding_flagged():
    from scripts.analysis.dataset_means_specific_flies import build_parser

    assert build_parser([]).include_flagged is False


def test_cli_include_flagged_flag():
    from scripts.analysis.dataset_means_specific_flies import build_parser

    assert build_parser(["--include-flagged"]).include_flagged is True


@pytest.mark.parametrize("include", [False, True])
def test_prepare_dataset_signature_accepts_include_flagged(include):
    import inspect

    from scripts.analysis.dataset_means_specific_flies import prepare_dataset

    param = inspect.signature(prepare_dataset).parameters["include_flagged"]
    assert param.default is False
