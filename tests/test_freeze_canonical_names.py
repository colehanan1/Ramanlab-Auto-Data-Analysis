"""Freeze lookups must survive canonicalisation of the dataset name.

The config spells the cohort ``3Oct-Training-24-0.1``; every figure path works
in canonical names, and ``odor_constants.canon_dataset`` renders that one as
``3OCT-Training-24-0.1`` (uppercase OCT). An exact dict lookup therefore misses,
and the whole 3Oct cohort re-rendered on every run while the config said it was
frozen -- observed 2026-08-28 on Raw-Testing-PER-Traces/3OCT-Training-24-0.1.
"""

from __future__ import annotations

import pytest

from fbpipe.freeze import figure_frozen_from_raw, freeze_flags
from fbpipe.odor_constants import canon_dataset


class _Ov:
    def __init__(self, data=True, figures=True):
        self.freeze_data = data
        self.freeze_figures = figures


class _Cfg:
    def __init__(self, overrides):
        self.dataset_overrides = overrides


CONFIG_SPELLING = "3Oct-Training-24-0.1"
CANON_SPELLING = "3OCT-Training-24-0.1"


def test_the_two_spellings_really_do_differ():
    """Guard the premise: if canon_dataset ever stops re-casing this, the rest
    of the file is testing nothing."""
    assert canon_dataset(CONFIG_SPELLING) == CANON_SPELLING
    assert CONFIG_SPELLING != CANON_SPELLING


@pytest.mark.parametrize("looked_up", [CONFIG_SPELLING, CANON_SPELLING])
def test_freeze_flags_matches_either_spelling(looked_up):
    cfg = _Cfg({CONFIG_SPELLING: _Ov()})
    assert freeze_flags(cfg, looked_up) == (True, True)


def test_freeze_flags_thaw_matches_either_spelling():
    cfg = _Cfg({CONFIG_SPELLING: _Ov()})
    assert freeze_flags(cfg, CANON_SPELLING, thawed=[CONFIG_SPELLING]) == (False, False)
    assert freeze_flags(cfg, CONFIG_SPELLING, thawed=[CANON_SPELLING]) == (False, False)


def test_freeze_flags_still_misses_a_genuinely_different_dataset():
    """Canonical matching must not turn into fuzzy matching."""
    cfg = _Cfg({CONFIG_SPELLING: _Ov()})
    assert freeze_flags(cfg, "3Oct-Control-24-0.1") == (False, False)
    assert freeze_flags(cfg, "EB-Training-24-1") == (False, False)


def test_figure_frozen_from_raw_carries_both_spellings():
    raw = {"dataset_overrides": {CONFIG_SPELLING: {"freeze": {"figures": True}}}}
    frozen = figure_frozen_from_raw(raw)
    assert CONFIG_SPELLING in frozen
    assert CANON_SPELLING in frozen


def test_figure_frozen_from_raw_thaw_uses_canonical_names_too():
    raw = {"dataset_overrides": {CONFIG_SPELLING: {"freeze": {"figures": True}}}}
    assert figure_frozen_from_raw(raw, thawed=[CANON_SPELLING]) == set()


def test_flagged_variant_of_a_frozen_dataset_is_frozen():
    """``<dataset>-flagged`` rows are the same cohort; a frozen cohort must not
    come back to life through its flagged split."""
    cfg = _Cfg({CONFIG_SPELLING: _Ov()})
    assert freeze_flags(cfg, f"{CANON_SPELLING}-flagged") == (True, True)
