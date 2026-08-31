"""``figure_frozen_from_raw``: the freeze set read straight off the config YAML.

Figure scripts run as subprocesses with only a ``--config`` path. Going through
``load_settings`` there is wrong: it validates the whole pipeline (it raises when
a data-frozen dataset's raw folder is missing from disk), and a figure script
must not fail for a reason that has nothing to do with drawing. This helper
reads the same two keys the dataclass does, and nothing else.
"""

from __future__ import annotations

import pytest

from fbpipe.freeze import figure_frozen_from_raw


RAW = {
    "dataset_overrides": {
        "Live": {},
        "DataOnly": {"freeze": {"data": True}},
        "FiguresOnly": {"freeze": {"figures": True}},
        "Both": {"freeze": {"data": True, "figures": True}},
        "ExplicitFalse": {"freeze": {"data": True, "figures": False}},
    }
}


def test_only_figure_freeze_counts():
    assert figure_frozen_from_raw(RAW) == {"FiguresOnly", "Both"}


def test_thaw_lifts_one_dataset():
    assert figure_frozen_from_raw(RAW, thawed=["Both"]) == {"FiguresOnly"}


def test_thaw_all_lifts_everything():
    assert figure_frozen_from_raw(RAW, thaw_all=True) == set()


@pytest.mark.parametrize("raw", [None, {}, {"dataset_overrides": None}, "nonsense"])
def test_missing_or_malformed_config_is_not_frozen(raw):
    """Unknown means LIVE. A config we cannot read must never silently suppress
    every figure -- the failure mode has to be a redundant render, not a
    missing one."""
    assert figure_frozen_from_raw(raw) == set()


def test_hand_typed_freeze_true_is_ignored():
    """``freeze: true`` (not a mapping) is a config typo; it must not count as
    a figure freeze -- same stance as fbpipe.config._freeze_block."""
    assert figure_frozen_from_raw({"dataset_overrides": {"A": {"freeze": True}}}) == set()


def test_truthy_non_bool_does_not_count():
    assert figure_frozen_from_raw(
        {"dataset_overrides": {"A": {"freeze": {"figures": "yes"}}}}
    ) == set()
