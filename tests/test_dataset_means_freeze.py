"""``dataset_means`` (and its training wrapper) must skip figure-frozen datasets.

It renders one figure per dataset found in the wide table, so a frozen dataset
whose rows are spliced back in from the freeze cache would otherwise be redrawn
on every run -- restyled by whatever palette and threshold are current.
"""

from __future__ import annotations

import scripts.analysis.dataset_means as dm


CFG = {
    "dataset_overrides": {
        "Frozen-Fig": {"freeze": {"data": True, "figures": True}},
        "Data-Only": {"freeze": {"data": True}},
    }
}

FOUND = ["Data-Only", "Frozen-Fig", "Live"]


def test_frozen_figures_dataset_is_dropped():
    assert dm.datasets_to_render(FOUND, CFG) == ["Data-Only", "Live"]


def test_data_freeze_alone_still_renders():
    """freeze.data means "reuse the rows", not "stop drawing"."""
    assert "Data-Only" in dm.datasets_to_render(FOUND, CFG)


def test_thaw_restores_a_frozen_dataset():
    assert dm.datasets_to_render(FOUND, CFG, thawed=["Frozen-Fig"]) == FOUND
    assert dm.datasets_to_render(FOUND, CFG, thaw_all=True) == FOUND


def test_no_config_renders_everything():
    assert dm.datasets_to_render(FOUND, {}) == FOUND


def test_order_is_preserved():
    assert dm.datasets_to_render(["b", "a"], {}) == ["b", "a"]


def test_cli_exposes_thaw_flags():
    args = dm.build_parser(["--thaw", "X", "--thaw", "Y"])
    assert list(args.thaw) == ["X", "Y"]
    assert dm.build_parser(["--thaw-all"]).thaw_all is True
    assert dm.build_parser([]).thaw_all is False
