"""Per-rig anchor resolution.

rig_3 is a physically mirrored rig (flies and odor tube on the opposite side),
so its geometric anchor sits on the left edge instead of the right.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fbpipe.utils.rig_anchor import (
    DEFAULT_ANCHOR,
    MIRRORED_ANCHOR,
    resolve_anchor,
    rig_token,
)


def test_default_and_mirrored_anchor_values():
    assert DEFAULT_ANCHOR == (1080.0, 540.0)
    assert MIRRORED_ANCHOR == (0.0, 540.0)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/data/EB-Training-24-1/july_17_batch_2_rig_3/trial_1", "rig_3"),
        ("/data/EB-Training-24-1/july_17_batch_2_rig_2/trial_1", "rig_2"),
        ("/data/3Oct-Training-24-0.1/july_20_batch_1_rig_3", "rig_3"),
        ("relative/july_18_batch_1_rig_3/x.parquet", "rig_3"),
        ("/data/no_rig_here/trial_1", None),
    ],
)
def test_rig_token(path, expected):
    assert rig_token(path) == expected


def test_rig_3_resolves_to_mirrored_anchor():
    p = "/data/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_1"
    assert resolve_anchor(p) == MIRRORED_ANCHOR


def test_rig_2_resolves_to_default_anchor():
    p = "/data/EB-Training-24-1/july_17_batch_2_rig_2/july_17_batch_2_testing_1"
    assert resolve_anchor(p) == DEFAULT_ANCHOR


def test_unknown_path_falls_back_to_default():
    assert resolve_anchor("/tmp/somewhere/else") == DEFAULT_ANCHOR


def test_accepts_path_objects():
    p = Path("/data/x/july_17_batch_2_rig_3/trial")
    assert resolve_anchor(p) == MIRRORED_ANCHOR


def test_deepest_rig_token_wins():
    """A nested path must resolve to the rig closest to the trial."""
    p = "/data/rig_2_archive/july_17_batch_2_rig_3/trial"
    assert resolve_anchor(p) == MIRRORED_ANCHOR
