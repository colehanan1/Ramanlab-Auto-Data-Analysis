"""Per-rig anchor resolution.

rig_3 is a physically mirrored rig (flies and odor tube on the opposite side),
so its geometric anchor sits on the left edge instead of the right.

The rig token match requires the underscore between "rig" and the digit
("rig_3", not "rig3"). ``rig_token`` scans every ancestor path component, so
without the underscore a directory merely *mentioning* rig3 in prose --
e.g. "EB-Training-24-1_excl_rig3" (real directory under
/home/ramanlab/Documents/cole/Results/Figures, meaning "excluding rig3") --
could be misread as a rig_3 trial. This is a defensive/latent concern rather
than an observed production bug: ``resolve_anchor`` is only ever called on
data paths, never on ``Results/Figures``, so that directory was never
actually reachable through this code. Every real rig directory on disk uses
the underscore form, so requiring it loses no real matches; see the
poisoner-name regression tests below.
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
        # No underscore between "rig" and the digit: must NOT match. This is
        # an intentional behaviour change (see module docstring) -- without
        # requiring the underscore, an ancestor directory that merely
        # *mentions* "rig3" in prose (e.g. "..._excl_rig3") would be
        # misread as a rig_3 trial.
        ("/data/x/july_17_batch_2_rig3/trial_1", None),
        ("/data/x/july_17_batch_2_Rig3/trial_1", None),
    ],
)
def test_rig_token(path, expected):
    assert rig_token(path) == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        # Real poisoner directory: means "EXCLUDING rig3", must not mirror.
        (
            "EB-Training-24-1_excl_rig3/july_18_batch_1",
            DEFAULT_ANCHOR,
        ),
        # Real poisoner directory that also mentions rig2 in prose (no
        # underscore) -- must still resolve to DEFAULT.
        (
            "EB-Training-24-1_excl_rig3_and_july17b2rig2/july_18_batch_1",
            DEFAULT_ANCHOR,
        ),
        # A genuine rig_3 token deeper in the path must still win, even
        # under a poisoner ancestor -- the deepest real token wins.
        (
            "EB-Training-24-1_excl_rig3/july_18_batch_1_rig_3",
            MIRRORED_ANCHOR,
        ),
        # Unchanged: a genuine underscored rig_3 token still mirrors.
        (
            "july_17_batch_2_rig_3",
            MIRRORED_ANCHOR,
        ),
    ],
)
def test_resolve_anchor_ignores_no_underscore_ancestor_mentions(path, expected):
    """Regression test for a defensive/latent ancestor false-positive.

    ``rig_token`` scans every ancestor path component, so any ancestor
    directory containing the bare substring "rig3"/"rig2" would, without the
    underscore requirement, be misread as naming a rig. The real directory
    ``EB-Training-24-1_excl_rig3_and_july17b2rig2`` (under
    /home/ramanlab/Documents/cole/Results/Figures) and
    ``scripts/analysis/reaction_matrix_specific_flies_vs_control.py``'s
    ``EB-Training-24-1_excl_rig3`` output both trip this: they mean
    "excluding rig3" but without the underscore requirement would resolve
    AS rig_3. This was never an observed production bug -- ``resolve_anchor``
    is only ever called on data paths, never on ``Results/Figures`` -- but
    requiring the underscore between "rig" and the digit closes the gap
    defensively without losing any real match (every real rig directory on
    disk uses the underscore form).

    Verified to bite: reverting ``_RIG_RE`` to the old ``rig_?(\\d+)``
    makes the first two cases here fail (they resolve to MIRRORED_ANCHOR
    instead of DEFAULT_ANCHOR).
    """
    assert resolve_anchor(path) == expected


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
