"""A cohort infix in the dataset name must not hide the trained odor.

``experiment.cohort`` (PiCode) pushes to ``<Odor>-<Cohort>-<Arm>-<h>-<conc>``:
``IAA-Odor10s-Training-24-1``, ``Hex-HexTest-Training-32-1``. ``_trained_label``
only accepted ``<Odor>-<Arm>``, so these fell through to the dataset name
itself. Nothing then matched the trained odor: its two post-training
presentations pooled into one column with n = 2 x flies, and no tick was bolded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.envelope_visuals import _trained_label


@pytest.mark.parametrize(
    ("dataset_canon", "expected"),
    [
        ("IAA-Odor10s-Training-24-1", "Isoamyl Acetate"),
        ("IAA-Odor10s-Control-24-1", "Isoamyl Acetate"),
        ("IAA-Odor10s-Training-1-1", "Isoamyl Acetate"),
        ("Hex-HexTest-Training-32-1", "Hexanol"),
        # Regressions: names without an infix resolve exactly as before.
        ("IAA-Sensitivity-24-1", "Isoamyl Acetate"),
        ("EB-Training-24-1", "Ethyl Butyrate"),
        ("Hex-Control-24-0.1", "Hexanol"),
        ("3OCT-Training-24-0.1", "3-Octanol"),
    ],
)
def test_trained_label_sees_past_a_cohort_infix(dataset_canon, expected):
    assert _trained_label(dataset_canon) == expected
