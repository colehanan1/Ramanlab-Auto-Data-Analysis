"""Odor labels drawn as "<Odor> <c>% (v/v)", whatever spelling they arrive in."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.odor_conc_labels import conc_text, odor_conc_label, with_conc


@pytest.mark.parametrize("raw, shown", [
    ("3-octonol-0.1", "3-Octanol 0.1% (v/v)"),
    ("3octanol-0.1", "3-Octanol 0.1% (v/v)"),
    ("3-octanol-0.1", "3-Octanol 0.1% (v/v)"),
    ("isoamylacetate-1", "Isoamyl Acetate 1% (v/v)"),
    ("ethylbutyrate-10", "Ethyl Butyrate 10% (v/v)"),
    ("EthylButyrate-0.1", "Ethyl Butyrate 0.1% (v/v)"),
    ("Hexanol-1", "Hexanol 1% (v/v)"),
    ("testing_3_citral-1", "Citral 1% (v/v)"),
    ("pretest_2_benzaldehyde-0.1", "Benzaldehyde 0.1% (v/v)"),
])
def test_raw_pins_tokens(raw, shown):
    assert odor_conc_label(raw) == shown


def test_parenthesised_display_names():
    assert odor_conc_label("3-Octanol (0.1%)") == "3-Octanol 0.1% (v/v)"
    assert odor_conc_label("Isoamyl Acetate (1%)") == "Isoamyl Acetate 1% (v/v)"


def test_no_concentration_is_not_invented():
    assert odor_conc_label("3-octonol") == "3-Octanol"
    assert odor_conc_label("Linalool") == "Linalool"
    assert with_conc("hexanol", None) == "Hexanol"
    assert with_conc("hexanol", math.nan) == "Hexanol"


def test_presentation_suffix_is_not_a_concentration():
    assert odor_conc_label("Hexanol 2") == "Hexanol 2"


def test_light_and_already_formatted_labels():
    assert odor_conc_label("lightonly") == "Optogenetic Light Control"
    assert odor_conc_label("3-Octanol 0.1% (v/v)") == "3-Octanol 0.1% (v/v)"


def test_explicit_parts_and_number_format():
    assert with_conc("3-octonol", 0.1) == "3-Octanol 0.1% (v/v)"
    assert with_conc("Isoamyl Acetate", "1.0") == "Isoamyl Acetate 1% (v/v)"
    assert conc_text("0.10") == "0.1% (v/v)"
