"""Legacy reaction-matrix odor handling.

The v2 "drop trials whose label has no odor suffix" filter must NOT run under the
legacy protocol: legacy labels are plain ``testing_N`` (odor comes from the
hardcoded schedule via _display_odor), so the filter would drop every trial and
produce an empty "No odors available" matrix. Regression for that empty-figure bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for c in (ROOT, ROOT / "src"):
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))

from scripts.analysis import reaction_matrix_from_spreadsheet as rm  # noqa: E402
from scripts.analysis.envelope_visuals import set_protocol  # noqa: E402


def test_legacy_keeps_plain_testing_n_labels():
    set_protocol("legacy")
    df = pd.DataFrame({"trial": ["testing_1", "testing_2", "testing_10"]})
    out = rm._drop_label_only_no_odor_trials(df)
    assert sorted(out["trial"]) == ["testing_1", "testing_10", "testing_2"]


def test_v2_drops_labels_without_odor_suffix():
    set_protocol("v2")
    df = pd.DataFrame({"trial": ["testing_1", "testing_3_Benzaldehyde"]})
    out = rm._drop_label_only_no_odor_trials(df)
    assert list(out["trial"]) == ["testing_3_Benzaldehyde"]
