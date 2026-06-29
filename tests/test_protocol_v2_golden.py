"""v2 byte-identical guard.

Builds the deterministic fixture, runs the combined builder under
``protocol: v2``, and compares the wide CSV + float16 matrix + code_maps.json
against goldens captured from the current code BEFORE any protocol gating was
added (``tests/golden/v2/``). This is the regression fence proving the ``v2``
path stays byte-identical as legacy gating is introduced — i.e. "v2 no issue".

Regenerate goldens only on an intentional v2 change:
    python tests/_capture_v2_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from scripts.analysis import envelope_combined as ec  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from _protocol_fixtures import build_envelope_fixture, MEASURE_COL  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "v2"
_SORT = ["fly", "trial_type", "trial_label"]


def _build_v2(tmp_path: Path):
    dataset_root = build_envelope_fixture(tmp_path)
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    ec.build_wide_csv([str(dataset_root)], str(wide_csv), measure_cols=[MEASURE_COL])
    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))
    return wide_csv, matrix_dir


def test_v2_wide_csv_matches_golden(tmp_path):
    wide_csv, _ = _build_v2(tmp_path)
    got = pd.read_csv(wide_csv).sort_values(_SORT).reset_index(drop=True)
    exp = pd.read_csv(GOLDEN / "wide.csv").sort_values(_SORT).reset_index(drop=True)
    assert list(got.columns) == list(exp.columns)
    pd.testing.assert_frame_equal(got, exp, rtol=1e-9, atol=1e-9, check_dtype=False)


def test_v2_matrix_matches_golden(tmp_path):
    _, matrix_dir = _build_v2(tmp_path)
    got = np.load(matrix_dir / "envelope_matrix_float16.npy")
    exp = np.load(GOLDEN / "envelope_matrix_float16.npy")
    assert got.shape == exp.shape
    assert np.array_equal(got, exp, equal_nan=True)


def test_v2_code_maps_matches_golden(tmp_path):
    _, matrix_dir = _build_v2(tmp_path)
    got = json.loads((matrix_dir / "code_maps.json").read_text())
    exp = json.loads((GOLDEN / "code_maps.json").read_text())
    assert got == exp
