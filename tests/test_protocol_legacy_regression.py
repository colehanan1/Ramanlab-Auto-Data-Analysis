"""Acceptance gate: ``protocol: legacy`` reproduces the v1 codebase exactly.

The old (v1) tree is the regression target. We run v1's ``envelope_combined``
in a SUBPROCESS (the two trees share the module names ``scripts.analysis.*`` and
``fbpipe.*``, so they cannot be imported into one interpreter) on the same
sidecar-less fixture, then run the CURRENT code in-process under
``protocol: legacy`` and assert the wide CSV, code_maps.json, and float16 matrix
match. Skipped when the v1 checkout is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from scripts.analysis import envelope_combined as ec  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from _protocol_fixtures import build_envelope_fixture, MEASURE_COL  # noqa: E402

V1_REPO = Path(
    "/home/ramanlab/Documents/cole/VSCode/"
    "Ramanlab-Auto-Data-Analysis-95dad19b65aa4f63599712ea7713b4b87c127538"
)

pytestmark = pytest.mark.skipif(
    not (V1_REPO / "scripts" / "analysis" / "envelope_combined.py").exists(),
    reason="v1 regression checkout not present",
)

_V1_RUNNER = textwrap.dedent(
    """
    import sys
    from scripts.analysis import envelope_combined as ec
    fixture, wide_csv, ref_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    ec.build_wide_csv([fixture], wide_csv, measure_cols=["envelope_of_rms"])
    ec.wide_to_matrix(wide_csv, ref_dir)
    """
)


def _run_v1(fixture: Path, wide_csv: Path, ref_dir: Path) -> None:
    runner = ref_dir.parent / "_v1_runner.py"
    runner.write_text(_V1_RUNNER)
    env = dict(os.environ)
    # Force v1's own fbpipe / scripts ahead of the editable-installed current one.
    env["PYTHONPATH"] = os.pathsep.join([str(V1_REPO / "src"), str(V1_REPO)])
    result = subprocess.run(
        [sys.executable, str(runner), str(fixture), str(wide_csv), str(ref_dir)],
        cwd=str(V1_REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"v1 subprocess failed:\n{result.stderr[-3000:]}"


@pytest.fixture()
def outputs(tmp_path):
    dataset_root = build_envelope_fixture(tmp_path)

    ev.set_protocol("legacy")
    cur_csv = tmp_path / "cur.csv"
    cur_dir = tmp_path / "cur_matrix"
    ec.build_wide_csv([str(dataset_root)], str(cur_csv), measure_cols=[MEASURE_COL])
    ec.wide_to_matrix(str(cur_csv), str(cur_dir))

    v1_csv = tmp_path / "v1.csv"
    v1_dir = tmp_path / "v1_matrix"
    v1_dir.mkdir(parents=True, exist_ok=True)
    _run_v1(dataset_root, v1_csv, v1_dir)

    sort = ["fly", "trial_type", "trial_label"]
    cur = pd.read_csv(cur_csv).sort_values(sort).reset_index(drop=True)
    v1 = pd.read_csv(v1_csv).sort_values(sort).reset_index(drop=True)
    return cur, v1, cur_dir, v1_dir


def test_legacy_wide_columns_match_v1(outputs):
    cur, v1, _, _ = outputs
    assert list(cur.columns) == list(v1.columns)
    # Legacy schema must not carry the v2-only per-trial seconds columns.
    for col in ("trial_odor_on_s", "trial_odor_off_s", "trial_duration_s", "trial_light_on_s"):
        assert col not in cur.columns


def test_legacy_wide_values_match_v1(outputs):
    cur, v1, _, _ = outputs
    str_cols = {"dataset", "fly", "trial_type", "trial_label"}
    for col in cur.columns:
        if col in str_cols:
            assert (cur[col].astype(str) == v1[col].astype(str)).all(), f"{col} differs"
        else:
            a = pd.to_numeric(cur[col], errors="coerce").to_numpy(float)
            b = pd.to_numeric(v1[col], errors="coerce").to_numpy(float)
            assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), f"{col} differs"


def test_legacy_code_maps_match_v1(outputs):
    _, _, cur_dir, v1_dir = outputs
    cur_cm = json.loads((cur_dir / "code_maps.json").read_text())
    v1_cm = json.loads((v1_dir / "code_maps.json").read_text())
    assert cur_cm == v1_cm


def test_legacy_matrix_matches_v1(outputs):
    _, _, cur_dir, v1_dir = outputs
    cur_m = np.load(cur_dir / "envelope_matrix_float16.npy")
    v1_m = np.load(v1_dir / "envelope_matrix_float16.npy")
    assert cur_m.shape == v1_m.shape
    assert np.array_equal(cur_m, v1_m, equal_nan=True)
