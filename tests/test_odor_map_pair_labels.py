"""The trial-label odor map must carry IAA odor-pair names (2026-11-13).

Pair recordings are named ``output_<batch>_testing_<N>_<odor1>_then_<odor2>_<ts>.csv``.
The old pattern's odor token could not contain ``_``, so pair trials got NO odor:
every one became a bare ``testing_<N>``, ``testing_9`` read as the light-only
control, and each fly's testing figure collapsed into one panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.analysis import envelope_combined, envelope_exports  # noqa: E402

FILES = [
    "output_november_13_batch_1_pretest_1_ParaffinOil-100_20261113_114806.csv",
    "output_november_13_batch_1_pretest_6_3-Octonol-0.1_20261113_121306.csv",
    "output_november_13_batch_1_testing_3_IsoamylAcetate-1_then_Linalool-1_20261113_125306.csv",
    "output_november_13_batch_1_testing_16_ParaffinOil-100_then_IsoamylAcetate-1_20261113_140810.csv",
    "output_november_13_batch_1_testing_9_3-Octonol-0.1_then_IsoamylAcetate-1_20261113_133809.csv",
    "sensors_output_november_13_batch_1_testing_3_IsoamylAcetate-1_then_Linalool-1_20261113_125306.csv",
]
EXPECTED = {
    "pretest_1": "ParaffinOil-100",
    "pretest_6": "3-Octonol-0.1",
    "testing_3": "IsoamylAcetate-1_then_Linalool-1",
    "testing_16": "ParaffinOil-100_then_IsoamylAcetate-1",
    "testing_9": "3-Octonol-0.1_then_IsoamylAcetate-1",
}


@pytest.fixture
def fly_dir(tmp_path):
    d = tmp_path / "IAA-PairX-Control-1-1" / "november_13_batch_1"
    d.mkdir(parents=True)
    for f in FILES:
        (d / f).write_text("FrameNumber\n1\n")
    return d


@pytest.mark.parametrize("mod", [envelope_combined, envelope_exports], ids=["combined", "exports"])
def test_pair_odor_names_reach_the_odor_map(fly_dir, mod):
    assert mod._build_odor_map(fly_dir) == EXPECTED


@pytest.mark.parametrize("mod", [envelope_combined, envelope_exports], ids=["combined", "exports"])
def test_single_odor_names_unchanged(tmp_path, mod):
    d = tmp_path / "Hex-HexTest-Training-24-1" / "october_05_batch_1"
    d.mkdir(parents=True)
    for f in ("output_october_05_batch_1_testing_1_Hexanol-1_20261005_101010.csv",
              "output_october_05_batch_1_training_2_Hexanol-0.1_20261005_111010.csv",
              "output_october_05_batch_1_testing_7_Apple Cider Vinegar_20261005_121010.csv"):
        (d / f).write_text("x\n")
    assert mod._build_odor_map(d) == {"testing_1": "Hexanol-1", "training_2": "Hexanol-0.1",
                                      "testing_7": "Apple Cider Vinegar"}

