"""iaa_rig_comparison: tie-aware Wilcoxon and the shared McNemar helper."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "iaa_rig_comparison.py"
_spec = importlib.util.spec_from_file_location("iaa_rig_comparison", _P)
irc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(irc)


def test_wilcoxon_p_is_tie_aware():
    # scipy's exact Wilcoxon reads 0.047 here; the tie-correct value is 0.0625
    assert irc.wilcoxon_p(pd.Series([-2, -2, -2, -2, -1, -1, 1.0])) == pytest.approx(0.0625)


def test_undefined_cases_keep_the_1_convention():
    assert irc.wilcoxon_p(pd.Series([0.0, 0.0])) == 1.0
    assert irc.wilcoxon_p(pd.Series([], dtype=float)) == 1.0
    same = pd.Series([True, False])
    assert irc.mcnemar_exact(same, same) == 1.0


def test_mcnemar_exact_matches_the_binomial():
    pre = pd.Series([False] * 5 + [True])
    post = pd.Series([True] * 5 + [True])
    assert irc.mcnemar_exact(pre, post) == pytest.approx(2 * 0.5 ** 5)
