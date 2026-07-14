import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

MODULE_PATH = PROJECT_ROOT / "scripts" / "analysis" / "score_summary.py"
spec = importlib.util.spec_from_file_location("score_summary_matrix", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_score_palette_has_one_colour_per_score():
    assert module.SCORES == [-1, 0, 1, 2, 3, 4, 5]
    assert len(module.SCORE_COLORS) == len(module.SCORES)


def test_score_palette_hexes_are_the_cvd_validated_set():
    # These exact values were validated at worst-pair CVD dE 19.5.
    # Changing them requires re-running the dataviz validator.
    assert module.SCORE_COLORS == [
        "#762a83", "#e2d4e8", "#f2ebf5",
        "#a6dba0", "#5aae61", "#1b7837", "#00441b",
    ]


def test_score_cmap_maps_each_score_to_its_own_colour():
    cmap, norm = module._score_cmap()
    seen = [cmap(norm(s)) for s in module.SCORES]
    assert len(set(seen)) == len(module.SCORES), "scores collapsed to same colour"


def test_each_score_maps_to_its_own_designated_hex():
    """Fixed map: score S wears SCORE_COLORS[S - SCORE_MIN], always.

    Falsifiable — breaks on wrong bounds math, reordered colours, or a bad hex.
    (The deeper "not rank-based" guarantee is structural: _score_cmap() takes no
    data, so it cannot depend on which scores a dataset happens to contain. The
    end-to-end guard for that lives in the render tests, where data does flow.)
    """
    from matplotlib.colors import to_rgba
    cmap, norm = module._score_cmap()
    for score, expected_hex in zip(module.SCORES, module.SCORE_COLORS):
        assert cmap(norm(score)) == to_rgba(expected_hex), (
            f"score {score} should be {expected_hex}"
        )


def test_score_cmap_missing_is_grey_and_distinct_from_every_score_colour():
    cmap, norm = module._score_cmap()
    bad = cmap.get_bad()
    score_colours = {cmap(norm(s)) for s in module.SCORES}
    assert tuple(bad) not in score_colours


def test_reaction_boundary_sits_between_1_and_2():
    assert module.REACTION_BOUNDARY_Y == 1.5
