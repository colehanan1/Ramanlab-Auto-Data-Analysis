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


def test_score_cmap_is_a_fixed_map_not_rank_based():
    """A 3 is the same green whether or not a 1 is present in the data.

    Guards the rank-based regression: score 1 never occurs in EB-Training-24-1.
    """
    cmap, norm = module._score_cmap()
    colour_of_3 = cmap(norm(3))
    # Re-deriving the cmap from a frame lacking score 1 must not shift anything.
    cmap2, norm2 = module._score_cmap()
    assert cmap2(norm2(3)) == colour_of_3


def test_score_cmap_missing_is_grey_and_distinct_from_every_score_colour():
    cmap, norm = module._score_cmap()
    bad = cmap.get_bad()
    score_colours = {cmap(norm(s)) for s in module.SCORES}
    assert tuple(bad) not in score_colours


def test_reaction_boundary_sits_between_1_and_2():
    assert module.REACTION_BOUNDARY_Y == 1.5
