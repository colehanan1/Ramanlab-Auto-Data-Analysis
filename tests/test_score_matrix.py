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


def _matrix_rows() -> list[dict[str, object]]:
    """Two flies x three odors, v2-style labels with odor suffixes."""
    rows = []
    data = {
        "fly_a": {"testing_1_hexanol": 0, "testing_2_ethylbutyrate": 5},
        "fly_b": {"testing_1_hexanol": -1, "testing_2_ethylbutyrate": 3},
    }
    for idx, (fly, trials) in enumerate(data.items(), start=1):
        for label, score in trials.items():
            rows.append({
                "dataset": "EB-Training",
                "fly": fly,
                "fly_number": str(idx),
                "trial_label": label,
                "score": score,
                "trial_type": "testing",
            })
    return rows


def test_per_fly_matrix_shape_and_column_alignment(tmp_path):
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    csv_path = tmp_path / "s.csv"
    pd.DataFrame(_matrix_rows()).to_csv(csv_path, index=False)
    df = module._load_scores(csv_path)
    summary = module._compute_summary(df)
    columns = summary["odor_col"].tolist()

    mat, flies = module._per_fly_score_matrix(df, "EB-Training", columns)

    assert mat.shape == (len(flies), len(columns))
    assert len(flies) == 2


def test_per_fly_matrix_values_land_in_the_right_cell(tmp_path):
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    csv_path = tmp_path / "s.csv"
    pd.DataFrame(_matrix_rows()).to_csv(csv_path, index=False)
    df = module._load_scores(csv_path)
    summary = module._compute_summary(df)
    columns = summary["odor_col"].tolist()

    mat, flies = module._per_fly_score_matrix(df, "EB-Training", columns)

    eb_col = columns.index("Ethyl Butyrate")
    hex_col = columns.index("Hexanol")
    a_row = flies.index("fly_a|1")
    b_row = flies.index("fly_b|2")
    assert mat[a_row, eb_col] == 5
    assert mat[a_row, hex_col] == 0
    assert mat[b_row, eb_col] == 3
    assert mat[b_row, hex_col] == -1


def test_per_fly_matrix_absent_pair_is_nan(tmp_path):
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    rows = _matrix_rows()
    # Drop fly_b's hexanol trial -> that cell must be NaN, not 0.
    rows = [r for r in rows
            if not (r["fly"] == "fly_b" and r["trial_label"] == "testing_1_hexanol")]
    csv_path = tmp_path / "s.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = module._load_scores(csv_path)
    summary = module._compute_summary(df)
    columns = summary["odor_col"].tolist()

    mat, flies = module._per_fly_score_matrix(df, "EB-Training", columns)

    assert np.isnan(mat[flies.index("fly_b|2"), columns.index("Hexanol")])


def test_per_fly_matrix_keys_rows_on_fly_and_fly_number(tmp_path):
    """Two flies sharing a `fly` value but differing `fly_number` stay distinct."""
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    rows = []
    for fly_number, score in (("1", 0), ("2", 5)):
        rows.append({
            "dataset": "EB-Training", "fly": "same_name",
            "fly_number": fly_number, "trial_label": "testing_2_ethylbutyrate",
            "score": score, "trial_type": "testing",
        })
    csv_path = tmp_path / "s.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = module._load_scores(csv_path)
    summary = module._compute_summary(df)
    columns = summary["odor_col"].tolist()

    mat, flies = module._per_fly_score_matrix(df, "EB-Training", columns)

    assert len(flies) == 2, "same fly name with different fly_number collapsed"
    assert mat.shape[0] == 2


def test_per_fly_matrix_ignores_other_datasets(tmp_path):
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    rows = _matrix_rows()
    rows.append({
        "dataset": "EB-Control", "fly": "ctrl_fly", "fly_number": "9",
        "trial_label": "testing_2_ethylbutyrate", "score": 5,
        "trial_type": "testing",
    })
    csv_path = tmp_path / "s.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = module._load_scores(csv_path)
    summary = module._compute_summary(df)
    columns = summary[summary["dataset_canon"] == "EB-Training"]["odor_col"].tolist()

    mat, flies = module._per_fly_score_matrix(df, "EB-Training", columns)

    assert "ctrl_fly|9" not in flies
    assert len(flies) == 2
