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


# ---------------------------------------------------------------------------
# Task 3: render the matrix band in `_plot_bar_charts`
# ---------------------------------------------------------------------------


def _v2_panel_rows() -> list[dict[str, object]]:
    """3 flies x 2 odors, EB-Training (trained odor = Ethyl Butyrate)."""
    rows = []
    scores = {"f1": (0, 5), "f2": (-1, 3), "f3": (0, 2)}
    for idx, (fly, (hex_s, eb_s)) in enumerate(scores.items(), start=1):
        for label, s in (("testing_1_hexanol", hex_s),
                         ("testing_2_ethylbutyrate", eb_s)):
            rows.append({
                "dataset": "EB-Training", "fly": fly, "fly_number": str(idx),
                "trial_label": label, "score": s, "trial_type": "testing",
            })
    return rows


def _render_v2(tmp_path, rows=None):
    """Render the v2 figure and hand back its live axes.

    ``rows`` defaults to the standard 3-fly/2-odor panel; callers that need a
    different score distribution (e.g. the rank-invariance guard) can pass
    their own.
    """
    import matplotlib.pyplot as plt
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    csv_path = tmp_path / "s.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(rows if rows is not None else _v2_panel_rows()).to_csv(
        csv_path, index=False)

    # Start from a clean figure registry. The `plt.close` monkeypatch below
    # is a no-op (it must be, so the figure this test wants to inspect stays
    # retrievable via get_fignums() after generate_score_summary() "closes"
    # it) -- so any figure another test left open would otherwise linger and
    # be mistaken below for the one this call renders.
    plt.close("all")
    closed = []
    real_close = plt.close
    plt.close = lambda *a, **k: closed.append(a[0] if a else None)
    try:
        module.generate_score_summary(csv_path=csv_path, out_dir=out_dir,
                                      overwrite=True)
    finally:
        plt.close = real_close
    fig = next(f for f in map(plt.figure, plt.get_fignums())
               if any(a.get_ylabel() == "Mean Score" for a in f.axes))
    return fig, out_dir


def test_v2_figure_is_written(tmp_path):
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    csv_path = tmp_path / "s.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(_v2_panel_rows()).to_csv(csv_path, index=False)
    module.generate_score_summary(csv_path=csv_path, out_dir=out_dir,
                                  overwrite=True)
    assert (out_dir / "mean_score_EB-Training.png").exists()


def test_v2_figure_has_matrix_and_bar_axes(tmp_path):
    fig, _ = _render_v2(tmp_path)
    ylabels = {a.get_ylabel() for a in fig.axes}
    assert "Mean Score" in ylabels
    assert any(lbl.endswith("Flies") for lbl in ylabels), "no matrix panel"


def test_matrix_and_bars_share_an_identical_x_span(tmp_path):
    """TRAP 1: fig.colorbar(ax=ax_m) shrinks only the matrix and breaks this."""
    fig, _ = _render_v2(tmp_path)
    ax_m = next(a for a in fig.axes if a.get_ylabel().endswith("Flies"))
    ax_b = next(a for a in fig.axes if a.get_ylabel() == "Mean Score")
    pm, pb = ax_m.get_position(), ax_b.get_position()
    assert pm.x0 == pytest.approx(pb.x0, abs=1e-9), "matrix/bars misaligned"
    assert pm.x1 == pytest.approx(pb.x1, abs=1e-9), "matrix/bars misaligned"
    assert ax_m.get_xlim() == ax_b.get_xlim()


def test_bar_labels_survive_matrix_labelling(tmp_path):
    """TRAP 2: sharex makes ax_m and the bar axis share ONE major-formatter
    instance (verified: `ax_m.xaxis.get_major_formatter() is
    ax_b.xaxis.get_major_formatter()`), so whichever axis last calls
    set_xticklabels() wins for BOTH. _draw_score_matrix runs before the bar
    labelling code, so labelling ax_m directly (instead of on its own
    secondary axis) would let the bars' later call silently overwrite the
    matrix's odor labels with its own "(n=...)" text -- checking only the
    bar axis would miss that, since the bar's text is untouched either way.
    """
    fig, _ = _render_v2(tmp_path)
    ax_m = next(a for a in fig.axes if a.get_ylabel().endswith("Flies"))
    ax_b = next(a for a in fig.axes if a.get_ylabel() == "Mean Score")
    bar_texts = [t.get_text() for t in ax_b.get_xticklabels()]
    matrix_texts = [t.get_text() for t in ax_m.get_xticklabels()]
    assert any("(n=" in t for t in bar_texts), f"bar labels clobbered: {bar_texts}"
    assert not any("(n=" in t for t in matrix_texts), (
        f"ax_m picked up the bar's own labels via the shared formatter -- "
        f"odor labels must live on ax_m's secondary axis: {matrix_texts}"
    )


def test_matrix_carries_its_own_odor_labels_below(tmp_path):
    """Positive guard for the odor-label block.

    The sibling bar-label test cannot cover this: with labelbottom=False,
    ax_m.get_xticklabels() is [] against the CORRECT implementation, so a
    negative assertion there passes over an empty list. Only a positive
    assertion on the secondary axis kills the mutants that delete or misplace
    the labels.
    """
    fig, _ = _render_v2(tmp_path)
    ax_m = next(a for a in fig.axes if a.get_ylabel().endswith("Flies"))
    assert ax_m.child_axes, "matrix has no secondary axis -> odor labels missing"
    ax_lab = ax_m.child_axes[0]
    texts = [t.get_text() for t in ax_lab.get_xticklabels()]
    assert any("HEXANOL" == t or "Hexanol" == t for t in texts), texts
    trained = [t for t in ax_lab.get_xticklabels()
               if t.get_text() == "ETHYL BUTYRATE"]
    assert trained, f"trained odor not uppercased on the matrix: {texts}"
    assert trained[0].get_color() == "#1a3a6b"
    assert trained[0].get_weight() == "bold"


def test_bar_y_axis_spans_the_full_score_range(tmp_path):
    fig, _ = _render_v2(tmp_path)
    ax_b = next(a for a in fig.axes if a.get_ylabel() == "Mean Score")
    assert ax_b.get_ylim() == (module.SCORE_MIN, module.SCORE_MAX)
    assert [int(t) for t in ax_b.get_yticks()] == module.SCORES


def test_matrix_has_no_y_tick_labels(tmp_path):
    fig, _ = _render_v2(tmp_path)
    ax_m = next(a for a in fig.axes if a.get_ylabel().endswith("Flies"))
    assert len(ax_m.get_yticks()) == 0


def test_score_colour_does_not_depend_on_which_scores_are_present(tmp_path):
    """THE rank-invariance guard: a 3 is the same green whether or not the
    dataset contains a 1.

    Task 1's unit test cannot prove this — _score_cmap() takes no data, so
    nothing can be "a frame missing score 1". Here data actually flows: this
    renders the real matrix panel for two datasets with different score sets
    and reads back the cmap/norm the render call site (_draw_score_matrix)
    actually handed to imshow -- via the live AxesImage artist -- rather
    than calling _score_cmap() a second time in isolation. A second, direct
    _score_cmap() call can never disagree with itself regardless of what
    data was loaded in between, so it would silently pass even if the
    renderer built a quantile/rank-based norm from the matrix's own
    min/max; reading the artist's actual norm closes that gap and this test
    would catch it.
    """
    from matplotlib.colors import to_rgba

    def _rendered_colour_of_3(rows):
        fig, _ = _render_v2(tmp_path, rows)
        ax_m = next(a for a in fig.axes if a.get_ylabel().endswith("Flies"))
        img = ax_m.images[0]
        return img.cmap(img.norm(3))

    dense = _v2_panel_rows()                       # scores {-1, 0, 2, 3, 5}
    sparse = [r for r in _v2_panel_rows()          # drop everything but 0 and 3
              if r["score"] in (0, 3)]
    assert _rendered_colour_of_3(dense) == _rendered_colour_of_3(sparse)
    assert _rendered_colour_of_3(dense) == to_rgba(module.SCORE_COLORS[
        module.SCORES.index(3)])


def test_legacy_figure_has_no_matrix_panel(tmp_path):
    """Regression guard: legacy mean_score_*.png must render as before."""
    import matplotlib.pyplot as plt
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("legacy")
    csv_path = tmp_path / "s.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(_v2_panel_rows()).to_csv(csv_path, index=False)

    # See _render_v2: without a clean registry, an earlier test's un-closed
    # (deliberately, via its own plt.close monkeypatch) v2 figure could be
    # mistaken below for the one this call renders.
    plt.close("all")
    real_close = plt.close
    plt.close = lambda *a, **k: None
    try:
        module.generate_score_summary(csv_path=csv_path, out_dir=out_dir,
                                      overwrite=True)
    finally:
        plt.close = real_close

    assert (out_dir / "mean_score_EB-Training.png").exists()
    fig = next(f for f in map(plt.figure, plt.get_fignums())
               if any(a.get_ylabel() == "Mean Score" for a in f.axes))
    assert not any(a.get_ylabel().endswith("Flies") for a in fig.axes), \
        "legacy figure grew a matrix panel"


def test_matrix_column_absent_from_data_is_all_nan(tmp_path):
    """Task 3 passes an independently-derived column list, so a column with no
    matching rows must yield an all-NaN column, not an error or a shift."""
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    csv_path = tmp_path / "s.csv"
    pd.DataFrame(_v2_panel_rows()).to_csv(csv_path, index=False)
    df = module._load_scores(csv_path)
    columns = ["Hexanol", "Ethyl Butyrate", "Nonexistent Odor"]
    mat, flies = module._per_fly_score_matrix(df, "EB-Training", columns)
    assert mat.shape == (len(flies), 3)
    assert np.isnan(mat[:, 2]).all(), "absent column should be entirely NaN"
    assert not np.isnan(mat[:, 0]).all(), "present column must still be filled"
