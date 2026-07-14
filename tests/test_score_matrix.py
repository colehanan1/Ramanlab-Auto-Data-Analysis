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
    assert "Hexanol" in texts, f"non-trained odor should not be uppercased: {texts}"
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


def test_colorbar_key_is_present_and_labelled(tmp_path):
    """Deleting the whole colorbar block otherwise survives every test."""
    fig, _ = _render_v2(tmp_path)
    cax = next((a for a in fig.axes
                if a.get_ylabel() == "Odor Response Score"), None)
    assert cax is not None, (
        f"colorbar key missing; ylabels={[a.get_ylabel() for a in fig.axes]}"
    )


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


def test_missing_cell_renders_as_missing_colour(tmp_path):
    """A data gap must render as MISSING_COLOR grey on the real artist.

    Guards a silent scientific misread: without set_bad(), a gap renders
    transparent (white), which against score 1's near-white #f2ebf5 would look
    like "no reaction" rather than "no data".
    """
    import numpy as np
    from matplotlib.colors import to_rgba
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    # Drop one (fly, odor) pair so the matrix has a real gap.
    rows = [r for r in _v2_panel_rows()
            if not (r["fly"] == "f2" and r["trial_label"] == "testing_1_hexanol")]
    csv_path = tmp_path / "gap.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    import matplotlib.pyplot as plt
    plt.close("all")
    real_close = plt.close
    plt.close = lambda *a, **k: None
    try:
        module.generate_score_summary(csv_path=csv_path, out_dir=out_dir,
                                      overwrite=True)
    finally:
        plt.close = real_close
    fig = next(f for f in map(plt.figure, plt.get_fignums())
               if any(a.get_ylabel() == "Mean Score" for a in f.axes))
    ax_m = next(a for a in fig.axes if a.get_ylabel().endswith("Flies"))
    img = ax_m.images[0]
    arr = img.get_array()
    rgba = img.to_rgba(arr)
    gap = np.argwhere(np.ma.getmaskarray(arr))
    assert len(gap), "fixture produced no gap - test would be vacuous"
    i, j = gap[0]
    assert tuple(rgba[i, j]) == to_rgba(module.MISSING_COLOR), (
        f"gap rendered {tuple(rgba[i, j])}, expected MISSING_COLOR "
        f"{to_rgba(module.MISSING_COLOR)}"
    )
    # The assertion above only proves MISSING_COLOR *propagates* to the
    # render -- it reads its expected value from module.MISSING_COLOR itself,
    # so it can never disagree with a wrong-but-still-propagated constant
    # (e.g. MISSING_COLOR accidentally set to a saturated colour instead of a
    # neutral grey: both sides of the comparison above would shift together
    # and it would still pass). Pin the literal semantics independently.
    r, g, b, _a = to_rgba(module.MISSING_COLOR)
    assert r == g == b, (
        f"MISSING_COLOR must be a neutral grey, got rgb=({r}, {g}, {b})"
    )
    # r == g == b still admits ANY grey, so MISSING_COLOR = "1.0" (white) would
    # survive -- and a white gap is indistinguishable from score 1's near-white
    # #f2ebf5, which is the exact silent misread this test exists to prevent.
    # Pin the literal, as SCORE_COLORS is pinned above.
    assert module.MISSING_COLOR == "0.70"


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


# ---------------------------------------------------------------------------
# Task 1: trained-odor remap must not merge presentations (pseudoreplication)
# ---------------------------------------------------------------------------


def test_trained_odor_remap_does_not_merge_presentations(tmp_path):
    """A remap of the trained odor's display name must NOT merge its two
    presentations, and must NOT inflate n_flies.

    Regression: _should_number gated on exact equality against _trained_label
    ("Ethyl Butyrate"), so remapping the display to "Ethyl Butyrate (1%)" made
    the equality fail, the odor stopped being numbered, and both presentations
    collapsed into one column with n = 2 x flies. Silent pseudoreplication.
    """
    from scripts.analysis.envelope_visuals import set_protocol, set_dataset_odor_remap
    set_protocol("v2")
    rows = []
    # 3 flies, Ethyl Butyrate presented TWICE each (trials 2 and 4).
    for idx, fly in enumerate(("f1", "f2", "f3"), start=1):
        for label in ("testing_1_hexanol", "testing_2_ethylbutyrate",
                      "testing_4_ethylbutyrate"):
            rows.append({
                "dataset": "EB-Training", "fly": fly, "fly_number": str(idx),
                "trial_label": label, "score": 3, "trial_type": "testing",
            })
    csv_path = tmp_path / "s.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    set_dataset_odor_remap({"EB-Training": {"Ethyl Butyrate": "Ethyl Butyrate (1%)"}})
    try:
        df = module._load_scores(csv_path)
        summary = module._compute_summary(df)
    finally:
        set_dataset_odor_remap({})

    eb = summary[summary["odor_col"].str.startswith("Ethyl Butyrate")]
    assert len(eb) == 2, (
        f"trained odor's 2 presentations merged into {len(eb)} column(s): "
        f"{eb['odor_col'].tolist()}"
    )
    assert sorted(eb["odor_col"]) == ["Ethyl Butyrate (1%) 1",
                                      "Ethyl Butyrate (1%) 2"]
    assert set(eb["n_flies"]) == {3}, (
        f"n_flies inflated by pseudoreplication: {eb['n_flies'].tolist()} "
        f"(3 flies measured twice must stay n=3, never 6)"
    )


# ---------------------------------------------------------------------------
# Task 3: control|training score matrix PAIR figure (_plot_score_pair)
# ---------------------------------------------------------------------------


def test_score_pair_figure_written_with_shared_key(tmp_path):
    """Control|training score matrices in ONE figure with ONE shared key."""
    import matplotlib.pyplot as plt
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    plt.close("all")
    rows = []
    for ds, base in (("EB-Training", 4), ("EB-Control", 0)):
        for idx, fly in enumerate(("a", "b", "c"), start=1):
            for label in ("testing_1_hexanol", "testing_2_ethylbutyrate"):
                rows.append({
                    "dataset": ds, "fly": f"{ds}_{fly}", "fly_number": str(idx),
                    "trial_label": label, "score": base, "trial_type": "testing",
                })
    csv_path = tmp_path / "s.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    module.generate_score_summary(csv_path=csv_path, out_dir=out_dir, overwrite=True)
    assert (out_dir / "mean_score_pair_EB-Training.png").exists()


def _pair_rows(train_flies: dict, ctrl_flies: dict) -> list[dict]:
    """Minimal v2 rows for an EB-Training/EB-Control pair.

    ``train_flies``/``ctrl_flies`` map a short fly name to a (hexanol_score,
    ethylbutyrate_score) tuple; dict LENGTH sets that panel's fly count, so
    callers can deliberately make the two panels unequal -- equal fly counts
    can never expose a stretched-image `extent` bug (see the cell-height
    test below).
    """
    rows = []
    for ds, flies in (("EB-Training", train_flies), ("EB-Control", ctrl_flies)):
        for idx, (fly, (hex_s, eb_s)) in enumerate(flies.items(), start=1):
            for label, score in (("testing_1_hexanol", hex_s),
                                  ("testing_2_ethylbutyrate", eb_s)):
                rows.append({
                    "dataset": ds, "fly": f"{ds}_{fly}", "fly_number": str(idx),
                    "trial_label": label, "score": score, "trial_type": "testing",
                })
    return rows


def _render_pair(tmp_path, rows):
    """Render generate_score_summary() with plt.close disabled and hand back
    the live PAIR figure.

    The pair figure is identified by its `fig.suptitle` text ("Per-Fly Odor
    Response - ...") -- unlike every other figure this module renders
    (_plot_bar_charts's band/bar figure sets an AXES title via
    `ax.set_title`, never a figure suptitle), so this lookup cannot
    accidentally grab the wrong figure the way a bare "does any image
    match" scan can (see test_score_pair_reuses_the_validated_palette).
    """
    import matplotlib.pyplot as plt
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("v2")
    csv_path = tmp_path / "s.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    plt.close("all")
    real_close = plt.close
    plt.close = lambda *a, **k: None
    try:
        module.generate_score_summary(csv_path=csv_path, out_dir=out_dir,
                                      overwrite=True)
    finally:
        plt.close = real_close
    fig = next(f for f in map(plt.figure, plt.get_fignums())
               if f.get_suptitle().startswith("Per-Fly Odor Response"))
    return fig, out_dir


def _pair_panels(fig):
    """Pick out the control/training imshow axes by their own titles (set in
    _plot_score_pair as "Control (...)"/"Training (...)"), so callers never
    have to guess GridSpec ordering."""
    ax_c = next(a for a in fig.axes if a.get_title().startswith("Control"))
    ax_t = next(a for a in fig.axes if a.get_title().startswith("Training"))
    return ax_c, ax_t


def _measured_cell_height_in(ax, fig):
    """Rendered height, in inches, of ONE matrix cell on ``ax``.

    Same approach as tests/test_train_vs_ctrl_split.py::_measured_cell_height_in
    (duplicated locally rather than imported across test modules, matching
    this file's existing convention of self-contained helpers): `imshow`
    stretches its image to fill `extent` in DATA coordinates; `set_ylim`
    then maps a DATA span onto the axes' fixed PHYSICAL height. A cell's
    rendered height is therefore the axes' physical height scaled by the
    fraction of the ylim span that one image row occupies. Reading only
    `get_ylim()` cannot detect a mismatched image `extent` -- it is equal by
    construction whenever `set_ylim` is shared, bug or no bug.
    """
    im = ax.images[0]
    x0, x1, y_bottom, y_top = im.get_extent()   # y_bottom > y_top (inverted)
    rows = im.get_array().shape[0]
    ylo, yhi = ax.get_ylim()                    # ylo > yhi (inverted)
    data_span = abs(ylo - yhi)
    ax_h_in = ax.get_position().height * fig.get_figheight()
    img_span = abs(y_bottom - y_top)
    return (ax_h_in * (img_span / data_span)) / rows


def test_score_pair_reuses_the_validated_palette(tmp_path):
    """The pair figure must not introduce a second ramp: its cells must use
    the same CVD-validated SCORE_COLORS as the band.

    STRENGTHENED beyond the brief's verbatim version. The brief's version
    scanned every image on every OPEN figure (`for f in map(plt.figure,
    plt.get_fignums()) for a in f.axes for im in a.images`) without
    restricting to the pair figure. But generate_score_summary() ALSO
    renders the pre-existing per-dataset band/bar figures
    (_plot_bar_charts -> _draw_score_matrix) for the very same
    EB-Training/EB-Control datasets in the same call, and those already use
    the correct SCORE_COLORS via _score_cmap() -- untouched by anything
    _plot_score_pair does. Confirmed empirically two ways: (1) the brief's
    unscoped assertion already PASSED at RED, before _plot_score_pair
    existed at all -- the textbook vacuous case, a test that is green when
    the feature under test is absent; (2) mutation testing -- replacing
    _plot_score_pair's `cmap, norm = _score_cmap()` with a plain
    Normalize-based ramp left the unscoped assertion passing (see the task
    report for both outputs). Restricting to the pair figure's own artists
    (via the suptitle-based _render_pair lookup) closes both gaps: it fails
    at real RED and fails again under the cmap mutant.
    """
    from matplotlib.colors import to_rgba
    rows = _pair_rows(
        train_flies={"a": (4, 4), "b": (4, 4), "c": (4, 4)},
        ctrl_flies={"a": (0, 0), "b": (0, 0), "c": (0, 0)},
    )
    fig, _ = _render_pair(tmp_path, rows)
    imgs = [im for ax in fig.axes for im in ax.images]
    assert imgs, "no matrix artist rendered on the pair figure"
    # A score of 4 must wear SCORE_COLORS[4 - SCORE_MIN] on the pair figure too.
    hit = [im for im in imgs
           if tuple(im.cmap(im.norm(4))) ==
           to_rgba(module.SCORE_COLORS[4 - module.SCORE_MIN])]
    assert hit, "pair figure does not use the validated SCORE_COLORS ramp"


def test_pair_panels_share_measured_cell_height_with_unequal_fly_counts(tmp_path):
    """THE cell-height trap (Task 2 shipped this exact bug and it was caught
    in review: control cells measured 1.54in vs training 1.03in). `imshow`
    STRETCHES its image to fill `extent`, so giving a 2-row matrix a 3-row
    extent renders its cells 1.5x too tall.

    Equal fly counts can NEVER expose this: both panels would end at the
    same row whether `extent` used its own true row count or the shared
    max, so this fixture deliberately uses 3 training flies vs 2 control
    flies. Checking only `get_ylim()` equality is also NOT enough -- it is
    equal BY CONSTRUCTION (`ax.set_ylim(n_max - 0.5, -0.5)` is shared by
    both panels) whether or not the `extent` bug is present; only the
    ACTUAL rendered cell height, measured from each panel's own image
    extent and axes geometry, can tell the two implementations apart.
    """
    rows = _pair_rows(
        train_flies={"t1": (0, 5), "t2": (-1, 3), "t3": (0, 2)},
        ctrl_flies={"c1": (0, 0), "c2": (1, -1)},
    )
    fig, _ = _render_pair(tmp_path, rows)
    ax_c, ax_t = _pair_panels(fig)
    fig.canvas.draw()

    assert ax_c.get_ylim() == ax_t.get_ylim(), (
        "panels must share one y-range for a fair visual comparison"
    )
    ctrl_cell = _measured_cell_height_in(ax_c, fig)
    train_cell = _measured_cell_height_in(ax_t, fig)
    assert ctrl_cell == pytest.approx(train_cell, rel=1e-6), (
        f"cell heights differ: control {ctrl_cell:.4f}in vs training "
        f"{train_cell:.4f}in -- the eye would read a size difference that "
        f"is not in the data"
    )


def test_pair_panels_have_equal_width_and_one_shared_colorbar(tmp_path):
    """THE cax trap: `fig.colorbar(mappable, ax=ax_t)` steals space from
    ax_t to make room for an auto-inserted colorbar axes, shrinking ax_t
    relative to ax_c and breaking the pair's visual symmetry. A plain
    get_ylim() or x-span check cannot see this -- ax_t's DATA limits stay
    exactly [-0.5, n_col - 0.5] either way; only its PHYSICAL width on the
    page changes when the colorbar steals space from it instead of owning
    its own dedicated gridspec column. Comparing rendered axes-position
    widths catches it directly.

    This also operationalises the "ONE shared key" half of
    test_score_pair_figure_written_with_shared_key's docstring, which that
    test cannot itself verify: it never patches plt.close, so by the time
    its assertion runs every figure this call rendered has already been
    closed and there is nothing left to inspect.
    """
    rows = _pair_rows(
        train_flies={"t1": (0, 5), "t2": (-1, 3)},
        ctrl_flies={"c1": (0, 0), "c2": (1, -1)},
    )
    fig, _ = _render_pair(tmp_path, rows)
    ax_c, ax_t = _pair_panels(fig)

    pc, pt = ax_c.get_position(), ax_t.get_position()
    assert pc.width == pytest.approx(pt.width, rel=1e-6), (
        f"control panel width {pc.width:.4f} != training panel width "
        f"{pt.width:.4f} -- the colorbar must own its own gridspec column "
        f"(cax=...), not shrink one panel via fig.colorbar(ax=...)"
    )
    colorbar_axes = [a for a in fig.axes
                     if a.get_ylabel() == "Odor Response Score"]
    assert len(colorbar_axes) == 1, (
        f"expected exactly ONE shared key, found {len(colorbar_axes)}: "
        f"{[a.get_ylabel() for a in fig.axes]}"
    )


def test_legacy_protocol_writes_no_score_pair_figure(tmp_path):
    """Regression guard: the pair figure is v2-only
    (`if get_protocol() != "v2" ...: return`). Legacy output must stay
    byte-for-byte identical, i.e. no mean_score_pair_*.png must appear at
    all under the legacy protocol -- mirrors the existing
    test_legacy_figure_has_no_matrix_panel guard for the band figure.
    """
    from scripts.analysis.envelope_visuals import set_protocol
    set_protocol("legacy")
    rows = []
    for idx, fly in enumerate(("f1", "f2"), start=1):
        for label, score in (("testing_1", 0), ("testing_2", 5)):
            rows.append({
                "dataset": "EB-Training", "fly": fly, "fly_number": str(idx),
                "trial_label": label, "score": score, "trial_type": "testing",
            })
    csv_path = tmp_path / "s.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    module.generate_score_summary(csv_path=csv_path, out_dir=out_dir, overwrite=True)
    assert not list(out_dir.glob("mean_score_pair_*.png")), (
        "legacy protocol must not emit the v2-only score-pair figure"
    )
