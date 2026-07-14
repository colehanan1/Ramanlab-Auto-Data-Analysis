"""Tests for splitting ``reaction_matrix_train_vs_ctrl_*`` into two figures:
a bars-only figure (today's file name, matrix removed) and a NEW
``reaction_matrix_pair_*`` figure with the control matrix on the left and the
training matrix on the right.

Covers:
  - ``_build_during_matrix`` exists and is reusable for both datasets.
  - Every filter carried over from the original inlined block (testing_11
    drop, non-reactive flagging, fly-pair sort order, light-only drop, the
    v2-only no-odor-suffix drop) still behaves as it did before extraction.
  - The load-bearing detail: BOTH panels resolve odor-display substitutions
    from the TRAINING dataset (``remap_from``), never from whichever dataset
    is actually being rendered.
  - Column-union: a caller-supplied ``columns`` list is honoured verbatim,
    and an odor absent from that dataset's own data becomes an all-NaN
    (grey) column rather than shifting the layout.
  - A real end-to-end run of ``generate_training_vs_control_matrices``
    produces both files, and the pair figure's two matrix panels have the
    same RENDERED cell height in inches -- measured from each panel's own
    image extent and axes geometry, not just an equal-by-construction
    y-limit -- and identical column labels.
  - ``_build_during_matrix`` filters by ``genotype`` (the ``fly_type``
    column) when one is given, excluding flies of other genotypes.
"""

from __future__ import annotations

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
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

MODULE_PATH = PROJECT_ROOT / "scripts" / "analysis" / "reaction_matrix_training_vs_control.py"
spec = importlib.util.spec_from_file_location("rm_tvc", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from scripts.analysis.envelope_visuals import (  # noqa: E402
    set_dataset_odor_remap,
    set_protocol,
)


def _v2_rows(dataset: str, flies: list[tuple[str, str, list[tuple[str, int]]]]) -> pd.DataFrame:
    """Build a minimal frame with the columns ``_build_during_matrix`` reads
    directly (bypassing the CSV-loading / trial-label-normalisation prologue
    in ``generate_training_vs_control_matrices``, which is not under test
    here)."""
    rows = []
    for fly, fly_number, trials in flies:
        for trial_label, hit in trials:
            rows.append(
                {
                    "dataset_canon": dataset,
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial": trial_label,
                    "during_hit": hit,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1 (brief): extraction + signature
# ---------------------------------------------------------------------------


def test_build_during_matrix_is_extracted_and_callable():
    """The matrix builder must be reusable for BOTH datasets, not inlined in
    the training-only loop."""
    assert hasattr(module, "_build_during_matrix"), (
        "matrix construction still inlined; cannot build the control panel"
    )


def test_build_during_matrix_respects_caller_supplied_columns():
    """The pair figure needs identical columns in both panels, so the control
    call must accept the training call's column list and honour its order."""
    import inspect

    sig = inspect.signature(module._build_during_matrix)
    assert "columns" in sig.parameters, (
        f"no way to force shared columns: {list(sig.parameters)}"
    )
    assert "remap_from" in sig.parameters, (
        "control panel must resolve its odor remap from the TRAINING dataset "
        f"(see the comment at reaction_matrix_training_vs_control.py:676): "
        f"{list(sig.parameters)}"
    )


# ---------------------------------------------------------------------------
# Behavioural tests: v2 matrix construction
# ---------------------------------------------------------------------------


def test_v2_matrix_values_land_in_the_right_fly_odor_cell():
    set_protocol("v2")
    df = _v2_rows(
        "Zz-Training",
        [
            ("f1", "1", [("testing_1_Hexanol", 1), ("testing_2_Ethylbutyrate", 0)]),
            ("f2", "2", [("testing_1_Hexanol", 0), ("testing_2_Ethylbutyrate", 1)]),
        ],
    )
    mat, fly_pairs, cols, flagged = module._build_during_matrix(
        df, "Zz-Training", None, remap_from="Zz-Training"
    )
    assert fly_pairs == [("f1", "1"), ("f2", "2")]
    assert sorted(cols) == ["Ethyl Butyrate", "Hexanol"]
    hex_idx, eb_idx = cols.index("Hexanol"), cols.index("Ethyl Butyrate")
    assert (mat[0, hex_idx], mat[1, hex_idx]) == (1.0, 0.0)
    assert (mat[0, eb_idx], mat[1, eb_idx]) == (0.0, 1.0)
    assert flagged == set()


def test_remap_resolves_from_remap_from_not_from_the_rendered_dataset():
    """THE load-bearing detail (reaction_matrix_training_vs_control.py:676-678):
    the control panel's odor-display substitution must come from the TRAINING
    dataset's remap, never its own. Otherwise the two panels can carry
    different column labels for what is supposed to be the same column, and
    the visual comparison is meaningless.
    """
    set_protocol("v2")
    set_dataset_odor_remap({"Zz-Training": {"Hexanol": "Hexanol Custom"}})
    try:
        train_df = _v2_rows("Zz-Training", [("f1", "1", [("testing_1_Hexanol", 1)])])
        ctrl_df = _v2_rows("Zz-Control", [("c1", "1", [("testing_1_Hexanol", 0)])])
        df = pd.concat([train_df, ctrl_df], ignore_index=True)

        _, _, train_cols, _ = module._build_during_matrix(
            df, "Zz-Training", None, remap_from="Zz-Training"
        )
        assert train_cols == ["Hexanol Custom"]

        # The exact call the loop makes for the control panel: remap_from is
        # still the TRAINING dataset, even though `dataset` is now Control.
        ctrl_mat, ctrl_pairs, ctrl_cols, _ = module._build_during_matrix(
            df, "Zz-Control", None, remap_from="Zz-Training", columns=train_cols
        )
        assert ctrl_cols == ["Hexanol Custom"], (
            "control panel resolved its own dataset's remap instead of the "
            f"training dataset's: got {ctrl_cols}"
        )
        assert ctrl_pairs == [("c1", "1")]
        assert ctrl_mat[0, 0] == 0.0

        # Sanity/contrast: resolving from the control's OWN dataset (which has
        # no remap registered) gives a DIFFERENT label. This proves the two
        # resolutions are not accidentally identical for some unrelated
        # reason (e.g. the remap dict being empty) -- remap_from is actually
        # doing the steering.
        _, _, ctrl_cols_own, _ = module._build_during_matrix(
            df, "Zz-Control", None, remap_from="Zz-Control"
        )
        assert ctrl_cols_own == ["Hexanol"]
    finally:
        set_dataset_odor_remap({})


def test_columns_forcing_absent_odor_becomes_all_nan_column_not_a_shift():
    """Column union: an odor tested in training but never tested in control
    must appear as an all-NaN (grey) column in the SAME position, not be
    dropped (which would shift every later column left)."""
    set_protocol("v2")
    train_df = _v2_rows(
        "Zz-Training",
        [("f1", "1", [("testing_1_Hexanol", 1), ("testing_2_Ethylbutyrate", 1)])],
    )
    # Control never tested Ethyl Butyrate at all.
    ctrl_df = _v2_rows("Zz-Control", [("c1", "1", [("testing_1_Hexanol", 0)])])
    df = pd.concat([train_df, ctrl_df], ignore_index=True)

    _, _, train_cols, _ = module._build_during_matrix(
        df, "Zz-Training", None, remap_from="Zz-Training"
    )
    assert sorted(train_cols) == ["Ethyl Butyrate", "Hexanol"]

    ctrl_mat, ctrl_pairs, ctrl_cols, _ = module._build_during_matrix(
        df, "Zz-Control", None, remap_from="Zz-Training", columns=train_cols
    )
    assert ctrl_cols == train_cols, "control must reuse training's exact column order"
    eb_idx, hex_idx = ctrl_cols.index("Ethyl Butyrate"), ctrl_cols.index("Hexanol")
    assert np.isnan(ctrl_mat[:, eb_idx]).all(), "odor absent from control must be all-NaN"
    assert ctrl_mat[0, hex_idx] == 0.0


# ---------------------------------------------------------------------------
# Behavioural tests: every filter from the original block, preserved exactly
# ---------------------------------------------------------------------------


def test_testing_11_trials_are_dropped():
    """_is_testing_11_label must drop testing_11 regardless of its odor
    suffix -- verified by a suffix that would otherwise create a brand-new,
    easily-detected column."""
    set_protocol("v2")
    df = _v2_rows(
        "Zz-Training",
        [("f1", "1", [("testing_1_Hexanol", 1), ("testing_11_Citral", 1)])],
    )
    _, _, cols, _ = module._build_during_matrix(
        df, "Zz-Training", None, remap_from="Zz-Training"
    )
    assert cols == ["Hexanol"], f"testing_11 leaked into columns: {cols}"


def test_light_only_trial_number_dropped_regardless_of_odor_suffix():
    """_is_light_only_label drops by TRIAL NUMBER (9 under v2), independent
    of whatever odor suffix happens to be in the label."""
    set_protocol("v2")
    df = _v2_rows(
        "Zz-Training",
        [("f1", "1", [("testing_1_Hexanol", 1), ("testing_9_Citral", 1)])],
    )
    _, _, cols, _ = module._build_during_matrix(
        df, "Zz-Training", None, remap_from="Zz-Training"
    )
    assert cols == ["Hexanol"], f"trial 9 (light-only slot) leaked into columns: {cols}"


def test_non_reactive_flies_are_excluded_and_reported_as_flagged():
    set_protocol("v2")
    df = _v2_rows(
        "Zz-Training",
        [
            ("f1", "1", [("testing_1_Hexanol", 1)]),
            ("f2", "2", [("testing_1_Hexanol", 0)]),
        ],
    )
    df["_non_reactive"] = [False, True]
    _, fly_pairs, _, flagged = module._build_during_matrix(
        df, "Zz-Training", None, remap_from="Zz-Training"
    )
    assert fly_pairs == [("f1", "1")], f"non-reactive fly not excluded: {fly_pairs}"
    assert flagged == {("f2", "2")}, f"flagged pair not reported back: {flagged}"


def test_genotype_filter_excludes_flies_of_other_genotypes():
    """``genotype`` (matched against the ``fly_type`` column) must restrict
    the matrix to flies of that genotype only. This is a verbatim 3-line
    substitution per the Task 2 brief (see reaction_matrix_training_vs_control.py
    around line 555-556), so risk is low, but it was previously untested --
    a fixture with two genotypes must show the other genotype's fly and its
    data excluded, not merely ignored positionally."""
    set_protocol("v2")
    df = _v2_rows(
        "Zz-Training",
        [
            ("f1", "1", [("testing_1_Hexanol", 1)]),
            ("f2", "2", [("testing_1_Hexanol", 0)]),
        ],
    )
    df["fly_type"] = ["GenoA", "GenoB"]
    mat, fly_pairs, cols, _ = module._build_during_matrix(
        df, "Zz-Training", "GenoA", remap_from="Zz-Training"
    )
    assert fly_pairs == [("f1", "1")], (
        f"genotype filter let a fly of another genotype through: {fly_pairs}"
    )
    assert cols == ["Hexanol"]
    assert list(mat[:, 0]) == [1.0], "wrong fly's data landed in the filtered matrix"


def test_fly_pairs_sorted_by_fly_sort_key_not_encounter_order():
    """fly_pairs must be name-then-numeric sorted (2 before 10), not row
    encounter order and not a plain string sort (which would put '10' before
    '2')."""
    set_protocol("v2")
    df = _v2_rows(
        "Zz-Training",
        [
            ("Zebra", "10", [("testing_1_Hexanol", 1)]),
            ("Zebra", "2", [("testing_1_Hexanol", 1)]),
            ("Alpha", "1", [("testing_1_Hexanol", 1)]),
        ],
    )
    _, fly_pairs, _, _ = module._build_during_matrix(
        df, "Zz-Training", None, remap_from="Zz-Training"
    )
    assert fly_pairs == [("Alpha", "1"), ("Zebra", "2"), ("Zebra", "10")], fly_pairs


def test_no_odor_suffix_drop_is_v2_only_legacy_keeps_plain_testing_n():
    """Legacy labels are plain testing_N (the odor comes from the fixed
    schedule via _display_odor, not the label). Applying the v2-only
    no-odor-suffix filter under legacy would drop every trial and empty the
    matrix -- exactly the regression tests/test_reaction_matrix_legacy_odor.py
    guards against in reaction_matrix_from_spreadsheet.py's sibling filter."""
    set_protocol("legacy")
    df = pd.DataFrame(
        [
            {"dataset_canon": "Hex-Training", "fly": "f1", "fly_number": "1",
             "trial": "testing_1", "during_hit": 1},
            {"dataset_canon": "Hex-Training", "fly": "f1", "fly_number": "1",
             "trial": "testing_2", "during_hit": 0},
        ]
    )
    mat, fly_pairs, cols, _ = module._build_during_matrix(
        df, "Hex-Training", None, remap_from="Hex-Training"
    )
    assert len(fly_pairs) == 1, "legacy matrix wrongly emptied by the v2-only filter"
    assert mat.size > 0
    assert cols == ["Apple Cider Vinegar", "Hexanol"]


def test_returns_empty_tuple_contract_when_dataset_absent():
    """Every `continue` guard in the original inlined block becomes an early
    return of this exact empty tuple."""
    set_protocol("v2")
    df = _v2_rows("Zz-Training", [("f1", "1", [("testing_1_Hexanol", 1)])])
    mat, fly_pairs, cols, flagged = module._build_during_matrix(
        df, "Nonexistent-Training", None, remap_from="Nonexistent-Training"
    )
    assert mat.shape == (0, 0)
    assert fly_pairs == []
    assert cols == []
    assert flagged == set()


# ---------------------------------------------------------------------------
# Legacy branch: values + order parameter
# ---------------------------------------------------------------------------


def test_legacy_matrix_values_and_schedule_based_labels():
    set_protocol("legacy")
    df = pd.DataFrame(
        [
            {"dataset_canon": "Hex-Training", "fly": "f1", "fly_number": "1",
             "trial": "testing_1", "during_hit": 1},
            {"dataset_canon": "Hex-Training", "fly": "f1", "fly_number": "1",
             "trial": "testing_2", "during_hit": 0},
        ]
    )
    mat, fly_pairs, cols, _ = module._build_during_matrix(
        df, "Hex-Training", None, remap_from="Hex-Training", order="observed"
    )
    assert fly_pairs == [("f1", "1")]
    # testing_1 -> Apple Cider Vinegar, testing_2 -> Hexanol per the fixed
    # legacy schedule resolved by envelope_visuals._display_odor.
    assert cols == ["Apple Cider Vinegar", "Hexanol"]
    assert list(mat[0]) == [1.0, 0.0]


def test_legacy_remap_from_controls_label_resolution_not_dataset():
    """Legacy has no per-dataset odor_remap dict, but _display_odor still
    takes a dataset_canon argument that selects WHICH fixed schedule to
    read (see envelope_visuals._display_odor's testing-trial branch). The
    load-bearing detail applies here too: label resolution must come from
    remap_from, not from whichever dataset is actually being rendered.

    Hex-Training and Hex-Control happen to resolve to identical labels in
    real data (both alias to the same testing schedule via
    resolve_testing_alias), which is why test_legacy_matrix_values_and_
    schedule_based_labels alone cannot distinguish remap_from from dataset
    -- confirmed by mutation testing (see the report). AIR-Training's
    testing-trial 2 slot ("AIR") has no such alias to a generic dataset
    name's slot 2 (falls through to DISPLAY_LABEL, i.e. the literal dataset
    name), so this pair of dataset_canon values DOES distinguish the two
    parameters.
    """
    set_protocol("legacy")
    df = pd.DataFrame(
        [
            {"dataset_canon": "Zz-Training", "fly": "f1", "fly_number": "1",
             "trial": "testing_2", "during_hit": 1},
        ]
    )
    _, _, cols_own, _ = module._build_during_matrix(
        df, "Zz-Training", None, remap_from="Zz-Training"
    )
    assert cols_own == ["Zz-Training"], cols_own

    _, _, cols_from_air, _ = module._build_during_matrix(
        df, "Zz-Training", None, remap_from="AIR-Training"
    )
    assert cols_from_air == ["AIR"], (
        f"legacy label resolution used `dataset` instead of `remap_from`: {cols_from_air}"
    )


def test_order_parameter_actually_changes_legacy_column_order():
    """`order` must be threaded all the way through to _trial_order_for, not
    silently defaulted to 'observed' for every call. TRAINED_FIRST_ORDER =
    (2, 4, 5, 1, 3, ...), so with trials 1-5 present, 'trained-first' must
    reorder them; 'observed' must not."""
    set_protocol("legacy")
    df = pd.DataFrame(
        [
            {"dataset_canon": "Hex-Training", "fly": "f1", "fly_number": "1",
             "trial": f"testing_{n}", "during_hit": 0}
            for n in range(1, 6)
        ]
    )
    _, _, cols_observed, _ = module._build_during_matrix(
        df, "Hex-Training", None, remap_from="Hex-Training", order="observed"
    )
    _, _, cols_trained_first, _ = module._build_during_matrix(
        df, "Hex-Training", None, remap_from="Hex-Training", order="trained-first"
    )
    assert cols_observed != cols_trained_first, (
        "changing order had no effect -- the `order` argument is not being "
        "threaded through to _trial_order_for"
    )
    # observed: trial-number order. Hex-Training testing_1/3 -> ACV,
    # testing_2/4/5 -> Hexanol (see envelope_visuals._display_odor).
    assert cols_observed == [
        "Apple Cider Vinegar", "Hexanol", "Apple Cider Vinegar", "Hexanol", "Hexanol",
    ]
    # trained-first: TRAINED_FIRST_ORDER = (2, 4, 5, 1, 3, ...) -> trials
    # 2,4,5,1,3 in that order -> Hexanol,Hexanol,Hexanol,ACV,ACV.
    assert cols_trained_first == [
        "Hexanol", "Hexanol", "Hexanol", "Apple Cider Vinegar", "Apple Cider Vinegar",
    ]


# ---------------------------------------------------------------------------
# End-to-end: real figure emission (bars-only + control|training pair)
# ---------------------------------------------------------------------------


def _write_predictions_csv(path: Path) -> None:
    rows = []
    # Training: 3 flies. Control: 2 flies -- deliberately unequal so the
    # cell-height-equalisation logic in Step 5 is actually exercised.
    for fly, fn in (("t1", "1"), ("t2", "2"), ("t3", "3")):
        for label, hit in (("testing_1_Hexanol", 1), ("testing_2_Ethylbutyrate", 0)):
            rows.append({"dataset": "Zz-Training", "fly": fly, "fly_number": fn,
                         "trial_label": label, "prediction": hit})
    for fly, fn in (("c1", "1"), ("c2", "2")):
        for label, hit in (("testing_1_Hexanol", 0), ("testing_2_Ethylbutyrate", 0)):
            rows.append({"dataset": "Zz-Control", "fly": fly, "fly_number": fn,
                         "trial_label": label, "prediction": hit})
    pd.DataFrame(rows).to_csv(path, index=False)


def _measured_cell_height_in(ax, fig):
    """Rendered height, in inches, of ONE matrix cell on ``ax``.

    ``imshow`` stretches its image to fill ``extent`` in DATA coordinates;
    ``set_ylim`` then maps a DATA span onto the axes' fixed PHYSICAL height.
    A cell's rendered height is therefore the axes' physical height scaled
    by the fraction of the ylim span that one image row occupies. Reading
    only ``get_ylim()`` cannot detect a mismatched image ``extent`` -- it is
    equal by construction whenever ``set_ylim`` is shared, bug or no bug.
    """
    im = ax.images[0]
    x0, x1, y_bottom, y_top = im.get_extent()   # y_bottom > y_top (inverted)
    rows = im.get_array().shape[0]
    ylo, yhi = ax.get_ylim()                    # ylo > yhi (inverted)
    data_span = abs(ylo - yhi)
    ax_h_in = ax.get_position().height * fig.get_figheight()
    img_span = abs(y_bottom - y_top)
    return (ax_h_in * (img_span / data_span)) / rows


def test_pair_figure_end_to_end_shares_cell_height_and_columns(tmp_path, monkeypatch):
    """Real end-to-end check driven through the actual rendering code, not a
    hand-rolled parallel implementation.

    The brief's Step 6 test (``test_pair_panels_share_cell_height_and_columns``)
    built its own throwaway figure using matplotlib calls that merely MIRROR
    Step 5's approach; it never called into the module, so it could not fail
    no matter what ``generate_training_vs_control_matrices`` actually
    renders -- it has been deleted as dead weight.

    This test's own ``get_ylim()`` equality checks below are ALSO
    insufficient on their own: ``set_ylim`` is fixed to
    ``max(n_ctrl, n_train)`` in both panels regardless of whether each
    panel's ``imshow`` ``extent`` uses its OWN row count (correct) or the
    shared max (buggy, stretches the shorter panel's cells), so those ylim
    values are equal BY CONSTRUCTION either way and can never expose that
    bug. The assertion below instead measures each panel's ACTUAL rendered
    cell height in inches (``_measured_cell_height_in``), which does differ
    between the two versions -- confirmed by reverting the extent fix and
    observing this assertion fail with a ~1.5x ratio (see the report).
    """
    import matplotlib.pyplot as plt

    set_protocol("v2")
    plt.close("all")
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    csv_path = tmp_path / "predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "out"
    cfg = module.SpreadsheetMatrixConfig(
        csv_path=csv_path, out_dir=out_dir, latency_sec=2.15,
        trial_orders=("observed",),
    )
    module.generate_training_vs_control_matrices(cfg)

    bar_files = list(out_dir.glob("**/reaction_matrix_train_vs_ctrl_Zz-Training_*.png"))
    pair_files = list(out_dir.glob("**/reaction_matrix_pair_Zz-Training_*.png"))
    assert bar_files, "bars-only figure was not written under the pre-existing filename"
    assert pair_files, "control|training pair figure was not written"

    pair_figs = [
        f for f in map(plt.figure, plt.get_fignums())
        if len(f.axes) == 2 and all(ax.images for ax in f.axes)
    ]
    assert pair_figs, "no rendered figure has two matrix (imshow) axes"
    fig = pair_figs[-1]
    ax_c, ax_t = fig.axes
    fig.canvas.draw()

    n_ctrl, n_train = 2, 3
    n_max = max(n_ctrl, n_train)
    for ax, name in ((ax_c, "control"), (ax_t, "training")):
        ylim = ax.get_ylim()
        assert ylim == pytest.approx((n_max - 0.5, -0.5)), (
            f"{name} panel's y-range is not fixed to the larger fly count: {ylim}"
        )
    assert ax_c.get_ylim() == ax_t.get_ylim(), (
        "both panels sit in one GridSpec row (equal physical height); an "
        "equal data range is what makes their cell height equal"
    )

    # The load-bearing check: the ylim equality above is necessary but NOT
    # sufficient -- it holds by construction even when imshow's extent is
    # wrong. Measure each panel's ACTUAL rendered cell height instead.
    ctrl_cell = _measured_cell_height_in(ax_c, fig)
    train_cell = _measured_cell_height_in(ax_t, fig)
    assert ctrl_cell == pytest.approx(train_cell, rel=1e-6), (
        f"cell heights differ: control {ctrl_cell:.4f}in vs training {train_cell:.4f}in "
        f"-- the eye would read a size difference that is not in the data"
    )

    labels_c = [t.get_text() for t in ax_c.get_xticklabels()]
    labels_t = [t.get_text() for t in ax_t.get_xticklabels()]
    assert labels_c, "no odor columns rendered on the control panel"
    assert labels_c == labels_t, (
        f"panels show different columns: control={labels_c} training={labels_t}"
    )

    bar_figs = [
        f for f in map(plt.figure, plt.get_fignums())
        if len(f.axes) == 1 and not f.axes[0].images
    ]
    assert bar_figs, "no bars-only (matrix-free) figure was rendered"


# ---------------------------------------------------------------------------
# Regression (task 6): trained-odor remap must not merge presentations
# ---------------------------------------------------------------------------
#
# Same class of bug already fixed in score_summary.py's _should_number
# (commit d2a7b99): `trained_dup_odors` gated on EXACT EQUALITY between a
# REMAPPED odor display name and the bare `_trained_label`. A per-dataset
# odor_remap that appends text to the trained odor's display name (e.g.
# config_new.yaml's "Ethyl Butyrate" -> "Ethyl Butyrate (1%)" for
# EB-Training) makes that equality fail, so the odor is never numbered and
# both its testing presentations silently collapse into ONE column/row.
# Observed live: reaction_matrix_pair_EB-Training-24-1_*.png rendered 7
# columns with "Ethyl Butyrate (1%)" appearing once, instead of 8 columns
# with "Ethyl Butyrate (1%) 1" / "Ethyl Butyrate (1%) 2" kept separate.


def test_trained_odor_remap_still_yields_two_numbered_columns_matrix():
    """Matrix-column site (_build_during_matrix, v2 branch, around
    reaction_matrix_training_vs_control.py:634-637) must still split the
    trained odor's two presentations into separate columns when a dataset
    odor_remap renames its display label."""
    set_protocol("v2")
    set_dataset_odor_remap({"EB-Training": {"Ethyl Butyrate": "Ethyl Butyrate (1%)"}})
    try:
        df = _v2_rows(
            "EB-Training",
            [
                ("f1", "1", [
                    ("testing_1_Ethylbutyrate", 1),
                    ("testing_2_Hexanol", 0),
                    ("testing_8_Ethylbutyrate", 0),
                ]),
                ("f2", "2", [
                    ("testing_1_Ethylbutyrate", 0),
                    ("testing_2_Hexanol", 1),
                    ("testing_8_Ethylbutyrate", 1),
                ]),
            ],
        )
        mat, fly_pairs, cols, _ = module._build_during_matrix(
            df, "EB-Training", None, remap_from="EB-Training"
        )
    finally:
        set_dataset_odor_remap({})

    eb_cols = [c for c in cols if c.startswith("Ethyl Butyrate (1%)")]
    assert eb_cols == ["Ethyl Butyrate (1%) 1", "Ethyl Butyrate (1%) 2"], (
        f"EB presentations merged into {len(eb_cols)} column(s): {eb_cols} "
        f"(all columns: {cols})"
    )
    col1 = cols.index("Ethyl Butyrate (1%) 1")
    col2 = cols.index("Ethyl Butyrate (1%) 2")
    assert (mat[0, col1], mat[1, col1]) == (1.0, 0.0), "1st EB presentation values wrong"
    assert (mat[0, col2], mat[1, col2]) == (0.0, 1.0), "2nd EB presentation values wrong"


def test_trained_odor_remap_still_yields_two_numbered_bar_stats_rows(tmp_path):
    """Bar-stats site (_load_rates_from_binary_csv, v2 branch, around
    reaction_matrix_training_vs_control.py:365-368) must still split the
    trained odor's two presentations into separate 'odor' rows feeding
    plot_training_vs_control_bars.

    binary_reactions_*.csv's odor_sent column already carries the REMAPPED
    display name -- it is written upstream via _display_odor ->
    apply_dataset_odor_remap (see reaction_matrix_from_spreadsheet.py:538)
    -- so this fixture supplies the post-remap value directly, exactly as
    the real CSV would contain it.
    """
    set_protocol("v2")
    out_dir = tmp_path / "matrix"
    ds_dir = module.resolve_dataset_output_dir(out_dir, "EB-Training")
    ds_dir.mkdir(parents=True, exist_ok=True)

    # 2 flies; each sees the (already-remapped) trained odor TWICE.
    pd.DataFrame(
        {
            "fly": ["f1", "f1", "f1", "f2", "f2", "f2"],
            "fly_number": ["1", "1", "1", "2", "2", "2"],
            "trial_num": [1, 2, 8, 1, 2, 8],
            "odor_sent": [
                "Ethyl Butyrate (1%)", "Hexanol", "Ethyl Butyrate (1%)",
                "Ethyl Butyrate (1%)", "Hexanol", "Ethyl Butyrate (1%)",
            ],
            "during_hit": [1, 0, 0, 0, 1, 1],
        }
    ).to_csv(ds_dir / "binary_reactions_EB-Training_unordered.csv", index=False)

    stats = module._load_rates_from_binary_csv(
        out_dir, "EB-Training", "EB-Training", include_hexanol=True,
    )

    eb_rows = stats[stats["odor"].str.startswith("Ethyl Butyrate")]
    assert sorted(eb_rows["odor"]) == [
        "Ethyl Butyrate (1%) 1", "Ethyl Butyrate (1%) 2",
    ], (
        f"EB presentations merged into {len(eb_rows)} row(s): "
        f"{eb_rows['odor'].tolist()} (all odors: {stats['odor'].tolist()})"
    )
    assert set(eb_rows["num_trials"]) == {2}, (
        "num_trials inflated by merged presentations: "
        f"{eb_rows[['odor', 'num_trials']].to_dict('records')}"
    )
