"""Tests for the rig/batch breakdown driver.

These figures (``Results/Figures/<dataset>_rig_batch_breakdowns/``) split one
training/control pair three ways:

* ``tvc_rig_N`` / ``tvc_batch_N`` — training vs control *within* one rig/batch,
* ``train_rig_1_vs_rig_N`` / ``ctrl_rig_1_vs_rig_N`` — one arm across rigs,
* the same across batches.

The rig lives in the fly folder name as a ``_rig_N`` suffix, and rig 1 is
implicit (``july_20_batch_1`` is rig 1, ``july_20_batch_1_rig_2`` is rig 2).
Getting that default wrong silently merges every rig-1 fly into "no rig", which
is exactly the confound these figures exist to expose — hence the parsing tests
below carry the real folder names from the 3Oct-24-0.1 cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.rig_batch_breakdowns import (
    Group,
    batch_of,
    build_comparisons,
    group_score_samples,
    group_score_stats,
    mannwhitney_per_column,
    rig_of,
    select_group,
)

TRAIN = "3OCT-Training-24-0.1"
CTRL = "3OCT-Control-24-0.1"


# ---------------------------------------------------------------------------
# Fly-folder parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fly, expected",
    [
        ("july_20_batch_1", 1),           # rig 1 is implicit — no suffix
        ("july_26_batch_1", 1),
        ("july_20_batch_1_rig_2", 2),
        ("july_24_batch_2_rig_3", 3),
        ("july_21_batch_2_rig_3", 3),
    ],
)
def test_rig_of(fly: str, expected: int) -> None:
    assert rig_of(fly) == expected


@pytest.mark.parametrize(
    "fly, expected",
    [
        ("july_20_batch_1", 1),
        ("july_20_batch_2_rig_3", 2),
        ("july_24_batch_2_rig_2", 2),
        ("some_fly_without_a_batch", None),
    ],
)
def test_batch_of(fly: str, expected: int | None) -> None:
    assert batch_of(fly) == expected


def test_rig_of_does_not_read_the_batch_number() -> None:
    """``batch_2`` must not be mistaken for ``rig 2`` — both are bare digits in
    the same folder name, and confusing them would relabel every rig-1 fly."""
    assert rig_of("july_20_batch_2") == 1
    assert rig_of("july_20_batch_3_rig_2") == 2


# ---------------------------------------------------------------------------
# Comparison enumeration
# ---------------------------------------------------------------------------


def _cohort_frame() -> pd.DataFrame:
    """(dataset, fly) rows spanning rigs 1-3 and batches 1-2 on both arms."""
    rows = []
    for dataset in (TRAIN, CTRL):
        for fly in (
            "july_20_batch_1",          # rig 1, batch 1
            "july_20_batch_2",          # rig 1, batch 2
            "july_20_batch_1_rig_2",    # rig 2, batch 1
            "july_21_batch_2_rig_2",    # rig 2, batch 2
            "july_24_batch_1_rig_3",    # rig 3, batch 1
            "july_24_batch_2_rig_3",    # rig 3, batch 2
        ):
            rows.append({"dataset_canon": dataset, "fly": fly, "fly_number": "1"})
    return pd.DataFrame(rows)


def test_build_comparisons_covers_the_published_tag_set() -> None:
    comps = build_comparisons(_cohort_frame(), TRAIN, CTRL)
    tags = [c.tag for c in comps]
    assert tags == [
        "tvc_rig_1",
        "tvc_rig_2",
        "tvc_rig_3",
        "tvc_batch_1",
        "tvc_batch_2",
        "train_rig_1_vs_rig_2",
        "train_rig_1_vs_rig_3",
        "train_batch_1_vs_batch_2",
        "ctrl_rig_1_vs_rig_2",
        "ctrl_rig_1_vs_rig_3",
        "ctrl_batch_1_vs_batch_2",
    ]


def test_build_comparisons_titles_and_groups() -> None:
    by_tag = {c.tag: c for c in build_comparisons(_cohort_frame(), TRAIN, CTRL)}

    tvc = by_tag["tvc_rig_1"]
    assert tvc.title_suffix == "Rig 1: Training vs Control"
    assert (tvc.a.label, tvc.a.dataset) == ("Training", TRAIN)
    assert (tvc.b.label, tvc.b.dataset) == ("Control", CTRL)

    across = by_tag["train_rig_1_vs_rig_3"]
    assert across.title_suffix == "Training: Rig 1 vs Rig 3"
    assert (across.a.label, across.a.dataset) == ("Rig 1", TRAIN)
    assert (across.b.label, across.b.dataset) == ("Rig 3", TRAIN)

    ctrl_batch = by_tag["ctrl_batch_1_vs_batch_2"]
    assert ctrl_batch.title_suffix == "Control: Batch 1 vs Batch 2"
    assert ctrl_batch.a.dataset == CTRL and ctrl_batch.b.dataset == CTRL


def test_build_comparisons_skips_groups_missing_from_one_arm() -> None:
    """A rig the control arm never ran has no training-vs-control comparison —
    emitting one would draw a 4-fly training bar against an empty control."""
    df = _cohort_frame()
    df = df[~((df["dataset_canon"] == CTRL) & (df["fly"].str.contains("rig_3")))]
    tags = [c.tag for c in build_comparisons(df, TRAIN, CTRL)]
    assert "tvc_rig_3" not in tags
    assert "train_rig_1_vs_rig_3" in tags   # training still has rig 3
    assert "ctrl_rig_1_vs_rig_3" not in tags


# ---------------------------------------------------------------------------
# Batch overrides / starvation-time labels
# ---------------------------------------------------------------------------


@pytest.fixture()
def _clean_batch_state():
    """Overrides and labels are module state; leaking them would silently
    regroup every other test's flies."""
    from scripts.analysis import rig_batch_breakdowns as mod

    yield
    mod.set_batch_overrides({})
    mod.set_batch_labels({})


def test_batch_of_honours_overrides(_clean_batch_state) -> None:
    """august_09_batch_2_rig_2/3 were starved on the batch-1 schedule; the
    folder token says batch 2, so an explicit override must win."""
    from scripts.analysis.rig_batch_breakdowns import set_batch_overrides

    set_batch_overrides({
        "august_09_batch_2_rig_2": 1,
        "august_09_batch_2_rig_3": 1,
    })
    assert batch_of("august_09_batch_2_rig_2") == 1
    assert batch_of("august_09_batch_2_rig_3") == 1
    # Non-overridden flies still read the folder token.
    assert batch_of("august_09_batch_1_rig_2") == 1
    assert batch_of("july_13_batch_2") == 2


def test_batch_override_does_not_touch_the_rig(_clean_batch_state) -> None:
    from scripts.analysis.rig_batch_breakdowns import set_batch_overrides

    set_batch_overrides({"august_09_batch_2_rig_3": 1})
    assert rig_of("august_09_batch_2_rig_3") == 3


def test_select_group_moves_overridden_fly_between_batches(_clean_batch_state) -> None:
    from scripts.analysis.rig_batch_breakdowns import set_batch_overrides

    set_batch_overrides({"july_20_batch_2": 1})
    df = _cohort_frame()
    batch1 = select_group(df, Group("Batch 1", TRAIN, "batch", 1))
    batch2 = select_group(df, Group("Batch 2", TRAIN, "batch", 2))
    assert "july_20_batch_2" in set(batch1["fly"])
    assert "july_20_batch_2" not in set(batch2["fly"])


def test_build_comparisons_uses_batch_labels(_clean_batch_state) -> None:
    """Labels rename the displayed group (titles/legends) only — tags, and
    therefore the output filenames, keep the batch numbers."""
    from scripts.analysis.rig_batch_breakdowns import set_batch_labels

    set_batch_labels({1: "Starved 24±3 h", 2: "Starved 27±3 h"})
    by_tag = {c.tag: c for c in build_comparisons(_cohort_frame(), TRAIN, CTRL)}

    tvc = by_tag["tvc_batch_1"]
    assert tvc.title_suffix == "Starved 24±3 h: Training vs Control"
    assert (tvc.a.label, tvc.b.label) == ("Training", "Control")

    across = by_tag["train_batch_1_vs_batch_2"]
    assert across.title_suffix == "Training: Starved 24±3 h vs Starved 27±3 h"
    assert (across.a.label, across.b.label) == ("Starved 24±3 h", "Starved 27±3 h")

    # Rig comparisons keep their plain names.
    assert by_tag["tvc_rig_1"].title_suffix == "Rig 1: Training vs Control"


def test_parse_batch_overrides() -> None:
    from scripts.analysis.rig_batch_breakdowns import _parse_batch_overrides

    assert _parse_batch_overrides(
        ["august_09_batch_2_rig_2=1", "august_09_batch_2_rig_3=1"]
    ) == {"august_09_batch_2_rig_2": 1, "august_09_batch_2_rig_3": 1}
    assert _parse_batch_overrides([]) == {}
    with pytest.raises(ValueError):
        _parse_batch_overrides(["missing_the_number"])
    with pytest.raises(ValueError):
        _parse_batch_overrides(["fly=not_a_number"])


def test_parse_batch_labels() -> None:
    from scripts.analysis.rig_batch_breakdowns import _parse_batch_labels

    assert _parse_batch_labels(["1=Starved 24±3 h", "2=Starved 27±3 h"]) == {
        1: "Starved 24±3 h",
        2: "Starved 27±3 h",
    }
    with pytest.raises(ValueError):
        _parse_batch_labels(["one=label"])


# ---------------------------------------------------------------------------
# Group selection
# ---------------------------------------------------------------------------


def test_select_group_picks_one_rig_of_one_dataset() -> None:
    df = _cohort_frame()
    sub = select_group(df, Group(label="Rig 2", dataset=TRAIN, kind="rig", value=2))
    assert set(sub["fly"]) == {"july_20_batch_1_rig_2", "july_21_batch_2_rig_2"}
    assert set(sub["dataset_canon"]) == {TRAIN}


def test_select_group_rig_1_keeps_the_unsuffixed_flies() -> None:
    df = _cohort_frame()
    sub = select_group(df, Group(label="Rig 1", dataset=CTRL, kind="rig", value=1))
    assert set(sub["fly"]) == {"july_20_batch_1", "july_20_batch_2"}


def test_select_group_whole_dataset_when_kind_is_none() -> None:
    df = _cohort_frame()
    sub = select_group(df, Group(label="Training", dataset=TRAIN, kind=None, value=None))
    assert set(sub["dataset_canon"]) == {TRAIN}
    assert len(sub) == 6


# ---------------------------------------------------------------------------
# Combined-rig comparisons
# ---------------------------------------------------------------------------


def test_select_group_accepts_multiple_rig_values() -> None:
    """A tuple value pools the levels — rigs 1 and 2 become one group."""
    df = _cohort_frame()
    sub = select_group(df, Group("Rig 1+2", TRAIN, "rig", (1, 2)))
    assert set(sub["fly"]) == {
        "july_20_batch_1",
        "july_20_batch_2",
        "july_20_batch_1_rig_2",
        "july_21_batch_2_rig_2",
    }
    assert set(sub["dataset_canon"]) == {TRAIN}


def test_parse_rig_compares() -> None:
    from scripts.analysis.rig_batch_breakdowns import _parse_rig_compares

    assert _parse_rig_compares(["train:1,2:3"]) == [("train", (1, 2), (3,))]
    assert _parse_rig_compares(["ctrl:1:2,3"]) == [("ctrl", (1,), (2, 3))]
    assert _parse_rig_compares([]) == []
    with pytest.raises(ValueError):
        _parse_rig_compares(["training:1,2:3"])   # arm must be train|ctrl
    with pytest.raises(ValueError):
        _parse_rig_compares(["train:1,2"])        # both sides required
    with pytest.raises(ValueError):
        _parse_rig_compares(["train:one:3"])


def test_rig_compare_comparison_shape() -> None:
    from scripts.analysis.rig_batch_breakdowns import _rig_compare_comparison

    comp = _rig_compare_comparison("train", (1, 2), (3,), TRAIN, CTRL)
    assert comp.tag == "train_rig_1_2_vs_rig_3"
    assert comp.title_suffix == "Training: Rig 1+2 vs Rig 3"
    assert (comp.a.label, comp.a.dataset, comp.a.kind, comp.a.value) == (
        "Rig 1+2", TRAIN, "rig", (1, 2)
    )
    assert (comp.b.label, comp.b.dataset, comp.b.kind, comp.b.value) == (
        "Rig 3", TRAIN, "rig", (3,)
    )


def test_main_rig_compare_with_only_writes_just_that_comparison(tmp_path: Path) -> None:
    """--only lets a new comparison be added to a published folder without
    regenerating (and possibly changing) every other figure in it."""
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
            "--rig-compare", "train:1:2",
            "--only", "train_rig_1_vs_rig_2",
        ]
    )

    names = {p.name for p in out_dir.iterdir()}
    stub = "30_latency_2.150s"
    assert names == {
        f"reaction_matrix_train_rig_1_vs_rig_2_{stub}.png",
        f"reaction_matrix_pair_train_rig_1_vs_rig_2_{stub}.png",
        f"reaction_matrix_train_rig_1_vs_rig_2_{stub}.json",
        "mean_score_train_rig_1_vs_rig_2.png",
        "mean_score_pair_train_rig_1_vs_rig_2.png",
        "mean_score_train_rig_1_vs_rig_2.json",
    }


def test_main_rig_compare_pools_the_combined_side(tmp_path: Path) -> None:
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
            "--rig-compare", "train:1,2:2",
            "--only", "train_rig_1_2_vs_rig_2",
        ]
    )
    sidecar = json.loads(
        (out_dir / "mean_score_train_rig_1_2_vs_rig_2.json").read_text()
    )
    assert sidecar["title_suffix"] == "Training: Rig 1+2 vs Rig 2"
    # rigs 1+2 = all four folders x 2 fly_numbers; rig 2 alone = 2 folders x 2
    assert sidecar["group_a"]["n_flies"] == 8
    assert sidecar["group_b"]["n_flies"] == 4


def test_parse_tvc_rigs() -> None:
    from scripts.analysis.rig_batch_breakdowns import _parse_tvc_rigs

    assert _parse_tvc_rigs(["1,2"]) == [(1, 2)]
    assert _parse_tvc_rigs(["3"]) == [(3,)]
    assert _parse_tvc_rigs([]) == []
    with pytest.raises(ValueError):
        _parse_tvc_rigs(["one,two"])
    with pytest.raises(ValueError):
        _parse_tvc_rigs([""])


def test_tvc_rig_comparison_shape() -> None:
    from scripts.analysis.rig_batch_breakdowns import _tvc_rig_comparison

    comp = _tvc_rig_comparison((1, 2), TRAIN, CTRL)
    assert comp.tag == "tvc_rig_1_2"
    assert comp.title_suffix == "Rig 1+2: Training vs Control"
    assert (comp.a.label, comp.a.dataset, comp.a.kind, comp.a.value) == (
        "Training", TRAIN, "rig", (1, 2)
    )
    assert (comp.b.label, comp.b.dataset, comp.b.kind, comp.b.value) == (
        "Control", CTRL, "rig", (1, 2)
    )


def test_main_tvc_rig_pools_both_arms(tmp_path: Path) -> None:
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
            "--tvc-rig", "1,2",
            "--only", "tvc_rig_1_2",
        ]
    )
    names = {p.name for p in out_dir.iterdir()}
    assert "mean_score_tvc_rig_1_2.json" in names
    assert len(names) == 6   # exactly the one comparison's file family

    sidecar = json.loads((out_dir / "mean_score_tvc_rig_1_2.json").read_text())
    assert sidecar["title_suffix"] == "Rig 1+2: Training vs Control"
    # rigs 1+2 = all four folders x 2 fly_numbers, on each arm
    assert sidecar["group_a"]["n_flies"] == 8
    assert sidecar["group_b"]["n_flies"] == 8


def test_main_restrict_batch_drops_other_batches(tmp_path: Path, _clean_batch_state) -> None:
    """--restrict-batch narrows BOTH arms to one starvation group before any
    comparison is built: batch-2 flies vanish, so no cross-batch figures and
    the rig figures count only batch-1 flies."""
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
            "--restrict-batch", "1",
        ]
    )

    names = {p.name for p in out_dir.iterdir()}
    assert "mean_score_tvc_batch_1.json" in names
    assert "mean_score_tvc_batch_2.json" not in names
    assert "mean_score_train_batch_1_vs_batch_2.json" not in names

    # each rig level now holds exactly one batch-1 folder x two fly_numbers
    sidecar = json.loads((out_dir / "mean_score_tvc_rig_1.json").read_text())
    assert sidecar["group_a"]["n_flies"] == 2
    assert sidecar["group_b"]["n_flies"] == 2


def test_main_restrict_batch_honours_overrides(tmp_path: Path, _clean_batch_state) -> None:
    """A fly overridden into batch 1 must survive --restrict-batch 1."""
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
            "--restrict-batch", "1",
            "--batch-override", "july_20_batch_2=1",
        ]
    )
    sidecar = json.loads((out_dir / "mean_score_tvc_rig_1.json").read_text())
    # rig 1 = july_20_batch_1 + the overridden july_20_batch_2, x2 fly_numbers
    assert sidecar["group_a"]["n_flies"] == 4
    assert sidecar["group_b"]["n_flies"] == 4


# ---------------------------------------------------------------------------
# Score statistics
# ---------------------------------------------------------------------------


def _score_frame() -> pd.DataFrame:
    """Two flies x two odors, one odor seen twice by fly 1."""
    return pd.DataFrame(
        [
            {"fly": "a", "fly_number": "1", "odor_col": "Hexanol (0.1%)", "score": 4.0},
            {"fly": "a", "fly_number": "1", "odor_col": "Hexanol (0.1%)", "score": 2.0},
            {"fly": "a", "fly_number": "1", "odor_col": "Citral (1%)", "score": 0.0},
            {"fly": "b", "fly_number": "1", "odor_col": "Hexanol (0.1%)", "score": 0.0},
            {"fly": "b", "fly_number": "1", "odor_col": "Citral (1%)", "score": 2.0},
        ]
    )


def test_group_score_stats_averages_within_fly_first() -> None:
    """Fly ``a`` saw hexanol twice (4 and 2). It must count once, at 3.0 — not
    twice, which would both bias the mean and inflate n."""
    stats = group_score_stats(_score_frame(), ["Hexanol (0.1%)", "Citral (1%)"])
    hexanol = stats.set_index("odor").loc["Hexanol (0.1%)"]
    assert hexanol["n_flies"] == 2
    assert hexanol["mean_score"] == pytest.approx(1.5)   # mean of 3.0 and 0.0
    assert hexanol["sem_score"] == pytest.approx(1.5)


def test_group_score_stats_reports_absent_odors_as_empty() -> None:
    stats = group_score_stats(_score_frame(), ["Hexanol (0.1%)", "Linalool (1%)"])
    lin = stats.set_index("odor").loc["Linalool (1%)"]
    assert lin["n_flies"] == 0
    assert np.isnan(lin["mean_score"])


def test_group_score_stats_preserves_column_order() -> None:
    columns = ["Citral (1%)", "Hexanol (0.1%)"]
    assert list(group_score_stats(_score_frame(), columns)["odor"]) == columns


def test_mannwhitney_per_column_matches_scipy() -> None:
    from scipy.stats import mannwhitneyu

    a = {"Hexanol (0.1%)": np.array([5.0, 4.0, 4.0, 3.0])}
    b = {"Hexanol (0.1%)": np.array([0.0, 0.0, 1.0, 0.0])}
    p = mannwhitney_per_column(a, b, ["Hexanol (0.1%)"])["Hexanol (0.1%)"]
    expected = mannwhitneyu(
        a["Hexanol (0.1%)"], b["Hexanol (0.1%)"], alternative="two-sided"
    ).pvalue
    assert p == pytest.approx(expected)


def test_mannwhitney_per_column_is_nan_when_a_side_is_empty() -> None:
    a = {"Citral (1%)": np.array([1.0, 2.0])}
    b = {"Citral (1%)": np.array([])}
    assert np.isnan(mannwhitney_per_column(a, b, ["Citral (1%)"])["Citral (1%)"])


def test_group_score_samples_are_one_value_per_fly() -> None:
    samples = group_score_samples(_score_frame(), ["Hexanol (0.1%)"])
    assert sorted(samples["Hexanol (0.1%)"].tolist()) == [0.0, 3.0]


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

ODORS = [
    "3-octanol",
    "3-octanol",
    "benzaldehyde",
    "citral",
    "ethylbutyrate",
    "hexanol",
    "linalool",
]


def _write_predictions_csv(path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    flies = [
        "july_20_batch_1",
        "july_20_batch_1_rig_2",
        "july_20_batch_2",
        "july_20_batch_2_rig_2",
    ]
    for dataset in ("3Oct-Training-24-0.1", "3Oct-Control-24-0.1"):
        for fly in flies:
            for n in (1, 2):
                for i, odor in enumerate(ODORS, start=1):
                    score = float(rng.integers(-1, 6))
                    rows.append(
                        {
                            "dataset": dataset,
                            "fly": fly,
                            "fly_number": n,
                            "trial_label": f"testing_{i}_{odor}",
                            "prediction": int(score >= 2),
                            "score": score,
                            "trial_type": "testing",
                            "fly_type": "wt",
                            "_non_reactive": False,
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_main_writes_the_four_figure_families_and_sidecars(tmp_path: Path) -> None:
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"

    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
        ]
    )

    names = {p.name for p in out_dir.iterdir()}
    stub = "30_latency_2.150s"
    for tag in ("tvc_rig_1", "tvc_batch_2", "train_rig_1_vs_rig_2", "ctrl_rig_1_vs_rig_2"):
        assert f"reaction_matrix_{tag}_{stub}.png" in names
        assert f"reaction_matrix_pair_{tag}_{stub}.png" in names
        assert f"reaction_matrix_{tag}_{stub}.json" in names
        assert f"mean_score_{tag}.png" in names
        assert f"mean_score_pair_{tag}.png" in names
        assert f"mean_score_{tag}.json" in names


def test_sidecar_schema_matches_the_published_figures(tmp_path: Path) -> None:
    """The published sidecars carry exactly these keys; downstream notes quote
    ``group_a.n_flies`` and the per-odor p-values straight out of them."""
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
        ]
    )

    reaction = json.loads(
        (out_dir / "reaction_matrix_tvc_rig_1_30_latency_2.150s.json").read_text()
    )
    assert reaction["tag"] == "tvc_rig_1"
    assert reaction["title_suffix"] == "Rig 1: Training vs Control"
    assert set(reaction) >= {"tag", "title_suffix", "group_a", "group_b",
                             "odor_columns", "fisher_p"}
    assert set(reaction["group_a"]) >= {"label", "dataset", "n_flies"}
    # rig 1 = the two unsuffixed folders x two fly_numbers each
    assert reaction["group_a"]["n_flies"] == 4
    assert set(reaction["fisher_p"]) == set(reaction["odor_columns"])

    score = json.loads((out_dir / "mean_score_tvc_rig_1.json").read_text())
    assert set(score) >= {"tag", "title_suffix", "group_a", "group_b",
                          "odor_columns", "mannwhitney_p"}
    assert set(score["mannwhitney_p"]) == set(score["odor_columns"])


def test_main_applies_batch_overrides_and_labels(tmp_path: Path, _clean_batch_state) -> None:
    """With every batch-2 folder overridden into batch 1 there is nothing left
    to compare across batches, and the tvc_batch_1 sidecar counts all flies
    under the starvation-time label."""
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
            "--batch-override", "july_20_batch_2=1",
            "--batch-override", "july_20_batch_2_rig_2=1",
            "--batch-label", "1=Starved 24±3 h",
        ]
    )

    names = {p.name for p in out_dir.iterdir()}
    assert "mean_score_tvc_batch_1.json" in names
    assert "mean_score_tvc_batch_2.json" not in names
    assert "mean_score_train_batch_1_vs_batch_2.json" not in names

    sidecar = json.loads((out_dir / "mean_score_tvc_batch_1.json").read_text())
    assert sidecar["title_suffix"] == "Starved 24±3 h: Training vs Control"
    # all 4 folders x 2 fly_numbers now count as batch 1
    assert sidecar["group_a"]["n_flies"] == 8
    assert sidecar["group_b"]["n_flies"] == 8


def _write_mixed_cohort_csv(path: Path) -> None:
    """Predictions CSV with two fly types and two recording months."""
    rng = np.random.default_rng(1)
    rows = []
    flies = [
        ("april_16_batch_1", "GR5a-GCaMP8"),
        ("april_20_batch_2_rig_2", "GR5a-Old"),
        ("august_05_batch_1_rig_2", "GR5a-Old"),
        ("august_06_batch_2_rig_3", "GR5a-Old"),
    ]
    for dataset in ("3Oct-Training-24-0.1", "3Oct-Control-24-0.1"):
        for fly, fly_type in flies:
            for n in (1, 2):
                for i, odor in enumerate(ODORS, start=1):
                    score = float(rng.integers(-1, 6))
                    rows.append(
                        {
                            "dataset": dataset,
                            "fly": fly,
                            "fly_number": n,
                            "trial_label": f"testing_{i}_{odor}",
                            "prediction": int(score >= 2),
                            "score": score,
                            "trial_type": "testing",
                            "fly_type": fly_type,
                            "_non_reactive": False,
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_main_fly_type_filter_keeps_only_that_type(tmp_path: Path) -> None:
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_mixed_cohort_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
            "--fly-type", "gr5a-old",   # case-insensitive
            "--only", "tvc_rig_2",
        ]
    )
    sidecar = json.loads((out_dir / "mean_score_tvc_rig_2.json").read_text())
    # rig 2 folders: april_20 (GR5a-Old, kept) + august_05 (kept); the GCaMP8
    # fly is rig 1 anyway — the count proves only Old flies remain: 2 x 2.
    assert sidecar["group_a"]["n_flies"] == 4
    assert sidecar["group_b"]["n_flies"] == 4


def test_main_fly_prefix_filter_keeps_only_matching_months(tmp_path: Path) -> None:
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_mixed_cohort_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
            "--fly-prefix", "august",
            "--only", "tvc_rig_2",
        ]
    )
    sidecar = json.loads((out_dir / "mean_score_tvc_rig_2.json").read_text())
    # only august_05 is rig 2 once april flies are dropped
    assert sidecar["group_a"]["n_flies"] == 2
    assert sidecar["group_b"]["n_flies"] == 2


def test_trained_odor_is_numbered_across_both_panels(tmp_path: Path) -> None:
    """3-Octanol is presented twice, so it must appear as two columns ("… 1",
    "… 2") in both the reaction and the score figures — a single merged column
    would double every fly's n and hide the exposure-order effect."""
    from scripts.analysis.rig_batch_breakdowns import main

    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(csv_path)
    out_dir = tmp_path / "figs"
    main(
        [
            "--csv-path", str(csv_path),
            "--out-dir", str(out_dir),
            "--train-dataset", "3Oct-Training-24-0.1",
            "--control-dataset", "3Oct-Control-24-0.1",
        ]
    )
    for name in ("reaction_matrix_tvc_rig_1_30_latency_2.150s.json",
                 "mean_score_tvc_rig_1.json"):
        cols = json.loads((out_dir / name).read_text())["odor_columns"]
        numbered = [c for c in cols if c.casefold().startswith("3-octanol")]
        assert len(numbered) == 2, f"{name}: {cols}"
