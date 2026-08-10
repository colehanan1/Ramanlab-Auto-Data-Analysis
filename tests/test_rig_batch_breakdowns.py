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
