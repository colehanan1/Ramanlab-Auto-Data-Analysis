"""Per-fly pre/post score matrix (scripts/analysis/per_fly_score_matrix.py)."""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import scripts.analysis.per_fly_score_matrix as pm

IAA = "Isoamyl Acetate"


@pytest.mark.parametrize("folder, key", [
    ("october_08_batch_2", (dt.date(2026, 10, 8), 2, 1)),
    ("october_08_batch_2_rig_3", (dt.date(2026, 10, 8), 2, 3)),
    ("october_12_batch_1_rig_2", (dt.date(2026, 10, 12), 1, 2)),
])
def test_folder_key_is_day_batch_rig(folder, key):
    assert pm.folder_key(folder) == key


def _paired():
    rows = []
    folders = ["october_12_batch_1_rig_2", "october_08_batch_2_rig_3", "october_08_batch_2",
               "october_12_batch_1", "october_08_batch_2_rig_2"]
    for f in folders:
        for n in (2, 1):
            for odor, pre, post in ((IAA, 0, 3), ("Hexanol", 1, 1), ("Citral", 2, 0)):
                rows.append({"dataset_canon": "IAA-Sensitivity-1-1", "fly": f, "fly_number": n,
                             "odor": odor, "score_pre": pre, "score_post": post})
    return pd.DataFrame(rows)


def test_rows_go_earliest_day_then_batch_then_rig_then_fly():
    m = pm.score_matrix(_paired(), trained=IAA)
    flies = list(m.index)
    assert flies[:6] == [("october_08_batch_2", 1), ("october_08_batch_2", 2),
                         ("october_08_batch_2_rig_2", 1), ("october_08_batch_2_rig_2", 2),
                         ("october_08_batch_2_rig_3", 1), ("october_08_batch_2_rig_3", 2)]
    assert flies[6:] == [("october_12_batch_1", 1), ("october_12_batch_1", 2),
                         ("october_12_batch_1_rig_2", 1), ("october_12_batch_1_rig_2", 2)]


def test_columns_are_pre_post_pairs_with_the_trained_odor_first():
    m = pm.score_matrix(_paired(), trained=IAA)
    assert list(m.columns[:2]) == [(IAA, "pre"), (IAA, "post")]
    assert list(m.columns[2:]) == [("Citral", "pre"), ("Citral", "post"),
                                   ("Hexanol", "pre"), ("Hexanol", "post")]
    assert m.loc[("october_08_batch_2", 1), (IAA, "post")] == 3


def test_group_boundaries_mark_rig_and_batch_changes():
    m = pm.score_matrix(_paired(), trained=IAA)
    rig, batch = pm.boundaries(list(m.index))
    # rig changes after rows 2 and 4 (0-based index of the first row of a new rig) ...
    assert rig == [2, 4, 8]
    # ... and a new day/batch starts at row 6
    assert batch == [6]


def test_figure_names_every_fly_and_shows_the_change_column():
    m = pm.score_matrix(_paired(), trained=IAA)
    fig = pm.build_figure(m, trained=IAA, title="IAA-Sensitivity-1-1")
    labels = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
    assert any("october_08_batch_2_rig_3" in s and "fly 2" in s for s in labels)
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "+3" in texts  # IAA change 0 -> 3
    plt.close(fig)


@pytest.mark.parametrize("dataset, phases, tag", [
    ("IAA-Sensitivity-1-1", "pre-exposure vs post-exposure", "(sensitized odor)"),
    ("Hex-HexTest-Training-32-1", "pre-test vs post-training", "(trained)"),
])
def test_title_and_trained_odor_header_follow_the_cohort(dataset, phases, tag):
    m = pm.score_matrix(_paired(), trained=IAA)
    fig = pm.build_figure(m, trained=IAA, title=dataset)
    assert fig._suptitle.get_text() == f"{dataset} — every fly's PER score, {phases}"
    assert f"{IAA}\n{tag}" in [t.get_text() for ax in fig.axes for t in ax.texts]
    plt.close(fig)
