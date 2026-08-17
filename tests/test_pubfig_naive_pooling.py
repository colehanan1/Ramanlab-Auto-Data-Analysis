"""The naive bar pools its trials; the correction family is selectable.

Two corrections to the naive-vs-trained-vs-control figure:

1. The naive panel presents each odor twice. Averaging each fly's two trials
   and then averaging those 20 fly-means is an average of averages -- with an
   unbalanced fly it silently reweights the group. The naive bar is the grand
   mean over all its trials: add up all 40 trial scores, divide by 40.

2. ``score_summary``'s trained-vs-control bars star the *uncorrected*
   Mann-Whitney p, so 3-octanol at p=0.034 reads ``*`` there while Holm over
   the three pairwise tests turned it into p=0.067 and no star here -- same
   data, same test, two conventions. ``--correction`` picks one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.pubfig_naive_vs_trained import (  # noqa: E402
    Comparison,
    cohort_scores,
    load_groups,
    render_comparison,
    score_stats,
)

NAIVE = "RandomPanel-24-0.1"
TRAIN = "3OCT-Training-24-0.1"
CONTROL = "3OCT-Control-24-0.1"


def _frame(naive_scores):
    """naive_scores: {fly: [score_presentation_1, score_presentation_2, ...]}."""
    rows = []
    for fly, scores in naive_scores.items():
        for occurrence, score in enumerate(scores, start=1):
            rows.append((NAIVE, fly, occurrence, score))
    for dataset, score in ((TRAIN, 4.0), (CONTROL, 1.0)):
        for i in range(4):
            rows.append((dataset, f"fly_{i}", 1, score))
    return pd.DataFrame(
        [
            {
                "dataset_canon": ds,
                "fly": fly,
                "fly_number": 1,
                "fly_type": "GR5a-Old",
                "odor_display": "3-Octanol" if ds == NAIVE else "3-Octanol (0.1%)",
                "occurrence": occ,
                "score": score,
            }
            for ds, fly, occ, score in rows
        ]
    )


def _comparison(**kw):
    return Comparison(
        odor="3-Octanol", concentration="0.1%",
        naive_dataset=NAIVE, train_dataset=TRAIN, control_dataset=CONTROL,
        **kw,
    )


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def test_pooled_scores_are_one_row_per_trial():
    df = _frame({"a": [0.0, 2.0], "b": [1.0, 3.0]})
    pooled = cohort_scores(df, NAIVE, "3-Octanol", pool_trials=True)
    assert len(pooled) == 4
    assert sorted(pooled["score"]) == [0.0, 1.0, 2.0, 3.0]


def test_the_pooled_mean_is_the_grand_mean_not_a_mean_of_fly_means():
    """The unbalanced fly is the whole point: 'b' contributed one trial, so
    per-fly averaging would give it the weight of a fly that gave two."""
    df = _frame({"a": [0.0, 0.0], "b": [3.0]})
    pooled = cohort_scores(df, NAIVE, "3-Octanol", pool_trials=True)["score"]
    assert pooled.mean() == 1.0                       # (0 + 0 + 3) / 3
    per_fly = cohort_scores(df, NAIVE, "3-Octanol", presentation=None)["score"]
    assert per_fly.mean() == 1.5                      # mean(mean(0,0), mean(3))
    assert pooled.mean() != per_fly.mean()


def test_pooling_keeps_every_presentation():
    df = _frame({"a": [0.0, 4.0]})
    first = cohort_scores(df, NAIVE, "3-Octanol", presentation=1)["score"].tolist()
    pooled = sorted(cohort_scores(df, NAIVE, "3-Octanol", pool_trials=True)["score"])
    assert first == [0.0]
    assert pooled == [0.0, 4.0]


def test_only_the_naive_arm_pools():
    df = _frame({"a": [0.0, 2.0], "b": [1.0, 3.0]})
    groups = load_groups(df, _comparison(naive_pool_trials=True))
    assert groups["Naive"].size == 4      # 2 flies x 2 presentations
    assert groups["Trained"].size == 4    # 4 flies, one trial each
    assert groups["Control"].size == 4


def test_pooling_is_off_for_the_hand_written_comparisons():
    """The published 3oct/eb figures must not silently change."""
    df = _frame({"a": [0.0, 2.0], "b": [1.0, 3.0]})
    groups = load_groups(df, _comparison())
    assert groups["Naive"].size == 2


def test_the_pooled_bar_is_labelled_in_trials_not_flies():
    df = _frame({"a": [0.0, 2.0], "b": [1.0, 3.0]})
    comparison = _comparison(naive_pool_trials=True)
    fig = render_comparison(comparison, load_groups(df, comparison), footnote=False)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels[0] == "Naive\n(n=4 trials)"
    assert labels[1] == "Trained\n(n=4)"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Correction family
# ---------------------------------------------------------------------------


def _groups():
    rng = np.random.default_rng(0)
    return {
        "Naive": np.repeat([0.0, 1.0], 10),
        "Trained": np.array([4.0, 5.0, 4.0, 3.0, 5.0, 4.0, 4.0, 3.0, 5.0]),
        "Control": np.array([1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0]),
    }


def test_holm_is_the_default_and_inflates_the_p_values():
    stats = score_stats(_groups())
    pair = stats["pairwise"][("Trained", "Control")]
    assert pair["p_adj"] > pair["p_raw"]


def test_no_correction_reports_the_raw_p_value():
    """Matching score_summary, whose bars star the uncorrected p."""
    stats = score_stats(_groups(), correction="none")
    for pair in stats["pairwise"].values():
        assert pair["p_adj"] == pair["p_raw"]


def test_an_unknown_correction_is_rejected():
    try:
        score_stats(_groups(), correction="bonferoni")
    except (ValueError, SystemExit) as exc:
        assert "bonferoni" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a typo'd correction must not be silently ignored")


def test_the_correction_reaches_the_drawn_figure():
    comparison = _comparison()
    corrected = render_comparison(comparison, _groups(), footnote=False)
    raw = render_comparison(comparison, _groups(), footnote=False, correction="none")
    stars = lambda fig: sorted(  # noqa: E731
        t.get_text() for ax in fig.axes for t in ax.texts if set(t.get_text()) == {"*"}
    )
    assert len(stars(raw)) >= len(stars(corrected))
    plt.close("all")
