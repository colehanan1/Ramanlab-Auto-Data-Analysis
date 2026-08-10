"""Tests for the mean-ordinal-score companion to the split reaction-rate figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.analysis import odor_bar_palette  # noqa: E402
from scripts.analysis.remake_score_bars_split import (  # noqa: E402
    AIR_COLOR,
    DEFAULT_EXCLUDE,
    attach_odors,
    load_scores,
    remake_score_bars,
    render_score_bars_figure,
    score_stats,
)

REAL_BINARY_CSV = Path(
    "/home/ramanlab/Documents/cole/Results/Opto-Fly-Figures/"
    "Matrix-PER-Reactions-Model/AIR-Training/binary_reactions_AIR-Training_unordered.csv"
)
REAL_PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/model_predictions.csv"
)

# The AIR-Training testing panel, in presentation order. Hexanol and AIR repeat;
# every other odor is presented once.
SCHEDULE = [
    (1, "Hexanol"),
    (2, "AIR"),
    (3, "Hexanol"),
    (4, "AIR"),
    (5, "AIR"),
    (6, "Apple Cider Vinegar"),
    (7, "Benzaldehyde"),
    (8, "Ethyl Butyrate"),
]


def _synthetic_binary() -> pd.DataFrame:
    rows = []
    for fly, fly_number in (("batch_a", 1), ("batch_a", 2)):
        for trial_num, odor in SCHEDULE:
            rows.append(
                {
                    "dataset": "AIR-Training",
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial_num": trial_num,
                    "odor_sent": odor,
                    "during_hit": 0,
                    "after_hit": 0,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_predictions() -> pd.DataFrame:
    """Scores chosen so every bar has a distinct, hand-checkable mean."""
    scores = {
        ("batch_a", 1): {1: 4, 2: 0, 3: 2, 4: -1, 5: 0, 6: 3, 7: 5, 8: 2},
        ("batch_a", 2): {1: 2, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 5, 8: 0},
    }
    rows = []
    for (fly, fly_number), per_trial in scores.items():
        for trial_num, odor in SCHEDULE:
            rows.append(
                {
                    "dataset": "AIR-Training",
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial_label": (
                        f"testing_{trial_num}_fly1_distances_fly1_angle_"
                        "distance_rms_envelope"
                    ),
                    "prediction": per_trial[trial_num],
                    "score": per_trial[trial_num],
                    "trial_type": "testing",
                }
            )
    return pd.DataFrame(rows)


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    binary_csv = tmp_path / "binary_reactions_AIR-Training_unordered.csv"
    _synthetic_binary().to_csv(binary_csv, index=False)
    predictions_csv = tmp_path / "model_predictions.csv"
    _synthetic_predictions().to_csv(predictions_csv, index=False)
    return binary_csv, predictions_csv


def _stats(**kwargs) -> pd.DataFrame:
    scores = attach_odors(
        load_scores(_synthetic_predictions(), dataset="AIR-Training"),
        _synthetic_binary(),
    )
    return score_stats(scores, trained_label="AIR", **kwargs)


# ---------------------------------------------------------------------------
# Odor attachment
# ---------------------------------------------------------------------------


def test_scores_take_their_odor_from_the_binary_reactions_export() -> None:
    """Prediction trial labels carry no odor token, so the odor comes from the join."""
    scores = load_scores(_synthetic_predictions(), dataset="AIR-Training")
    assert "odor" not in scores.columns

    joined = attach_odors(scores, _synthetic_binary())

    assert joined["odor"].notna().all()
    trial_two = joined.loc[joined["trial_num"] == 2, "odor"].unique().tolist()
    assert trial_two == ["AIR"]


def test_unmatched_scores_are_an_error_not_a_silent_drop() -> None:
    binary = _synthetic_binary()
    binary = binary.loc[binary["trial_num"] != 5]

    with pytest.raises(RuntimeError, match="odor"):
        attach_odors(load_scores(_synthetic_predictions(), dataset="AIR-Training"), binary)


# ---------------------------------------------------------------------------
# Bar selection
# ---------------------------------------------------------------------------


def test_benzaldehyde_is_excluded_by_default() -> None:
    stats = _stats(first_presentation_only=False)
    assert "Benzaldehyde" not in set(stats["odor"])
    assert DEFAULT_EXCLUDE == frozenset({"Benzaldehyde"})


def test_all_presentations_are_kept_when_the_flag_is_off() -> None:
    stats = _stats(first_presentation_only=False)
    assert stats["odor"].tolist() == [
        "Hexanol",
        "AIR",
        "Hexanol",
        "AIR",
        "AIR",
        "Apple Cider Vinegar",
        "Ethyl Butyrate",
    ]


def test_first_presentation_only_keeps_the_earliest_of_each_repeated_odor() -> None:
    stats = _stats(first_presentation_only=True)

    assert stats["odor"].tolist() == [
        "Hexanol",
        "AIR",
        "Apple Cider Vinegar",
        "Ethyl Butyrate",
    ]
    # The kept Hexanol/AIR bars are trials 1 and 2, not the later repeats.
    assert stats["trial_num"].tolist() == [1, 2, 6, 8]


def test_first_presentation_only_uses_the_first_trials_scores() -> None:
    """The dropped repeats must not leak into the surviving bar's mean."""
    stats = _stats(first_presentation_only=True)
    hexanol = stats.loc[stats["odor"] == "Hexanol"].iloc[0]

    # Trial 1 scores are 4 and 2 -> mean 3.0; trial 3 (2 and 0) is dropped.
    assert hexanol["mean_score"] == pytest.approx(3.0)
    assert hexanol["n_flies"] == 2
    assert hexanol["sem_score"] == pytest.approx(1.0)


def test_trained_odor_is_flagged() -> None:
    stats = _stats(first_presentation_only=True)
    trained = stats.loc[stats["is_trained"]]
    assert trained["odor"].tolist() == ["AIR"]


def test_air_mean_can_be_negative() -> None:
    """Scores span -1..5, so the axis and stats must survive a below-zero bar."""
    stats = _stats(first_presentation_only=False)
    air_first = stats.loc[stats["trial_num"] == 4].iloc[0]
    assert air_first["mean_score"] == pytest.approx(0.0)
    air_neg = stats.loc[stats["trial_num"] == 2].iloc[0]
    assert air_neg["mean_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def test_figure_axis_labels_and_range() -> None:
    fig = render_score_bars_figure(_stats(first_presentation_only=True))
    try:
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Mean PER Score"
        assert ax.get_xlabel() == ""
        assert ax.get_ylim() == (-1.0, 5.0)
    finally:
        plt.close(fig)


def test_cohort_n_is_stated_once_in_the_legend_not_on_every_bar() -> None:
    fig = render_score_bars_figure(_stats(first_presentation_only=True))
    try:
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Training (n=2)"]
        assert all("n=" not in t.get_text() for t in ax.texts), (
            "the n belongs in the legend, not repeated above every bar"
        )
    finally:
        plt.close(fig)


def test_figure_capitalises_the_trained_bar() -> None:
    stats = _stats(first_presentation_only=True)
    fig = render_score_bars_figure(stats)
    try:
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["Hexanol", "AIR", "Apple Cider Vinegar", "Ethyl Butyrate"]
    finally:
        plt.close(fig)


def test_bars_use_the_shared_odor_palette() -> None:
    """Each odorant gets the colour it has in the ACV-Training score figure."""
    stats = _stats(first_presentation_only=True)
    fig = render_score_bars_figure(stats)
    try:
        ax = fig.axes[0]
        colours = [bar.get_facecolor() for bar in ax.patches]
        expected = [
            odor_bar_palette.HEX_COLOR,
            AIR_COLOR,
            odor_bar_palette.ACV_COLOR,
            odor_bar_palette.PINK,
        ]
        assert colours == [matplotlib.colors.to_rgba(c) for c in expected]
    finally:
        plt.close(fig)


def test_air_keeps_its_blue_and_is_not_repainted_by_the_palette() -> None:
    """AIR has no palette entry; it must stay blue rather than fall through."""
    assert odor_bar_palette.odor_color("AIR") is None

    stats = _stats(first_presentation_only=True)
    fig = render_score_bars_figure(stats)
    try:
        ax = fig.axes[0]
        trained_idx = int(np.flatnonzero(stats["is_trained"].to_numpy())[0])
        assert ax.patches[trained_idx].get_facecolor() == matplotlib.colors.to_rgba(
            AIR_COLOR
        )
        tick = ax.get_xticklabels()[trained_idx]
        assert matplotlib.colors.to_rgba(tick.get_color()) == matplotlib.colors.to_rgba(
            AIR_COLOR
        )
    finally:
        plt.close(fig)


def test_untrained_odor_ticks_stay_black() -> None:
    """The palette greens and yellows are unreadable as text, so only AIR is coloured."""
    stats = _stats(first_presentation_only=True)
    fig = render_score_bars_figure(stats)
    try:
        ax = fig.axes[0]
        black = matplotlib.colors.to_rgba("black")
        for tick, trained in zip(ax.get_xticklabels(), stats["is_trained"]):
            if not bool(trained):
                assert matplotlib.colors.to_rgba(tick.get_color()) == black
    finally:
        plt.close(fig)


def test_figure_draws_sem_error_bars_and_a_zero_line() -> None:
    stats = _stats(first_presentation_only=True)
    fig = render_score_bars_figure(stats)
    try:
        ax = fig.axes[0]
        assert ax.containers, "no bar container"
        assert any(
            getattr(container, "has_yerr", False) for container in ax.containers
        ), "SEM error bars missing"
        zero_lines = [
            line
            for line in ax.get_lines()
            if np.size(line.get_ydata())
            and np.allclose(np.asarray(line.get_ydata(), dtype=float), 0.0)
        ]
        assert zero_lines, "no baseline at score 0"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_remake_writes_a_figure_and_a_csv(tmp_path: Path) -> None:
    binary_csv, predictions_csv = _write_pair(tmp_path)
    out_dir = tmp_path / "out"

    written = remake_score_bars(
        csv_path=binary_csv,
        predictions_csv=predictions_csv,
        out_dir=out_dir,
        first_presentation_only=True,
    )

    png = out_dir / "score_bars_AIR-Training_unordered_first-presentation_copy.png"
    csv = out_dir / "score_bars_AIR-Training_unordered_first-presentation_copy.csv"
    assert png.exists() and csv.exists()
    assert png in written and csv in written

    saved = pd.read_csv(csv)
    assert saved["odor"].tolist() == [
        "Hexanol",
        "AIR",
        "Apple Cider Vinegar",
        "Ethyl Butyrate",
    ]


def test_remake_never_overwrites_the_reaction_figures(tmp_path: Path) -> None:
    binary_csv, predictions_csv = _write_pair(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    original = out_dir / "reaction_rates_AIR-Training_30_latency_2.150s_unordered_copy.png"
    original.write_bytes(b"ORIGINAL")

    remake_score_bars(
        csv_path=binary_csv,
        predictions_csv=predictions_csv,
        out_dir=out_dir,
        first_presentation_only=True,
    )

    assert original.read_bytes() == b"ORIGINAL"


@pytest.mark.skipif(
    not (REAL_BINARY_CSV.exists() and REAL_PREDICTIONS_CSV.exists()),
    reason="AIR-Training results not available",
)
def test_real_air_training_scores() -> None:
    scores = attach_odors(
        load_scores(REAL_PREDICTIONS_CSV, dataset="AIR-Training"),
        REAL_BINARY_CSV,
    )
    stats = score_stats(scores, trained_label="AIR", first_presentation_only=True)

    assert stats["odor"].tolist() == [
        "Hexanol",
        "AIR",
        "Apple Cider Vinegar",
        "Ethyl Butyrate",
        "Citral",
        "3-Octanol",
    ]
    assert stats["n_flies"].tolist() == [12] * 6
    # Means computed straight from model_predictions.csv, trial by trial.
    assert stats["mean_score"].round(4).tolist() == [
        0.75,
        -0.0833,
        0.3333,
        2.6667,
        0.1667,
        0.8333,
    ]
    assert "Benzaldehyde" not in set(stats["odor"])
