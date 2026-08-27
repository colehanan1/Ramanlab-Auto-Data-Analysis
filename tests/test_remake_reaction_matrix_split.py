"""Tests for regenerating reaction matrices as two split figures, sans Benzaldehyde."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.analysis.remake_reaction_matrix_split import (
    DEFAULT_EXCLUDE,
    build_matrix,
    load_binary_reactions,
    plot_reaction_rate_bars,
    rate_stats,
    remake,
    render_bars_figure,
)

REAL_CSV = Path(
    "/home/ramanlab/Documents/cole/Results/Opto-Fly-Figures/"
    "Matrix-PER-Reactions-Model/AIR-Training/binary_reactions_AIR-Training_unordered.csv"
)


def _synthetic() -> pd.DataFrame:
    """Two flies, five presentations each, mirroring the AIR-Training layout."""
    rows = []
    schedule = [
        (1, "Hexanol"),
        (2, "AIR"),
        (3, "Ethyl Butyrate"),
        (4, "Benzaldehyde"),
        (5, "Citral"),
    ]
    hits = {
        ("batch_a", 1): {1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
        ("batch_a", 2): {1: 0, 2: 0, 3: 1, 4: 1, 5: 0},
    }
    for (fly, fly_number), per_trial in hits.items():
        for trial_num, odor in schedule:
            rows.append(
                {
                    "dataset": "AIR-Training",
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial_num": trial_num,
                    "odor_sent": odor,
                    "during_hit": per_trial[trial_num],
                    "after_hit": per_trial[trial_num],
                    "prob_reaction": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _write(tmp_path: Path, df: pd.DataFrame) -> Path:
    csv_path = tmp_path / "binary_reactions_AIR-Training_unordered.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_benzaldehyde_column_is_dropped_from_matrix() -> None:
    matrix, labels, fly_pairs = build_matrix(_synthetic(), exclude=DEFAULT_EXCLUDE)

    assert not any("benzaldehyde" in label.lower() for label in labels)
    assert labels == ["Hexanol", "AIR", "Ethyl Butyrate", "Citral"]
    assert matrix.shape == (len(fly_pairs), len(labels)) == (2, 4)


def test_matrix_keeps_trial_order_and_hit_values() -> None:
    matrix, labels, fly_pairs = build_matrix(_synthetic(), exclude=DEFAULT_EXCLUDE)

    assert fly_pairs == [("batch_a", 1), ("batch_a", 2)]
    # Row 0: Hexanol hit, AIR miss, Ethyl Butyrate hit, Citral miss.
    assert matrix[0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert matrix[1].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_rate_stats_exclude_benzaldehyde_and_flag_trained_odor() -> None:
    stats = rate_stats(_synthetic(), trained_label="AIR", exclude=DEFAULT_EXCLUDE)

    assert "Benzaldehyde" not in set(stats["odor"])
    assert stats["odor"].tolist() == ["Hexanol", "AIR", "Ethyl Butyrate", "Citral"]
    assert stats.loc[stats["odor"] == "Ethyl Butyrate", "rate"].iloc[0] == pytest.approx(1.0)
    assert stats.loc[stats["odor"] == "Hexanol", "rate"].iloc[0] == pytest.approx(0.5)
    assert stats["is_trained"].tolist() == [False, True, False, False]


def test_remake_writes_two_separate_figures(tmp_path: Path) -> None:
    csv_path = _write(tmp_path, _synthetic())
    out_dir = tmp_path / "out"

    written = remake(csv_path=csv_path, out_dir=out_dir, latency_sec=2.15)

    matrix_png = out_dir / "reaction_matrix_AIR-Training_30_latency_2.150s_unordered_copy.png"
    bars_png = out_dir / "reaction_rates_AIR-Training_30_latency_2.150s_unordered_copy.png"
    assert matrix_png.exists(), "matrix figure missing"
    assert bars_png.exists(), "bar figure missing"
    assert matrix_png in written and bars_png in written
    # Two figures, not one stacked figure.
    assert matrix_png != bars_png


def test_remake_never_overwrites_the_originals(tmp_path: Path) -> None:
    csv_path = _write(tmp_path, _synthetic())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    original = out_dir / "reaction_matrix_AIR-Training_30_latency_2.150s_unordered.png"
    original.write_bytes(b"ORIGINAL")

    remake(csv_path=csv_path, out_dir=out_dir, latency_sec=2.15)

    assert original.read_bytes() == b"ORIGINAL"
    for path in out_dir.rglob("*"):
        if path.is_file() and path != original:
            assert "_copy" in path.name, f"unexpected non-copy output: {path.name}"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="AIR-Training results not available")
def test_real_air_training_rates_match_published_figure() -> None:
    df = load_binary_reactions(REAL_CSV)
    stats = rate_stats(df, trained_label="AIR", exclude=DEFAULT_EXCLUDE)

    assert "Benzaldehyde" not in set(stats["odor"])
    # Presentation order and percentages read off the published figure.
    assert stats["odor"].tolist() == [
        "Hexanol",
        "AIR",
        "Hexanol",
        "AIR",
        "AIR",
        "Apple Cider Vinegar",
        "Ethyl Butyrate",
        "Citral",
        "3-Octanol",
    ]
    percentages = [round(rate * 100) for rate in stats["rate"]]
    assert percentages == [25, 8, 17, 8, 8, 17, 67, 8, 25]
    assert set(stats["num_trials"]) == {12}


def test_bars_figure_axis_labels() -> None:
    """The split bar figure names the y-axis and carries no x-axis label."""
    stats = rate_stats(_synthetic(), trained_label="AIR", exclude=DEFAULT_EXCLUDE)

    fig = render_bars_figure(stats)
    try:
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Mean PER response %"
        assert ax.get_xlabel() == ""
    finally:
        plt.close(fig)


def test_shared_bar_helper_keeps_its_own_defaults() -> None:
    """Other figures that call the helper are unaffected by the split-figure labels."""
    stats = rate_stats(_synthetic(), trained_label="AIR", exclude=DEFAULT_EXCLUDE)

    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, stats, title="Reaction Rates by Odor")
        assert ax.get_ylabel() == "Mean PER response %"
        assert ax.get_xlabel() == "Presented Odor"
    finally:
        plt.close(fig)
