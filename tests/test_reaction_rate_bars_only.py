"""Tests for the bar-only re-render of RandomPanel reaction-matrix figures.

The golden percentages below were read directly off the published stacked
figures in ``Matrix-PER-Reactions-Model/<dataset>/reaction_matrix_*.png``; the
bar-only figures must reproduce them bar-for-bar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.reaction_rate_bars_only import (
    bar_stats,
    load_binary_reactions,
    make_bars,
    resolve_genotype,
)

RESULTS_ROOT = Path(
    "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Matrix-PER-Reactions-Model"
)

# (csv path, expected bar labels, expected PER percentages, expected n per bar)
PUBLISHED = {
    "RandomPanel-24-0.1": (
        RESULTS_ROOT / "RandomPanel-24-0.1" / "binary_reactions_RandomPanel-24-0.1_unordered.csv",
        [
            "3-Octanol 1", "3-Octanol 2",
            "Apple Cider Vinegar 1", "Apple Cider Vinegar 2",
            "Benzaldehyde 1", "Benzaldehyde 2",
            "Citral 1", "Citral 2",
            "Ethyl Butyrate 1", "Ethyl Butyrate 2",
            "Hexanol 1", "Hexanol 2",
            "Isoamyl Acetate 1", "Isoamyl Acetate 2",
        ],
        [15, 30, 30, 25, 10, 30, 25, 30, 30, 35, 55, 60, 35, 40],
        [20] * 14,
    ),
    "RandomPanel-24-1": (
        RESULTS_ROOT / "RandomPanel-24-1" / "binary_reactions_RandomPanel-24-1_unordered.csv",
        [
            "3-Octanol 1", "3-Octanol 2",
            "Apple Cider Vinegar 1", "Apple Cider Vinegar 2",
            "Benzaldehyde 1", "Benzaldehyde 2",
            "Citral 1", "Citral 2",
            "Ethyl Butyrate 1", "Ethyl Butyrate 2", "Ethyl Butyrate 3",
            "Hexanol 1", "Hexanol 2",
            "Isoamyl Acetate 1", "Isoamyl Acetate 2",
        ],
        [20, 5, 25, 10, 60, 50, 20, 10, 30, 30, 33, 60, 55, 20, 5],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 3, 20, 20, 20, 20],
    ),
    "RandomPanel-Training-24-10/GR5a-Old": (
        RESULTS_ROOT
        / "RandomPanel-Training-24-10"
        / "GR5a-Old"
        / "binary_reactions_RandomPanel-Training-24-10_unordered.csv",
        [
            "3-Octanol 1", "3-Octanol 2",
            "Apple Cider Vinegar 1", "Apple Cider Vinegar 2",
            "Benzaldehyde 1", "Benzaldehyde 2",
            "Citral 1", "Citral 2",
            "Ethyl Butyrate 1", "Ethyl Butyrate 2",
            "Hexanol 1", "Hexanol 2",
            "Isoamyl Acetate 1", "Isoamyl Acetate 2",
        ],
        [15, 15, 20, 5, 55, 35, 15, 20, 75, 75, 60, 60, 65, 55],
        [20] * 14,
    ),
    "RandomPanel-Training-24-10/GR5a-GCaMP8": (
        RESULTS_ROOT
        / "RandomPanel-Training-24-10"
        / "GR5a-GCaMP8"
        / "binary_reactions_RandomPanel-Training-24-10_unordered.csv",
        [
            "3-Octanol 1", "3-Octanol 2",
            "Apple Cider Vinegar 1", "Apple Cider Vinegar 2",
            "Benzaldehyde 1", "Benzaldehyde 2",
            "Citral 1", "Citral 2",
            "Ethyl Butyrate 1", "Ethyl Butyrate 2",
            "Hexanol 1", "Hexanol 2",
            "Isoamyl Acetate 1", "Isoamyl Acetate 2",
        ],
        [12, 12, 0, 12, 12, 12, 0, 12, 12, 50, 25, 12, 25, 38],
        [8] * 14,
    ),
}


def _synthetic() -> pd.DataFrame:
    """Two flies, a 2-odor panel presented twice each — the RandomPanel shape."""
    schedule = {
        ("july_01_batch_1", 1): [(1, "Hexanol", 1), (2, "Citral", 0), (3, "Hexanol", 1), (4, "Citral", 0)],
        # Fly 2 sees the same panel in a different (randomised) trial order.
        ("july_01_batch_1", 2): [(1, "Citral", 1), (2, "Hexanol", 0), (3, "Citral", 0), (4, "Hexanol", 0)],
    }
    rows = []
    for (fly, fly_number), trials in schedule.items():
        for trial_num, odor, hit in trials:
            rows.append(
                {
                    "dataset": "RandomPanel-24-0.1",
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial_num": trial_num,
                    "odor_sent": odor,
                    "during_hit": hit,
                    "after_hit": hit,
                    "prob_reaction": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def test_bar_stats_numbers_each_presentation_of_a_repeated_odor() -> None:
    stats = bar_stats(_synthetic(), "RandomPanel-24-0.1")

    # Alphabetical by label, one bar per (odor, occurrence) — not per trial number.
    assert stats["odor"].tolist() == ["Citral 1", "Citral 2", "Hexanol 1", "Hexanol 2"]
    assert stats["num_trials"].tolist() == [2, 2, 2, 2]
    # Citral 1 = fly1 trial2 (miss) + fly2 trial1 (hit) -> 50%.
    assert stats["rate"].tolist() == pytest.approx([0.5, 0.0, 0.5, 0.5])


def test_bar_stats_groups_by_occurrence_not_trial_number() -> None:
    """Randomised order means (trial_num, odor) grouping would fragment the bars."""
    stats = bar_stats(_synthetic(), "RandomPanel-24-0.1")

    assert len(stats) == 4, "expected 2 odors x 2 presentations, not one bar per trial number"
    assert "trial_num" not in stats.columns


def test_make_bars_writes_a_bar_only_figure(tmp_path: Path) -> None:
    csv_path = tmp_path / "binary_reactions_RandomPanel-24-0.1_unordered.csv"
    _synthetic().to_csv(csv_path, index=False)
    out_dir = tmp_path / "out"

    written = make_bars(csv_path=csv_path, out_dir=out_dir)

    png = out_dir / "reaction_rate_bars_RandomPanel-24-0.1_unordered.png"
    assert png.exists(), "bar figure missing"
    assert png in written
    assert not any("reaction_matrix" in path.name for path in written), "no matrix should be drawn"


def test_make_bars_leaves_the_published_figures_untouched(tmp_path: Path) -> None:
    csv_path = tmp_path / "binary_reactions_RandomPanel-24-0.1_unordered.csv"
    _synthetic().to_csv(csv_path, index=False)
    original = tmp_path / "reaction_matrix_RandomPanel-24-0.1_30_latency_2.150s_unordered.png"
    original.write_bytes(b"ORIGINAL")

    make_bars(csv_path=csv_path, out_dir=tmp_path)

    assert original.read_bytes() == b"ORIGINAL"


def test_resolve_genotype_reads_the_split_subfolder(tmp_path: Path) -> None:
    plain = tmp_path / "RandomPanel-24-1" / "binary_reactions_RandomPanel-24-1_unordered.csv"
    split = (
        tmp_path
        / "RandomPanel-Training-24-10"
        / "GR5a-Old"
        / "binary_reactions_RandomPanel-Training-24-10_unordered.csv"
    )
    assert resolve_genotype(plain, "RandomPanel-24-1") is None
    assert resolve_genotype(split, "RandomPanel-Training-24-10") == "GR5a-Old"


@pytest.mark.parametrize("key", sorted(PUBLISHED))
def test_bars_match_the_published_matrix_figures(key: str) -> None:
    csv_path, labels, percentages, counts = PUBLISHED[key]
    if not csv_path.exists():
        pytest.skip(f"{key} results not available")

    df = load_binary_reactions(csv_path)
    dataset = key.split("/")[0]
    stats = bar_stats(df, dataset)

    assert stats["odor"].tolist() == labels
    assert [round(rate * 100) for rate in stats["rate"]] == percentages
    assert stats["num_trials"].tolist() == counts
    # RandomPanel has no trained odor, so no bar is highlighted blue.
    assert not stats["is_trained"].any()
