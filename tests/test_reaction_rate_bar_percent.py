"""The reaction-rate bar chart must read as a percentage (0-100%), matching the
training/control bars in reaction_matrix_training_vs_control.py, instead of the
old 0.0-1.0 fractional axis.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.envelope_visuals import plot_reaction_rate_bars  # noqa: E402


def _stats(rates: list[float]) -> pd.DataFrame:
    odors = ["Hexanol", "Citral", "Benzaldehyde"][: len(rates)]
    return pd.DataFrame(
        {
            "odor": odors,
            "rate": rates,
            "num_trials": [10] * len(rates),
            "is_trained": [False] * len(rates),
            "trial_num": list(range(1, len(rates) + 1)),
        }
    )


def test_bar_heights_are_scaled_to_percent():
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, _stats([0.5, 0.25]), title="t")
        heights = [round(p.get_height(), 6) for p in ax.patches]
        assert heights == [50.0, 25.0]  # 0.5 -> 50%, 0.25 -> 25%
    finally:
        plt.close(fig)


def test_y_axis_spans_zero_to_one_hundred():
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, _stats([1.0, 0.0]), title="t")
        top = ax.get_ylim()[1]
        assert 100.0 <= top <= 115.0  # 0..110 like the training/model bars
    finally:
        plt.close(fig)
