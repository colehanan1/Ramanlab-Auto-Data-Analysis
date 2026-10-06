"""The trial-profile figure must colour every test presentation.

``figure_trial_profile`` coloured only four presentation labels, so a cohort
that presents its conditioning odor at more test positions crashed the whole
command with ``KeyError: 'Hex 5'``. Hex-Sensitivity-24-0.1 presents hexanol at
six (testing 1, 2, 4, 5, 6, 7): every night since 2026-09-24 the control-arm
training_auc_vs_control_response.py run died there, after that cohort's
trial_matrix figure, leaving its trial_profile / per_fly.csv / stats.json, the
three cohorts after it and the root sidecar unwritten.

The fix extends the categorical palette in its fixed slot order, so the four
labels coloured before keep their colours and existing figures are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import training_auc_vs_control_response as mod  # noqa: E402

DATASET = "Hex-Sensitivity-24-0.1"
#: The colours the four original slots drew before the fix (pinned, so no
#: existing figure changes colour).
OLD_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _table(presentations, n_flies=6):
    train, test = [], []
    for fly in range(n_flies):
        for trial in (1, 2, 3, 4):
            train.append({
                "dataset": DATASET, "fly": f"day_{fly}", "fly_number": 1,
                "trial_type": "training",
                "trial_label": f"training_{trial}_hexanol",
                "AUC-During": float(10 * fly + trial * (fly % 3)),
            })
        for k, trial in enumerate(presentations):
            score = (fly + k) % 6 - 1
            test.append({
                "dataset": DATASET, "fly": f"day_{fly}", "fly_number": 1,
                "trial_type": "testing",
                "trial_label": f"testing_{trial}_hexanol",
                "score": score, "prediction": int(score >= 2),
            })
    table = mod.build_fly_table(
        pd.DataFrame(train), pd.DataFrame(test), dataset=DATASET
    )
    assert list(table.attrs["presentations"]) == list(presentations)
    return table


def _line_colours(fig):
    """{label: colour} of the series lines drawn on the first panel."""
    ax = fig.axes[0]
    return {
        line.get_label(): matplotlib.colors.to_hex(line.get_color())
        for line in ax.get_lines() if not line.get_label().startswith("_")
    }


def test_six_presentations_render_without_a_key_error():
    """Hex-Sensitivity-24-0.1's layout: hexanol at six test positions."""
    table = _table([1, 2, 4, 5, 6, 7])
    fig, payload = mod.figure_trial_profile(
        table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        labels = [f"Hex {i}" for i in range(1, 7)]
        for panel in payload:
            assert list(panel["series"]) == labels
        colours = _line_colours(fig)
        assert list(colours) == labels
        # Every presentation is its own colour: identity must not repeat.
        assert len(set(colours.values())) == 6
    finally:
        matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_labels_coloured_before_keep_their_colours(n):
    table = _table([1, 2, 4, 5][:n])
    fig, _payload = mod.figure_trial_profile(
        table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert list(_line_colours(fig).values()) == OLD_SLOTS[:n]
    finally:
        matplotlib.pyplot.close(fig)


def test_new_slots_follow_the_validated_categorical_order():
    """Slots 5-8 continue the same validated palette the first four came
    from, in its fixed order (magenta, green, violet, red)."""
    assert list(mod.SERIES_COLORS[:4]) == OLD_SLOTS
    assert list(mod.SERIES_COLORS[4:]) == ["#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def test_series_past_the_palette_fold_to_a_neutral_not_a_repeat():
    """A ninth series never reuses a hue (that would claim two presentations
    are the same); it folds to the neutral ink and keeps its direct label."""
    table = _table(list(range(1, 11)))
    fig, _payload = mod.figure_trial_profile(
        table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        colours = list(_line_colours(fig).values())
        assert colours[:8] == list(mod.SERIES_COLORS)
        assert colours[8:] == [mod.MUTED, mod.MUTED]
    finally:
        matplotlib.pyplot.close(fig)
