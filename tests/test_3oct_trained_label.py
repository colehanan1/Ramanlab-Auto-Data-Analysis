"""Regression tests: the 3Oct cohorts must resolve their trained odor.

``_trained_label(dataset_canon)`` names the odor a dataset trained on. Every
reaction-rate figure uses it twice: to decide which duplicate presentations get
numbered ("3-Octanol (0.1%) 1" / "... 2" instead of one pooled column), and to
bold the trained odor on the axis.

For "3OCT-Training-24-0.1" it used to fall all the way through to the dataset
name itself: ``PRIMARY_ODOR_LABEL`` has no entry for this cohort, and the
auto-derive fallback looked the base token "3OCT" up in ``_DISPLAY_LABEL_LOWER``
(which only knows the rig spelling "3-octonol"). Nothing then started with the
"trained label", so the two 3-Octanol presentations silently merged into one
bar with double the n (n=10 vs n=5 for every other odor) and no odor was bolded.

``odor_constants._auto_display_label`` already derives "3-Octanol" from the
canonical name via ``_BASE_ODORS``; ``_trained_label`` must consult it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis import envelope_visuals as ev
from scripts.analysis.envelope_visuals import _trained_label
from scripts.analysis.reaction_matrix_training_vs_control import _build_during_matrix


@pytest.fixture()
def v2_protocol():
    """These cohorts run the v2 protocol (odor suffix in every trial label)."""
    saved = ev.get_protocol()
    ev.set_protocol("v2")
    try:
        yield
    finally:
        ev.set_protocol(saved)


@pytest.mark.parametrize(
    ("dataset_canon", "expected"),
    [
        ("3OCT-Training-24-0.1", "3-Octanol"),
        ("3OCT-Control-24-0.1", "3-Octanol"),
        # Regressions: cohorts that already resolved must not change.
        ("EB-Training-24-1", "Ethyl Butyrate"),
        ("EB-Control-24-1", "Ethyl Butyrate"),
        ("Hex-Training-24-0.1", "Hexanol"),
    ],
)
def test_trained_label_resolves_odor_not_dataset_name(dataset_canon, expected) -> None:
    assert _trained_label(dataset_canon) == expected


def test_trained_label_is_not_the_dataset_name() -> None:
    """The old failure mode returned the dataset name itself — pin it out."""
    for ds in ("3OCT-Training-24-0.1", "3OCT-Control-24-0.1"):
        assert _trained_label(ds) != ds


# ---------------------------------------------------------------------------
# End-to-end: the trained odor's two presentations stay separate columns
# ---------------------------------------------------------------------------

def _binary_rows(dataset: str, n_flies: int) -> pd.DataFrame:
    """One fly-block per fly: 3-Octanol twice (trials 1 and 8) plus 6 singles."""
    odors = [
        "3-Octanol (0.1%)",
        "Benzaldehyde (0.1%)",
        "Citral (1%)",
        "Ethyl Butyrate (1%)",
        "Hexanol (0.1%)",
        "Isoamyl Acetate (1%)",
        "Linalool (1%)",
        "3-Octanol (0.1%)",
    ]
    rows = []
    for fly_idx in range(n_flies):
        for trial_num, odor in enumerate(odors, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "fly": f"batch_{fly_idx}",
                    "fly_number": str(fly_idx + 1),
                    "trial": f"testing_{trial_num}_{odor}",
                    "trial_num": trial_num,
                    "odor_sent": odor,
                    # First presentation reacts, second does not — so a pooled
                    # column would read 50% while the split columns read 100/0.
                    "during_hit": 1 if trial_num == 1 else 0,
                }
            )
    return pd.DataFrame(rows)


def test_trained_odor_presentations_are_separate_columns(v2_protocol) -> None:
    """3-Octanol 1 and 3-Octanol 2 must be distinct matrix columns, not pooled."""
    train_ds = "3OCT-Training-24-0.1"
    df = pd.concat(
        [_binary_rows(train_ds, n_flies=5), _binary_rows("3OCT-Control-24-0.1", n_flies=4)],
        ignore_index=True,
    )
    df["dataset_canon"] = df["dataset"]

    _matrix, fly_pairs, odor_columns, _flagged = _build_during_matrix(
        df, train_ds, None, remap_from=train_ds, order="observed"
    )

    assert len(fly_pairs) == 5
    trained_cols = [c for c in odor_columns if str(c).startswith("3-Octanol")]
    assert trained_cols == ["3-Octanol (0.1%) 1", "3-Octanol (0.1%) 2"], (
        f"trained odor was not split per presentation; got {odor_columns}"
    )
    # Non-trained odors stay single columns.
    assert sum(str(c).startswith("Citral") for c in odor_columns) == 1
