"""Unit tests for the RandomPanel trial-position aggregation.

These lock the trial-level aggregation contract (n, mean/SEM score, % reaction
with Wilson CI, block-position collapse, per-group Spearman trend) BEFORE the
figures are trusted, per the project's test-first workflow.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import randompanel_trial_position as rtp


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "conc": [1.0, 1.0, 1.0],
            "odor": ["Citral", "Citral", "Citral"],
            "fly": ["a", "b", "c"],
            "fly_number": [1, 1, 1],
            "trial_num": [3, 3, 3],
            "score": [0, 2, 4],
        }
    )


def test_add_position_columns_block_and_reacted():
    df = pd.DataFrame({"trial_num": [1, 7, 8, 14], "score": [-1, 2, 1, 5]})
    out = rtp.add_position_columns(df)
    # trial 1..7 -> block 1, 8..14 -> block 2
    assert list(out["block"]) == [1, 1, 2, 2]
    # block position collapses the two exposures: 8->1, 14->7
    assert list(out["blockpos"]) == [1, 7, 1, 7]
    # reaction boundary is score >= 2
    assert list(out["reacted"]) == [0, 1, 0, 1]


def test_aggregate_counts_mean_sem_and_pct():
    df = rtp.add_position_columns(_base_frame())
    summ = rtp.aggregate(df, ["conc", "trial_num"])
    assert len(summ) == 1
    row = summ.iloc[0]
    assert row["n"] == 3
    assert row["mean_score"] == pytest.approx(2.0)
    # sample SD of [0,2,4] = 2.0 -> SEM = 2/sqrt(3)
    assert row["sem_score"] == pytest.approx(2.0 / math.sqrt(3))
    # two of three trials clear score >= 2
    assert row["pct_react"] == pytest.approx(2.0 / 3.0)


def test_aggregate_wilson_ci_matches_reference():
    df = rtp.add_position_columns(_base_frame())
    row = rtp.aggregate(df, ["conc", "trial_num"]).iloc[0]
    # Wilson 95% CI for k=2, n=3 (independently computed).
    assert row["ci_lo"] == pytest.approx(0.2076, abs=1e-3)
    assert row["ci_hi"] == pytest.approx(0.9385, abs=1e-3)


def test_aggregate_singleton_group_has_zero_sem():
    df = rtp.add_position_columns(
        pd.DataFrame(
            {
                "conc": [1.0],
                "odor": ["Citral"],
                "fly": ["a"],
                "fly_number": [1],
                "trial_num": [5],
                "score": [4],
            }
        )
    )
    row = rtp.aggregate(df, ["conc", "trial_num"]).iloc[0]
    assert row["n"] == 1
    assert row["sem_score"] == 0.0


def test_spearman_by_group_perfect_monotonic():
    # score rises monotonically with trial position -> rho = +1.
    df = rtp.add_position_columns(
        pd.DataFrame(
            {
                "conc": [1.0] * 5,
                "odor": ["Citral"] * 5,
                "fly": list("abcde"),
                "fly_number": [1] * 5,
                "trial_num": [1, 2, 3, 4, 5],
                "score": [0, 2, 3, 4, 5],
            }
        )
    )
    sp = rtp.spearman_by_group(df, ["conc", "odor"], "trial_num")
    row = sp.iloc[0]
    assert row["rho"] == pytest.approx(1.0)
    assert row["n"] == 5


def test_load_trials_end_to_end(tmp_path: Path):
    """A tiny model_predictions.csv round-trips through the real loader."""
    rows = []
    odors = ["citral", "benzaldehyde", "acv", "hexanol", "linalool", "ethylbutyrate", "3-octonol"]
    # Two flies, all 14 odor trials + two light-only trials that must be dropped.
    for fly in ("rig_a", "rig_b"):
        for t in range(1, 15):
            od = odors[(t - 1) % 7]
            rows.append(
                {
                    "dataset": "RandomPanel-24-1",
                    "fly": fly,
                    "fly_number": 1,
                    "trial_label": f"training_{t}_{od}",
                    "prediction": 0,
                    "score": (t % 6),
                    "trial_type": "testing",
                    "fly_type": "GR5a-Old",
                }
            )
        # light-only trials (no odor token) -> dropped by _load_scores
        for t in (15, 16):
            rows.append(
                {
                    "dataset": "RandomPanel-24-1",
                    "fly": fly,
                    "fly_number": 1,
                    "trial_label": f"training_{t}",
                    "prediction": 0,
                    "score": 0,
                    "trial_type": "testing",
                    "fly_type": "GR5a-Old",
                }
            )
    csv = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    df = rtp.load_trials(csv, fly_type="GR5a-Old", config="")
    # Only odor trials 1..14 survive; light-only 15/16 dropped.
    assert set(df["trial_num"].unique()) == set(range(1, 15))
    assert df["conc"].unique().tolist() == [1.0]
    # 2 flies x 14 trials
    assert len(df) == 28
    # With no --config there is no odor_remap, so the default display names come
    # through. (The real run passes --config, which applies the per-dataset
    # remap: RandomPanel maps Linalool -> "Isoamyl Acetate".)
    seen_odors = set(df["odor"])
    assert "Citral" in seen_odors
    assert "Linalool" in seen_odors
