import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.analysis.envelope_visuals import reaction_rate_stats_from_rows, resolve_dataset_output_dir

TRAIN_VS_CTRL_PATH = PROJECT_ROOT / "scripts" / "analysis" / "reaction_matrix_training_vs_control.py"
spec = importlib.util.spec_from_file_location("reaction_matrix_training_vs_control", TRAIN_VS_CTRL_PATH)
tvc_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tvc_module
spec.loader.exec_module(tvc_module)


def _trial_rows() -> pd.DataFrame:
    rows = []
    patterns = {
        "fly1": [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        "fly2": [0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
    }
    for fly, values in patterns.items():
        for trial_num, value in enumerate(values, start=1):
            rows.append(
                {
                    "trial": f"testing_{trial_num}",
                    "during_hit": value,
                    "fly": fly,
                    "fly_number": fly[-1],
                }
            )
    return pd.DataFrame(rows)


def test_reaction_rate_stats_keep_duplicate_odors_separate_by_trial():
    stats = reaction_rate_stats_from_rows(
        _trial_rows(),
        "Benz-Training",
        include_hexanol=True,
        context="unit-test",
        trial_col="trial",
        reaction_col="during_hit",
        separate_presentations=True,
    )

    assert stats["trial_num"].tolist() == list(range(1, 11))
    assert stats["odor"].tolist().count("Hexanol") == 2
    assert stats["odor"].tolist().count("Benzaldehyde") == 3
    assert stats.loc[stats["trial_num"] == 2, "is_trained"].iloc[0]
    assert stats.loc[stats["trial_num"] == 1, "odor"].iloc[0] == "Hexanol"
    assert stats.loc[stats["trial_num"] == 4, "odor"].iloc[0] == "Benzaldehyde"


def test_training_vs_control_loader_keeps_presentations_separate(tmp_path):
    out_dir = tmp_path / "matrix"
    ds_dir = resolve_dataset_output_dir(out_dir, "Benz-Training")
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "trial_num": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "odor_sent": [
                "Hexanol",
                "Hexanol",
                "Benzaldehyde",
                "Benzaldehyde",
                "Hexanol",
                "Hexanol",
                "Benzaldehyde",
                "Benzaldehyde",
                "Benzaldehyde",
                "Benzaldehyde",
            ],
            "during_hit": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        }
    ).to_csv(ds_dir / "binary_reactions_Benz-Training_unordered.csv", index=False)

    stats = tvc_module._load_rates_from_binary_csv(
        out_dir,
        "Benz-Training",
        "Benz-Training",
        include_hexanol=True,
    )

    assert stats["trial_num"].tolist() == [1, 2, 3, 4, 5]
    assert stats["odor"].tolist() == [
        "Hexanol",
        "Benzaldehyde",
        "Hexanol",
        "Benzaldehyde",
        "Benzaldehyde",
    ]
    assert stats["is_trained"].tolist() == [False, True, False, True, True]


def _raw_v2_frame(reactions: dict[str, list[int]]) -> pd.DataFrame:
    """Two flies, each presented "EB" twice (trial_num 1 and 3), plus one
    "Hexanol" presentation (trial_num 2) that is never duplicated."""
    rows = []
    for fly, hits in reactions.items():
        eb_hit_1, eb_hit_2, hex_hit = hits
        rows.append({"fly": fly, "fly_number": "1", "trial_num": 1, "odor_sent": "EB", "during_hit": eb_hit_1})
        rows.append({"fly": fly, "fly_number": "1", "trial_num": 2, "odor_sent": "Hexanol", "during_hit": hex_hit})
        rows.append({"fly": fly, "fly_number": "1", "trial_num": 3, "odor_sent": "EB", "during_hit": eb_hit_2})
    return pd.DataFrame(rows)


def test_fisher_test_per_presentation_scores_each_occurrence_independently():
    """The training-vs-control Fisher's exact test for a twice-presented
    trained odor ("EB 1" / "EB 2") must compare each occurrence against its
    matching control occurrence separately, not pool both presentations into
    one combined contingency table."""
    train_raw = _raw_v2_frame({
        "fly1": [1, 0, 1],  # EB occurrence 1: hit, EB occurrence 2: miss
        "fly2": [1, 0, 1],
    })
    ctrl_raw = _raw_v2_frame({
        "fly1": [0, 0, 0],
        "fly2": [0, 0, 0],
    })

    p_values = tvc_module._fisher_test_per_presentation(
        train_raw, ctrl_raw, [(0, "EB 1"), (0, "EB 2")]
    )

    p_occ1 = p_values[(0, "EB 1")]
    p_occ2 = p_values[(0, "EB 2")]

    # occurrence 1: train 2/2 react vs ctrl 0/2 react -> fisher_exact([[2,0],[0,2]])
    assert p_occ1 == pytest.approx(0.3333333333333333)
    # occurrence 2: train 0/2 react vs ctrl 0/2 react -> fisher_exact([[0,2],[0,2]])
    assert p_occ2 == pytest.approx(1.0)
    # The two occurrences must not collapse onto the same pooled p-value.
    assert p_occ1 != pytest.approx(p_occ2)
