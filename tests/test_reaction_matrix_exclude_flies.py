"""Tests for excluding flies (by exact key or substring pattern) from BOTH
the training and control sides in ``reaction_matrix_specific_flies_vs_control.py``.

Unlike ``--fly`` (which restricts the TRAINING side to a hand-picked allow
list, control untouched), exclusions must drop matching flies everywhere —
a bad rig affects both arms of the comparison.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

MODULE_PATH = PROJECT_ROOT / "scripts" / "analysis" / "reaction_matrix_specific_flies_vs_control.py"
spec = importlib.util.spec_from_file_location("rm_specific", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _df():
    return pd.DataFrame(
        {
            "dataset_canon": [
                "EB-Training-24-1", "EB-Training-24-1", "EB-Training-24-1",
                "EB-Control-24-1", "EB-Control-24-1",
            ],
            "fly": [
                "july_17_batch_2_rig_2", "july_17_batch_2_rig_3", "july_18_batch_1",
                "july_17_batch_1_rig_3", "july_13_batch_1",
            ],
            "fly_number": ["1", "1", "1", "1", "1"],
        }
    )


def test_exclusion_mask_drops_pattern_match_case_insensitively_on_both_datasets():
    mask = module._exclusion_mask(_df(), exclude_flies=set(), exclude_patterns=["RIG_3"])
    assert mask.tolist() == [False, True, False, True, False]


def test_exclusion_mask_drops_exact_fly_key_regardless_of_dataset():
    mask = module._exclusion_mask(
        _df(),
        exclude_flies={("july_17_batch_2_rig_2", "1")},
        exclude_patterns=[],
    )
    assert mask.tolist() == [True, False, False, False, False]


def test_exclusion_mask_combines_pattern_and_exact_without_double_dropping_others():
    mask = module._exclusion_mask(
        _df(),
        exclude_flies={("july_17_batch_2_rig_2", "1")},
        exclude_patterns=["rig_3"],
    )
    assert mask.tolist() == [True, True, False, True, False]


def test_exclusion_mask_empty_criteria_excludes_nothing():
    mask = module._exclusion_mask(_df(), exclude_flies=set(), exclude_patterns=[])
    assert not mask.any()


def test_flagged_flies_truth_table_overrides_a_stale_non_reactive_column(tmp_path):
    """The predictions CSV can already carry a ``_non_reactive`` column that
    was computed WITHOUT the truth-table CSV (e.g. all False). When a
    ``--flagged-flies-csv`` truth table is supplied it must still be applied
    — it is authoritative and must not be shadowed by a stale precomputed
    column, matching ``reaction_matrix_training_vs_control.py``'s behavior."""
    truth_csv = tmp_path / "flagged-flys-truth.csv"
    truth_csv.write_text(
        'dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment\n'
        "EB-Training-24-1,july_18_batch_1,1,-1,dead\n"
    )

    df = pd.DataFrame(
        {
            "dataset": ["EB-Training-24-1", "EB-Training-24-1"],
            "fly": ["july_18_batch_1", "july_18_batch_1"],
            "fly_number": ["1", "2"],
            "_non_reactive": [False, False],  # stale: computed before truth table existed
        }
    )

    mask_without_truth_table = module._resolve_non_reactive_mask(df, "")
    assert not mask_without_truth_table.any(), (
        "sanity check: with no truth table, the stale column (all False) is trusted"
    )

    mask_with_truth_table = module._resolve_non_reactive_mask(df, str(truth_csv))
    assert mask_with_truth_table.tolist() == [True, False], (
        "truth-table exclusion for july_18_batch_1:1 must apply despite the "
        "stale _non_reactive column saying False"
    )


def test_truth_table_float_fly_number_still_matches_clean_int_fly_number(tmp_path):
    """The truth CSV's ``fly_number`` column parses as float when ANY row has a
    blank fly_number (one NaN promotes the whole column to float64), so a naive
    ``str(fly_number)`` yields ``'1.0'``. The predictions data has clean ``'1'``.
    Matching must survive this ``'1.0'`` vs ``'1'`` mismatch, else flagged
    (state != 1) flies are silently NOT removed."""
    truth_csv = tmp_path / "flagged-flys-truth.csv"
    # Trailing blank-fly_number row forces the column to float64 -> "1.0" keys.
    truth_csv.write_text(
        'dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment\n'
        "EB-Training-24-1,july_19_batch_1,1,-1,dead\n"
        "EB-Training-24-1,july_16_batch_1_rig_2,2,0,bad\n"
        "EB-Training-24-1,some_absent_fly,,0,no fly_number\n"
    )

    df = pd.DataFrame(
        {
            "dataset": ["EB-Training-24-1"] * 3,
            "fly": ["july_19_batch_1", "july_16_batch_1_rig_2", "july_20_batch_1"],
            "fly_number": ["1", "2", "1"],  # clean int-strings from the predictions data
        }
    )

    mask = module._resolve_non_reactive_mask(df, str(truth_csv))
    assert mask.tolist() == [True, True, False], (
        "float-formatted truth-table fly_number ('1.0') must still match the "
        f"predictions data's '1'; got {mask.tolist()}"
    )
