"""A cohort may need different odor labels than its dataset.

``Hex-Training-24-0.1`` ran two different rig plumbings over its life. Its
May/June flies had sourdough yeast on the Citral channel and isoamyl acetate on
the Linalool channel; its August flies use the newer arrangement where the ACV
channel carries the isoamyl acetate and Citral/Linalool are genuinely
themselves. Both live under one dataset name, so the config's per-dataset
``odor_remap`` cannot describe both -- setting it either way silently mislabels
the other era.

``--odor-remap`` overrides the labels for one figure only. The dataset-level
remap is left alone, so every other figure over those same May/June flies keeps
the sourdough/isoamyl correction it needs.

The override must reach the CONTROL cohort too: figures pair training and
control by display label, so relabelling only one side splits an odor into two
unpaired columns and still renders a plausible-looking figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import pubfig_score_train_vs_control as pub  # noqa: E402


def _predictions(tmp_path: Path) -> Path:
    rows = []
    for dataset in ("Hex-Training-24-0.1", "Hex-Control-24-0.1"):
        for fly_number in (1, 2):
            for trial in ("testing_2_acv", "testing_3_citral", "testing_4_linalool"):
                rows.append(
                    {
                        "dataset": dataset,
                        "fly": "august_10_batch_1_rig_2",
                        "fly_number": fly_number,
                        "trial_label": trial,
                        "score": 4.0,
                        "trial_type": "testing",
                        "fly_type": "GR5a-Old",
                    }
                )
    out = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _odors(tmp_path: Path, overrides) -> list[str]:
    rows = pub.rows_from_score_summary(
        _predictions(tmp_path), "Hex-Training-24-0.1",
        config=None, odor_remap=overrides,
    )
    return list(rows["odor"])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_splits_on_the_first_equals_only():
    """Values contain '=' in no case today, but names must survive one anyway."""
    assert pub.parse_odor_remap(["A=B (1%)"]) == {"A": "B (1%)"}
    assert pub.parse_odor_remap(["A=B=C"]) == {"A": "B=C"}


def test_parse_trims_surrounding_space():
    assert pub.parse_odor_remap([" Apple Cider Vinegar = Isoamyl Acetate (1%) "]) == {
        "Apple Cider Vinegar": "Isoamyl Acetate (1%)"
    }


def test_parse_rejects_a_pair_with_no_equals():
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        pub.parse_odor_remap(["Apple Cider Vinegar"])


def test_parse_rejects_an_empty_key():
    with pytest.raises(SystemExit, match="empty"):
        pub.parse_odor_remap(["=Isoamyl Acetate (1%)"])


def test_parse_of_nothing_is_an_empty_mapping():
    assert pub.parse_odor_remap([]) == {}
    assert pub.parse_odor_remap(None) == {}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def test_without_an_override_the_bare_names_are_used(tmp_path):
    assert _odors(tmp_path, None) == [
        "Apple Cider Vinegar", "Citral", "Linalool"
    ]


def test_override_relabels_the_training_cohort(tmp_path):
    odors = _odors(tmp_path, {"Apple Cider Vinegar": "Isoamyl Acetate (1%)"})
    assert "Isoamyl Acetate (1%)" in odors
    assert "Apple Cider Vinegar" not in odors


def test_override_reaches_the_control_cohort_so_bars_stay_paired(tmp_path):
    """If only training were relabelled, the odor would split into two columns
    with one empty cohort each -- and the figure would still render."""
    rows = pub.rows_from_score_summary(
        _predictions(tmp_path), "Hex-Training-24-0.1", config=None,
        odor_remap={"Apple Cider Vinegar": "Isoamyl Acetate (1%)"},
    )
    row = rows[rows["odor"] == "Isoamyl Acetate (1%)"].iloc[0]
    assert int(row["n_train"]) == 2 and int(row["n_ctrl"]) == 2


def test_override_applies_to_the_percent_metric_too(tmp_path):
    rows = pub.rows_from_percent_responding(
        _predictions(tmp_path), "Hex-Training-24-0.1", config=None,
        odor_remap={"Citral": "Citral (1%)"},
    )
    assert "Citral (1%)" in list(rows["odor"])


def test_override_does_not_leak_into_other_datasets(tmp_path):
    """Registering the override must not relabel a dataset this figure is not
    about -- the -0.1 May/June figures still need their own labels."""
    from scripts.analysis import envelope_visuals as ev

    pub.rows_from_score_summary(
        _predictions(tmp_path), "Hex-Training-24-0.1", config=None,
        odor_remap={"Apple Cider Vinegar": "Isoamyl Acetate (1%)"},
    )
    ev.set_protocol("v2")
    assert ev._display_odor("EB-Training-24-1", "testing_2_acv") == (
        "Apple Cider Vinegar"
    )


def test_cli_accepts_repeated_odor_remap_flags(tmp_path):
    out_dir = tmp_path / "figs"
    pub.main(
        [
            "dataset",
            "--train-dataset", "Hex-Training-24-0.1",
            "--predictions-csv", str(_predictions(tmp_path)),
            "--figures-dir", str(out_dir),
            "--out-stem", "t",
            "--odor-remap", "Apple Cider Vinegar=Isoamyl Acetate (1%)",
            "--odor-remap", "Citral=Citral (1%)",
        ]
    )
    labels = (out_dir / "t.svg").read_text(errors="ignore")
    assert "Isoamyl Acetate (1%)" in labels
    assert "Citral (1%)" in labels
