"""Raw testing traces for the folder-FROZEN flies a dataset dropped from its figures.

``3Oct-Training-24-0.1`` freezes ``*_rig_3`` before 2026-08-11, so those flies'
rows stay in the wide table marked ``frozen`` and never reach a figure. This
driver renders them anyway — one raw-testing-trace figure per fly — restricted
to flies the flagged-flies truth CSV does NOT exclude.

The tests pin the selection, which is the part that can silently go wrong:
  * only frozen rows are picked (a live fly must never leak in);
  * only the requested folder pattern is picked;
  * only ``testing`` trials are picked;
  * flagged flies are dropped, matched on the canonical fly_number so a
    float-parsed ``1.0`` still matches the CSV's ``1``;
  * the rendered slice is un-frozen, because ``_load_wide_table`` drops frozen
    rows unconditionally and would otherwise discard the whole selection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis import frozen_folder_testing_traces as fft


DATASET = "3Oct-Training-24-0.1"


def _row(fly: str, fly_number, trial_label: str, *, frozen: bool, trial_type="testing"):
    return {
        "dataset": DATASET,
        "fly": fly,
        "fly_number": fly_number,
        "fly_type": "GR5a-Old",
        "trial_type": trial_type,
        "trial_label": trial_label,
        "fps": 40.0,
        "frozen": frozen,
        "global_min": 1.0,
        "global_max": 25.0,
        "trace_len": 4,
        "dir_val_0": 1.0,
        "dir_val_1": 8.5,
        "dir_val_2": 16.25,
        "dir_val_3": 4.0,
    }


@pytest.fixture
def wide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # frozen rig_3, not flagged -> KEEP
            _row("july_24_batch_2_rig_3", 1, "testing_1_3-octonol", frozen=True),
            _row("july_24_batch_2_rig_3", 1, "testing_8_3-octonol", frozen=True),
            # frozen rig_3, flagged -> DROP
            _row("july_26_batch_1_rig_3", 1, "testing_1_3-octonol", frozen=True),
            # frozen rig_3 but a training trial -> DROP
            _row(
                "july_24_batch_2_rig_3", 1, "training_1", frozen=True,
                trial_type="training",
            ),
            # live rig_3 (post-repair) -> DROP, it is already in the figures
            _row("august_15_batch_1_rig_3", 1, "testing_1_3-octonol", frozen=False),
            # frozen rig_2 -> DROP, wrong folder pattern
            _row("july_24_batch_2_rig_2", 1, "testing_1_3-octonol", frozen=True),
            # a different dataset's frozen rig_3 -> DROP
            {**_row("july_24_batch_2_rig_3", 1, "testing_1_3-octonol", frozen=True),
             "dataset": "3Oct-Control-24-0.1"},
        ]
    )


@pytest.fixture
def flagged_csv(tmp_path: Path) -> Path:
    path = tmp_path / "flagged-flys-truth.csv"
    pd.DataFrame(
        [
            {
                "dataset": DATASET,
                "fly": "july_26_batch_1_rig_3",
                "fly_number": 1.0,  # float-parsed, as the real CSV is
                "FLY-State(1, 0, -1)": 0,
                "comment": "Bad",
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_selects_only_frozen_nonflagged_testing_rows(wide, flagged_csv):
    kept, dropped = fft.select_rows(
        wide, dataset=DATASET, pattern="*_rig_3", flagged_flies_csv=str(flagged_csv)
    )
    assert list(kept["fly"].unique()) == ["july_24_batch_2_rig_3"]
    assert sorted(kept["trial_label"]) == ["testing_1_3-octonol", "testing_8_3-octonol"]
    assert len(kept) == 2
    # The flagged fly is reported, not silently vanished.
    assert (DATASET, "july_26_batch_1_rig_3", "1") in dropped["flagged"]


def test_selection_is_unfrozen_so_the_loader_keeps_it(wide, flagged_csv):
    from fbpipe.utils.frozen_folders import drop_frozen

    kept, _ = fft.select_rows(
        wide, dataset=DATASET, pattern="*_rig_3", flagged_flies_csv=str(flagged_csv)
    )
    assert not kept["frozen"].any()
    # drop_frozen() runs unconditionally inside _load_wide_table; the whole
    # selection has to survive it or the figures come out empty.
    assert len(drop_frozen(kept)) == len(kept)


def test_no_flagged_csv_keeps_every_frozen_fly(wide):
    kept, dropped = fft.select_rows(
        wide, dataset=DATASET, pattern="*_rig_3", flagged_flies_csv=""
    )
    assert set(kept["fly"]) == {"july_24_batch_2_rig_3", "july_26_batch_1_rig_3"}
    assert dropped["flagged"] == []


def test_empty_selection_raises_rather_than_writing_nothing(wide, flagged_csv):
    with pytest.raises(fft.NoFrozenRowsError):
        fft.select_rows(
            wide, dataset=DATASET, pattern="*_rig_1", flagged_flies_csv=str(flagged_csv)
        )


def test_manifest_is_one_row_per_trial(wide, flagged_csv):
    kept, _ = fft.select_rows(
        wide, dataset=DATASET, pattern="*_rig_3", flagged_flies_csv=str(flagged_csv)
    )
    manifest = fft.build_manifest(kept)
    assert len(manifest) == len(kept)
    for col in ("dataset", "fly", "fly_number", "trial_label", "odor", "figure"):
        assert col in manifest.columns
    # One figure name per fly, matching the pipeline's Raw-Testing filename.
    assert set(manifest["figure"]) == {
        "july_24_batch_2_rig_3_fly1_testing_envelope_trials_by_odor_30_shifted.png"
    }


def test_panel_count_excludes_the_light_only_trial(wide, flagged_csv):
    """9 testing trials render 8 odor panels — the renderer drops testing_9."""
    kept, _ = fft.select_rows(
        wide, dataset=DATASET, pattern="*_rig_3", flagged_flies_csv=str(flagged_csv)
    )
    light = kept.iloc[[0]].copy()
    light["trial_label"] = "testing_9_lightonly"
    manifest = fft.build_manifest(pd.concat([kept, light], ignore_index=True))

    assert len(manifest) == 3                    # the light trial IS a trial...
    assert fft.odor_panel_count(manifest) == 2   # ...but not a panel
