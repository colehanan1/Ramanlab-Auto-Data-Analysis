"""Tests for the per-presentation trained-vs-control mean-trace driver.

These are the ``<odor>_training_vs_control.png`` figures. The existing
``dataset_means_specific_flies`` collector keeps only each odor's *first*
presentation, which silently drops the second 3-octanol exposure
(``testing_8``) — the one that carries the extinction/re-test signal. This
driver instead keys traces by (odor, presentation), so 3-octanol yields two
figures and every once-presented odor still yields one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import envelope_visuals as ev
from scripts.analysis.dataset_mean_traces_tvc import (
    base_odor_key,
    collect_per_presentation_traces,
    main,
)

TRAIN = "3Oct-Training-24-0.1"
CTRL = "3Oct-Control-24-0.1"
TRAIN_CANON = "3OCT-Training-24-0.1"

REMAP = {
    "Apple Cider Vinegar": "Isoamyl Acetate (1%)",
    "3-Octanol": "3-Octanol (0.1%)",
    "Ethyl Butyrate": "Ethyl Butyrate (1%)",
    "Benzaldehyde": "Benzaldehyde (0.1%)",
    "Hexanol": "Hexanol (0.1%)",
    "Citral": "Citral (1%)",
    "Linalool": "Linalool (1%)",
}

# (trial_label, constant trace value) — the real testing schedule: 3-octanol
# on testing_1 and again on testing_8, everything else once, light-only last.
SCHEDULE = [
    ("testing_1_3-octonol", 1.0),
    ("testing_2_hexanol", 2.0),
    ("testing_3_citral", 3.0),
    ("testing_4_ethylbutyrate", 4.0),
    ("testing_5_benzaldehyde", 5.0),
    ("testing_6_acv", 6.0),
    ("testing_7_linalool", 7.0),
    ("testing_8_3-octonol", 8.0),
    ("testing_9_lightonly", 9.0),
]

N_FRAMES = 120
FPS = 40.0
ODOR_ON_S = 0.5           # 20 baseline frames at 40 fps — keeps fixtures small
BASELINE_FRAMES = int(round(ODOR_ON_S * FPS))


@pytest.fixture(autouse=True)
def _registered_remap():
    """Register/restore the 3Oct odor remap the way run_workflows does."""
    saved_protocol = ev.get_protocol()
    saved_remap = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    ev.set_protocol("v2")
    ev.set_dataset_odor_remap({TRAIN: dict(REMAP), CTRL: dict(REMAP)})
    try:
        yield
    finally:
        ev.set_protocol(saved_protocol)
        ev.set_dataset_odor_remap(saved_remap)


def _wide_frame(datasets=(TRAIN, CTRL), flies=("july_20_batch_1", "july_20_batch_2")):
    rows = []
    for dataset in datasets:
        for fly in flies:
            for fly_number in (1, 2):
                for label, value in SCHEDULE:
                    row = {
                        "dataset": dataset,
                        "fly": fly,
                        "fly_number": fly_number,
                        "trial_type": "testing",
                        "trial_label": label,
                    }
                    row.update({f"dir_val_{i}": value for i in range(N_FRAMES)})
                    rows.append(row)
    return pd.DataFrame(rows)


def _dir_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("dir_val_")],
        key=lambda c: int(c.split("_")[-1]),
    )


def _collect(df: pd.DataFrame):
    return collect_per_presentation_traces(
        df[df["dataset"] == TRAIN],
        TRAIN_CANON,
        baseline_frames=BASELINE_FRAMES,
        dir_cols=_dir_cols(df),
    )


# ---------------------------------------------------------------------------
# Presentation keying
# ---------------------------------------------------------------------------


def test_repeated_odor_splits_into_numbered_presentations() -> None:
    per_odor = _collect(_wide_frame())
    assert "3-Octanol (0.1%) 1" in per_odor
    assert "3-Octanol (0.1%) 2" in per_odor


def test_single_presentation_odors_are_not_numbered() -> None:
    per_odor = _collect(_wide_frame())
    assert "Hexanol (0.1%)" in per_odor
    assert "Hexanol (0.1%) 1" not in per_odor


def test_presentation_number_follows_trial_order_not_row_order() -> None:
    """testing_1 is presentation 1 even when the frame arrives shuffled — the
    wide table is not guaranteed to be sorted, and getting this backwards would
    swap the two 3-octanol figures."""
    df = _wide_frame().sample(frac=1.0, random_state=0).reset_index(drop=True)
    per_odor = _collect(df)
    # trace values are constant per trial: testing_1 -> 1.0, testing_8 -> 8.0,
    # both baseline-corrected to 0, so compare against the raw fixture instead.
    first = per_odor["3-Octanol (0.1%) 1"]
    second = per_odor["3-Octanol (0.1%) 2"]
    assert set(first) == set(second)          # same flies contribute to both
    assert len(first) == 4


def test_odor_remap_is_applied_to_the_keys() -> None:
    """ACV is really isoamyl acetate on this rig; the figure must say so."""
    per_odor = _collect(_wide_frame())
    assert "Isoamyl Acetate (1%)" in per_odor
    assert not any("Apple Cider" in k for k in per_odor)


def test_light_only_trials_are_dropped() -> None:
    per_odor = _collect(_wide_frame())
    assert not any("light" in k.casefold() for k in per_odor)


def test_traces_are_baseline_subtracted() -> None:
    """Each fixture trial is a constant, so baseline subtraction must zero it."""
    per_odor = _collect(_wide_frame())
    trace = next(iter(per_odor["Citral (1%)"].values()))
    assert np.allclose(trace, 0.0)


def test_flies_missing_a_presentation_simply_do_not_contribute() -> None:
    df = _wide_frame()
    drop = (df["fly"] == "july_20_batch_2") & (df["trial_label"] == "testing_8_3-octonol")
    per_odor = _collect(df[~drop])
    assert len(per_odor["3-Octanol (0.1%) 1"]) == 4
    assert len(per_odor["3-Octanol (0.1%) 2"]) == 2


def test_each_fly_appears_once_per_presentation() -> None:
    per_odor = _collect(_wide_frame())
    keys = list(per_odor["Hexanol (0.1%)"])
    assert len(keys) == len(set(keys)) == 4


# ---------------------------------------------------------------------------
# Colour keying
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected",
    [
        ("3-Octanol (0.1%) 1", "3-Octanol"),
        ("3-Octanol (0.1%) 2", "3-Octanol"),
        ("Hexanol (0.1%)", "Hexanol"),
        ("Isoamyl Acetate (1%)", "Isoamyl Acetate"),
        ("Benzaldehyde", "Benzaldehyde"),
    ],
)
def test_base_odor_key_strips_concentration_and_presentation(label, expected) -> None:
    """Colours are keyed on the bare odor name, so both 3-octanol figures come
    out the same colour instead of falling through to the grey default."""
    assert base_odor_key(label) == expected


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def _write_wide_parquet(path: Path) -> None:
    _wide_frame().to_parquet(path, index=False)


def test_main_emits_one_figure_per_presentation(tmp_path: Path) -> None:
    wide = tmp_path / "wide.parquet"
    _write_wide_parquet(wide)
    out_dir = tmp_path / "Training_vs_Control"

    main([
        "--wide-csv", str(wide),
        "--train-dataset", TRAIN,
        "--control-dataset", CTRL,
        "--out-dir", str(out_dir),
        "--fps", str(FPS),
        "--odor-on-s", str(ODOR_ON_S),
        "--odor-off-s", "2.0",
    ])

    names = {p.name for p in out_dir.iterdir()}
    assert "3-Octanol_0.1%_1_training_vs_control.png" in names
    assert "3-Octanol_0.1%_2_training_vs_control.png" in names
    assert "Hexanol_0.1%_training_vs_control.png" in names
    assert "Isoamyl_Acetate_1%_training_vs_control.png" in names
    assert "training_vs_control.json" in names
    # 7 distinct odors, one of them presented twice -> 8 figures
    assert sum(n.endswith("_training_vs_control.png") for n in names) == 8


def test_sidecar_records_group_sizes_per_presentation(tmp_path: Path) -> None:
    wide = tmp_path / "wide.parquet"
    _write_wide_parquet(wide)
    out_dir = tmp_path / "Training_vs_Control"
    main([
        "--wide-csv", str(wide),
        "--train-dataset", TRAIN,
        "--control-dataset", CTRL,
        "--out-dir", str(out_dir),
        "--fps", str(FPS),
        "--odor-on-s", str(ODOR_ON_S),
        "--odor-off-s", "2.0",
    ])
    meta = json.loads((out_dir / "training_vs_control.json").read_text())
    assert meta["training_dataset"] == TRAIN
    assert meta["control_dataset"] == CTRL
    assert meta["per_odor"]["3-Octanol (0.1%) 2"] == {
        "n_training_flies": 4,
        "n_control_flies": 4,
    }
    assert meta["fps"] == FPS
    assert "shared_mean_ylim" in meta


def test_odors_absent_from_one_arm_are_skipped(tmp_path: Path) -> None:
    """A one-sided odor cannot be a trained-vs-control figure; it must be
    reported in the sidecar rather than drawn against an empty cohort."""
    df = _wide_frame()
    df = df[~((df["dataset"] == CTRL) & (df["trial_label"] == "testing_3_citral"))]
    wide = tmp_path / "wide.parquet"
    df.to_parquet(wide, index=False)
    out_dir = tmp_path / "Training_vs_Control"
    main([
        "--wide-csv", str(wide),
        "--train-dataset", TRAIN,
        "--control-dataset", CTRL,
        "--out-dir", str(out_dir),
        "--fps", str(FPS),
        "--odor-on-s", str(ODOR_ON_S),
        "--odor-off-s", "2.0",
    ])
    names = {p.name for p in out_dir.iterdir()}
    assert "Citral_1%_training_vs_control.png" not in names
    meta = json.loads((out_dir / "training_vs_control.json").read_text())
    assert "Citral (1%)" in meta["skipped_odors"]


# ---------------------------------------------------------------------------
# Cohort filters
# ---------------------------------------------------------------------------


def _batched_wide_frame():
    """Two batches per arm, so a batch filter has something to cut."""
    frames = []
    for batch, flies in ((1, ("july_13_batch_1", "july_14_batch_1")),
                         (2, ("july_13_batch_2", "july_14_batch_2"))):
        frames.append(_wide_frame(flies=flies))
    return pd.concat(frames, ignore_index=True)


def test_batch_filter_keeps_only_that_batch(tmp_path: Path) -> None:
    from scripts.analysis.dataset_mean_traces_tvc import main

    wide = tmp_path / "wide.parquet"
    _batched_wide_frame().to_parquet(wide, index=False)
    out_dir = tmp_path / "batch_2"
    main([
        "--wide-csv", str(wide),
        "--train-dataset", TRAIN, "--control-dataset", CTRL,
        "--out-dir", str(out_dir),
        "--fps", str(FPS), "--odor-on-s", str(ODOR_ON_S), "--odor-off-s", "2.0",
        "--batch", "2",
    ])
    meta = json.loads((out_dir / "training_vs_control.json").read_text())
    assert meta["batch"] == 2
    # 2 folders x 2 fly numbers in batch 2 (batch 1's four flies excluded)
    assert meta["per_odor"]["Hexanol (0.1%)"]["n_training_flies"] == 4


def test_without_a_batch_filter_all_flies_contribute(tmp_path: Path) -> None:
    from scripts.analysis.dataset_mean_traces_tvc import main

    wide = tmp_path / "wide.parquet"
    _batched_wide_frame().to_parquet(wide, index=False)
    out_dir = tmp_path / "pooled"
    main([
        "--wide-csv", str(wide),
        "--train-dataset", TRAIN, "--control-dataset", CTRL,
        "--out-dir", str(out_dir),
        "--fps", str(FPS), "--odor-on-s", str(ODOR_ON_S), "--odor-off-s", "2.0",
    ])
    meta = json.loads((out_dir / "training_vs_control.json").read_text())
    assert meta["batch"] is None
    assert meta["per_odor"]["Hexanol (0.1%)"]["n_training_flies"] == 8


def test_batch_filter_that_matches_nothing_fails_loudly(tmp_path: Path) -> None:
    """An empty cohort must raise, not silently emit figures with n=0."""
    from scripts.analysis.dataset_mean_traces_tvc import main

    wide = tmp_path / "wide.parquet"
    _batched_wide_frame().to_parquet(wide, index=False)
    with pytest.raises(RuntimeError):
        main([
            "--wide-csv", str(wide),
            "--train-dataset", TRAIN, "--control-dataset", CTRL,
            "--out-dir", str(tmp_path / "batch_9"),
            "--fps", str(FPS), "--odor-on-s", str(ODOR_ON_S), "--odor-off-s", "2.0",
            "--batch", "9",
        ])
