"""build_wide_csv marks frozen experiment folders without dropping their rows.

The user's rule: frozen folders stay in the CSV, out of the plots. So
``build_wide_csv`` writes every row it always wrote and adds a ``frozen``
boolean column; the figure layer is what filters on it.

The caller resolves WHICH folders are frozen (run_workflows, via
fbpipe.utils.frozen_folders) and passes a ``{dataset: {folder, ...}}`` map,
mirroring how ``frozen_slices`` is already threaded in. envelope_combined
stays free of config policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev


@pytest.fixture(autouse=True)
def _v2():
    ev.set_protocol("v2")


def _make_batch(root, fly, n_samples=8):
    out = root / fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"envelope_of_rms": np.linspace(0, 100, n_samples, dtype=float)}
    ).to_csv(out / f"{fly}_testing_1_angle_distance_rms_envelope.csv", index=False)
    return root


def _build(roots, out_csv, **kw):
    ec.build_wide_csv(
        [str(r) for r in roots], str(out_csv), measure_cols=["envelope_of_rms"], **kw
    )
    return pd.read_csv(out_csv)


def test_frozen_folder_row_is_still_written(tmp_path):
    """THE point of the design: freezing must not delete data from the CSV."""
    root = tmp_path / "Hex-Training-24-0.1"
    _make_batch(root, "may_22_batch_1_rig_2")
    _make_batch(root, "august_10_batch_1_rig_2")

    df = _build(
        [root],
        tmp_path / "wide.csv",
        frozen_folders={"Hex-Training-24-0.1": {"may_22_batch_1_rig_2"}},
    )
    assert set(df["fly"]) == {"may_22_batch_1_rig_2", "august_10_batch_1_rig_2"}


def test_frozen_column_marks_only_the_frozen_folder(tmp_path):
    root = tmp_path / "Hex-Training-24-0.1"
    _make_batch(root, "may_22_batch_1_rig_2")
    _make_batch(root, "august_10_batch_1_rig_2")

    df = _build(
        [root],
        tmp_path / "wide.csv",
        frozen_folders={"Hex-Training-24-0.1": {"may_22_batch_1_rig_2"}},
    )
    frozen_by_fly = dict(zip(df["fly"], df["frozen"]))
    assert frozen_by_fly == {
        "may_22_batch_1_rig_2": True,
        "august_10_batch_1_rig_2": False,
    }


def test_frozen_column_is_present_and_all_false_when_nothing_is_frozen(tmp_path):
    """The column is unconditional, so downstream readers never branch on its
    existence -- and an all-live run is self-describing rather than silent."""
    root = tmp_path / "EB-Training-24-1"
    _make_batch(root, "july_14_batch_2_rig_2")
    df = _build([root], tmp_path / "wide.csv")
    assert list(df["frozen"]) == [False]


def test_frozen_folders_are_scoped_per_dataset(tmp_path):
    """Batch folder names repeat across datasets; a name frozen under one must
    not freeze the same-named folder under another."""
    train = tmp_path / "EB-Training-24-1"
    ctrl = tmp_path / "EB-Control-24-1"
    _make_batch(train, "july_14_batch_2_rig_2")
    _make_batch(ctrl, "july_14_batch_2_rig_2")

    df = _build(
        [train, ctrl],
        tmp_path / "wide.csv",
        frozen_folders={"EB-Training-24-1": {"july_14_batch_2_rig_2"}},
    )
    got = dict(zip(df["dataset"], df["frozen"]))
    assert got == {"EB-Training-24-1": True, "EB-Control-24-1": False}


def test_trial_type_export_carries_the_frozen_column(tmp_path):
    """The training-trial export is a second wide file fed to its own figure
    stage; it needs the same marking or training plots keep the old flies."""
    root = tmp_path / "Hex-Training-24-0.1"
    out = root / "may_22_batch_1_rig_2" / "angle_distance_rms_envelope"
    out.mkdir(parents=True)
    pd.DataFrame({"envelope_of_rms": np.linspace(0, 100, 8)}).to_csv(
        out / "may_22_batch_1_rig_2_training_1_angle_distance_rms_envelope.csv",
        index=False,
    )
    training_csv = tmp_path / "wide_training.csv"
    _build(
        [root],
        tmp_path / "wide.csv",
        frozen_folders={"Hex-Training-24-0.1": {"may_22_batch_1_rig_2"}},
        extra_trial_exports={"training": training_csv},
    )
    assert list(pd.read_csv(training_csv)["frozen"]) == [True]


def test_spliced_frozen_dataset_rows_read_as_live(tmp_path):
    """A data-frozen dataset's cached slice predates this column. reindex fills
    NaN; NaN must not read as frozen, or every cached dataset would vanish from
    the figures the first time this ships."""
    live = tmp_path / "LIVE"
    _make_batch(live, "august_10_batch_1")
    cached = pd.DataFrame(
        [{"dataset": "CACHED", "fly": "april_11_batch_2", "fly_number": "1",
          "trial_type": "testing", "trial_label": "testing_1", "trace_len": 4,
          "dir_val_0": 1.0, "dir_val_1": 2.0, "dir_val_2": 3.0, "dir_val_3": 4.0}]
    )
    df = _build(
        [live],
        tmp_path / "wide.csv",
        frozen_slices={"CACHED": (cached, 4)},
    )
    assert df.loc[df["dataset"] == "CACHED", "frozen"].tolist() == [False]
