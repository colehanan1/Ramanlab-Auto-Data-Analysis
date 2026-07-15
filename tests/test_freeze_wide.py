"""build_wide_csv splices frozen slices without walking their roots."""

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev


def _make_dataset(root, n_samples, fly="october_01_fly1"):
    """One testing trial of *n_samples* points, in the layout build_wide_csv walks."""
    out = root / fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0, 100, n_samples, dtype=float)
    pd.DataFrame({"envelope_of_rms": values}).to_csv(
        out / f"{fly}_testing_1_angle_distance_rms_envelope.csv", index=False
    )
    return root


def _build(roots, out_csv, **kw):
    ec.build_wide_csv(
        [str(r) for r in roots], str(out_csv), measure_cols=["envelope_of_rms"], **kw
    )
    return pd.read_csv(out_csv)


@pytest.fixture(autouse=True)
def _v2():
    ev.set_protocol("v2")


def test_frozen_slice_round_trips_identically_when_frozen_is_longer(tmp_path):
    """THE load-bearing test.

    The frozen dataset is given a LONGER trace than the live one on purpose. With
    equal lengths this test would pass even if the own_max_len fold were missing
    entirely -- the global max would be correct by accident. Unequal lengths are
    what make it bite.
    """
    live = _make_dataset(tmp_path / "LIVE", 8)
    frozen = _make_dataset(tmp_path / "FROZEN", 20)  # LONGER than live

    baseline_csv = tmp_path / "baseline.csv"
    baseline = _build([live, frozen], baseline_csv)

    # Seed the cache from the baseline, exactly as Task 4's caller will.
    frozen_rows = baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True)
    own = int(frozen_rows["trace_len"].max())
    assert own == 20

    spliced_csv = tmp_path / "spliced.csv"
    spliced = _build(
        [live, frozen], spliced_csv, frozen_slices={"FROZEN": (frozen_rows, own)}
    )

    # Byte-identical output, frozen or not.
    assert baseline_csv.read_bytes() != b""
    pd.testing.assert_frame_equal(
        baseline.sort_values(["dataset", "fly"]).reset_index(drop=True),
        spliced.sort_values(["dataset", "fly"]).reset_index(drop=True),
    )
    # The global max grew to the FROZEN dataset's length, not the live one's.
    assert "dir_val_19" in spliced.columns
    assert "dir_val_20" not in spliced.columns


def test_frozen_rows_not_truncated_when_no_live_dataset_is_as_long(tmp_path):
    """Directly pins the failure the own_max_len fold prevents: silent chopping."""
    live = _make_dataset(tmp_path / "LIVE", 5)
    frozen = _make_dataset(tmp_path / "FROZEN", 30)

    baseline = _build([live, frozen], tmp_path / "b.csv")
    frozen_rows = baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True)

    spliced = _build(
        [live, frozen], tmp_path / "s.csv", frozen_slices={"FROZEN": (frozen_rows, 30)}
    )
    row = spliced[spliced["dataset"] == "FROZEN"].iloc[0]
    # Sample 29 is real data, not padding, and must survive.
    assert not pd.isna(row["dir_val_29"])
    assert int(row["trace_len"]) == 30


def test_frozen_shorter_than_live_is_nan_padded(tmp_path):
    live = _make_dataset(tmp_path / "LIVE", 25)
    frozen = _make_dataset(tmp_path / "FROZEN", 6)

    baseline = _build([live, frozen], tmp_path / "b.csv")
    frozen_rows = baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True)

    spliced = _build(
        [live, frozen], tmp_path / "s.csv", frozen_slices={"FROZEN": (frozen_rows, 6)}
    )
    row = spliced[spliced["dataset"] == "FROZEN"].iloc[0]
    assert not pd.isna(row["dir_val_5"])   # last real sample
    assert pd.isna(row["dir_val_24"])      # padding out to the live max
    assert int(row["trace_len"]) == 6


def test_frozen_root_is_never_walked(tmp_path):
    """The entire performance claim. If this passes vacuously the feature is a lie."""
    live = _make_dataset(tmp_path / "LIVE", 8)
    frozen = _make_dataset(tmp_path / "FROZEN", 8)

    baseline = _build([live, frozen], tmp_path / "b.csv")
    frozen_rows = baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True)

    walked = []
    real_iterdir = ec.Path.iterdir

    def _spy(self):
        walked.append(str(self))
        return real_iterdir(self)

    ec.Path.iterdir = _spy
    try:
        _build(
            [live, frozen], tmp_path / "s.csv",
            frozen_slices={"FROZEN": (frozen_rows, 8)},
        )
    finally:
        ec.Path.iterdir = real_iterdir

    assert not any("FROZEN" in w for w in walked), f"frozen root was walked: {walked}"
    # Guard against the spy simply never firing -- the live root MUST be walked.
    assert any("LIVE" in w for w in walked), "spy never fired; test proves nothing"


def test_all_datasets_frozen_does_not_raise(tmp_path):
    """items is empty when every root is frozen. The pre-existing guard at
    envelope_combined.py:2733 raises RuntimeError on empty items -- it must not
    fire when frozen rows are present."""
    a = _make_dataset(tmp_path / "A", 8)
    b = _make_dataset(tmp_path / "B", 12)

    baseline = _build([a, b], tmp_path / "b.csv")
    rows_a = baseline[baseline["dataset"] == "A"].reset_index(drop=True)
    rows_b = baseline[baseline["dataset"] == "B"].reset_index(drop=True)

    out = _build(
        [a, b], tmp_path / "s.csv",
        frozen_slices={"A": (rows_a, 8), "B": (rows_b, 12)},
    )
    assert set(out["dataset"]) == {"A", "B"}
    assert len(out) == len(baseline)


def test_no_eligible_data_and_no_frozen_still_raises(tmp_path):
    """The empty-items guard must survive for the case it was written for."""
    empty = tmp_path / "EMPTY"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No eligible"):
        _build([empty], tmp_path / "s.csv")


def test_frozen_slices_none_is_todays_behavior(tmp_path):
    """Default off: absent frozen_slices must not perturb output."""
    live = _make_dataset(tmp_path / "LIVE", 8)
    a = _build([live], tmp_path / "a.csv")
    b = _build([live], tmp_path / "b.csv", frozen_slices=None)
    pd.testing.assert_frame_equal(a, b)


def test_frozen_training_rows_route_to_extra_export(tmp_path):
    """A cached slice holds ALL trial types; the splice must route them the same
    way live rows are routed (testing -> main, training -> extra export)."""
    root = tmp_path / "DS"
    fly = "october_01_fly1"
    out = root / fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True, exist_ok=True)
    for tt in ("testing", "training"):
        pd.DataFrame({"envelope_of_rms": np.linspace(0, 100, 9)}).to_csv(
            out / f"{fly}_{tt}_1_angle_distance_rms_envelope.csv", index=False
        )

    main_csv = tmp_path / "main.csv"
    train_csv = tmp_path / "train.csv"
    ec.build_wide_csv(
        [str(root)], str(main_csv),
        measure_cols=["envelope_of_rms"],
        extra_trial_exports={"training": str(train_csv)},
    )
    all_rows = pd.concat(
        [pd.read_csv(main_csv), pd.read_csv(train_csv)], ignore_index=True
    )
    assert set(all_rows["trial_type"].str.lower()) == {"testing", "training"}

    m2 = tmp_path / "main2.csv"
    t2 = tmp_path / "train2.csv"
    ec.build_wide_csv(
        [str(root)], str(m2),
        measure_cols=["envelope_of_rms"],
        extra_trial_exports={"training": str(t2)},
        frozen_slices={"DS": (all_rows, 9)},
    )
    assert set(pd.read_csv(m2)["trial_type"].str.lower()) == {"testing"}
    assert set(pd.read_csv(t2)["trial_type"].str.lower()) == {"training"}
