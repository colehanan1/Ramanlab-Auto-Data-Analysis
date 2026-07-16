"""Freeze end-to-end: derive live, freeze, re-run, output is unchanged."""

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev
from fbpipe import freeze
from fbpipe.config import DatasetOverride


class _Tracking:
    """Stub of fbpipe.config.TrackingConfig -- build_fingerprint duck-types it.

    build_wide_csv reads settings.tracking internally (envelope_combined.py:2622)
    to derive the tracking_* columns, so it is part of the fingerprint.
    """

    apply_missing_frame_check = True
    max_missing_frames_per_trial = 5000
    max_missing_frames_pct_per_trial = 50.0


def _make(root, n, fly="october_01_fly1"):
    out = root / fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"envelope_of_rms": np.linspace(0, 100, n)}).to_csv(
        out / f"{fly}_testing_1_angle_distance_rms_envelope.csv", index=False
    )
    return root


@pytest.fixture(autouse=True)
def _v2():
    ev.set_protocol("v2")


def test_freeze_then_rerun_is_byte_identical(tmp_path):
    """The user-facing promise, exercised through the real cache.

    FROZEN is deliberately longer than LIVE so the own_max_len fold is genuinely
    exercised rather than correct by accident.
    """
    live = _make(tmp_path / "LIVE", 9)
    frozen = _make(tmp_path / "FROZEN", 21)
    cache = tmp_path / "cache"

    fp_kw = dict(
        protocol="v2",
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=None,
        low_max_threshold_px=ec.LOW_MAX_FLAG_THRESHOLD_PX,
        use_per_trial_baseline=False,
        trial_type_filter=None,
        override=DatasetOverride(),
        tracking=_Tracking(),
    )
    fp = freeze.build_fingerprint(**fp_kw)

    # Pass 1: fully live.
    first = tmp_path / "first.csv"
    ec.build_wide_csv(
        [str(live), str(frozen)], str(first), measure_cols=["envelope_of_rms"]
    )
    baseline = pd.read_csv(first)

    # Cache FROZEN's slice, as the pipeline does after a live derivation.
    freeze.save_slice(
        cache, "wide", "FROZEN",
        baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True), fp,
    )

    # Pass 2: FROZEN spliced from cache.
    got = freeze.load_slice(cache, "wide", "FROZEN", fp)
    assert got is not None and got.own_max_len == 21

    second = tmp_path / "second.csv"
    ec.build_wide_csv(
        [str(live), str(frozen)], str(second),
        measure_cols=["envelope_of_rms"],
        frozen_slices={"FROZEN": (got.rows, got.own_max_len)},
    )

    assert first.read_bytes() == second.read_bytes()

    pd.testing.assert_frame_equal(
        baseline.sort_values(["dataset", "fly"]).reset_index(drop=True),
        pd.read_csv(second).sort_values(["dataset", "fly"]).reset_index(drop=True),
    )


def test_deleting_the_cache_is_safe(tmp_path):
    """The cache is a derived artifact: deleting it must only cost time."""
    import shutil

    live = _make(tmp_path / "LIVE", 9)
    frozen = _make(tmp_path / "FROZEN", 21)
    cache = tmp_path / "cache"
    fp = freeze.build_fingerprint(
        protocol="v2", measure_cols=["envelope_of_rms"], fps_fallback=40.0,
        distance_limits=None, non_reactive_threshold=None,
        low_max_threshold_px=ec.LOW_MAX_FLAG_THRESHOLD_PX,
        use_per_trial_baseline=False, trial_type_filter=None,
        override=DatasetOverride(),
        tracking=_Tracking(),
    )
    out1 = tmp_path / "a.csv"
    ec.build_wide_csv([str(live), str(frozen)], str(out1), measure_cols=["envelope_of_rms"])
    base = pd.read_csv(out1)
    freeze.save_slice(
        cache, "wide", "FROZEN", base[base["dataset"] == "FROZEN"].reset_index(drop=True), fp
    )
    shutil.rmtree(cache)
    assert freeze.load_slice(cache, "wide", "FROZEN", fp) is None

    out2 = tmp_path / "b.csv"
    ec.build_wide_csv([str(live), str(frozen)], str(out2), measure_cols=["envelope_of_rms"])
    pd.testing.assert_frame_equal(base, pd.read_csv(out2))
