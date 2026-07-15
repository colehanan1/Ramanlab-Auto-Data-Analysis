"""Freeze resolution + cache writing in run_workflows."""

import numpy as np
import pandas as pd
import pytest

import scripts.pipeline.run_workflows as rw
from fbpipe import freeze
from fbpipe.config import DatasetOverride


class _Tracking:
    """Stub of fbpipe.config.TrackingConfig -- build_fingerprint duck-types it."""

    apply_missing_frame_check = True
    max_missing_frames_per_trial = 5000
    max_missing_frames_pct_per_trial = 50.0


class _Settings:
    def __init__(self, tmp_path, overrides, datasets):
        self.cache_dir = str(tmp_path / "cache")
        self.dataset_overrides = overrides
        self.datasets = tuple(datasets)
        self.protocol = "v2"
        self.tracking = _Tracking()


def _rows(dataset, tl=5):
    rec = {"dataset": dataset, "fly": "f1", "trial_type": "testing", "trace_len": tl}
    for i in range(tl):
        rec[f"dir_val_{i}"] = float(i)
    return pd.DataFrame([rec])


def _fp_kw():
    return dict(
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=12.5,
        low_max_threshold_px=5.0,
        use_per_trial_baseline=False,
    )


def test_frozen_dataset_with_cache_resolves(tmp_path):
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    fp = freeze.build_fingerprint(
        protocol="v2", override=s.dataset_overrides["DS"], tracking=s.tracking,
        **_fp_kw()
    )
    freeze.save_slice(s.cache_dir, "combined_base", "DS", _rows("DS"), fp)

    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **_fp_kw()
    )
    assert "DS" in got
    assert got["DS"].own_max_len == 5


def test_unfrozen_dataset_is_not_resolved_even_with_a_cache(tmp_path):
    """A cache exists but the dataset is not frozen -- it must be derived live."""
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=False)}, ["DS"])
    fp = freeze.build_fingerprint(
        protocol="v2", override=s.dataset_overrides["DS"], tracking=s.tracking,
        **_fp_kw()
    )
    freeze.save_slice(s.cache_dir, "combined_base", "DS", _rows("DS"), fp)

    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **_fp_kw()
    )
    assert got == {}


def test_frozen_but_cache_miss_auto_rebuilds(tmp_path):
    """No cache -> absent from frozen_slices -> root gets walked. That IS the
    auto-rebuild."""
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **_fp_kw()
    )
    assert got == {}


def test_frozen_but_config_drift_auto_rebuilds(tmp_path):
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    fp = freeze.build_fingerprint(
        protocol="v2", override=s.dataset_overrides["DS"], tracking=s.tracking,
        **_fp_kw()
    )
    freeze.save_slice(s.cache_dir, "combined_base", "DS", _rows("DS"), fp)

    kw = _fp_kw()
    kw["non_reactive_threshold"] = 10.0  # drift
    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **kw
    )
    assert got == {}


def test_thaw_ignores_freeze(tmp_path):
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    fp = freeze.build_fingerprint(
        protocol="v2", override=s.dataset_overrides["DS"], tracking=s.tracking,
        **_fp_kw()
    )
    freeze.save_slice(s.cache_dir, "combined_base", "DS", _rows("DS"), fp)

    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=["DS"], thaw_all=False, **_fp_kw()
    )
    assert got == {}


def test_write_freeze_cache_stores_each_dataset_slice(tmp_path):
    s = _Settings(tmp_path, {}, ["A", "B"])
    out = tmp_path / "wide.csv"
    pd.concat([_rows("A", 4), _rows("B", 6)], ignore_index=True).to_csv(out, index=False)

    rw._write_freeze_cache(
        s, "combined_base", str(out), [],
        fingerprint_for=lambda ds: freeze.build_fingerprint(
            protocol="v2", override=DatasetOverride(), tracking=s.tracking,
            **_fp_kw()
        ),
    )
    fp = freeze.build_fingerprint(
        protocol="v2", override=DatasetOverride(), tracking=s.tracking, **_fp_kw()
    )
    a = freeze.load_slice(s.cache_dir, "combined_base", "A", fp)
    b = freeze.load_slice(s.cache_dir, "combined_base", "B", fp)
    assert a is not None and b is not None
    assert a.own_max_len == 4
    assert b.own_max_len == 6
    assert set(a.rows["dataset"]) == {"A"}   # slices must not bleed into each other
    assert set(b.rows["dataset"]) == {"B"}


def test_write_freeze_cache_includes_extra_trial_exports(tmp_path):
    """A slice must hold ALL trial types, or freezing loses the training rows."""
    s = _Settings(tmp_path, {}, ["A"])
    main = tmp_path / "wide.csv"
    train = tmp_path / "wide_training.csv"
    _rows("A", 4).to_csv(main, index=False)
    tr = _rows("A", 4)
    tr["trial_type"] = "training"
    tr.to_csv(train, index=False)

    rw._write_freeze_cache(
        s, "combined_base", str(main), [str(train)],
        fingerprint_for=lambda ds: freeze.build_fingerprint(
            protocol="v2", override=DatasetOverride(), tracking=s.tracking,
            **_fp_kw()
        ),
    )
    fp = freeze.build_fingerprint(
        protocol="v2", override=DatasetOverride(), tracking=s.tracking, **_fp_kw()
    )
    got = freeze.load_slice(s.cache_dir, "combined_base", "A", fp)
    assert set(got.rows["trial_type"].str.lower()) == {"testing", "training"}
