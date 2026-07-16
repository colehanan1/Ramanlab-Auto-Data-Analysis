"""Freeze resolution + cache writing in run_workflows."""

import pandas as pd

import scripts.pipeline.run_workflows as rw
from fbpipe import freeze
from fbpipe.config import DatasetOverride


class _Tracking:
    """Stub of fbpipe.config.TrackingConfig -- build_fingerprint duck-types it."""

    apply_missing_frame_check = True
    max_missing_frames_per_trial = 5000
    max_missing_frames_pct_per_trial = 50.0


class _Settings:
    # Class-level defaults for the extra attributes _run_combined touches
    # (flagged-dir auto-discovery, non-reactive threshold, class2 limits).
    # None/neutral so they're no-ops unless a test overrides them.
    flagged_root = None
    flagged_secured_root = None
    non_reactive_span_px = 12.5
    class2_min = 0.0
    class2_max = 100.0

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


# --- FIX 2: a real round-trip through _freeze_fingerprint -----------------
#
# The tests above verify the write half (test_write_freeze_cache_*) and the
# read half (test_frozen_dataset_with_cache_resolves) SEPARATELY, each
# building its own fingerprint by hand. Neither exercises the exact pattern
# _run_combined relies on: _write_freeze_cache(..., fingerprint_for=lambda
# ds: _freeze_fingerprint(settings, ds, **kw)) followed by
# _resolve_frozen_slices(settings, roots, block, **kw) with the SAME kw. A
# mutation that made the two call sites' fingerprints diverge (e.g. one using
# base_measure_cols where the other uses wide_measure_cols) would slip past
# every test above while silently disabling freeze for good.


def test_freeze_round_trip_write_then_resolve(tmp_path):
    """_write_freeze_cache (via _freeze_fingerprint) seeds the cache; then
    _resolve_frozen_slices, given the identical kw, must find that dataset."""
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    out = tmp_path / "wide.csv"
    _rows("DS", 4).to_csv(out, index=False)
    kw = _fp_kw()

    rw._write_freeze_cache(
        s, "combined_base", str(out), [],
        fingerprint_for=lambda ds: rw._freeze_fingerprint(s, ds, **kw),
    )

    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **kw
    )
    assert "DS" in got
    assert got["DS"].own_max_len == 4


def test_freeze_round_trip_misses_when_resolve_kw_diverges(tmp_path):
    """Proves the round-trip test above is not vacuous: if the resolve-side kw
    differs from what was used to write the cache (e.g. a different
    measure_cols, mimicking a base_measure_cols/wide_measure_cols mixup), the
    dataset must NOT resolve -- it must fall back to a live rebuild."""
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    out = tmp_path / "wide.csv"
    _rows("DS", 4).to_csv(out, index=False)
    write_kw = _fp_kw()

    rw._write_freeze_cache(
        s, "combined_base", str(out), [],
        fingerprint_for=lambda ds: rw._freeze_fingerprint(s, ds, **write_kw),
    )

    resolve_kw = _fp_kw()
    resolve_kw["measure_cols"] = ["distance_percentage"]  # diverges from write_kw
    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **resolve_kw
    )
    assert got == {}


# --- FIX 1: build_wide_csv must honor the caller's --config ---------------
#
# build_wide_csv derives tracking_missing_frames / tracking_pct_missing /
# tracking_flagged from `load_settings(config_path or DEFAULT_CONFIG_PATH)`
# (scripts/analysis/envelope_combined.py:2633). None of _run_combined's three
# build_wide_csv call sites passed config_path, so a run started with
# `--config config_new.yaml` silently computed those columns from
# config/config.yaml's thresholds instead. This test proves _run_combined
# threads its config_path through to every build_wide_csv call by spying on
# the call rather than trying to compute real tracking columns (which needs
# video/RMS data this unit test doesn't have).


def test_run_combined_threads_config_path_to_build_wide_csv(tmp_path, monkeypatch):
    captured_config_paths = []

    def _spy_build_wide_csv(*args, **kwargs):
        captured_config_paths.append(kwargs.get("config_path"))

    monkeypatch.setattr(rw, "build_wide_csv", _spy_build_wide_csv)
    # wide_to_matrix runs unconditionally after the pair_groups build_wide_csv
    # call; stub it so the (nonexistent, since build_wide_csv is a spy) wide
    # CSV is never read from disk.
    monkeypatch.setattr(rw, "wide_to_matrix", lambda *a, **k: None)

    ds_root = tmp_path / "DS1"
    ds_root.mkdir()

    cfg = {
        "wide": {
            "roots": [str(ds_root)],
            "output_csv": str(tmp_path / "wide.csv"),
        },
        # Exercises the second build_wide_csv call site (_process_base_block).
        "combined_base": {
            "wide": {
                "output_csv": str(tmp_path / "wide_base.csv"),
            },
        },
        # Exercises the third build_wide_csv call site (pair_groups).
        "pair_groups": [
            {
                "name": "pair1",
                "datasets": [ds_root.name],
                "out_dir": str(tmp_path / "pair1"),
            }
        ],
    }
    s = _Settings(tmp_path, {}, ["DS1"])
    sentinel_config_path = tmp_path / "config_new.yaml"

    rw._run_combined(cfg, s, config_path=sentinel_config_path)

    # All three build_wide_csv call sites must have fired, and every one must
    # have received the caller's config_path -- not the DEFAULT_CONFIG_PATH
    # fallback build_wide_csv silently applies when config_path is None.
    assert len(captured_config_paths) == 3
    assert all(p == sentinel_config_path for p in captured_config_paths)
