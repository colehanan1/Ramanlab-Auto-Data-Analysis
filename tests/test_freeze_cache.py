"""Freeze cache: fingerprint, own_max_len, save/load round-trip."""

import numpy as np
import pandas as pd
import pytest

from fbpipe import freeze
from fbpipe.config import DatasetOverride


def _rows(trace_lens, n_val_cols):
    """Build a wide-CSV-shaped frame: trace_len + NaN-padded dir_val_* columns."""
    recs = []
    for tl in trace_lens:
        rec = {"dataset": "DS", "trial_type": "testing", "trace_len": tl}
        for i in range(n_val_cols):
            rec[f"dir_val_{i}"] = float(i) if i < tl else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def _fp(**kw):
    base = dict(
        protocol="v2",
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=12.5,
        low_max_threshold_px=5.0,
        use_per_trial_baseline=False,
        override=DatasetOverride(),
    )
    base.update(kw)
    return freeze.build_fingerprint(**base)


def test_own_max_len_is_max_trace_len(tmp_path):
    assert freeze.own_max_len(_rows([10, 7, 3], 10)) == 10


def test_own_max_len_clamped_to_present_columns():
    """A truncated row can carry trace_len > the dir_val columns it actually has.
    own_max_len must not over-report, or reload pads to a width with no data."""
    rows = _rows([10], 10)
    rows["trace_len"] = 99  # claims 99 samples, only 10 dir_val columns exist
    assert freeze.own_max_len(rows) == 10


def test_save_load_round_trip(tmp_path):
    rows = _rows([10, 7], 10)
    fp = _fp()
    freeze.save_slice(tmp_path, "combined_base", "DS", rows, fp)
    got = freeze.load_slice(tmp_path, "combined_base", "DS", fp)
    assert got is not None
    assert got.own_max_len == 10
    pd.testing.assert_frame_equal(
        got.rows.reset_index(drop=True), rows.reset_index(drop=True)
    )


def test_load_miss_returns_none(tmp_path):
    assert freeze.load_slice(tmp_path, "combined_base", "NOPE", _fp()) is None


def test_fingerprint_drift_returns_none(tmp_path):
    """A changed analysis parameter must invalidate -- else one CSV mixes two
    parameterizations."""
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    drifted = _fp(non_reactive_threshold=10.0)
    assert freeze.load_slice(tmp_path, "combined_base", "DS", drifted) is None


def test_protocol_drift_returns_none(tmp_path):
    """legacy and v2 have different column sets -- a schema mismatch, not a value
    drift."""
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    assert freeze.load_slice(tmp_path, "combined_base", "DS", _fp(protocol="legacy")) is None


def test_odor_remap_drift_returns_none(tmp_path):
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    drifted = _fp(override=DatasetOverride(odor_remap={"Citral": "Yeast"}))
    assert freeze.load_slice(tmp_path, "combined_base", "DS", drifted) is None


def test_figure_output_subdir_does_not_invalidate(tmp_path):
    """figure_output_subdir routes figures; it cannot change a row value, so it
    must NOT invalidate a data cache."""
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    same = _fp(override=DatasetOverride(figure_output_subdir="RandomPanel"))
    assert freeze.load_slice(tmp_path, "combined_base", "DS", same) is not None


def test_blocks_do_not_collide(tmp_path):
    """combined_base and distance_base have different measure_cols and so
    different rows. Serving one for the other is a silent data corruption."""
    cb = _rows([5], 5)
    db = _rows([9], 9)
    freeze.save_slice(tmp_path, "combined_base", "DS", cb, _fp(measure_cols=["combined_pct"]))
    freeze.save_slice(tmp_path, "distance_base", "DS", db, _fp(measure_cols=["distance_percentage"]))
    got_cb = freeze.load_slice(tmp_path, "combined_base", "DS", _fp(measure_cols=["combined_pct"]))
    got_db = freeze.load_slice(tmp_path, "distance_base", "DS", _fp(measure_cols=["distance_percentage"]))
    assert got_cb.own_max_len == 5
    assert got_db.own_max_len == 9


def test_corrupt_cache_returns_none_not_raise(tmp_path):
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    (freeze.slice_dir(tmp_path, "combined_base", "DS") / "meta.json").write_text("{not json")
    assert freeze.load_slice(tmp_path, "combined_base", "DS", _fp()) is None


class _Cfg:
    def __init__(self, overrides):
        self.dataset_overrides = overrides


def test_freeze_flags_reads_override():
    cfg = _Cfg({"DS": DatasetOverride(freeze_data=True, freeze_figures=False)})
    assert freeze.freeze_flags(cfg, "DS") == (True, False)


def test_freeze_flags_unknown_dataset_is_unfrozen():
    assert freeze.freeze_flags(_Cfg({}), "NOPE") == (False, False)


def test_thaw_overrides_named_dataset():
    cfg = _Cfg({"DS": DatasetOverride(freeze_data=True, freeze_figures=True)})
    assert freeze.freeze_flags(cfg, "DS", thawed=["DS"]) == (False, False)


def test_thaw_does_not_affect_other_datasets():
    cfg = _Cfg({
        "DS": DatasetOverride(freeze_data=True, freeze_figures=True),
        "OTHER": DatasetOverride(freeze_data=True, freeze_figures=True),
    })
    assert freeze.freeze_flags(cfg, "OTHER", thawed=["DS"]) == (True, True)


def test_thaw_all_overrides_everything():
    cfg = _Cfg({"DS": DatasetOverride(freeze_data=True, freeze_figures=True)})
    assert freeze.freeze_flags(cfg, "DS", thaw_all=True) == (False, False)
