"""_write_freeze_cache: read the just-written Parquet siblings, skip unchanged slices.

build_wide_csv writes ``<stem>.parquet`` from ``pd.read_csv(<stem>.csv)`` right
after each wide CSV, so the freeze cache does not need its own second pass of
``pd.read_csv`` over ~2 GB of CSV. The sibling write is best-effort (a warning
on failure), so a missing, stale or unreadable sibling falls back to the CSV.
A dataset spliced from its cache is not re-saved when the re-save would write
that cache back unchanged -- and only then.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import scripts.pipeline.run_workflows as rw
from fbpipe import freeze
from fbpipe.analysis.threshold import ThresholdRule
from fbpipe.config import DatasetOverride
from fbpipe.utils.tables import write_table


class _Tracking:
    apply_missing_frame_check = True
    max_missing_frames_per_trial = 5000
    max_missing_frames_pct_per_trial = 50.0


class _Settings:
    flagged_root = None
    flagged_secured_root = None
    non_reactive_span_px = 12.5
    class2_min = 0.0
    class2_max = 100.0

    def __init__(self, tmp_path, overrides, datasets, cache="cache"):
        self.cache_dir = str(tmp_path / cache)
        self.dataset_overrides = overrides
        self.datasets = tuple(datasets)
        self.protocol = "v2"
        self.tracking = _Tracking()


def _fp(ds):
    return freeze.build_fingerprint(
        protocol="v2",
        override=DatasetOverride(),
        tracking=_Tracking(),
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=12.5,
        low_max_threshold_px=5.0,
        use_per_trial_baseline=False,
        trial_type_filter=None,
        threshold_rule=ThresholdRule(),
    )


def _wide(dataset, n, trial_type="testing", tl=6, pad=3):
    """Rows with every dtype a wide CSV carries (str w/ NaN, int, bool, float).

    ``tl`` values then ``pad`` NaN columns; the values do not depend on ``pad``.
    """
    rng = np.random.default_rng(len(dataset) + n)
    rec = {
        "dataset": [dataset] * n,
        "fly": [f"october_0{i % 3}_batch_1_rig_2" for i in range(n)],
        "fly_number": [i % 4 + 1 for i in range(n)],
        "fly_type": [np.nan if i % 2 else "Gr5a" for i in range(n)],
        "trial_type": [trial_type] * n,
        "trace_len": [tl] * n,
        "frozen": [bool(i % 2) for i in range(n)],
    }
    df = pd.DataFrame(rec)
    vals = np.hstack([rng.normal(size=(n, tl)), np.full((n, pad), np.nan)])
    return pd.concat([df, pd.DataFrame(vals, columns=[f"dir_val_{k}" for k in range(tl + pad)])], axis=1)


def _write_outputs(tmp_path, *, siblings=True, pad=3, live_fly_number_nan=False):
    main, train = tmp_path / "wide.csv", tmp_path / "wide_training.csv"
    live = _wide("LIVE", 5, pad=pad)
    if live_fly_number_nan:
        live.loc[0, "fly_number"] = np.nan  # an int column read back as float64
    pd.concat([live, _wide("FROZEN", 4, pad=pad)], ignore_index=True).to_csv(main, index=False)
    pd.concat(
        [_wide("LIVE", 3, "training", pad=pad), _wide("FROZEN", 2, "training", pad=pad)], ignore_index=True
    ).to_csv(train, index=False)
    if siblings:
        for p in (main, train):  # exactly what build_wide_csv does
            write_table(pd.read_csv(p), p, replace_legacy_csv=False)
    return main, train


def _slices(settings, datasets):
    out = {}
    for ds in datasets:
        d = freeze.slice_dir(settings.cache_dir, "combined_base", ds)
        rows = pd.read_parquet(d / "rows.parquet")
        out[ds] = (rows, hashlib.md5((d / "rows.parquet").read_bytes()).hexdigest(), (d / "meta.json").read_text())
    return out


def _run(settings, main, train, fingerprint_for=_fp, **kw):
    rw._write_freeze_cache(
        settings, "combined_base", str(main), [str(train)], fingerprint_for=fingerprint_for, **kw
    )


def test_reads_current_parquet_siblings_instead_of_the_csvs(tmp_path, monkeypatch):
    main, train = _write_outputs(tmp_path)
    # Reference: today's behaviour, pd.read_csv of each CSV (siblings moved away).
    reference = _Settings(tmp_path, {}, ["LIVE", "FROZEN"], cache="ref")
    for p in (main, train):
        p.with_suffix(".parquet").rename(p.with_suffix(".bak"))
    _run(reference, main, train)
    for p in (main, train):
        p.with_suffix(".bak").rename(p.with_suffix(".parquet"))

    read_csv_calls = []
    real_read_csv = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: read_csv_calls.append(a) or real_read_csv(*a, **k))
    fast = _Settings(tmp_path, {}, ["LIVE", "FROZEN"], cache="fast")
    _run(fast, main, train)

    assert read_csv_calls == []
    want, got = _slices(reference, ["LIVE", "FROZEN"]), _slices(fast, ["LIVE", "FROZEN"])
    for ds in want:
        assert got[ds][0].equals(want[ds][0])
        assert list(got[ds][0].dtypes) == list(want[ds][0].dtypes)
        assert got[ds][1] == want[ds][1], f"{ds} rows.parquet bytes differ"
        assert got[ds][2] == want[ds][2], f"{ds} meta.json differs"


@pytest.mark.parametrize("how", ["older", "missing", "corrupt"])
def test_falls_back_to_the_csv_when_the_sibling_cannot_be_trusted(tmp_path, how):
    main, train = _write_outputs(tmp_path)
    if how == "older":
        # A stale sibling from an earlier run: different rows, older mtime.
        write_table(_wide("LIVE", 1), main, replace_legacy_csv=False)
        st = main.stat()
        os.utime(main.with_suffix(".parquet"), ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))
    elif how == "missing":
        main.with_suffix(".parquet").unlink()
    else:
        main.with_suffix(".parquet").write_bytes(b"not a parquet file")
        os.utime(main.with_suffix(".parquet"))
    s = _Settings(tmp_path, {}, ["LIVE", "FROZEN"])
    _run(s, main, train)
    rows = _slices(s, ["LIVE"])["LIVE"][0]
    expect = pd.concat([pd.read_csv(main), pd.read_csv(train)], ignore_index=True)
    expect = expect[expect["dataset"] == "LIVE"].reset_index(drop=True)
    assert rows.equals(expect)


def test_a_sibling_two_outputs_share_is_never_read(tmp_path):
    """``wide.csv`` and ``wide.txt`` both map to ``wide.parquet``, so
    build_wide_csv's sibling loop leaves only the LAST output's rows there:
    it stands in for neither. Each is read from its own CSV, as before."""
    main, other = tmp_path / "wide.csv", tmp_path / "wide.txt"
    _wide("LIVE", 5).to_csv(main, index=False)
    _wide("LIVE", 3, "training").to_csv(other, index=False)
    for p in (main, other):  # exactly what build_wide_csv's loop does
        write_table(pd.read_csv(p), p, replace_legacy_csv=False)
    ref = _Settings(tmp_path, {}, ["LIVE"], cache="ref")
    sibling = main.with_suffix(".parquet")
    sibling.rename(tmp_path / "away.bak")
    _run(ref, main, other)
    (tmp_path / "away.bak").rename(sibling)
    os.utime(sibling)  # newer than both CSVs

    s = _Settings(tmp_path, {}, ["LIVE"], cache="fast")
    _run(s, main, other)
    got, want = _slices(s, ["LIVE"])["LIVE"], _slices(ref, ["LIVE"])["LIVE"]
    assert got[0].equals(want[0]) and len(got[0]) == 8
    assert got[1:] == want[1:]


def _fp_spelled(odor_on_s):
    """A fingerprint whose override carries a raw config number (int or float)."""

    def fp(ds):
        return freeze.build_fingerprint(
            protocol="v2",
            override=DatasetOverride(odor_on_s=odor_on_s),
            tracking=_Tracking(),
            measure_cols=["envelope_of_rms"],
            fps_fallback=40.0,
            distance_limits=None,
            non_reactive_threshold=12.5,
            low_max_threshold_px=5.0,
            use_per_trial_baseline=False,
            trial_type_filter=None,
            threshold_rule=ThresholdRule(),
        )

    return fp


def _slice_files(settings, ds):
    d = freeze.slice_dir(settings.cache_dir, "combined_base", ds)
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


@pytest.mark.parametrize(
    "drift",
    ["none", "dtype", "width", "fingerprint_spelling", "meta_missing", "meta_stale", "meta_reformatted"],
)
def test_spliced_slice_is_skipped_only_when_a_re_save_would_rewrite_it_as_is(tmp_path, drift):
    """Night 1 primes the cache; night 2 splices FROZEN back from it.

    Night 2's slice of FROZEN is cut from the whole re-read output, so it can
    differ from the cached one even though FROZEN's own rows did not change:
    a live NaN turns an int column float64 (the next splice would print 2.0,
    not 2), a longer live trace widens every row, and a config number spelled
    30 -> 30.0 rewrites meta.json. A meta.json written by an older save_slice
    (another own_max_len, other formatting) under the same fingerprint is
    rewritten too. Today every slice is re-saved. The skip must leave exactly
    the bytes that re-save writes, and skip only when it would have rewritten
    the slice unchanged.
    """
    prime_fp = _fp_spelled(30) if drift == "fingerprint_spelling" else _fp
    night2_fp = _fp_spelled(30.0) if drift == "fingerprint_spelling" else _fp
    main, train = _write_outputs(tmp_path)
    new = _Settings(tmp_path, {"FROZEN": DatasetOverride(freeze_data=True)}, ["LIVE", "FROZEN"], cache="new")
    _run(new, main, train, fingerprint_for=prime_fp)
    ref = _Settings(tmp_path, {"FROZEN": DatasetOverride(freeze_data=True)}, ["LIVE", "FROZEN"], cache="ref")
    shutil.copytree(new.cache_dir, ref.cache_dir)

    # What _resolve_frozen_slices hands build_wide_csv on night 2.
    spliced = {"FROZEN": freeze.load_slice(new.cache_dir, "combined_base", "FROZEN", night2_fp("FROZEN"))}
    assert spliced["FROZEN"] is not None
    if drift == "dtype":
        _write_outputs(tmp_path, live_fly_number_nan=True)
    elif drift == "width":
        _write_outputs(tmp_path, pad=5)
    elif drift == "meta_missing":
        for s in (new, ref):
            (freeze.slice_dir(s.cache_dir, "combined_base", "FROZEN") / "meta.json").unlink()
    elif drift in ("meta_stale", "meta_reformatted"):
        for s in (new, ref):
            meta_path = freeze.slice_dir(s.cache_dir, "combined_base", "FROZEN") / "meta.json"
            meta = json.loads(meta_path.read_text())
            if drift == "meta_stale":
                meta["own_max_len"] += 1  # what an older own_max_len may have stored
                meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            else:
                meta_path.write_text(json.dumps(meta))  # same content, other layout
    frozen_dir = freeze.slice_dir(new.cache_dir, "combined_base", "FROZEN")
    for p in frozen_dir.iterdir():
        os.utime(p, ns=(0, 0))

    _run(ref, main, train, fingerprint_for=night2_fp)  # today: re-save every slice
    _run(new, main, train, fingerprint_for=night2_fp, spliced=spliced)

    for ds in ("FROZEN", "LIVE"):
        assert _slice_files(new, ds) == _slice_files(ref, ds), f"{ds} cache bytes differ from a re-save"
    skipped = all(p.stat().st_mtime_ns == 0 for p in frozen_dir.iterdir())
    assert skipped == (drift == "none")
    if drift == "dtype":
        assert rw.pd.read_parquet(frozen_dir / "rows.parquet")["fly_number"].dtype == np.float64


def test_spliced_tuple_slices_are_compared_too(tmp_path):
    """build_wide_csv's frozen_slices contract is (rows, own_max_len) tuples."""
    main, train = _write_outputs(tmp_path)
    s = _Settings(tmp_path, {"FROZEN": DatasetOverride(freeze_data=True)}, ["LIVE", "FROZEN"])
    _run(s, main, train)
    loaded = freeze.load_slice(s.cache_dir, "combined_base", "FROZEN", _fp("FROZEN"))
    d = freeze.slice_dir(s.cache_dir, "combined_base", "FROZEN")
    for p in d.iterdir():
        os.utime(p, ns=(0, 0))
    _run(s, main, train, spliced={"FROZEN": (loaded.rows, loaded.own_max_len)})
    assert all(p.stat().st_mtime_ns == 0 for p in d.iterdir())


def test_unchanged_spliced_slice_is_compared_without_copying_its_rows(tmp_path, monkeypatch):
    """Frozen rows are most of the wide table (19k of 26k a night). Skipping
    their re-save must not still pay reset_index's full copy of each slice."""
    main, train = _write_outputs(tmp_path)
    s = _Settings(tmp_path, {"FROZEN": DatasetOverride(freeze_data=True)}, ["LIVE", "FROZEN"])
    _run(s, main, train)
    spliced = {"FROZEN": freeze.load_slice(s.cache_dir, "combined_base", "FROZEN", _fp("FROZEN"))}
    reset = []
    real_reset = pd.DataFrame.reset_index

    def _spy(self, *a, **k):
        if "dataset" in self.columns:
            reset.extend(set(self["dataset"].astype(str)))
        return real_reset(self, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "reset_index", _spy)
    _run(s, main, train, spliced=spliced)
    assert reset == ["LIVE"]


def test_spliced_default_saves_every_dataset(tmp_path):
    """The old call (no ``spliced``) keeps priming every slice."""
    main, train = _write_outputs(tmp_path)
    s = _Settings(tmp_path, {}, ["LIVE", "FROZEN"])
    _run(s, main, train)
    assert set(_slices(s, ["LIVE", "FROZEN"])) == {"LIVE", "FROZEN"}


def test_run_combined_passes_the_spliced_slices(tmp_path, monkeypatch):
    """All three build_wide_csv call sites hand the cache writer the very
    slices they spliced, so it can tell an unchanged slice from a drifted one."""
    captured, returned = [], []

    def _resolve(settings, roots, block, **kw):
        out = {"DS1": freeze.FrozenSlice(_wide("DS1", 1), 6)}
        returned.append(out)
        return out

    monkeypatch.setattr(rw, "build_wide_csv", lambda *a, **k: None)
    monkeypatch.setattr(rw, "wide_to_matrix", lambda *a, **k: None)
    monkeypatch.setattr(rw, "_resolve_frozen_slices", _resolve)
    monkeypatch.setattr(rw, "_write_freeze_cache", lambda *a, **k: captured.append(k.get("spliced")))
    ds_root = tmp_path / "DS1"
    ds_root.mkdir()
    cfg = {
        "wide": {"roots": [str(ds_root)], "output_csv": str(tmp_path / "wide.csv")},
        "combined_base": {"wide": {"output_csv": str(tmp_path / "wide_base.csv")}},
        "pair_groups": [{"name": "pair1", "datasets": [ds_root.name], "out_dir": str(tmp_path / "pair1")}],
    }
    rw._run_combined(cfg, _Settings(tmp_path, {}, ["DS1"]), config_path=tmp_path / "c.yaml")
    assert len(captured) == len(returned) == 3
    assert all(c is r for c, r in zip(captured, returned))


def test_auto_freeze_caches_the_walked_datasets_stamped_with_the_run(tmp_path, monkeypatch):
    """Under an active auto-freeze plan only the datasets this run walked are
    cached, each stamped with the run that wrote it -- the commit trusts no
    other slice -- and a walked dataset that produced no rows gets an empty
    slice (no rows IS its truth). A spliced dataset is never re-saved."""
    from fbpipe import autofreeze as af

    monkeypatch.setattr(af, "_ACTIVE", {"schema": af.PLAN_SCHEMA, "run_id": "R1", "datasets": {}})
    main, train = _write_outputs(tmp_path)  # rows for LIVE and FROZEN only
    s = _Settings(tmp_path, {}, ["LIVE", "FROZEN", "EMPTY"])
    _run(s, main, train, spliced={"FROZEN": (_wide("FROZEN", 4), 6)}, walked={"LIVE", "FROZEN", "EMPTY"})

    def _meta(ds):
        return json.loads((freeze.slice_dir(s.cache_dir, "combined_base", ds) / "meta.json").read_text())

    assert _meta("LIVE")["built_by_run"] == "R1" and _meta("LIVE")["row_count"] == 8
    assert _meta("EMPTY")["built_by_run"] == "R1" and _meta("EMPTY")["row_count"] == 0
    assert not freeze.slice_dir(s.cache_dir, "combined_base", "FROZEN").exists()


def test_walked_rows_are_found_when_dataset_is_not_the_first_column(tmp_path, monkeypatch):
    """The walked datasets' lines are picked by their leading field. Were
    ``dataset`` not the first column, no line would match and every walked
    dataset would be cached EMPTY -- then recorded frozen without its rows."""
    from fbpipe import autofreeze as af

    monkeypatch.setattr(af, "_ACTIVE", {"schema": af.PLAN_SCHEMA, "run_id": "R1", "datasets": {}})
    main, train = tmp_path / "wide.csv", tmp_path / "wide_training.csv"
    live = _wide("LIVE", 5)
    cols = ["fly", *[c for c in live.columns if c != "fly"]]
    pd.concat([live, _wide("OTHER", 4)], ignore_index=True)[cols].to_csv(main, index=False)
    _wide("LIVE", 3, "training")[cols].to_csv(train, index=False)
    s = _Settings(tmp_path, {}, ["LIVE", "OTHER"])
    _run(s, main, train, walked={"LIVE"})
    rows = pd.read_parquet(freeze.slice_dir(s.cache_dir, "combined_base", "LIVE") / "rows.parquet")
    expected = pd.concat([live, _wide("LIVE", 3, "training")], ignore_index=True)[cols]
    assert len(rows) == 8 and set(rows["dataset"]) == {"LIVE"}
    assert list(rows["fly"]) == list(expected["fly"])
    assert not freeze.slice_dir(s.cache_dir, "combined_base", "OTHER").exists()


def test_a_wide_output_without_a_dataset_column_caches_nothing(tmp_path, monkeypatch):
    """No ``dataset`` column: no row can be told apart, so nothing is cached --
    every walked dataset stays thawing and is rebuilt next run, instead of
    being frozen empty."""
    from fbpipe import autofreeze as af

    monkeypatch.setattr(af, "_ACTIVE", {"schema": af.PLAN_SCHEMA, "run_id": "R1", "datasets": {}})
    main, train = tmp_path / "wide.csv", tmp_path / "wide_training.csv"
    _wide("LIVE", 5).drop(columns=["dataset"]).to_csv(main, index=False)
    _wide("LIVE", 3, "training").drop(columns=["dataset"]).to_csv(train, index=False)
    s = _Settings(tmp_path, {}, ["LIVE", "EMPTY"])
    _run(s, main, train, walked={"LIVE", "EMPTY"})
    for ds in ("LIVE", "EMPTY"):
        assert not freeze.slice_dir(s.cache_dir, "combined_base", ds).exists()


def test_auto_freeze_never_caches_an_exact_duplicate_row(tmp_path, monkeypatch):
    """A fresh build emits each trial once; a repeated line could only come
    from overlapping outputs, and caching it would splice it back every night."""
    from fbpipe import autofreeze as af

    monkeypatch.setattr(af, "_ACTIVE", {"schema": af.PLAN_SCHEMA, "run_id": "R1", "datasets": {}})
    main, train = tmp_path / "wide.csv", tmp_path / "wide_training.csv"
    live = _wide("LIVE", 4)
    pd.concat([live, live.iloc[[1, 2]]], ignore_index=True).to_csv(main, index=False)
    _wide("LIVE", 2, "training").to_csv(train, index=False)
    s = _Settings(tmp_path, {}, ["LIVE"])
    _run(s, main, train, walked={"LIVE"})
    rows = pd.read_parquet(freeze.slice_dir(s.cache_dir, "combined_base", "LIVE") / "rows.parquet")
    assert len(rows) == 6
    assert not rows.duplicated().any()
