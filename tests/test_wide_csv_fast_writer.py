"""build_wide_csv's row writers must produce exactly the bytes pandas wrote.

Every wide row used to be written as a 1-row, 12,242-column DataFrame
(``pd.DataFrame([row], columns=...).to_csv(path, mode="a")``) and every spliced
frozen slice as ``group.to_csv(path, mode="a")``. The fast writers keep pandas
for the ~38 metadata/AUC cells and format the float64 ``dir_val_*`` block with
the same ``ndarray.astype(str)`` + ``na_rep=""`` rule pandas applies to float64
columns. These tests pin that equivalence against pandas itself -- run them
under both pandas versions the pipeline meets (nightly PYTHONNOUSERSITE=1 ->
2.3.x, an interactive ``python3`` -> 3.x).
"""

from __future__ import annotations

import bz2
import gzip
import io

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev

N_META = 38
META_COLS = [f"m{i}" for i in range(N_META - 7)] + list(ec.AUC_COLUMNS)

EDGE_VALUES = np.array(
    [
        np.nan,
        np.inf,
        -np.inf,
        -0.0,
        0.0,
        1e-05,
        1e16,
        1e22,
        0.1 + 0.2,
        5e-324,  # smallest subnormal
        1e-320,  # subnormal
        2.2250738585072014e-308,  # smallest normal
        1.7976931348623157e308,
        123456789012345678.0,
        -1e-7,
        2.5,
        100.0,
        3.0,
    ],
    dtype=np.float64,
)


def _meta_cells(i: int) -> list:
    """38 cells of every Python/numpy type build_wide_csv puts in a row."""
    cells = [
        "Hex-Sensitivity-24-0.1",
        f"august_2{i % 10}_batch_1_rig_3",
        "UNKNOWN" if i % 5 == 0 else str(i % 4 + 1),
        "Gr5a>CsChrimson ü" if i % 3 == 0 else "UNKNOWN",
        float(i) / 7,
        np.float64(-i) / 3,
        float("nan"),
        float("inf") if i % 4 == 0 else 12.5,
        -0.0,
        1e16,
        0.1 + 0.2,
        5e-324,
        np.float64(1e-05),
        1.0 if i % 2 else 0.0,
        0.0,
        int(i),
        np.int64(i * 3),
        float(i) / 9,
        bool(i % 2),
        np.bool_(i % 3 == 0),
        int(3600 + i),
        30.0,
        60.0,
        90.0,
        float("nan"),
        "testing" if i % 3 == 0 else ("training" if i % 3 == 1 else "pretest"),
        f"testing_{i}",
        40.0,
        None if i == 7 else 'a,"quoted" label',
        bool(i % 4 == 0),
        float("-inf"),
    ]
    auc = [
        float(i) * 1.5,
        float("nan"),
        3.0,
        np.float64(0.1),
        float("inf") if i == 2 else 2.0,
        12.25,
        float(i) - 0.5,
    ]
    assert len(cells) + len(auc) == N_META
    return [*cells, *auc]


def _pandas_line(meta_cells, values, max_len) -> str:
    """The pre-change writer, verbatim: 1-row DataFrame -> to_csv."""
    row = [*meta_cells, *list(values)]
    if len(values) < max_len:
        row.extend([np.nan] * (max_len - len(values)))
    elif len(values) > max_len:
        row = row[: len(meta_cells) + max_len]
    cols = META_COLS + [f"dir_val_{k}" for k in range(max_len)]
    return pd.DataFrame([row], columns=cols).to_csv(index=False, header=False)


def _traces():
    rng = np.random.default_rng(1234)
    real = rng.normal(40.0, 25.0, 3600)
    real[::97] = np.nan  # interior gaps
    with_tail_nan = real.copy()
    with_tail_nan[-40:] = np.nan
    return {
        "real": real,
        "edge": EDGE_VALUES,
        "all_nan": np.full(500, np.nan),
        "empty": np.array([], dtype=np.float64),
        "single": np.array([7.25]),
        "leading_nan": np.concatenate([np.full(10, np.nan), real[:100]]),
        "tail_nan": with_tail_nan,
        "float32_origin": rng.normal(0, 1, 300).astype(np.float32).astype(np.float64),
        "integral_floats": np.arange(50, dtype=np.float64),
    }


@pytest.mark.parametrize("name", list(_traces()))
@pytest.mark.parametrize("max_len", [0, 1, 20, 3600, 4000])
def test_row_line_matches_pandas(name, max_len):
    values = _traces()[name]
    meta = _meta_cells(len(name) + max_len)
    cols = META_COLS + [f"dir_val_{k}" for k in range(max_len)]
    assert ec._wide_row_line(meta, values, max_len, cols) == _pandas_line(meta, values, max_len)


@pytest.mark.parametrize(
    "values",
    [
        np.arange(6, dtype=np.int64),  # pandas prints '5'; a float cast would print '5.0'
        np.array([1.5, np.nan, 2.25], dtype=np.float32),
        np.array(["a", "b"], dtype=object),
        [1.0, 2.0, float("nan")],
        np.array([True, False]),
    ],
    ids=["int64", "float32", "strings", "list", "bool"],
)
def test_row_line_falls_back_to_pandas_for_non_float64(values):
    meta = _meta_cells(3)
    max_len = 8
    cols = META_COLS + [f"dir_val_{k}" for k in range(max_len)]
    assert ec._wide_row_line(meta, values, max_len, cols) == _pandas_line(meta, values, max_len)


def _verbatim_old_writer(meta_cells, values, max_len, columns) -> str:
    """build_wide_csv's pre-change row writer, character for character: the
    truncation keeps ``len(metadata) + len(AUC_COLUMNS) + max_len`` cells,
    i.e. the header width, whatever the length of the metadata cells."""
    row = [*meta_cells, *list(values)]
    if len(values) < max_len:
        row.extend([np.nan] * (max_len - len(values)))
    elif len(values) > max_len:
        row = row[: len(columns)]
    return pd.DataFrame([row], columns=list(columns)).to_csv(index=False, header=False)


@pytest.mark.parametrize("meta_delta", [-1, 1])
@pytest.mark.parametrize("n_values", [3, 6, 9], ids=["padded", "exact", "truncated"])
def test_row_line_with_wrong_meta_width_does_what_the_old_writer_did(meta_delta, n_values):
    """Metadata one cell short or long (a protocol/schema mismatch): whatever the
    old writer did -- raise, or write the row it truncated to the header width
    -- the fallback does too."""
    meta = _meta_cells(5)
    meta = meta[:-1] if meta_delta < 0 else [*meta, "extra"]
    max_len = 6
    cols = META_COLS + [f"dir_val_{k}" for k in range(max_len)]
    values = _traces()["real"][:n_values]
    try:
        want = _verbatim_old_writer(meta, values, max_len, cols)
    except ValueError:
        with pytest.raises(ValueError):
            ec._wide_row_line(meta, values, max_len, cols)
        return
    assert ec._wide_row_line(meta, values, max_len, cols) == want


def test_row_line_int_trace_keeps_pandas_int_formatting():
    """Guard the fallback itself: an int trace must not come out as '5.0'."""
    meta = _meta_cells(1)
    cols = META_COLS + [f"dir_val_{k}" for k in range(3)]
    line = ec._wide_row_line(meta, np.array([5, 6, 7]), 3, cols)
    assert line.endswith(",5,6,7\n")


def test_row_line_bytes_match_pandas_file_append(tmp_path):
    """Same bytes on disk as the old to_csv(path, mode='a'): utf-8, '\\n' only."""
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    max_len = 64
    cols = META_COLS + [f"dir_val_{k}" for k in range(max_len)]
    with new.open("a", encoding="utf-8", newline="") as fh:
        for i, values in enumerate(_traces().values()):
            meta = _meta_cells(i)
            row = [*meta, *list(values)]
            if len(values) < max_len:
                row.extend([np.nan] * (max_len - len(values)))
            else:
                row = row[: N_META + max_len]
            pd.DataFrame([row], columns=cols).to_csv(old, index=False, mode="a", header=False)
            fh.write(ec._wide_row_line(meta, values, max_len, cols))
    assert old.read_bytes() == new.read_bytes()
    assert "ü".encode("utf-8") in new.read_bytes()


# --------------------------------------------------------------------------
# Frozen splice: a whole cached slice at once.
# --------------------------------------------------------------------------


def _frozen_frame(n_rows: int, n_vals: int, seed: int = 0) -> pd.DataFrame:
    """Shaped like a reindexed freeze slice: meta of every dtype + float64 dir_vals."""
    rng = np.random.default_rng(seed)
    meta = {
        "dataset": ["RandomPanel-Training-24-10"] * n_rows,
        "fly": [f"june_30_batch_{i % 3}" for i in range(n_rows)],
        "fly_number": [i % 4 + 1 for i in range(n_rows)],
        "fly_type": [np.nan if i % 6 == 0 else "Gr5a ü" for i in range(n_rows)],
        "global_min": rng.normal(size=n_rows),
        "non_reactive_flag": [float(i % 2) for i in range(n_rows)],
        "tracking_missing_frames": np.arange(n_rows, dtype=np.int64),
        "tracking_flagged": [bool(i % 2) for i in range(n_rows)],
        "trace_len": np.full(n_rows, n_vals, dtype=np.int64),
        "trial_light_on_s": [np.nan if i % 2 else 35.05 for i in range(n_rows)],
        "trial_type": ["testing" if i % 2 else "training" for i in range(n_rows)],
        "trial_label": [f"testing_{i}" for i in range(n_rows)],
        "fps": [40.02] * n_rows,
        "frozen": [False] * n_rows,
    }
    frame = pd.DataFrame(meta)
    for name in ec.AUC_COLUMNS:
        frame[name] = rng.normal(size=n_rows)
    vals = rng.normal(50.0, 20.0, (n_rows, n_vals))
    vals[:, ::53] = np.nan
    for i in range(n_rows):
        # own trace length varies per row -> trailing NaN padding of every size
        vals[i, max(0, n_vals - 7 * i):] = np.nan
    k = min(5, n_vals)
    vals[0, :k] = EDGE_VALUES[:k]
    if n_rows > 1:
        vals[1, :] = np.nan  # an all-NaN row
    dv = pd.DataFrame(vals, columns=[f"dir_val_{k}" for k in range(n_vals)])
    return pd.concat([frame, dv], axis=1)


def _n_meta(frame: pd.DataFrame) -> int:
    return sum(1 for c in frame.columns if not str(c).startswith("dir_val_"))


@pytest.mark.parametrize("shape", [(1, 10), (5, 0), (7, 120), (300, 40), (2, 1)])
def test_frame_writer_matches_pandas(shape):
    frame = _frozen_frame(*shape)
    buf = io.StringIO()
    ec._write_wide_frame(buf, frame, _n_meta(frame))
    assert buf.getvalue() == frame.to_csv(index=False, header=False)


def test_frame_writer_matches_pandas_after_reindex_padding():
    """The splice reindexes to the global width: new columns are float64 NaN."""
    frame = _frozen_frame(9, 30)
    meta = [c for c in frame.columns if not c.startswith("dir_val_")]
    target = meta + [f"dir_val_{k}" for k in range(75)]
    aligned = frame.reindex(columns=target)
    buf = io.StringIO()
    ec._write_wide_frame(buf, aligned, len(meta))
    assert buf.getvalue() == aligned.to_csv(index=False, header=False)


def test_frame_writer_empty_frame_writes_nothing():
    frame = _frozen_frame(3, 5).iloc[0:0]
    buf = io.StringIO()
    ec._write_wide_frame(buf, frame, _n_meta(frame))
    assert buf.getvalue() == frame.to_csv(index=False, header=False) == ""


@pytest.mark.parametrize("kind", ["object_vals", "int_vals", "float32_vals", "newline_meta", "one_meta"])
def test_frame_writer_falls_back_to_pandas(kind):
    frame = _frozen_frame(6, 12)
    n_meta = _n_meta(frame)
    if kind == "object_vals":
        frame["dir_val_3"] = frame["dir_val_3"].astype(object)
    elif kind == "int_vals":
        frame["dir_val_0"] = np.arange(len(frame), dtype=np.int64)
    elif kind == "float32_vals":
        frame["dir_val_2"] = frame["dir_val_2"].astype(np.float32)
    elif kind == "newline_meta":
        frame.loc[2, "trial_label"] = "line\nbreak"  # quoted by pandas
    elif kind == "one_meta":
        frame = frame[["fly_type"] + [c for c in frame.columns if c.startswith("dir_val_")]]
        frame.loc[1, "fly_type"] = ""  # csv writes a lone empty field as ""
        n_meta = 1
    buf = io.StringIO()
    ec._write_wide_frame(buf, frame, n_meta)
    assert buf.getvalue() == frame.to_csv(index=False, header=False)


# --------------------------------------------------------------------------
# End to end: the whole build, fast writers vs the pre-change pandas writers.
# --------------------------------------------------------------------------


def _write_trace(out_dir, stem, values):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"envelope_of_rms": values}).to_csv(out_dir / f"{stem}.csv", index=False)


def _make_live(root, n=40):
    rng = np.random.default_rng(7)
    for batch in ("october_01_batch_1_rig_2", "october_02_batch_1"):
        env = root / batch / "angle_distance_rms_envelope"
        for tt, idx in (("pretest", 1), ("training", 1), ("training", 2), ("testing", 1), ("testing", 2)):
            for fly in (1, 2):
                vals = rng.normal(30.0, 10.0, n + idx)
                vals[::11] = np.nan
                if tt == "testing" and idx == 2 and fly == 2:
                    vals[:] = np.nan  # a trace with no finite sample
                _write_trace(env, f"{tt}_{idx}_hexanol_fly{fly}_distances_fly{fly}_angle_distance_rms_envelope", vals)
    return root


def _reference_writers(monkeypatch):
    """Swap the fast writers for the exact pre-change pandas calls."""

    def _frame(fh, frame, n_meta):
        fh.write(frame.to_csv(index=False, header=False))

    monkeypatch.setattr(ec, "_wide_row_line", _verbatim_old_writer)
    monkeypatch.setattr(ec, "_write_wide_frame", _frame)


def _build(tmp_path, tag, *, frozen=None, frozen_folders=None):
    out = tmp_path / tag
    out.mkdir()
    main = out / "wide.csv"
    extras = {"training": str(out / "wide_training.csv"), "pretest": str(out / "wide_pretest.csv")}
    ec.build_wide_csv(
        [str(tmp_path / "LIVE-DS"), str(tmp_path / "AAA-FROZEN")],
        str(main),
        measure_cols=["envelope_of_rms"],
        extra_trial_exports=extras,
        frozen_slices=frozen,
        frozen_folders=frozen_folders,
    )
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


@pytest.mark.parametrize("protocol", ["v2", "legacy"])
def test_build_wide_csv_bytes_match_pre_change_writers(tmp_path, monkeypatch, protocol):
    ev.set_protocol(protocol)
    _make_live(tmp_path / "LIVE-DS")
    # The frozen dataset's traces are the LONGEST, so live rows get NaN padding
    # from the global max -- the production shape (12,204 vs ~3,600).
    _make_live(tmp_path / "AAA-FROZEN", n=90)
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    ec.build_wide_csv(
        [str(tmp_path / "AAA-FROZEN")],
        str(seed_dir / "w.csv"),
        measure_cols=["envelope_of_rms"],
        extra_trial_exports={"training": str(seed_dir / "t.csv"), "pretest": str(seed_dir / "p.csv")},
    )
    rows = pd.concat([pd.read_csv(seed_dir / n) for n in ("w.csv", "t.csv", "p.csv")], ignore_index=True)
    frozen = {"AAA-FROZEN": (rows, int(rows["trace_len"].max()))}
    folders = {"LIVE-DS": {"october_02_batch_1"}}

    appends = []
    real_to_csv = pd.DataFrame.to_csv

    def _spy(self, path_or_buf=None, *a, **k):
        if k.get("mode") == "a":
            appends.append(path_or_buf)
        return real_to_csv(self, path_or_buf, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", _spy)
    fast = _build(tmp_path, "fast", frozen=frozen, frozen_folders=folders)
    assert appends == [], "a row was written through a per-row DataFrame.to_csv append"

    _reference_writers(monkeypatch)
    ref = _build(tmp_path, "ref", frozen=frozen, frozen_folders=folders)

    assert sorted(fast) == sorted(ref)
    assert {"wide.csv", "wide_training.csv", "wide_pretest.csv", "wide.parquet"} <= set(fast)
    for name in ref:
        assert fast[name] == ref[name], name
    main = pd.read_csv(tmp_path / "fast" / "wide.csv")
    assert set(main["dataset"]) == {"AAA-FROZEN", "LIVE-DS"}
    if protocol == "v2":
        live = main[main["dataset"] == "LIVE-DS"]
        assert live.loc[live["fly"] == "october_02_batch_1", "frozen"].all()
        assert not live.loc[live["fly"] != "october_02_batch_1", "frozen"].any()
        assert not main.loc[main["dataset"] == "AAA-FROZEN", "frozen"].any()


def test_rows_written_before_a_failure_are_on_disk(tmp_path, monkeypatch):
    """The old writer opened, appended and closed per row, so a build that
    raises part-way left every row written so far on disk. The shared append
    handles must too -- while the traceback still holds the failed frame, and
    with it the open handles."""
    ev.set_protocol("v2")
    _make_live(tmp_path / "LIVE-DS")
    full = tmp_path / "full"
    full.mkdir()
    extras = lambda d: {"training": str(d / "t.csv"), "pretest": str(d / "p.csv")}  # noqa: E731
    ec.build_wide_csv(
        [str(tmp_path / "LIVE-DS")], str(full / "w.csv"), measure_cols=["envelope_of_rms"], extra_trial_exports=extras(full)
    )

    calls = []
    real = ec._wide_row_line

    def _fail_on_fourth(*a, **k):
        if len(calls) == 3:
            raise RuntimeError("injected")
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ec, "_wide_row_line", _fail_on_fourth)
    part = tmp_path / "part"
    part.mkdir()
    with pytest.raises(RuntimeError, match="injected") as excinfo:
        ec.build_wide_csv(
            [str(tmp_path / "LIVE-DS")], str(part / "w.csv"), measure_cols=["envelope_of_rms"], extra_trial_exports=extras(part)
        )
    assert excinfo.traceback  # the failed frame is still alive
    rows = 0
    for name in ("w.csv", "t.csv", "p.csv"):
        got = (part / name).read_text(encoding="utf-8").splitlines(keepends=True)
        want = (full / name).read_text(encoding="utf-8").splitlines(keepends=True)
        assert got == want[: len(got)], name
        rows += len(got) - 1  # minus the header
    assert rows == 3


def test_a_compressed_output_still_decompresses_to_the_plain_build(tmp_path):
    """to_csv(path, mode="a") compresses every append when pandas infers a
    compression from the path (.gz, .bz2, ...), so the per-row writer left a
    valid multi-member file. The shared append handles must not write plain
    text into such a file: each output decompresses to the plain build."""
    ev.set_protocol("v2")
    _make_live(tmp_path / "LIVE-DS")
    built = {}
    for ext in (".csv", ".csv.gz", ".csv.bz2"):
        out = tmp_path / ext.lstrip(".").replace(".", "_")
        out.mkdir()
        ec.build_wide_csv(
            [str(tmp_path / "LIVE-DS")],
            str(out / f"w{ext}"),
            measure_cols=["envelope_of_rms"],
            extra_trial_exports={"training": str(out / f"t{ext}"), "pretest": str(out / f"p{ext}")},
        )
        built[ext] = out
    for name in ("w", "t", "p"):
        plain = (built[".csv"] / f"{name}.csv").read_bytes()
        assert plain.count(b"\n") > 1, name  # rows, not just the header
        assert gzip.decompress((built[".csv.gz"] / f"{name}.csv.gz").read_bytes()) == plain, name
        assert bz2.decompress((built[".csv.bz2"] / f"{name}.csv.bz2").read_bytes()) == plain, name


def _spy_to_csv(monkeypatch):
    widths = []
    real_to_csv = pd.DataFrame.to_csv

    def _spy(self, *a, **k):
        widths.append(self.shape[1])
        return real_to_csv(self, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", _spy)
    return widths


def test_fast_paths_are_taken_for_float64(monkeypatch):
    """Guard against a writer that is correct only because it always falls back."""
    widths = _spy_to_csv(monkeypatch)
    max_len = 500
    cols = META_COLS + [f"dir_val_{k}" for k in range(max_len)]
    ec._wide_row_line(_meta_cells(2), _traces()["real"], max_len, cols)
    frame = _frozen_frame(20, 300)
    ec._write_wide_frame(io.StringIO(), frame, _n_meta(frame))
    assert widths == [N_META, _n_meta(frame)]
