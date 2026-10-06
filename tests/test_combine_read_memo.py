"""combine_distance_angle decodes each per-trial table once per batch.

Before the memo, one batch read every per-trial parquet six or seven times:
``_ensure_angle_percentages`` three times (reference angle, fly max, write
pass), the two ``_locate_trials`` header sniffs once each (``_read_trial_table``
read the whole file and then sliced five rows), and pass 1 once as "distance"
and once as "angle". The 2026-10-05 nightly made 1,113 parquet decodes for 84
trials. The memo keeps one decode per file per batch, hands out copies, and
drops a file's entry whenever combine rewrites it.

None of this may change a byte of output; the tests below hold the memo to
that against the uncached path on the same inputs.
"""

from __future__ import annotations

import contextlib
import io
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev
from _combine_fixtures import tree_digest, write_batch

DATASET = "Hex-Sensitivity-24-0.1"


@pytest.fixture(autouse=True)
def _v2():
    ev.set_protocol("v2")


def _dataset(base: Path) -> Path:
    root = base / DATASET
    write_batch(root, "august_28_batch_1_rig_3", flies=(1, 2, 3, 4))
    write_batch(
        root, "september_29_batch_1", flies=(1, 2), host="Flybehavior2", rec_date="20260929"
    )
    return root


def _combine(root: Path) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ec.combine_distance_angle(ec.CombineConfig(root=root))
    return buf.getvalue()


@pytest.fixture
def decode_counts(monkeypatch):
    """Count real parquet decodes per file (read_table and the header sniff
    both go through ``pd.read_parquet``)."""

    counts: Counter[Path] = Counter()
    real = pd.read_parquet

    def counting(path, *args, **kwargs):
        counts[Path(path).resolve()] += 1
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counting)
    return counts


def test_each_table_is_decoded_at_most_twice_per_batch(tmp_path, decode_counts):
    """First run: every per-fly file lacks angle columns, so combine rewrites
    it. One decode before the rewrite, one after; nothing more."""

    root = _dataset(tmp_path)
    _combine(root)

    per_fly = {p: n for p, n in decode_counts.items() if "merged" not in p.name}
    merged = {p: n for p, n in decode_counts.items() if "merged" in p.name}
    assert per_fly and merged
    assert max(per_fly.values()) <= 2, sorted(per_fly.values())
    # merged tables are never rewritten (no per-fly coordinates), so a single
    # decode serves the angle passes and both header sniffs
    assert set(merged.values()) == {1}


def test_unchanged_tables_are_decoded_exactly_once(tmp_path, decode_counts):
    """Second run: the angle columns already match, nothing is rewritten, and
    every table is decoded once for the whole batch."""

    root = _dataset(tmp_path)
    _combine(root)
    decode_counts.clear()
    _combine(root)

    assert decode_counts
    assert set(decode_counts.values()) == {1}


def test_the_memo_changes_no_output(tmp_path, monkeypatch):
    """Same inputs, memo on vs. memo off: every file byte-identical and the
    same log, on the first run (rewrites) and the second (no rewrites)."""

    cached = _dataset(tmp_path / "cached")
    uncached = tmp_path / "uncached" / DATASET
    shutil.copytree(cached, uncached)

    def norm(log: str, base: Path) -> str:
        return log.replace(str(base), "<ROOT>")

    logs_cached = [norm(_combine(cached), cached.parent) for _ in range(2)]
    with monkeypatch.context() as m:
        m.setattr(ec, "_batch_read_memo", contextlib.nullcontext)
        logs_uncached = [norm(_combine(uncached), uncached.parent) for _ in range(2)]

    assert tree_digest(cached) == tree_digest(uncached)
    assert logs_cached == logs_uncached


def test_rewritten_files_leave_the_memo(tmp_path):
    """_ensure_angle_percentages drops a file's entry when it writes the file,
    rather than trusting the next stat to notice."""

    batch = write_batch(tmp_path / DATASET, "august_28_batch_1_rig_3", flies=(1, 2))
    cfg = ec.CombineConfig(root=batch.parent)
    before = {p: p.stat().st_mtime_ns for p in batch.rglob("*.parquet")}
    with ec._batch_read_memo() as memo, contextlib.redirect_stdout(io.StringIO()):
        ec._ensure_angle_percentages(batch, cfg.angle_suffixes)
        rewritten = [p for p, ns in before.items() if p.stat().st_mtime_ns != ns]
        untouched = [p for p in before if p not in rewritten]
        assert rewritten and untouched
        assert not any(p in memo for p in rewritten)
        assert all(p in memo for p in untouched)


def test_combine_writes_invalidate_even_when_stat_cannot_tell(tmp_path, monkeypatch):
    """A rewrite inside one timestamp tick can leave every stat field the memo
    checks unchanged. Pin the signature to a constant -- the worst case -- and
    combine must still see its own writes, because each write invalidates."""

    batch = write_batch(tmp_path / DATASET, "august_28_batch_1_rig_3", flies=(1, 2))
    cfg = ec.CombineConfig(root=batch.parent)
    monkeypatch.setattr(ec, "_stat_signature", lambda st: "unchanged")
    before = {p: p.stat().st_mtime_ns for p in batch.rglob("*.parquet")}
    with ec._batch_read_memo(), contextlib.redirect_stdout(io.StringIO()):
        ec._ensure_angle_percentages(batch, cfg.angle_suffixes)
        rewritten = [p for p, ns in before.items() if p.stat().st_mtime_ns != ns]
        assert rewritten
        for p in rewritten:
            assert "angle_centered_pct" in ec._read_full_table(p).columns


def test_every_table_combine_writes_leaves_the_memo(tmp_path, monkeypatch):
    """Every table write invalidates its entry explicitly -- the angle-column
    rewrites and pass 2's envelope CSVs alike -- so no write is left to the
    stat check."""

    root = _dataset(tmp_path)
    written: set[Path] = set()
    forgotten: set[Path] = set()
    real_to_csv, real_to_parquet = pd.DataFrame.to_csv, pd.DataFrame.to_parquet
    real_forget = ec._forget_table

    def to_csv(self, path_or_buf=None, *args, **kwargs):
        if isinstance(path_or_buf, (str, Path)):
            written.add(Path(path_or_buf).resolve())
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    def to_parquet(self, path=None, *args, **kwargs):
        if isinstance(path, (str, Path)):
            written.add(Path(path).resolve())
        return real_to_parquet(self, path, *args, **kwargs)

    def forget(path):
        forgotten.add(Path(path).resolve())
        real_forget(path)

    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)
    monkeypatch.setattr(ec, "_forget_table", forget)
    _combine(root)

    assert any(p.suffix == ".parquet" for p in written)  # angle columns written back
    assert any(p.parent.name == "angle_distance_rms_envelope" for p in written)
    assert written <= forgotten, sorted(str(p) for p in written - forgotten)[:5]


def test_invalidate_drops_the_entry(tmp_path):
    p = tmp_path / "t_testing_1_fly1_distances.csv"
    p.write_text("a\n1\n", encoding="utf-8")
    with ec._batch_read_memo() as memo:
        assert ec._read_full_table(p)["a"].tolist() == [1]
        assert p in memo
        memo.invalidate(p)
        assert p not in memo
        p.write_text("a\n2\n", encoding="utf-8")
        assert ec._read_full_table(p)["a"].tolist() == [2]


def test_the_whole_combine_is_correct_with_a_blind_stat(tmp_path, monkeypatch):
    """End to end with the stat check disabled: same bytes as the normal run.
    Combine's correctness rests on its explicit invalidation, not on mtimes."""

    normal = _dataset(tmp_path / "normal")
    blind = tmp_path / "blind" / DATASET
    shutil.copytree(normal, blind)
    _combine(normal)
    with monkeypatch.context() as m:
        m.setattr(ec, "_stat_signature", lambda st: "unchanged")
        _combine(blind)
    assert tree_digest(normal) == tree_digest(blind)


def test_a_changed_stat_is_never_served_stale(tmp_path):
    p = tmp_path / "t_testing_1_fly1_distances.parquet"
    pd.DataFrame({"a": [1.0, 2.0]}).to_parquet(p, index=False)
    with ec._batch_read_memo():
        assert ec._read_full_table(p)["a"].tolist() == [1.0, 2.0]
        pd.DataFrame({"a": [3.0, 4.0, 5.0]}).to_parquet(p, index=False)
        assert ec._read_full_table(p)["a"].tolist() == [3.0, 4.0, 5.0]


def test_memo_hands_out_independent_copies(tmp_path):
    p = tmp_path / "t_testing_1_fly1_distances.parquet"
    pd.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]}).to_parquet(p, index=False)
    with ec._batch_read_memo():
        first = ec._read_full_table(p)
        first.loc[0, "a"] = 99.0
        first["extra"] = 1
        second = ec._read_full_table(p)
    assert second["a"].tolist() == [1.0, 2.0]
    assert "extra" not in second.columns


def test_the_memo_lives_for_one_batch(tmp_path, monkeypatch):
    """One memo per batch, emptied when the batch ends; nothing leaks into
    build_wide_csv (which shares read_table) or the next batch."""

    root = _dataset(tmp_path)
    made: list = []

    class Recording(ec._TableReadMemo):
        def __init__(self):
            super().__init__()
            made.append(self)

    monkeypatch.setattr(ec, "_TableReadMemo", Recording)
    _combine(root)

    assert len(made) == 2  # two batches
    assert all(len(m) == 0 for m in made)
    assert ec._READ_MEMO is None


def test_reads_outside_a_batch_go_straight_to_disk(tmp_path, decode_counts):
    p = tmp_path / "t_testing_1_fly1_distances.parquet"
    pd.DataFrame({"a": [1.0]}).to_parquet(p, index=False)
    ec._read_full_table(p)
    ec._read_full_table(p)
    assert decode_counts[p.resolve()] == 2


# ── the header sniff (_read_trial_table with nrows) ───────────────────────


def _mixed_table(path: Path) -> Path:
    pd.DataFrame(
        {
            "frame": np.arange(12, dtype=np.int64),
            "distance_percentage_0_1": np.linspace(0, 1, 12, dtype=np.float32),
            "x_class1": [np.nan] + list(np.linspace(1, 2, 11)),
            "OFM_State": ["before"] * 6 + ["during"] * 6,
            "flag": [True, False] * 6,
        }
    ).to_parquet(path, index=False)
    return path


def test_header_sniff_equals_the_sliced_full_read(tmp_path):
    p = _mixed_table(tmp_path / "t_testing_1_fly1_distances.parquet")
    expected = pd.read_parquet(p, engine="pyarrow").iloc[:5]
    pd.testing.assert_frame_equal(ec._read_trial_table(p, nrows=5), expected)
    with ec._batch_read_memo():
        for _ in range(2):  # miss, then hit
            pd.testing.assert_frame_equal(ec._read_trial_table(p, nrows=5), expected)


def test_header_sniff_shares_the_batch_decode(tmp_path, decode_counts):
    p = _mixed_table(tmp_path / "t_testing_1_fly1_distances.parquet")
    with ec._batch_read_memo():
        ec._read_full_table(p)
        ec._read_trial_table(p, nrows=5)
        ec._read_trial_table(p, nrows=5)
    assert decode_counts[p.resolve()] == 1


def _corrupt_body(src: Path, dst: Path) -> Path:
    """Copy ``src`` and scribble over its data pages, keeping the footer, so
    the schema still reads but the rows do not."""

    raw = bytearray(src.read_bytes())
    footer_len = int.from_bytes(raw[-8:-4], "little")
    body_end = len(raw) - 8 - footer_len
    raw[4:body_end] = b"\xff" * (body_end - 4)
    dst.write_bytes(bytes(raw))
    return dst


def test_a_parquet_with_a_corrupt_body_is_still_rejected_at_discovery(tmp_path):
    """Why the sniff is not a schema-only read: a file whose footer is intact
    but whose rows cannot be decoded is rejected here today. A schema read
    would accept it and leave the failure for pass 1, where it can shift the
    pairing. The memoised full read keeps today's behaviour."""

    fly_dir = tmp_path / DATASET / "august_28_batch_1_rig_3"
    trial = fly_dir / "august_28_batch_1_testing_1"
    trial.mkdir(parents=True)
    good = _mixed_table(trial / "august_28_batch_1_testing_1_fly1_distances.parquet")
    bad = _corrupt_body(good, trial / "august_28_batch_1_testing_1_fly2_distances.parquet")
    assert pq.read_schema(bad).names == pq.read_schema(good).names
    with pytest.raises(Exception):
        pd.read_parquet(bad, engine="pyarrow")

    def located() -> set[str]:
        with contextlib.redirect_stdout(io.StringIO()):
            entries = ec._locate_trials(fly_dir, ("*fly*_distances.parquet",), ec.DIST_COLS)
        return {path.name for _, path, _ in entries}

    assert located() == {good.name}
    with ec._batch_read_memo():
        assert located() == {good.name}
