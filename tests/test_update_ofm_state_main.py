"""End-to-end tests for ``update_ofm_state.main``.

The step annotates every ``RMS_calculations/updated_*`` table with an
``OFM_State`` column (before / during / after the batch's OFM window). The
window comes from the rig's ``output_*.csv``: the FIRST one, in ``iterdir``
order, whose header has an OFM column. It depends only on the batch dir, so it
is probed and read once per batch, lazily, and an RMS table is read only once
its skip check has failed.

``_reference_main`` below is a verbatim copy of the implementation before that
hoist. Every scenario runs both on identical trees and requires byte-identical
output, so the rule above (and its error paths) stays exactly as it was.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.config import ForceSettings, ParallelSettings, Settings, get_main_directories
from fbpipe.steps import update_ofm_state
from fbpipe.steps.update_ofm_state import _find_ofm_col
from fbpipe.utils.frozen_folders import iter_live_batch_dirs
from fbpipe.utils.tables import (
    read_schema_columns,
    read_table,
    resolve_existing,
    table_path,
    write_table,
)

BATCH = "october_05_batch_2"
OLD = 1_700_000_000  # an mtime well before NEW
NEW = 1_800_000_000


def _reference_main(cfg: Settings):
    """update_ofm_state.main as it was before the per-batch hoist (verbatim)."""
    force_recompute = bool(getattr(getattr(cfg, "force", None), "pipeline", False))
    roots = get_main_directories(cfg)
    for root in roots:
        for fly in iter_live_batch_dirs(cfg, root):
            rdir = fly / "RMS_calculations"
            if not rdir.is_dir():
                continue
            outs = [p for p in fly.iterdir() if p.is_file() and p.suffix.lower()==".csv" and p.name.startswith("output_")]
            _rms_by_stem: dict[str, Path] = {}
            for p in sorted(rdir.iterdir()):
                if not (p.is_file() and p.name.startswith("updated_") and p.suffix in (".csv", ".parquet")):
                    continue
                if p.stem not in _rms_by_stem or p.suffix == ".parquet":
                    _rms_by_stem[p.stem] = p
            for f in _rms_by_stem.values():
                dfu = read_table(f)
                out_csv = None
                for o in outs:
                    try:
                        dfo = pd.read_csv(o, nrows=5)
                        if _find_ofm_col(dfo) is not None:
                            out_csv = o; break
                    except Exception:
                        continue
                if out_csv is None:
                    continue
                if not force_recompute:
                    try:
                        existing = resolve_existing(f)
                        if (
                            existing is not None
                            and "OFM_State" in read_schema_columns(existing)
                            and existing.stat().st_mtime >= out_csv.stat().st_mtime
                        ):
                            print(f"[OFM] Skipping up-to-date file: {f.name}")
                            continue
                    except Exception:
                        pass
                dfo = read_table(out_csv)
                col = _find_ofm_col(dfo)
                if col is None:
                    continue
                on_idx = dfo.index[dfo[col].astype(str) != "off"].tolist()
                if not on_idx:
                    continue
                first_on, last_on = min(on_idx), max(on_idx)
                if "Frame" in dfu.columns:
                    dfu["OFM_State"] = dfu["Frame"].apply(lambda x: "before" if x < first_on else ("during" if first_on <= x <= last_on else "after"))
                elif "frame" in dfu.columns:
                    dfu["OFM_State"] = dfu["frame"].apply(lambda x: "before" if x < first_on else ("during" if first_on <= x <= last_on else "after"))
                else:
                    dfu["OFM_State"] = "unknown"
                write_table(dfu, f)
                print(f"[OFM] updated {f.name} using {out_csv.name}")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _cfg(root: Path, *, force: bool) -> Settings:
    return Settings(
        model_path="",
        main_directories=[str(root)],
        auto_discover_flagged=False,
        force=ForceSettings(pipeline=force),
        parallel=ParallelSettings(enabled=False),
    )


def _rig_csv(path: Path, states: list[str], *, ofm_col: str | None = "ActiveOFM") -> Path:
    data = {"Frame": list(range(len(states))), "odor": ["x"] * len(states)}
    if ofm_col is not None:
        data[ofm_col] = states
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _malformed_after_row5(path: Path, states: list[str]) -> Path:
    """Header + 5 good rows parse (the nrows=5 probe passes); row 8 has an extra
    field, so a FULL read raises a ParserError."""
    _rig_csv(path, states)
    lines = path.read_text().splitlines()
    lines[8] = lines[8] + ",extra"
    path.write_text("\n".join(lines) + "\n")
    return path


def _rms(path: Path, n: int = 20, *, frame_col: str | None = "frame", ofm: bool = False) -> Path:
    data = {"timestamp": [i * 0.025 for i in range(n)], "x_class1": [float(i) for i in range(n)]}
    if frame_col is not None:
        data[frame_col] = list(range(n))
    if ofm:
        data["OFM_State"] = ["stale"] * n
    return write_table(pd.DataFrame(data), path)


def _window_states(n: int, first: int, last: int) -> list[str]:
    return ["on" if first <= i <= last else "off" for i in range(n)]


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _build_twice(tmp_path: Path, build) -> tuple[Path, Path]:
    """Build the same dataset tree twice: one for the reference, one for main."""
    roots = []
    for label in ("ref", "new"):
        root = tmp_path / label / "Hex-HexTest-Training-24-1"
        build(root)
        roots.append(root)
    return roots[0], roots[1]


def _run_both(tmp_path: Path, build, *, force: bool):
    ref_root, new_root = _build_twice(tmp_path, build)
    _reference_main(_cfg(ref_root, force=force))
    update_ofm_state.main(_cfg(new_root, force=force))
    return ref_root, new_root


def _assert_same_tree(ref_root: Path, new_root: Path) -> None:
    ref, new = _tree_bytes(ref_root), _tree_bytes(new_root)
    assert sorted(ref) == sorted(new)
    differing = [k for k in ref if ref[k] != new[k]]
    assert differing == []


def _first_ofm_csv_in_iterdir_order(batch: Path) -> Path:
    outs = [p for p in batch.iterdir() if p.is_file() and p.suffix.lower() == ".csv" and p.name.startswith("output_")]
    for o in outs:
        try:
            if _find_ofm_col(pd.read_csv(o, nrows=5)) is not None:
                return o
        except Exception:
            continue
    raise AssertionError("fixture has no OFM csv")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _several_rig_csvs(root: Path) -> None:
    """Three rig CSVs: one without an OFM column, two with DIFFERENT windows,
    so the output shows which one the step picked. Plus a frame-less RMS
    table, a 'Frame'-cased one, and a stale .csv twin of a parquet."""
    b = root / BATCH
    rdir = b / "RMS_calculations"
    rdir.mkdir(parents=True)
    _rig_csv(b / f"output_{BATCH}_testing_1_Hexanol-1_20261005_150000.csv", ["off"] * 20, ofm_col=None)
    _rig_csv(b / f"output_{BATCH}_testing_2_Hexanol-1_20261005_151000.csv", _window_states(20, 3, 6))
    _rig_csv(b / f"output_{BATCH}_testing_3_Hexanol-1_20261005_152000.csv", _window_states(20, 10, 12), ofm_col="Active OFM Pin")
    (b / f"output_{BATCH}_testing_3_Hexanol-1_20261005_152000.txt").write_text("sidecar\n")
    (b / "sensors_output_x.csv").write_text("a,b\n1,2\n")
    _rms(rdir / f"updated_{BATCH}_testing_1_fly1_distances.parquet")
    _rms(rdir / f"updated_{BATCH}_testing_1_fly2_distances.parquet", frame_col="Frame")
    _rms(rdir / f"updated_{BATCH}_testing_2_fly1_distances.parquet", frame_col=None)
    # Stale legacy csv twin: the parquet wins, the csv is left alone.
    pd.DataFrame({"frame": [0, 1], "x_class1": [9.0, 9.0]}).to_csv(
        rdir / f"updated_{BATCH}_testing_1_fly1_distances.csv", index=False
    )
    # A second batch with no RMS dir at all, and a third with an empty one.
    (root / "october_05_batch_3").mkdir()
    _rig_csv(root / "october_05_batch_3" / "output_october_05_batch_3_testing_1_x_20261005_150000.csv", _window_states(20, 1, 2))
    (root / "october_05_batch_3_rig_2" / "RMS_calculations").mkdir(parents=True)


@pytest.mark.parametrize("force", [True, False])
def test_matches_reference_with_several_rig_csvs(tmp_path, force):
    ref_root, new_root = _run_both(tmp_path, _several_rig_csvs, force=force)
    _assert_same_tree(ref_root, new_root)

    # And the window really is the first OFM csv in iterdir order.
    b = new_root / BATCH
    chosen = _first_ofm_csv_in_iterdir_order(b)
    first_on, last_on = (3, 6) if "testing_2" in chosen.name else (10, 12)
    df = read_table(b / "RMS_calculations" / f"updated_{BATCH}_testing_1_fly1_distances.parquet")
    expected = ["before" if i < first_on else ("during" if i <= last_on else "after") for i in range(20)]
    assert df["OFM_State"].tolist() == expected
    df_upper = read_table(b / "RMS_calculations" / f"updated_{BATCH}_testing_1_fly2_distances.parquet")
    assert df_upper["OFM_State"].tolist() == expected
    df_frameless = read_table(b / "RMS_calculations" / f"updated_{BATCH}_testing_2_fly1_distances.parquet")
    assert set(df_frameless["OFM_State"]) == {"unknown"}
    # The parquet won the stem; rewriting it retired its legacy csv twin
    # (write_table's single-source-of-truth rule), in both trees alike.
    assert not (b / "RMS_calculations" / f"updated_{BATCH}_testing_1_fly1_distances.csv").exists()


def _up_to_date_with_malformed_rig_csv(root: Path) -> None:
    """Every RMS table already annotated and newer than the rig CSV; the rig
    CSV itself only parses as far as row 5."""
    b = root / BATCH
    rdir = b / "RMS_calculations"
    rdir.mkdir(parents=True)
    rig = _malformed_after_row5(b / f"output_{BATCH}_training_1_Hexanol-1_20261005_150000.csv", _window_states(20, 3, 6))
    os.utime(rig, (OLD, OLD))
    for name in ("training_1_fly1", "training_1_fly2"):
        p = _rms(rdir / f"updated_{BATCH}_{name}_distances.parquet", ofm=True)
        os.utime(p, (NEW, NEW))


def test_non_force_up_to_date_batch_never_parses_the_rig_csv(tmp_path, monkeypatch):
    """An eager read of the rig CSV above the skip check would crash here."""
    reads: list[Path] = []
    real = update_ofm_state.read_table

    def spy(path, *args, **kwargs):
        reads.append(Path(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(update_ofm_state, "read_table", spy)
    ref_root, new_root = _run_both(tmp_path, _up_to_date_with_malformed_rig_csv, force=False)
    _assert_same_tree(ref_root, new_root)
    assert reads == [], "skipped files must not be read, nor the rig csv"


def test_force_with_malformed_rig_csv_still_raises(tmp_path):
    """Forced, the window is needed, and a rig CSV that cannot be parsed in full
    aborts the step exactly as before."""
    ref_root, new_root = _build_twice(tmp_path, _up_to_date_with_malformed_rig_csv)
    with pytest.raises(pd.errors.ParserError):
        _reference_main(_cfg(ref_root, force=True))
    with pytest.raises(pd.errors.ParserError):
        update_ofm_state.main(_cfg(new_root, force=True))


def _one_stale_file(root: Path) -> None:
    b = root / BATCH
    rdir = b / "RMS_calculations"
    rdir.mkdir(parents=True)
    rig = _rig_csv(b / f"output_{BATCH}_training_1_Hexanol-1_20261005_150000.csv", _window_states(20, 4, 9))
    os.utime(rig, (OLD + 10, OLD + 10))
    fresh = _rms(rdir / f"updated_{BATCH}_training_1_fly1_distances.parquet", ofm=True)
    os.utime(fresh, (NEW, NEW))
    unannotated = _rms(rdir / f"updated_{BATCH}_training_1_fly2_distances.parquet")
    os.utime(unannotated, (NEW, NEW))
    older = _rms(rdir / f"updated_{BATCH}_training_1_fly3_distances.parquet", ofm=True)
    os.utime(older, (OLD, OLD))


def test_non_force_rewrites_only_the_files_that_need_it(tmp_path):
    ref_root, new_root = _run_both(tmp_path, _one_stale_file, force=False)
    _assert_same_tree(ref_root, new_root)
    rdir = new_root / BATCH / "RMS_calculations"
    assert read_table(rdir / f"updated_{BATCH}_training_1_fly1_distances.parquet")["OFM_State"].iloc[0] == "stale"
    assert read_table(rdir / f"updated_{BATCH}_training_1_fly2_distances.parquet")["OFM_State"].iloc[0] == "before"
    assert read_table(rdir / f"updated_{BATCH}_training_1_fly3_distances.parquet")["OFM_State"].iloc[5] == "during"
    assert os.stat(rdir / f"updated_{BATCH}_training_1_fly1_distances.parquet").st_mtime == NEW


def _unparseable_first_candidate(root: Path) -> None:
    """A rig CSV the probe itself cannot parse is passed over, not fatal."""
    b = root / BATCH
    rdir = b / "RMS_calculations"
    rdir.mkdir(parents=True)
    (b / f"output_{BATCH}_testing_1_Hexanol-1_20261005_150000.csv").write_bytes(b"\xff\xfe\x00garbage\x00\n\x00\x01")
    _rig_csv(b / f"output_{BATCH}_testing_2_Hexanol-1_20261005_151000.csv", _window_states(20, 2, 5))
    _rms(rdir / f"updated_{BATCH}_testing_1_fly1_distances.parquet")


@pytest.mark.parametrize("force", [True, False])
def test_unparseable_rig_csv_is_skipped_by_the_probe(tmp_path, force):
    ref_root, new_root = _run_both(tmp_path, _unparseable_first_candidate, force=force)
    _assert_same_tree(ref_root, new_root)
    df = read_table(new_root / BATCH / "RMS_calculations" / f"updated_{BATCH}_testing_1_fly1_distances.parquet")
    assert df["OFM_State"].iloc[3] == "during"


def _no_on_rows(root: Path) -> None:
    b = root / BATCH
    rdir = b / "RMS_calculations"
    rdir.mkdir(parents=True)
    _rig_csv(b / f"output_{BATCH}_testing_9_LightOnly_20261005_150000.csv", ["off"] * 20)
    _rms(rdir / f"updated_{BATCH}_testing_9_fly1_distances.parquet")
    _rms(rdir / f"updated_{BATCH}_testing_9_fly2_distances.parquet")


@pytest.mark.parametrize("force", [True, False])
def test_a_window_with_no_on_rows_annotates_nothing(tmp_path, force):
    ref_root, new_root = _run_both(tmp_path, _no_on_rows, force=force)
    _assert_same_tree(ref_root, new_root)
    for p in (new_root / BATCH / "RMS_calculations").glob("*.parquet"):
        assert "OFM_State" not in read_schema_columns(p)


def _many_rms_files(root: Path) -> None:
    b = root / BATCH
    rdir = b / "RMS_calculations"
    rdir.mkdir(parents=True)
    _rig_csv(b / f"output_{BATCH}_testing_1_noofm_20261005_150000.csv", ["off"] * 20, ofm_col=None)
    _rig_csv(b / f"output_{BATCH}_testing_2_Hexanol-1_20261005_151000.csv", _window_states(20, 3, 6))
    for trial in range(1, 6):
        for slot in (1, 2):
            _rms(rdir / f"updated_{BATCH}_testing_{trial}_fly{slot}_distances.parquet")


def test_force_reads_the_rig_csv_once_per_batch(tmp_path, monkeypatch):
    full_reads: list[Path] = []
    probes: list[Path] = []
    real_read_table = update_ofm_state.read_table
    real_read_csv = pd.read_csv

    def spy_read_table(path, *args, **kwargs):
        full_reads.append(Path(path))
        return real_read_table(path, *args, **kwargs)

    def spy_read_csv(path, *args, **kwargs):
        if kwargs.get("nrows") == 5:  # the header probe, not read_table's parse
            probes.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    root = tmp_path / "Hex-HexTest-Training-24-1"
    _many_rms_files(root)
    monkeypatch.setattr(update_ofm_state, "read_table", spy_read_table)
    monkeypatch.setattr(update_ofm_state.pd, "read_csv", spy_read_csv)

    update_ofm_state.main(_cfg(root, force=True))

    rig_reads = [p for p in full_reads if p.name.startswith("output_")]
    rms_reads = [p for p in full_reads if p.name.startswith("updated_")]
    assert len(rig_reads) == 1
    assert len(rms_reads) == 10
    # One header probe per rig csv up to (and including) the chosen one.
    assert len(probes) <= 2
    for p in (root / BATCH / "RMS_calculations").glob("*.parquet"):
        assert "OFM_State" in read_schema_columns(p)


def test_batch_without_an_ofm_csv_reads_no_rms_table(tmp_path, monkeypatch):
    """Nothing to annotate: the RMS tables are not opened at all (the old code
    read every one first, so a corrupt table aborted a batch with no OFM)."""
    root = tmp_path / "Hex-HexTest-Training-24-1"
    b = root / BATCH
    rdir = b / "RMS_calculations"
    rdir.mkdir(parents=True)
    _rig_csv(b / f"output_{BATCH}_testing_1_noofm_20261005_150000.csv", ["off"] * 20, ofm_col=None)
    (rdir / f"updated_{BATCH}_testing_1_fly1_distances.parquet").write_bytes(b"not a parquet")
    reads: list[Path] = []
    real = update_ofm_state.read_table

    def spy(path, *args, **kwargs):
        reads.append(Path(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(update_ofm_state, "read_table", spy)
    update_ofm_state.main(_cfg(root, force=True))
    assert reads == []
