"""Auto-freeze's input signature: what counts as "something new was added".

New: a batch folder appearing or disappearing, and a raw rig file directly in a
batch folder (``output_*``, ``sensors_output_*``, ``session_metadata.txt``,
``odor_vials.csv``, a hand note) added, changed or removed -- by content.
Never new: anything the pipeline writes (trial folders and every other
directory below a batch, ``*global_distance_stats_class_*.json`` beside the
rig files), videos (the secure cleanup deletes processed ones; pending videos
are judged by yolo_discovery instead), dot/tilde files, files directly in the
dataset root, and a file only touched (same bytes).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from fbpipe import autofreeze as af

STEM = "output_october_06_batch_1_testing_1_Citral-1_20261006_110000"


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "IAA-Sensitivity-12-1"
    batch = root / "october_06_batch_1"
    batch.mkdir(parents=True)
    for name in (f"{STEM}.csv", f"{STEM}.txt", f"sensors_{STEM}.csv", "session_metadata.txt", "odor_vials.csv"):
        (batch / name).write_text(f"{name}\n")
    return root


def _diff(root: Path, before: dict) -> af.InputDiff:
    return af.diff_inputs(before, af.scan_inputs(root, before))


def test_the_signature_covers_batch_folders_and_their_raw_files(tmp_path):
    root = _root(tmp_path)
    sig = af.scan_inputs(root)
    assert sig["dirs"] == ["october_06_batch_1"]
    assert sorted(sig["files"]) == sorted(
        f"october_06_batch_1/{n}"
        for n in (f"{STEM}.csv", f"{STEM}.txt", f"sensors_{STEM}.csv", "session_metadata.txt", "odor_vials.csv")
    )
    assert af.diff_inputs(sig, af.scan_inputs(root, sig)).is_new is False


def test_a_new_batch_folder_is_new(tmp_path):
    root = _root(tmp_path)
    before = af.scan_inputs(root)
    (root / "october_07_batch_1").mkdir()
    diff = _diff(root, before)
    assert diff.is_new and diff.added_dirs == ["october_07_batch_1"]
    assert "+1 batch folder(s) (october_07_batch_1)" in diff.describe()


def test_a_removed_batch_folder_is_new(tmp_path):
    root = _root(tmp_path)
    (root / "october_07_batch_1").mkdir()
    before = af.scan_inputs(root)
    (root / "october_07_batch_1").rmdir()
    diff = _diff(root, before)
    assert diff.is_new and diff.removed_dirs == ["october_07_batch_1"]


def test_a_new_raw_file_is_new(tmp_path):
    root = _root(tmp_path)
    before = af.scan_inputs(root)
    (root / "october_06_batch_1" / "output_october_06_batch_1_testing_2_X_20261006_111000.csv").write_text("x")
    diff = _diff(root, before)
    assert diff.is_new and len(diff.added) == 1


def test_a_changed_raw_file_is_new(tmp_path):
    root = _root(tmp_path)
    before = af.scan_inputs(root)
    (root / "october_06_batch_1" / "session_metadata.txt").write_text("Born (approx): 2026-10-01\n")
    diff = _diff(root, before)
    assert diff.is_new and diff.changed == ["october_06_batch_1/session_metadata.txt"]


def test_a_removed_raw_file_is_new(tmp_path):
    root = _root(tmp_path)
    before = af.scan_inputs(root)
    (root / "october_06_batch_1" / "odor_vials.csv").unlink()
    diff = _diff(root, before)
    assert diff.is_new and diff.removed == ["october_06_batch_1/odor_vials.csv"]


def test_a_touched_file_is_not_new(tmp_path):
    root = _root(tmp_path)
    before = af.scan_inputs(root)
    target = root / "october_06_batch_1" / f"{STEM}.csv"
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    diff = _diff(root, before)
    assert diff.is_new is False
    assert diff.touched == [f"october_06_batch_1/{STEM}.csv"]


def test_an_mtime_preserving_replacement_is_caught(tmp_path):
    """``cp -p``/``rsync -t`` keep the mtime; the new inode/ctime still differ,
    so the file is re-hashed and its new content seen."""
    root = _root(tmp_path)
    target = root / "october_06_batch_1" / f"{STEM}.csv"
    before = af.scan_inputs(root)
    replacement = tmp_path / "replacement.csv"
    replacement.write_text(f"{STEM}.csv\n".replace("output", "OUTPUT"))  # same size, other bytes
    st = target.stat()
    os.utime(replacement, ns=(st.st_atime_ns, st.st_mtime_ns))
    shutil.copy2(replacement, target.with_suffix(".tmp"))
    os.replace(target.with_suffix(".tmp"), target)
    assert target.stat().st_mtime_ns == st.st_mtime_ns
    diff = _diff(root, before)
    assert diff.changed == [f"october_06_batch_1/{STEM}.csv"]


@pytest.mark.parametrize(
    "rel",
    [
        "october_06_batch_1/fly1_global_distance_stats_class_0.json",  # distance_stats
        "october_06_batch_1/global_distance_stats_class_2.json",
        "october_06_batch_1/batch_manifest.json",  # curate_yolo_dataset
        "october_06_batch_1/october_06_batch_1_testing_1/october_06_batch_1_testing_1_distances_merged.parquet",
        "october_06_batch_1/angle_distance_rms_envelope/testing_1_fly1.csv",
        "october_06_batch_1/RMS_calculations/updated_x.parquet",
        f"october_06_batch_1/{STEM}.mp4",  # videos never count
        f"october_06_batch_1/{STEM}.h264",
        "october_06_batch_1/.~lock.output_x.csv#",
        "october_06_batch_1/~scratch.txt",
        "remux_20261006.log",  # directly in the dataset root
        ".Trash-1000/files/x.csv",  # a hidden folder is no batch
    ],
)
def test_pipeline_writes_and_videos_are_never_new(tmp_path, rel):
    root = _root(tmp_path)
    before = af.scan_inputs(root)
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("written by the pipeline")
    assert _diff(root, before).is_new is False


def test_deleting_a_processed_video_is_not_new(tmp_path):
    root = _root(tmp_path)
    video = root / "october_06_batch_1" / f"{STEM}.mp4"
    video.write_bytes(b"v")
    before = af.scan_inputs(root)
    video.unlink()
    assert _diff(root, before).is_new is False


def test_a_quiet_scan_reads_no_file_contents(tmp_path, monkeypatch):
    root = _root(tmp_path)
    before = af.scan_inputs(root)
    opened = []
    real = af._file_hash
    monkeypatch.setattr(af, "_file_hash", lambda p: opened.append(p) or real(p))
    after = af.scan_inputs(root, before)
    assert opened == []
    (root / "october_06_batch_1" / "odor_vials.csv").write_text("changed\n")
    af.scan_inputs(root, after)
    assert [Path(p).name for p in opened] == ["odor_vials.csv"]


def test_a_missing_root_scans_empty(tmp_path):
    assert af.scan_inputs(tmp_path / "gone") == {"dirs": [], "files": {}}


def test_processed_batches_need_a_per_fly_table(tmp_path):
    root = _root(tmp_path)
    trial = root / "october_06_batch_1" / "october_06_batch_1_testing_1"
    trial.mkdir()
    (trial / "october_06_batch_1_testing_1_distances_merged.parquet").write_bytes(b"x")
    assert af.processed_batches(root) == set()
    (trial / "october_06_batch_1_testing_1_fly1_distances.parquet").write_bytes(b"x")
    assert af.processed_batches(root) == {"october_06_batch_1"}
