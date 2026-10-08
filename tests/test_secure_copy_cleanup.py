"""secure_copy_and_cleanup: copy everything, delete only videos YOLO has finished.

The copy phase is unchanged: every file absent from the secured mirror is
copied, an existing target is never overwritten. The cleanup used to delete
every local ``.mp4``/``.h264`` with a secured copy, processed or not, so a
video that landed after its dataset's YOLO slot was mirrored and deleted
without ever being inferred (IAA-Sensitivity-12-1 october_06_batch_1 lost 7
trials on 2026-10-06; the copies restored on 2026-10-07 went the same way).

Now a local video is deleted only when its secured copy exists AND YOLO fully
processed its trial (fbpipe.utils.yolo_discovery.trial_tables_complete, by
yolo_infer's own naming). Every kept video is logged with the reason.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import envelope_combined as ec  # noqa: E402

STEM = "output_may_01_batch_1_testing_{n}_Hexanol-1_20260501_10{n:02d}00"


def _trial(fly: Path, n: int, *, eyes=(True,), per_fly=True) -> Path:
    """The trial folder yolo_infer writes for STEM n, with its tables."""
    name = f"may_01_batch_1_testing_{n}"
    trial = fly / name
    trial.mkdir(parents=True, exist_ok=True)
    cols = {"frame": [0, 1]}
    for i, has in enumerate(eyes):
        cols[f"eye_{i}_x"] = [1.0, 2.0] if has else [np.nan, np.nan]
    pd.DataFrame(cols).to_parquet(trial / f"{name}_distances_merged.parquet")
    if per_fly:
        for i, has in enumerate(eyes):
            if has:
                pd.DataFrame({"frame": [0]}).to_parquet(trial / f"{name}_fly{i + 1}_distances.parquet")
    return trial


def _scene(tmp_path):
    source = tmp_path / "Hex-Training"
    dest = tmp_path / "secured"
    fly = source / "may_01_batch_1"  # month-named -> eligible for cleanup
    fly.mkdir(parents=True)
    return source, dest, fly


def _video(fly: Path, n: int, ext=".mp4") -> Path:
    path = fly / f"{STEM.format(n=n)}{ext}"
    path.write_bytes(b"video")
    return path


def test_processed_videos_are_pruned_after_their_secured_copy(tmp_path):
    source, dest, fly = _scene(tmp_path)
    mp4, h264 = _video(fly, 1), _video(fly, 1, ".h264")
    trial = _trial(fly, 1)
    annotated = trial / "may_01_batch_1_testing_1_distance_annotated.mp4"
    annotated.write_bytes(b"annotated")
    keep = fly / f"{STEM.format(n=1)}.csv"  # non-video, must be preserved
    keep.write_bytes(b"x")

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)

    secured_fly = dest / "Hex-Training" / "may_01_batch_1"
    for f in (mp4, h264, keep):
        assert (secured_fly / f.name).is_file()
    assert (secured_fly / trial.name / annotated.name).is_file()
    assert not mp4.exists()
    assert not h264.exists()
    assert not annotated.exists()
    assert keep.exists()


def test_an_unprocessed_video_is_copied_but_kept(tmp_path, capsys):
    source, dest, fly = _scene(tmp_path)
    mp4 = _video(fly, 2)

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)

    assert (dest / "Hex-Training" / "may_01_batch_1" / mp4.name).is_file()
    assert mp4.exists()
    out = capsys.readouterr().out
    assert f"Keeping video (YOLO has not processed it: no trial folder): {mp4}" in out


@pytest.mark.parametrize(
    "build,reason",
    [
        (lambda fly: (fly / "may_01_batch_1_testing_3").mkdir(), "no merged table"),
        (lambda fly: _trial(fly, 3, eyes=(True, True), per_fly=False), "no per-fly table for fly1, fly2"),
    ],
    ids=["empty trial folder", "per-fly tables missing"],
)
def test_an_incomplete_trial_keeps_its_video(tmp_path, capsys, build, reason):
    source, dest, fly = _scene(tmp_path)
    mp4, h264 = _video(fly, 3), _video(fly, 3, ".h264")
    build(fly)

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)

    assert mp4.exists() and h264.exists()
    assert reason in capsys.readouterr().out


def test_an_eyeless_trial_counts_as_processed(tmp_path):
    """yolo_infer writes no per-fly table for a slot with no eye data."""
    source, dest, fly = _scene(tmp_path)
    mp4 = _video(fly, 4)
    _trial(fly, 4, eyes=(False, False))

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)
    assert not mp4.exists()


def test_a_lone_h264_waits_for_its_remux_and_inference(tmp_path, capsys):
    source, dest, fly = _scene(tmp_path)
    h264 = _video(fly, 5, ".h264")

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)
    assert h264.exists()
    assert "Keeping video (YOLO has not processed it" in capsys.readouterr().out


def test_an_annotated_video_in_an_incomplete_trial_is_kept(tmp_path):
    source, dest, fly = _scene(tmp_path)
    trial = fly / "may_01_batch_1_testing_6"
    trial.mkdir()
    annotated = trial / "may_01_batch_1_testing_6_distance_annotated.mp4"
    annotated.write_bytes(b"half written")

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)
    assert annotated.exists()


def test_no_secured_copy_keeps_the_video_as_before(tmp_path, capsys, monkeypatch):
    source, dest, fly = _scene(tmp_path)
    mp4 = _video(fly, 7)
    _trial(fly, 7)
    real_copy2 = ec.shutil.copy2

    def _copy_all_but_videos(src, dst, *a, **k):
        if str(src).endswith(".mp4"):
            return None
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr(ec.shutil, "copy2", _copy_all_but_videos)
    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)
    assert mp4.exists()
    assert f"Keeping video (no secured copy): {mp4}" in capsys.readouterr().out


def test_an_existing_secured_copy_is_never_overwritten(tmp_path):
    source, dest, fly = _scene(tmp_path)
    mp4 = _video(fly, 8)
    target = dest / "Hex-Training" / "may_01_batch_1" / mp4.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"first version")

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)
    assert target.read_bytes() == b"first version"
    assert mp4.exists()  # unprocessed


def test_non_month_folders_are_never_cleaned(tmp_path):
    source, dest, _ = _scene(tmp_path)
    flagged = source / "flagged" / "may_20_batch_2"
    flagged.mkdir(parents=True)
    legacy = flagged / "trial.mp4"
    legacy.write_bytes(b"x")

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)
    assert legacy.exists()


def test_cleanup_off_deletes_nothing(tmp_path):
    source, dest, fly = _scene(tmp_path)
    mp4 = _video(fly, 9)
    _trial(fly, 9)
    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=False)
    assert mp4.exists()
