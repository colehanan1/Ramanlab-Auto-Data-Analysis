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


def test_a_trial_recorded_more_than_once_keeps_every_session(tmp_path, capsys):
    """A restarted session names one trial folder from every attempt, and that
    folder may hold an ABORTED attempt: EB-Control-24-1 july_17_batch_1_rig_3
    training_1 was built from its 09:39 attempt (no eyes detected) while the
    real 11:51 session was never tracked. A complete folder therefore proves
    nothing about which session YOLO saw, so every session's video is kept."""
    source, dest, fly = _scene(tmp_path)
    first = fly / "output_may_01_batch_1_testing_2_Hexanol-1_20260501_093939.mp4"
    last = fly / "output_may_01_batch_1_testing_2_Hexanol-1_20260501_115131.mp4"
    for video in (first, last):
        video.write_bytes(b"video")
    capture = last.with_suffix(".h264")
    capture.write_bytes(b"raw")
    _trial(fly, 2)  # complete tables
    single = _video(fly, 1)
    _trial(fly, 1)

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)

    assert first.exists() and last.exists() and capture.exists()
    assert not single.exists()  # one session: pruned as before
    out = capsys.readouterr().out
    assert "recorded more than once (2 sessions)" in out
    assert (dest / "Hex-Training" / "may_01_batch_1" / first.name).is_file()


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
    real_replace = ec.os.replace

    def _land_all_but_videos(src, dst, *a, **k):
        if str(dst).endswith(".mp4"):
            return None   # the video's copy never lands under its real name
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(ec.os, "replace", _land_all_but_videos)
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


def test_a_processed_video_whose_secured_copy_differs_in_size_is_kept(tmp_path, capsys):
    """The copy phase never overwrites an existing target, so a secured copy cut
    short (an interrupted copy, a file still growing when it was copied) stays
    short for good. Deleting the local video then would leave only the broken
    copy: a processed video is deleted only when its secured copy has its size."""
    source, dest, fly = _scene(tmp_path)
    mp4 = _video(fly, 4)
    _trial(fly, 4)
    secured = dest / "Hex-Training" / "may_01_batch_1" / mp4.name
    secured.parent.mkdir(parents=True)
    secured.write_bytes(b"vid")  # truncated: 3 of 5 bytes

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)

    assert mp4.exists() and secured.read_bytes() == b"vid"  # copy behaviour unchanged
    out = capsys.readouterr().out
    assert f"Keeping video (its secured copy is 3 bytes, the local file 5): {mp4}" in out


class _Killed(Exception):
    """Stands in for the run being killed partway through a copy."""


def test_a_copy_cut_short_leaves_nothing_under_the_real_name(tmp_path, monkeypatch):
    """2026-11-09: a run killed mid-copy could leave a short file under the real name;
    the next run skips any existing target, and the file-server and Box syncs pass the
    short copy on. Copies now land under a temporary name and are renamed when whole."""
    source, dest, fly = _scene(tmp_path)
    table = fly / "may_01_batch_1_testing_1" / "trial.csv"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"x" * 1000)
    real_copy2 = ec.shutil.copy2

    def killed_halfway(src, dst, *a, **k):
        Path(dst).write_bytes(Path(src).read_bytes()[:400])
        raise _Killed

    monkeypatch.setattr(ec.shutil, "copy2", killed_halfway)
    with pytest.raises(_Killed):
        ec.secure_copy_and_cleanup([str(source)], str(dest))
    target = dest / "Hex-Training" / table.relative_to(source)
    assert not target.exists()

    monkeypatch.setattr(ec.shutil, "copy2", real_copy2)
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    assert target.read_bytes() == b"x" * 1000
    assert [p.name for p in target.parent.iterdir()] == ["trial.csv"]


def test_already_secured_files_are_counted_not_listed(tmp_path, capsys):
    """One line per already-secured file was ~900k lines a night in the nightly log."""
    source, dest, fly = _scene(tmp_path)
    for n in (1, 2, 3):
        (fly / f"note_{n}.txt").write_text("n")
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    capsys.readouterr()
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    out = capsys.readouterr().out
    assert "Skipping (already exists)" not in out
    assert "3 file(s) already in secured storage" in out


# ── hardening (2026-11-09): leftover partials, bad source files, os.sync ──


def test_a_leftover_partial_is_removed_before_the_copy(tmp_path):
    source, dest, fly = _scene(tmp_path)
    (fly / "trial.csv").write_text("good")
    leftover = dest / "Hex-Training" / fly.name / ".trial.csv.partial"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("junk from a killed run")
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    assert not leftover.exists()
    assert (leftover.parent / "trial.csv").read_text() == "good"


def _eio_for(monkeypatch, bad_name):
    real = ec.shutil.copy2

    def flaky(src, dst, *a, **k):
        if Path(src).name == bad_name:
            Path(dst).write_bytes(b"half")  # a partial exists when the read fails
            raise OSError(5, "Input/output error")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(ec.shutil, "copy2", flaky)


def test_an_unreadable_source_file_does_not_stop_the_copy(tmp_path, monkeypatch, capsys):
    from fbpipe.utils import notify

    sent = []
    monkeypatch.setattr(notify, "ntfy_notify", lambda *a, **k: sent.append((a, k)))
    source, dest, fly = _scene(tmp_path)
    (fly / "a_bad.csv").write_text("bad")
    (fly / "b_good.csv").write_text("good")
    _eio_for(monkeypatch, "a_bad.csv")
    ec.secure_copy_and_cleanup([str(source)], str(dest))  # must not raise
    out_dir = dest / "Hex-Training" / fly.name
    assert (out_dir / "b_good.csv").read_text() == "good"
    assert not (out_dir / "a_bad.csv").exists()
    assert not (out_dir / ".a_bad.csv.partial").exists()
    out = capsys.readouterr().out
    assert "a_bad.csv" in out and "Input/output error" in out
    assert len(sent) == 1 and "a_bad.csv" in str(sent[0])


def test_the_alert_lists_at_most_twenty_failures(tmp_path, monkeypatch):
    from fbpipe.utils import notify

    sent = []
    monkeypatch.setattr(notify, "ntfy_notify", lambda *a, **k: sent.append((a, k)))
    source, dest, fly = _scene(tmp_path)
    for n in range(25):
        (fly / f"bad_{n:02d}.csv").write_text("x")
    monkeypatch.setattr(
        ec.shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError(5, "EIO"))
    )
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    assert len(sent) == 1
    text = " ".join(str(x) for x in sent[0][0])
    assert sum(f"bad_{n:02d}.csv" in text for n in range(25)) == 20
    assert "25" in text


def test_no_failures_means_no_alert(tmp_path, monkeypatch):
    from fbpipe.utils import notify

    sent = []
    monkeypatch.setattr(notify, "ntfy_notify", lambda *a, **k: sent.append(a))
    source, dest, fly = _scene(tmp_path)
    (fly / "ok.csv").write_text("ok")
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    assert sent == []


def test_a_failed_copy_keeps_its_video(tmp_path, monkeypatch):
    """No equal-size copy under the real name -> the local video stays."""
    from fbpipe.utils import notify

    monkeypatch.setattr(notify, "ntfy_notify", lambda *a, **k: None)
    source, dest, fly = _scene(tmp_path)
    _trial(fly, 1)
    video = _video(fly, 1)
    _eio_for(monkeypatch, video.name)
    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)
    assert video.exists()


def test_os_sync_runs_once_after_copying_and_before_any_delete(tmp_path, monkeypatch):
    source, dest, fly = _scene(tmp_path)
    _trial(fly, 1)
    video = _video(fly, 1)
    (fly / "t.csv").write_text("t")
    events = []
    monkeypatch.setattr(ec.os, "sync", lambda: events.append(("sync", video.exists())))
    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)
    assert events == [("sync", True)]  # once, and the video was still there
    assert not video.exists()


def test_a_source_still_changing_while_copied_is_not_landed(tmp_path, monkeypatch):
    """A complete-looking copy of a file that grew mid-copy would never be redone."""
    source, dest, fly = _scene(tmp_path)
    growing = fly / "growing.csv"
    growing.write_bytes(b"a" * 100)
    real_copy2 = ec.shutil.copy2

    def copy_then_grow(src, dst, *a, **k):
        out = real_copy2(src, dst, *a, **k)
        if str(src).endswith("growing.csv"):
            with open(src, "ab") as fh:
                fh.write(b"b" * 50)
        return out

    monkeypatch.setattr(ec.shutil, "copy2", copy_then_grow)
    monkeypatch.setattr("fbpipe.utils.notify.ntfy_notify", lambda *a, **k: None)
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    target = dest / "Hex-Training" / growing.relative_to(source)
    assert not target.exists()
    monkeypatch.setattr(ec.shutil, "copy2", real_copy2)
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    assert target.stat().st_size == 150


def test_a_folder_that_cannot_be_made_costs_only_its_files(tmp_path, monkeypatch):
    source, dest, fly = _scene(tmp_path)
    (fly / "bad_dir").mkdir()
    (fly / "bad_dir" / "x.csv").write_text("x")
    (fly / "ok.csv").write_text("y")
    real_mkdir = Path.mkdir

    def mkdir(self, *a, **k):
        if self.name == "bad_dir" and "secured" in str(self):
            raise OSError(36, "File name too long", str(self))
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr("fbpipe.utils.notify.ntfy_notify", lambda *a, **k: None)
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    assert (dest / "Hex-Training" / "may_01_batch_1" / "ok.csv").read_text() == "y"


def test_the_failure_alert_fits_in_one_ntfy_message(tmp_path, monkeypatch):
    """ntfy turns a body over 4,096 bytes into an attachment instead of showing it."""
    source, dest, fly = _scene(tmp_path)
    for n in range(30):
        (fly / (f"trial_{n:02d}_" + "x" * 150 + ".csv")).write_text("x")

    def eio(src, dst, *a, **k):
        raise OSError(5, "Input/output error", str(src))

    sent = []
    monkeypatch.setattr(ec.shutil, "copy2", eio)
    monkeypatch.setattr("fbpipe.utils.notify.ntfy_notify",
                        lambda title, message, **k: sent.append((title, message)))
    ec.secure_copy_and_cleanup([str(source)], str(dest))
    assert len(sent) == 1
    title, message = sent[0]
    assert len(message.encode()) < 4096
    assert "30" in title + message
