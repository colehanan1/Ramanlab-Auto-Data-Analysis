"""fbpipe.utils.yolo_discovery: one definition of "a video YOLO would process".

yolo_infer.main, the auto-freeze plan (a video YOLO would still process is
"something new") and the secure cleanup (never delete a video YOLO has not
fully processed) must agree on it exactly, so the rules live in one torch-free
module:

* a raw video is a ``.mp4``/``.avi`` directly inside a live batch folder;
* its trial folder is ``<batch>/<fields 1-6 of the stem>`` (odor and stamp
  dropped), and YOLO skips a video whose trial folder exists;
* a trial recorded more than once (a restarted session) names one trial folder
  from every attempt: YOLO processes the session the trial's metadata is read
  from (trial_metadata.find_sidecar: most recent sidecar), never the last name
  in sorted order -- the odor label precedes the stamp -- and none while that
  session has no video; without rig files, the latest stamp decides;
* "fully processed" = the trial folder holds the merged table and a per-fly
  table for every fly slot the merged table has eye data for.
"""

from __future__ import annotations

import datetime as dt
import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.steps import yolo_infer
from fbpipe.utils import yolo_discovery as yd
from fbpipe.utils.frozen_folders import FolderRule
from fbpipe.utils.trial_metadata import find_sidecar


def _inline_name(video: Path) -> str:
    """yolo_infer.main's own naming, as it was written inline before the helper."""
    base = video.stem
    parts = base.split("_")
    return "_".join(parts[1:7]) if len(parts) >= 7 else base


@pytest.mark.parametrize(
    "name",
    [
        "output_october_06_batch_1_training_1_IsoamylAcetate-1_20261006_120331.mp4",
        "output_july_17_batch_1_training_1_EthylButyrate_20260717_115131.mp4",
        "output_may_20_batch_2_testing_9_LightOnly_20260520_101010_preprocessed.mp4",
        "short_name.mp4",
        "a_b_c_d_e_f.avi",
        "a_b_c_d_e_f_g.AVI",
    ],
)
def test_trial_folder_name_matches_yolo_infers_naming(name):
    assert yd.trial_folder_name(Path("/b") / name) == _inline_name(Path(name))


def test_yolo_infer_uses_the_shared_helpers():
    src = inspect.getsource(yolo_infer.main)
    assert "trial_folder_name(" in src
    assert "raw_videos(" in src
    assert "superseded_videos(" in src


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


V = "output_october_06_batch_1_training_{n}_IsoamylAcetate-1_20261006_12{n:02d}00{ext}"


def test_raw_videos_are_mp4_and_avi_directly_in_the_batch_sorted(tmp_path):
    batch = tmp_path / "october_06_batch_1"
    b = _touch(batch / V.format(n=2, ext=".MP4"))
    a = _touch(batch / V.format(n=1, ext=".mp4"))
    c = _touch(batch / V.format(n=3, ext=".avi"))
    _touch(batch / V.format(n=4, ext=".h264"))  # needs a remux first
    _touch(batch / "october_06_batch_1_training_1" / "nested.mp4")
    _touch(batch / V.format(n=5, ext=".csv"))
    assert yd.raw_videos(batch) == sorted([a, b, c], key=str)


def test_latest_session_supersedes_the_earlier_ones(tmp_path):
    batch = tmp_path / "july_17_batch_1_rig_3"
    stem = "output_july_17_batch_1_training_1_EthylButyrate_20260717_{}.mp4"
    early = _touch(batch / stem.format("093939"))
    mid = _touch(batch / stem.format("102030"))
    late = _touch(batch / stem.format("115131"))
    other = _touch(batch / "output_july_17_batch_1_training_2_EthylButyrate_20260717_120000.mp4")
    videos = yd.raw_videos(batch)
    assert yd.superseded_videos(videos) == {early, mid}
    assert late in videos and other in videos


# Hex-Control-24-0.01 april_17_batch_1 testing_3 was recorded under Linalool at
# 14:13:50, then again under 3-Octonol at 14:30:27. The odor label precedes the
# stamp, so the LATER session sorts FIRST ('3' < 'L').
APR_TRIAL = "april_17_batch_1_testing_3"
APR_EARLY = "output_april_17_batch_1_testing_3_Linalool_20260417_141350"
APR_LATE = "output_april_17_batch_1_testing_3_3-Octonol_20260417_143027"
EB_STEM = "output_july_17_batch_1_training_1_EthylButyrate_20260717_{}"


def _written(path: Path, when: dt.datetime) -> Path:
    stamp = when.timestamp()
    os.utime(path, (stamp, stamp))
    return path


def _session(batch: Path, stem: str, when: dt.datetime, *, video=".mp4", sidecar=True):
    """One recording session as the rig leaves it: rig CSV, sidecar, video."""
    files = [batch / f"{stem}.csv"] + ([batch / f"{stem}.txt"] if sidecar else [])
    for path in files:
        _written(_touch(path), when)
    return _written(_touch(batch / f"{stem}{video}"), when) if video else None


def _metadata_stems(batch: Path, trial: str) -> tuple:
    txt, csv = find_sidecar(batch / trial)
    return (txt.stem if txt else None, csv.stem if csv else None)


def test_a_restart_under_another_odor_label_keeps_the_session_its_metadata_reads(tmp_path):
    """Picking the last name would process the EARLIER Linalool session while
    the trial's metadata (odor, frames, odor window) comes from 3-Octonol."""
    batch = tmp_path / "april_17_batch_1"
    early = _session(batch, APR_EARLY, dt.datetime(2026, 4, 17, 14, 16, 20))
    late = _session(batch, APR_LATE, dt.datetime(2026, 4, 17, 14, 32, 57))
    videos = yd.raw_videos(batch)
    assert videos == [late, early]  # sorted by name: the later session first
    assert yd.superseded_videos(videos) == {early}
    assert yd.latest_session(videos, early) == late
    assert yd.latest_session(videos, late) == late
    assert _metadata_stems(batch, APR_TRIAL) == (late.stem, late.stem)


def test_without_rig_files_the_latest_stamp_decides_not_the_name(tmp_path):
    """Only videos (nothing to read metadata from): the latest stamp is the
    restart. The videos' own times are ignored -- a copy back from the
    secured mirror rewrites them."""
    batch = tmp_path / "april_17_batch_1"
    late = _written(_touch(batch / f"{APR_LATE}.mp4"), dt.datetime(2026, 10, 7, 9, 0, 0))
    early = _written(_touch(batch / f"{APR_EARLY}.mp4"), dt.datetime(2026, 10, 7, 9, 5, 0))
    assert yd.superseded_videos(yd.raw_videos(batch)) == {early}
    assert yd.latest_session(yd.raw_videos(batch), early) == late


def test_the_metadata_session_wins_over_the_stamps(tmp_path):
    """Whatever made an earlier session's sidecar the most recent, the trial's
    metadata is read from it: its tables must come from that same recording."""
    batch = tmp_path / "july_17_batch_1_rig_3"
    later = _session(batch, EB_STEM.format("115131"), dt.datetime(2026, 7, 17, 11, 53, 1))
    earlier = _session(batch, EB_STEM.format("093939"), dt.datetime(2026, 7, 17, 12, 30, 0))
    assert yd.superseded_videos(yd.raw_videos(batch)) == {later}
    assert _metadata_stems(batch, "july_17_batch_1_training_1") == (earlier.stem, earlier.stem)


def test_an_aborted_attempt_without_a_sidecar_never_wins(tmp_path):
    """Hex-Control-24-0.01 may_13_batch_1 training_1: the 09:24 attempt was
    aborted (no sidecar .txt, an 8 KB rig CSV); 10:54 is the real session."""
    batch = tmp_path / "may_13_batch_1"
    stem = "output_may_13_batch_1_training_1_Hexanol_20260513_{}"
    aborted = _session(batch, stem.format("092405"), dt.datetime(2026, 5, 13, 9, 24, 8), sidecar=False)
    real = _session(batch, stem.format("105423"), dt.datetime(2026, 5, 13, 10, 56, 53))
    assert yd.superseded_videos(yd.raw_videos(batch)) == {aborted}
    assert yd.latest_session(yd.raw_videos(batch), aborted) == real


def test_no_video_is_processed_while_the_metadata_session_has_none(tmp_path):
    """EB-Control-24-1 july_17_batch_1_rig_3 training_1, had only the aborted
    09:39 video been local: building the folder from it would pair its tables
    with the 11:51 metadata. Nothing is processed, and nothing is pending."""
    root = tmp_path / "EB-Control-24-1"
    batch = root / "july_17_batch_1_rig_3"
    aborted = _session(batch, EB_STEM.format("093939"), dt.datetime(2026, 7, 17, 9, 41, 10))
    _session(batch, EB_STEM.format("115131"), dt.datetime(2026, 7, 17, 11, 53, 1), video=None)
    videos = yd.raw_videos(batch)
    assert videos == [aborted]
    assert yd.superseded_videos(videos) == {aborted}
    assert yd.latest_session(videos, aborted) is None
    report = yd.trial_report(_Cfg(), root)
    assert report.pending == []
    assert report.collisions == {"july_17_batch_1_rig_3/july_17_batch_1_training_1": 2}


def test_one_sessions_several_videos_keep_yolo_infers_first_pick(tmp_path):
    """Two files of ONE recording (an .avi beside the .mp4, a _preprocessed
    copy) are no restart: yolo_infer keeps processing the first in its order."""
    batch = tmp_path / "july_17_batch_1_rig_3"
    stem = EB_STEM.format("115131")
    when = dt.datetime(2026, 7, 17, 11, 53, 1)
    avi = _session(batch, stem, when, video=".avi")
    mp4 = _written(_touch(batch / f"{stem}.mp4"), when)
    pre = _written(_touch(batch / f"{stem}_preprocessed.mp4"), when)
    videos = yd.raw_videos(batch)
    assert videos == [avi, mp4, pre]
    assert yd.superseded_videos(videos) == {mp4, pre}
    assert yd.latest_session(videos, pre) == avi


class _Cfg:
    def __init__(self, overrides=None, cutoff=None):
        self.freeze_folders_before = cutoff
        self.freeze_folders_born_on_or_after = None
        self.keep_folders_born_on_or_after = None
        self.dataset_overrides = dict(overrides or {})


def test_pending_videos_mirror_yolo_discovery(tmp_path):
    root = tmp_path / "IAA-Sensitivity-12-1"
    batch = root / "october_06_batch_1"
    done = _touch(batch / V.format(n=1, ext=".mp4"))
    (batch / yd.trial_folder_name(done)).mkdir()  # YOLO skips it
    new = _touch(batch / V.format(n=2, ext=".mp4"))
    _touch(batch / V.format(n=3, ext=".h264"))  # not a YOLO input
    retired = root / "october_06_batch_1_rig_3"
    _touch(retired / V.format(n=4, ext=".mp4"))
    from fbpipe.config import DatasetOverride

    cfg = _Cfg({"IAA-Sensitivity-12-1": DatasetOverride(freeze_folders=(FolderRule("*_rig_3"),))})
    assert yd.pending_videos(cfg, root) == [new]


def test_a_collision_group_is_pending_once_as_its_latest_video(tmp_path):
    root = tmp_path / "EB-Control-24-1"
    batch = root / "july_17_batch_1_rig_3"
    stem = "output_july_17_batch_1_training_1_EthylButyrate_20260717_{}.mp4"
    _touch(batch / stem.format("093939"))
    late = _touch(batch / stem.format("115131"))
    assert yd.pending_videos(_Cfg(), root) == [late]
    (batch / "july_17_batch_1_training_1").mkdir()
    assert yd.pending_videos(_Cfg(), root) == []


def test_a_missing_root_has_no_pending_videos(tmp_path):
    assert yd.pending_videos(_Cfg(), tmp_path / "nope") == []


# --- fully processed ------------------------------------------------------------


def _merged(trial: Path, name: str, eyes=(True, True, False, False), *, csv=False):
    trial.mkdir(parents=True, exist_ok=True)
    data = {"frame": [0, 1], "timestamp": [0.0, 0.025]}
    for i, has in enumerate(eyes):
        data[f"eye_{i}_x"] = [1.0, np.nan] if has else [np.nan, np.nan]
    df = pd.DataFrame(data)
    if csv:
        df.to_csv(trial / f"{name}_distances_merged.csv", index=False)
    else:
        df.to_parquet(trial / f"{name}_distances_merged.parquet", index=False)


def _fly(trial: Path, name: str, n: int, *, csv=False):
    path = trial / f"{name}_fly{n}_distances.{'csv' if csv else 'parquet'}"
    pd.DataFrame({"frame": [0]}).to_csv(path, index=False) if csv else pd.DataFrame({"frame": [0]}).to_parquet(path)


NAME = "october_06_batch_1_training_1"


def test_no_trial_folder_is_not_processed(tmp_path):
    ok, why = yd.trial_tables_complete(tmp_path / NAME, NAME)
    assert ok is False and "no trial folder" in why


def test_an_empty_trial_folder_is_not_processed(tmp_path):
    (tmp_path / NAME).mkdir()
    ok, why = yd.trial_tables_complete(tmp_path / NAME, NAME)
    assert ok is False and "merged" in why


def test_missing_per_fly_tables_are_not_processed(tmp_path):
    trial = tmp_path / NAME
    _merged(trial, NAME)
    _fly(trial, NAME, 1)
    ok, why = yd.trial_tables_complete(trial, NAME)
    assert ok is False and "fly2" in why


def test_merged_plus_every_expected_per_fly_table_is_processed(tmp_path):
    trial = tmp_path / NAME
    _merged(trial, NAME)
    _fly(trial, NAME, 1)
    _fly(trial, NAME, 2)
    assert yd.trial_tables_complete(trial, NAME) == (True, "")


def test_a_trial_with_no_eyes_detected_needs_no_per_fly_table(tmp_path):
    """yolo_infer writes no per-fly table for an eyeless slot (EB-Control-24-1
    july_17_batch_1_rig_3 training_1: all four slots empty)."""
    trial = tmp_path / NAME
    _merged(trial, NAME, eyes=(False, False, False, False))
    assert yd.trial_tables_complete(trial, NAME) == (True, "")


def test_legacy_csv_tables_count(tmp_path):
    trial = tmp_path / NAME
    _merged(trial, NAME, csv=True)
    _fly(trial, NAME, 1, csv=True)
    _fly(trial, NAME, 2, csv=True)
    assert yd.trial_tables_complete(trial, NAME) == (True, "")


def test_an_unreadable_merged_table_is_not_processed(tmp_path):
    trial = tmp_path / NAME
    trial.mkdir()
    (trial / f"{NAME}_distances_merged.parquet").write_bytes(b"not parquet")
    ok, why = yd.trial_tables_complete(trial, NAME)
    assert ok is False and "unreadable" in why
