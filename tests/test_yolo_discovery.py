"""fbpipe.utils.yolo_discovery: one definition of "a video YOLO would process".

yolo_infer.main, the auto-freeze plan (a video YOLO would still process is
"something new") and the secure cleanup (never delete a video YOLO has not
fully processed) must agree on it exactly, so the rules live in one torch-free
module:

* a raw video is a ``.mp4``/``.avi`` directly inside a live batch folder;
* its trial folder is ``<batch>/<fields 1-6 of the stem>`` (odor and stamp
  dropped), and YOLO skips a video whose trial folder exists;
* several videos naming one trial folder are a restarted session: the LATEST
  (last in sorted order -- the stamp is in the name) is the real one;
* "fully processed" = the trial folder holds the merged table and a per-fly
  table for every fly slot the merged table has eye data for.
"""

from __future__ import annotations

import datetime as dt
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.steps import yolo_infer
from fbpipe.utils import yolo_discovery as yd
from fbpipe.utils.frozen_folders import FolderRule


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
