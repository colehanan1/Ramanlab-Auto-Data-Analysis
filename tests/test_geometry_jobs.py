"""Batch jobs of the geometry pipeline: production batches in, a separate output tree out.

In integrated mode (``GeometrySettings.out_base``) the steps enumerate the PRODUCTION batches
(sidecars and session_metadata.txt, no mp4s), take each trial's video from the secured mirror first
and from the local batch second, and write every output to
``<out_base>/<dataset folder>/<batch>/geometry/``. Pinned here: which batches are walked (cut-off
date, dataset subset, freeze judged on the production dir), where each trial's video comes from (never a
superseded attempt of a restarted trial: production's YOLO skips those, so they are dropped per source before the
first source wins), that the read-only trees are never written, that one bad video is logged and counted instead of stopping
the batch, that normalize/wide/figures need nothing but ``geometry/``, that a track with nothing
pending spawns no workers and loads no model, and that the integrated track stops at the next video
while a PER run holds the shared run lock (resumable, nothing half-written). Legacy mode (no
``out_base``) keeps walking exactly what ``iter_batches`` yields.
"""
import dataclasses
import json
import os
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import pandas as pd
import pytest

from fbpipe import geometry_pipeline as GP
from fbpipe.config import DatasetOverride, Settings
from fbpipe.geometry import columns as C
from fbpipe.geometry import detect as D
from fbpipe.geometry import steps as S
from fbpipe.geometry import stats as ST
from fbpipe.geometry.config import GeometrySettings
from fbpipe.geometry.trial_meta import trial_name_from_video
from fbpipe.utils.frozen_folders import FolderRule
from fbpipe.utils.run_locks import per_run_lock
from test_geometry_track import H, NAMES, W, fake_predict, video  # noqa: F401  (``video``: the synthetic 40-frame clip)

DATASET = "IAA-X"
BATCH = "october_07_batch_2"
RIG_BATCH = BATCH + "_rig_2"
STAMP = "20261007"
CUTOFF = date(2026, 10, 6)
# (type, index, odor) of BATCH's three trials: 1 and 2 are on the secured mirror, 3 only in the local batch.
TRIALS = (("pretest", 1, "Citral-1"), ("pretest", 2, "Hexanol-0.1"), ("pretest", 3, "Linalool-1"))
SESSION_METADATA = ("Session Metadata\nFly Type: GR5a-Retinol\nBorn (approx): 2026-10-05\n"
                    "Host: BehaviorLocust | OS: Linux\n")


def trial_stem(n: int, batch: str = BATCH) -> str:
    return f"{batch}_pretest_{n}"


def add_trial(folder: Path, template: Path, batch: str, kind: str, n: int, odor: str, *, day: str,
              video: Optional[str] = "avi") -> Path:
    """One trial's sidecars in ``folder`` (the csv is the template's: ActiveOFM on for frames 10-19) and, per
    ``video``, its video: "avi" a real 40-frame clip, "corrupt" a 0-byte mp4, "no_frames" a header-only avi that
    opens but has no decodable frame, None no video at all. The video path is returned. (Names are built by
    hand: ``with_suffix`` would eat the dot of ``Hexanol-0.1``.)"""
    folder.mkdir(parents=True, exist_ok=True)
    base = f"output_{batch}_{kind}_{n}_{odor}_{day}_{10 + n:02d}0000"
    shutil.copyfile(template.with_suffix(".csv"), folder / f"{base}.csv")
    (folder / f"{base}.txt").write_text(
        f"Cycle Name: {kind}_{n}_{odor}\nTotal Frames Captured: 40\nAchieved Frame Rate: 40.0 FPS\n", encoding="utf-8")
    if video == "avi":
        shutil.copyfile(template, folder / f"{base}.avi")
    elif video == "corrupt":
        (folder / f"{base}.mp4").write_bytes(b"")
    elif video == "no_frames":
        cv2.VideoWriter(str(folder / f"{base}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 40.0, (W, H)).release()
    return folder / f"{base}.{'mp4' if video == 'corrupt' else 'avi'}"


def production_batch(root: Path, name: str, template: Path, *, day: Optional[str]) -> Path:
    """A production batch dir: session_metadata.txt and, when ``day`` is given, one sidecar-only trial stamped then."""
    batch = root / name
    batch.mkdir(parents=True)
    (batch / "session_metadata.txt").write_text(SESSION_METADATA, encoding="utf-8")
    if day is not None:
        add_trial(batch, template, name, "pretest", 1, "Citral-1", day=day, video=None)
    return batch


def snapshot(*roots: Path) -> dict:
    """(is_dir, size, mtime_ns) of every path under the roots, the roots themselves included."""
    out = {}
    for root in roots:
        for p in [root, *sorted(root.rglob("*"))]:
            st = p.stat()
            out[str(p)] = (p.is_dir(), st.st_size, st.st_mtime_ns)
    return out


class Tree:
    """The tmp trees of integrated mode. ``flys_New`` (production) holds sidecars + session_metadata.txt only, except
    trial 3 of BATCH, whose video exists only there. ``secured`` holds trials 1 and 2 of BATCH (video + sidecars) and
    a stale video of the out-of-window september batch. ``flys_Abdomen`` (the out base) starts empty."""

    def __init__(self, root: Path, template: Path, *, corrupt: Optional[int] = None,
                 no_frames: Optional[int] = None, folder: str = DATASET):
        self.root, self.template = root, template
        self.folder = folder                                  # the dataset's directory name on disk
        self.prod, self.secured, self.out_base = root / "flys_New", root / "secured", root / "flys_Abdomen"
        self.ds_prod = self.prod / folder
        self.batch_dir = self.ds_prod / BATCH
        self.secured_batch = self.secured / folder / BATCH
        self.ds_prod.mkdir(parents=True)
        production_batch(self.ds_prod, "september_30_batch_1", template, day="20260930")
        production_batch(self.ds_prod, RIG_BATCH, template, day=STAMP)
        production_batch(self.ds_prod, BATCH, template, day=None)
        (self.ds_prod / "misc_run").mkdir()                     # no stamps, no metadata, no date in its name
        for kind, n, odor in TRIALS:
            on_mirror = n in (1, 2)
            add_trial(self.batch_dir, template, BATCH, kind, n, odor, day=STAMP, video=None if on_mirror else "avi")
            if on_mirror:
                add_trial(self.secured_batch, template, BATCH, kind, n, odor, day=STAMP,
                          video="corrupt" if n == corrupt else "no_frames" if n == no_frames else "avi")
        add_trial(self.secured / folder / "september_30_batch_1", template, "september_30_batch_1", "pretest", 1,
                  "Citral-1", day="20260930")

    def settings(self, **kw) -> Settings:
        return Settings(model_path=str(self.root / "abdomen_best.pt"), main_directories=[str(self.ds_prod)],
                        datasets=(DATASET,), conf_thres=0.25, max_flies=5, inference_batch_size=7, allow_cpu=True,
                        fps_default=40.0, **kw)

    def frozen(self, *folders: str) -> Settings:
        rules = tuple(FolderRule(f) for f in folders)
        return self.settings(dataset_overrides={DATASET: DatasetOverride(freeze_folders=rules)})

    def gcfg(self, **kw) -> GeometrySettings:
        base = dict(warmup_frames=5, annotated_video=False, out_base=str(self.out_base),
                    video_bases=(str(self.secured),), batch_date_from=CUTOFF,
                    wide_out_dir=str(self.root / "wide"), figures_out_dir=str(self.root / "figs"))
        return GeometrySettings(**{**base, **kw})

    def job(self, name: str = BATCH) -> "S.BatchJob":
        return next(j for j in S.iter_batch_jobs(self.settings(), self.gcfg()) if j.name == name)

    def out_batch(self, name: str = BATCH) -> Path:
        return self.out_base / self.folder / name

    def trial_dir(self, n: int) -> Path:
        return self.out_batch() / "geometry" / trial_stem(n)

    def meta(self, n: int) -> Path:
        return C.meta_path(self.trial_dir(n), trial_stem(n))

    def read_only(self) -> dict:
        return snapshot(self.prod, self.secured)


@pytest.fixture
def tree(tmp_path, video):
    return Tree(tmp_path, video)


@pytest.fixture
def fake_model(monkeypatch):
    """The abdomen model replaced by test_geometry_track's fake detector: no weights, no GPU. Returns the list of
    model paths that were loaded, so a test can assert none was."""
    loaded = []
    monkeypatch.delenv("NUM_WORKERS", raising=False)
    monkeypatch.delenv("WORKER_INDEX", raising=False)
    monkeypatch.setattr(D, "load_model", lambda path: loaded.append(path) or SimpleNamespace(names=NAMES))
    monkeypatch.setattr(D, "make_predict_fn", lambda model, **kw: fake_predict)
    monkeypatch.setattr(D, "model_sha1", lambda path: "0" * 40)
    return loaded


def _cli(monkeypatch, settings, gcfg):
    """GP.main over ``settings``/``gcfg`` with no config file; returns the worker counts spawned and the steps run."""
    spawned, ran = [], []
    monkeypatch.setattr("fbpipe.geometry.config.load_geometry_config", lambda path: (settings, gcfg))
    monkeypatch.setattr(GP, "fan_out_track", lambda cfg, n, *, settings, options: spawned.append(n) or 0)
    monkeypatch.setattr(S, "run_normalize", lambda settings, gcfg, *, force=False: ran.append("normalize"))
    return spawned, ran


# ---- the job record -------------------------------------------------------------------------------

def test_a_batch_job_is_a_frozen_record_of_the_six_agreed_fields():
    fields = [f.name for f in dataclasses.fields(S.BatchJob)]
    assert fields == ["dataset", "root_name", "name", "prod_dir", "out_dir", "frozen"]
    job = S.BatchJob(DATASET, DATASET, BATCH, Path("/p"), Path("/o"), False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        job.name = "other"


# ---- which batches are walked ---------------------------------------------------------------------

def test_integrated_jobs_keep_only_batches_recorded_on_or_after_the_cutoff(tree):
    jobs = list(S.iter_batch_jobs(tree.settings(), tree.gcfg()))
    assert [j.name for j in jobs] == [BATCH, RIG_BATCH]             # september_30_batch_1 (stamped 20260930) is out
    assert all(j.dataset == DATASET and j.root_name == DATASET and not j.frozen for j in jobs)
    assert [j.prod_dir for j in jobs] == [tree.ds_prod / BATCH, tree.ds_prod / RIG_BATCH]


def test_a_batch_that_cannot_be_dated_at_all_is_excluded(tree):
    assert "misc_run" not in [j.name for j in S.iter_batch_jobs(tree.settings(), tree.gcfg())]
    assert "misc_run" not in [j.name for j in S.iter_batch_jobs(tree.settings(), tree.gcfg(), live_only=False)]


def test_a_batch_without_stamps_is_dated_by_its_folder_name_in_the_cutoffs_year(tmp_path, video):
    root = tmp_path / "flys_New" / DATASET
    root.mkdir(parents=True)
    for name in ("october_05_batch_1", "october_06_batch_1", "october_08_batch_1", "november_02_batch_1",
                 "september_30_batch_9", "february_30_batch_1"):
        (root / name).mkdir()
    production_batch(root, "october_09_batch_3", video, day="20260930")        # stamps say Sep 30: they beat the name
    settings = Settings(model_path="m", main_directories=[str(root)], datasets=(DATASET,))
    gcfg = GeometrySettings(out_base=str(tmp_path / "out"), batch_date_from=CUTOFF)
    assert [j.name for j in S.iter_batch_jobs(settings, gcfg)] == ["november_02_batch_1", "october_06_batch_1",
                                                                   "october_08_batch_1"]


def test_without_a_cutoff_every_batch_is_kept_even_the_undated_ones(tree):
    names = [j.name for j in S.iter_batch_jobs(tree.settings(), tree.gcfg(batch_date_from=None))]
    assert names == ["misc_run", BATCH, RIG_BATCH, "september_30_batch_1"]


def test_the_dataset_subset_keeps_only_the_listed_datasets(tmp_path, video):
    roots = {}
    for ds in ("IAA-X", "Hex-Y"):
        roots[ds] = tmp_path / "flys_New" / ds
        production_batch(roots[ds], BATCH, video, day=STAMP)
    settings = Settings(model_path="m", main_directories=[str(r) for r in roots.values()], datasets=tuple(roots))
    out = str(tmp_path / "out")
    everything = S.iter_batch_jobs(settings, GeometrySettings(out_base=out, datasets=()))
    assert sorted(j.dataset for j in everything) == ["Hex-Y", "IAA-X"]                      # empty = all
    only = list(S.iter_batch_jobs(settings, GeometrySettings(out_base=out, datasets=("Hex-Y",))))
    assert [(j.dataset, j.name) for j in only] == [("Hex-Y", BATCH)]


def test_out_dirs_and_the_dataset_subset_follow_the_root_folder_name_not_the_dataset_label(tmp_path, video):
    root = tmp_path / "flys_New" / "iaa-x"           # on disk lower case; the config spells the dataset IAA-X
    production_batch(root, BATCH, video, day=STAMP)
    settings = Settings(model_path="m", main_directories=[str(root)], datasets=(DATASET,))
    out_base = tmp_path / "flys_Abdomen"
    jobs = list(S.iter_batch_jobs(settings, GeometrySettings(out_base=str(out_base))))
    assert len(jobs) == 1
    job = jobs[0]
    assert (job.dataset, job.root_name, job.name) == (DATASET, "iaa-x", BATCH)
    assert job.out_dir == out_base / "iaa-x" / BATCH and job.prod_dir == root / BATCH
    for listed in ("iaa-x", DATASET):                # the subset matches the folder name or the label
        assert len(list(S.iter_batch_jobs(settings, GeometrySettings(out_base=str(out_base), datasets=(listed,))))) == 1
    assert not list(S.iter_batch_jobs(settings, GeometrySettings(out_base=str(out_base), datasets=("other",))))


def test_out_dir_is_out_base_dataset_folder_batch(tree):
    for job in S.iter_batch_jobs(tree.settings(), tree.gcfg()):
        assert job.out_dir == tree.out_base / DATASET / job.name
        assert not job.out_dir.exists(), "enumerating jobs creates nothing"


def test_frozen_is_judged_on_the_production_dir_and_no_warning_names_the_out_tree(tree, capsys):
    # A recording-date rule needs the folder's date: the production dir has stamps, an out dir never does. Out dirs
    # exist from an earlier run, so a freeze check that wandered into them would warn "Cannot date batch".
    rule = {DATASET: DatasetOverride(freeze_folders=(FolderRule("*_rig_2", before=date(2026, 10, 8)),))}
    settings = tree.settings(dataset_overrides=rule)
    for name in (BATCH, RIG_BATCH):
        (tree.out_batch(name) / "geometry").mkdir(parents=True)
    assert [j.name for j in S.iter_batch_jobs(settings, tree.gcfg())] == [BATCH]
    allj = list(S.iter_batch_jobs(settings, tree.gcfg(), live_only=False))
    assert {j.name: j.frozen for j in allj} == {BATCH: False, RIG_BATCH: True}
    seen = capsys.readouterr()
    assert str(tree.out_base) not in seen.out + seen.err


def test_legacy_jobs_are_exactly_what_iter_batches_yields(tmp_path):
    root = tmp_path / "flys_Abdomen" / DATASET
    for name in ("september_02_batch_1_rig_3_copy", "august_28_batch_1_rig_3_copy"):
        (root / name).mkdir(parents=True)
    settings = Settings(model_path="x", main_directories=[str(root)], datasets=(DATASET,),
                        dataset_overrides={DATASET: DatasetOverride(freeze_folders=(FolderRule("august_28*"),))})
    for live_only in (True, False):
        jobs = list(S.iter_batch_jobs(settings, GeometrySettings(), live_only=live_only))
        assert [(j.prod_dir, j.dataset, j.frozen) for j in jobs] == list(S.iter_batches(settings, live_only=live_only))
        assert all(j.out_dir == j.prod_dir and j.root_name == root.name and j.name == j.prod_dir.name for j in jobs)
    assert [j.frozen for j in S.iter_batch_jobs(settings, GeometrySettings(), live_only=False)] == [True, False]


# ---- where each trial's video comes from ----------------------------------------------------------

def test_videos_come_per_trial_from_the_secured_mirror_then_the_local_batch(tree):
    gcfg = tree.gcfg()
    videos = S.videos_for_job(tree.job(), gcfg)
    assert [trial_name_from_video(v) for v in videos] == [trial_stem(n) for n in (1, 2, 3)]
    assert [v.parent for v in videos] == [tree.secured_batch, tree.secured_batch, tree.batch_dir]
    assert [v.name for v in videos] == sorted(v.name for v in videos)
    # a stale local copy of trial 1 (another recording stamp) does not displace the secured one: the trial is the key
    add_trial(tree.batch_dir, tree.template, BATCH, "pretest", 1, "Citral-1", day="20261007", video="avi")
    (tree.batch_dir / f"output_{BATCH}_pretest_1_Citral-1_20261007_235959.avi").write_bytes(b"stale")
    assert S.videos_for_job(tree.job(), gcfg) == videos


def test_each_video_base_is_tried_in_order_before_the_local_batch(tree, tmp_path):
    second = tmp_path / "secured2" / DATASET / BATCH
    add_trial(second, tree.template, BATCH, "pretest", 1, "Citral-1", day=STAMP)            # loses to the first base
    add_trial(second, tree.template, BATCH, "pretest", 3, "Linalool-1", day=STAMP)          # beats the local copy
    gcfg = tree.gcfg(video_bases=(str(tree.secured), str(tmp_path / "secured2")))
    videos = S.videos_for_job(tree.job(), gcfg)
    assert [v.parent for v in videos] == [tree.secured_batch, tree.secured_batch, second]


def test_a_batch_missing_on_the_mirror_falls_back_to_its_local_videos(tree, tmp_path):
    gcfg = tree.gcfg(video_bases=(str(tmp_path / "no_such_mirror"),))
    videos = S.videos_for_job(tree.job(), gcfg)
    assert [v.parent for v in videos] == [tree.batch_dir] and trial_name_from_video(videos[0]) == trial_stem(3)
    assert S.videos_for_job(tree.job(RIG_BATCH), tree.gcfg()) == []


def newest_session(folder: Path, n: int, odor: str, day: str) -> None:
    """Make the sidecar of attempt *day* of trial *n* the newest in *folder*: the session production's metadata is read
    from (``find_sidecar`` takes the most recently modified), so the one production's YOLO processes."""
    for txt in folder.glob(f"output_{BATCH}_pretest_{n}_{odor}_*.txt"):
        stamp = 2_000_000_000 if day in txt.name else 1_000_000_000
        os.utime(txt, (stamp, stamp))


def trial_videos(videos, n: int) -> list:
    return [v for v in videos if trial_name_from_video(v) == trial_stem(n)]


def test_a_restarted_trial_is_tracked_from_the_session_production_processed_not_the_first_one_by_name(tree):
    """Every attempt at a trial names one trial folder, and production's YOLO processes the session its metadata is read
    from. The abdomen tracks must pair with those tables, so that is the video that is tracked -- not the first by name,
    which is the aborted attempt when the restart came later."""
    aborted = next(tree.secured_batch.glob(f"output_{BATCH}_pretest_1_*.avi"))
    restart = add_trial(tree.secured_batch, tree.template, BATCH, "pretest", 1, "Citral-1", day="20261008")
    assert aborted.name < restart.name, "precondition: first-by-name would pick the aborted attempt"
    newest_session(tree.secured_batch, 1, "Citral-1", "20261008")
    videos = S.videos_for_job(tree.job(), tree.gcfg())
    assert trial_videos(videos, 1) == [restart]
    assert [trial_name_from_video(v) for v in videos] == [trial_stem(n) for n in (1, 2, 3)]


def test_the_older_attempt_wins_when_it_is_the_session_production_processed(tree):
    first = next(tree.secured_batch.glob(f"output_{BATCH}_pretest_1_*.avi"))
    add_trial(tree.secured_batch, tree.template, BATCH, "pretest", 1, "Citral-1", day="20261008")
    newest_session(tree.secured_batch, 1, "Citral-1", "20261007")           # the metadata still says the first one
    assert trial_videos(S.videos_for_job(tree.job(), tree.gcfg()), 1) == [first]


def test_a_source_that_holds_only_a_superseded_attempt_hands_the_trial_to_the_next_source(tree):
    """The mirror has the aborted attempt's video; the restart's video (the session production processed) is only in the
    local batch. The mirror's video is superseded -- production's YOLO would skip it -- so it must not win the trial."""
    add_trial(tree.secured_batch, tree.template, BATCH, "pretest", 1, "Citral-1", day="20261008", video=None)
    restart = add_trial(tree.batch_dir, tree.template, BATCH, "pretest", 1, "Citral-1", day="20261008")
    newest_session(tree.secured_batch, 1, "Citral-1", "20261008")
    newest_session(tree.batch_dir, 1, "Citral-1", "20261008")
    videos = S.videos_for_job(tree.job(), tree.gcfg())
    assert trial_videos(videos, 1) == [restart]
    assert [v.parent for v in videos] == [tree.batch_dir, tree.secured_batch, tree.batch_dir]


def test_a_superseded_video_is_never_tracked_even_when_no_source_has_the_real_session(tree, fake_model):
    add_trial(tree.secured_batch, tree.template, BATCH, "pretest", 1, "Citral-1", day="20261008", video=None)
    newest_session(tree.secured_batch, 1, "Citral-1", "20261008")
    settings, gcfg = tree.settings(), tree.gcfg()
    videos = S.videos_for_job(tree.job(), gcfg)
    assert trial_videos(videos, 1) == [] and [trial_name_from_video(v) for v in videos] == [trial_stem(2), trial_stem(3)]
    assert S.pending_trials(settings, gcfg) == 2
    assert S.run_track(settings, gcfg)["tracked"] == 2 and not tree.meta(1).exists()


def test_video_paths_are_built_from_the_root_folder_name_not_the_dataset_label(tmp_path, video):
    root = tmp_path / "flys_New" / "iaa-x"
    production_batch(root, BATCH, video, day=STAMP)
    mirror = tmp_path / "secured"
    right = add_trial(mirror / "iaa-x" / BATCH, video, BATCH, "pretest", 1, "Citral-1", day=STAMP)
    add_trial(mirror / DATASET / BATCH, video, BATCH, "pretest", 2, "Hexanol-0.1", day=STAMP)    # decoy under the label
    settings = Settings(model_path="m", main_directories=[str(root)], datasets=(DATASET,))
    gcfg = GeometrySettings(out_base=str(tmp_path / "out"), video_bases=(str(mirror),))
    job = next(iter(S.iter_batch_jobs(settings, gcfg)))
    assert S.videos_for_job(job, gcfg) == [right]


def test_a_root_folder_spelled_unlike_its_dataset_label_is_found_everywhere(tmp_path, video, fake_model):
    tree = Tree(tmp_path, video, folder="iaa-x")      # lower case on disk and on the mirror; the config says IAA-X
    settings, gcfg = tree.settings(), tree.gcfg()
    assert S.run_track(settings, gcfg)["tracked"] == 3, "the videos are found under <mirror>/iaa-x/<batch>"
    S.run_normalize(settings, gcfg)
    wide = S.run_wide(settings, gcfg)
    S.run_figures(settings, gcfg)
    assert [p.name for p in tree.out_base.iterdir()] == ["iaa-x"], "paths follow the folder"
    meta = json.loads(tree.meta(1).read_text(encoding="utf-8"))
    assert meta["dataset"] == DATASET and meta["trial"]["dataset"] == DATASET, "metadata keeps the label"
    assert set(pd.read_csv(wide["files"]["per"])["dataset"]) == {DATASET}
    assert len(list((Path(gcfg.figures_out_dir) / DATASET / BATCH).glob("*.png"))) == 7


def test_legacy_videos_are_the_batch_dirs_own_and_video_bases_do_not_apply(tmp_path, video):
    batch = production_batch(tmp_path / "flys_Abdomen" / DATASET, BATCH, video, day=None)
    own = add_trial(batch, video, BATCH, "pretest", 1, "Citral-1", day=STAMP)
    mirror = tmp_path / "secured"
    add_trial(mirror / DATASET / BATCH, video, BATCH, "pretest", 2, "Hexanol-0.1", day=STAMP)
    settings = Settings(model_path="m", main_directories=[str(batch.parent)], datasets=(DATASET,))
    gcfg = GeometrySettings(video_bases=(str(mirror),))                    # no out_base: legacy
    job = next(iter(S.iter_batch_jobs(settings, gcfg)))
    assert S.videos_for_job(job, gcfg) == S.iter_videos(batch) == [own]


# ---- tracking: the out tree, the read-only trees, one bad video -----------------------------------

def test_track_writes_only_into_the_out_tree_and_never_touches_production_or_the_mirror(tree, fake_model):
    before = tree.read_only()
    result = S.run_track(tree.settings(), tree.gcfg())
    assert (result["tracked"], result["skipped"], result["failed"]) == (3, 0, 0)
    assert tree.read_only() == before, "flys_New and the secured mirror must come out byte- and mtime-identical"
    assert sorted(p.name for p in tree.out_batch().iterdir()) == ["geometry"]
    trial_dirs = sorted(p.name for p in (tree.out_batch() / "geometry").iterdir() if p.is_dir())
    assert trial_dirs == [trial_stem(n) for n in (1, 2, 3)]
    for n in (1, 2, 3):
        assert tree.meta(n).is_file()
        assert sorted(s for s, _ in C.iter_geometry_tables(tree.trial_dir(n))) == [1, 2, 3]
    assert not tree.out_batch(RIG_BATCH).exists(), "a batch without any video leaves nothing behind"
    assert not tree.out_batch("september_30_batch_1").exists(), "nor does a batch before the cut-off"


def test_track_reads_sidecars_beside_each_video_and_batch_metadata_from_the_production_dir(tree, fake_model):
    S.run_track(tree.settings(), tree.gcfg())
    metas = {n: json.loads(tree.meta(n).read_text(encoding="utf-8")) for n in (1, 2, 3)}
    for n, sidecar_dir in ((1, tree.secured_batch), (2, tree.secured_batch), (3, tree.batch_dir)):
        trial = metas[n]["trial"]
        assert Path(trial["sidecar_csv"]).parent == sidecar_dir and Path(trial["sidecar_txt"]).parent == sidecar_dir
        assert Path(trial["video"]).parent == sidecar_dir
        assert trial["batch_dir"] == str(tree.batch_dir)
        assert trial["fly_type"] == "GR5a-Old", "the Fly Type comes from the production session_metadata.txt"
        assert trial["dataset"] == DATASET and metas[n]["dataset"] == DATASET
        assert trial["has_odor"] and trial["odor_on_frame"] == 11
        assert Path(trial["out_dir"]) == tree.trial_dir(n)
    assert metas[2]["trial"]["odor"] == "Hexanol-0.1"


def test_source_json_records_the_production_dir_and_the_first_video_dir(tree, fake_model):
    S.run_track(tree.settings(), tree.gcfg())
    source = tree.out_batch() / "geometry" / "source.json"
    assert json.loads(source.read_text(encoding="utf-8")) == {"prod_dir": str(tree.batch_dir),
                                                              "video_dir": str(tree.secured_batch)}
    assert not (tree.out_batch(RIG_BATCH) / "geometry").exists()
    # a rerun that has nothing to track leaves the record alone
    stamp = source.stat().st_mtime_ns
    S.run_track(tree.settings(), tree.gcfg())
    assert source.stat().st_mtime_ns == stamp


def test_a_corrupt_video_is_logged_and_counted_and_the_other_trials_still_tracked(tmp_path, video, fake_model, capsys):
    tree = Tree(tmp_path, video, corrupt=2)
    result = S.run_track(tree.settings(), tree.gcfg())
    assert (result["tracked"], result["skipped"], result["failed"]) == (2, 0, 1)
    log = capsys.readouterr().out
    line = next(ln for ln in log.splitlines() if ln.startswith("[GEOMETRY] failed "))
    assert ".mp4" in line and "_pretest_2_" in line and "cannot open video" in line
    assert tree.meta(1).is_file() and tree.meta(3).is_file(), "the trials after the bad one are tracked"
    assert not tree.meta(2).exists(), "a failed trial writes no meta, so it is retried"
    assert S.pending_trials(tree.settings(), tree.gcfg()) == 1
    again = S.run_track(tree.settings(), tree.gcfg())
    assert (again["tracked"], again["skipped"], again["failed"]) == (0, 2, 1)


def test_a_video_that_opens_but_yields_no_frame_is_failed_and_retried_not_tracked(tmp_path, video, fake_model, capsys):
    tree = Tree(tmp_path, video, no_frames=2)
    clip = next(tree.secured_batch.glob("*_pretest_2_*.avi"))
    cap = cv2.VideoCapture(str(clip))
    assert cap.isOpened() and not cap.read()[0], "precondition: the clip opens, yet no frame can be decoded"
    cap.release()
    settings, gcfg = tree.settings(), tree.gcfg()
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["skipped"], result["failed"]) == (2, 0, 1)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[GEOMETRY] failed "))
    assert "_pretest_2_" in line and "no decodable frame" in line
    assert tree.meta(1).is_file() and tree.meta(3).is_file()
    assert not tree.meta(2).exists(), "no meta, so the trial is not 'done'"
    assert list(C.iter_geometry_tables(tree.trial_dir(2))) == []
    assert S.pending_trials(settings, gcfg) == 1
    shutil.copyfile(video, clip)                              # the file is repaired or remuxed
    again = S.run_track(settings, gcfg)
    assert (again["tracked"], again["skipped"], again["failed"]) == (1, 2, 0)
    assert tree.meta(2).is_file() and S.pending_trials(settings, gcfg) == 0


def test_a_readable_video_with_no_flies_is_still_recorded_as_tracked(tree, fake_model, monkeypatch):
    # Every frame decodes; the detector just finds nothing (an empty arena). That is a result, not a failure.
    monkeypatch.setattr(D, "make_predict_fn",
                        lambda model, **kw: lambda images, conf: [SimpleNamespace(obb=None) for _ in images])
    settings, gcfg = tree.settings(), tree.gcfg()
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["skipped"], result["failed"]) == (3, 0, 0)
    for n in (1, 2, 3):
        meta = json.loads(tree.meta(n).read_text(encoding="utf-8"))
        assert meta["slots"] == [] and meta["n_frames_processed"] == 0 and meta["warmup"]["frames_used"] == 5
        assert list(C.iter_geometry_tables(tree.trial_dir(n))) == []
    assert S.pending_trials(settings, gcfg) == 0


def test_normalize_wide_and_figures_carry_on_past_a_failed_video(tmp_path, video, fake_model):
    tree = Tree(tmp_path, video, corrupt=2)
    settings, gcfg = tree.settings(), tree.gcfg()
    assert S.run_track(settings, gcfg)["failed"] == 1
    assert tree.trial_dir(2).is_dir() and not tree.meta(2).exists(), "the failed trial leaves a bare folder, no meta"
    assert S.run_normalize(settings, gcfg) == {str(tree.out_batch()): [1, 2, 3]}
    assert set(S.run_wide(settings, gcfg)["rows"].values()) == {6}                  # 2 tracked trials x 3 flies
    assert S.run_figures(settings, gcfg)["n_figures"] == 7


def test_a_failed_video_does_not_stop_the_steps_that_follow_track(tmp_path, video, fake_model, monkeypatch):
    tree = Tree(tmp_path, video, corrupt=2)
    settings = dataclasses.replace(tree.settings(), yolo_num_workers=1)
    spawned, ran = _cli(monkeypatch, settings, tree.gcfg())
    assert GP.main(["--config", "c.yaml", "track", "normalize"]) == 0
    assert tree.meta(1).is_file() and tree.meta(3).is_file() and not tree.meta(2).exists()
    assert spawned == [] and ran == ["normalize"]


def test_an_error_while_tracking_a_video_is_isolated_the_same_way(tree, fake_model, monkeypatch, capsys):
    from fbpipe.geometry import track as T
    real = T.track_video

    def flaky(video_path, *a, **kw):
        if "_pretest_3_" in Path(video_path).name:
            raise ValueError("boom")
        return real(video_path, *a, **kw)

    monkeypatch.setattr(T, "track_video", flaky)
    result = S.run_track(tree.settings(), tree.gcfg())
    assert (result["tracked"], result["failed"]) == (2, 1)
    assert "[GEOMETRY] failed " in capsys.readouterr().out
    assert tree.meta(1).is_file() and tree.meta(2).is_file() and not tree.meta(3).exists()


def test_workers_split_the_per_trial_video_list_without_overlap(tree, fake_model, monkeypatch):
    monkeypatch.setattr(S, "_cap_cpu_threads", lambda n: None)      # keep cv2/torch thread settings out of the suite
    tracked = []
    for worker in (0, 1):
        monkeypatch.setenv("NUM_WORKERS", "2")
        monkeypatch.setenv("WORKER_INDEX", str(worker))
        tracked.append(S.run_track(tree.settings(), tree.gcfg())["tracked"])
    assert tracked == [2, 1], "videos 1 and 3 go to worker 0, video 2 to worker 1"
    assert all(tree.meta(n).is_file() for n in (1, 2, 3))


def test_legacy_track_still_writes_beside_the_batch_and_records_no_source(tmp_path, video, fake_model):
    batch = production_batch(tmp_path / "flys_Abdomen" / DATASET, BATCH, video, day=None)
    for kind, n, odor in TRIALS[:2]:
        add_trial(batch, video, BATCH, kind, n, odor, day=STAMP)
    settings = Settings(model_path=str(tmp_path / "m.pt"), main_directories=[str(batch.parent)], datasets=(DATASET,),
                        conf_thres=0.25, max_flies=5, inference_batch_size=7, allow_cpu=True, fps_default=40.0)
    gcfg = GeometrySettings(warmup_frames=5, annotated_video=False)
    assert S.pending_trials(settings, gcfg) == 2
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["skipped"], result["failed"]) == (2, 0, 0)
    for n in (1, 2):
        tdir = batch / "geometry" / trial_stem(n)
        assert C.meta_path(tdir, trial_stem(n)).is_file() and len(list(C.iter_geometry_tables(tdir))) == 3
    assert not (batch / "geometry" / "source.json").exists()
    assert S.pending_trials(settings, gcfg) == 0
    again = S.run_track(settings, gcfg)
    assert (again["tracked"], again["skipped"], again["failed"]) == (0, 2, 0)
    assert S.run_normalize(settings, gcfg) == {str(batch.resolve()): [1, 2, 3]}


# ---- yielding to a running PER pass ---------------------------------------------------------------

def test_track_yields_to_a_per_run_that_starts_midway_and_writes_no_partial_meta(tree, tmp_path, fake_model,
                                                                                monkeypatch, capsys):
    asked = []

    def per_run_active(cache_dir):                      # PER starts while the first video is being tracked
        asked.append(cache_dir)
        return len(asked) > 1

    monkeypatch.setattr(S, "per_run_active", per_run_active)
    add_trial(tree.ds_prod / RIG_BATCH, tree.template, RIG_BATCH, "pretest", 2, "Hexanol-0.1", day=STAMP)  # a 2nd batch
    settings, gcfg = tree.settings(cache_dir=str(tmp_path / "cache")), tree.gcfg()
    result = S.run_track(settings, gcfg)
    assert result["yielded"] is True
    assert (result["tracked"], result["skipped"], result["failed"]) == (1, 0, 0)
    assert asked == [settings.cache_dir] * 2, "asked before each video, in the cache dir the PER lock lives in"
    assert tree.meta(1).is_file(), "the video tracked before PER started is kept"
    assert not tree.trial_dir(2).exists() and not tree.trial_dir(3).exists(), "nothing written for the trials left"
    assert not tree.out_batch(RIG_BATCH).exists(), "the loop stops for good: later batches are not even visited"
    out = capsys.readouterr().out
    assert out.count("[GEOMETRY] PER run active — yielding; remaining trials resume next run") == 1
    assert S.pending_trials(settings, gcfg) == 3, "the rest resume next run"

    monkeypatch.setattr(S, "per_run_active", lambda cache_dir: False)
    resumed = S.run_track(settings, gcfg)
    assert resumed["yielded"] is False and (resumed["tracked"], resumed["skipped"]) == (3, 1)
    assert all(tree.meta(n).is_file() for n in (1, 2, 3))


def test_a_pass_that_holds_the_real_per_lock_makes_track_yield_before_the_first_video(tree, tmp_path, fake_model):
    settings, gcfg = tree.settings(cache_dir=str(tmp_path / "cache")), tree.gcfg()
    with per_run_lock(settings.cache_dir):
        result = S.run_track(settings, gcfg)
    assert result["yielded"] is True and result["tracked"] == 0
    assert not tree.out_base.exists(), "no meta, no trial folder, not even source.json"
    resumed = S.run_track(settings, gcfg)
    assert resumed["yielded"] is False and resumed["tracked"] == 3


def test_a_run_with_nothing_left_to_track_does_not_report_yielding(tree, fake_model, monkeypatch, capsys):
    settings, gcfg = tree.settings(), tree.gcfg()
    S.run_track(settings, gcfg)
    monkeypatch.setattr(S, "per_run_active", lambda cache_dir: True)
    capsys.readouterr()
    result = S.run_track(settings, gcfg)
    assert result["yielded"] is False and (result["tracked"], result["skipped"]) == (0, 3)
    assert "yielding" not in capsys.readouterr().out


def test_legacy_track_never_looks_at_the_per_lock(tmp_path, video, fake_model, monkeypatch):
    batch = production_batch(tmp_path / "flys_Abdomen" / DATASET, BATCH, video, day=None)
    add_trial(batch, video, BATCH, "pretest", 1, "Citral-1", day=STAMP)
    settings = Settings(model_path=str(tmp_path / "m.pt"), main_directories=[str(batch.parent)], datasets=(DATASET,),
                        conf_thres=0.25, max_flies=5, inference_batch_size=7, allow_cpu=True, fps_default=40.0)
    monkeypatch.setattr(S, "per_run_active", lambda cache_dir: pytest.fail("legacy mode must not ask"))
    result = S.run_track(settings, GeometrySettings(warmup_frames=5, annotated_video=False))
    assert result["tracked"] == 1 and result["yielded"] is False


# ---- normalize / wide / figures / stats read only geometry/ ---------------------------------------

def test_normalize_wide_and_figures_need_nothing_but_the_geometry_dir_of_each_out_batch(tree, fake_model):
    settings, gcfg = tree.settings(), tree.gcfg()
    S.run_track(settings, gcfg)
    out = tree.out_batch()
    assert [p.name for p in out.iterdir()] == ["geometry"], "no raw file, sidecar or video sits in the out batch"
    before = tree.read_only()

    summary = S.run_normalize(settings, gcfg)
    assert summary == {str(out): [1, 2, 3]}
    for slot in (1, 2, 3):
        assert C.stats_path(out / "geometry", slot).is_file()
    assert C.NORMALIZED_COLUMNS[0] in C.read_table_any(next(iter(C.iter_geometry_tables(tree.trial_dir(1))))[1]).columns

    wide = S.run_wide(settings, gcfg)
    assert wide["rows"] == {m: 9 for m in wide["rows"]}, "3 trials x 3 flies per measure"
    per = pd.read_csv(wide["files"]["per"])
    assert set(per["dataset"]) == {DATASET} and set(per["fly"]) == {BATCH} and set(per["frozen"]) == {False}

    figs = S.run_figures(settings, gcfg)
    assert figs["n_figures"] == 7                                         # (traces + size) x 3 flies + the QC grid
    assert len(list((Path(gcfg.figures_out_dir) / DATASET / BATCH).glob("*.png"))) == 7

    assert [p.name for p in out.iterdir()] == ["geometry"]
    assert not tree.out_batch(RIG_BATCH).exists(), "jobs without a geometry dir are skipped, not created"
    assert tree.read_only() == before


def test_the_steps_do_not_even_visit_jobs_without_a_geometry_dir(tree, fake_model, monkeypatch):
    from fbpipe.geometry import figures as F
    from fbpipe.geometry import normalize as N
    from fbpipe.geometry import wide as W
    settings, gcfg = tree.settings(), tree.gcfg()
    assert S.run_normalize(settings, gcfg) == {} and not tree.out_base.exists()       # nothing tracked yet
    S.run_track(settings, gcfg)                                                       # only BATCH gets a geometry dir
    visited = []
    monkeypatch.setattr(N, "normalize_batch", lambda batch_dir, *a: visited.append(("normalize", batch_dir))
                        or {"n_trials": 0, "slots": []})
    monkeypatch.setattr(W, "batch_wide_rows", lambda batch_dir, *a: visited.append(("wide", batch_dir))
                        or {m: [] for m in W.WIDE_MEASURES})
    monkeypatch.setattr(F, "generate", lambda batch_dir, *a: visited.append(("figures", batch_dir))
                        or {"n_figures": 0, "out_dir": None})
    S.run_normalize(settings, gcfg)
    S.run_wide(settings, gcfg)
    S.run_figures(settings, gcfg)
    assert visited == [(step, tree.out_batch()) for step in ("normalize", "wide", "figures")]


def test_collect_fly_trial_metrics_walks_the_jobs_of_the_out_tree(tree, fake_model):
    settings, gcfg = tree.settings(), tree.gcfg()
    S.run_track(settings, gcfg)
    S.run_normalize(settings, gcfg)
    df = ST.collect_fly_trial_metrics(settings, gcfg)
    assert len(df) == 9 and set(df["fly"]) == {BATCH} and set(df["dataset"]) == {DATASET}
    frozen = tree.frozen(BATCH)
    assert ST.collect_fly_trial_metrics(frozen, gcfg).empty, "a frozen job stays out unless asked for"
    kept = ST.collect_fly_trial_metrics(frozen, gcfg, include_frozen=True)
    assert len(kept) == 9 and set(kept["frozen"]) == {True}


# ---- pending trials and the CLI -------------------------------------------------------------------

def test_pending_trials_counts_the_videos_of_live_jobs_that_have_no_meta_yet(tree, fake_model):
    settings, gcfg = tree.settings(), tree.gcfg()
    assert S.pending_trials(settings, gcfg) == 3                # the stale september video is outside the cut-off
    S.run_track(settings, gcfg)
    assert S.pending_trials(settings, gcfg) == 0
    tree.meta(2).unlink()
    assert S.pending_trials(settings, gcfg) == 1
    assert S.pending_trials(tree.frozen(BATCH), gcfg) == 0, "a frozen batch is not live, so nothing in it is pending"


def test_main_spawns_no_workers_and_loads_no_model_when_nothing_is_pending(tree, fake_model, monkeypatch, capsys):
    settings, gcfg = dataclasses.replace(tree.settings(), yolo_num_workers=3), tree.gcfg()
    S.run_track(settings, gcfg)
    fake_model.clear()
    spawned, ran = _cli(monkeypatch, settings, gcfg)
    assert GP.main(["--config", "c.yaml", "track", "normalize"]) == 0
    assert spawned == [] and fake_model == [], "no fan-out and no model load"
    assert ran == ["normalize"], "the remaining steps still run"
    assert "nothing pending" in capsys.readouterr().out
    # the same holds for a worker process and for a single in-process run
    assert GP.main(["--config", "c.yaml", "--no-fan-out", "track"]) == 0
    assert GP.main(["--config", "c.yaml", "--workers", "1", "track"]) == 0
    assert fake_model == []


def test_main_still_tracks_when_trials_are_pending(tree, fake_model, monkeypatch):
    settings, gcfg = dataclasses.replace(tree.settings(), yolo_num_workers=3), tree.gcfg()
    spawned, ran = _cli(monkeypatch, settings, gcfg)
    assert GP.main(["--config", "c.yaml", "track", "normalize"]) == 0
    assert spawned == [3] and fake_model == [] and ran == ["normalize"], "the workers do the tracking, not this process"
    assert GP.main(["--config", "c.yaml", "--workers", "1", "track"]) == 0
    assert len(fake_model) == 1 and all(tree.meta(n).is_file() for n in (1, 2, 3))


def test_force_tracks_even_when_every_trial_already_has_a_meta(tree, fake_model, monkeypatch):
    settings, gcfg = dataclasses.replace(tree.settings(), yolo_num_workers=3), tree.gcfg()
    S.run_track(settings, gcfg)
    fake_model.clear()
    spawned, ran = _cli(monkeypatch, settings, gcfg)
    assert GP.main(["--config", "c.yaml", "--force", "track"]) == 0
    assert spawned == [3], "--force means redo: an all-tracked tree is not 'nothing to do'"
    assert GP.main(["--config", "c.yaml", "--force", "--workers", "1", "track"]) == 0
    assert len(fake_model) == 1


def test_the_fan_out_workers_are_spawned_to_die_with_their_parent(tmp_path, monkeypatch):
    """The stage kills a stage command's process group on a timeout or a SIGTERM of the driver, but nothing can catch a
    SIGKILL: when that takes the geometry process, its workers must go with it (``PR_SET_PDEATHSIG``) instead of running on
    with no deadline."""
    sentinel, spawned = object(), []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            spawned.append(kwargs)
            self.pid = 1

        def wait(self):
            return 0

    monkeypatch.setattr(GP, "die_with_parent_preexec", lambda: sentinel)
    monkeypatch.setattr(GP.subprocess, "Popen", FakePopen)
    settings = Settings(model_path="x", main_directories=[], cache_dir=str(tmp_path))
    assert GP.fan_out_track(Path("c.yaml"), 3, settings=settings, options={}) == 0
    assert [kwargs["preexec_fn"] for kwargs in spawned] == [sentinel] * 3


def test_the_fan_out_summary_reports_each_workers_failed_videos(tmp_path, monkeypatch, capsys):
    class FakePopen:
        def __init__(self, cmd, **kwargs):
            kwargs["stdout"].write("[GEOMETRY] a.avi: 3 slot(s), 40 frames in 0.1s -> /out\n"
                                   "[GEOMETRY] failed /mirror/b.mp4: RuntimeError: cannot open video: /mirror/b.mp4\n")
            kwargs["stdout"].flush()
            self.pid = 1

        def wait(self):
            return 0

    monkeypatch.setattr(GP.subprocess, "Popen", FakePopen)
    settings = Settings(model_path="x", main_directories=[], cache_dir=str(tmp_path))
    assert GP.fan_out_track(Path("c.yaml"), 2, settings=settings, options={}) == 0
    assert capsys.readouterr().out.count("OK | tracked~1 failed~1") == 2
