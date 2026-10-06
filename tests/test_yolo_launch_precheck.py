"""run_workflows skips the YOLO worker launch for a root holding no video.

Every night main() launched 3 YOLO workers for each of 27 dataset roots; 26
had nothing to infer and each launch still cost ~1.4 s of python + torch
start-up. The pre-check must be a SUPERSET of yolo_infer's own discovery: it
may launch more often than needed, never less. A pending video it missed would
be copied to secured storage and deleted by the secure cleanup without ever
being inferred, silently losing the trial. So it counts any video-suffixed
entry anywhere under the root -- ignoring folder freeze, depth and type -- and
the superset property is pinned against yolo_infer.main's real discovery.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbpipe.config import DatasetOverride, Settings  # noqa: E402
from fbpipe.utils.frozen_folders import FolderRule  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402

from _loop_harness import (  # noqa: E402
    isolate_env,
    make_dataset,
    per_trial_runs,
    run_main,
    write_config,
    yolo_launches,
)

DS = "EB-Training-24-1"
BATCH = "august_13_batch_2_rig_3"
STEM = "output_august_13_batch_2_rig_3_testing_1_Hexanol_20260813_120000"

# (label, files relative to the dataset root). Every layout but the last three
# holds something with a video suffix.
LAYOUTS = [
    ("batch-level mp4", [f"{BATCH}/{STEM}.csv", f"{BATCH}/{STEM}.mp4"]),
    ("batch-level AVI, upper case", [f"{BATCH}/{STEM}.AVI"]),
    ("raw h264 only", [f"{BATCH}/{STEM}.h264"]),
    ("nested below a trial dir", [f"{BATCH}/trial_1/clip.mp4"]),
    ("deeply nested", [f"flagged/{BATCH}/{STEM}.mp4"]),
    ("in a folder-frozen batch", [f"frozen_batch/{STEM}.mp4"]),
    ("loose at the dataset root", [f"{STEM}.mp4"]),
    ("a container move_videos relocates", [f"{BATCH}/{BATCH}_testing_1/clip.MKV"]),
    ("tables only", [f"{BATCH}/{STEM}.csv", f"{BATCH}/t/fly1_distances.parquet"]),
    ("video-like names that are not videos", [f"{BATCH}/{STEM}.mp4.txt", f"{BATCH}/mp4"]),
    ("empty batch", []),
]
NO_VIDEO = {"tables only", "video-like names that are not videos", "empty batch"}


def _build(tmp_path: Path, files) -> Path:
    root = tmp_path / DS
    (root / BATCH).mkdir(parents=True)
    (root / "frozen_batch").mkdir()
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("label,files", LAYOUTS, ids=[label for label, _ in LAYOUTS])
def test_tree_has_video(tmp_path, label, files):
    root = _build(tmp_path, files)
    assert rw._tree_has_video(root) is (label not in NO_VIDEO)


def test_suffixes_cover_every_step_that_acts_on_a_video():
    """yolo_infer's own .mp4/.avi, the .mp4/.h264 the secure cleanup prunes and
    every container the per-trial move_videos step relocates. A suffix one of
    them acts on but the pre-check ignored could skip a launch -- or a frozen
    dataset's per-trial run -- that had a video to handle."""
    from fbpipe.steps import move_videos
    from scripts.analysis.envelope_combined import PRUNABLE_VIDEO_SUFFIXES

    covered = set(rw._PRECHECK_VIDEO_SUFFIXES)
    assert {".mp4", ".avi"} <= covered  # yolo_infer.main's own filter
    assert set(PRUNABLE_VIDEO_SUFFIXES) <= covered
    assert set(move_videos.VIDEO_EXTS) <= covered
    assert all(s == s.lower() for s in covered)  # matched against name.lower()


def test_suffixes_cover_compose_videos_rms():
    compose = pytest.importorskip("fbpipe.steps.compose_videos_rms")
    assert set(compose.VIDEO_EXTS) <= set(rw._PRECHECK_VIDEO_SUFFIXES)


def test_a_directory_with_a_video_suffix_counts(tmp_path):
    # yolo_infer filters on suffix alone, so it would "discover" this too.
    root = _build(tmp_path, [])
    (root / BATCH / f"{STEM}.mp4").mkdir()
    assert rw._tree_has_video(root) is True


def test_missing_root_has_no_video(tmp_path):
    # yolo_infer prints "does not exist" and moves on: nothing to launch for.
    assert rw._tree_has_video(tmp_path / "nope") is False


def test_symlinked_batch_dir_is_followed(tmp_path):
    # iter_live_batch_dirs keeps a symlinked batch (is_dir follows links).
    elsewhere = tmp_path / "elsewhere" / BATCH
    elsewhere.mkdir(parents=True)
    (elsewhere / f"{STEM}.mp4").write_text("x\n", encoding="utf-8")
    root = tmp_path / DS
    root.mkdir()
    (root / BATCH).symlink_to(elsewhere, target_is_directory=True)
    assert rw._tree_has_video(root) is True


def test_symlink_loop_terminates(tmp_path):
    root = _build(tmp_path, [f"{BATCH}/{STEM}.csv"])
    (root / BATCH / "loop").symlink_to(root, target_is_directory=True)
    assert rw._tree_has_video(root) is False


def test_unreadable_dir_counts_as_maybe_video(tmp_path, monkeypatch):
    root = _build(tmp_path, [f"{BATCH}/{STEM}.csv"])
    real_scandir = rw.os.scandir

    def _scandir(path):
        if Path(path).name == BATCH:
            raise PermissionError(path)
        return real_scandir(path)

    monkeypatch.setattr(rw.os, "scandir", _scandir)
    assert rw._tree_has_video(root) is True


# --- superset of yolo_infer.main's own discovery ----------------------------

def _yolo_discovery(monkeypatch, tmp_path, root: Path) -> list[Path]:
    """Every video yolo_infer.main tries to open under *root* (its own walk)."""
    yolo_infer = pytest.importorskip("fbpipe.steps.yolo_infer")
    opened: list[Path] = []

    class _Model:
        def __init__(self, *a, **k):
            pass

        def to(self, *a, **k):
            return self

    class _Capture:
        def __init__(self, path):
            opened.append(Path(path))

        def isOpened(self):
            return False

    monkeypatch.delenv("NUM_WORKERS", raising=False)
    monkeypatch.delenv("WORKER_INDEX", raising=False)
    monkeypatch.setattr(yolo_infer, "YOLO", _Model)
    monkeypatch.setattr(yolo_infer.cv2, "VideoCapture", _Capture)
    monkeypatch.setattr(yolo_infer.torch.cuda, "is_available", lambda: False)
    cfg = Settings(
        model_path=str(tmp_path / "m.pt"),
        main_directories=[str(root)],
        allow_cpu=True,
        dataset_overrides={DS: DatasetOverride(freeze_folders=(FolderRule("frozen_*"),))},
    )
    yolo_infer.main(cfg)
    return opened


@pytest.mark.parametrize("label,files", LAYOUTS, ids=[label for label, _ in LAYOUTS])
def test_precheck_is_a_superset_of_yolo_discovery(tmp_path, monkeypatch, label, files):
    root = _build(tmp_path, files)
    # Decide BEFORE yolo_infer runs: it creates trial out_dirs as it goes.
    launches = rw._tree_has_video(root)
    discovered = _yolo_discovery(monkeypatch, tmp_path, root)
    if discovered:
        assert launches, f"{label}: yolo_infer would process {discovered}"


def test_superset_covers_yolos_whole_batch_level_discovery(tmp_path, monkeypatch):
    """Sanity check on the harness: it really is yolo_infer's walk."""
    root = _build(tmp_path, [f"{BATCH}/{STEM}.mp4", f"frozen_batch/{STEM}.mp4"])
    discovered = _yolo_discovery(monkeypatch, tmp_path, root)
    # The folder-frozen batch is skipped by yolo, the live one is not.
    assert [p.parent.name for p in discovered] == [BATCH]


# --- launches that walk more than the one root ------------------------------

def test_main_directories_env_always_launches(tmp_path, monkeypatch):
    # The worker reads MAIN_DIRECTORIES ahead of the MAIN_DIRECTORY it is
    # handed, so the one root says nothing about what the launch would see.
    isolate_env(monkeypatch)
    root = _build(tmp_path, [])
    settings = Settings(model_path="m.pt", main_directories=[str(root)])
    assert rw._yolo_may_have_work(settings, str(root)) is False
    monkeypatch.setenv("MAIN_DIRECTORIES", str(tmp_path / "other"))
    assert rw._yolo_may_have_work(settings, str(root)) is True


@pytest.mark.parametrize(
    "name", ["Hex,Control", f"Hex{os.pathsep}Control", "Hex-Control "],
    ids=["comma", "pathsep", "trailing space"],
)
def test_a_root_the_worker_would_not_read_as_given_always_launches(tmp_path, monkeypatch, name):
    # _run_pipeline hands the root over as MAIN_DIRECTORY, which load_settings
    # splits on "," / os.pathsep and strips (config._parse_env_paths): that
    # worker walks other dirs than the one this pre-check could look at.
    isolate_env(monkeypatch)
    root = tmp_path / name
    (root / BATCH).mkdir(parents=True)
    settings = Settings(model_path="m.pt", main_directories=[str(root)])
    assert rw._tree_has_video(root) is False
    assert rw._yolo_may_have_work(settings, str(root)) is True


def test_auto_discovered_flagged_dirs_always_launch(tmp_path, monkeypatch):
    # get_main_directories appends every flagged experiment dir to each run.
    isolate_env(monkeypatch)
    root = _build(tmp_path, [])
    flagged = tmp_path / "flagged_root"
    (flagged / "some_flagged_experiment").mkdir(parents=True)
    settings = Settings(
        model_path="m.pt", main_directories=[str(root)], flagged_root=str(flagged)
    )
    assert rw._yolo_may_have_work(settings, str(root)) is True
    settings.auto_discover_flagged = False
    assert rw._yolo_may_have_work(settings, str(root)) is False


# --- the dataset loop ---------------------------------------------------------

def test_loop_launches_yolo_only_where_a_video_exists(tmp_path, monkeypatch, capsys):
    make_dataset(tmp_path, "Live-A", [f"{BATCH}/{STEM}.csv", f"{BATCH}/{STEM}.mp4"])
    make_dataset(tmp_path, "Live-B", [f"{BATCH}/{STEM}.csv"])
    cfg = write_config(tmp_path, ["Live-A", "Live-B"])

    calls = run_main(monkeypatch, cfg)

    assert yolo_launches(calls) == ["Live-A"]
    # Still 3 workers for the launch that does happen.
    assert [n for d, steps, n in calls["pipeline"] if steps == ("yolo",)] == [3]
    # The per-trial steps are untouched by the YOLO pre-check.
    assert per_trial_runs(calls) == ["Live-A", "Live-B"]
    assert "no video files under" in capsys.readouterr().out
