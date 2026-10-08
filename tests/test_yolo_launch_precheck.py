"""run_workflows skips the YOLO worker launch for a root holding no video.

Every night main() launched 3 YOLO workers for each of 27 dataset roots; 26
had nothing to infer and each launch still cost ~1.4 s of python + torch
start-up. The pre-check must be a SUPERSET of yolo_infer's own discovery: it
may launch more often than needed, never less. A pending video it missed would
be copied to secured storage and deleted by the secure cleanup without ever
being inferred, silently losing the trial. So it counts any video-suffixed
entry anywhere under the root -- ignoring folder freeze, depth and type -- and
the superset property is pinned against yolo_infer.main's real discovery.

A launch that would FAIL must happen too. Without CUDA (allow_cpu off), or
with no model file, yolo_infer.main raises before it looks for a video, so
today's launch stops the night loudly even for an empty root. Skipped, the
night would go on and the per-trial steps would run their CPU variants, whose
outputs differ from the GPU path's. The contract, pinned against the real
worker below: a skipped launch is one that would have run to completion
without opening a video or writing a file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

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


def _model(tmp_path: Path, name: str = "best.engine") -> Path:
    """A stand-in for the live model, laid out like it (runs/obb/.../weights).
    Ultralytics loads an engine lazily, at the first predict, so YOLO() on it
    only needs the file to exist."""
    path = tmp_path / "runs" / "obb" / "train" / "weights" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stand-in engine")
    return path


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


def test_a_root_that_cannot_be_stat_ed_counts_as_maybe_video(tmp_path, monkeypatch):
    # Python 3.12's Path.is_dir() re-raises PermissionError. The pre-check must
    # not crash main() over it: launching lets the worker meet the error exactly
    # where today's launch would.
    root = _build(tmp_path, [f"{BATCH}/{STEM}.csv"])
    real_is_dir = Path.is_dir

    def _is_dir(self):
        if self == root:
            raise PermissionError(str(self))
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", _is_dir)
    assert rw._tree_has_video(root) is True


def test_loop_launches_when_the_precheck_itself_fails(tmp_path, monkeypatch, capsys):
    make_dataset(tmp_path, "Live-A", [f"{BATCH}/{STEM}.csv"])
    cfg = write_config(tmp_path, ["Live-A"])

    def _boom(settings, root):
        raise OSError("flagged root unreadable")

    monkeypatch.setattr(rw, "_yolo_work_reason", _boom)
    calls = run_main(monkeypatch, cfg)
    assert yolo_launches(calls) == ["Live-A"]
    assert per_trial_runs(calls) == ["Live-A"]
    assert "flagged root unreadable" in capsys.readouterr().out


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


def _snapshot(top: Path) -> dict:
    return {
        str(p.relative_to(top)): (p.is_dir(), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(top.rglob("*"))
    }


@pytest.mark.parametrize(
    "label,files",
    [(label, files) for label, files in LAYOUTS if label in NO_VIDEO],
    ids=[label for label, _ in LAYOUTS if label in NO_VIDEO],
)
def test_a_launch_the_precheck_skips_would_have_written_nothing(tmp_path, monkeypatch, label, files):
    """Skipping a launch preserves outputs only while yolo_infer, given a root
    with no video, creates or touches nothing. Should yolo_infer ever start to,
    the skip has to be revisited."""
    root = _build(tmp_path, files)
    assert rw._tree_has_video(root) is False
    before = _snapshot(tmp_path)
    assert _yolo_discovery(monkeypatch, tmp_path, root) == []
    assert _snapshot(tmp_path) == before


# --- launches that walk more than the one root ------------------------------

def test_main_directories_env_always_launches(tmp_path, monkeypatch):
    # The worker reads MAIN_DIRECTORIES ahead of the MAIN_DIRECTORY it is
    # handed, so the one root says nothing about what the launch would see.
    isolate_env(monkeypatch)
    root = _build(tmp_path, [])
    settings = Settings(model_path=str(_model(tmp_path)), main_directories=[str(root)])
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
    settings = Settings(model_path=str(_model(tmp_path)), main_directories=[str(root)])
    assert rw._tree_has_video(root) is False
    assert rw._yolo_may_have_work(settings, str(root)) is True


def test_auto_discovered_flagged_dirs_always_launch(tmp_path, monkeypatch):
    # get_main_directories appends every flagged experiment dir to each run.
    isolate_env(monkeypatch)
    root = _build(tmp_path, [])
    flagged = tmp_path / "flagged_root"
    (flagged / "some_flagged_experiment").mkdir(parents=True)
    settings = Settings(
        model_path=str(_model(tmp_path)), main_directories=[str(root)], flagged_root=str(flagged)
    )
    assert rw._yolo_may_have_work(settings, str(root)) is True
    settings.auto_discover_flagged = False
    assert rw._yolo_may_have_work(settings, str(root)) is False


# --- launches that would fail before they look for a video -------------------

def test_no_cuda_keeps_the_launch_so_the_night_still_fails_loudly(tmp_path, monkeypatch):
    isolate_env(monkeypatch)
    root = _build(tmp_path, [])
    settings = Settings(model_path=str(_model(tmp_path)), main_directories=[str(root)])
    assert rw._yolo_may_have_work(settings, str(root)) is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert rw._yolo_may_have_work(settings, str(root)) is True
    assert "CUDA is unavailable" in rw._yolo_work_reason(settings, str(root))
    # allow_cpu (smoke tests only) runs YOLO on the CPU instead of raising, so
    # an empty root is a no-op again.
    settings.allow_cpu = True
    assert rw._yolo_may_have_work(settings, str(root)) is False


@pytest.mark.parametrize(
    "present,model_name,must_launch",
    [
        ((), "best.engine", True),
        (("best.engine",), "best.engine", False),  # the live layout
        (("best.engine",), "best.pt", False),  # yolo_infer prefers the .engine sibling
        (("best.pt",), "best.pt", False),  # ...and falls back to the .pt itself
        (("other.engine",), "best.pt", True),  # nothing YOLO() would load
    ],
    ids=["nothing", "engine", "engine beside the .pt", ".pt only", "unrelated file"],
)
def test_a_missing_model_keeps_the_launch(tmp_path, monkeypatch, present, model_name, must_launch):
    # YOLO() raises FileNotFoundError before discovery: today's launch fails.
    isolate_env(monkeypatch)
    root = _build(tmp_path, [])
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in present:
        (weights / name).write_bytes(b"x")
    settings = Settings(model_path=str(weights / model_name), main_directories=[str(root)])
    assert rw._yolo_may_have_work(settings, str(root)) is must_launch
    if must_launch:
        assert "is missing" in rw._yolo_work_reason(settings, str(root))


# (label, CUDA available, allow_cpu, model file present)
WORKER_STATES = [
    ("gpu", True, False, True),  # the nightly
    ("no cuda", False, False, True),
    ("no cuda, allow_cpu", False, True, True),
    ("no model", True, False, False),
    ("no model, allow_cpu", False, True, False),
]


def _real_yolo_main(monkeypatch, settings):
    """Run yolo_infer.main as a launch would, with the REAL ultralytics YOLO
    class; only cv2.VideoCapture is stubbed. Returns (exception, opened)."""
    yolo_infer = pytest.importorskip("fbpipe.steps.yolo_infer")
    opened: list[Path] = []

    class _Capture:
        def __init__(self, path):
            opened.append(Path(path))

        def isOpened(self):
            return False

    monkeypatch.delenv("NUM_WORKERS", raising=False)
    monkeypatch.delenv("WORKER_INDEX", raising=False)
    monkeypatch.setattr(yolo_infer.cv2, "VideoCapture", _Capture)
    # yolo_infer.main sets these process-wide; have them put back afterwards.
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    try:
        yolo_infer.main(settings)
    except Exception as exc:  # what a failing worker would exit with
        return exc, opened
    return None, opened


def _worker_settings(tmp_path, monkeypatch, root, state):
    _, cuda, allow_cpu, model_present = state
    model = _model(tmp_path)
    if not model_present:
        model.unlink()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    return Settings(
        model_path=str(model),
        main_directories=[str(root)],
        allow_cpu=allow_cpu,
        dataset_overrides={DS: DatasetOverride(freeze_folders=(FolderRule("frozen_*"),))},
    )


@pytest.mark.parametrize("state", WORKER_STATES, ids=[s[0] for s in WORKER_STATES])
@pytest.mark.parametrize("label,files", LAYOUTS, ids=[label for label, _ in LAYOUTS])
def test_a_skipped_launch_is_one_that_would_have_succeeded_as_a_no_op(
    tmp_path, monkeypatch, label, files, state
):
    isolate_env(monkeypatch)
    root = _build(tmp_path, files)
    settings = _worker_settings(tmp_path, monkeypatch, root, state)
    # Decide BEFORE yolo_infer runs: it creates trial out_dirs as it goes.
    skipped = not rw._yolo_may_have_work(settings, str(root))
    before = _snapshot(tmp_path)
    raised, opened = _real_yolo_main(monkeypatch, settings)
    if skipped:
        assert raised is None, f"{label}/{state[0]}: the skipped launch would have failed: {raised!r}"
        assert opened == []
        assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "state,error",
    [(WORKER_STATES[1], RuntimeError), (WORKER_STATES[3], FileNotFoundError),
     (WORKER_STATES[4], FileNotFoundError)],
    ids=["no cuda", "no model", "no model, allow_cpu"],
)
def test_the_real_worker_fails_an_empty_root_in_those_states(tmp_path, monkeypatch, state, error):
    """Sanity check on the harness above: these launches really do fail, so the
    skip test cannot pass by never meeting a failing one."""
    isolate_env(monkeypatch)
    root = _build(tmp_path, [])
    settings = _worker_settings(tmp_path, monkeypatch, root, state)
    raised, opened = _real_yolo_main(monkeypatch, settings)
    assert isinstance(raised, error)
    assert opened == []
    assert rw._yolo_may_have_work(settings, str(root)) is True


# --- the dataset loop ---------------------------------------------------------

def test_loop_without_cuda_launches_every_root_as_today(tmp_path, monkeypatch):
    for name in ("Live-A", "Live-B"):
        make_dataset(tmp_path, name, [f"{BATCH}/{STEM}.csv"])
    cfg = write_config(tmp_path, ["Live-A", "Live-B"])
    calls = run_main(monkeypatch, cfg, cuda=False)
    assert yolo_launches(calls) == ["Live-A", "Live-B"]


def test_loop_without_cuda_still_stops_at_the_first_failing_launch(tmp_path, monkeypatch):
    # Today the first YOLO worker exits non-zero, _run_pipeline raises and the
    # night ends there -- before any per-trial step could run on the CPU.
    for name in ("Live-A", "Live-B"):
        make_dataset(tmp_path, name, [f"{BATCH}/{STEM}.csv"])
    cfg = write_config(tmp_path, ["Live-A", "Live-B"])

    def _worker_fails(main_directory, steps):
        if steps == ("yolo",):
            raise subprocess.CalledProcessError(1, ["fbpipe.pipeline", "yolo"])

    calls = run_main(
        monkeypatch, cfg, cuda=False, on_pipeline=_worker_fails,
        expect=subprocess.CalledProcessError,
    )
    assert yolo_launches(calls) == ["Live-A"]
    assert per_trial_runs(calls) == []


def test_loop_with_allow_cpu_skips_empty_launches_without_cuda(tmp_path, monkeypatch):
    for name in ("Live-A", "Live-B"):
        make_dataset(tmp_path, name, [f"{BATCH}/{STEM}.csv"])
    cfg = write_config(tmp_path, ["Live-A", "Live-B"], allow_cpu=True)
    calls = run_main(monkeypatch, cfg, cuda=False)
    assert yolo_launches(calls) == []
    assert per_trial_runs(calls) == ["Live-A", "Live-B"]


def test_loop_decides_on_the_env_overrides_the_workers_inherit(tmp_path, monkeypatch):
    # load_settings lets MODEL_PATH / ALLOW_CPU in the environment override the
    # config, and every worker inherits the driver's environment. The pre-check
    # has to see the same values the worker will act on.
    for name in ("Live-A", "Live-B"):
        make_dataset(tmp_path, name, [f"{BATCH}/{STEM}.csv"])
    cfg = write_config(tmp_path, ["Live-A", "Live-B"])  # its model file exists
    calls = run_main(monkeypatch, cfg, env={"MODEL_PATH": str(tmp_path / "gone.engine")})
    assert yolo_launches(calls) == ["Live-A", "Live-B"]
    calls = run_main(monkeypatch, cfg, cuda=False, env={"ALLOW_CPU": "true"})
    assert yolo_launches(calls) == []
    assert per_trial_runs(calls) == ["Live-A", "Live-B"]


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
