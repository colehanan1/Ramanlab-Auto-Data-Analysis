"""How yolo_infer uses the batch eye template.

The template is built once per batch, before any of its videos are processed,
because the alternative -- filling a cache as trials go by -- depends on which
worker got which trial (``_job_idx % num_workers``) and fails outright on the
batch that prompted this: ``october_09_batch_3``'s first trial is one of the
eighteen that miss the edge fly, so there would be nothing to borrow.

These tests drive the pre-pass with a stand-in scanner, so they need no GPU,
no model and no video.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbpipe.config import EyeTemplateSettings, Settings  # noqa: E402
from fbpipe.steps import yolo_infer  # noqa: E402
from fbpipe.utils import eye_template as et  # noqa: E402


def _box(cx, cy, w=60.0, h=52.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _boxes(*centres):
    if not centres:
        return np.zeros((0, 4), np.float32)
    return np.array([_box(cx, cy) for cx, cy in centres], dtype=np.float32)


EDGE_FLY = (721.0, 4.0)
ALWAYS_SEEN = [(635.0, 280.0), (495.0, 570.0), (467.0, 830.0)]


@pytest.fixture
def batch(tmp_path):
    """21 trial videos; 3 of them show the edge fly."""
    paths = []
    for i in range(21):
        p = tmp_path / f"out_october_09_batch_3_trial_{i:02d}.mp4"
        p.write_bytes(b"")
        paths.append(p)
    return tmp_path, paths


def _scanner(paths, with_edge_fly=(1, 2, 6), edge_pairs=True):
    """Stand-in for the 5-frame warm-up scan; records what it was asked for.

    Returns ``(boxes, paired)`` the way ``_scan_initial_eye_boxes`` does:
    whether each detected eye had a proboscis inside its gate. ``edge_pairs``
    is the october (True) vs june (False) case -- an edge eye whose proboscis
    is in frame, or one whose proboscis never is.
    """
    calls = []

    def scan(video_path):
        calls.append(video_path)
        idx = paths.index(video_path)
        centres = list(ALWAYS_SEEN)
        paired = [True] * len(centres)
        if idx in with_edge_fly:
            centres = [EDGE_FLY] + centres
            paired = [bool(edge_pairs)] + paired
        return _boxes(*centres), paired

    return scan, calls


def _cfg(**kw):
    return Settings(
        model_path="model.pt",
        main_directories=[],
        eye_template=EyeTemplateSettings(**kw),
    )


def test_the_pre_pass_scans_every_video_once_and_finds_four_slots(batch):
    batch_dir, paths = batch
    scan, calls = _scanner(paths)
    slots, scanned = yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(), scan)

    assert len(slots) == 4
    assert round(slots[0].cy) == 4
    assert sorted(calls) == sorted(paths)
    assert len(calls) == len(paths)
    assert set(scanned) == set(paths)


def test_the_pre_pass_hands_back_its_scans_so_the_main_loop_need_not_repeat_them(batch):
    batch_dir, paths = batch
    scan, _ = _scanner(paths)
    _, scanned = yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(), scan)
    assert np.allclose(scanned[paths[1]], _boxes(EDGE_FLY, *ALWAYS_SEEN))
    assert np.allclose(scanned[paths[0]], _boxes(*ALWAYS_SEEN))


def test_a_cached_template_costs_no_scans_at_all(batch):
    """Every worker slicing the batch would otherwise re-scan all 21 videos."""
    batch_dir, paths = batch
    scan, _ = _scanner(paths)
    yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(), scan)

    scan2, calls2 = _scanner(paths)
    slots, scanned = yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(), scan2)
    assert calls2 == []
    assert scanned == {}
    assert len(slots) == 4


def test_a_cache_built_under_a_different_min_trials_is_rebuilt(batch):
    batch_dir, paths = batch
    scan, _ = _scanner(paths)
    yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(min_trials=2), scan)

    scan2, calls2 = _scanner(paths)
    slots, _ = yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(min_trials=4), scan2)
    assert calls2 != []
    assert len(slots) == 3  # the edge fly is in 3 trials, one short of the 4 required


def test_a_slot_short_of_min_trials_does_not_join_the_template(batch):
    batch_dir, paths = batch
    scan, _ = _scanner(paths, with_edge_fly=(1,))  # one trial only
    slots, _ = yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(), scan)
    assert len(slots) == 3


def test_disabled_never_scans_and_yields_no_template(batch):
    batch_dir, paths = batch
    scan, calls = _scanner(paths)
    slots, scanned = yolo_infer.load_or_build_eye_template(
        batch_dir, paths, _cfg(enabled=False), scan
    )
    assert slots is None
    assert scanned == {}
    assert calls == []


def test_disabled_writes_no_cache_file(batch):
    batch_dir, paths = batch
    scan, _ = _scanner(paths)
    yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(enabled=False), scan)
    assert not et.template_path(batch_dir).exists()


def test_a_single_video_batch_falls_back_rather_than_emptying_the_slot_set(tmp_path):
    """min_trials=2 cannot be met by one trial. The template must read as
    absent, not as a batch with zero flies."""
    only = tmp_path / "out_batch_trial_1.mp4"
    only.write_bytes(b"")
    scan, _ = _scanner([only], with_edge_fly=())
    slots, _ = yolo_infer.load_or_build_eye_template(tmp_path, [only], _cfg(), scan)
    assert slots is None


def test_a_batch_with_no_videos_yields_no_template(tmp_path):
    scan, calls = _scanner([])
    slots, _ = yolo_infer.load_or_build_eye_template(tmp_path, [], _cfg(), scan)
    assert slots is None
    assert calls == []


# ── Slot count handed to the rest of the pipeline ────────────────────────────


def test_the_template_sets_the_trials_slot_count(batch):
    batch_dir, paths = batch
    scan, _ = _scanner(paths)
    slots, _ = yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(), scan)
    cfg = dataclasses.replace(_cfg(), max_flies=4)
    assert yolo_infer._template_slot_count(slots, cfg) == 4


def test_max_flies_still_caps_the_template(batch):
    """A template longer than max_flies must not widen the tracker past it."""
    batch_dir, paths = batch
    scan, _ = _scanner(paths)
    slots, _ = yolo_infer.load_or_build_eye_template(batch_dir, paths, _cfg(), scan)
    cfg = dataclasses.replace(_cfg(), max_flies=2)
    assert yolo_infer._template_slot_count(slots, cfg) == 2


def test_no_template_means_the_old_per_video_sizing(batch):
    cfg = _cfg()
    assert yolo_infer._template_slot_count(None, cfg) is None


# ── The scan reports pairing, and the template uses it ───────────────────────


def test_an_edge_slot_that_never_pairs_a_proboscis_is_left_out(batch):
    """june_03_batch_2: the bottom-edge eye clears min_trials but its
    proboscis is out of frame, so it would only ever give an empty trace."""
    batch_dir, paths = batch
    slots, _ = yolo_infer.load_or_build_eye_template(
        batch_dir, paths, _cfg(), _scanner(paths, edge_pairs=False)[0]
    )
    assert len(slots) == 3
    assert all(round(s.cy) != 4 for s in slots)


def test_an_edge_slot_that_pairs_a_proboscis_is_kept(batch):
    """october_09_batch_3: the top-edge eye pairs in every frame it is seen."""
    batch_dir, paths = batch
    slots, _ = yolo_infer.load_or_build_eye_template(
        batch_dir, paths, _cfg(), _scanner(paths, edge_pairs=True)[0]
    )
    assert len(slots) == 4
    assert round(slots[0].cy) == 4


def test_turning_the_requirement_off_keeps_the_unpairable_slot(batch):
    batch_dir, paths = batch
    slots, _ = yolo_infer.load_or_build_eye_template(
        batch_dir, paths, _cfg(require_proboscis=False), _scanner(paths, edge_pairs=False)[0]
    )
    assert len(slots) == 4
