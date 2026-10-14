"""_resolve_trial_meta_for_csv: rig batches resolve their own trial folders.

A rig batch (``august_28_batch_1_rig_3``) names its trial folders WITHOUT the
rig suffix (``august_28_batch_1_testing_1``). The resolver inferred
``<batch>_<type>_<N>``, missed, and keyed every trial of the batch on ONE
``angle_distance_rms_envelope/_trial_meta.json`` -- a cache keyed only on the
sidecar's mtime. That thrashed on every call, and a trial with no sidecar
(mtime None skips the check) read back another trial's metadata.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev
from fbpipe.utils import trial_metadata as tm

ENV = "angle_distance_rms_envelope"


@pytest.fixture(autouse=True)
def _v2_without_settings(monkeypatch):
    ev.set_protocol("v2")
    # Config defaults (fps 40, odor 30-60 s); no dataset overrides.
    monkeypatch.setattr(ec, "_RUNTIME_SETTINGS", None)


def _sidecar(batch_dir: Path, stem: str, trial_type: str, idx: int, *, n_frames: int, on: int, off: int):
    base = f"output_{stem}_{trial_type}_{idx}_Hexanol_20260828_10{idx:02d}00"
    (batch_dir / f"{base}.txt").write_text(
        f"Cycle Name: {trial_type}_{idx}_Hexanol\n"
        f"Total Frames Captured: {n_frames}\n"
        "Achieved Frame Rate: 40.00 FPS\n",
        encoding="utf-8",
    )
    frames = np.arange(1, n_frames + 1)
    state = np.where((frames >= on) & (frames < off), "OFM_C", "off")
    pd.DataFrame({"FrameNumber": frames, "ActiveOFM": state}).to_csv(batch_dir / f"{base}.csv", index=False)


def _envelope(batch_dir: Path, trial_type: str, idx: int, fly: int, n: int = 50) -> Path:
    env = batch_dir / ENV
    env.mkdir(parents=True, exist_ok=True)
    path = env / f"{trial_type}_{idx}_hexanol_fly{fly}_distances_fly{fly}_angle_distance_rms_envelope.csv"
    pd.DataFrame({"envelope_of_rms": np.linspace(0, 1, n)}).to_csv(path, index=False)
    return path


def _rig_batch(tmp_path: Path, name: str = "august_28_batch_1_rig_3", stem: str = "august_28_batch_1") -> Path:
    batch = tmp_path / "Hex-Sensitivity-24-0.1" / name
    batch.mkdir(parents=True)
    for idx in (1, 2):
        (batch / f"{stem}_testing_{idx}").mkdir()
    return batch


def test_rig_batch_resolves_its_real_trial_folder(tmp_path):
    batch = _rig_batch(tmp_path)
    _sidecar(batch, "august_28_batch_1", "testing", 1, n_frames=400, on=100, off=300)
    csv = _envelope(batch, "testing", 1, fly=1)

    meta = ec._resolve_trial_meta_for_csv(csv, n_frames_hint=50)

    assert meta.trial_dir == (batch / "august_28_batch_1_testing_1").resolve()
    assert (meta.odor_on_frame, meta.odor_off_frame, meta.n_frames) == (100, 300, 400)
    assert (batch / "august_28_batch_1_testing_1" / tm.CACHE_FILENAME).is_file()
    assert not (batch / ENV / tm.CACHE_FILENAME).exists(), "shared envelope-dir cache written"


def test_plain_batch_still_uses_its_inferred_trial_folder(tmp_path):
    batch = _rig_batch(tmp_path, name="august_28_batch_1")
    _sidecar(batch, "august_28_batch_1", "testing", 2, n_frames=400, on=120, off=320)
    csv = _envelope(batch, "testing", 2, fly=2)

    meta = ec._resolve_trial_meta_for_csv(csv, n_frames_hint=50)

    assert meta.trial_dir == (batch / "august_28_batch_1_testing_2").resolve()
    assert (meta.odor_on_frame, meta.odor_off_frame) == (120, 320)


def test_sidecar_less_trial_never_inherits_another_trials_metadata(tmp_path):
    batch = _rig_batch(tmp_path)
    # Trial 1 has a sidecar with an unusual odor window; trial 2 has none.
    _sidecar(batch, "august_28_batch_1", "testing", 1, n_frames=400, on=100, off=300)
    first = ec._resolve_trial_meta_for_csv(_envelope(batch, "testing", 1, fly=1), n_frames_hint=50)
    assert first.odor_on_frame == 100

    second = ec._resolve_trial_meta_for_csv(_envelope(batch, "testing", 2, fly=1, n=3600), n_frames_hint=3600)

    # Its own fallback (30-60 s at 40 fps over its own 3600 frames), not trial 1's.
    assert (second.odor_on_frame, second.odor_off_frame, second.n_frames) == (1200, 2400, 3600)
    assert second.sidecar_txt is None


@pytest.mark.parametrize("layout", ["two-candidates", "no-folder"])
def test_without_one_trial_folder_the_synthetic_path_is_kept(tmp_path, layout):
    """No guessing between candidates, and no change where no folder exists
    (sidecar-less fixtures such as tests/golden/v2 depend on that path)."""
    batch = _rig_batch(tmp_path)
    if layout == "two-candidates":
        (batch / "august_28_batch_9_testing_1").mkdir()
    else:
        (batch / "august_28_batch_1_testing_1").rmdir()
    _sidecar(batch, "august_28_batch_1", "testing", 1, n_frames=400, on=100, off=300)

    meta = ec._resolve_trial_meta_for_csv(_envelope(batch, "testing", 1, fly=1), n_frames_hint=50)

    assert meta.trial_dir == (batch / ENV).resolve()
    assert (meta.odor_on_frame, meta.odor_off_frame) == (100, 300)


def test_memo_loads_each_trial_once_per_hint(tmp_path, monkeypatch):
    batch = _rig_batch(tmp_path)
    _sidecar(batch, "august_28_batch_1", "testing", 1, n_frames=400, on=100, off=300)
    calls = []
    real = ec.load_trial_metadata
    monkeypatch.setattr(ec, "load_trial_metadata", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    fly1 = _envelope(batch, "testing", 1, fly=1)
    fly2 = _envelope(batch, "testing", 1, fly=2)

    memo: dict = {}
    a = ec._resolve_trial_meta_for_csv(fly1, n_frames_hint=50, memo=memo)
    b = ec._resolve_trial_meta_for_csv(fly2, n_frames_hint=50, memo=memo)
    assert a == b and len(calls) == 1
    ec._resolve_trial_meta_for_csv(fly2, n_frames_hint=None, memo=memo)
    assert len(calls) == 2  # a different hint is a different call
    ec._resolve_trial_meta_for_csv(fly1, n_frames_hint=50)
    assert len(calls) == 3  # no memo -> no memoisation


def test_build_wide_csv_memo_does_not_outlive_the_build(tmp_path):
    """A sidecar edited between two builds must reach the second build."""
    batch = _rig_batch(tmp_path)
    _sidecar(batch, "august_28_batch_1", "testing", 1, n_frames=400, on=100, off=300)
    _envelope(batch, "testing", 1, fly=1, n=400)
    root = batch.parent

    out1 = tmp_path / "w1.csv"
    ec.build_wide_csv([str(root)], str(out1), measure_cols=["envelope_of_rms"])
    old = {p.name: p.stat().st_mtime for p in batch.glob("output_*")}
    for p in batch.glob("output_*"):
        p.unlink()
    _sidecar(batch, "august_28_batch_1", "testing", 1, n_frames=400, on=160, off=360)
    for p in batch.glob("output_*"):
        os.utime(p, (old[p.name] + 10, old[p.name] + 10))  # a real, later edit
    out2 = tmp_path / "w2.csv"
    ec.build_wide_csv([str(root)], str(out2), measure_cols=["envelope_of_rms"])

    on1 = pd.read_csv(out1)["trial_odor_on_s"].iloc[0]
    on2 = pd.read_csv(out2)["trial_odor_on_s"].iloc[0]
    assert (on1, on2) == (100 / 40.0, 160 / 40.0)
