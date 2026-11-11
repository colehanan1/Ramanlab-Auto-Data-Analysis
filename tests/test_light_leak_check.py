"""Light-leak QC: the LED must not be visible in trials (or parts of trials)
where the sensors log never commanded it.

``check_light_stimulus`` only confirms the light was ON inside its commanded
window; it skips every trial with no light event. This QC is the mirror image:
it looks for light where the sensors log says there should be none.

Measured on real videos (2026-11-10) the light moves the frame redness
``R - (G+B)/2`` in a rig-dependent *direction*: Pi 1 (``_rig_2``) +7.6,
Pi 2 (no suffix, grayscale) about +8, Pi 3 (``_rig_3``) -35 to -54 (the frame
darkens). No-light trials stay within about +/-2. The detector is therefore
sign-agnostic: any sustained deviation from the trial baseline counts.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from fbpipe.utils.light_stimulus import (
    FrameSample,
    check_trial_light_leak,
    detect_unexpected_light,
    flag_baseline_shifts,
)
from fbpipe.utils.trial_metadata import parse_light_span_seconds

FPS = 20.0


def _samples(levels: list[tuple[float, float, float]], duration: float = 60.0,
             base: float = 50.0, noise: float = 0.3, seed: int = 0) -> list[FrameSample]:
    """Redness trace at ``base`` with ``(start, end, delta)`` steps added."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / FPS)
    v = base + rng.normal(0.0, noise, t.size)
    for start, end, delta in levels:
        v[(t >= start) & (t < end)] += delta
    return [FrameSample(t=float(a), redness=float(b)) for a, b in zip(t, v)]


# ── detector ──────────────────────────────────────────────────────────────


def test_clean_no_light_trial_is_not_flagged():
    res = detect_unexpected_light(_samples([]), allowed_windows=[])
    assert res["leaked"] is False
    assert res["leak_intervals"] == []
    assert abs(res["baseline"] - 50.0) < 0.5


@pytest.mark.parametrize("delta", [+8.0, -30.0], ids=["pi1_pi2_brighter", "pi3_darker"])
def test_light_in_a_no_light_trial_is_flagged_either_direction(delta):
    res = detect_unexpected_light(_samples([(30.0, 36.0, delta)]), allowed_windows=[])
    assert res["leaked"] is True
    (start, end, peak), = res["leak_intervals"]
    assert 29.5 <= start <= 30.5 and 35.5 <= end <= 36.5
    assert np.sign(peak) == np.sign(delta)


def test_commanded_window_is_allowed():
    res = detect_unexpected_light(_samples([(30.0, 36.0, -30.0)]), allowed_windows=[(30.0, 36.0)])
    assert res["leaked"] is False


def test_light_outside_its_commanded_window_is_flagged():
    """Light commanded 30-36 s but still on until 50 s: the tail is a leak."""
    res = detect_unexpected_light(_samples([(30.0, 50.0, 8.0)]), allowed_windows=[(30.0, 36.0)])
    assert res["leaked"] is True
    (start, end, _), = res["leak_intervals"]
    assert start >= 36.0 and 49.5 <= end <= 50.5


def test_brief_blip_shorter_than_min_duration_is_ignored():
    """A one-frame glitch (fly leg, compression burst) is not a light."""
    res = detect_unexpected_light(_samples([(20.0, 20.1, 15.0)]), allowed_windows=[])
    assert res["leaked"] is False


def test_small_sustained_drift_below_threshold_is_ignored():
    """No-light trials wander up to ~2 redness units; that is not a light."""
    res = detect_unexpected_light(_samples([(10.0, 40.0, 2.0)]), allowed_windows=[])
    assert res["leaked"] is False


def test_camera_warmup_at_video_start_is_ignored():
    res = detect_unexpected_light(_samples([(0.0, 0.8, -20.0)]), allowed_windows=[])
    assert res["leaked"] is False


# ── batch-level stuck-on check ────────────────────────────────────────────


def test_trial_lit_for_its_whole_length_is_caught_by_batch_baseline():
    """A light stuck on for an entire trial has no step inside the trial, so
    only a comparison with the batch's other trials can see it."""
    rows = [{"trial_dir": f"t{i}", "status": "checked", "baseline_redness": 55.8 + 0.1 * i,
             "leaked": False} for i in range(8)]
    rows.append({"trial_dir": "stuck", "status": "checked", "baseline_redness": 22.0, "leaked": False})
    flag_baseline_shifts(rows, min_dev=4.0)
    by = {r["trial_dir"]: r for r in rows}
    assert by["stuck"]["leaked"] is True
    assert by["stuck"]["baseline_shift"] < -30
    assert all(by[f"t{i}"]["leaked"] is False for i in range(8))


def test_baseline_shift_skips_unchecked_rows_and_small_batches():
    rows = [{"trial_dir": "a", "status": "no_video"},
            {"trial_dir": "b", "status": "checked", "baseline_redness": 50.0, "leaked": False}]
    flag_baseline_shifts(rows, min_dev=4.0)
    assert "baseline_shift" not in rows[0]
    assert rows[1]["leaked"] is False


# ── sensors-log span (pulse trains toggle many times) ─────────────────────


def _sensors(tmp_path: Path, events: list[tuple[float, str]]) -> Path:
    start = 1_000_000_000_000
    rows = [{"MonoNs": start, "OFM_Event": "RECORDING_START"}]
    rows += [{"MonoNs": start + int(t * 1e9), "OFM_Event": e} for t, e in events]
    p = tmp_path / "sensors_output_x_testing_1_20260101_000000.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_light_span_covers_a_whole_pulse_train(tmp_path):
    ev = [(30.0, "LIGHT_PULSE_ON"), (30.1, "LIGHT_PULSE_OFF"),
          (30.5, "LIGHT_PULSE_ON"), (30.6, "LIGHT_PULSE_OFF"),
          (35.0, "LIGHT_PULSE_ON"), (35.1, "LIGHT_OFF")]
    assert parse_light_span_seconds(_sensors(tmp_path, ev)) == pytest.approx((30.0, 35.1))


def test_light_span_none_when_no_light_events(tmp_path):
    assert parse_light_span_seconds(_sensors(tmp_path, [(10.0, "OFM_A_ON"), (40.0, "OFM_A_OFF")])) == (None, None)


def test_light_span_on_without_off_runs_open_ended(tmp_path):
    on, off = parse_light_span_seconds(_sensors(tmp_path, [(30.0, "LIGHT_SOLID_ON")]))
    assert on == pytest.approx(30.0) and off is None


# ── per-trial wrapper on a real (tiny) video ──────────────────────────────


def _write_video(path: Path, lit: tuple[float, float] | None, seconds: float = 6.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 64))
    for i in range(int(seconds * FPS)):
        t = i / FPS
        frame = np.full((64, 64, 3), 60, dtype=np.uint8)
        if lit and lit[0] <= t < lit[1]:
            frame[..., 2] = 140  # red up -> redness +80
        w.write(frame)
    w.release()


def test_check_trial_light_leak_flags_and_caches(tmp_path):
    trial = tmp_path / "batch" / "x_testing_1"
    trial.mkdir(parents=True)
    _write_video(trial / f"{trial.name}_distance_annotated.mp4", lit=(3.0, 4.5))

    row = check_trial_light_leak(trial, allowed_windows=[], stride=1,
                                 extra_columns={"dataset": "D"})
    assert row["status"] == "checked"
    assert row["leaked"] is True and row["passed"] is False
    assert row["dataset"] == "D"
    assert (trial / "_light_leak_check.json").exists()

    again = check_trial_light_leak(trial, allowed_windows=[], stride=1)
    assert again["from_cache"] is True and again["leaked"] is True

    # Same video, but the light was commanded there: the cache key changes.
    allowed = check_trial_light_leak(trial, allowed_windows=[(3.0, 4.5)], stride=1)
    assert allowed["from_cache"] is False and allowed["leaked"] is False


def test_check_trial_light_leak_without_video(tmp_path):
    trial = tmp_path / "batch" / "x_testing_2"
    trial.mkdir(parents=True)
    row = check_trial_light_leak(trial, allowed_windows=[])
    assert row["status"] == "no_video" and row["passed"] is None


# ── pipeline step ─────────────────────────────────────────────────────────

from types import SimpleNamespace

from fbpipe.steps import check_light_leak as step

BATCH = "november_10_batch_1_rig_3"


def _trial_with(root: Path, phase_idx: str, *, lit, light_events=None, sensors=True) -> Path:
    batch = root / "IAA-Test-24-1" / BATCH
    trial = batch / f"{BATCH}_{phase_idx}"
    trial.mkdir(parents=True, exist_ok=True)
    _write_video(trial / f"{trial.name}_distance_annotated.mp4", lit=lit)
    if sensors:
        start = 1_000_000_000_000
        rows = [{"MonoNs": start, "OFM_Event": "RECORDING_START"}]
        rows += [{"MonoNs": start + int(t * 1e9), "OFM_Event": e} for t, e in (light_events or [])]
        pd.DataFrame(rows).to_csv(batch / f"sensors_output_{BATCH}_{phase_idx}_20261110_120000.csv", index=False)
    return trial


def _cfg(root: Path):
    return SimpleNamespace(force=SimpleNamespace(pipeline=False), datasets=(),
                           parallel=SimpleNamespace(enabled=False, n_jobs=1), main_directory=str(root))


def test_step_flags_light_in_a_no_light_trial_but_never_a_trial_without_a_log(tmp_path, monkeypatch):
    monkeypatch.setattr(step, "DEFAULT_PLOTS_DIR", tmp_path / "plots")
    root = tmp_path / "data"
    _trial_with(root, "testing_1", lit=None)                                  # clean
    _trial_with(root, "testing_2", lit=(3.0, 4.5))                            # LEAK
    _trial_with(root, "training_1", lit=(3.0, 4.5),
                light_events=[(3.0, "LIGHT_SOLID_ON"), (4.5, "LIGHT_OFF")])   # commanded, fine
    _trial_with(root, "testing_3", lit=(3.0, 4.5), sensors=False)             # no log: unknown

    rows = step._process_fly_dir(root / "IAA-Test-24-1" / BATCH, _cfg(root))
    by = {Path(r["trial_dir"]).name.split(BATCH + "_")[1]: r for r in rows}

    assert by["testing_1"]["leaked"] is False
    assert by["testing_2"]["leaked"] is True and by["testing_2"]["light_commanded"] is False
    assert by["training_1"]["leaked"] is False and by["training_1"]["light_commanded"] is True
    assert by["testing_3"]["status"] == "no_sensors_log" and by["testing_3"]["passed"] is None
    assert (tmp_path / "plots" / f"{BATCH}_testing_2_leak_timeseries.png").exists()


def test_step_main_notifies_once_per_new_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(step, "DEFAULT_PLOTS_DIR", tmp_path / "plots")
    root = tmp_path / "data"
    _trial_with(root, "testing_1", lit=None)
    _trial_with(root, "testing_2", lit=(3.0, 4.5))
    csv = tmp_path / "leak.csv"
    monkeypatch.setenv("LIGHT_LEAK_CSV", str(csv))
    monkeypatch.setattr(step, "get_main_directories", lambda cfg: [root / "IAA-Test-24-1"])
    monkeypatch.setattr(step, "iter_live_batch_dirs", lambda cfg, r: [r / BATCH])
    sent = []
    monkeypatch.setattr(step, "ntfy_notify", lambda title, body, **kw: sent.append((title, body)))

    step.main(_cfg(root))
    assert len(sent) == 1 and "1 trial(s)" in sent[0][0] and "testing_2" in sent[0][1]
    df = pd.read_csv(csv)
    assert set(df["passed"]) == {True, False}

    step.main(_cfg(root))  # unchanged videos: no repeat alert
    assert len(sent) == 1


def test_step_is_registered_after_the_report_only_qcs():
    from fbpipe import pipeline
    names = [s.name for s in pipeline.ORDERED_STEPS]
    assert names.index("check_light_leak") == names.index("trial_timing_qc") + 1
    assert names[names.index("check_light_stimulus") + 1] == "eye_drift_qc"
