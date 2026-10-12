"""eye_drift_qc: flag flies whose eye drifts off its frozen anchor.

yolo_infer records, per frame, where each slot's eye is detected
(``eye_{i}_live_x/_y``) and how far that sits from the frozen anchor
(``eye_{i}_drift_px``). This step turns that into one report row per trial x
fly. A drift whose trailing median over ``eye_qc.sustain_s`` exceeds
``eye_qc.max_drift_px`` (a fly that came loose, a shifted mount) is flagged, and
so is an eye that goes unmatched too often or for too long in a row (one that
jumped past ``match_radius_px``). It is report-only: a CSV, a log line and one
phone ping.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fbpipe.config as fbpipe_config
from fbpipe.config import (
    DatasetOverride,
    EyeQCSettings,
    ParallelSettings,
    Settings,
    load_flagged_fly_exclusions,
)
from fbpipe.steps import eye_drift_qc as qc

FPS = 40.0
N_FRAMES = 3600
# Four flies stacked ~290 px apart (Flybehavior2 framing), slot 0 on top.
ANCHORS = ((520.0, 140.0), (515.0, 430.0), (510.0, 735.0), (505.0, 1010.0))
DATASET = "E2E-EyeQC"
BATCH = "october_07_batch_1_rig_2"
TRIAL_PREFIX = "october_07_batch_1"  # trial folders drop the rig suffix, as on disk


# ---------------------------------------------------------------------------
# Real-shaped merged tables
# ---------------------------------------------------------------------------


def _timestamps(n: int, fps: float = FPS) -> np.ndarray:
    # The rig stamps frames 0 and 1 alike, then one frame every 1/fps.
    return np.maximum(np.arange(n) - 1, 0) / fps


def _jitter(n: int = N_FRAMES) -> np.ndarray:
    """Detection jitter of a good trial: 0-0.5 px, deterministic."""
    return 0.25 + 0.25 * np.sin(np.arange(n) * 0.7)


def _loose(n: int = N_FRAMES, at: int = 1200, shift: float = 8.0) -> np.ndarray:
    """The eye moves ``shift`` px off its anchor at frame ``at`` and stays there."""
    drift = _jitter(n)
    drift[at:] += shift
    return drift


def _merged_table(drifts, *, max_flies=4, timestamps=None, track_ids=None, live=True, n=None):
    """A ``<trial>_distances_merged`` table with the column groups yolo_infer
    writes for every slot in range(max_flies). Slot ``i`` has an eye anchor iff
    ``i < len(drifts)``; ``drifts[i]`` is its per-frame drift (NaN = unmatched),
    laid along +x from the anchor. ``live=False`` gives a table from before
    per-frame eye tracking."""
    if n is None:
        n = len(drifts[0]) if len(drifts) else N_FRAMES
    data = {
        "frame": np.arange(n),
        "timestamp": _timestamps(n) if timestamps is None else np.asarray(timestamps, float),
        "x_anchor": np.full(n, 1080.0),
        "y_anchor": np.full(n, 540.0),
    }
    for i in range(max_flies):
        if i < len(drifts):
            drift = np.asarray(drifts[i], dtype=float)
            ids = None if track_ids is None else track_ids[i]
            tid = np.full(n, 1001.0 + i) if ids is None else np.asarray(ids, float)
            ex, ey = np.full(n, ANCHORS[i][0]), np.full(n, ANCHORS[i][1])
        else:
            drift = tid = ex = ey = np.full(n, np.nan)
        data[f"eye_{i}_track_id"] = tid
        data[f"eye_{i}_x"] = ex
        data[f"eye_{i}_y"] = ey
        if live:
            data[f"eye_{i}_live_x"] = ex + drift
            data[f"eye_{i}_live_y"] = ey + np.where(np.isnan(drift), np.nan, 0.0)
            data[f"eye_{i}_drift_px"] = drift
        data[f"dist_eye_{i}_anchor"] = np.hypot(ex - 1080.0, ey - 540.0)
        data[f"cls8_{i}_track_id"] = np.full(n, np.nan)
        data[f"cls8_{i}_x"] = np.full(n, np.nan)
        data[f"cls8_{i}_y"] = np.full(n, np.nan)
        data[f"dist_eye_{i}_cls8_{i}"] = np.full(n, np.nan)
        data[f"angle_eye_{i}_cls8_vs_anchor"] = np.full(n, np.nan)
    return pd.DataFrame(data)


def _assess(drifts, *, eq=None, fps_default=FPS, **table_kw):
    return qc.assess_trial(_merged_table(drifts, **table_kw), eq or EyeQCSettings(), fps_default)


# ---------------------------------------------------------------------------
# assess_trial
# ---------------------------------------------------------------------------


def test_steady_sub_pixel_jitter_is_not_flagged():
    (row,) = _assess([_jitter()])

    assert row["status"] == "assessed"
    assert row["eye_qc_flag"] is False
    assert row["eye_qc_reason"] == ""
    assert (row["fly_number"], row["slot_idx"], row["eye_track_id"]) == (1, 0, 1001)
    assert row["n_frames"] == N_FRAMES
    assert (row["eye_detected_frac"], row["eye_missing_frac"]) == (1.0, 0.0)
    assert row["longest_missing_run_s"] == 0.0
    assert math.isnan(row["longest_missing_run_start_s"])  # no run, so no start
    assert row["drift_max_px"] <= 0.5
    assert row["drift_median_px"] == pytest.approx(np.median(_jitter()), abs=1e-3)
    assert row["drift_sustained_max_px"] <= 0.5
    assert row["drift_frac_over_thresh"] == 0.0
    assert row["longest_over_thresh_run_s"] == 0.0
    assert math.isnan(row["first_over_thresh_s"])
    assert (row["anchor_x"], row["anchor_y"]) == ANCHORS[0]


def test_a_shift_that_persists_is_flagged_as_drift_from_when_it_starts():
    good, loose = _assess([_jitter(), _loose(at=1200, shift=8.0)])

    assert good["eye_qc_flag"] is False
    assert loose["eye_qc_flag"] is True
    assert loose["fly_number"] == 2
    # Frame 1200 is 30 s in; the 0.5 s rolling median crosses within ~0.25 s.
    assert loose["first_over_thresh_s"] == pytest.approx(30.0, abs=0.5)
    assert 8.0 <= loose["drift_sustained_max_px"] <= 8.5
    assert loose["longest_over_thresh_run_s"] == pytest.approx(2400 / FPS)
    assert loose["drift_frac_over_thresh"] == pytest.approx(2400 / N_FRAMES, abs=1e-4)
    assert loose["eye_qc_reason"].startswith("drift > 5.0 px (median over 0.5 s) from 30.")
    assert loose["eye_qc_reason"].endswith(f"(max {loose['drift_sustained_max_px']:.1f} px)")


@pytest.mark.parametrize("frames, flagged", [(12, True), (8, False)], ids=["0.3s", "0.2s"])
def test_the_drift_rule_is_a_median_over_sustain_s_not_a_minimum_duration(frames, flagged):
    """A 20 px excursion over just over half of the 20-frame (0.5 s) window
    already lifts the trailing median past 5 px. The reason therefore says
    "median over 0.5 s", never that the drift was sustained for 0.5 s."""
    drift = _jitter()
    drift[2000:2000 + frames] = 20.0
    (row,) = _assess([drift])

    assert row["eye_qc_flag"] is flagged
    if flagged:
        # The 10th excursion frame (2009) is the first to hold half the window.
        assert row["eye_qc_reason"] == "drift > 5.0 px (median over 0.5 s) from 50.2 s (max 20.0 px)"
    else:
        assert row["drift_sustained_max_px"] < 5.0


def test_a_few_frame_spike_is_not_flagged():
    drift = _jitter()
    drift[2000:2005] = 20.0
    (row,) = _assess([drift])

    assert row["eye_qc_flag"] is False
    assert row["drift_max_px"] == 20.0
    assert row["longest_over_thresh_run_s"] == pytest.approx(5 / FPS)
    assert row["drift_sustained_max_px"] < 5.0


def test_a_few_frame_spike_at_the_trial_start_is_not_flagged_either():
    # Frames 1-5 are half of a 10-frame partial window; a median over that
    # alone would read the spike as sustained. The window must span sustain_s.
    drift = _jitter()
    drift[1:6] = 20.0
    (row,) = _assess([drift])
    assert row["eye_qc_flag"] is False
    assert row["drift_max_px"] == 20.0
    assert row["drift_sustained_max_px"] < 5.0


@pytest.mark.parametrize("lost, spike", [(10, 5), (2, 9)], ids=["10-lost-then-5", "2-lost-then-9"])
def test_a_few_frame_spike_right_after_a_dropout_is_not_flagged(lost, spike):
    """Unmatched frames count as not over the threshold. Skipped instead, a
    window half lost to grooming or occlusion would let a 5-frame spike make up
    half of what is left of it -- and read as a shift."""
    drift = _jitter()
    drift[2000:2000 + lost] = np.nan
    drift[2000 + lost:2000 + lost + spike] = 20.0
    (row,) = _assess([drift])

    assert row["eye_qc_flag"] is False
    assert row["drift_sustained_max_px"] < 5.0


def test_a_loose_fly_whose_eye_drops_out_now_and_then_is_still_flagged():
    """A fly 8 px loose from 30 s whose eye goes unmatched in 1 of every 15
    frames (6.7 %, under max_missing_frac): the windows holding a dropout must
    still give a median, or the shift is never seen."""
    drift = _loose(at=1200)
    drift[np.arange(N_FRAMES) % 15 == 7] = np.nan
    (row,) = _assess([drift])

    assert row["eye_missing_frac"] == pytest.approx(240 / N_FRAMES, abs=1e-4)
    assert row["eye_qc_flag"] is True
    assert row["first_over_thresh_s"] == pytest.approx(30.2, abs=0.1)
    assert row["eye_qc_reason"].startswith("drift > 5.0 px (median over 0.5 s) from 30.2 s")


def test_a_lone_detection_inside_a_dropout_is_not_drift():
    """At least half the window must hold a detection: one 20 px detection in
    the middle of a 1 s dropout is no median at all, let alone a shift."""
    drift = _jitter()
    drift[2000:2040] = np.nan
    drift[2020] = 20.0
    (row,) = _assess([drift])

    assert row["eye_qc_flag"] is False
    assert row["drift_sustained_max_px"] < 5.0


def test_an_eye_unmatched_in_30_percent_of_frames_is_flagged_eye_missing():
    drift = _jitter()
    drift[1000:2080] = np.nan  # 1080 of 3600 frames, in one run
    (row,) = _assess([drift])

    assert row["eye_qc_flag"] is True
    assert row["eye_missing_frac"] == pytest.approx(0.3)
    assert row["eye_detected_frac"] == pytest.approx(0.7)
    assert row["longest_missing_run_s"] == pytest.approx(1080 / FPS)
    assert row["longest_missing_run_start_s"] == pytest.approx(999 / FPS)
    # Both missing rules trip; one reason gives the fraction and the run.
    assert row["eye_qc_reason"] == (
        "eye unmatched in 30% of frames, 27.0 s in a row from 25.0 s (moved > 50 px?)"
    )


def test_scattered_dropouts_still_flag_on_the_missing_fraction_alone():
    """The fraction rule is unchanged: 30 % of frames lost three at a time never
    makes a long run, and still flags with the reason it always gave."""
    drift = _jitter()
    drift[np.arange(N_FRAMES) % 10 < 3] = np.nan
    (row,) = _assess([drift])

    assert row["eye_missing_frac"] == pytest.approx(0.3)
    assert row["longest_missing_run_s"] == pytest.approx(3 / FPS)
    assert row["eye_qc_flag"] is True
    assert row["eye_qc_reason"] == "eye unmatched in 30% of frames"


def test_a_slot_whose_eye_never_matches_reads_as_a_phantom():
    good, phantom = _assess([_jitter(), np.full(N_FRAMES, np.nan)])

    assert good["eye_qc_flag"] is False
    assert phantom["eye_qc_flag"] is True
    assert phantom["eye_missing_frac"] == 1.0
    assert phantom["eye_qc_reason"] == "eye never matched (phantom slot?)"
    assert math.isnan(phantom["drift_median_px"])
    assert math.isnan(phantom["drift_sustained_max_px"])


def test_a_phantom_slot_that_only_matched_the_frame_it_was_made_from_is_still_a_phantom():
    """yolo_infer builds an anchor from one frame's eye detection and matches
    that same frame against it, so a phantom anchor always matches itself once,
    at 0 px, on the frame it was made from -- then never again."""
    drift = np.full(N_FRAMES, np.nan)
    drift[0] = 0.0
    good, phantom = _assess([_jitter(), drift])

    assert good["eye_qc_flag"] is False
    assert phantom["eye_qc_flag"] is True
    assert phantom["eye_qc_reason"] == "eye never matched (phantom slot?)"


def test_drift_and_missing_together_give_both_reasons():
    drift = _loose(at=1200)
    drift[:1200] = np.nan  # 1/3 of the trial unmatched, then shifted for good
    (row,) = _assess([drift])

    assert row["eye_qc_flag"] is True
    reason = row["eye_qc_reason"]
    assert reason.startswith("drift > 5.0 px (median over 0.5 s) from 30.")
    assert reason.endswith("; eye unmatched in 33% of frames, 30.0 s in a row from 0.0 s (moved > 50 px?)")


def test_missing_frames_below_the_threshold_do_not_flag():
    drift = _jitter()
    drift[::10] = np.nan  # 10 % of frames
    (row,) = _assess([drift])
    assert row["eye_qc_flag"] is False
    assert row["eye_missing_frac"] == pytest.approx(0.1)
    assert row["longest_missing_run_s"] == pytest.approx(1 / FPS)


# A fly that comes loose LATE with a big jump: its eye lands beyond
# match_radius_px, so it reads as unmatched rather than as drift, and late in a
# trial that is too small a fraction of the frames for max_missing_frac.


def _gap(n: int, start: int, stop: int) -> np.ndarray:
    """Good jitter, with the eye unmatched from frame ``start`` up to ``stop``."""
    drift = _jitter(n)
    drift[start:stop] = np.nan
    return drift


@pytest.mark.parametrize(
    "n, start, stop, from_s",
    [
        (1200, 1080, 1200, "27.0"),      # the last 3 s of a 30 s trial: lost for good at 90 %
        (N_FRAMES, 3240, 3360, "81.0"),  # 3 s at 90 % of a 90 s trial, then found again
    ],
    ids=["tail", "gap"],
)
def test_an_eye_lost_for_3_s_late_in_a_trial_is_flagged_eye_missing(n, start, stop, from_s):
    (row,) = _assess([_gap(n, start, stop)])

    assert row["eye_missing_frac"] < EyeQCSettings().max_missing_frac  # 10 % / 3 %: the fraction misses it
    assert row["eye_qc_flag"] is True
    assert row["longest_missing_run_s"] == pytest.approx(3.0)
    # Timed like first_over_thresh_s: from the first assessed frame's timestamp.
    assert row["longest_missing_run_start_s"] == pytest.approx((start - 1) / FPS)
    assert row["eye_qc_reason"] == f"eye unmatched for 3.0 s in a row from {from_s} s (moved > 50 px?)"
    assert row["drift_sustained_max_px"] < 5.0  # never read as drift


def test_a_one_second_gap_is_not_flagged():
    (row,) = _assess([_gap(N_FRAMES, 2000, 2040)])
    assert row["eye_qc_flag"] is False
    assert row["eye_qc_reason"] == ""
    assert row["longest_missing_run_s"] == pytest.approx(1.0)
    assert row["longest_missing_run_start_s"] == pytest.approx(1999 / FPS)


def test_of_two_equally_long_missing_runs_the_earlier_is_reported():
    drift = _jitter()
    drift[1000:1100] = np.nan
    drift[2500:2600] = np.nan
    (row,) = _assess([drift])

    assert row["longest_missing_run_start_s"] == pytest.approx(999 / FPS)
    assert row["eye_qc_reason"] == "eye unmatched for 2.5 s in a row from 25.0 s (moved > 50 px?)"


@pytest.mark.parametrize("frames, flagged", [(80, False), (81, True)], ids=["2.000s", "2.025s"])
def test_a_missing_run_must_last_longer_than_max_missing_run_s(frames, flagged):
    (row,) = _assess([_gap(N_FRAMES, 2000, 2000 + frames)])  # the default limit is 2.0 s
    assert row["eye_qc_flag"] is flagged


def test_the_run_limit_and_the_radius_in_the_hint_come_from_the_settings():
    eq = EyeQCSettings(match_radius_px=35.0, max_missing_run_s=0.75)
    (row,) = _assess([_gap(N_FRAMES, 2000, 2040)], eq=eq)  # the 1 s gap above
    assert row["eye_qc_flag"] is True
    assert row["eye_qc_reason"] == "eye unmatched for 1.0 s in a row from 50.0 s (moved > 35 px?)"
    assert row["max_missing_run_s"] == 0.75


def test_the_longest_run_is_timed_from_the_first_assessed_frame():
    # The slot's final anchor only took over at frame 400: a run from frame
    # 2000 starts 40 s into the frames assessed, with or without a clock.
    tid = np.full(N_FRAMES, 1001.0)
    tid[:400] = 1003.0
    for timestamps in (None, np.full(N_FRAMES, np.nan)):
        (row,) = _assess([_gap(N_FRAMES, 2000, 2100)], track_ids=[tid], timestamps=timestamps)
        assert row["longest_missing_run_start_s"] == pytest.approx(1600 / FPS)
        assert row["eye_qc_reason"] == "eye unmatched for 2.5 s in a row from 40.0 s (moved > 50 px?)"


def test_frames_under_another_track_id_are_ignored():
    # Before confirmation this slot held another partial anchor (slots are
    # re-sorted top-to-bottom at confirmation), and before that none at all.
    drift = _jitter()
    drift[:40] = 30.0
    tid = np.full(N_FRAMES, 1001.0)
    tid[:5] = np.nan
    tid[5:40] = 1003.0
    (row,) = _assess([drift], track_ids=[tid])

    assert row["eye_track_id"] == 1001
    assert row["n_frames"] == N_FRAMES - 40
    assert row["drift_max_px"] <= 0.5
    assert row["eye_qc_flag"] is False


def test_only_slots_with_an_eye_anchor_get_a_row():
    rows = _assess([_jitter(), _jitter()], max_flies=5)
    assert [r["fly_number"] for r in rows] == [1, 2]
    assert _assess([], max_flies=4) == []


def test_fps_comes_from_the_timestamps():
    # A 30 fps recording: a 30-frame dropout lasts exactly 1 s.
    n = 1800
    drift = _jitter(n)
    drift[600:630] = np.nan
    (row,) = _assess([drift], timestamps=np.arange(n) / 30.0, fps_default=50.0)
    assert row["longest_missing_run_s"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "timestamps",
    [
        np.full(1800, np.nan),        # no clock at all
        np.zeros(1800),               # a frozen clock
        np.arange(1800) * 25.0,       # milliseconds written as seconds -> 0.04 fps
    ],
    ids=["missing", "frozen", "ms_scaled"],
)
def test_unusable_timestamps_fall_back_to_fps_default(timestamps):
    drift = _jitter(1800)
    drift[600:630] = np.nan
    (row,) = _assess([drift], timestamps=timestamps, fps_default=50.0)
    assert row["longest_missing_run_s"] == pytest.approx(30 / 50.0)


def test_first_over_thresh_counts_frames_when_there_are_no_timestamps():
    (with_clock,) = _assess([_loose(at=1200)])
    (no_clock,) = _assess([_loose(at=1200)], timestamps=np.full(N_FRAMES, np.nan))
    # The crossing frame is the same; only how it is turned into seconds differs.
    assert no_clock["first_over_thresh_s"] == pytest.approx(with_clock["first_over_thresh_s"] + 1 / FPS)


def test_first_over_thresh_is_timed_from_the_first_assessed_frame():
    # The slot's final anchor only took over at frame 400 (9.975 s); the crossing
    # at frame 1210 (30.225 s) is reported relative to that.
    tid = np.full(N_FRAMES, 1001.0)
    tid[:400] = 1003.0
    (row,) = _assess([_loose(at=1200)], track_ids=[tid])
    assert row["n_frames"] == N_FRAMES - 400
    assert row["first_over_thresh_s"] == pytest.approx(1209 / FPS - 399 / FPS)


def test_the_rows_carry_the_thresholds_they_were_judged_by():
    eq = EyeQCSettings(
        max_drift_px=3.0, sustain_s=0.25, match_radius_px=40.0, max_missing_frac=0.35, max_missing_run_s=1.5
    )
    (row,) = _assess([_jitter()], eq=eq)
    assert (
        row["max_drift_px"], row["sustain_s"], row["max_missing_frac"], row["max_missing_run_s"],
        row["match_radius_px"],
    ) == (3.0, 0.25, 0.35, 1.5, 40.0)
    for column in ("longest_missing_run_start_s", "max_missing_run_s"):
        assert column in qc.REPORT_COLUMNS


def test_a_lower_threshold_catches_a_smaller_shift():
    eq = EyeQCSettings(max_drift_px=3.0)
    (row,) = _assess([_loose(shift=4.0)], eq=eq)
    assert row["eye_qc_flag"] is True
    assert row["eye_qc_reason"].startswith("drift > 3.0 px")
    (default_row,) = _assess([_loose(shift=4.0)])
    assert default_row["eye_qc_flag"] is False


# ---------------------------------------------------------------------------
# Threshold sanitising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,bad",
    [
        ("max_drift_px", -1.0),
        ("max_drift_px", float("nan")),
        ("sustain_s", -0.5),
        ("sustain_s", float("inf")),
        ("match_radius_px", 0.0),
        ("match_radius_px", -3.0),
        ("max_missing_frac", 1.5),
        ("max_missing_frac", -0.1),
        ("max_missing_frac", float("nan")),
        ("max_missing_run_s", -1.0),
        ("max_missing_run_s", float("nan")),
        ("max_missing_run_s", float("inf")),
    ],
)
def test_an_unusable_threshold_falls_back_to_the_dataclass_default(field, bad, capsys):
    eq = qc._sanitized_eye_qc(EyeQCSettings(**{field: bad}))
    assert getattr(eq, field) == getattr(EyeQCSettings(), field)
    assert f"eye_qc.{field}" in capsys.readouterr().out


def test_usable_thresholds_pass_through_silently(capsys):
    eq = EyeQCSettings(
        max_drift_px=3.5, sustain_s=0.25, match_radius_px=40.0, max_missing_frac=0.35,
        max_missing_run_s=1.75, report_dir="/x",
    )
    assert qc._sanitized_eye_qc(eq) == eq
    assert capsys.readouterr().out == ""


def test_a_zero_run_limit_is_usable(capsys):
    """0 s is a legal, strict choice -- any unmatched frame then flags."""
    assert qc._sanitized_eye_qc(EyeQCSettings(max_missing_run_s=0.0)).max_missing_run_s == 0.0
    assert capsys.readouterr().out == ""


def test_a_match_radius_inside_the_drift_threshold_is_kept_with_a_warning(capsys):
    eq = qc._sanitized_eye_qc(EyeQCSettings(max_drift_px=6.0, match_radius_px=4.0))
    assert (eq.max_drift_px, eq.match_radius_px) == (6.0, 4.0)
    out = capsys.readouterr().out
    assert "match_radius_px" in out and "missing" in out


# ---------------------------------------------------------------------------
# A tmp batch tree
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_side_effects(monkeypatch, tmp_path):
    """No test may reach the real report dir or phone."""
    monkeypatch.setattr(fbpipe_config, "DEFAULT_EYE_QC_REPORT_DIR", tmp_path / "default-report-dir")
    monkeypatch.setenv("FBPIPE_DISABLE_NTFY", "1")


@pytest.fixture
def ntfy_calls(monkeypatch):
    calls = []

    def _record(title, message, **kwargs):
        calls.append({"title": title, "message": message, **kwargs})
        return True

    monkeypatch.setattr(qc, "ntfy_notify", _record)
    return calls


def _write_merged(trial: Path, table: pd.DataFrame, suffix: str = ".parquet") -> Path:
    trial.mkdir(parents=True, exist_ok=True)
    path = trial / f"{trial.name}_distances_merged{suffix}"
    if suffix == ".parquet":
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False)
    return path


def _sidecar(batch: Path, trial: str, cycle: str, odor: str) -> None:
    (batch / f"output_{trial}_{odor}_20261007_110648.txt").write_text(
        f"Cycle Name: {cycle}_{odor}\nAchieved Frame Rate: 40.02 FPS\n", encoding="utf-8"
    )


def _tree(tmp_path: Path, *, variety: bool = False) -> Path:
    """A dataset root with one real-shaped batch.

    pretest_1  4 flies, fly 2 came loose 30 s in (merged parquet)
    pretest_2  a legacy CSV table from before per-frame eye tracking

    With ``variety``:
    pretest_3  a corrupt parquet
    testing_1  no merged table (YOLO has not processed it)
    testing_2  live columns but no eye anchor in any slot
    training_1 an empty table
    RMS_calculations/ a loose-fly copy that must never be read
    """
    root = tmp_path / "data" / DATASET
    batch = root / BATCH
    batch.mkdir(parents=True)
    (batch / f"output_{TRIAL_PREFIX}_pretest_1_Hexanol-0.1_20261007_110648.mp4").write_bytes(b"")

    t1 = f"{TRIAL_PREFIX}_pretest_1"
    _sidecar(batch, t1, "pretest_1", "Hexanol-0.1")
    _write_merged(batch / t1, _merged_table([_jitter(), _loose(), _jitter(), _jitter()]))

    t2 = batch / f"{TRIAL_PREFIX}_pretest_2"
    _write_merged(t2, _merged_table([_jitter(), _jitter()], live=False), suffix=".csv")
    # yolo_infer writes all-NaN per-fly live columns when the merged table has
    # none; they must not read as "eye never matched".
    pd.DataFrame({"frame": [0, 1], "eye_drift_px": [np.nan, np.nan]}).to_parquet(
        t2 / f"{t2.name}_fly1_distances.parquet", index=False
    )

    if variety:
        t3 = batch / f"{TRIAL_PREFIX}_pretest_3"
        t3.mkdir()
        (t3 / f"{t3.name}_distances_merged.parquet").write_bytes(b"PAR1 this is not a parquet file")
        (batch / f"{TRIAL_PREFIX}_testing_1").mkdir()
        _write_merged(batch / f"{TRIAL_PREFIX}_testing_2", _merged_table([], n=400))
        _write_merged(batch / f"{TRIAL_PREFIX}_training_1", _merged_table([_jitter()]).iloc[:0])
        decoy = batch / "RMS_calculations" / f"{TRIAL_PREFIX}_testing_5"
        _write_merged(decoy, _merged_table([_loose()]))
    return root


def _cfg(root: Path, report_dir: Path, *, eye_qc: dict | None = None, **overrides) -> Settings:
    kwargs = dict(
        model_path="model.pt",
        main_directories=[str(root)],
        datasets=(DATASET,),
        eye_qc=EyeQCSettings(report_dir=str(report_dir), **(eye_qc or {})),
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _trial(root: Path, name: str, batch: str = BATCH) -> str:
    return str(root / batch / f"{TRIAL_PREFIX}_{name}")


# ---------------------------------------------------------------------------
# _process_fly_dir
# ---------------------------------------------------------------------------


def test_process_fly_dir_reports_every_processed_trial(tmp_path):
    root = _tree(tmp_path, variety=True)
    cfg = _cfg(root, tmp_path / "qc")

    rows = qc._process_fly_dir(root / BATCH, cfg)

    got = sorted(
        (Path(r["trial_dir"]).name, r["status"], r["fly_number"] or 0) for r in rows
    )
    p = TRIAL_PREFIX
    assert got == [
        (f"{p}_pretest_1", "assessed", 1),
        (f"{p}_pretest_1", "assessed", 2),
        (f"{p}_pretest_1", "assessed", 3),
        (f"{p}_pretest_1", "assessed", 4),
        (f"{p}_pretest_2", "no_live_eye_columns", 0),
        (f"{p}_pretest_3", "read_error", 0),
        (f"{p}_testing_2", "no_eye_anchor", 0),
        (f"{p}_training_1", "empty", 0),
    ]
    for row in rows:
        assert set(row) == set(qc.REPORT_COLUMNS)
        assert row["qc_version"] == qc.QC_VERSION
        assert row["dataset"] == DATASET
        assert row["fly"] == BATCH

    flagged = [r for r in rows if r["eye_qc_flag"]]
    assert [(Path(r["trial_dir"]).name, r["fly_number"]) for r in flagged] == [(f"{p}_pretest_1", 2)]


def test_process_fly_dir_reads_trial_metadata_like_check_light_stimulus(tmp_path):
    root = _tree(tmp_path)
    rows = qc._process_fly_dir(root / BATCH, _cfg(root, tmp_path / "qc"))
    loose = next(r for r in rows if r["fly_number"] == 2)

    assert loose["trial_dir"] == _trial(root, "pretest_1")
    assert (loose["phase"], loose["trial_index"], loose["trial_type"], loose["odor"]) == (
        "pretest", 1, "pretest", "Hexanol-0.1",
    )
    legacy = next(r for r in rows if r["status"] == "no_live_eye_columns")
    assert (legacy["phase"], legacy["trial_index"]) == ("pretest", 2)
    assert legacy["fly_number"] is None and legacy["eye_qc_flag"] is False


def test_process_fly_dir_never_raises_on_a_bad_table_or_bad_metadata(tmp_path, monkeypatch):
    root = _tree(tmp_path, variety=True)

    def _broken_metadata(trial_dir, **kwargs):
        raise RuntimeError("sidecar exploded")

    monkeypatch.setattr(qc, "load_trial_metadata", _broken_metadata)
    rows = qc._process_fly_dir(root / BATCH, _cfg(root, tmp_path / "qc"))

    loose = next(r for r in rows if r["fly_number"] == 2)
    assert loose["eye_qc_flag"] is True
    # Folder-derived fallback: the dataset root, the phase and index of the name.
    assert (loose["dataset"], loose["phase"], loose["trial_index"]) == (DATASET, "pretest", 1)
    bad = next(r for r in rows if r["status"] == "read_error")
    assert "pretest_3_distances_merged.parquet" in bad["eye_qc_reason"]
    assert bad["eye_qc_flag"] is False


def test_a_csv_merged_table_with_live_columns_is_assessed_too(tmp_path):
    root = tmp_path / "data" / DATASET
    _write_merged(root / BATCH / f"{TRIAL_PREFIX}_testing_3", _merged_table([_jitter(), _loose()]), suffix=".csv")
    rows = qc._process_fly_dir(root / BATCH, _cfg(root, tmp_path / "qc"))
    assert [(r["status"], r["fly_number"], r["eye_qc_flag"]) for r in rows] == [
        ("assessed", 1, False),
        ("assessed", 2, True),
    ]


def test_an_unlistable_batch_folder_gives_no_rows_instead_of_raising(tmp_path, capsys):
    root = tmp_path / "data" / DATASET
    assert qc._process_fly_dir(root / "gone_batch", _cfg(root, tmp_path / "qc")) == []
    assert "cannot list" in capsys.readouterr().out


def test_the_report_never_reads_as_a_fly_exclusion_list(tmp_path):
    """load_flagged_fly_exclusions keys on dataset/fly/fly_number plus a
    FLY-State column; the report shares the first three, so it must never carry
    anything it could take for the fourth, nor the auto-excluding wide flag."""
    for col in qc.REPORT_COLUMNS:
        lowered = col.lower()
        assert "fly-state" not in lowered and "fly_state" not in lowered
        assert lowered not in {"flagged", "tracking_flagged"}
    root = _tree(tmp_path)
    qc.main(_cfg(root, tmp_path / "qc"))
    assert load_flagged_fly_exclusions(tmp_path / "qc" / qc.REPORT_FILENAME) == set()


# ---------------------------------------------------------------------------
# main: report, flagged CSV, logging, ntfy
# ---------------------------------------------------------------------------


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def test_main_writes_the_report_and_the_flagged_csv(tmp_path, ntfy_calls, capsys):
    root = _tree(tmp_path)
    qc.main(_cfg(root, tmp_path / "qc"))

    report = _read(tmp_path / "qc" / qc.REPORT_FILENAME)
    assert list(report.columns) == list(qc.REPORT_COLUMNS)
    assert len(report) == 5  # 4 flies + 1 legacy trial
    legacy = report[report["status"] == "no_live_eye_columns"]
    assert len(legacy) == 1 and legacy["fly_number"].isna().all()

    flagged = _read(tmp_path / "qc" / qc.FLAGGED_FILENAME)
    assert list(flagged.columns) == list(qc.REPORT_COLUMNS)
    assert flagged[["trial_dir", "fly_number"]].values.tolist() == [[_trial(root, "pretest_1"), 2]]
    assert flagged["eye_qc_reason"].iloc[0].startswith("drift > 5.0 px (median over 0.5 s) from 30.")

    out = capsys.readouterr().out
    assert f"[EYE-DRIFT][FLAG] {DATASET}/{BATCH}/{TRIAL_PREFIX}_pretest_1 fly2: drift > 5.0 px" in out
    summary = [line for line in out.splitlines() if "trial-flies" in line]
    assert len(summary) == 1
    assert "4 trial-flies in 1 trial" in summary[0]
    assert "1 flagged (1 drift / 0 eye_missing)" in summary[0]
    assert "1 trial without per-frame eye columns" in summary[0]
    assert str(tmp_path / "qc" / qc.REPORT_FILENAME) in summary[0]
    # Legacy trials are counted, never logged one by one.
    assert f"{TRIAL_PREFIX}_pretest_2" not in out


def test_main_reports_a_long_missing_run_as_eye_missing(tmp_path, ntfy_calls, capsys):
    root = tmp_path / "data" / DATASET
    trial = root / BATCH / f"{TRIAL_PREFIX}_testing_1"
    _write_merged(trial, _merged_table([_jitter(), _gap(N_FRAMES, 3240, 3360)]))

    qc.main(_cfg(root, tmp_path / "qc"))

    out = capsys.readouterr().out
    assert (
        f"[EYE-DRIFT][FLAG] {DATASET}/{BATCH}/{trial.name} fly2: "
        "eye unmatched for 3.0 s in a row from 81.0 s (moved > 50 px?)"
    ) in out
    assert "1 flagged (0 drift / 1 eye_missing)" in out
    flagged = _read(tmp_path / "qc" / qc.FLAGGED_FILENAME)
    assert list(flagged.columns) == list(qc.REPORT_COLUMNS)
    assert flagged["fly_number"].tolist() == [2]
    assert flagged["longest_missing_run_start_s"].tolist() == [pytest.approx(3239 / FPS)]
    assert flagged["max_missing_run_s"].tolist() == [2.0]
    assert len(ntfy_calls) == 1


def test_the_report_is_upserted_never_overwritten(tmp_path, ntfy_calls):
    root = _tree(tmp_path)
    report_path = tmp_path / "qc" / qc.REPORT_FILENAME
    report_path.parent.mkdir(parents=True)
    other = str(tmp_path / "data" / "Other-Set" / "may_01_batch_1" / "may_01_batch_1_testing_1")
    pretest_1 = _trial(root, "pretest_1")
    pd.DataFrame(
        [
            # Another dataset's subprocess wrote this; it must survive untouched.
            {"dataset": "Other-Set", "trial_dir": other, "fly_number": 1, "status": "assessed",
             "eye_qc_flag": True, "eye_qc_reason": "eye unmatched in 48% of frames"},
            # pretest_1 as it was before re-inference: one trial-level row ...
            {"dataset": DATASET, "trial_dir": pretest_1, "fly_number": None,
             "status": "no_live_eye_columns", "eye_qc_flag": False},
            # ... and a key this run assesses again, with its old values.
            {"dataset": DATASET, "trial_dir": pretest_1, "fly_number": 2, "status": "assessed",
             "eye_qc_flag": False, "eye_qc_reason": ""},
        ]
    ).to_csv(report_path, index=False)

    qc.main(_cfg(root, tmp_path / "qc"))
    report = _read(report_path)

    kept = report[report["trial_dir"] == other]
    assert len(kept) == 1
    assert kept["eye_qc_reason"].iloc[0] == "eye unmatched in 48% of frames"
    mine = report[report["trial_dir"] == pretest_1]
    assert sorted(mine["fly_number"].astype(int)) == [1, 2, 3, 4]
    assert set(mine["status"]) == {"assessed"}
    assert bool(mine.loc[mine["fly_number"] == 2, "eye_qc_flag"].iloc[0]) is True
    assert not report.duplicated(subset=["trial_dir", "fly_number"]).any()


@pytest.mark.parametrize("gone", ["trial_folder", "merged_table"])
def test_a_trial_gone_from_disk_leaves_the_report(tmp_path, ntfy_calls, gone):
    """Rows are upserted per trial, so a trial deleted from disk (the
    orphan-trial cleanup, a hand delete) or left without its merged table would
    otherwise keep its rows -- and its flag in the flagged CSV and the
    end-of-run count -- for good."""
    root = _tree(tmp_path)
    cfg = _cfg(root, tmp_path / "qc")
    qc.main(cfg)
    loose = _trial(root, "pretest_1")
    assert loose in set(_read(tmp_path / "qc" / qc.FLAGGED_FILENAME)["trial_dir"])

    if gone == "trial_folder":
        shutil.rmtree(loose)
    else:
        (Path(loose) / f"{Path(loose).name}_distances_merged.parquet").unlink()
    qc.main(cfg)

    assert set(_read(tmp_path / "qc" / qc.REPORT_FILENAME)["trial_dir"]) == {_trial(root, "pretest_2")}
    assert _read(tmp_path / "qc" / qc.FLAGGED_FILENAME).empty


def test_a_batch_moved_on_disk_is_reported_once_under_its_new_name(tmp_path, ntfy_calls):
    root = _tree(tmp_path)
    cfg = _cfg(root, tmp_path / "qc")
    qc.main(cfg)

    os.rename(root / BATCH, root / "october_07_batch_1_rig_3")  # re-filed under another rig
    qc.main(cfg)

    report = _read(tmp_path / "qc" / qc.REPORT_FILENAME)
    assert {Path(t).parent.name for t in report["trial_dir"]} == {"october_07_batch_1_rig_3"}
    assert len(report) == 5  # 4 flies + 1 legacy trial, once
    assert report["eye_qc_flag"].sum() == 1


def test_rows_under_a_root_without_batch_folders_are_kept(tmp_path, ntfy_calls):
    """An empty root is what an unmounted drive's mount point looks like: its
    batches are out of sight, not gone, so their rows stay."""
    root = _tree(tmp_path)
    empty_root = tmp_path / "data" / "Other-Set"
    empty_root.mkdir(parents=True)
    other = str(empty_root / "may_01_batch_1" / "may_01_batch_1_testing_1")
    report_path = tmp_path / "qc" / qc.REPORT_FILENAME
    report_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"dataset": "Other-Set", "trial_dir": other, "fly_number": 1, "status": "assessed",
          "eye_qc_flag": True, "eye_qc_reason": "eye unmatched in 48% of frames"}]
    ).to_csv(report_path, index=False)

    qc.main(_cfg(root, tmp_path / "qc", main_directories=[str(root), str(empty_root)],
                 datasets=(DATASET, "Other-Set")))

    report = _read(report_path)
    assert other in set(report["trial_dir"])
    assert _trial(root, "pretest_1") in set(report["trial_dir"])


def test_the_flagged_csv_leaves_out_a_folder_frozen_since(tmp_path, ntfy_calls):
    root = _tree(tmp_path)
    retired = root / "may_22_batch_1"
    retired.mkdir()
    retired_trial = str(retired / "may_22_batch_1_testing_1")
    report_path = tmp_path / "qc" / qc.REPORT_FILENAME
    report_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"dataset": DATASET, "fly": retired.name, "trial_dir": retired_trial, "fly_number": 1,
          "status": "assessed", "eye_qc_flag": True, "eye_qc_reason": "eye unmatched in 60% of frames"}]
    ).to_csv(report_path, index=False)

    cfg = _cfg(
        root,
        tmp_path / "qc",
        dataset_overrides={DATASET: DatasetOverride(freeze_folders=("may_22_batch_1",))},
    )
    qc.main(cfg)

    report = _read(report_path)
    flagged = _read(tmp_path / "qc" / qc.FLAGGED_FILENAME)
    assert retired_trial in set(report["trial_dir"])  # the full record keeps it
    assert retired_trial not in set(flagged["trial_dir"])
    assert _trial(root, "pretest_1") in set(flagged["trial_dir"])


def test_frozen_row_mask_marks_rows_of_frozen_batches(tmp_path):
    root = tmp_path / DATASET
    (root / "may_22_batch_1").mkdir(parents=True)
    (root / "october_07_batch_1").mkdir()
    report = pd.DataFrame(
        {
            "trial_dir": [
                str(root / "may_22_batch_1" / "may_22_batch_1_testing_1"),
                str(root / "october_07_batch_1" / "october_07_batch_1_testing_1"),
                str(root / "may_22_batch_1" / "may_22_batch_1_testing_2"),
            ]
        }
    )
    cfg = _cfg(root, tmp_path / "qc", dataset_overrides={DATASET: DatasetOverride(freeze_folders=("may_22*",))})
    assert qc.frozen_row_mask(report, cfg).tolist() == [True, False, True]
    assert qc.frozen_row_mask(report, cfg, thaw_all=True).tolist() == [False, False, False]
    assert qc.frozen_row_mask(report.iloc[:0], cfg).tolist() == []


def test_ntfy_fires_only_for_newly_flagged_flies(tmp_path, ntfy_calls):
    root = _tree(tmp_path)
    cfg = _cfg(root, tmp_path / "qc")

    qc.main(cfg)
    assert len(ntfy_calls) == 1
    first = ntfy_calls[0]
    assert first["priority"] == "high"
    assert f"{TRIAL_PREFIX}_pretest_1 fly2" in first["message"]
    assert str(tmp_path / "qc") in first["message"]

    qc.main(cfg)  # same tables, same flags: nothing new to say
    assert len(ntfy_calls) == 1

    t4 = f"{TRIAL_PREFIX}_pretest_4"
    _write_merged(root / BATCH / t4, _merged_table([_jitter(), _jitter(), _jitter(), _loose()]))
    qc.main(cfg)
    assert len(ntfy_calls) == 2
    second = ntfy_calls[1]["message"]
    assert f"{t4} fly4" in second
    assert f"{TRIAL_PREFIX}_pretest_1 fly2" not in second


def test_the_ntfy_lists_at_most_20_flies(tmp_path, ntfy_calls):
    root = tmp_path / "data" / DATASET
    batch = root / BATCH
    for k in range(1, 7):  # 6 trials x 4 loose flies = 24 flagged
        _write_merged(
            batch / f"{TRIAL_PREFIX}_testing_{k}",
            _merged_table([_loose(400, at=100)] * 4),
        )
    qc.main(_cfg(root, tmp_path / "qc"))

    (call,) = ntfy_calls
    assert "24" in call["title"]
    lines = call["message"].splitlines()
    assert sum(line.startswith("- ") for line in lines) == 20
    assert "...and 4 more" in call["message"]
    assert str(tmp_path / "qc" / qc.FLAGGED_FILENAME) in call["message"]


def test_serial_and_parallel_runs_write_identical_reports(tmp_path, ntfy_calls):
    root = _tree(tmp_path, variety=True)
    second = root / "october_08_batch_1_rig_3"
    _write_merged(second / "october_08_batch_1_testing_1", _merged_table([_jitter(), _loose(), _jitter()]))

    serial = _cfg(root, tmp_path / "serial")
    parallel = replace(
        serial,
        parallel=ParallelSettings(enabled=True, n_jobs=2),
        eye_qc=replace(serial.eye_qc, report_dir=str(tmp_path / "parallel")),
    )
    qc.main(serial)
    qc.main(parallel)

    for name in (qc.REPORT_FILENAME, qc.FLAGGED_FILENAME):
        serial_bytes = (tmp_path / "serial" / name).read_bytes()
        assert serial_bytes == (tmp_path / "parallel" / name).read_bytes()
    assert len(_read(tmp_path / "serial" / qc.FLAGGED_FILENAME)) == 2


def _infer_trial_with_yolo(monkeypatch, cfg, trial: Path, *, n_frames: int, shift_at: int, shift_px: float):
    """Write ``trial``'s merged table the way yolo_infer's chunked inference
    does, over canned detections of two eyes 300 px apart: fly 2's eye moves
    ``shift_px`` right at frame ``shift_at`` and stays there."""
    yolo_infer = pytest.importorskip("fbpipe.steps.yolo_infer")
    from fbpipe.utils.multi_fly import EyeAnchorManager, StablePairing
    from fbpipe.utils.tables import write_table

    eyes = ((300.0, 100.0), (300.0, 400.0))

    def _detections(frame_idx):
        boxes = [
            [x + dx - 8, y - 8, x + dx + 8, y + 8]
            for k, (x, y) in enumerate(eyes)
            for dx in [shift_px if k == 1 and frame_idx >= shift_at else 0.0]
        ]
        return {
            0: {"boxes": np.array(boxes, np.float32), "scores": np.full(2, 0.9, np.float32)},
            1: {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32)},
        }

    monkeypatch.setattr(yolo_infer, "collect_detections", lambda result, classes: _detections(result))

    class _Cap:
        def __init__(self):
            self.i = 0

        def isOpened(self):
            return self.i < n_frames

        def read(self):
            if self.i >= n_frames:
                return False, None
            self.i += 1
            return True, np.zeros((600, 600, 3), np.uint8)

    class _Writer:
        def write(self, frame):
            pass

    frame_ids = iter(range(n_frames))  # each canned "result" is its frame index

    def _predict(frames, conf):
        return [next(frame_ids) for _ in frames]

    rows = yolo_infer._run_chunked_inference(
        _Cap(), n_frames - 1, (600, 600), _Writer(), {}, FPS, (600.0, 300.0), cfg, _predict,
        {}, None, EyeAnchorManager(max_eyes=2), yolo_infer._build_proboscis_tracker(cfg, 2),
        StablePairing(max_pairs=2), 2, 16,
    )
    trial.mkdir(parents=True)
    write_table(pd.DataFrame(rows), trial / f"{trial.name}_distances_merged.csv")


def test_a_loose_fly_tracked_by_yolo_infer_is_flagged(tmp_path, monkeypatch, ntfy_calls):
    """The contract with yolo_infer: the merged table its chunked inference
    writes is the table this step reads. A column rename there would otherwise
    turn every new trial into a silent no_live_eye_columns row."""
    root = tmp_path / "data" / DATASET
    cfg = _cfg(root, tmp_path / "qc", max_flies=2, use_optical_flow=False)
    shift_at = 200
    _infer_trial_with_yolo(
        monkeypatch, cfg, root / BATCH / f"{TRIAL_PREFIX}_testing_1", n_frames=600, shift_at=shift_at, shift_px=8.0
    )

    qc.main(cfg)

    report = _read(tmp_path / "qc" / qc.REPORT_FILENAME)
    assert report["status"].tolist() == ["assessed", "assessed"]
    assert report["fly_number"].tolist() == [1, 2]
    assert report["eye_qc_flag"].tolist() == [False, True]
    assert report["first_over_thresh_s"].iloc[1] == pytest.approx(shift_at / FPS, abs=0.3)


def test_an_eye_that_jumps_past_the_match_radius_late_is_flagged_through_yolo_infer(
    tmp_path, monkeypatch, ntfy_calls, capsys
):
    """What the run rule is for, end to end: fly 2's eye jumps 60 px -- past
    the 50 px match radius -- 12.5 s into a 15 s trial and stays there.
    yolo_infer records every later frame as unmatched, never as drift, and
    those 2.5 s are only 17 % of the frames, under max_missing_frac."""
    root = tmp_path / "data" / DATASET
    cfg = _cfg(root, tmp_path / "qc", max_flies=2, use_optical_flow=False)
    _infer_trial_with_yolo(
        monkeypatch, cfg, root / BATCH / f"{TRIAL_PREFIX}_testing_1", n_frames=600, shift_at=500, shift_px=60.0
    )

    qc.main(cfg)

    report = _read(tmp_path / "qc" / qc.REPORT_FILENAME)
    assert report["eye_qc_flag"].tolist() == [False, True]
    fly2 = report.iloc[1]
    assert fly2["drift_max_px"] < 1.0
    assert fly2["eye_missing_frac"] < cfg.eye_qc.max_missing_frac
    assert fly2["eye_qc_reason"] == "eye unmatched for 2.5 s in a row from 12.5 s (moved > 50 px?)"
    assert "1 flagged (0 drift / 1 eye_missing)" in capsys.readouterr().out
    (call,) = ntfy_calls
    assert "fly2: eye unmatched for 2.5 s in a row from 12.5 s" in call["message"]


class _NoProboscis:
    def step(self, boxes, scores):
        return []


def test_a_phantom_anchor_tracked_by_yolo_infer_reads_as_a_phantom(monkeypatch):
    """The fly-count scan over-counted: a false-positive "eye" in frame 0 became
    a 4th anchor among 3 real flies and never came back. yolo_infer matches the
    frame an anchor was made from against that anchor, so the slot has one
    0 px match -- it must still read as a phantom, not as a fly that came loose."""
    yolo_infer = pytest.importorskip("fbpipe.steps.yolo_infer")
    from fbpipe.utils.multi_fly import EyeAnchorManager, StablePairing

    real = (ANCHORS[0], ANCHORS[1], ANCHORS[3])
    false_eye = (300.0, 600.0)
    dets = {}
    monkeypatch.setattr(yolo_infer, "collect_detections", lambda result, classes: dets)
    settings = Settings(model_path="model.pt", main_directories="/tmp/videos", max_flies=4, use_optical_flow=False)
    eye_mgr, pairer, rows = EyeAnchorManager(max_eyes=4), StablePairing(max_pairs=4), []
    n = 400
    for f, ts in enumerate(_timestamps(n)):
        eyes = real + ((false_eye,) if f == 0 else ())
        dets[0] = {
            "boxes": np.array([[x - 6, y - 6, x + 6, y + 6] for x, y in eyes], np.float32),
            "scores": np.array([0.9, 0.85, 0.8, 0.5][: len(eyes)], np.float32),
        }
        dets[1] = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32)}
        _, row, _ = yolo_infer._process_frame(
            np.zeros((1100, 600, 3), np.uint8), f, ts, {}, None, (1080.0, 540.0), settings, object(),
            eye_mgr, _NoProboscis(), pairer, 4,
        )
        rows.append(row)

    report = qc.assess_trial(pd.DataFrame(rows), EyeQCSettings(), FPS)

    assert [r["fly_number"] for r in report] == [1, 2, 3, 4]
    assert (report[2]["anchor_x"], report[2]["anchor_y"]) == false_eye
    assert [r["eye_qc_flag"] for r in report] == [False, False, True, False]
    assert report[2]["eye_qc_reason"] == "eye never matched (phantom slot?)"


# ---------------------------------------------------------------------------
# main never raises, and says so when it skips
# ---------------------------------------------------------------------------


def test_main_never_raises(tmp_path, monkeypatch, capsys):
    root = _tree(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(qc, "_process_fly_dir", _boom)
    qc.main(_cfg(root, tmp_path / "qc"))
    assert "worker crashed" in capsys.readouterr().out


def test_an_unwritable_report_dir_does_not_raise_and_still_pings_the_flag(tmp_path, ntfy_calls, capsys):
    root = _tree(tmp_path)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where the report dir should be", encoding="utf-8")
    qc.main(_cfg(root, blocker))
    out = capsys.readouterr().out
    assert "[EYE-DRIFT]" in out and "not written" in out.lower()
    # The flag must still reach the phone, saying the report could not be
    # written -- so it repeats until it can -- and not pointing at a flagged
    # CSV this run never wrote.
    (call,) = ntfy_calls
    assert f"{TRIAL_PREFIX}_pretest_1 fly2" in call["message"]
    assert f"{qc.REPORT_FILENAME} could NOT be written" in call["message"]
    assert "sent again" in call["message"]
    assert qc.FLAGGED_FILENAME not in call["message"]


_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.mark.skipif(_AS_ROOT, reason="root ignores directory permissions")
def test_a_read_only_report_dir_still_pings_a_fly_flagged_that_night(tmp_path, ntfy_calls, capsys):
    """A report dir left root-owned by a sudo run, or a full disk: the report
    cannot be updated, but a fly that came loose that night must still reach the
    phone -- once per run until the report can be written again."""
    root = _tree(tmp_path)
    loose = Path(_trial(root, "pretest_1"))
    table = loose / f"{loose.name}_distances_merged.parquet"
    _merged_table([_jitter()] * 4).to_parquet(table, index=False)
    qcdir = tmp_path / "qc"
    cfg = _cfg(root, qcdir)
    qc.main(cfg)  # night 1: nothing loose yet
    assert ntfy_calls == []

    _merged_table([_jitter(), _loose(), _jitter(), _jitter()]).to_parquet(table, index=False)
    before = (qcdir / qc.REPORT_FILENAME).read_bytes()
    qcdir.chmod(0o555)
    try:
        qc.main(cfg)  # night 2: fly 2 came loose, and the report dir is read-only
    finally:
        qcdir.chmod(0o755)

    (call,) = ntfy_calls
    assert f"{loose.name} fly2" in call["message"]
    assert f"{qc.REPORT_FILENAME} could NOT be written" in call["message"]
    assert (qcdir / qc.REPORT_FILENAME).read_bytes() == before
    assert "Report NOT written" in capsys.readouterr().out


@pytest.mark.skipif(_AS_ROOT, reason="root ignores directory permissions")
def test_a_prior_report_that_can_be_neither_read_nor_set_aside_is_left_alone(tmp_path, ntfy_calls):
    """Writing a fresh report over it would lose every other dataset's rows, so
    it is left as it is; with no prior state to compare against, every flag of
    the run counts as new."""
    root = _tree(tmp_path)
    qcdir = tmp_path / "qc"
    qcdir.mkdir()
    report_path = qcdir / qc.REPORT_FILENAME
    report_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00 not a csv report")
    qcdir.chmod(0o555)
    try:
        qc.main(_cfg(root, qcdir))
    finally:
        qcdir.chmod(0o755)

    assert report_path.read_bytes().startswith(b"\x89PNG")
    assert list(qcdir.iterdir()) == [report_path]
    (call,) = ntfy_calls
    assert f"{TRIAL_PREFIX}_pretest_1 fly2" in call["message"]
    assert f"{qc.REPORT_FILENAME} could NOT be written" in call["message"]


def test_an_unreadable_prior_report_is_set_aside_not_lost(tmp_path, ntfy_calls, capsys):
    root = _tree(tmp_path)
    report_path = tmp_path / "qc" / qc.REPORT_FILENAME
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00 not a csv report")

    qc.main(_cfg(root, tmp_path / "qc"))

    assert list(_read(report_path).columns) == list(qc.REPORT_COLUMNS)
    aside = list(report_path.parent.glob(f"{qc.REPORT_FILENAME}.unreadable-*"))
    assert len(aside) == 1
    assert aside[0].read_bytes().startswith(b"\x89PNG")
    assert "WARNING" in capsys.readouterr().out


def test_unusable_thresholds_never_stop_the_step(tmp_path, ntfy_calls):
    root = _tree(tmp_path)
    qc.main(_cfg(root, tmp_path / "qc", eye_qc={"max_drift_px": -2.0, "max_missing_frac": 3.0}))
    report = _read(tmp_path / "qc" / qc.REPORT_FILENAME)
    assert set(report["max_drift_px"]) == {5.0}
    assert set(report["max_missing_frac"]) == {0.2}
    assert report["eye_qc_flag"].sum() == 1


def test_disabled_eye_qc_skips_the_step(tmp_path, ntfy_calls, capsys):
    root = _tree(tmp_path)
    qc.main(_cfg(root, tmp_path / "qc", eye_qc={"enabled": False}))
    assert not (tmp_path / "qc").exists()
    assert "eye_qc.enabled is false" in capsys.readouterr().out
    assert ntfy_calls == []


def test_a_root_without_merged_tables_warns_loudly(tmp_path, ntfy_calls, capsys):
    root = tmp_path / "data" / DATASET
    batch = root / BATCH
    (batch / f"{TRIAL_PREFIX}_testing_1").mkdir(parents=True)
    (batch / f"output_{TRIAL_PREFIX}_testing_1_Hexanol_20261007_110648.mp4").write_bytes(b"")

    qc.main(_cfg(root, tmp_path / "qc"))

    out = capsys.readouterr().out
    assert "WARNING" in out and "_distances_merged" in out and str(root) in out
    assert not (tmp_path / "qc" / qc.REPORT_FILENAME).exists()


def test_an_all_frozen_root_is_not_a_warning(tmp_path, ntfy_calls, capsys):
    root = _tree(tmp_path)
    cfg = _cfg(root, tmp_path / "qc", dataset_overrides={DATASET: DatasetOverride(freeze_folders=(BATCH,))})
    qc.main(cfg)
    assert "WARNING" not in capsys.readouterr().out
    assert not (tmp_path / "qc" / qc.REPORT_FILENAME).exists()


def _summary_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("[EYE-DRIFT] assessed ")]


@pytest.mark.parametrize(
    "scenario", ["tables", "no_tables", "all_frozen", "no_batch_folders", "missing_root"]
)
def test_every_call_prints_exactly_one_summary_line(tmp_path, ntfy_calls, capsys, scenario):
    """Each pipeline subprocess says what the step did -- also when it found
    nothing to assess, or a silent step reads exactly like one that never ran."""
    root = tmp_path / "data" / DATASET
    overrides = {}
    if scenario in ("tables", "all_frozen"):
        root = _tree(tmp_path)
    if scenario == "all_frozen":
        overrides["dataset_overrides"] = {DATASET: DatasetOverride(freeze_folders=(BATCH,))}
    elif scenario == "no_tables":
        (root / BATCH / f"{TRIAL_PREFIX}_testing_1").mkdir(parents=True)
    elif scenario == "no_batch_folders":
        root.mkdir(parents=True)

    qc.main(_cfg(root, tmp_path / "qc", **overrides))

    (summary,) = _summary_lines(capsys.readouterr().out)
    report = tmp_path / "qc" / qc.REPORT_FILENAME
    if scenario == "tables":
        assert summary.endswith(f". Report: {report}")
        return
    assert summary == (
        "[EYE-DRIFT] assessed 0 trial-flies in 0 trials; 0 flagged (0 drift / 0 eye_missing); "
        f"0 trials without per-frame eye columns. Report unchanged: {report}"
    )
    assert not report.exists()


def test_a_root_without_batch_folders_says_so_plainly(tmp_path, ntfy_calls, capsys):
    """A dataset root nothing was recorded into yet is no fault: one plain line,
    no WARNING banner (that is for batches YOLO left without merged tables)."""
    root = tmp_path / "data" / DATASET
    root.mkdir(parents=True)
    (root / "notes.txt").write_text("a file is not a batch folder", encoding="utf-8")

    qc.main(_cfg(root, tmp_path / "qc"))

    out = capsys.readouterr().out
    (line,) = [line for line in out.splitlines() if "no batch folders" in line.lower()]
    assert line.startswith("[EYE-DRIFT] ") and str(root) in line
    assert "WARNING" not in out
    assert not (tmp_path / "qc").exists()


def test_an_all_frozen_root_does_not_read_as_an_empty_one(tmp_path, ntfy_calls, capsys):
    root = _tree(tmp_path)
    qc.main(_cfg(root, tmp_path / "qc", dataset_overrides={DATASET: DatasetOverride(freeze_folders=(BATCH,))}))
    out = capsys.readouterr().out
    assert "no batch folders" not in out.lower()
    assert f"skipping 1 frozen folder(s): {BATCH}" in out  # iter_live_batch_dirs already said why


def test_the_step_imports_without_torch_cv2_or_ultralytics():
    """It runs in every pipeline subprocess; it must stay a light import."""
    src = str(Path(qc.__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    code = (
        "import sys, fbpipe.steps.eye_drift_qc as m; "
        "print(m.__file__); "
        "print(sorted(k for k in ('torch', 'cv2', 'ultralytics') if k in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    module_file, heavy = out.stdout.strip().splitlines()
    assert Path(module_file).resolve() == Path(qc.__file__).resolve()
    assert heavy == "[]"
