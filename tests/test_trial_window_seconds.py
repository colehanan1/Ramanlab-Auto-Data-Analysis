"""Wide-table odor-window seconds use the trace's own frame rate.

2026-11-05: Paraffin-Sensitivity-1-100/november_05_batch_1 testing_3 (Citral)
had an 8.4 s camera stall, so the rig sidecar reported 3261 frames at an
"achieved" 36.27 fps. The valve opened on frame 1196 (29.9 s on the camera
clock), but frame / 36.27 stored it as 32.97 s; the figure (trace at 40 fps)
shaded 33-66 s and the scorer (seconds x 40 fps) read frames 1319-2644. Seconds
must be frame / the fps the trace columns are indexed at, so seconds x fps
returns the valve's own frame.
"""

from types import SimpleNamespace

import pytest

from scripts.analysis.envelope_combined import _trial_window_seconds


def _meta(on, off, n, fps):
    return SimpleNamespace(odor_on_frame=on, odor_off_frame=off, n_frames=n, fps=fps)


def test_a_stalled_trial_keeps_its_valve_frames():
    on_s, off_s, dur_s = _trial_window_seconds(_meta(1196, 2397, 3261, 36.27), 40.0)
    assert on_s == pytest.approx(29.9)
    assert off_s == pytest.approx(59.925)
    assert dur_s == pytest.approx(3261 / 40.0)
    assert round(on_s * 40.0) == 1196 and round(off_s * 40.0) == 2397


def test_a_normal_trial_is_unchanged():
    on_s, off_s, dur_s = _trial_window_seconds(_meta(1196, 2397, 3601, 40.02), 40.0)
    assert (on_s, off_s) == (pytest.approx(29.9), pytest.approx(59.925))


def test_no_window_gives_nan():
    on_s, off_s, _ = _trial_window_seconds(_meta(None, None, 3600, 40.0), 40.0)
    assert on_s != on_s and off_s != off_s
