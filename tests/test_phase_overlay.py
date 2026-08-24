"""Text layout for the on-Pi QTGL preview overlay (phase_status.overlay_lines).

The overlay mirrors the web status page's render() switch: given a
PhaseState snapshot and "now", produce (title, big, banner) where ``banner``
is True only while a stimulus is actively on — the manual-reward cue.
Pure function, so it is testable off the rig; combinedv2_1.py owns the
cv2/picamera2 rendering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PiCode"))
import phase_status as ps  # noqa: E402


def _snap(phase, *, label="", trial_label="", odor="", stim="",
          started_at=None, ends_at=None):
    return {
        "phase": phase, "label": label, "trial_label": trial_label,
        "odor": odor, "stim": stim, "started_at": started_at,
        "ends_at": ends_at, "session": {}, "server_time": 0.0,
    }


def test_fmt_countdown_under_a_minute_is_tenths():
    assert ps.fmt_countdown(12.34) == "12.3 s"


def test_fmt_countdown_minutes_and_hours():
    assert ps.fmt_countdown(75) == "01:15"
    assert ps.fmt_countdown(3600 + 61) == "1:01:01"


def test_fmt_countdown_never_negative():
    assert ps.fmt_countdown(-3) == "0.0 s"


def test_idle_shows_waiting_placeholder():
    title, big, banner = ps.overlay_lines(_snap(ps.PHASE_IDLE), now=100.0)
    assert title == "WAITING FOR EXPERIMENT"
    assert big == "--:--"
    assert banner is False


def test_waiting_uses_label_and_countdown():
    snap = _snap(ps.PHASE_WAITING, label="Training 1 starts in", ends_at=190.0)
    title, big, banner = ps.overlay_lines(snap, now=100.0)
    assert title == "Training 1 starts in"
    assert big == "01:30"
    assert banner is False


def test_baseline_names_odor_and_trial():
    snap = _snap(ps.PHASE_BASELINE, trial_label="Training 2", odor="Hexanol",
                 ends_at=112.0)
    title, big, banner = ps.overlay_lines(snap, now=100.0)
    assert title == "Training 2 — BASELINE — Hexanol in"
    assert big == "12.0 s"
    assert banner is False


def test_stimulus_shows_elapsed_and_raises_banner():
    snap = _snap(ps.PHASE_STIMULUS, trial_label="Training 2", odor="Hexanol",
                 stim="ODOR", started_at=100.0, ends_at=110.0)
    title, big, banner = ps.overlay_lines(snap, now=103.0)
    assert title == "Training 2 — Hexanol ON"
    assert big == "3.0 s"
    assert banner is True


def test_light_stimulus_says_light_on():
    snap = _snap(ps.PHASE_STIMULUS, stim="LIGHT", started_at=100.0, ends_at=110.0)
    title, _, banner = ps.overlay_lines(snap, now=101.0)
    assert title == "LIGHT ON"
    assert banner is True


def test_post_counts_down_recording_end():
    snap = _snap(ps.PHASE_POST, trial_label="Testing 3", ends_at=130.0)
    title, big, banner = ps.overlay_lines(snap, now=100.0)
    assert title == "Testing 3 — POST — recording ends in"
    assert big == "30.0 s"
    assert banner is False


def test_gap_uses_label_default():
    snap = _snap(ps.PHASE_GAP, ends_at=340.0)
    title, big, banner = ps.overlay_lines(snap, now=100.0)
    assert title == "NEXT TRIAL IN"
    assert big == "04:00"
    assert banner is False


def test_complete_is_a_checkmark():
    title, big, banner = ps.overlay_lines(_snap(ps.PHASE_COMPLETE, label="all cycles finished"), now=0.0)
    assert title == "EXPERIMENT COMPLETE"
    assert big == "✓"
    assert banner is False


def test_missing_deadline_gives_placeholder():
    _, big, _ = ps.overlay_lines(_snap(ps.PHASE_WAITING, label="x"), now=5.0)
    assert big == "--:--"


def test_overlay_lines_from_live_state():
    """End-to-end through PhaseState so snapshot keys stay in sync."""
    st = ps.PhaseState(clock=lambda: 50.0)
    st.enter(ps.PHASE_STIMULUS, trial_label="Training 1", odor="ACV",
             stim="ODOR", ends_at=60.0)
    title, big, banner = ps.overlay_lines(st.snapshot(), now=52.5)
    assert (title, big, banner) == ("Training 1 — ACV ON", "2.5 s", True)
