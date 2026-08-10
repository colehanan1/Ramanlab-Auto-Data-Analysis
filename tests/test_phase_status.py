"""Unit tests for PiCode/phase_status.py — the import-safe half of the live
experiment status page.

``phase_status`` deliberately holds everything that can run off-Pi: the phase
state machine, the step classifier, and the self-contained HTML page. The
hardware-bound wiring inside ``combinedv2_1.py`` is covered separately by
``test_status_page_wiring.py`` (AST-based — that module cannot be imported off
the rig).
"""

import json
import threading
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "PiCode"))

import phase_status as ps  # noqa: E402


# ── step classification ──────────────────────────────────────────────────────

ODOR_STEP = {
    "name": "Odor + Light Pattern",
    "duration": 30,
    "odor_label": "Hexanol",
    "actions": [{"device": "ofm", "pin": "OFM_H", "state": "on"}],
    "light_schedule": [{"start": 5, "end": 30}],
}

LIGHT_ONLY_STEP = {
    "name": "Testing Light Only (No Odor)",
    "duration": 25,
    "odor_label": "LightOnly",
    "light_schedule": [{"start": 0, "end": 25}],
}

BASELINE_STEP = {"name": "Baseline Period", "duration": 30}


def test_odor_step_is_stimulus():
    phase = ps.classify_step(ODOR_STEP, recording=True, stimulus_delivered=False)
    assert phase == ps.PHASE_STIMULUS


def test_odor_beats_light_for_stimulus_kind():
    """Training odor step carries a light_schedule too — it must read ODOR."""
    assert ps.stimulus_kind(ODOR_STEP) == "ODOR"


def test_light_only_step_is_stimulus_of_kind_light():
    phase = ps.classify_step(LIGHT_ONLY_STEP, recording=True, stimulus_delivered=False)
    assert phase == ps.PHASE_STIMULUS
    assert ps.stimulus_kind(LIGHT_ONLY_STEP) == "LIGHT"


def test_recording_baseline_before_stimulus_is_baseline():
    phase = ps.classify_step(BASELINE_STEP, recording=True, stimulus_delivered=False)
    assert phase == ps.PHASE_BASELINE


def test_recording_after_stimulus_is_post():
    phase = ps.classify_step(
        {"name": "Post-Stimulus Period", "duration": 30},
        recording=True, stimulus_delivered=True,
    )
    assert phase == ps.PHASE_POST


def test_gap_when_not_recording():
    """The 240 s inter-trial step is also named 'Baseline Period' — only the
    recording flag distinguishes it from the pre-odor baseline."""
    phase = ps.classify_step(BASELINE_STEP, recording=False, stimulus_delivered=True)
    assert phase == ps.PHASE_GAP


def test_control_steps_are_not_classified():
    for name in ("Start Recording", "Stop Recording", "Ntfy Notification"):
        step = {"name": name, "duration": 0}
        assert ps.classify_step(step, recording=True, stimulus_delivered=False) is None


def test_non_stimulus_step_has_no_stimulus_kind():
    assert ps.stimulus_kind(BASELINE_STEP) is None


def test_ofm_off_action_is_not_a_stimulus():
    step = {
        "name": "Odor Off",
        "duration": 5,
        "actions": [{"device": "ofm", "pin": "OFM_H", "state": "off"}],
    }
    assert ps.stimulus_kind(step) is None


# ── trial titles ─────────────────────────────────────────────────────────────

def test_trial_title_training():
    assert ps.trial_title("training_3") == "Training 3"


def test_trial_title_testing_with_odor_suffix():
    assert ps.trial_title("testing_11_ACV") == "Testing 11"


def test_trial_title_unknown_is_generic():
    assert ps.trial_title("unknown") == "Trial"


# ── PhaseState ───────────────────────────────────────────────────────────────

def _fixed_clock(t):
    return lambda: t


def test_initial_snapshot_is_idle():
    state = ps.PhaseState(clock=_fixed_clock(100.0))
    snap = state.snapshot()
    assert snap["phase"] == ps.PHASE_IDLE
    assert snap["server_time"] == 100.0


def test_enter_stores_absolute_deadlines():
    state = ps.PhaseState(clock=_fixed_clock(1000.0))
    state.enter(ps.PHASE_BASELINE, label="Odor in", trial_label="Training 3/6",
                odor="Hexanol", ends_at=1030.0)
    snap = state.snapshot()
    assert snap["phase"] == ps.PHASE_BASELINE
    assert snap["ends_at"] == 1030.0          # absolute, not remaining
    assert snap["started_at"] == 1000.0       # defaults to clock at enter()
    assert snap["trial_label"] == "Training 3/6"
    assert snap["odor"] == "Hexanol"
    assert snap["label"] == "Odor in"


def test_enter_clears_stale_fields_from_previous_phase():
    """A field set in one phase must not leak into the next snapshot."""
    state = ps.PhaseState(clock=_fixed_clock(1.0))
    state.enter(ps.PHASE_STIMULUS, stim="ODOR", odor="Hexanol", ends_at=31.0)
    state.enter(ps.PHASE_GAP, label="Next trial in", ends_at=271.0)
    snap = state.snapshot()
    assert snap["stim"] == ""
    assert snap["odor"] == ""


def test_session_metadata_survives_phase_changes():
    state = ps.PhaseState(clock=_fixed_clock(1.0))
    state.set_session(fly=101, odor="Hex", mode="Training", kind="Manual",
                      dataset="Hex-Training-24-0.005-Manual")
    state.enter(ps.PHASE_WAITING, ends_at=5400.0)
    snap = state.snapshot()
    assert snap["session"]["fly"] == 101
    assert snap["session"]["dataset"] == "Hex-Training-24-0.005-Manual"


def test_to_json_round_trips():
    state = ps.PhaseState(clock=_fixed_clock(42.0))
    state.enter(ps.PHASE_POST, label="Recording ends in", ends_at=72.0)
    parsed = json.loads(state.to_json())
    assert parsed["phase"] == ps.PHASE_POST
    assert parsed["server_time"] == 42.0


def test_snapshot_returns_a_copy():
    state = ps.PhaseState(clock=_fixed_clock(1.0))
    state.snapshot()["phase"] = "TAMPERED"
    assert state.snapshot()["phase"] == ps.PHASE_IDLE


def test_concurrent_enter_and_snapshot_do_not_corrupt_state():
    state = ps.PhaseState()
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            state.enter(ps.PHASE_BASELINE, label=f"iter {i}", ends_at=float(i))
            i += 1

    def reader():
        while not stop.is_set():
            try:
                snap = state.snapshot()
                json.dumps(snap)
                assert snap["phase"] in (ps.PHASE_IDLE, ps.PHASE_BASELINE)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
                stop.set()

    threads = [threading.Thread(target=writer), threading.Thread(target=reader),
               threading.Thread(target=reader)]
    for t in threads:
        t.start()
    stop.wait(timeout=0.5)
    stop.set()
    for t in threads:
        t.join(timeout=2)
    assert not errors


# ── the HTML page ────────────────────────────────────────────────────────────

def test_page_embeds_the_preview_stream():
    assert 'src="/preview"' in ps.STATUS_PAGE_HTML


def test_page_polls_status_json():
    assert "/status.json" in ps.STATUS_PAGE_HTML


def test_page_is_fully_self_contained():
    """The rig may have no internet — no CDN scripts, fonts, or stylesheets."""
    for marker in ("http://", "https://", "cdn.", "googleapis", "@import"):
        assert marker not in ps.STATUS_PAGE_HTML, f"external reference: {marker}"


def test_page_is_phone_friendly():
    assert 'name="viewport"' in ps.STATUS_PAGE_HTML


def test_page_handles_every_phase():
    """The client-side renderer must know all published phase constants."""
    for phase in (ps.PHASE_IDLE, ps.PHASE_WAITING, ps.PHASE_BASELINE,
                  ps.PHASE_STIMULUS, ps.PHASE_POST, ps.PHASE_GAP,
                  ps.PHASE_COMPLETE):
        assert phase in ps.STATUS_PAGE_HTML, f"page ignores phase {phase}"


def test_page_corrects_for_clock_skew():
    """Countdowns must come from server deadlines + skew offset, not the
    client clock alone."""
    assert "server_time" in ps.STATUS_PAGE_HTML
