import json
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "rigmon" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")


def test_page_is_self_contained():
    assert not re.search(r'(src|href)="https?://', PAGE), "no CDN — the hub may be offline"


def test_page_uses_the_api():
    for needle in ("/api/stream", "/api/mute/", "/media/", "EventSource"):
        assert needle in PAGE


def test_page_is_phone_friendly_and_installable():
    assert 'name="viewport"' in PAGE and 'rel="manifest"' in PAGE
    manifest = json.loads((STATIC / "manifest.webmanifest").read_text())
    assert manifest["display"] == "standalone" and manifest["start_url"] == "/"


def test_page_handles_every_phase():
    for phase in ("IDLE", "WAITING", "BASELINE", "STIMULUS", "POST", "GAP", "COMPLETE"):
        assert phase in PAGE


def test_page_corrects_both_clock_offsets():
    assert "server_time" in PAGE and "fetched_at" in PAGE


def test_page_has_dark_mode_tokens():
    assert "prefers-color-scheme: dark" in PAGE and ":root" in PAGE


def test_page_has_no_inline_handlers_and_escapes_quotes():
    assert "onclick=" not in PAGE
    assert 'addEventListener("click"' in PAGE
    assert "&#39;" in PAGE


def test_page_ticks_without_rebuilding_and_guards_phase_lookup():
    assert "setInterval(tick" in PAGE and "setInterval(render" not in PAGE
    assert "hasOwnProperty" in PAGE


def test_page_says_plainly_when_nothing_is_running():
    """An unreachable rig between batches is normal: say so instead of 'offline'.
    Only a rig that vanished mid-run is an alarm."""
    for text in ("No experiment running", "Not running", "Lost contact",
                 "Waiting for intake", "Experiment complete"):
        assert text in PAGE
    assert 'phaseName : "offline"' not in PAGE


def test_page_reconnects_after_the_stream_closes():
    assert "EventSource.CLOSED" in PAGE and "setTimeout(connect" in PAGE


def test_page_has_mount_check_ui():
    for needle in ("Check mount now", "data-mount=", "/api/mountcheck/", "/qc/", "mount_check",
                   "Mount ✓", "issue"):
        assert needle in PAGE
    assert "onclick=" not in PAGE


def test_mount_ui_survives_rerender():
    assert "const mcOpen = new Set()" in PAGE and "mcOpen.has(name)" in PAGE
    t = PAGE.index('addEventListener("toggle"')
    assert "}, true);" in PAGE[t:t + 400]
    assert "data-card=" in PAGE and "lastCard" in PAGE
    assert "S.mount_check_enabled" in PAGE
    # running branch returns before any button is built
    mb = PAGE[PAGE.index("function mountBlock"):]
    run = mb.index('m.state === "running"')
    assert "Mount check running" in mb[run:run + 200] and "return" in mb[run:run + 200]
    assert mb.index("const button") > run


def test_page_partial_sample_is_amber_not_a_tick():
    assert "partial sample" in PAGE and "re-run" in PAGE
    mb = PAGE[PAGE.index("function mountBlock"):]
    assert mb.index('m.state === "incomplete"') < mb.index("Mount ✓")
    assert ".mc .partial" in PAGE and "var(--base)" in PAGE[PAGE.index(".mc .partial"):][:120]


def test_page_shows_mount_check_unavailable_and_uses_jpg_name():
    assert "Mount check unavailable" in PAGE and "S.mount_check_error" in PAGE
    assert "jpg_name" in PAGE and "jpg_path" not in PAGE
    assert "Could not start" in PAGE and ".catch(" in PAGE
    assert "[data-card] > .card { height: 100%; }" in PAGE


def test_page_has_trace_button_and_dialog():
    for needle in ("data-trace", "/traces/", "tracebox", "trace failed"):
        assert needle in PAGE
    assert "onclick=" not in PAGE


def test_page_trial_rows_not_rebuilt_unchanged_and_trace_closes():
    assert "lastTrials" in PAGE
    assert 'getElementById("tracebox").addEventListener("close"' in PAGE
    assert 'removeAttribute("src")' in PAGE


def test_wait_buttons_present():
    for marker in ('data-wait-add="300"', 'data-wait-add="900"', "data-wait-pause", "data-wait-resume",
                   "/api/rig/", "/wait"):
        assert marker in PAGE
    assert "onclick=" not in PAGE.lower()


def test_wait_buttons_gated_on_startup_wait():
    i = PAGE.index("function waitControls")
    seg = PAGE[i:i + 1500]
    for cond in ("rig.control", '"WAITING"', "trials_done", '"running"'):
        assert cond in seg


def test_page_is_view_only_until_admin_login():
    assert 'fetch("/api/whoami")' in PAGE and '"viewonly"' in PAGE
    assert 'href="/login"' in PAGE
    for sel in ("body.viewonly button[data-mute]", "body.viewonly button[data-mount]",
                "body.viewonly .wait-ctl"):
        assert sel in PAGE
