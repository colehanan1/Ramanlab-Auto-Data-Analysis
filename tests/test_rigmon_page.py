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
