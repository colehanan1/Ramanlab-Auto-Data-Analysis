"""Tests for scripts/analysis/make_lab_flying_words_clip.py (flying-words title clip)."""

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "make_lab_flying_words_clip.py"
spec = importlib.util.spec_from_file_location("make_lab_flying_words_clip", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


# ---------------------------------------------------------------- ported demo math
def test_rand_matches_the_demo_glsl_hash():
    # rand(seed) = fract(sin(seed*127.1 + 311.7) * 43758.5453), as in Motion.tsx
    for seed in (0, 1, 7, 13.5, 99):
        x = math.sin(seed * 127.1 + 311.7) * 43758.5453
        assert m.rand(seed) == pytest.approx(x - math.floor(x), abs=1e-12)
        assert 0.0 <= m.rand(seed) < 1.0


def test_project_is_css_perspective_scaling():
    x, y, k = m.project(100.0, -40.0, 0.0)
    assert (x, y, k) == pytest.approx((100.0, -40.0, 1.0))
    x, y, k = m.project(100.0, -40.0, -1100.0)          # one perspective distance away -> half size
    assert (x, y, k) == pytest.approx((50.0, -20.0, 0.5))
    assert m.project(0, 0, 800.0)[2] == pytest.approx(1100 / 300)
    with pytest.raises(ValueError):
        m.project(0, 0, 1100.0)                          # at/behind the camera plane


def test_travel_constants_are_the_tuned_demo_values():
    assert m.PERSPECTIVE == 1100
    assert (m.Z_FAR, m.Z_NEAR) == (-1750, 800)
    assert m.z_at(0.0) == -1750 and m.z_at(1.0) == 800
    assert m.drift(0.0) == pytest.approx(0.5) and m.drift(1.0) == pytest.approx(1.85)
    assert m.near_blur(0.86) == 0.0 and m.near_blur(0.5) == 0.0
    assert m.near_blur(0.96) == pytest.approx(0.1 * 26)


@pytest.mark.parametrize("u, want", [(0.0, 0.0), (0.25, 1.0), (0.6, 0.5), (0.85, 0.2), (1.0, 0.0),
                                     (0.125, 0.5), (0.425, 0.75), (0.725, 0.35), (0.925, 0.1)])
def test_life_opacity_curve(u, want):
    assert m.life_opacity(u) == pytest.approx(want)


def test_golden_angle_placement_on_a_flattened_ellipse():
    for i, it in enumerate(m.ITEMS):
        a = i * 2.39996 + m.rand(i * 7 + 1) * 0.8
        r = 82 + m.rand(i * 13 + 2) * 165
        assert it.x == pytest.approx(math.cos(a) * r)
        assert it.y == pytest.approx(math.sin(a) * r * 0.6)
        assert 82 <= math.hypot(it.x, it.y / 0.6) <= 247 + 1e-9
    # phases stay evenly spaced i/N (the demo's density); review 2026-10-30 re-phases which
    # placement gets which phase so readable words never collide
    n = len(m.ITEMS)
    assert sorted(it.ph for it in m.ITEMS) == pytest.approx([k / n for k in range(n)])


def test_cycles_integer_and_frame_count_whole():
    assert isinstance(m.CYCLES, int) and m.CYCLES >= 1
    assert 10.0 <= m.DURATION_S <= 12.0
    assert m.N_FRAMES == round(m.DURATION_S * m.FPS)
    assert m.N_FRAMES == m.DURATION_S * m.FPS
    assert (m.W, m.H, m.FPS) == (1080, 1080, 40)


def test_slower_than_the_demo():
    # demo: 2 cycles in 6 s -> one pass every 3 s
    assert m.DURATION_S / m.CYCLES >= 2.5 * 3.0


# ---------------------------------------------------------------- loop + determinism
def test_word_phase_wraps_exactly():
    for i in range(len(m.ITEMS)):
        assert m.word_u(i, 0) == m.word_u(i, m.N_FRAMES)


def test_word_layer_loops_seamlessly():
    a = m.render_word_layer(0)
    b = m.render_word_layer(m.N_FRAMES)
    assert a.shape == (m.H, m.W, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_word_layer_actually_moves():
    assert not np.array_equal(m.render_word_layer(0), m.render_word_layer(m.N_FRAMES // 3))


def test_rendering_is_deterministic():
    f = m.N_FRAMES // 2
    assert np.array_equal(np.asarray(m.render_frame(f)), np.asarray(m.render_frame(f)))
    assert np.array_equal(m.render_word_layer(37), m.render_word_layer(37))


# ---------------------------------------------------------------- legibility
def test_every_word_has_a_legible_window_in_the_clip():
    for i, it in enumerate(m.ITEMS):
        s = m.legible_window_s(i)
        assert s >= 0.5, f"{it.text!r}: only {s:.2f} s at >= 32 px and >= 0.8 opacity"


def test_legible_window_frames_really_meet_the_floor():
    for i in range(len(m.ITEMS)):
        frames = m.legible_frames(i)
        assert frames, m.ITEMS[i].text
        for f in frames:
            st = m.word_state(i, f)
            assert st.px >= 32
            assert st.alpha_eff >= 0.8
            assert st.blur_px == 0.0
            l, t, r, b = st.bbox
            assert l >= 0 and t >= 0 and r <= m.W and b <= m.H


def test_legibility_rejects_small_or_faint_states():
    # far end of travel: tiny and fading in -> never counts
    st = m.word_state(0, 0)
    assert m.is_legible(st._replace(px=31.9, alpha_eff=1.0, blur_px=0.0, bbox=(10, 10, 20, 20))) is False
    assert m.is_legible(st._replace(px=40.0, alpha_eff=0.79, blur_px=0.0, bbox=(10, 10, 20, 20))) is False
    assert m.is_legible(st._replace(px=40.0, alpha_eff=0.9, blur_px=0.0, bbox=(-1, 10, 20, 20))) is False
    assert m.is_legible(st._replace(px=40.0, alpha_eff=0.9, blur_px=0.0, bbox=(10, 10, 20, 20))) is True


def test_scrim_dims_words_behind_the_title_only():
    # word_attenuation(x, y): y is the bottom of the word's ink box
    assert m.word_attenuation(540, 200) == pytest.approx(1.0)
    assert m.word_attenuation(540, m.H - 60) == 0.0
    assert m.word_attenuation(540, m.TITLE_Y - 20) == 0.0      # gone before it reaches the accent bar


# ---------------------------------------------------------------- review 2026-10-30: collisions, edges, title
READABLE_PX, READABLE_ALPHA, OVERLAP_TOL, MIN_GAP_PX = 32, 0.5, 0.03, 8.0


def _readable(st):
    return st.px >= READABLE_PX and st.alpha_eff >= READABLE_ALPHA and st.blur_px == 0.0


def _poly(st):
    from shapely.geometry import Polygon
    return Polygon(st.quad)


def test_word_quad_is_the_rotated_ink_box():
    st = m.word_state(3, 200)
    xs = [x for x, _ in st.quad]
    ys = [y for _, y in st.quad]
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx(st.bbox)
    assert _poly(st).is_valid and _poly(st).area > 0


def test_readable_words_never_overlap():
    worst = []
    for f in range(m.N_FRAMES):
        sts = [(m.ITEMS[i].text, m.word_state(i, f)) for i in range(len(m.ITEMS))]
        rd = [(w, _poly(st)) for w, st in sts if _readable(st)]
        for a in range(len(rd)):
            for b in range(a + 1, len(rd)):
                (wa, pa), (wb, pb) = rd[a], rd[b]
                frac = pa.intersection(pb).area / min(pa.area, pb.area)
                if frac > OVERLAP_TOL:
                    worst.append((round(f / m.FPS, 2), wa, wb, round(frac, 3)))
    assert worst == [], worst[:10]


def test_readable_words_keep_a_visible_gap():
    # stricter than the overlap tolerance: readable ink boxes never even touch (halo room)
    tight = []
    for f in range(m.N_FRAMES):
        rd = [(m.ITEMS[i].text, _poly(st)) for i in range(len(m.ITEMS))
              if _readable(st := m.word_state(i, f))]
        for a in range(len(rd)):
            for b in range(a + 1, len(rd)):
                d = rd[a][1].distance(rd[b][1])
                if d < MIN_GAP_PX:
                    tight.append((round(f / m.FPS, 2), rd[a][0], rd[b][0], round(d, 1)))
    assert tight == [], tight[:10]


def test_readable_words_never_clipped_by_the_frame():
    cut = []
    for f in range(m.N_FRAMES):
        for i, it in enumerate(m.ITEMS):
            st = m.word_state(i, f)
            if _readable(st):
                l, t, r, b = st.bbox
                if l < 0 or t < 0 or r > m.W or b > m.H:
                    cut.append((round(f / m.FPS, 2), it.text))
    assert cut == [], cut[:10]


def test_no_word_ink_reaches_the_title_block():
    # every word that is drawn at all has its ink box clear of the accent bar
    for f in range(m.N_FRAMES):
        for i, it in enumerate(m.ITEMS):
            st = m.word_state(i, f)
            if st.alpha * m.word_attenuation(st.cx, st.bbox[3]) > 0.0:
                assert st.bbox[3] < m.TITLE_Y - 20, (f, it.text, st.bbox)


@pytest.mark.parametrize("f", [40, 100, 160, 240, 320, 400, 440])
def test_title_area_pixels_carry_no_word_ghosts(f):
    # rows from just above the accent bar down: word layer == background + glow (no words)
    words = m._word_layer_f(f)
    bare = m._word_layer_f(f, words=False)
    y0 = m.TITLE_Y - 12
    assert np.abs(words[y0:] - bare[y0:]).max() <= 2.0


def test_word_lengths_follow_the_card():
    # flying-words pitfall: keep the demo's 4-9 letter words
    for it in m.ITEMS:
        letters = sum(c.isalpha() for c in it.text)
        assert 4 <= letters <= 9, it.text


def test_subtitle_wraps_like_the_intro_card():
    assert m.title_lines() == ("Systems Neuroscience &", "Neuromorphic Engineering Lab")


def test_spread_is_reduced():
    assert m.SPREAD <= 1.3


def test_title_lands_and_holds_at_least_one_second():
    landed = m.title_landed_s()
    fade_start = m.DURATION_S - m.FADE_OUT_S
    assert fade_start - landed >= 1.0


def test_title_text_at_least_32px():
    for size in m.title_font_sizes():
        assert size >= 32


def test_clip_starts_and_ends_on_the_surface():
    surface = np.array(m.rgb(m.SURFACE), dtype=int)
    for f in (0, m.N_FRAMES - 1):
        img = np.asarray(m.render_frame(f)).astype(int)
        assert np.abs(img - surface).max() <= 3, f


# ---------------------------------------------------------------- content + palette
FORBIDDEN = ("drosophila", "melanogaster", "female", "batch", "rig_", "rig ", "trial", "fly1", "fly 1",
             "chandak", "saha")


def test_word_list_size_and_uniqueness():
    words = [it.text for it in m.ITEMS]
    assert 18 <= len(words) <= 26
    assert len(set(w.lower() for w in words)) == len(words)
    assert all(len(w) <= 18 for w in words)


def test_forbidden_strings_absent():
    for s in m.rendered_strings():
        low = s.lower()
        assert not re.search(r"\d", s), s
        for bad in FORBIDDEN:
            assert bad not in low, (bad, s)
    src = MODULE_PATH.read_text().lower()
    for bad in ("drosophila", "melanogaster", "female"):
        assert bad not in src


def test_title_matches_the_lab_card():
    strings = m.rendered_strings()
    assert m.LAB_SHORT in strings
    assert any(m.LAB_FULL.startswith(s) or s in m.LAB_FULL for s in strings if s != m.LAB_SHORT)


def test_palette_is_lab_tokens_only():
    allowed = {m.SURFACE, m.TEXT_1, m.TEXT_2, m.MUTED, m.ACCENT, m.ACCENT_LINK}
    used = m.palette_used()
    assert used <= allowed, used - allowed
    word_cols = [it.color for it in m.ITEMS]
    assert set(word_cols) <= {m.TEXT_1, m.TEXT_2, m.ACCENT, m.ACCENT_LINK}
    reds = sum(c in (m.ACCENT, m.ACCENT_LINK) for c in word_cols)
    assert 2 <= reds <= len(word_cols) // 3          # "a few" in red, the rest neutral
    assert m.GLOW_COLOR in (m.ACCENT, m.ACCENT_LINK)


def test_rendered_frames_have_no_blue_violet():
    for f in (m.N_FRAMES // 4, m.N_FRAMES // 2):
        img = np.asarray(m.render_frame(f)).astype(int)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        assert (b - np.maximum(r, g)).max() <= 2
