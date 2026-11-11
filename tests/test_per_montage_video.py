"""Pure helpers and timing contract of the PER montage video."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/make_per_montage_video.py"
spec = importlib.util.spec_from_file_location("per_montage", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
scope = mod.lab_scope


def test_total_runtime_is_about_a_minute():
    total = (mod.TITLE_S + mod.PROTOCOL_S * mod.INCLUDE_PROTOCOL
             + sum(mod.clip_frames(i)[1] / mod.FPS for i in range(len(mod.SEGMENTS))))
    assert total == pytest.approx(mod.total_runtime_s())
    # protocol slide removed (user 2026-10-30); tight odor-onset clip windows, but the CS+ clip
    # runs ~10 s longer to reach its full extension (user 2026-11-02)
    assert 50.0 <= total <= 60.0
    # the encoded video is the concatenation of the segments: frame count = sum of segment frames
    frames = sum(int(round(d * mod.FPS)) for _, d in mod.segment_durations())
    assert frames == int(round(total * mod.FPS))
    # clip durations are their exact encoded frame counts (one per data frame, fade included)
    for (name, d), idx in zip(mod.segment_durations()[1:], range(len(mod.SEGMENTS))):
        assert abs(d * mod.FPS - mod.clip_frames(idx)[1]) < 1e-6, name


def _fade_frames(n_total):
    """Encoded frame indices whose segment fade is below 1 (the fade-out)."""
    st = mod.style
    dur = n_total / mod.FPS
    out = [i for i in range(n_total) if st.segment_fade(i / mod.FPS, dur, 0.0, st.FADE_OUT_S) < 1.0]
    assert len(out) >= int(st.FADE_OUT_S * mod.FPS) - 1
    return out


def test_no_clip_ends_on_a_pause_data_keeps_running_through_the_fade():
    """User rule 2026-10-30: no frozen last frame. Footage, trace, playhead and clock keep
    running in real time while the clip fades out: every encoded frame of the fade shows a new
    source frame (strictly increasing, never two in a row the same)."""
    assert not hasattr(mod, "HOLD_S")
    st = mod.style
    for idx in range(len(mod.SEGMENTS)):
        n_data, n_total = mod.clip_frames(idx)
        seg = mod.SEGMENTS[idx]
        assert n_data == int(round(seg.end_s * mod.SRC_FPS)) - int(round(seg.start_s * mod.SRC_FPS)) + 1
        assert n_total == n_data                                    # no hold frames appended
        src = [mod.source_frame(i, n_data) for i in range(n_total)]
        assert src == list(range(n_data))                           # real time to the last frame
        fade = _fade_frames(n_total)
        assert fade[-1] == n_total - 1
        sf = [src[i] for i in fade]
        assert all(b > a for a, b in zip(sf, sf[1:]))               # strictly increasing in the fade
        assert len(set(sf)) == len(sf)                              # no consecutive repeat
        heads = [mod.head_rel(idx, i) for i in range(n_total)]      # the scope's write head ...
        assert all(b > a for a, b in zip(heads, heads[1:]))         # ... advances on every frame
        assert heads[0] == pytest.approx(seg.start_rel, abs=0.5 / mod.SRC_FPS)
        assert heads[-1] == pytest.approx(seg.end_rel, abs=0.5 / mod.SRC_FPS)
        assert sf[-1] == n_data - 1                                 # the last fade frame is real data
        dur = n_total / mod.FPS
        assert st.segment_fade(dur - 1e-3, dur, 0.0, st.FADE_OUT_S) < 0.05
        with pytest.raises(IndexError):
            mod.source_frame(n_data, n_data)                        # nothing past the data exists


def test_clip_windows_end_inside_the_recorded_trial():
    """The windows reach far enough into the source for real frames through the whole fade."""
    for idx, seg in enumerate(mod.SEGMENTS):
        last = int(round(seg.end_s * mod.SRC_FPS))
        try:
            tr = mod.load_trial(seg.trial)
        except (FileNotFoundError, OSError):
            pytest.skip("trial data not available")
        assert last <= int(tr["frame"].max()), seg.trial
        # the fade (last FADE_OUT_S) plays real frames, not anything past the data
        n_fade = int(round(mod.style.FADE_OUT_S * mod.FPS))
        assert mod.clip_frames(idx)[0] > n_fade


class _FrameSpy:
    """Stands in for the hero footage array and records which frame each call reads."""

    def __init__(self, n):
        self.n, self.read = n, []

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.read.append(int(i))
        return np.zeros((8, 8), np.uint8)


def test_title_footage_keeps_moving_through_the_title_fade_out():
    """The hero card's footage has a real frame for every encoded frame, fade included, and
    advances on every frame of the fade (no frozen last frame)."""
    st = mod.style
    a, b = mod.TITLE_FOOTAGE_S
    n_src = int(round(b * mod.SRC_FPS)) - int(round(a * mod.SRC_FPS)) + 1     # title_footage()'s frames
    spy = _FrameSpy(n_src)
    hero = st.FootageHero(spy, mod.TITLE_S, [], kb=mod.TITLE_KB, src_fps=mod.SRC_FPS)
    n_out = int(round(mod.TITLE_S * mod.FPS))
    fade = [i for i in range(n_out)
            if st.segment_fade(i / mod.FPS, mod.TITLE_S, hero.fade_in, hero.fade_out) < 1.0
            and i / mod.FPS > mod.TITLE_S / 2]
    assert len(fade) >= int(hero.fade_out * mod.FPS) - 1 and fade[-1] == n_out - 1
    for i in fade:
        hero.background(i / mod.FPS)
    assert all(y > x for x, y in zip(spy.read, spy.read[1:]))      # strictly increasing
    # ... because the footage is never clamped onto its last frame
    assert int(round((n_out - 1) / mod.FPS * mod.SRC_FPS)) <= n_src - 1
    # the Ken Burns drift keeps moving too
    zs = [st.kb_params(mod.TITLE_KB, i / mod.FPS / mod.TITLE_S)[0] for i in fade]
    assert all(y > x for x, y in zip(zs, zs[1:]))


def test_extension_is_zero_at_rest_and_ignores_the_odor_window():
    t = np.arange(0, 40, 0.025)
    dist = np.full_like(t, 47.0)
    dist[t >= 30] = 110.0                    # extension once the odor is on
    ext = mod.extension_trace(dist, t, odor_on_s=30.0)
    assert abs(ext[t < 29.9]).max() == 0
    assert np.allclose(ext[t > 30.1], 63.0)


def test_fade_ramps_in_and_out_and_holds_in_between():
    n, f = 100, 10
    a = [mod.fade_alpha(i, n, f) for i in range(n)]
    assert a[0] == 0.1 and a[-1] == 0.1
    assert a[9] == 1.0 and a[50] == 1.0 and a[90] == 1.0
    assert mod.fade_alpha(3, 10, 0) == 1.0


def test_trial_dir_strips_rig_suffix():
    assert mod.trial_dir("testing_1").name == "august_11_batch_1_testing_1"


# ---------------------------------------------------------------- on-screen text
FORBIDDEN = [r"female", r"single fly", r"drosophila", r"yolo26", r"fly ?\d", r"batch", r"pretest_",
             r"testing_", r"training_", r"representative", r"Hex-Training", r"IAA-Sensitivity",
             r"_rig_"]
DATA_IDENTIFIERS = {"FLY", "TITLE_FOOTAGE_TRIAL", "FFMPEG", "FFPROBE"}   # names in code, never drawn


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def rendered_strings(m, data_ids=DATA_IDENTIFIERS):
    """Every string constant that can reach the screen: module-level text constants, the
    segment copy, and every item of the reading schedule (headers, blocks, labels)."""
    out = []
    for name, val in vars(m).items():
        if name.isupper() and name not in data_ids:
            out += list(_strings(val))
    for seg in getattr(m, "SEGMENTS", []):
        out += [seg.step, seg.title, seg.subtitle]
    for _, _, _, items in m.reading_schedule():
        out += [it.text for it in items]
    return out


def test_no_identity_provenance_or_boilerplate_text_on_screen():
    texts = rendered_strings(mod)
    assert len(texts) > 40
    for t in texts:
        for pat in FORBIDDEN:
            assert not re.search(pat, t, re.I), (pat, t)
    src = SCRIPT.read_text()
    assert "provenance(" not in src and "fly_label(" not in src and "SUBHEADER_RUNS" not in src
    assert "BATCH.parent.name" not in src
    for line in src.splitlines():                        # nothing data-derived is drawn as text
        if ".text(" in line:
            assert not re.search(r"\b(seg\.trial|FLY|BATCH|trial_dir)\b", line), line


def test_genotype_is_the_only_fly_identity_and_italic():
    assert mod.GENOTYPE_RUNS == [("Gr5a-GAL4 > UAS-CsChrimson", True)]
    assert "Female" not in mod.PROTOCOL_BODY and "food-deprived 24 h" in mod.PROTOCOL_BODY
    assert "Food-deprived 24 h" in mod.STARVATION_NOTE
    kicker = [it.text for it in mod.reading_schedule()[0][3]]
    assert any(k.startswith("Gr5a-GAL4 > UAS-CsChrimson") for k in kicker)
    # it appears on the title card only
    others = [it.text for _, _, _, items in mod.reading_schedule()[1:] for it in items]
    assert not any("Gr5a-GAL4" in t for t in others)


def test_copy_is_academic_and_titles_are_final():
    assert mod.HEADER_TITLE == "Optogenetic Conditioning of PER (Proboscis Extension Reflex)"
    assert [s.step for s in mod.SEGMENTS] == ["Training", "Test: CS\u207a", "Test: Untrained Odor"]
    assert [s.title for s in mod.SEGMENTS] == [
        "Training: Hexanol Paired with Light",
        "Test: Hexanol Alone (CS\u207a, No Light)",
        "Test: Citral (Untrained Odor)"]
    assert mod.Y_LABEL == "Labellum Displacement (px)"
    assert mod.X_LABEL == "Time from Odor Onset (s)"
    assert mod.PROTOCOL_HEADING == "Experimental Protocol"


def test_pairing_glyphs_light_one_after_another_then_test_block_highlights():
    counts = [mod.pairings_lit(t) for t in np.arange(0, mod.PROTOCOL_S, 0.05)]
    assert counts[0] == 0 and counts[-1] == 6
    assert all(b >= a for a, b in zip(counts, counts[1:]))
    assert mod.pairings_lit(mod.PAIR_T0 - 0.01) == 0 and mod.pairings_lit(mod.PAIR_T0 + 0.01) == 1
    assert mod.pairings_lit(mod.PAIR_T0 + mod.PAIR_DT + 0.01) == 2
    assert mod.PAIR_T0 + 6 * mod.PAIR_DT <= mod.TEST_HIGHLIGHT_T < mod.PROTOCOL_S


def _ngrams(text, n=5):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def test_title_body_is_one_paragraph_and_does_not_repeat_a_clip_caption():
    # review 2026-10-30: the second sentence near-duplicated clip 02's caption -> deleted
    assert "\n" not in mod.TITLE_BODY
    assert "After training" not in mod.TITLE_BODY and "elicits proboscis extension" not in mod.TITLE_BODY
    for seg in mod.SEGMENTS:
        assert not _ngrams(mod.TITLE_BODY, 4) & _ngrams(seg.subtitle, 4), seg.subtitle


def test_title_card_holds_long_enough_to_read_the_whole_card():
    """Review 2026-10-30: the card showed ~43 words but was fully up for ~6.7 s.  Once its last
    element lands, it must hold for 0.3 s per word on the card + 1 s for the headline."""
    st = mod.style
    name, dur, fade, items = mod.reading_schedule()[0]
    assert name == "title" and dur == mod.TITLE_S and fade == st.HERO_FADE_OUT
    words = sum(st.word_count(it.text) for it in items)
    assert words <= 35                                     # second body sentence removed (was 43)
    assert st.card_reading_time(items) == pytest.approx(st.READ_S_PER_WORD * words + st.HEADING_EXTRA_S)
    assert st.card_reading_shortfall(items, dur, fade) == 0.0
    assert st.card_hold(items, dur, fade) <= st.card_reading_time(items) + 1.0   # but no dead air
    # the body itself (the longest element) is fully up for the whole card's reading time
    body = max(items, key=lambda it: st.word_count(it.text))
    assert dur - fade - body.shown >= st.card_reading_time(items)


def test_protocol_card_is_paced_for_reading():
    # pinned: ~1.9x slower pairings than the 0.45 s of the previous cut, longer card and holds
    assert mod.PROTOCOL_S == 12.5
    assert (mod.PAIR_T0, mod.PAIR_DT, mod.GLYPH_FADE) == (3.0, 0.85, 0.4)
    assert (mod.TEST_HIGHLIGHT_T, mod.TEST_HIGHLIGHT_DUR) == (8.6, 1.0)
    assert (mod.BLOCK_T0, mod.BLOCK_DT, mod.BLOCK_DUR) == (0.6, 0.8, 0.7)
    assert 1.5 <= mod.PAIR_DT / 0.45 <= 2.0
    # every glyph is fully lit before the highlight, and the highlight holds >= 2 s before the fade
    assert mod.PAIR_T0 + 5 * mod.PAIR_DT + mod.GLYPH_FADE < mod.TEST_HIGHLIGHT_T
    hold = mod.PROTOCOL_S - mod.style.FADE_OUT_S - (mod.TEST_HIGHLIGHT_T + mod.TEST_HIGHLIGHT_DUR)
    assert hold >= 2.0
    # glyphs ease on (no snapping) and blocks fade in one after another
    assert mod.glyph_level(0, mod.PAIR_T0) == 0.0 and mod.glyph_level(0, mod.PAIR_T0 + mod.GLYPH_FADE) == 1.0
    assert 0 < mod.glyph_level(0, mod.PAIR_T0 + mod.GLYPH_FADE / 2) < 1
    assert mod.block_alpha(1, mod.BLOCK_T0 + mod.BLOCK_DUR) == 0.0 < mod.block_alpha(0, mod.BLOCK_T0 + 0.3)


def test_every_text_element_stays_up_long_enough_to_read():
    """Per slide: each text element, once revealed, stays on screen >= 0.3 s per word
    (+1 s for a heading) before the slide fades out."""
    st = mod.style
    assert (st.READ_S_PER_WORD, st.HEADING_EXTRA_S) == (0.30, 1.0)
    sched = mod.reading_schedule()
    assert [name for name, *_ in sched] == [n for n, _ in mod.segment_durations()]
    for name, dur, fade, items in sched:
        assert any(it.heading for it in items), name
        assert st.reading_shortfalls(items, dur, fade) == [], name
        # slide-level reading rate: no element asks for more than 1 word per 0.3 s of its hold
        for it in items:
            hold = dur - fade - it.shown
            assert st.word_count(it.text) / hold <= 1 / st.READ_S_PER_WORD + 1e-9, (name, it.text)


# font-size arguments that are not literals must be one of these floor-derived names
FLOOR_NAMES = r"(?:TEXT_PT|style\.MIN_TEXT_PT|BLOCK_TITLE_PT|fontsize)(?:\s*\+\s*\d+(?:\.\d+)?)?"


def assert_sizes_at_the_floor(src: str, floor: float) -> None:
    """Every fontsize= / labelsize= in the source (switched-off slides included) is a literal at or
    above the floor or a floor-derived name; none slips through unparsed."""
    args = re.findall(r"(?:fontsize|labelsize)\s*=\s*([^,)\n]+)", src)
    assert len(args) >= 10
    for a in args:
        a = a.strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", a):
            assert float(a) >= floor, a
        else:
            assert re.fullmatch(FLOOR_NAMES, a), a


def test_no_text_below_the_caption_floor():
    """32 px floor: no literal matplotlib font size under MIN_TEXT_PT (23.04 pt = 32 px at dpi 100)."""
    assert mod.TEXT_PT == mod.style.MIN_TEXT_PT and mod.TEXT_PT * 100 / 72 >= mod.style.MIN_TEXT_PX - 1e-9
    assert_sizes_at_the_floor(SCRIPT.read_text(), mod.style.MIN_TEXT_PT)
    fig = mod._new_fig()
    fig.text(0.1, 0.1, "ok", fontsize=mod.TEXT_PT)
    mod.assert_text_floor(fig)
    fig.text(0.2, 0.2, "too small", fontsize=13)
    with pytest.raises(ValueError):
        mod.assert_text_floor(fig)


def test_frame_is_square_1080_and_derived_from_w_h():
    assert mod.W == mod.H == 1080
    assert mod.G * 2 + mod.CONTENT_W == mod.W
    src = SCRIPT.read_text()
    assert "1920" not in src


def test_px_helpers_map_top_left_pixels_to_figure_fractions():
    assert mod.fx(0) == 0.0 and mod.fx(mod.W) == 1.0
    assert mod.fy(0) == 1.0 and mod.fy(mod.H) == 0.0


def test_fit_aspect_keeps_crop_aspect_inside_the_box():
    w, h = mod.fit_aspect(360, 320, 900, 500)
    assert h == 500 and w == 562
    w2, h2 = mod.fit_aspect(340, 280, 460, 600)
    assert w2 == 460 and abs(h2 - 460 * 280 / 340) < 1


def _len_measure(text, italic=False):
    return 10.0 * len(text)


def test_wrap_words_never_exceeds_width_and_keeps_every_word():
    text = "Optogenetic Conditioning of the Proboscis Extension Reflex"
    lines = mod.wrap_words(text, 250, _len_measure)
    assert " ".join(lines) == text
    assert all(_len_measure(l) <= 250 for l in lines)
    balanced = mod.wrap_words(text, 450, _len_measure, balance=True)
    greedy = mod.wrap_words(text, 450, _len_measure)
    assert len(balanced) == len(greedy)
    assert max(map(_len_measure, balanced)) <= max(map(_len_measure, greedy))


RUNS = [("Gr5a-GAL4 > UAS-CsChrimson", True), (mod.SEP + "Odor Panel" + mod.SEP, False),
        ("Hexanol", False), (mod.SEP + "Light Off", False)]


def test_wrap_runs_breaks_only_at_separators_and_keeps_italics():
    lines = mod.wrap_runs(RUNS, 300, _len_measure)
    assert len(lines) >= 2
    joined = mod.SEP.join("".join(t for t, _ in line) for line in lines)
    assert joined == "".join(t for t, _ in RUNS)
    for line in lines:
        assert line[0][0] != mod.SEP and line[-1][0] != mod.SEP
        assert sum(_len_measure(t) for t, _ in line) <= 300
    flat = [r for line in lines for r in line]
    assert ("Gr5a-GAL4 > UAS-CsChrimson", True) in flat
    one = mod.wrap_runs(RUNS, 10_000, _len_measure)
    assert len(one) == 1
    assert "".join(t for t, _ in one[0]) == "".join(t for t, _ in RUNS)


# ---------------------------------------------------------------- lab-montage restyle
def test_segments_are_numbered_after_the_title_card():
    assert mod.N_NUMBERED == len(mod.SEGMENTS) == 3
    assert mod.PROTOCOL_INDEX == 0
    assert [mod.segment_index(i) for i in range(len(mod.SEGMENTS))] == [1, 2, 3]


def test_title_footage_spans_light_onset_and_keeps_burned_in_label_out():
    a, b = mod.TITLE_FOOTAGE_S
    assert abs((b - a) - mod.TITLE_S) < 1e-9
    assert a < 35.06 < b                       # training_4 light onset (meta light_on_s)
    x0, y0, w, h = mod.TITLE_CROP
    assert x0 + w <= mod.FRAME_WH[0] and y0 + h <= mod.FRAME_WH[1]
    assert x0 >= 120 or y0 >= 70               # burned-in odor label: top-left ~110 x 60 px
    cx0, cy0, cw, ch = mod.CROP                # the tracked fly sits inside the hero footage
    assert x0 <= cx0 and cx0 + cw <= x0 + w and y0 <= cy0 and cy0 + ch <= y0 + h


def test_data_colours_unchanged_and_never_the_accent():
    assert (mod.LIGHT_C, mod.ODOR_C, mod.LABELLUM_C) == ("#3987e5", "#199e70", "#d95926")
    assert mod.style.ACCENT == "#a51417" and mod.style.ACCENT_LINK == "#d01c29"
    assert not {mod.style.ACCENT, mod.style.ACCENT_LINK} & {mod.LIGHT_C, mod.ODOR_C, mod.LABELLUM_C}
    assert (mod.SURFACE, mod.TEXT_1, mod.TEXT_2, mod.MUTED) == \
        (mod.style.SURFACE, mod.style.TEXT_1, mod.style.TEXT_2, mod.style.MUTED)


# ---------------------------------------------------------------- stabilisation
def _texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.5)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 200 + 20).astype(np.float32)


def _shaken(ref, shifts, dot=None):
    """Frames whose content is displaced by +d (frame(x) = ref(x - d)); optional bright dot
    that rides on the shaken scene."""
    frames = []
    for d in shifts:
        f = ref.copy()
        if dot is not None:
            cv2.circle(f, dot, 3, 255.0, -1)
        frames.append(mod.align(f, -np.asarray(d)))
    return np.stack(frames)


def test_align_moves_content_by_minus_d():
    img = np.zeros((40, 40), np.float32)
    img[20, 20] = 1.0
    out = mod.align(img, (3.0, -2.0))                 # sample at x + d -> the point moves by -d
    y, x = np.unravel_index(np.argmax(out), out.shape)
    assert (x, y) == (17, 22)


def test_estimate_shifts_recovers_sub_pixel_shake_to_a_quarter_pixel():
    rng = np.random.default_rng(3)
    true = rng.uniform(-3, 3, size=(30, 2))
    ref = _texture(200, 220)
    frames = _shaken(ref, true).clip(0, 255).astype(np.uint8)
    mask = np.ones(ref.shape, bool)
    mask[:14] = mask[-14:] = False
    mask[:, :14] = mask[:, -14:] = False
    est = mod.estimate_shifts(frames, mask)
    err = (est - np.median(est, 0)) - (true - np.median(true, 0))
    assert np.abs(err).max() < 0.25
    # re-aligned, the sequence is still to < 0.25 px
    out = mod.stabilize(frames, est)
    assert out.shape == (30, 200 - 28, 220 - 28) and out.dtype == np.uint8
    inner = np.ones(out.shape[1:], bool)
    resid = mod.estimate_shifts(out, inner & np.pad(np.ones((out.shape[1] - 8, out.shape[2] - 8), bool), 4,
                                                    constant_values=False))
    assert np.abs(resid).max() < 0.25
    rms_before, f2f_before = mod.jitter_stats(true)
    rms_after, f2f_after = mod.jitter_stats(resid)
    assert f2f_after < 0.1 * f2f_before


def test_moving_labellum_outside_the_mask_does_not_drag_the_estimate():
    rng = np.random.default_rng(5)
    true = rng.uniform(-2, 2, size=(20, 2))
    ref = _texture(200, 220, seed=1)
    frames = _shaken(ref, true)
    lab = []
    for i, f in enumerate(frames):                    # a bright blob that swings 30 px on its own
        c = (150 + int(30 * np.sin(i)), 150)
        cv2.circle(f, c, 10, 255.0, -1)
        lab.append(c)
    frames = frames.clip(0, 255).astype(np.uint8)
    mask = mod.stab_mask(ref.shape, (0, 0), eye_xy=(110, 120), lab_xy=np.array(lab, float),
                         roi=(90, 100, 60), lab_pad=12)
    assert not mask[150, 150] and not mask[150, 180] and mask[60, 60]
    est = mod.estimate_shifts(frames, mask)
    err = (est - np.median(est, 0)) - (true - np.median(true, 0))
    assert np.abs(err).max() < 0.25


def test_ring_gets_the_same_shift_as_its_frame():
    """A feature riding on the shaken scene lands at a constant stabilised position when its
    raw (per-frame) coordinates are passed through stabilized_ring with the frame's shift."""
    rng = np.random.default_rng(9)
    true = rng.uniform(-3, 3, size=(25, 2))
    ref = _texture(200, 220, seed=2)
    frames = _shaken(ref, true).clip(0, 255).astype(np.uint8)
    mask = np.ones(ref.shape, bool)
    mask[:14] = mask[-14:] = False
    mask[:, :14] = mask[:, -14:] = False
    est = mod.estimate_shifts(frames, mask)
    raw_xy = np.array([100.0, 90.0]) + true           # where the tracker saw it, frame by frame
    crop_origin = (14, 14)
    ring = np.array([mod.stabilized_ring(xy, crop_origin, d) for xy, d in zip(raw_xy, est)])
    assert ring.std(0).max() < 0.25
    assert mod.stabilized_ring((50.0, 60.0), (10, 20), (1.5, -0.5)) == (38.5, 40.5)


def test_stab_mask_is_anchored_on_the_eye_and_excludes_the_labellum_path_and_border():
    lab = np.array([[160.0, 180.0], [180.0, 200.0], [np.nan, np.nan]])
    m = mod.stab_mask((400, 400), (0, 0), eye_xy=(170, 150), lab_xy=lab)
    half, up, down = mod.STAB_ROI
    assert m[150, 170 - half + 1] and not m[150, 170 - half - 2]           # eye box, left edge
    assert m[150 - up + 1, 100] and not m[150 + down + 1, 60]                # rows above / below
    assert not m[190, 170] and not m[180 - mod.STAB_LAB_PAD + 1, 165]        # labellum path + pad
    assert not m[:mod.STAB_MARGIN].any() and not m[:, :mod.STAB_MARGIN].any()
    with pytest.raises(ValueError):
        mod.stab_mask((400, 400), (0, 0), eye_xy=(-500, -500), lab_xy=lab)


def test_read_window_mirrors_rows_beyond_the_frame(monkeypatch):
    calls = []

    def fake_read(video, first, last, crop):
        calls.append(crop)
        x0, y0, w, h = crop
        for _ in range(first, last + 1):
            yield np.tile(np.arange(y0, y0 + h, dtype=np.uint8)[:, None], (1, w))
    monkeypatch.setattr(mod, "read_clip", fake_read)
    win = mod.read_window(Path("x.mp4"), 0, 2, (560, 0, 340, 280), margin=14)
    assert win.shape == (3, 280 + 28, 340 + 28)
    assert calls == [(546, 0, 368, 294)]
    assert (win[0, 14] == 0).all() and (win[0, 13] == 1).all()             # reflected about the top row


def test_title_footage_window_matches_the_title_card():
    a, b = mod.TITLE_FOOTAGE_S
    assert abs((b - a) - mod.TITLE_S) < 1e-9
    (z0, *_), (z1, *_) = mod.TITLE_KB
    assert (z1 - z0) / mod.TITLE_S < 0.024 / 1.0            # drift slower than the old 0.12 per 5 s


def test_protocol_slide_is_left_out_and_starvation_moves_to_the_title():
    # user 2026-10-30: remove the experimental-protocol slide; the 24 h note stays (title card)
    assert mod.INCLUDE_PROTOCOL is False
    assert "protocol" not in [n for n, _ in mod.segment_durations()]
    kicker = [it.text for it in mod.reading_schedule()[0][3]]
    assert any("Gr5a-GAL4 > UAS-CsChrimson" in k and "Food-deprived 24 h" in k for k in kicker)
    on_screen = " ".join(it.text for *_, items in mod.reading_schedule() for it in items)
    assert "Experimental Protocol" not in on_screen


# ---------------------------------------------------------------- 32 px layout (no overlaps)
# Synthetic stimulus timing per clip (odor on / off, light on / off, s): the layout only
# depends on these, so the check runs without the video data.
# The real trials' timing (_trial_meta.json): in clips 2 and 3 the odor is already on at the
# clip's start, so the odor band spans the whole plot.
FAKE_TIMING = [((30.06, 60.07), (35.06, 60.05), "Hexanol"),
               ((30.08, 60.07), None, "Hexanol"),
               ((30.06, 60.07), None, "Citral")]


def _overlap(a, b, pad=0.0):
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def _segment_boxes(idx):
    odor, light, name = FAKE_TIMING[idx]
    fig, h = mod.segment_figure(idx, odor, light, name)
    h["clock"].set_text("+13.5 s")                         # widest clock string
    mod._set_pill(h["odor_pill"], "ODOR", mod.ODOR_C, False)   # OFF is the wider pill state
    mod._set_pill(h["light_pill"], "LIGHT", mod.LIGHT_C, False)
    h["readout"].update(100.0, h["scope"].thr)            # widest, largest readout: '100 %' at max scale
    boxes = mod.text_boxes(fig)
    return fig, h, boxes


@pytest.mark.parametrize("idx", range(3))
def test_segment_layout_snaps_to_its_header_and_gives_the_footage_the_room(idx):
    lay = mod.segment_layout(idx)
    vx, vy, vw, vh = lay["vid"]
    tx, ty, tw, th = lay["trace"]
    # review 2026-10-30: footage ~520 x 462 at x 60-580, status column from x ~604
    assert vx == mod.G and 515 <= vw <= 525 and vx + vw <= 582
    assert abs(vw / vh - mod.CROP[2] / mod.CROP[3]) < 0.01
    assert vw / mod.CROP[2] <= 1.46                      # upscale of the 360 x 320 crop stays mild
    assert 600 <= lay["col_x"] <= 610
    # the video top snaps to this clip's own header, not the tallest one
    assert vy == mod.clip_header(idx).bottom + mod.VID_GAP
    assert vy + vh < lay["lane_y"] - 18 < ty             # label lane sits between video and trace
    assert ty + th == mod.TRACE_BOTTOM
    assert th >= 100                                     # the trace stays readable
    assert lay["col_x"] + 2 * lay["pill_w"] + mod.PILL_GAP <= mod.W - mod.G   # pills share one row
    # live scope: one speed for every clip and a window of at least 4 s of history
    assert mod.SCOPE_SPEED == scope.SPEED_PX_S == 165
    assert tw / mod.SCOPE_SPEED >= 4.0
    assert mod.segment_layout(0)["trace"][2] == tw       # same plot width (= same window) in every clip
    # the live readout sits right of the write head, clear of its halo and the 'now' label
    assert lay["readout_x"] == mod.W - mod.G
    assert tx + tw + mod.READOUT_GAP + lay["readout_w"] <= mod.W - mod.G + 1e-6
    assert ty < lay["readout_base"] < ty + th


def test_default_segment_layout_is_the_tallest_header_layout():
    """segment_layout() without an index keeps working for importers (tallest header)."""
    tallest = max(range(len(mod.SEGMENTS)), key=lambda i: mod.clip_header(i).bottom)
    assert mod.segment_layout() == mod.segment_layout(tallest)


def test_x_ticks_carry_their_unit():
    assert [mod.x_tick_label(v) for v in (-2.0, 0.0, 2.0, 12.0)] == ["\u22122 s", "0 s", "2 s", "12 s"]


@pytest.mark.parametrize("idx", range(3))
def test_x_tick_labels_keep_clear_gaps(idx):
    fig, h, boxes = _segment_boxes(idx)
    try:
        ticks = sorted((b for n, b in boxes if n == "now" or re.fullmatch("\u2212\\d+ s", n)), key=lambda b: b[0])
        assert len(ticks) >= 4
        assert all(b[0] - a[2] >= 24 for a, b in zip(ticks, ticks[1:]))
        ax = h["ax_t"]                                   # 'now' sits under the write head
        assert ax.get_xticks()[-1] == pytest.approx(h["scope"].plot_w)
    finally:
        mod.plt.close(fig)


@pytest.mark.parametrize("idx", range(3))
def test_no_segment_text_overlaps_or_leaves_the_frame(idx):
    fig, h, boxes = _segment_boxes(idx)
    try:
        mod.assert_text_floor(fig)
        names = [n for n, _ in boxes]
        odor, light, name = FAKE_TIMING[idx]
        for want in ("ODOR OFF", "LIGHT OFF", f"Odor: {name}", mod.LEGEND_LABEL, scope.Y_LABEL,
                     scope.THRESH_LABEL, "Odor", "+13.5 s", "0", "50", "100", "now", "%"):
            assert want in names, want
        assert names.count("100") == 2                   # top y tick + the live readout
        assert mod.X_LABEL not in names and mod.Y_LABEL not in names and "rest" not in names
        assert ("Light" in names) == (idx == 0)
        for na, b in boxes:                              # inside the frame, clear of the edge
            assert b[0] >= 16 and b[2] <= mod.W - 16 and b[1] >= 16, (na, b)
            assert b[3] <= mod.H - mod.BOTTOM_MARGIN + 1, (na, b)     # ~40 px bottom margin
            assert b[0] >= mod.G - 4 and b[2] <= mod.W - mod.G + 4, (na, b)
        for i, (na, a) in enumerate(boxes):
            for nb, b in boxes[i + 1:]:
                assert not _overlap(a, b, pad=4), (na, a, nb, b)
        # header (PIL) above everything matplotlib draws; only the clock sits on the footage
        hdr = mod.clip_header(idx)
        lay = mod.segment_layout(idx)
        vid = mod.rect_box(lay["vid"])
        trace = mod.rect_box(lay["trace"])
        for na, a in boxes:
            assert a[1] > hdr.bottom, na
            if na != "+13.5 s":
                assert not _overlap(a, vid, pad=6), na
        assert not _overlap(vid, trace, pad=20)
        # the band / threshold key sits where the data never goes: outside the plot area
        for na in ("Odor", "Light", scope.THRESH_LABEL):
            for nb, b in boxes:
                if nb == na:
                    assert not _overlap(b, trace, pad=2), (na, b, trace)
        # ... and every other label stays out of the plot area too (the trace streams through it)
        for na, a in boxes:
            assert not _overlap(a, (trace[0] + 1, trace[1] + 1, trace[2] - 1, trace[3] - 1)), na
        # the readout ('100' + '%', at its largest) sits right of the head, clear of its halo
        ro = [b for n, b in boxes if n in ("100", "%") and b[0] > trace[2]]
        assert len(ro) == 2
        for b in ro:
            assert b[0] >= trace[2] + scope.HALO_R_PX + 6 and b[2] <= mod.W - mod.G + 1
            assert trace[1] - 4 <= b[1] and b[3] <= trace[3] + 4
    finally:
        mod.plt.close(fig)


@pytest.mark.parametrize("idx", range(3))
def test_stimulus_bands_scroll_with_the_data_and_stop_at_the_head(idx):
    odor, light, name = FAKE_TIMING[idx]
    fig, h = mod.segment_figure(idx, odor, light, name, thr=12.0)
    try:
        sc = h["scope"]
        bands = sc.band_patches
        assert set(bands) == ({"odor", "light"} if light else {"odor"})
        ylo, yhi = sc.ylim
        for p in bands.values():                          # full plot height
            assert p.get_y() == ylo and p.get_height() == yhi - ylo
        t = np.arange(-12, 20, 1 / 40.02)
        v = np.full_like(t, 3.0)
        sc.update(t, v, -0.5)                             # before odor onset: no band yet
        assert not any(p.get_visible() for p in bands.values())
        sc.update(t, v, 2.0)                              # 2 s into the odor: band = the last 2 s only
        p = bands["odor"]
        assert p.get_visible() and p.get_x() + p.get_width() == pytest.approx(sc.plot_w)
        assert p.get_width() == pytest.approx(2.0 * mod.SCOPE_SPEED)
        if light:
            assert not bands["light"].get_visible()      # light comes on at +5 s
            sc.update(t, v, 6.0)
            q = bands["light"]
            assert q.get_visible() and q.get_width() == pytest.approx((6.0 - (light[0] - odor[0])) * mod.SCOPE_SPEED)
    finally:
        mod.plt.close(fig)


def _step_trace(t_on=4.0, lo=3.0, hi=50.0):
    t = np.arange(-12, 20, 1 / 40.02)
    return t, np.where(t >= t_on, hi, lo)


@pytest.mark.parametrize("idx", range(3))
def test_scope_streams_per_percent_with_the_head_pinned_right(idx):
    """The write head sits at the plot's right edge; history streams left at SCOPE_SPEED px/s;
    above the PER threshold the trace, head dot and readout turn the labellum orange."""
    odor, light, name = FAKE_TIMING[idx]
    fig, h = mod.segment_figure(idx, odor, light, name, thr=12.0)
    try:
        sc, ro = h["scope"], h["readout"]
        assert sc.speed == mod.SCOPE_SPEED and sc.thr == 12.0
        assert sc.ylim == mod.SCOPE_YLIM and tuple(h["ax_t"].get_yticks()) == mod.SCOPE_YTICKS == (0, 50, 100)
        thr_lines = [l for l in h["ax_t"].lines if l is sc.thr_line]
        assert thr_lines and tuple(sc.thr_line.get_ydata()) == (12.0, 12.0)
        assert sc.thr_line.get_color() == mod.LABELLUM_C and sc.thr_line.get_linestyle() == "--"
        t, v = _step_trace()
        hv = sc.update(t, v, 1.0)                         # at rest: grey head, white readout
        ro.update(hv, sc.thr)
        assert hv == pytest.approx(3.0)
        assert sc.dot.get_color() == mod.TEXT_1 and ro.value.get_text() == "3"
        assert ro.value.get_color() == mod.TEXT_1
        x, y = sc.lo.get_data()
        assert x[-1] == pytest.approx(sc.plot_w) and y[-1] == pytest.approx(3.0)
        hv = sc.update(t, v, 5.0)                         # 1 s into a PER at 50 %
        ro.update(hv, sc.thr)
        assert hv == pytest.approx(50.0)
        assert sc.dot.get_color() == mod.LABELLUM_C
        assert ro.value.get_text() == "50" and ro.value.get_color() == mod.LABELLUM_C
        x, y = sc.hi.get_data()
        assert x[-1] == pytest.approx(sc.plot_w) and y[-1] == pytest.approx(50.0)
        # the onset (t = 4 s) sits one second = SCOPE_SPEED px left of the head
        xs = np.asarray(x)[np.isfinite(np.asarray(y, float))]
        assert xs.min() == pytest.approx(sc.plot_w - 1.0 * mod.SCOPE_SPEED, abs=mod.SCOPE_SPEED / 40)
        # ticks read '-N s ... now' and are 1 s apart in px
        labels = [tl.get_text() for tl in h["ax_t"].get_xticklabels()]
        assert labels[-1] == "now" and all(re.fullmatch("\u2212\\d+ s", l) for l in labels[:-1])
        assert np.allclose(np.diff(h["ax_t"].get_xticks()), mod.SCOPE_SPEED)
    finally:
        mod.plt.close(fig)


def test_head_halo_stays_clear_of_the_lane_key_and_the_readout():
    """The head dot's halo (drawn unclipped at the plot's right edge) never reaches the label lane
    at 100 % nor the readout to its right."""
    for idx in range(3):
        lay = mod.segment_layout(idx)
        tx, ty, tw, th = lay["trace"]
        ylo, yhi = mod.SCOPE_YLIM
        y100 = ty + (yhi - 100) / (yhi - ylo) * th
        lane_bottom = lay["lane_y"] + mod.LANE_H / 2
        assert y100 - (scope.HALO_R_PX + 6) >= lane_bottom - 8      # outer (8 % alpha) ring at most grazes
        assert y100 - scope.HALO_R_PX >= lane_bottom                # the visible rings stay clear
        assert tx + tw + scope.HALO_R_PX + 6 < lay["readout_x"] - lay["readout_w"]


@pytest.mark.parametrize("idx", range(3))
def test_real_trace_never_meets_a_label(idx):
    """With the real trial data (skipped without the secured share): every trace sample (and its
    dot) stays inside the plot box, which no label enters."""
    seg = mod.SEGMENTS[idx]
    try:
        tr = mod.load_trial(seg.trial)
    except (FileNotFoundError, OSError):
        pytest.skip("trial data not available")
    fig, h = mod.segment_figure(idx, tr["odor"], tr["light"], tr["odor_name"])
    try:
        rel = tr["t"] - tr["odor"][0]
        sel = (rel >= seg.start_rel - h["scope"].window_s) & (rel <= seg.end_rel)
        ylo, yhi = h["ax_t"].get_ylim()
        v = tr["pct"][sel]
        v = v[np.isfinite(v)]
        assert 0.0 <= v.min() and v.max() <= 100.0                       # PER % never leaves 0-100
        # the head dot (drawn unclipped) stays inside the plot box, give or take the axis line: the
        # x tick labels start a tick + pad (10 px) below it, the lane far above
        th = mod.segment_layout(idx)["trace"][3]
        px = th / (yhi - ylo)
        assert (v.min() - ylo) * px - scope.HEAD_R_PX > -2.0
        assert (yhi - v.max()) * px - scope.HEAD_R_PX > -2.0
    finally:
        mod.plt.close(fig)


def test_pills_switch_state_without_moving_into_each_other():
    fig, h, _ = _segment_boxes(0)
    try:
        for odor_on in (False, True):
            for light_on in (False, True):
                mod._set_pill(h["odor_pill"], "ODOR", mod.ODOR_C, odor_on)
                mod._set_pill(h["light_pill"], "LIGHT", mod.LIGHT_C, light_on)
                b = dict(mod.text_boxes(fig))
                o = b["ODOR ON" if odor_on else "ODOR OFF"]
                l = b["LIGHT ON" if light_on else "LIGHT OFF"]
                assert o[2] + mod.PILL_GAP - 1 <= l[0] and abs(o[1] - l[1]) < 1
    finally:
        mod.plt.close(fig)


def test_trace_ticks_are_sparse_enough_for_32px_labels():
    fig, h, boxes = _segment_boxes(1)
    try:
        ax = h["ax_t"]
        xt = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        assert all(b - a >= mod.SCOPE_SPEED - 1e-9 for a, b in zip(xt, xt[1:]))     # 1 s = 165 px
        assert xt[0] >= mod.X_TICK_EDGE                  # no tick label hanging off the plot's left end
        assert list(ax.get_yticks()) == [0, 50, 100]
    finally:
        mod.plt.close(fig)


def test_text_boxes_and_overlap_helpers():
    fig = mod._new_fig()
    try:
        fig.text(0.1, 0.9, "A", fontsize=mod.TEXT_PT)
        fig.text(0.1, 0.9, "B", fontsize=mod.TEXT_PT)
        boxes = mod.text_boxes(fig)
        assert [n for n, _ in boxes] == ["A", "B"]
        (_, a), (_, b) = boxes
        assert a[1] < 0.1 * mod.H + 5 and a[3] > 0.1 * mod.H      # top-left pixel coordinates
        assert _overlap(a, b)
        assert mod.rect_box((10, 20, 30, 40)) == (10, 20, 40, 60)
    finally:
        mod.plt.close(fig)


def test_protocol_card_layout_fits_when_switched_back_on(monkeypatch):
    monkeypatch.setattr(mod, "PROTOCOL_INDEX", 1)
    monkeypatch.setattr(mod, "N_NUMBERED", 4)
    n, frame, fig = mod.build_protocol()
    try:
        boxes = mod.text_boxes(fig)
        assert len(boxes) == 6
        for i, (na, a) in enumerate(boxes):
            assert a[0] >= mod.G and a[2] <= mod.W - mod.G and a[1] > mod.protocol_header().bottom, na
            for nb, b in boxes[i + 1:]:
                assert not _overlap(a, b, pad=4), (na, nb)
        # every text sits inside its own block
        for (title, sub), (by, bh) in zip(mod.PROTOCOL_TEXTS, mod.BLOCK_YH):
            for na, a in boxes:
                if na in (title, sub) or na.replace("\n", " ") == sub:
                    assert by < a[1] and a[3] < by + bh, na
        assert mod.BLOCK_YH[-1][0] + mod.BLOCK_YH[-1][1] < mod.style.CAPTION_Y - 10
    finally:
        mod.plt.close(fig)


def test_wrapped_headings_are_balanced():
    for i, seg in enumerate(mod.SEGMENTS):
        lines = mod.clip_header(i).head_lines
        assert " ".join(lines) == seg.title
        if len(lines) > 1:
            assert min(len(l) for l in lines) >= 0.5 * max(len(l) for l in lines), lines


# ---------------------------------------------------------------- clip windows (user 2026-11-02)
def _per(seg):
    """(t_rel, PER %, threshold) of a clip's trial; skips without the secured share."""
    try:
        tr = mod.load_trial(seg.trial)
    except (FileNotFoundError, OSError):
        pytest.skip("trial data not available")
    t = tr["t"] - tr["odor"][0]
    return t, tr["pct"], scope.per_threshold(tr["pct"], t), tr


def _runs(t, v, level, t0, t1):
    """(start, end, peak) of every contiguous run above `level` inside [t0, t1]."""
    sel = (t >= t0) & (t <= t1)
    t, v = t[sel], v[sel]
    above = np.r_[False, v > level, False]
    d = np.diff(above.astype(int))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(t[a], t[b], v[a:b + 1].max()) for a, b in zip(starts, ends)]


def test_windows_are_pinned_relative_to_odor_onset():
    """Every clip opens at odor onset (lab_scope.clip_start_rel is 0 for all three trials) and
    the ends are as tight as the data and the reading time allow."""
    assert [(s.trial, s.start_rel, s.end_rel) for s in mod.SEGMENTS] == [
        ("training_4", 0.0, 9.7), ("testing_1", 0.0, 22.6), ("testing_3", 0.0, 9.1)]
    for idx, seg in enumerate(mod.SEGMENTS):
        assert seg.first_frame == seg.odor_on_frame + int(round(seg.start_rel * mod.SRC_FPS))
        assert seg.start_s == pytest.approx(seg.first_frame / mod.SRC_FPS)
        assert mod.clip_frames(idx)[0] == seg.last_frame - seg.first_frame + 1


@pytest.mark.parametrize("idx", range(3))
def test_clip_start_follows_the_odor_onset_rule(idx):
    """Start rule: odor onset, or 1 s before it when the fly moves within the first 2 s of odor."""
    seg = mod.SEGMENTS[idx]
    t, v, thr, tr = _per(seg)
    assert seg.odor_on_frame == tr["meta"]["odor_on_frame"]
    assert seg.start_rel == scope.clip_start_rel(v, t)
    assert seg.start_s == pytest.approx(tr["odor"][0] + seg.start_rel, abs=0.5 / mod.SRC_FPS)


def test_clip_header_is_readable_as_one_block():
    """Heading + description are read as one card once both are up (0.3 s / word + 1 s for the
    heading), like the title card -- and no clip is longer than its data needs beyond that."""
    st = mod.style
    for idx, seg in enumerate(mod.SEGMENTS):
        items = mod.clip_header(idx).read_items()
        dur = mod.clip_frames(idx)[1] / mod.FPS
        assert st.card_reading_shortfall(items, dur, st.FADE_OUT_S) == 0.0, seg.trial
    # training_4 and testing_3 are reading-bound: 0.1 s shorter would cut the header short
    for idx in (0, 2):
        items = mod.clip_header(idx).read_items()
        dur = mod.clip_frames(idx)[1] / mod.FPS
        assert st.card_reading_shortfall(items, dur - 0.1, st.FADE_OUT_S) > 0.0


def test_training_clip_ends_soon_after_a_sustained_extension():
    seg = mod.SEGMENTS[0]
    t, v, thr, _ = _per(seg)
    onset = scope.first_crossing(v, t, thr)
    assert 4.9 <= onset <= 5.3                                    # PER at light onset (+5 s)
    fade = seg.end_rel - mod.style.FADE_OUT_S
    assert 8.5 <= fade <= 9.5                                     # user: fade starts ~+8.5 to +9.5 s
    sel = (t >= onset + 0.3) & (t <= seg.end_rel)
    assert (v[sel] > thr).all()                                   # sustained through the end
    assert fade - onset >= 3.0                                    # >= 3 s of full extension on screen


def test_cs_plus_clip_runs_through_the_full_extension_and_ends_right_after_it():
    """User 2026-11-02: the CS+ clip must show the trial's maximal extension (95 % at +20.2 s),
    not only the small early bouts -- so it plays all four bouts and fades as the proboscis
    retracts after the full extension."""
    seg = mod.SEGMENTS[1]
    t, v, thr, tr = _per(seg)
    fade = seg.end_rel - mod.style.FADE_OUT_S
    bouts = _runs(t, v, thr + 10.0, seg.start_rel, seg.end_rel)   # clear bouts: > threshold + 10 points
    assert len(bouts) >= 4
    assert all(peak_t <= fade for peak_t in
               (t[(t >= a) & (t <= b)][np.argmax(v[(t >= a) & (t <= b)])] for a, b, _ in bouts))
    assert bouts[0][0] == pytest.approx(scope.first_crossing(v, t, thr), abs=0.4)
    assert seg.end_rel <= bouts[-1][1] + 1.2                       # tight: <= 1.2 s after the last bout
    assert max(b[2] for b in bouts) >= 90.0                       # the full extension is on screen


@pytest.mark.parametrize("idx", range(3))
def test_every_clip_shows_its_trials_maximal_extension(idx):
    """User 2026-11-02: wherever a trial reaches a large extension during odor (+3 s), the clip
    includes that peak before its fade begins."""
    seg = mod.SEGMENTS[idx]
    t, v, thr, tr = _per(seg)
    odor_len = tr["odor"][1] - tr["odor"][0]
    trial = (t >= 0) & (t <= odor_len + 3.0)
    peak = np.nanmax(v[trial])
    if peak <= thr:
        return                                                    # no response to show
    shown = (t >= seg.start_rel) & (t <= seg.end_rel - mod.style.FADE_OUT_S)
    assert np.nanmax(v[shown]) >= peak - 1.0


def test_untrained_odor_clip_is_the_shortest_readable_window_past_the_cs_plus_latency():
    seg = mod.SEGMENTS[2]
    t, v, thr, _ = _per(seg)
    assert np.isnan(scope.first_crossing(v[t <= seg.end_rel], t[t <= seg.end_rel], thr))   # no PER
    cs = mod.SEGMENTS[1]
    t1, v1, thr1, _ = _per(cs)
    latency = scope.first_crossing(v1, t1, thr1)
    assert seg.end_rel - mod.style.FADE_OUT_S > latency + 2.0      # well past the CS+ response time
