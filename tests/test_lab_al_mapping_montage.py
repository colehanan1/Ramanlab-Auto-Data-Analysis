"""Tests for scripts/analysis/make_lab_al_mapping_montage.py (locust antennal-lobe montage).

Run with:
    PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_lab_al_mapping_montage.py -q -p no:warnings
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "make_lab_al_mapping_montage.py"
spec = importlib.util.spec_from_file_location("make_lab_al_mapping_montage", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


# ---------------------------------------------------------------- format
def test_output_format_matches_the_lab_videos():
    assert (m.W, m.H, m.FPS) == (1080, 1080, 40)
    assert m.OUT_PATH == Path("/home/ramanlab/Documents/cole/Results/Figures/"
                              "raman_lab_antennal_lobe_mapping_montage.mp4")
    cmd = m.encoder_cmd(Path("x.mp4"))
    assert "libx264" in cmd and "yuv420p" in cmd
    assert "-an" in cmd                                  # silent: never an audio track


def test_lab_tokens_are_imported_not_redefined():
    assert m.SURFACE == "#1a1a19" and m.ACCENT == "#a51417" and m.ACCENT_LINK == "#d01c29"
    assert m.style.READ_S_PER_WORD == 0.3 and m.style.HEADING_EXTRA_S == 1.0
    src = MODULE_PATH.read_text()
    for tok in ("#1a1a19", "#a51417", "#d01c29", "#f0efec", "#c3c2b7"):
        assert tok not in src, f"{tok} must come from the lab modules, not be retyped"


# ---------------------------------------------------------------- text floor
def test_every_text_is_at_least_32_px_and_headings_54():
    inv = m.text_inventory()
    assert inv, "inventory is empty"
    for seg, text, px, role in inv:
        assert px >= 32, (seg, text, px)
        if role in ("heading", "statement"):
            assert px >= 54, (seg, text, px)


def test_inventory_covers_every_tagged_element():
    """Each text element the scenes draw carries its size, so the floor test sees all of them."""
    for name, els in m.all_text_elements().items():
        for el in els:
            if hasattr(el, "img"):
                assert getattr(el, "px", None), (name, el)


def test_key_statement_flies_in_at_54_px_or_more():
    els = m.statement_els(m.section_header(4))
    assert len(els) == 1
    el = els[0]
    assert isinstance(el, m.style.FlyInEl)
    assert el.px >= 54 and el.text == m.COPY["objective"]


def test_statement_wraps_balanced_inside_the_frame():
    hdr = m.section_header(4)
    (el,) = m.statement_els(hdr)
    lines = m.style.wrap_balanced(el.text, m.W - 2 * m.MARGIN, m.style.run_measure(m.STATEMENT_SIZE))
    assert 2 <= len(lines) <= 6
    meas = m.style.run_measure(m.STATEMENT_SIZE)
    widths = [meas(ln) for ln in lines]
    assert max(widths) <= m.W - 2 * m.MARGIN
    assert min(widths[:-1]) >= 0.6 * max(widths)          # balanced: no ragged short line mid-block
    x0, y0, x1, y1 = m.style.el_box(el)
    assert y0 >= hdr.bottom + 40 and y1 <= m.H - m.SAFE_BOTTOM
    assert x0 >= m.MARGIN and x1 <= m.W - m.MARGIN


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_no_text_elements_overlap_and_all_stay_on_frame():
    # 'closing' is the lab's own closing card (make_lab_overview_montage.hero_layout), laid out there.
    for name, els in m.all_text_elements().items():
        if name == "closing":
            continue
        boxes = [(getattr(e, "label", "?"), m.style.el_box(e)) for e in els if hasattr(e, "img")]
        for lab_a, a in boxes:
            assert 0 <= a[0] and a[2] <= m.W and 0 <= a[1] and a[3] <= m.H, (name, lab_a, a)
        for i, (lab_a, a) in enumerate(boxes):
            for lab_b, b in boxes[i + 1:]:
                assert not _overlap(a, b), (name, lab_a, lab_b)


def test_header_text_stays_above_the_footage():
    for name in ("sections", "projection", "local"):
        hdr = m.section_header(m.SECTION_INDEX[name])
        top = min(b[1] for b in m.media_boxes(name))
        assert hdr.bottom < top, name
        assert len(hdr.body_lines) <= 3, (name, hdr.body_lines)
        for _, y, _, h in m.media_boxes(name):
            assert y + h <= m.CAPTION_Y - m.CAPTION_GAP, name


# ---------------------------------------------------------------- reading time
def test_every_text_element_is_held_long_enough_to_read():
    for name, duration, fade_out, items in m.reading_schedule():
        assert items, name
        short = m.style.reading_shortfalls(items, duration, fade_out)
        assert not short, (name, [(it.text, round(h, 2), round(n, 2)) for it, h, n in short])


def test_reading_rule_uses_the_lab_constants():
    it = m.style.ReadItem("one two three four", 0.0, heading=True)
    assert m.style.required_hold(it) == pytest.approx(0.3 * 4 + 1.0)


# ---------------------------------------------------------------- timing
def test_segment_order_and_total_duration():
    segs = m.segments()
    names = ["title", "sections", "projection", "local"]
    names += ["objective"] if m.INCLUDE_OBJECTIVE else []
    names += ["closing"] if m.INCLUDE_CLOSING else []
    assert [s.name for s in segs] == names
    total = sum(s.duration for s in segs)
    assert 45.0 <= total <= 70.0, total
    assert m.frame_count(total) == sum(m.frame_count(s.duration) for s in segs)


def test_no_freeze_logic_remains():
    """User rule (2026-10-30): no video or slide ends on a pause."""
    for name in ("FREEZE_S", "freeze_start"):
        assert not hasattr(m, name), name
    src = MODULE_PATH.read_text()
    assert "freeze" not in src.lower().replace("no freeze", "")
    assert not re.search(r"\bend_frame\b", src)            # (blend_frames is fine)


def test_text_lands_before_the_fade():
    for s in m.segments():
        for el in m.all_text_elements()[s.name]:
            end = el.t0 + getattr(el, "dur", 0.0)
            assert end <= m.fade_start(s.name) - 1.0 + 1e-9, (s.name, getattr(el, "label", el))


FOOTAGE_SEGMENTS = tuple(n for n in ("title", "sections", "projection", "local", "closing")
                         if n in {s.name for s in m.segments()})


def test_objective_slide_is_off_and_the_video_ends_on_the_closing_card():
    """User 2026-11-02: drop 04 / 04 Research Objective, keep the lab closing card at the end."""
    assert m.INCLUDE_OBJECTIVE is False and m.INCLUDE_CLOSING is True
    names = [s.name for s in m.segments()]
    assert "objective" not in names
    assert names[-1] == "closing"
    assert m.seg("objective").duration == m.objective_duration()   # still defined, can be switched back on


def test_numbered_headers_count_only_the_slides_shown():
    """Without the objective the headers read 01 / 03 .. 03 / 03, never '/ 04'."""
    assert m.N_NUMBERED == 3
    for name in ("sections", "projection", "local"):
        hdr = m.section_header(m.SECTION_INDEX[name])
        assert hdr.of_el.label == "/ 03", (name, hdr.of_el.label)


def _fade_frames(name):
    s = m.seg(name)
    n = m.frame_count(s.duration)
    first = int(np.floor(m.fade_start(name) * m.FPS)) - 1          # one frame before the fade begins
    return [i / m.FPS for i in range(first, n)]


def test_every_footage_segment_has_moving_media():
    for name in FOOTAGE_SEGMENTS:
        assert m.segment_clips(name), name
    assert m.segment_clips("objective") == []
    # hero and closing backgrounds both turn slowly (the closing background is no longer a still)
    assert m.segment_clips("title")[0].loop and m.segment_clips("closing")[0].loop


def test_source_position_strictly_increases_on_every_frame_of_every_fade():
    for name in FOOTAGE_SEGMENTS:
        ts = _fade_frames(name)
        for clip in m.segment_clips(name):
            pos = [m.clip_pos(clip, t) for t in ts]
            step = clip.src_fps * clip.speed / m.FPS
            for p0, p1 in zip(pos, pos[1:]):
                if clip.loop:
                    d = (p1 - p0) % clip.n_src                      # wraps 119 -> 0 as an ordinary step
                else:
                    d = p1 - p0
                    assert p1 < clip.n_src - 1, (name, clip.key, p1)   # never clamped on the last plane
                assert d > 0, (name, clip.key, p0, p1)
                assert d == pytest.approx(step), (name, clip.key, d)


def test_playback_rate_is_source_rate_times_speed():
    for s in m.segments():
        for clip in m.segment_clips(s.name):
            a, b = m.clip_pos(clip, clip.t0 + 1.0), m.clip_pos(clip, clip.t0 + 2.0)
            assert (b - a) % clip.n_src == pytest.approx(clip.src_fps * clip.speed)


def test_closing_push_in_keeps_zooming_through_the_fade():
    ts = _fade_frames("closing")
    z = [m.closing_zoom(t) for t in ts]
    assert all(b > a for a, b in zip(z, z[1:]))
    assert m.closing_zoom(0.0) == 1.0
    assert m.closing_zoom(m.seg("closing").duration) == pytest.approx(1.0 + m.CLOSING_ZOOM)


def test_slice_movie_plays_without_looping_and_reaches_its_last_plane_at_the_cut():
    (clip,) = m.segment_clips("sections")
    assert not clip.loop
    s = m.seg("sections")
    last_t = (m.frame_count(s.duration) - 1) / m.FPS
    assert clip.n_src - 2 < m.clip_pos(clip, last_t) < clip.n_src - 1    # still advancing on the very last frame
    assert m.clip_pos(clip, s.duration) == pytest.approx(clip.n_src - 1)
    assert clip.speed <= 1.0                       # real-time or slower, so the tufts can be seen


def test_objective_hold_grows_with_its_word_count(monkeypatch):
    base = m.objective_duration()
    monkeypatch.setitem(m.COPY, "objective", m.COPY["objective"] + " one two three four five six seven eight nine ten")
    longer = m.objective_duration()
    assert longer - base == pytest.approx(10 * m.style.READ_S_PER_WORD, abs=0.1 + 1e-9)


# ---------------------------------------------------------------- blending
def test_blend_endpoints_equal_source_frames():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (5, 4, 6, 3), dtype=np.uint8)
    for i in range(5):
        out = m.blend_frames(frames, float(i), loop=False)
        assert out.dtype == np.uint8
        assert np.array_equal(out, frames[i])
    assert np.array_equal(m.blend_frames(frames, 4.0, loop=True), frames[4])
    assert np.array_equal(m.blend_frames(frames, 5.0, loop=True), frames[0])     # wraps


def test_blend_midpoint_is_the_mean_and_loop_wraps():
    a = np.zeros((2, 2, 3), np.uint8)
    b = np.full((2, 2, 3), 200, np.uint8)
    frames = np.stack([a, b])
    assert np.array_equal(m.blend_frames(frames, 0.5, loop=False), np.full((2, 2, 3), 100, np.uint8))
    assert np.array_equal(m.blend_frames(frames, 0.25, loop=False), np.full((2, 2, 3), 50, np.uint8))
    assert np.array_equal(m.blend_frames(frames, 1.5, loop=True), np.full((2, 2, 3), 100, np.uint8))
    # without looping the last frame holds past the end
    assert np.array_equal(m.blend_frames(frames, 3.7, loop=False), b)


# ---------------------------------------------------------------- brightness lift
def test_lift_is_linear_and_clipped():
    x = np.arange(256, dtype=np.uint8)
    y = m.lift(x, 6, 70)
    assert y.dtype == np.uint8
    assert y[:7].max() == 0 and y[70:].min() == 255          # clipped at both ends
    inside = np.arange(6, 71)
    expect = (inside - 6) * 255.0 / 64.0
    assert np.abs(y[inside].astype(float) - expect).max() <= 0.5   # affine, rounding only
    steps = np.diff(y[inside].astype(int))
    assert set(steps.tolist()) <= {3, 4, 5}                   # constant slope ~3.98, no gamma curve


def test_lift_applies_the_same_map_to_every_channel():
    rgb = np.stack([np.arange(256, dtype=np.uint8)] * 3, -1)
    y = m.lift(rgb, 10, 200)
    assert np.array_equal(y[:, 0], y[:, 1]) and np.array_equal(y[:, 1], y[:, 2])


def test_lift_rejects_a_non_increasing_range():
    with pytest.raises(ValueError):
        m.lift(np.zeros(3, np.uint8), 50, 50)


def test_every_movie_uses_a_linear_lift():
    for key, mv in m.MOVIES.items():
        assert 0 <= mv.black < mv.white <= 255, key


# ---------------------------------------------------------------- scale bar
def test_scale_bar_keeps_the_burned_in_calibration():
    # measured in the 40x source: horizontal and vertical arms both 176 px = 100 um
    assert m.SCALE_BAR_SRC_PX == 176 and m.SCALE_BAR_UM == 100
    mv = m.MOVIES[m.SECTION_MEDIA["local"][0]]
    box = m.media_boxes("local")[0]
    s = m.display_scale(mv, box)
    assert m.scale_bar_px(s) == round(176 * s)
    bar = m.scale_bar_el()
    assert bar.bar_px == m.scale_bar_px(s)
    x0, y0, x1, y1 = m.style.el_box(bar)                # inset inside the card, clear of the caption row
    assert box[0] <= x0 and x1 <= box[0] + box[2] and box[1] <= y0 and y1 <= box[1] + box[3]
    assert y1 <= m.CAPTION_Y
    # the crop excludes the tiny burned-in bar + label (rows 753-946 of the source)
    l, t, r, b = mv.crop
    assert b <= 753
    assert m.COPY["scale"] == "100 µm"


# ---------------------------------------------------------------- copy
FORBIDDEN = [r"drosophila", r"melanogaster", r"\.avi", r"composite", r"slicemov", r"\b\d+\s*x\b",
             r"moesm", r"sectioning", r"standard", r"axonless", r"gaba"]


def test_no_forbidden_strings_on_screen():
    for s in m.onscreen_strings():
        for pat in FORBIDDEN + m.FORBIDDEN_PATTERNS:
            assert not re.search(pat, s, re.I), (pat, s)


def test_no_counts_of_neurons_or_glomeruli():
    for s in m.onscreen_strings():
        assert not re.search(r"\d+\s*(glomerul|neuron|cell|PN|LN)", s, re.I), s
        assert not re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred)s?\s+"
                             r"(glomerul|neuron)", s, re.I), s


def test_copy_matches_the_fact_checked_wording():
    c = m.COPY
    assert c["headline"] == "Mapping the Locust Antennal Lobe"
    assert c["title_line"] == ("Is the spatial arrangement of antennal lobe glomeruli organized by odor "
                               "chemical identity or by odor valence?")
    assert c["sections_heading"] == "Optical Sections"
    assert c["sections_body"] == ("Serial confocal sections through the antennal lobe reveal the glomeruli "
                                  "innervated by a single filled neuron.")
    assert c["sections_caption"] == "Serial optical sections through the antennal lobe"
    assert c["projection_heading"] == "Projection Neurons"
    assert c["projection_body"] == ("Projection neurons innervate multiple glomeruli and send axons to the "
                                    "mushroom body calyces and the lateral protocerebrum.")
    assert c["projection_caption"] == "Rotating 3D projections of filled projection neurons"
    assert c["local_heading"] == "Local Neurons"
    assert c["local_body"] == ("Local neurons arborize extensively throughout the antennal lobe, "
                               "interconnecting many glomeruli.")
    assert c["local_caption"] == "Rotating 3D projection of a filled local neuron"
    assert c["objective_heading"] == "Research Objective"
    assert c["objective"] == ("Map the glomerular innervation of individual antennal lobe neurons and "
                              "determine whether its spatial organization reflects odor chemical identity "
                              "or behavioral valence.")


def test_lobe_is_always_the_antennal_lobe():
    """Formal wording: never the bare 'lobe' on screen, always 'antennal lobe'."""
    for s in m.onscreen_strings():
        for hit in re.finditer(r"\blobes?\b", s, re.I):
            assert re.search(r"antennal\s+$", s[:hit.start()], re.I), s


def test_lobe_rule_catches_the_bare_word():
    pat = r"\blobes?\b"
    for bad in ("Imaging through the lobe", "Lobe structure"):
        hit = re.search(pat, bad, re.I)
        assert hit and not re.search(r"antennal\s+$", bad[:hit.start()], re.I)


def test_copy_is_centralized_in_one_block():
    """Every on-screen string comes from COPY (or the lab's own name/URL tokens), and no
    copy string is typed anywhere else in the module."""
    lab_tokens = {m.lab.LAB_SHORT, m.lab.LAB_FULL, m.lab.UNIVERSITY, m.lab.URL}
    allowed = set(m.COPY.values()) | lab_tokens
    for s in m.onscreen_strings():
        if re.fullmatch(r"(\d\d|/ \d\d)", s):           # header numerals are generated
            continue
        assert s in allowed, s
    src = MODULE_PATH.read_text()
    for key, val in m.COPY.items():
        # the string literal appears once (in COPY); long strings may be split across lines
        head = val[:24]
        assert src.count(head) == 1, (key, src.count(head))


def test_cell_type_labels_live_in_one_swappable_block():
    types = {mv.cell_type for mv in m.MOVIES.values()}
    assert types <= set(m.CELL_TYPES)
    for sec, keys in m.SECTION_MEDIA.items():
        for k in keys:
            assert k in m.MOVIES
    # sections pick their movies by cell type, so swapping a movie's label moves it
    assert all(m.MOVIES[k].cell_type == "PN" for k in m.SECTION_MEDIA["projection"])
    assert all(m.MOVIES[k].cell_type == "LN" for k in m.SECTION_MEDIA["local"])


def test_movie_sources_are_read_from_the_asset_copy_not_downloads():
    assert "Downloads" not in str(m.ASSET_DIR)
    assert m.ASSET_DIR.parent == m.lab.ASSET_DIR.parent
    for mv in m.MOVIES.values():
        assert " " not in mv.file


# ---------------------------------------------------------------- rendered frames (needs the asset copy)
needs_assets = pytest.mark.skipif(not all((m.ASSET_DIR / mv.file).exists() for mv in m.MOVIES.values()),
                                  reason="movie asset copy not present")


@pytest.fixture(scope="module")
def scenes():
    return {s.name: m.build_scene(s.name) for s in m.segments()}


def _footage_region(name):
    if name in ("title", "closing"):
        return (0, 0, m.W, m.H)
    xs = [b[0] for b in m.media_boxes(name)] + [b[0] + b[2] for b in m.media_boxes(name)]
    ys = [b[1] for b in m.media_boxes(name)] + [b[1] + b[3] for b in m.media_boxes(name)]
    return (min(xs), min(ys), max(xs), max(ys))


@needs_assets
def test_rendered_footage_keeps_moving_through_the_fade(scenes):
    """Consecutive frames at the start of each fade differ inside the footage region (no held frame)."""
    for name in FOOTAGE_SEGMENTS:
        sc = scenes[name]
        x0, y0, x1, y1 = _footage_region(name)
        fs = m.fade_start(name)
        for t in (fs - 1.0 / m.FPS, fs + 0.1, fs + 0.3):
            a = np.asarray(sc.frame(t))[y0:y1, x0:x1].astype(int)
            b = np.asarray(sc.frame(t + 1.0 / m.FPS))[y0:y1, x0:x1].astype(int)
            assert np.abs(a - b).mean() > 0.0, (name, t)


@needs_assets
def test_rendered_frames_end_on_the_surface(scenes):
    for s in m.segments():
        sc = scenes[s.name]
        assert np.asarray(sc.frame(1.0)).shape == (1080, 1080, 3)
        last = np.asarray(sc.frame((m.frame_count(s.duration) - 1) / m.FPS)).astype(int)
        surf = np.array(m.rgb(m.SURFACE))
        assert np.abs(last - surf).max() <= 12, s.name          # ends (nearly) on the bare surface


@needs_assets
def test_rendering_is_deterministic(scenes):
    sc = scenes["projection"]
    assert np.array_equal(np.asarray(sc.frame(5.3)), np.asarray(m.build_scene("projection").frame(5.3)))
