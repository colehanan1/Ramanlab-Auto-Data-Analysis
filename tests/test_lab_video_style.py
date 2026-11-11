"""Shared lab-montage visual language used by the fly-behaviour montage videos."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("lab_video_style", SCRIPTS / "lab_video_style.py")
st = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = st
spec.loader.exec_module(st)
lab = st.lab


def _rgb(hex_):
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


# ---------------------------------------------------------------- tokens
def test_style_tokens_are_the_lab_montage_tokens():
    assert st.ACCENT == lab.ACCENT == "#a51417"
    assert st.ACCENT_LINK == lab.ACCENT_LINK == "#d01c29"
    assert (st.SURFACE, st.TEXT_1, st.TEXT_2, st.MUTED) == ("#1a1a19", "#f0efec", "#c3c2b7", "#8a8984")
    assert (st.HEAD_SIZE, st.BODY_SIZE, st.CAPTION_SIZE) == (lab.HEAD_SIZE, lab.BODY_SIZE, lab.CAPTION_SIZE)
    assert (st.W, st.H, st.FPS) == (1080, 1080, 40)
    assert st.TextEl is lab.TextEl and st.BarEl is lab.BarEl and st.Motif is lab.Motif


# ---------------------------------------------------------------- numbering
def test_header_numerals_are_zero_padded_index_over_total():
    assert st.header_numerals(1, 4) == ("01", "/ 04")
    assert st.header_numerals(6, 6) == ("06", "/ 06")
    assert st.header_numerals(3, 12) == ("03", "/ 12")
    for bad in ((0, 4), (5, 4), (1, 0)):
        with pytest.raises(ValueError):
            st.header_numerals(*bad)


# ---------------------------------------------------------------- staged reveal
def test_header_reveal_is_the_lab_order_slowed_by_pace():
    lab_t = {"num": 0.10, "of": 0.18, "brand": 0.18, "rule": 0.25, "heading": 0.32, "body": 0.62}
    assert st.LAB_HEADER_T0 == lab_t
    assert 1.5 <= st.PACE <= 2.0
    assert st.HEADER_T0 == {k: pytest.approx(v * st.PACE) for k, v in lab_t.items()}
    assert sorted(st.HEADER_T0, key=st.HEADER_T0.get) == sorted(lab_t, key=lab_t.get)
    assert (st.CONTENT_T0, st.CONTENT_DUR) == (pytest.approx(0.40 * st.PACE), pytest.approx(0.9 * st.PACE))
    assert st.REVEAL_DUR == pytest.approx(0.7 * st.PACE, abs=0.01) and st.RULE_DUR == pytest.approx(0.8 * st.PACE)
    assert st.FADE_OUT_S > st.SEG_FADE_OUT_S and st.HERO_FADE_IN > 0.7
    assert st.NUM_Y == 62 and st.HEAD_Y == 112 and st.HEADER_RULE_W == 90


def test_hero_reveal_times_are_staggered_and_finish_before_the_fade_out():
    tt = st.hero_reveal_times(n_kicker=2, n_items=3)
    order = [tt["rule"], *tt["kicker"], tt["headline"], tt["body"], tt["divider"], *tt["items"]]
    assert all(b > a for a, b in zip(order, order[1:]))
    sp = st.HERO_SPACING
    assert sp == pytest.approx(0.72 * st.PACE)
    assert tt["rule"] == pytest.approx(0.30 * sp) and tt["headline"] == pytest.approx(0.70 * sp)
    assert tt["items"][0] == pytest.approx(1.50 * sp)
    assert tt["items"][-1] + st.REVEAL_DUR < 9.0 - st.HERO_FADE_OUT     # shortest title card is 9 s


# ---------------------------------------------------------------- header layout
@pytest.mark.parametrize("sec", lab.SECTIONS, ids=lambda s: s.number)
def test_segment_header_layout_is_identical_to_lab_section_layout(sec):
    hdr = st.SegmentHeader(1, 3, sec.heading, sec.body)
    lay = lab.section_text_layout(sec)
    assert hdr.head_lines == lay["head"] and hdr.body_lines == lay["body"]
    assert (hdr.head_y, hdr.body_y, hdr.bottom) == (lay["head_y"], lay["body_y"], lay["bottom"])


def test_segment_header_accent_rule_and_numeral_colours():
    hdr = st.SegmentHeader(2, 4, "Training: Hexanol Paired with Light", "Short body.")
    assert (hdr.rule_el.color, hdr.rule_el.w, hdr.rule_el.h) == (st.ACCENT, 90, 2)
    num = np.asarray(hdr.numeral_el.img)
    assert tuple(num[..., :3][num[..., 3] > 0][0]) == _rgb(st.ACCENT_LINK)
    of = np.asarray(hdr.of_el.img)
    assert tuple(of[..., :3][of[..., 3] > 0][0]) == _rgb(st.MUTED)
    brand = np.asarray(hdr.brand_el.img)
    assert tuple(brand[..., :3][brand[..., 3] > 0][0]) == _rgb(st.MUTED)
    assert hdr.brand_el.x + hdr.brand_el.img.width - 4 == st.W - st.MARGIN


def test_segment_header_heading_paragraph_break_is_respected():
    hdr = st.SegmentHeader(5, 6, "Population Summary:\nPre- vs. Post-Exposure", "x")
    assert hdr.head_lines == ["Population Summary:", "Pre- vs. Post-Exposure"]


def test_segment_header_reveals_numeral_before_heading_before_body():
    hdr = st.SegmentHeader(1, 4, "Experimental Protocol", "Flies food-deprived 24 h before training")
    blank = Image.new("RGBA", (st.W, st.H), _rgb(st.SURFACE) + (255,))

    def drawn(t, box):
        fr = blank.copy()
        hdr.draw(fr, t)
        return (np.asarray(fr.crop(box))[..., :3] != np.array(_rgb(st.SURFACE))).any()

    num_box = (st.MARGIN, st.NUM_Y, st.MARGIN + 40, st.NUM_Y + 30)
    head_box = (st.MARGIN, hdr.head_y, st.MARGIN + 300, hdr.head_y + 50)
    body_box = (st.MARGIN, hdr.body_y, st.MARGIN + 300, hdr.body_y + 30)
    t = st.HEADER_T0
    assert not any(drawn(0.0, b) for b in (num_box, head_box, body_box))
    assert drawn(t["num"] + 0.15, num_box) and not drawn(t["num"] + 0.15, head_box)
    assert drawn(t["heading"] + 0.15, head_box) and not drawn(t["heading"] + 0.15, body_box)
    assert all(drawn(3.0, b) for b in (num_box, head_box, body_box))


def test_heading_only_header_has_no_body_and_a_tighter_bottom():
    full = st.SegmentHeader(2, 6, "IAA (Exposed Odor)", "A body line.")
    bare = st.SegmentHeader(2, 6, "IAA (Exposed Odor)")
    assert bare.body_el is None and bare.body_lines == [] and bare.body_el not in bare.els
    assert bare.bottom < full.bottom
    assert [it.text for it in bare.read_items()] == ["IAA (Exposed Odor)"]
    assert bare.read_items()[0].heading


# ---------------------------------------------------------------- reading time
def test_word_count_ignores_symbols_and_counts_numbers():
    assert st.word_count("Hexanol 30 s · light from +5 s") == 7
    assert st.word_count("1.21 → 1.90 ▲") == 2
    assert st.word_count("") == 0


def test_required_hold_and_shortfalls():
    assert (st.READ_S_PER_WORD, st.HEADING_EXTRA_S) == (0.30, 1.0)
    head = st.ReadItem("Experimental Protocol", shown=1.0, heading=True)
    body = st.ReadItem("one two three four five six seven eight nine ten", shown=2.0)
    assert st.required_hold(head) == pytest.approx(1.6) and st.required_hold(body) == pytest.approx(3.0)
    assert st.reading_shortfalls([head, body], duration=5.0 + st.FADE_OUT_S) == []
    short = st.reading_shortfalls([head, body], duration=4.5 + st.FADE_OUT_S)
    assert [it for it, _, _ in short] == [body]


def test_card_reading_rule_counts_every_word_on_the_card_after_it_is_all_up():
    """A card whose elements are all on screen at once is read as a whole: once its last
    element has landed it must hold for the reading time of every word on it."""
    head = st.ReadItem("Experimental Protocol", shown=1.0, heading=True)     # 1.6 s
    body = st.ReadItem("one two three four five six seven eight nine ten", shown=2.0)   # 3.0 s
    assert st.card_reading_time([head, body]) == pytest.approx(4.6)
    assert st.card_hold([head, body], duration=10.0, fade_out=1.0) == pytest.approx(7.0)
    assert st.card_reading_shortfall([head, body], duration=2.0 + 4.6 + 1.0, fade_out=1.0) == pytest.approx(0.0)
    assert st.card_reading_shortfall([head, body], duration=2.0 + 4.0 + 1.0, fade_out=1.0) == pytest.approx(0.6)
    # the per-element rule alone would have passed that short card
    assert st.reading_shortfalls([head, body], duration=2.0 + 4.0 + 1.0, fade_out=1.0) == []


def test_header_and_title_elements_carry_their_words():
    hdr = st.SegmentHeader(1, 4, "Experimental Protocol", "Flies food-deprived 24 h before training")
    items = hdr.read_items()
    assert [i.text for i in items] == ["Experimental Protocol", "Flies food-deprived 24 h before training"]
    assert items[0].shown == pytest.approx(st.HEADER_T0["heading"] + st.REVEAL_DUR)
    els = st.title_els([[("Kicker", False)]], "Headline", ["Body."], [("line", "#199e70", "Odor")])
    assert [i.text for i in st.read_items(els)] == ["Kicker", "Headline", "Body.", "Odor"]


def test_legibility_floor_is_the_lab_caption_size():
    """Secondary text >= 3 % of frame height (32 px at 1080); headings >= 5 % (54 px)."""
    assert st.MIN_TEXT_PX == st.CAPTION_SIZE == 32 >= 0.0295 * st.H
    assert st.MIN_TEXT_PT * 100 / 72 >= st.MIN_TEXT_PX - 1e-9
    assert min(st.BODY_SIZE, st.KICKER_SIZE, st.HERO_BODY_SIZE) >= st.MIN_TEXT_PX
    assert min(st.HEAD_SIZE, st.HERO_HEAD_SIZE) >= 54


def test_text_images_refuse_sizes_below_the_floor():
    st.runs_image([[("ok", False)]], st.MIN_TEXT_PX, st.TEXT_1, st.MIN_TEXT_PX)
    with pytest.raises(ValueError):
        st.runs_image([[("too small", False)]], st.MIN_TEXT_PX - 1, st.TEXT_1, 30)
    with pytest.raises(ValueError):
        lab.text_image(["too small"], lab.font(lab.MIN_TEXT_PX - 1), lab.TEXT_1, 30)


def test_header_meta_text_meets_the_floor():
    hdr = st.SegmentHeader(1, 3, "Heading")
    assert min(hdr.of_el.img.height, hdr.brand_el.img.height) >= int(st.MIN_TEXT_PX * 1.3)


# ---------------------------------------------------------------- captions
def test_no_provenance_helpers_remain():
    assert not hasattr(st, "fly_label") and not hasattr(st, "provenance")
    assert st.SEP == "  ·  "


def test_caption_lines_use_text2_then_muted():
    cap = st.Caption([[("A description line.", False)]], [[("A muted note", False)]], x=62, y=988)
    d = np.asarray(cap.desc_img)
    p = np.asarray(cap.prov_img)
    assert tuple(d[..., :3][d[..., 3] > 0][0]) == _rgb(st.TEXT_2)
    assert tuple(p[..., :3][p[..., 3] > 0][0]) == _rgb(st.MUTED)
    assert cap.prov_y == 988 + st.CAPTION_LINE * 1 + 1


# ---------------------------------------------------------------- rich text
def test_superscript_plus_is_drawn_without_a_missing_glyph_box():
    m = st.run_measure(st.HEAD_SIZE, bold=True)
    assert m("CS") < m("CS⁺") < m("CS+")
    img = st.runs_image([[("⁺", False)]], 52, st.TEXT_1, 62, bold=True)
    a = np.asarray(img)[..., 3]
    ys, xs = np.nonzero(a > 128)
    assert len(ys) > 0
    assert ys.max() < 52 * 0.75          # raised: sits in the upper part of the line


def test_italic_runs_render_in_italic_font():
    up = np.asarray(st.runs_image([[("Drosophila", False)]], st.BODY_SIZE, st.TEXT_2, st.BODY_LINE))[..., 3]
    it = np.asarray(st.runs_image([[("Drosophila", True)]], st.BODY_SIZE, st.TEXT_2, st.BODY_LINE))[..., 3]
    assert up.shape != it.shape or (up != it).any()


# ---------------------------------------------------------------- compositing
def test_compose_reveals_content_after_header_and_fades_to_surface():
    content = np.zeros((st.H, st.W, 4), np.uint8)
    content[..., :3] = 200
    content[..., 3] = 255
    first = st.compose(content, 0.0, 5.0, [])
    mid = st.compose(content, 2.5, 5.0, [])
    last = st.compose(content, 5.0 - 1 / st.FPS, 5.0, [])
    s = np.array(_rgb(st.SURFACE))
    assert first.shape == (st.H, st.W, 4)
    assert (first[..., :3] == s).all()
    assert (mid[..., :3] == 200).all()
    assert np.abs(last[..., :3].astype(int) - s).max() < 20


def test_footage_hero_drifts_and_starts_on_surface():
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 255, size=(10, 300, 300)).astype(np.uint8)
    hero = st.FootageHero(frames, 2.0, [], kb=((1.0, 0.4, 0.5), (1.2, 0.6, 0.5)))
    f0 = np.asarray(hero.frame(0.0))
    assert f0.shape == (st.H, st.W, 3)
    assert (f0 == np.array(_rgb(st.SURFACE))).all()
    b1 = np.asarray(hero.background(0.5))
    b2 = np.asarray(hero.background(1.5))
    assert b1.shape == (st.H, st.W, 4) and (b1 != b2).any()


def test_title_stack_is_left_aligned_and_vertically_centred():
    els = st.title_els([[("Kicker", False)]], "A Headline That Wraps Over Two Lines Here",
                       ["Body paragraph one.", "Two."], [("line", "#199e70", "Odor")])
    xs = {e.x for e in els if isinstance(e, st.TextEl)}
    assert min(xs) >= st.HERO_X - 6
    ys = [e.y for e in els]
    bottom = max(e.y + (e.img.height if isinstance(e, st.TextEl) else e.h) for e in els)
    assert abs((min(ys) + bottom) / 2 - st.H / 2) < 40
    rules = [e for e in els if isinstance(e, st.BarEl)]
    assert rules[0].color == st.ACCENT and rules[0].h == 6
    assert rules[1].color == st.MUTED and rules[1].h == 1


def test_fly_in_slides_horizontally_and_lands_on_x():
    img = Image.new("RGBA", (20, 10), (255, 255, 255, 255))
    el = st.FlyInEl(img, 50, 30, t0=1.0, dur=1.0, dx=-40)
    def col(t):
        f = Image.new("RGBA", (200, 100), (0, 0, 0, 255)); el.draw(f, t)
        a = np.asarray(f)[35, :, 0]
        nz = np.flatnonzero(a > 0)
        return (nz[0] if nz.size else None), a.max()
    assert col(0.5) == (None, 0)                      # not started
    x_mid, v_mid = col(1.3)
    assert x_mid < 50 and 0 < v_mid < 255             # still sliding in from the left, half faded
    assert col(2.5) == (50, 255)                      # landed, fully opaque


def test_fly_in_clips_at_the_frame_edge_instead_of_raising():
    img = Image.new("RGBA", (40, 10), (255, 255, 255, 255))
    f = Image.new("RGBA", (100, 50), (0, 0, 0, 255))
    st.FlyInEl(img, 10, 5, t0=0.0, dur=1.0, dx=-200).draw(f, 0.2)   # starts far off-frame left
    st.FlyInEl(img, 90, 5, t0=0.0, dur=1.0, dx=0).draw(f, 2.0)      # overhangs the right edge
    assert np.asarray(f)[10, 95, 0] > 0


# ---------------------------------------------------------------- 32 px layout (no overlaps)
def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_el_box_is_the_landed_position_of_text_bars_and_fly_ins():
    img = Image.new("RGBA", (20, 10))
    assert st.el_box(st.TextEl(img, 5, 7, 0.0)) == (5, 7, 25, 17)
    assert st.el_box(st.FlyInEl(img, 5, 7, 0.0, dx=-100)) == (5, 7, 25, 17)
    assert st.el_box(st.BarEl(3, 4, 50, 2, st.ACCENT, 0.0)) == (3, 4, 53, 6)


@pytest.mark.parametrize("kicker,head,body,keys", [
    ([[("Gr5a-GAL4 > UAS-CsChrimson", True), ("  ·  Food-deprived 24 h", False)]],
     "Optogenetic Conditioning of the Proboscis Extension Reflex",
     ["Conditioned stimulus (CS): hexanol. Unconditioned stimulus (US): red-light activation of "
      "Gr5a⁺ sugar-sensing neurons via CsChrimson.",
      "After training, the conditioned odor alone elicits proboscis extension."],
     [("line", "#199e70", "Odor (CS)"), ("line", "#3987e5", "Light (US)"), ("line", "#d95926", "Labellum")]),
    ([[("Proboscis Extension Reflex (PER)", False)], [("Gr5a-GAL4 > UAS-CsChrimson", True)]],
     "Odor Experience Bidirectionally Modulates the Proboscis Extension Reflex",
     ["Responses to an odor panel were measured before and after six consecutive presentations "
      "of isoamyl acetate (IAA), without light or reward."],
     [("line", "#199e70", "Odor"), ("line", "#d95926", "Labellum (Tracked)")]),
])
def test_hero_stack_text_never_overlaps_and_stays_in_frame(kicker, head, body, keys):
    els = st.title_els(kicker, head, body, keys)
    texts = [e for e in els if getattr(e, "text", None)]
    boxes = [st.el_box(e) for e in texts]
    for i, a in enumerate(boxes):
        assert a[0] >= st.HERO_X - 6 and a[2] <= st.W - st.HERO_X + 6 and a[1] >= 60 and a[3] <= st.H - 60
        for b in boxes[i + 1:]:
            assert not _overlap(a, b), (texts[i].text, a, b)


def test_segment_header_elements_never_overlap():
    for heading, body in [("Training: Hexanol Paired with Light",
                           "Proboscis extension is elicited at light onset and sustained for the duration of "
                           "illumination."), ("Conclusion", None)]:
        hdr = st.SegmentHeader(1, 3, heading, body)
        boxes = [(type(e).__name__, st.el_box(e)) for e in hdr.els]
        for i, (na, a) in enumerate(boxes):
            assert a[2] <= st.W - st.MARGIN + 6
            for nb, b in boxes[i + 1:]:
                assert not _overlap(a, b), (na, a, nb, b)
        assert max(b[3] for _, b in boxes) <= hdr.bottom + 2
