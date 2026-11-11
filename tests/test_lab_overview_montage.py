"""Pure-helper tests for scripts/analysis/make_lab_overview_montage.py."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "make_lab_overview_montage.py"
spec = importlib.util.spec_from_file_location("make_lab_overview_montage", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


# ---------------------------------------------------------------- Ken Burns
@pytest.mark.parametrize("src", [(1200, 800), (1448, 1376), (789, 500), (400, 1000)])
@pytest.mark.parametrize("aspect", [1.0, 960 / 536, 16 / 9])
@pytest.mark.parametrize("zoom", [1.0, 1.1, 1.5])
@pytest.mark.parametrize("centre", [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.3, 0.8)])
def test_ken_burns_rect_inside_source_with_output_aspect(src, aspect, zoom, centre):
    l, t, r, b = m.ken_burns_rect(*src, aspect, zoom, *centre)
    assert l >= -1e-9 and t >= -1e-9
    assert r <= src[0] + 1e-9 and b <= src[1] + 1e-9
    assert (r - l) / (b - t) == pytest.approx(aspect)


def test_ken_burns_zoom_one_is_full_cover_crop_and_zoom_two_halves_it():
    l, t, r, b = m.ken_burns_rect(1200, 800, 1.0, 1.0, 0.5, 0.5)
    assert (l, t, r, b) == pytest.approx((200, 0, 1000, 800))
    l2, t2, r2, b2 = m.ken_burns_rect(1200, 800, 1.0, 2.0, 0.5, 0.5)
    assert (r2 - l2, b2 - t2) == pytest.approx((400, 400))
    assert ((l2 + r2) / 2, (t2 + b2) / 2) == pytest.approx((600, 400))


def test_ken_burns_rejects_zoom_below_one():
    with pytest.raises(ValueError):
        m.ken_burns_rect(100, 100, 1.0, 0.9, 0.5, 0.5)


def test_kb_params_endpoints_and_clamping():
    kb = ((1.0, 0.2, 0.3), (1.2, 0.6, 0.7))
    assert m.kb_params(kb, 0.0) == pytest.approx((1.0, 0.2, 0.3))
    assert m.kb_params(kb, 1.0) == pytest.approx((1.2, 0.6, 0.7))
    assert m.kb_params(kb, 0.5) == pytest.approx((1.1, 0.4, 0.5))
    assert m.kb_params(kb, -3) == m.kb_params(kb, 0.0)
    assert m.kb_params(kb, 7) == m.kb_params(kb, 1.0)


def test_cover_assets_never_zoom_below_one_and_cards_stay_inside_box():
    for a in m.ASSETS.values():
        zooms = [a.kb[0][0], a.kb[1][0]]
        if a.mode == "cover":
            assert min(zooms) >= 1.0, a.key
        else:
            assert max(zooms) <= 1.0, a.key   # card only ever shrinks inside the area


def test_fit_contain_preserves_aspect_and_caps_upscale():
    w, h = m.fit_contain(1448, 892, 916, 492)
    assert w <= 916 and h <= 492
    assert w / h == pytest.approx(1448 / 892, rel=0.01)
    assert m.fit_contain(230, 219, 916, 492, max_up=2.0) == (460, 438)


# ---------------------------------------------------------------- timing contract
def test_card_runtime_is_6_to_8_s():
    assert 6.0 <= m.CARD_S <= 8.0


def test_montage_runtime_is_45_to_60_s_and_segments_are_contiguous():
    segs = m.montage_timeline()
    assert 45.0 <= m.montage_duration() <= 60.0
    assert segs[0].start == 0.0
    for a, b in zip(segs, segs[1:]):
        assert b.start == pytest.approx(a.start + a.duration)
    assert [s.name for s in segs] == ["opening", "section01", "section02", "section03", "closing"]


def test_section_order_and_headings():
    assert [s.heading for s in m.SECTIONS] == [
        "Biological Olfaction",
        "Artificial Olfaction",
        "Insect-Based Brain–Machine Interface for Bio-Hybrid Olfaction",
    ]
    for s in m.SECTIONS:
        assert 2 <= len(s.assets) <= 3
        assert all(k in m.ASSETS for k in s.assets)


def test_every_image_has_caption_and_credit_and_publications_cite_journal_year():
    for a in m.ASSETS.values():
        assert a.caption and a.credit and a.url.startswith("https://")
        if not a.credit.startswith("Image: "):
            assert a.credit.count(",") >= 2 and a.credit[-4:].isdigit()


def test_publication_figures_at_most_two_per_theme():
    for s in m.SECTIONS:
        pubs = [k for k in s.assets if not m.ASSETS[k].credit.startswith("Image: ")]
        assert len(pubs) <= 2


def test_frame_count():
    assert m.frame_count(7.0, 40) == 280
    assert m.frame_count(m.montage_duration()) == round(m.montage_duration() * m.FPS)


# ---------------------------------------------------------------- slots / easing
@pytest.mark.parametrize("n,dur", [(1, 5.0), (2, 13.0), (3, 14.0), (3, 15.0)])
def test_image_slots_cover_duration_with_dissolve_overlaps(n, dur):
    d = 0.9
    slots = m.image_slots(n, dur, d)
    assert slots[0][0] == 0.0
    assert slots[-1][1] == pytest.approx(dur)
    for (s0, e0), (s1, e1) in zip(slots, slots[1:]):
        assert e0 - s1 == pytest.approx(d)


def test_slot_weights_sum_to_one_everywhere():
    d = 0.9
    slots = m.image_slots(3, 14.0, d)
    for i in range(0, 1400):
        t = i / 100
        assert sum(m.slot_weights(slots, t, d)) == pytest.approx(1.0, abs=1e-9)


def test_easing_monotone_and_clamped():
    vals = [m.ease_out_cubic(u / 50) for u in range(-10, 61)]
    assert vals[0] == 0.0 and vals[-1] == 1.0
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert m.smoothstep(0.5) == pytest.approx(0.5)
    assert m.reveal(0.0, 1.0, 0.5) == 0.0 and m.reveal(2.0, 1.0, 0.5) == 1.0


def test_segment_fade_starts_and_ends_on_surface():
    assert m.segment_fade(0.0, 7.0, 0.9, 0.8) == 0.0
    assert m.segment_fade(7.0, 7.0, 0.9, 0.8) == 0.0
    assert m.segment_fade(3.5, 7.0, 0.9, 0.8) == 1.0


# ---------------------------------------------------------------- text
def test_wrap_text_greedy_with_fake_measure():
    measure = lambda s: 10 * len(s)
    lines = m.wrap_text("aaa bbb ccc ddd", 75, measure)
    assert lines == ["aaa bbb", "ccc ddd"]
    assert all(measure(s) <= 75 for s in lines)
    assert m.wrap_text("averyveryverylongword x", 50, measure) == ["averyveryverylongword", "x"]
    assert m.wrap_text("", 50, measure) == []


def test_wrap_text_keeps_every_word_in_order():
    text = m.SECTIONS[2].body
    lines = m.wrap_text(text, 400, lambda s: 9 * len(s))
    assert " ".join(lines).split() == text.split()


def test_section_text_fits_above_image_area_with_real_fonts():
    for s in m.SECTIONS:
        lay = m.section_text_layout(s)
        x, y, w, h = m.section_area(s)
        assert y >= lay["bottom"] + 20, s.number
        assert h >= m.MIN_AREA_H, s.number
        assert y + h + m.CAPTION_GAP == m.section_caption_y(s), s.number
        assert len(lay["body"]) <= 4
        assert " ".join(lay["body"]).split() == s.body.split()
        f = m.font(m.BODY_SIZE)
        assert all(f.getlength(x) <= m.W - 2 * m.MARGIN for x in lay["body"])
        hf = m.font(m.HEAD_SIZE, True)
        assert all(hf.getlength(x) <= m.W - 2 * m.MARGIN for x in lay["head"])


def test_section_captions_wrap_whole_and_stay_inside_the_frame():
    f = m.font(m.CAPTION_SIZE)
    for s in m.SECTIONS:
        cy = m.section_caption_y(s)
        for k in s.assets:
            a = m.ASSETS[k]
            lines = m.caption_lines(a)
            assert " ".join(lines).split() == a.caption.split(), k      # never truncated
            assert all(f.getlength(x) <= m.AREA_W - 8 for x in lines), k
            assert f.getlength(a.credit) <= m.AREA_W - 8, k
            n = len(lines) + 1                                            # + credit line
            assert cy + m.caption_block_h(n) <= m.H - m.SAFE_BOTTOM, k
    # a single-line footnote at the shared CAPTION_Y (sibling montages) also clears the edge
    assert m.CAPTION_Y + m.caption_block_h(1) <= m.H - m.SAFE_BOTTOM


def test_wrap_balanced_keeps_line_count_and_words():
    measure = lambda s: 10 * len(s)
    text = "aaaa bbbb cccc dddd eeee ff"
    greedy = m.wrap_text(text, 100, measure)
    bal = m.wrap_balanced(text, 100, measure)
    assert len(bal) == len(greedy) and " ".join(bal).split() == text.split()
    assert max(map(measure, bal)) - min(map(measure, bal)) <= max(map(measure, greedy)) - min(map(measure, greedy))
    assert m.wrap_balanced("one", 100, measure) == ["one"]


def test_caption_block_height():
    assert m.caption_block_h(1) == int(math.ceil(m.CAPTION_SIZE * 1.3))
    assert m.caption_block_h(3) == 2 * m.CAPTION_LINE + int(math.ceil(m.CAPTION_SIZE * 1.3))


def _text_boxes(els):
    """Inked extent (alpha bbox) of each text element, in frame pixels."""
    out = []
    for e in els:
        if isinstance(e, m.TextEl) and e.img.width > 20:
            l, t, r, b = e.img.getchannel("A").getbbox()
            out.append((e.x + l, e.y + t, e.x + r, e.y + b))
    return out


@pytest.mark.parametrize("variant", ["card", "opening", "closing"])
def test_hero_layout_builds_at_the_text_floor_inside_frame_without_overlap(variant):
    els = m.hero_layout(variant)                  # text_image raises for any font < MIN_TEXT_PX
    boxes = _text_boxes(els)
    assert len(boxes) >= 4
    for l, t, r, b in boxes:
        assert l >= 48 and t >= 48 and r <= m.W - 48 and b <= m.H - 120, (variant, l, t, r, b)
    for i, a in enumerate(boxes):
        for c in boxes[i + 1:]:
            overlap = min(a[2], c[2]) > max(a[0], c[0]) and min(a[3], c[3]) > max(a[1], c[1])
            assert not overlap, (variant, a, c)


def test_hero_hierarchy_lab_name_largest():
    for v in ("card", "opening", "closing"):
        sizes = m.hero_font_sizes(v)
        assert min(sizes.values()) >= m.MIN_TEXT_PX
        assert sizes["lab_short"] == max(sizes.values())
        assert sizes["lab_full"] > sizes["university"]


def test_text_image_enforces_the_floor():
    with pytest.raises(ValueError):
        m.text_image(["x"], m.font(m.MIN_TEXT_PX - 1), m.TEXT_1, 40)
    assert m.text_image(["x"], m.font(m.MIN_TEXT_PX), m.TEXT_1, 40).width > 0


def test_section_header_row_does_not_collide():
    num_w = m.font(m.NUMERAL_SIZE, True).getlength("01")
    of_w = m.font(m.CAPTION_SIZE).getlength(f"/ {len(m.SECTIONS):02d}")
    brand_w = m.font(m.CAPTION_SIZE).getlength(m.LAB_SHORT)
    rule_end = m.MARGIN + num_w + of_w + 18 + 90 + 8
    assert rule_end + 24 < m.W - m.MARGIN - brand_w


def test_legibility_minimum_sizes():
    assert m.MIN_TEXT_PX == 32
    assert m.BODY_SIZE >= m.MIN_TEXT_PX and m.CAPTION_SIZE >= m.MIN_TEXT_PX and m.HEAD_SIZE >= 54


def test_palette_tokens_match_sibling_montages_and_site_accent():
    assert (m.SURFACE, m.TEXT_1, m.TEXT_2, m.MUTED) == ("#1a1a19", "#f0efec", "#c3c2b7", "#8a8984")
    assert m.ACCENT == "#a51417" and m.ACCENT_LINK == "#d01c29"
    assert m.W == m.H == 1080


# ---------------------------------------------------------------- review 2026-10-30: dip through the surface
@pytest.mark.parametrize("n,dur", [(2, 13.0), (3, 14.0), (3, 15.0)])
def test_dip_weights_never_show_two_cards_at_once(n, dur):
    d = m.DISSOLVE_S
    slots = m.image_slots(n, dur, d)
    for k in range(int(dur * 200) + 1):
        t = k / 200
        ws = m.dip_weights(slots, t, d)
        assert len(ws) == n and all(0.0 <= w <= 1.0 for w in ws)
        assert sum(w > 0 for w in ws) <= 1, (t, ws)          # outgoing fully gone before incoming starts
    for (s, e), (s1, _) in zip(slots, slots[1:]):
        mid = (s1 + e) / 2                                    # middle of each hand-over: bare surface
        assert sum(m.dip_weights(slots, mid, d)) == pytest.approx(0.0, abs=1e-9)
    for i, (s, e) in enumerate(slots):                        # outside the hand-overs, exactly one card at full
        t = s + d + 0.01 if i else 0.01
        ws = m.dip_weights(slots, t, d)
        assert ws[i] == 1.0 and sum(ws) == 1.0


def test_captions_align_with_heading_and_body():
    assert m.CAPTION_X == m.MARGIN
    f = m.font(m.CAPTION_SIZE)
    for a in m.ASSETS.values():
        assert all(m.CAPTION_X + f.getlength(x) <= m.W - m.MARGIN for x in m.caption_lines(a)), a.key
        assert m.CAPTION_X + f.getlength(a.credit) <= m.W - m.MARGIN, a.key


def _assets_present():
    return all((m.ASSET_DIR / a.rel).exists() for a in m.ASSETS.values())


@pytest.mark.skipif(not _assets_present(), reason="lab image assets not cached")
@pytest.mark.parametrize("sec_i", [0, 1, 2])
def test_section_handover_dips_through_the_surface_with_its_caption(sec_i):
    import numpy as np
    sc = m.SectionScene(m.SECTIONS[sec_i])
    surf = np.array(m.rgb(m.SURFACE), dtype=int)
    x, y, w, h = m.section_area(m.SECTIONS[sec_i])
    for (s, e), (s1, _) in zip(sc.slots, sc.slots[1:]):
        img = np.asarray(sc.frame((s1 + e) / 2)).astype(int)
        # image area and the whole caption/credit band are bare surface at the dip
        assert np.abs(img[y:y + h, x:x + w] - surf).max() <= 2
        assert np.abs(img[sc.cap_y - 4:m.H, :] - surf).max() <= 2
        # a quarter into the hand-over only the outgoing card's caption is up (dimmed, not both)
        q = np.asarray(sc.frame(s1 + m.DISSOLVE_S * 0.2)).astype(int)
        assert np.abs(q[sc.cap_y:m.H, :] - surf).max() > 10
