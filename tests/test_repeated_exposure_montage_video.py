"""Pure helpers and timing contract of the repeated-exposure montage video."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPTS))          # the script does `import make_per_montage_video as base`
SCRIPT = SCRIPTS / "make_repeated_exposure_montage_video.py"
spec = importlib.util.spec_from_file_location("repeated_exposure_montage", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_per_montage_video import FORBIDDEN, assert_sizes_at_the_floor, rendered_strings  # noqa: E402


def test_total_runtime_is_about_a_minute():
    total = mod.total_runtime_s()
    assert 40.0 <= total <= 60.0                     # simple version: title, three pairs, conclusion
    assert total == (mod.TITLE_S + mod.PROTOCOL_S * mod.INCLUDE_PROTOCOL
                     + mod.SUMMARY_S * mod.INCLUDE_SUMMARY + mod.CLOSING_S
                     + sum(p.end_rel - p.start_rel for p in mod.PAIRS))
    assert abs(sum(d for _, d in mod.segment_durations()) - total) < 1e-9
    frames = sum(int(round(d * mod.FPS)) for _, d in mod.segment_durations())
    assert frames == int(round(total * mod.FPS))


def test_pairs_are_the_three_chosen_flies_with_the_exposed_odor_first():
    assert [p.odor for p in mod.PAIRS] == ["IAA", "Hexanol", "Benzaldehyde"]
    assert mod.PAIRS[0].odor == mod.EXPOSED_ODOR == "IAA"
    assert [p.direction for p in mod.PAIRS] == ["up", "up", "down"]
    assert all(abs((p.end_rel - p.start_rel) - 9.0) < 1e-9 for p in mod.PAIRS)


def test_clip_window_frames_cover_exact_duration():
    first, last = mod.clip_window(odor_on_frame=1196, fps=40.02, start_rel=-1.5, end_rel=7.5)
    assert first == round(1196 - 1.5 * 40.02)
    assert last - first + 1 == 9 * 40          # 360 output frames at 40 fps
    f2, l2 = mod.clip_window(1196, 40.02, 4.0, 13.0)
    assert f2 > first and l2 - f2 == last - first


def test_source_frame_index_maps_output_frames_by_time():
    # 40.02 fps source played at 40 fps: output frame 0 -> first, last -> within window
    k0 = mod.source_frame(0, first=1000, fps_src=40.02)
    k_end = mod.source_frame(359, first=1000, fps_src=40.02)
    assert k0 == 1000
    assert 1000 + 358 <= k_end <= 1000 + 360


def test_dumbbell_label_and_glyph():
    assert mod.fmt_change(0.8333, 1.6905) == "0.83 \u2192 1.69"
    assert mod.fmt_change(1.357, 0.976) == "1.36 \u2192 0.98"
    assert mod.fmt_p(0.009382) == "p = 0.009"
    assert mod.direction_glyph(0.83, 1.69) == "\u25b2"
    assert mod.direction_glyph(1.36, 0.98) == "\u25bc"
    assert mod.direction_glyph(1.0, 1.0) == "\u25a0"


def test_cohort_numbers_match_user_correction():
    d = {r.odor: (r.pre, r.post, r.p) for r in mod.COHORT}
    assert d["Hexanol"] == (0.83, 1.69, 0.009)
    assert d["IAA"] == (1.21, 1.90, 0.044)
    assert d["Benzaldehyde"] == (1.36, 0.98, 0.039)
    assert mod.N_FLIES == 42
    assert all(r.p < 0.05 for r in mod.COHORT)
    assert "42 Flies" in mod.SUMMARY_TITLE and "Wilcoxon" in mod.SUMMARY_FOOTNOTE


def test_no_stale_wording_in_script():
    src = SCRIPT.read_text().lower()
    for bad in ("trend", "mcnemar", "% responding", "n = 80", "not yet statistically"):
        assert bad not in src


def test_iaa_example_fly_is_the_october_09_batch_2_rig_3_fly():
    iaa = mod.PAIRS[0]
    assert (iaa.batch, iaa.fly, iaa.before_trial, iaa.after_trial) == (
        "october_09_batch_2_rig_3", "fly1", "pretest_6", "testing_6")
    assert iaa.crop[0] >= 350            # keeps the burned-in top-left odor label out of frame


def test_puffs_light_up_one_after_another_then_stay_lit():
    counts = [mod.puffs_lit(t) for t in np.arange(0, mod.PROTOCOL_S, 0.05)]
    assert counts[0] == 0 and counts[-1] == 6
    assert all(b >= a for a, b in zip(counts, counts[1:]))
    assert set(counts) == set(range(7))
    t_first = mod.PUFF_T0
    assert mod.puffs_lit(t_first - 0.01) == 0 and mod.puffs_lit(t_first + 0.01) == 1
    assert mod.puffs_lit(t_first + mod.PUFF_DT + 0.01) == 2
    assert mod.PUFF_T0 + 5 * mod.PUFF_DT + mod.PUFF_FADE < mod.AFTER_HIGHLIGHT_T < mod.PROTOCOL_S


def test_protocol_and_summary_are_paced_slower():
    # pinned: one puff every 0.8 s (1.6x the previous 0.5 s), longer cards, eased fills
    assert (mod.TITLE_S, mod.PROTOCOL_S, mod.SUMMARY_S) == (10.0, 10.5, 10.0)
    assert (mod.PUFF_T0, mod.PUFF_DT, mod.PUFF_FADE) == (3.0, 0.8, 0.4)
    assert (mod.AFTER_HIGHLIGHT_T, mod.AFTER_HIGHLIGHT_DUR) == (7.7, 0.9)
    assert (mod.BLOCK_T0, mod.BLOCK_DT, mod.BLOCK_DUR) == (0.6, 0.6, 0.7)
    assert (mod.ROW_T0, mod.ROW_DT, mod.ROW_DUR) == (2.0, 1.8, 1.5)
    assert 1.5 <= mod.PUFF_DT / 0.5 <= 2.0 and mod.ROW_DT / 1.3 >= 1.35
    assert mod.puff_level(0, mod.PUFF_T0) == 0.0 and mod.puff_level(0, mod.PUFF_T0 + mod.PUFF_FADE) == 1.0
    hold = mod.PROTOCOL_S - mod.style.FADE_OUT_S - (mod.AFTER_HIGHLIGHT_T + mod.AFTER_HIGHLIGHT_DUR)
    assert hold >= 1.0
    last_row = mod.ROW_T0 + mod.ROW_DT * (len(mod.COHORT) - 1) + mod.LABEL_DELAY + mod.LABEL_DUR
    assert mod.SUMMARY_S - mod.style.FADE_OUT_S - last_row >= 1.5     # whole chart holds before the fade


def test_every_text_element_stays_up_long_enough_to_read():
    st = mod.style
    sched = mod.reading_schedule()
    assert [name for name, *_ in sched] == mod.NAMES == [n for n, _ in mod.segment_durations()]
    for name, dur, fade, items in sched:
        assert any(it.heading for it in items), name
        assert st.reading_shortfalls(items, dur, fade) == [], name
        for it in items:
            hold = dur - fade - it.shown
            assert st.word_count(it.text) / hold <= 1 / st.READ_S_PER_WORD + 1e-9, (name, it.text)


def test_no_identity_provenance_or_boilerplate_text_on_screen():
    texts = rendered_strings(mod, data_ids={"FFMPEG", "FFPROBE"})
    texts += [p.odor for p in mod.PAIRS]
    assert len(texts) > 60
    for t in texts:
        for pat in FORBIDDEN:
            assert not re.search(pat, t, re.I), (pat, t)
    src = SCRIPT.read_text()
    for gone in ("PAIR_FOOTER", "PAIR_SUBLINE", "pair_provenance", "provenance(", "fly_label(", "DATA_ROOT.name"):
        assert gone not in src
    for line in src.splitlines():
        if ".text(" in line:
            assert not re.search(r"\b(trial|spec\.batch|spec\.fly|before_trial|after_trial|DATA_ROOT)\b", line), line


def test_exposed_odor_leads_every_odor_listing():
    assert mod.COHORT[0].odor == "IAA"
    assert mod.ODOR_LABELS["IAA"] == "IAA (Exposed Odor)"
    assert mod.pair_heading(mod.PAIRS[0]) == "IAA (Exposed Odor)"
    assert [mod.pair_heading(p) for p in mod.PAIRS[1:]] == ["Hexanol", "Benzaldehyde"]
    assert mod.PANEL_CHIPS[0] == "IAA"
    assert mod.PAIR_NAMES[0] == "pair_iaa" and mod.NAMES[1] == "pair_iaa"   # first slide after the title
    assert mod.SEGMENT_INDEX["pair_iaa"] == 1


def test_no_text_below_the_caption_floor():
    assert_sizes_at_the_floor(SCRIPT.read_text(), mod.style.MIN_TEXT_PT)
    assert mod.TEXT_PT == mod.base.TEXT_PT == mod.style.MIN_TEXT_PT


def test_contrast_stretch_uses_clip_percentiles_and_maps_to_full_range():
    rng = np.random.default_rng(0)
    frames = rng.integers(20, 90, size=(5, 40, 40)).astype(np.uint8)
    lo, hi = mod.contrast_limits(frames)
    assert 20 <= lo < hi <= 90
    out = mod.stretch(frames[0], lo, hi)
    assert out.dtype == np.uint8 and out.max() >= 200
    # same limits for every frame -> a constant frame stays constant (no per-frame flicker)
    flat = np.full((40, 40), 55, np.uint8)
    assert len(np.unique(mod.stretch(flat, lo, hi))) == 1


def test_conclusion_is_the_single_simple_message():
    # user 2026-10-30: the point of the video, nothing else on the last slide
    assert mod.CLOSING_HEADLINE == "Conclusion"
    assert mod.CLOSING_LINES == [
        "Repeated presentations of a single odor can change a fly\u2019s response to that odor "
        "and to other odors, and not in a uniform way."]
    on_screen = " ".join(it.text for *_, items in mod.reading_schedule() for it in items).lower()
    assert "sensitiz" not in on_screen


def test_no_example_fly_labels_and_final_copy():
    src = SCRIPT.read_text()
    assert "Example fly" not in src and "fly_tag" not in src
    assert mod.PANEL_TITLES == ("Pre-Exposure", "Post-Exposure")
    assert mod.BADGES == {"up": "\u25b2 Increased Response", "down": "\u25bc Decreased Response"}
    assert mod.SUMMARY_TITLE == "Population Summary: Pre- vs. Post-Exposure (n = 42 Flies)"
    assert mod.SUMMARY_SUBTITLE == "Mean PER score \u00b1 SEM; sessions initiated 12:00\u201315:00"
    assert mod.SUMMARY_FOOTNOTE == "Two-sided Wilcoxon signed-rank test, paired by fly."
    assert mod.TITLE_TEXT == "Odor Experience Bidirectionally Modulates the Proboscis Extension Reflex"
    assert mod.FUTURE_BULLETS[0].startswith("Functional imaging of antennal lobe (AL)")
    assert len(mod.FUTURE_BULLETS) == 3                 # kept in code, switched off (INCLUDE_FUTURE)


def test_frame_size_comes_from_base_and_is_square():
    assert (mod.W, mod.H) == (mod.base.W, mod.base.H) == (1080, 1080)
    assert "1920" not in SCRIPT.read_text()


def test_flies_are_identified_exactly_as_in_video_1():
    assert mod.GENOTYPE_RUNS == mod.base.GENOTYPE_RUNS == [("Gr5a-GAL4 > UAS-CsChrimson", True)]
    kicker = "".join(t for t, _ in mod.TITLE_KICKER_RUNS)
    assert kicker == "Proboscis Extension Reflex (PER)"
    title = [it.text for it in mod.reading_schedule()[0][3]]
    assert "Gr5a-GAL4 > UAS-CsChrimson" in title
    others = [it.text for _, _, _, items in mod.reading_schedule()[1:] for it in items]
    assert not any("Gr5a-GAL4" in t for t in others)


# ---------------------------------------------------------------- lab-montage restyle
def test_every_segment_after_the_title_is_numbered_in_order():
    assert mod.NAMES == ["title", "pair_iaa", "pair_hexanol", "pair_benzaldehyde", "closing"]
    assert mod.N_NUMBERED == len(mod.NAMES) - 1 == 4
    assert [mod.SEGMENT_INDEX[n] for n in mod.NAMES[1:]] == [1, 2, 3, 4]
    assert "title" not in mod.SEGMENT_INDEX


def test_title_footage_is_the_iaa_post_exposure_response_without_the_burned_in_label():
    spec = mod.PAIRS[mod.TITLE_PAIR]
    assert spec.odor == "IAA"
    a, b = mod.TITLE_FOOTAGE_REL
    assert abs((b - a) - mod.TITLE_S) < 1e-9 and a < spec.preview_rel < b
    x0, y0, w, h = mod.TITLE_CROP
    assert x0 + w <= mod.FRAME_WH[0] and y0 + h <= mod.FRAME_WH[1]
    assert x0 >= 120 or y0 >= 70
    cx0, cy0, cw, ch = spec.crop
    assert x0 <= cx0 and cx0 + cw <= x0 + w and y0 <= cy0 and cy0 + ch <= y0 + h


def test_style_comes_from_the_shared_lab_module():
    assert mod.style is mod.base.style
    assert (mod.SURFACE, mod.TEXT_2, mod.MUTED) == (mod.style.SURFACE, mod.style.TEXT_2, mod.style.MUTED)
    assert mod.ODOR_C == "#199e70" and mod.LABELLUM_C == "#d95926"


def test_pair_layout_uses_the_freed_caption_band():
    # the x-axis title row is gone ("-4 s ... now" ticks): the scope runs down to 40 px + one
    # 32 px tick-label row above the frame bottom
    assert mod.SCOPE_BOTTOM == mod.H - 40 - 48
    for idx in range(len(mod.PAIRS)):
        assert all(y + h == mod.SCOPE_BOTTOM for _, y, _, h in mod.pair_layout(idx)["scopes"])
    hdr = mod.pair_header(0)
    assert hdr.body_el is None and hdr.bottom < mod.style.HEAD_Y + 80      # heading-only header


@pytest.mark.skipif(not mod.INCLUDE_FUTURE, reason="future directions switched off")
def test_closing_gives_each_point_time_and_flies_it_in():
    # user: part 6 longer, more time to read the points, have them fly in
    assert mod.CLOSING_S == 19.0
    assert mod.FUTURE_RULE_T >= 5.0                    # conclusion gets ~4 s alone before the list starts
    assert mod.BULLET_DT >= 2.4                        # each bullet ~2.4 s before the next arrives
    assert mod.BULLET_T0 + mod.BULLET_DT * (len(mod.FUTURE_BULLETS) - 1) < mod.SCHEMATIC_T
    els, _ = mod.closing_els(mod.closing_header())
    fly = [e for e in els if isinstance(e, mod.style.FlyInEl)]
    assert len(fly) == 2 * len(mod.FUTURE_BULLETS)    # marker + text per bullet
    assert all(e.dx < 0 for e in fly)                  # slide in from the left


def test_results_slide_is_left_out_but_can_be_switched_back_on():
    # user 2026-10-30: drop the population-results slide from the montage
    assert mod.INCLUDE_SUMMARY is False
    assert "summary" not in mod.NAMES and "summary" not in [n for n, _ in mod.segment_durations()]
    assert "summary" not in [name for name, *_ in mod.reading_schedule()]
    assert len(mod.BUILDERS) == len(mod.NAMES)
    assert mod.build_summary not in mod.BUILDERS
    on_screen = " ".join(it.text for *_, items in mod.reading_schedule() for it in items)
    assert "Wilcoxon" not in on_screen and "n = 42" not in on_screen and "p = " not in on_screen


def test_protocol_and_future_directions_are_left_out():
    # user 2026-10-30: "remove experimental summary and the next steps ... keep it simple"
    assert mod.INCLUDE_PROTOCOL is False and mod.INCLUDE_FUTURE is False
    names = [n for n, _ in mod.segment_durations()]
    assert "protocol" not in names and "protocol" not in mod.NAMES
    assert len(mod.BUILDERS) == len(mod.NAMES)
    on_screen = " ".join(it.text for *_, items in mod.reading_schedule() for it in items)
    for gone in ("Experimental Protocol", "Future Directions", "Antennal Lobe", "Functional imaging"):
        assert gone not in on_screen
    els, _ = mod.closing_els(mod.closing_header())
    statement = [e for e in els if getattr(e, "text", None) == mod.CLOSING_LINES[0]]
    assert len(statement) == 1 and isinstance(statement[0], mod.style.FlyInEl)


# ---------------------------------------------------------------- review 2026-10-30: one ODOR pill per column
def test_odor_pill_state_follows_each_clip():
    assert mod.odor_pill_states(-0.5, [2.0, 3.0]) == [False, False]
    assert mod.odor_pill_states(0.0, [2.0, 3.0]) == [True, True]
    assert mod.odor_pill_states(2.5, [2.0, 3.0]) == [False, True]
    assert mod.odor_pill_states(3.5, [2.0, 3.0]) == [False, False]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_each_column_header_carries_its_own_odor_pill_without_collisions(idx):
    import matplotlib.pyplot as plt
    mod.base._setup_fonts()
    fig, h = mod.pair_figure(idx, [2.0, 2.0], [60.0, 40.0])
    try:
        pills = h["odor_pills"]
        assert len(pills) == 2
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        H = fig.bbox.height
        lay = mod.pair_layout(idx)
        labels = [t for t in fig.texts if t.get_text() in mod.PANEL_TITLES]
        assert [t.get_text() for t in labels] == list(mod.PANEL_TITLES)
        for col, (pill, label) in enumerate(zip(pills, labels)):
            assert pill.get_text() in ("ODOR ON", "ODOR OFF")
            pb = pill.get_bbox_patch().get_window_extent(r)
            lb = label.get_window_extent(r)
            x0, x1 = mod.PANEL_X[col], mod.PANEL_X[col] + mod.PANEL_W
            assert x0 <= pb.x0 and pb.x1 <= x1 + 0.5, (col, pb, x0, x1)          # inside its own column
            assert lb.x1 + 24 <= pb.x0, (col, lb, pb)                           # clear of the panel title
            assert abs((H - (pb.y0 + pb.y1) / 2) - lay["label_y"]) <= 3         # on the column-header row
            vid_top = lay["vids"][col][1]
            assert H - pb.y0 <= vid_top - 4                                     # above the video
        hdr_bottom = mod.pair_header(idx).bottom
        for pill in pills:
            assert H - pill.get_bbox_patch().get_window_extent(r).y1 >= hdr_bottom + 4
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- 2026-11-02: live PER % scope under each video
scope = mod.scope                                      # scripts/analysis/lab_scope.py
needs_data = pytest.mark.skipif(not mod.DATA_ROOT.exists(), reason="secured data not mounted")


def test_flies_trials_and_windows_are_unchanged():
    # user 2026-11-02: "the flies you chose are good, don't change them"
    assert [(p.odor, p.direction, p.batch, p.fly, p.crop, p.before_trial, p.after_trial,
             p.start_rel, p.end_rel, p.preview_rel) for p in mod.PAIRS] == [
        ("IAA", "up", "october_09_batch_2_rig_3", "fly1", (560, 0, 340, 280), "pretest_6", "testing_6",
         10.0, 19.0, 14.5),
        ("Hexanol", "up", "october_14_batch_1", "fly2", (330, 330, 340, 280), "pretest_4", "testing_2",
         4.0, 13.0, 9.6),
        ("Benzaldehyde", "down", "october_12_batch_1_rig_3", "fly1", (480, 0, 340, 280), "pretest_2", "testing_2",
         15.0, 24.0, 21.4)]


def test_one_scroll_speed_for_every_column_with_a_four_second_window():
    plots = [r for idx in range(len(mod.PAIRS)) for r in mod.pair_layout(idx)["scopes"]]
    assert len(plots) == 6 and len({(w, h) for _, _, w, h in plots}) == 1     # every scope the same size
    w = plots[0][2]
    assert w == mod.SCOPE_W
    assert 3.5 <= w / mod.SCOPE_SPEED <= 4.5                                   # user: a ~3.5-4.5 s window
    assert mod.SCOPE_SPEED < scope.SPEED_PX_S                                  # slower than the 165 px/s default
    assert mod.SCOPE_WINDOW_S == pytest.approx(w / mod.SCOPE_SPEED)
    # each scope's head (right edge) lines up with its video's right edge
    for idx in range(len(mod.PAIRS)):
        lay = mod.pair_layout(idx)
        for (vx, _, vw, _), (sx, _, sw, _) in zip(lay["vids"], lay["scopes"]):
            assert sx + sw == vx + vw


def test_x_ticks_read_back_from_now_and_y_runs_0_to_100():
    assert mod.scope_x_tick_labels() == ("−4 s", "−3 s", "−2 s", "−1 s", "now")
    lo, hi = mod.SCOPE_YLIM
    assert lo < 0 and hi > 100                                                 # 0-100 % plus a top text lane
    assert mod.SCOPE_YTICKS == (0, 50, 100)
    assert mod.Y_LABEL == scope.Y_LABEL == "Proboscis Extension (%)"
    src = SCRIPT.read_text()
    assert "X_LABEL" not in src and "Time from Odor Onset" not in src         # the "now" ticks replace it
    assert mod.base.X_LABEL == "Time from Odor Onset (s)"                     # base itself is untouched


@pytest.mark.parametrize("fps_src", [39.83, 40.0, 40.02])
def test_no_freeze_every_output_frame_shows_a_new_source_frame(fps_src):
    # user rule 2026-10-30: footage keeps playing through the fade, never a repeated frame
    for spec in mod.PAIRS:
        n = mod._frame_count(spec.end_rel - spec.start_rel)
        src = [mod.source_frame(i, 1000, fps_src) for i in range(n)]
        assert all(b > a for a, b in zip(src, src[1:])), (spec.odor, fps_src)
        heads = [mod.head_rel(i, 1000, 900, fps_src) for i in range(n)]
        assert all(b > a for a, b in zip(heads, heads[1:]))                   # the scope streams to the last frame
        # the head stays on real time: within 2 source frames of the shared clock at the end
        assert abs(heads[-1] - (1000 - 900) / fps_src - (n - 1) / mod.FPS) <= 2 / fps_src
    assert mod.style.FADE_OUT_S < 9.0                                          # the fade is inside the stream


def _trace(thr_hit: bool):
    t = np.arange(-12, 30, 1 / 40)
    v = np.full_like(t, 20.0)
    if thr_hit:
        v[(t > 11) & (t < 14)] = 95.0
    return t, v


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_scope_head_is_pinned_at_the_right_edge_and_both_columns_share_the_clock(idx):
    import matplotlib.pyplot as plt
    mod.base._setup_fonts()
    fig, h = mod.pair_figure(idx, [30.0, 30.0], [60.0, 55.0])
    try:
        panels = h["panels"]
        assert len(panels) == 2
        for p, thr in zip(panels, (60.0, 55.0)):
            sc = p["scope"]
            assert isinstance(sc, scope.ScopePanel)
            assert sc.speed == mod.SCOPE_SPEED and sc.plot_w == mod.SCOPE_W
            assert sc.thr == thr                                               # each side its own threshold
            assert tuple(sc.ax.get_ylim()) == mod.SCOPE_YLIM
            assert tuple(sc.ax.get_yticks()) == mod.SCOPE_YTICKS
            assert sc.thr_line.get_linestyle() == "--"                         # dashed threshold line
            t, v = _trace(True)
            hv = sc.update(t, v, 12.0)
            assert hv == pytest.approx(95.0)
            assert list(sc.dot.get_xdata()) == [mod.SCOPE_W]                   # write head at the right edge
            xs = np.asarray(sc.lo.get_xdata(), float)
            assert np.nanmax(xs) == pytest.approx(mod.SCOPE_W)
            assert np.nanmin(xs) <= 0.0                                        # history fills the window
            assert sc.dot.get_color() == mod.LABELLUM_C                        # orange above threshold
            hv = sc.update(t, v, 20.0)
            assert sc.dot.get_color() != mod.LABELLUM_C                        # back to idle below it
    finally:
        plt.close(fig)


def _boxes(fig):
    """(label, (x0, y0, x1, y1)) top-left px of every visible, non-empty text (pill = its patch)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.bbox.height
    arts = list(fig.texts)
    for ax in fig.axes:
        arts += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        arts += [t.label1 for t in ax.xaxis.get_major_ticks()[:len(ax.get_xticks())]
                 if x0 - 1e-9 <= t.get_loc() <= x1 + 1e-9]
        arts += [t.label1 for t in ax.yaxis.get_major_ticks()[:len(ax.get_yticks())]
                 if y0 - 1e-9 <= t.get_loc() <= y1 + 1e-9]
    out = []
    for a in arts:
        if not a.get_visible() or not a.get_text().strip():
            continue
        patch = a.get_bbox_patch()
        bb = (patch if patch is not None else a).get_window_extent(r)
        out.append((a.get_text(), (bb.x0, H - bb.y1, bb.x1, H - bb.y0)))
    return out


def _overlap(a, b, pad=0.0):
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def test_odor_label_stays_pinned_at_the_left_in_every_clip():
    """Every clip starts with odor onset at (or past) the window's left edge, so the "Odor" band
    label sits at the plot's left end for the whole clip and never rides into the readout."""
    for spec in mod.PAIRS:
        head0 = spec.start_rel - 0.5 / 39.83                                  # first frame, worst rounding
        x0, _ = scope.band_extent(0.0, 30.0, head0, mod.SCOPE_W, mod.SCOPE_SPEED)
        assert scope.sticky_x(x0, mod.SCOPE_W, 80.0, mod.SCOPE_W) <= 12.0, spec.odor


def _real_heads(spec):
    n = mod._frame_count(spec.end_rel - spec.start_rel)
    return (spec.start_rel, spec.preview_rel, spec.start_rel + (n - 1) / mod.FPS)


@pytest.mark.parametrize("idx", [0, 1, 2])
@pytest.mark.parametrize("which", [0, 1, 2])                                 # first frame, response, last frame
def test_no_text_boxes_overlap_with_the_largest_readout(idx, which):
    import matplotlib.pyplot as plt
    mod.base._setup_fonts()
    head = _real_heads(mod.PAIRS[idx])[which]
    fig, h = mod.pair_figure(idx, [30.0, 30.0], [10.0, 10.0])
    try:
        t, v = _trace(True)
        for p in h["panels"]:
            p["scope"].update(t, v, head)
            p["readout"].update(100.0, 10.0)                                 # widest text, full 1.32x scale
            p["clock"].set_text("+24.0 s")
        mod.base.assert_text_floor(fig)
        boxes = _boxes(fig)
        names = [n for n, _ in boxes]
        assert "100" in names and "%" in names and mod.Y_LABEL in names
        for i, (na, a) in enumerate(boxes):
            for nb, b in boxes[i + 1:]:
                if {na, nb} == {"100", "%"}:                                 # value + unit are one readout
                    continue
                assert not _overlap(a, b), (na, a, nb, b)
            assert 0 <= a[0] and a[2] <= mod.W and 0 <= a[1] and a[3] <= mod.H, (na, a)   # nothing clipped
        lay = mod.pair_layout(idx)
        readouts = [b for n, b in boxes if n in ("100", "%")]
        for b in readouts:                                                    # readouts never touch a video
            for vx, vy, vw, vh in lay["vids"]:
                assert not _overlap(b, (vx, vy, vx + vw, vy + vh)), b
            for sx, sy, sw, sh in lay["scopes"]:                              # ... and stay clear of the head halo
                if sx <= b[0] <= sx + sw:
                    assert b[2] <= sx + sw - scope.HALO_R_PX - 6, b
    finally:
        plt.close(fig)


def test_reading_schedule_lists_the_scope_text():
    for name, *_, items in mod.reading_schedule():
        if name.startswith("pair_"):
            texts = [it.text for it in items]
            assert mod.Y_LABEL in texts and "Odor" in texts
            assert "Time from Odor Onset (s)" not in texts and "Labellum Displacement (px)" not in texts


@needs_data
@pytest.mark.parametrize("idx", [0, 1, 2])
def test_scope_plots_per_pct_with_each_sides_own_threshold(idx):
    spec = mod.PAIRS[idx]
    for trial in (spec.before_trial, spec.after_trial):
        meta, df = mod._trial_table(spec, trial)
        tr = mod.scope_trace(meta, df)
        np.testing.assert_array_equal(tr["pct"], scope.per_pct(df["distance_percentage"]))
        assert tr["thr"] == pytest.approx(scope.per_threshold(tr["pct"], tr["rel"]))
        assert tr["odor_len"] == pytest.approx(30.0, abs=0.05)


@needs_data
@pytest.mark.parametrize("idx", [0, 1, 2])
def test_orange_event_lands_only_on_the_badges_responding_side(idx):
    """Over everything the scope ever shows (history included), only the side the badge names
    rises above its threshold: Post-Exposure for an increase, Pre-Exposure for a decrease."""
    spec = mod.PAIRS[idx]
    n = mod._frame_count(spec.end_rel - spec.start_rel)
    hot = []
    for trial in (spec.before_trial, spec.after_trial):
        meta, df = mod._trial_table(spec, trial)
        tr = mod.scope_trace(meta, df)
        fps, on = float(meta["fps"]), int(meta["odor_on_frame"])
        first = mod.clip_window(on, fps, spec.start_rel, spec.end_rel)[0]
        last_head = mod.head_rel(n - 1, first, on, fps)
        shown = (tr["rel"] >= spec.start_rel - mod.SCOPE_WINDOW_S - 0.1) & (tr["rel"] <= last_head)
        hot.append(bool(np.any(tr["pct"][shown] > tr["thr"])))
        if hot[-1]:                                                          # the head itself passes the crossing
            c = scope.first_crossing(tr["pct"], tr["rel"], tr["thr"], after=spec.start_rel - mod.SCOPE_WINDOW_S)
            assert spec.start_rel < c < spec.end_rel, (trial, c)
    assert hot == ([False, True] if spec.direction == "up" else [True, False])
