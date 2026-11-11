"""Pure helpers, timing and text contract of the PER behaviour-tracking oscilloscope clips."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/make_per_tracking_scope_clip.py"
spec = importlib.util.spec_from_file_location("per_tracking_scope", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

KEYS = ("training", "cs_plus_test", "untrained_odor_test")


# ------------------------------------------------------------------ ported recipe parameters
def test_recipe_parameters_are_the_tuned_demo_values():
    # shot card: "5.5 already perceptible, 8 clearer" -> the documented lower bound, 5.5 px/f at
    # 30 fps (165 px/s), shared by all three clips so the time axis is identical across the cuts
    assert mod.SPEED_PX_S == pytest.approx(5.5 * 30)
    assert mod.SPEED_PX_S / mod.FPS == pytest.approx(4.125)
    assert mod.WINDOW_S == pytest.approx(840 / 165)
    assert mod.HEAD_R_PX == 10.5 and mod.TAIL_PX == 160 and mod.HALO_R_PX == 24
    assert mod.READOUT_GAIN == 0.32
    # user rule 2026-10-30: never end on a pause -- the demo's brake and still hold are gone
    for gone in ("BRAKE_S", "BRAKE_FRACTION", "HOLD_S", "out_cubic"):
        assert not hasattr(mod, gone), gone
    assert mod.WINDOW_S == pytest.approx(mod.PLOT_W / mod.SPEED_PX_S)
    assert (mod.W, mod.H, mod.FPS) == (1080, 1080, 40)


# ------------------------------------------------------------------ pure helpers
def test_stream_offset_pins_the_head_to_the_right_edge_and_streams_left():
    assert mod.stream_x(5.0, 5.0) == pytest.approx(mod.PLOT_W)
    assert mod.stream_x(4.0, 5.0) == pytest.approx(mod.PLOT_W - mod.SPEED_PX_S)
    assert mod.stream_x(5.0 - mod.WINDOW_S, 5.0) == pytest.approx(0.0)
    # one output frame later every sample has moved left by SPEED / FPS px
    xs = np.array([1.0, 2.0, 3.0])
    d = mod.stream_x(xs, 3.0) - mod.stream_x(xs, 3.0 + 1 / mod.FPS)
    assert np.allclose(d, mod.SPEED_PX_S / mod.FPS)


def test_timeline_eases_in_still_then_streams_at_1x_to_the_last_frame():
    """Still while the content eases in, then 1x real time to the very last frame: the stream
    keeps writing at constant speed through the fade-out (no brake, no hold)."""
    tl = mod.Timeline(end_rel=4.6)
    ht = np.array([tl.head_time(i) for i in range(tl.n_frames)])
    assert np.all(ht[: tl.stream_start + 1] == ht[0])
    assert ht[0] == pytest.approx(-mod.LEAD_S, abs=0.5 / mod.FPS)
    assert np.allclose(np.diff(ht[tl.stream_start:]), 1 / mod.FPS)     # constant speed to the end
    assert ht[tl.fade_start] == pytest.approx(4.6)                    # end_rel = where the fade starts
    assert ht[-1] == pytest.approx(4.6 + (tl.n_fade - 1) / mod.FPS) == pytest.approx(tl.last_rel)
    assert tl.n_frames == tl.fade_start + tl.n_fade
    assert not hasattr(tl, "glow") and not hasattr(tl, "frozen")


def _fade_frames(tl):
    st = mod.style
    out = [i for i in range(tl.n_frames) if st.segment_fade(i / mod.FPS, tl.duration, 0.0, st.FADE_OUT_S) < 1.0]
    assert out[0] >= tl.fade_start - 1 and out[-1] == tl.n_frames - 1
    assert len(out) >= tl.n_fade - 1
    return out


def test_stream_and_footage_keep_moving_through_every_frame_of_the_fade():
    """No frozen last frame: across the fade the head advances 1/FPS per frame and the footage
    shows a new source frame every frame (strictly increasing, never a repeat)."""
    odor_on, fps_src = 30.0599700149925, 40.02
    for key in KEYS:
        tl = mod.timeline(key)
        fade = _fade_frames(tl)
        ht = np.array([tl.head_time(i) for i in fade])
        assert np.allclose(np.diff(ht), 1 / mod.FPS)
        q = mod.footage_positions(tl, odor_on, fps_src)
        qf = q[fade]
        assert np.all(np.diff(qf) >= 1)                                  # strictly increasing
        assert len(set(qf.tolist())) == len(qf)                          # no consecutive repeat
        # the streaming part is the same: one new frame per output frame after the still intro
        assert np.all(np.diff(q[tl.stream_start:]) >= 1)


def test_head_glow_and_tail_stay_lit_through_the_fade():
    """update_scope draws the head dot, halo and afterglow tail at full strength on every frame
    (the global lab fade dims them with everything else)."""
    t = np.arange(-2, 10, 1 / 40.02)
    tr = dict(thr=10.0, t_rel=t, ext=np.where(t > 1, 40.0, 0.0), odor_off_rel=30.0, light_rel=None)
    fig, art = mod.build_figure("cs_plus_test", "Hexanol", False, 10.0)
    try:
        tl = mod.timeline("cs_plus_test")
        for i in _fade_frames(tl):
            mod.update_scope(art, tr, tl.head_time(i))
            assert art["dot"].get_visible() and art["dot"].get_alpha() == 1.0
            assert all(h.get_visible() and h.get_alpha() == a for h, a in art["halo"])
            assert art["glow_hi"].get_visible() and art["glow_hi"].get_alpha() == pytest.approx(0.45)
            assert len(art["glow_hi"].get_xdata()) > 0
    finally:
        mod.plt.close(fig)


def test_clip_durations():
    for key in KEYS:
        tl = mod.timeline(key)
        assert tl.n_fade / mod.FPS == pytest.approx(mod.style.FADE_OUT_S, abs=0.5 / mod.FPS)
        assert tl.stream_start / mod.FPS == pytest.approx(mod.HOLD_IN_S)
        assert tl.duration == pytest.approx(tl.n_frames / mod.FPS)
        assert tl.n_frames == tl.stream_start + tl.n_stream + tl.n_fade      # nothing after the stream
        # the tests start ~1 s before odor onset; training starts at odor ON
        lo, hi = (-0.05, 0.0) if key == "training" else (-1.1, -0.9)
        assert lo <= tl.head_time(0) <= hi
    assert mod.LEAD_S == pytest.approx(1.0)


def test_x_ticks_every_second_at_the_stream_speed():
    assert list(mod.X_TICK_S) == list(range(int(mod.WINDOW_S), -1, -1))
    assert mod.X_TICK_LABELS[-1] == "now"
    for s, lab in zip(mod.X_TICK_S[:-1], mod.X_TICK_LABELS[:-1]):
        assert lab == f"−{s} s"
    xs = [mod.stream_x(-s, 0.0) for s in mod.X_TICK_S]
    assert np.allclose(np.diff(xs), mod.SPEED_PX_S)                 # 1 s apart at the stream speed
    assert min(xs) >= 0 and max(xs) == pytest.approx(mod.PLOT_W)


def test_footage_is_always_the_heads_own_frame():
    odor_on, fps_src = 30.0599700149925, 40.02
    for key in KEYS:
        tl = mod.timeline(key)
        q = mod.footage_positions(tl, odor_on, fps_src)
        p = np.array([(odor_on + tl.head_time(i)) * fps_src for i in range(tl.n_frames)])
        assert q.dtype.kind == "i"                                        # whole frames, no blending
        assert np.all(np.abs(q - p) <= 0.5 + 1e-9)
        assert np.all(np.diff(q) >= 0)


def test_view_box_inside_the_stabilised_crop():
    x0, y0, w, h = mod.VIEW
    assert x0 >= 0 and y0 >= 0 and x0 + w <= mod.CROP[2] and y0 + h <= mod.CROP[3]
    assert (mod.VID_W, mod.VID_H) == mod.montage.fit_aspect(w, h, mod.VID_MAX_W, mod.VID_MAX_H)
    assert mod.VID_W >= 470 and mod.VID_H >= 380                   # was 386 x 343
    assert mod.VID_W / w <= 1.75                                    # upscale stays sharp


def test_head_position_equals_the_current_video_frame():
    odor_on, fps_src = 30.0599700149925, 40.02
    tl = mod.Timeline(end_rel=7.5)
    for i in range(tl.n_frames):
        t = tl.head_time(i)
        f = mod.source_frame(t, odor_on, fps_src)
        assert abs(f / fps_src - (odor_on + t)) <= 0.5 / fps_src + 1e-12
        # the head sample is drawn at the right edge
        assert mod.stream_x(t, t) == pytest.approx(mod.PLOT_W)


# ------------------------------------------------------------------ event state
def test_threshold_floor_and_noise_rule():
    t = np.arange(-10, 10, 0.025)
    quiet = np.where(t < 0, 0.1 * np.sin(t * 7), 60.0)
    assert mod.per_threshold(quiet, t) == mod.PER_FLOOR_PX
    noisy = np.where(t < 0, 3.0 * np.sign(np.sin(t * 7)), 0.0)
    assert mod.per_threshold(noisy, t) > mod.PER_FLOOR_PX


def test_event_state_toggles_only_above_threshold():
    thr = 10.0
    assert not mod.is_event(10.0, thr) and mod.is_event(10.01, thr)
    assert not mod.is_event(np.nan, thr)
    assert mod.readout_k(thr, thr) == 0.0 and mod.readout_k(thr - 5, thr) == 0.0
    assert mod.readout_k(thr + mod.K_SPAN_PX, thr) == 1.0 and mod.readout_k(1e6, thr) == 1.0
    vals = np.array([0, 3, 9.9, 10, 12, 40, 80, 11, 9, 2, -4])
    for v in vals:
        text, scale, color = mod.readout_state(v, thr)
        hot = v > thr
        assert (color == mod.EVENT_C) == hot
        assert scale == pytest.approx(1 + mod.READOUT_GAIN * mod.readout_k(v, thr))
        assert (scale > 1) == hot
    assert mod.EVENT_C == "#d95926" == mod.montage.LABELLUM_C


def test_readout_formatting():
    assert mod.format_readout(0.4) == "0" and mod.format_readout(-0.4) == "0"
    assert mod.format_readout(-2.2) == "−2" and mod.format_readout(65.6) == "66"
    assert mod.format_readout(np.nan) == "–"


def test_readout_floors_below_rest_displacement_at_zero():
    # the trace keeps the real (slightly negative) data; only the number is floored
    for v in (-3.2, -1.0, -0.4):
        assert mod.readout_state(v, 10.0)[0] == "0"
    assert mod.readout_state(4.6, 10.0)[0] == "5"
    assert mod.readout_state(np.nan, 10.0)[0] == "–"


def test_split_at_threshold_inserts_crossings_into_both_parts():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 20.0, 20.0, 0.0])
    xs, lo, hi = mod.split_at_threshold(x, y, 10.0)
    assert list(xs) == [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    assert np.isnan(lo[2]) and np.isnan(lo[3]) and lo[1] == 10.0 and lo[4] == 10.0
    assert np.isnan(hi[0]) and np.isnan(hi[-1]) and hi[1] == 10.0 and hi[2] == 20.0


def test_upward_crossings_count():
    assert mod.upward_crossings(np.array([0, 11, 12, 3, 15, 0]), 10) == 2
    assert mod.upward_crossings(np.array([11, 12]), 10) == 0     # already above: not a new event
    assert mod.upward_crossings(np.array([0, 1, 2]), 10) == 0


def test_band_extent_never_draws_the_future_and_sticky_labels_get_pushed_out():
    assert mod.band_extent(0.0, 30.0, head_t=-1.0) is None          # odor not on yet
    x0, x1 = mod.band_extent(0.0, 30.0, head_t=1.0)
    assert x0 == pytest.approx(mod.PLOT_W - mod.SPEED_PX_S) and x1 == pytest.approx(mod.PLOT_W)
    assert mod.sticky_x(-500, 600, 80) == pytest.approx(mod.BAND_LABEL_PAD)
    assert mod.sticky_x(100, 600, 80) == pytest.approx(100 + mod.BAND_LABEL_PAD)
    assert mod.sticky_x(-500, 50, 80) == pytest.approx(50 - 80 - mod.BAND_LABEL_PAD)   # pushed out left


# ------------------------------------------------------------------ text
FORBIDDEN = [r"female", r"single fly", r"drosophila", r"melanogaster", r"yolo", r"fly ?\d", r"batch",
             r"pretest_", r"testing_", r"training_", r"representative", r"Hex-Training", r"_rig",
             r"august", r"\brig\b"]


def test_no_identity_or_boilerplate_text_on_screen_or_in_file_names():
    for key in KEYS:
        texts = mod.onscreen_strings(key)
        assert len(texts) >= 10
        for t in texts:
            for pat in FORBIDDEN:
                assert not re.search(pat, t, re.I), (pat, t)
        name = mod.out_path(key).name
        assert name == f"PER_tracking_scope_{key}.mp4"
        for pat in FORBIDDEN + [r"fly"]:
            assert not re.search(pat, name, re.I), (pat, name)
    src = SCRIPT.read_text()
    for line in src.splitlines():                        # nothing data-derived is drawn as text
        if ".text(" in line:
            assert not re.search(r"\b(clip\.trial|FLY|BATCH|trial_dir)\b", line), line


def test_genotype_if_shown_is_italic_and_only_identity():
    for key in KEYS:
        for t in mod.onscreen_strings(key):
            assert "Gr5a" not in t or t == "Gr5a-GAL4 > UAS-CsChrimson"


def test_text_size_floor_heading_and_readout():
    assert mod.style.MIN_TEXT_PX >= 32 and mod.TEXT_PT >= mod.style.MIN_TEXT_PT
    assert mod.style.HEAD_SIZE >= 54
    assert mod.READOUT_PX >= 64
    for key in KEYS:
        fig, art = mod.build_figure(key, odor_name="Hexanol", has_light=key == "training", thr=10.0)
        try:
            mod.assert_text_floor(fig)                    # raises on any text < MIN_TEXT_PT
            # readout at full jump stays clear of the plot and the card title
            mod.set_readout(art, 80.0, 10.0)
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            rb = art["readout"].get_window_extent(r)
            pb = art["ax_scope"].get_window_extent(r)
            assert rb.y0 > pb.y1 + 4, (rb, pb)
            tb = art["card_title"].get_window_extent(r)
            assert rb.x0 > tb.x1 + 20
            assert art["readout"].get_fontsize() >= mod.READOUT_PX * 72 / 100 - 1e-6
        finally:
            mod.plt.close(fig)


def test_every_text_stays_up_long_enough_to_read():
    for key in KEYS:
        tl = mod.timeline(key)
        items = mod.reading_items(key)
        assert any(it.heading for it in items)
        short = mod.style.reading_shortfalls(items, tl.duration, mod.style.FADE_OUT_S)
        assert not short, short


def test_rendering_is_deterministic():
    src = SCRIPT.read_text()
    assert "random" not in src and "time.time" not in src


# ------------------------------------------------------------------ real data (skipped off the lab box)
def _have_data():
    try:
        return mod.montage.trial_dir("training_4").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _have_data(), reason="lab data not mounted")
@pytest.mark.parametrize("key,n_events", [("training", 1), ("cs_plus_test", 1), ("untrained_odor_test", 0)])
def test_real_trace_event_happens_at_most_once_and_the_fade_shows_it(key, n_events):
    tr = mod.load_clip_data(key)
    tl = mod.timeline(key)
    heads = np.array([mod.head_value(tr, tl.head_time(i)) for i in range(tl.n_frames)])
    assert np.isfinite(heads).all()
    assert mod.upward_crossings(heads, tr["thr"]) == n_events
    assert not mod.is_event(heads[0], tr["thr"])                     # starts at rest
    # the extension is under the head for the whole fade (start ... last frame), or never
    assert mod.is_event(heads[tl.fade_start], tr["thr"]) == (n_events == 1)
    assert mod.is_event(heads[-1], tr["thr"]) == (n_events == 1)
    assert tr["thr"] == pytest.approx(mod.PER_FLOOR_PX)


@pytest.mark.skipif(not _have_data(), reason="lab data not mounted")
@pytest.mark.parametrize("key", [k for k in KEYS if k != "untrained_odor_test"])
def test_onset_stays_inside_the_window_throughout_the_fade(key):
    """From the fade's first frame to its last, the window shows the PER's onset (the upward
    threshold crossing) and what follows; for training also the plateau after it."""
    tr = mod.load_clip_data(key)
    tl = mod.timeline(key)
    heads = np.array([tl.head_time(i) for i in range(tl.n_frames)])
    vals = np.array([mod.head_value(tr, h) for h in heads])
    k = int(np.flatnonzero(~(vals[:-1] > tr["thr"]) & (vals[1:] > tr["thr"]))[0]) + 1
    assert k < tl.fade_start                                            # the onset is written before the fade
    for i in range(tl.fade_start, tl.n_frames):
        x = float(mod.stream_x(heads[k], heads[i]))
        assert 0.5 * mod.SPEED_PX_S <= x <= mod.PLOT_W, (key, i, x)     # >= 0.5 s inside the left edge
    if key == "training":
        # at the fade's start >= 2.5 s of sustained extension (plateau) sits between onset and head
        t, e = tr["t_rel"], tr["ext"]
        plateau = (t >= heads[k] + 0.5) & (t <= heads[tl.fade_start])
        assert heads[tl.fade_start] - heads[k] >= 2.5
        assert np.nanmin(e[plateau]) > tr["thr"] and np.nanmedian(e[plateau]) > 50


@pytest.mark.skipif(not _have_data(), reason="lab data not mounted")
@pytest.mark.parametrize("key", KEYS)
def test_tighter_view_keeps_the_eye_and_the_whole_labellum_path(key):
    tr = mod.load_clip_data(key)
    tl = mod.timeline(key)
    x0, y0, w, h = mod.VIEW
    lo, hi = tl.head_time(0), tl.head_time(tl.n_frames - 1)
    s = (tr["t_rel"] >= lo - 0.1) & (tr["t_rel"] <= hi + 0.1)
    pts = np.vstack([tr["lab_xy"][s], [tr["eye_xy"]]]) - np.array(mod.CROP[:2])
    pts = pts[np.isfinite(pts).all(1)]
    m = 15                                                              # ring radius + stabiliser shift
    assert np.all((pts[:, 0] >= x0 + m) & (pts[:, 0] <= x0 + w - m)), key
    assert np.all((pts[:, 1] >= y0 + m) & (pts[:, 1] <= y0 + h - m)), key


@pytest.mark.skipif(not _have_data(), reason="lab data not mounted")
@pytest.mark.parametrize("key", KEYS)
def test_trace_never_runs_through_the_threshold_label(key):
    tr = mod.load_clip_data(key)
    tl = mod.timeline(key)
    fig, art = mod.build_figure(key, tr["odor_name"], bool(tr["light"]), tr["thr"])
    try:
        fig.canvas.draw()
        x0, x1, y0, y1 = mod.label_data_box(art["thresh_label"], art["ax_scope"])
    finally:
        mod.plt.close(fig)
    pad_x = mod.SPEED_PX_S / tr["fps"] + 4                              # one sample + half the line
    pad_y = 3.0 * (mod.Y_MAX - mod.Y_MIN) / mod.PLOT_H                # half the 5 px line + 0.5 px
    t, e = tr["t_rel"], tr["ext"]
    for i in range(tl.n_frames):
        h = tl.head_time(i)
        s = t <= h
        xs = mod.stream_x(t[s], h)
        inside = (xs >= x0 - pad_x) & (xs <= x1 + pad_x)
        ys = e[s][inside]
        assert not np.any((ys > y0 - pad_y) & (ys < y1 + pad_y)), (key, i)


def test_training_clip_starts_at_odor_onset_and_dwells_on_the_extension():
    """User 2026-10-30: training starts at odor ON and shows more of the extended proboscis;
    review: the onset must still be on screen to the clip's last (fading) frame."""
    spec, tl = mod.CLIPS["training"], mod.timeline("training")
    assert spec.lead_s == 0.0
    assert tl.head_time(0) == pytest.approx(0.0, abs=0.5 / mod.FPS)
    assert spec.end_rel - mod.PER_ONSET_TRAINING_S >= 2.5              # plateau in view as the fade starts
    dwell_last = tl.last_rel - mod.PER_ONSET_TRAINING_S
    assert dwell_last >= 3.5                                           # extension fills most of the window
    assert 0.9 <= mod.WINDOW_S - dwell_last <= 1.6                     # onset 1-1.5 s inside the left edge
    # the tests keep a pre-odor rest lead
    assert mod.CLIPS["cs_plus_test"].lead_s == mod.CLIPS["untrained_odor_test"].lead_s == mod.LEAD_S


def test_untrained_clip_fades_past_the_cs_plus_latency():
    assert mod.CLIPS["untrained_odor_test"].end_rel == pytest.approx(4.7)
    assert mod.CLIPS["untrained_odor_test"].end_rel > mod.CS_PLUS_LATENCY_S
    # the two test clips are cut identically (same lead, same fade start, same length)
    a, b = mod.timeline("cs_plus_test"), mod.timeline("untrained_odor_test")
    assert (a.n_frames, a.start_rel) == (b.n_frames, b.start_rel)


def test_header_has_no_orphan_accent_rule():
    assert all(not isinstance(el, mod.style.BarEl) for el in mod.HDR.els)
