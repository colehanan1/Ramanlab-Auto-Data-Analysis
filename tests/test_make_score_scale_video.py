"""Pure logic of the PER score-scale explainer video renderer.

Covers spec loading / validation, segment timing (slowed playback that never
freezes), frame <-> trial-time mapping, the fixed score palette, caption text
rules and the on-screen layout (no overlaps, legible sizes). No video is decoded.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analysis/make_score_scale_video.py"
spec = importlib.util.spec_from_file_location("score_scale_video", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from scripts.analysis import score_summary  # noqa: E402


def entry(**over) -> dict:
    """A valid spec entry (paths need not exist: validation without check_files)."""
    d = {
        "score": 3, "variant": None,
        "caption": "Clear extension during the odor, held for a few seconds.",
        "dataset": "Hex-Training-24-0.1", "batch": "august_11_batch_1_rig_2", "fly_number": 3,
        "trial_label": "testing_8", "trial_dir": "/data/b/august_11_batch_1_testing_8",
        "video": "/data/b/output_august_11_batch_1_testing_8_Hexanol_20260811_142415.mp4",
        "envelope_csv": "/data/b/angle_distance_rms_envelope/t8_fly3.csv",
        "trace_column": "combined_pct", "fps": 40.02, "odor_on_s": 30.08, "odor_off_s": 60.09,
        "clip_start_s": 28.0, "clip_end_s": 36.0, "peak_px_over_rest": 31.1, "peak_pct": 57.3,
        "alternates": [{"anything": 1}],
    }
    d.update(over)
    return d


CANON = [(-1, None), (0, None), (1, "a"), (1, "b"), (2, None), (3, None), (4, None), (5, None)]


def full_spec() -> list[dict]:
    return [entry(score=s, variant=v) for s, v in CANON]


# ------------------------------------------------------------------ score palette
def test_score_colours_are_the_fixed_palette_imported_not_redefined():
    assert mod.SCORE_COLORS is score_summary.SCORE_COLORS
    cmap, norm = score_summary._score_cmap()
    for s in range(-1, 6):
        assert mod.score_color(s) == score_summary.SCORE_COLORS[s + 1]
        assert np.allclose(cmap(norm(s))[:3], mod.to_rgb(mod.score_color(s)))
    with pytest.raises(ValueError):
        mod.score_color(6)
    with pytest.raises(ValueError):
        mod.score_color(-2)


def test_script_defines_no_score_hexes_of_its_own():
    src = SCRIPT.read_text()
    for hx in score_summary.SCORE_COLORS:
        assert hx.lower() not in src.lower(), f"score colour {hx} hard-coded in the renderer"


def _rel_lum(hex_):
    c = np.array(mod.to_rgb(hex_))
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


@pytest.mark.parametrize("score", range(-1, 6))
def test_badge_number_contrasts_with_its_badge(score):
    a, b = _rel_lum(mod.score_color(score)), _rel_lum(mod.badge_text_color(score))
    ratio = (max(a, b) + 0.05) / (min(a, b) + 0.05)
    assert ratio >= 4.5


def test_score_labels_use_a_true_minus_and_variants_are_suffixed():
    assert mod.score_label(-1) == "−1"
    assert mod.score_label(0) == "0" and mod.score_label(5) == "5"
    clips = mod.parse_spec(full_spec())
    assert [c.label for c in clips] == ["-1", "0", "1a", "1b", "2", "3", "4", "5"]


# ------------------------------------------------------------------ spec loading
def test_full_canonical_spec_parses_without_warnings(tmp_path):
    p = tmp_path / "clips.json"
    p.write_text(json.dumps(full_spec()))
    clips = mod.load_spec(p)
    assert len(clips) == 8
    assert mod.order_warnings(clips) == []
    c = clips[0]
    assert c.score == -1 and c.variant is None and c.fly_number == 3
    assert c.duration_s == pytest.approx(8.0)
    assert isinstance(c.trial_dir, Path) and isinstance(c.video, Path)


def test_unknown_keys_and_alternates_are_ignored():
    clips = mod.parse_spec([entry(extra_note="x", alternates=[1, 2, 3])])
    assert len(clips) == 1


@pytest.mark.parametrize("bad, needle", [
    ({"score": 6}, "score"),
    ({"score": -2}, "score"),
    ({"score": True}, "score"),
    ({"score": 2.0}, "score"),
    ({"variant": "c"}, "variant"),
    ({"fly_number": 5}, "fly_number"),
    ({"fly_number": 0}, "fly_number"),
    ({"clip_end_s": 27.0}, "clip"),
    ({"clip_end_s": 32.0}, "5"),            # 4 s: too short
    ({"clip_end_s": 39.0}, "10"),           # 11 s: too long
    ({"odor_off_s": 20.0}, "odor"),
    ({"fps": 0}, "fps"),
    ({"caption": ""}, "caption"),
    ({"trace_column": ""}, "trace_column"),
    ({"video": "relative/out.mp4"}, "absolute"),
    ({"clip_start_s": -1.0, "clip_end_s": 6.0}, "clip_start_s"),
])
def test_invalid_entries_are_rejected_with_a_named_reason(bad, needle):
    with pytest.raises(mod.SpecError) as ei:
        mod.parse_spec([entry(**bad)])
    assert needle in str(ei.value)


def test_missing_field_is_named():
    d = entry()
    del d["envelope_csv"]
    with pytest.raises(mod.SpecError, match="envelope_csv"):
        mod.parse_spec([d])


def test_top_level_must_be_a_non_empty_list():
    with pytest.raises(mod.SpecError):
        mod.parse_spec({"clips": []})
    with pytest.raises(mod.SpecError):
        mod.parse_spec([])


def test_every_error_is_reported_at_once():
    with pytest.raises(mod.SpecError) as ei:
        mod.parse_spec([entry(score=9), entry(fly_number=7)])
    msg = str(ei.value)
    assert "[0]" in msg and "[1]" in msg


def test_check_files_reports_missing_paths(tmp_path):
    with pytest.raises(mod.SpecError, match="does not exist"):
        mod.parse_spec([entry(trial_dir=str(tmp_path / "nope"))], check_files=True)


def test_order_warnings_flag_out_of_order_and_duplicate_segments():
    clips = mod.parse_spec([entry(score=3), entry(score=0)])
    assert any("order" in w for w in mod.order_warnings(clips))
    dup = mod.parse_spec([entry(score=2), entry(score=2)])
    assert any("twice" in w for w in mod.order_warnings(dup))
    # a placeholder subset in canonical order is fine
    ok = mod.parse_spec([entry(score=0), entry(score=1, variant="b"), entry(score=4)])
    assert mod.order_warnings(ok) == []


def test_example_tags_number_repeated_scores_only():
    clips = mod.parse_spec(full_spec())
    tags = mod.example_tags(clips)
    assert tags[2] == "Example 1 of 2" and tags[3] == "Example 2 of 2"
    assert all(t is None for i, t in enumerate(tags) if i not in (2, 3))


# ------------------------------------------------------------------ caption rules
@pytest.mark.parametrize("caption", [
    "Fly 3 extends strongly.",
    "fly3 extends strongly.",
    "From august_11_batch_1, a full extension.",
    "Second batch, full extension.",
    "Seen on rig 2.",
    "testing_8 shows extension.",
    "Female fly, full extension.",
    "Single fly response.",
    "Drosophila melanogaster extends.",
    "A representative individual.",
    "Hex-Training-24-0.1 fly extends.",
])
def test_captions_with_provenance_or_boilerplate_are_rejected(caption):
    assert mod.caption_violations(caption, mod.parse_spec([entry()])[0])
    with pytest.raises(mod.SpecError):
        mod.parse_spec([entry(caption=caption)])


@pytest.mark.parametrize("caption", [
    "No extension: the proboscis stays at rest through the odor.",
    "Brief, partial extension. Hard to say whether it is odor-evoked or random.",
    "Full extension, held for most of the odor window.",
    "The proboscis retracts below its resting position during the odor.",
])
def test_scientific_captions_pass(caption):
    assert mod.caption_violations(caption, mod.parse_spec([entry()])[0]) == []


def test_two_line_caption_for_1b_fits_and_reads_in_time():
    cap = ("A brief, partial extension during the odor. Hard to say whether it is "
           "odor-evoked or a random, spontaneous extension.")
    c = mod.parse_spec([entry(score=1, variant="b", caption=cap)])[0]
    lines = mod.caption_lines(cap)
    assert 2 <= len(lines) <= mod.MAX_CAPTION_LINES
    assert " ".join(lines).split() == cap.split()             # every word kept, in order
    m = mod.style.run_measure(mod.CAPTION_PX)
    assert all(m(ln) <= mod.CAPTION_W for ln in lines)
    assert mod.reading_shortfalls(c) == []


def test_caption_too_long_to_fit_is_rejected():
    cap = " ".join(["extension"] * 60)
    with pytest.raises(mod.SpecError, match="caption"):
        mod.parse_spec([entry(caption=cap)])


def test_caption_too_long_to_read_in_a_short_clip_is_rejected():
    # fits in 3 lines but 26 words cannot be read in a 5 s clip (10 s on screen at 0.5x)
    cap = " ".join(["a b c d e f g h i j k l m n o p q r s t u v w x y z".split()[i] for i in range(26)])
    with pytest.raises(mod.SpecError, match="read"):
        mod.parse_spec([entry(caption=cap, clip_start_s=30.0, clip_end_s=35.0)])


# ------------------------------------------------------------------ timing
@pytest.mark.parametrize("speed", [0.5, 0.4, 1.0])
def test_slowed_playback_never_freezes_and_covers_the_window(speed):
    c = mod.parse_spec([entry(clip_start_s=28.0, clip_end_s=36.0)])[0]
    n = mod.n_output_frames(c, speed)
    assert n == int(round(8.0 / speed * mod.FPS))
    assert mod.on_screen_s(c, speed) == pytest.approx(n / mod.FPS)
    ts = np.array([mod.source_time(k, c, speed) for k in range(n)])
    assert ts[0] == pytest.approx(c.clip_start_s)
    assert np.all(np.diff(ts) > 0)                            # strictly advancing: never held
    assert np.allclose(np.diff(ts), speed / mod.FPS)          # constant slow-motion rate
    assert c.clip_end_s - speed / mod.FPS - 1e-9 <= ts[-1] <= c.clip_end_s + 1e-9
    # the fade-out plays real, advancing footage: the window covers its own fade
    fade = [k for k in range(n) if mod.style.segment_fade(k / mod.FPS, n / mod.FPS, 0, mod.style.FADE_OUT_S) < 1]
    assert fade and fade[-1] == n - 1
    assert len(set(ts[fade])) == len(fade)


def test_default_speed_is_half():
    assert mod.SPEED == 0.5


def test_frame_range_covers_every_blend_neighbour():
    c = mod.parse_spec([entry(clip_start_s=28.013, clip_end_s=35.987)])[0]
    first, last = mod.source_frame_range(c)
    for k in range(mod.n_output_frames(c, 0.5)):
        i0, i1, w = mod.blend_index(mod.source_time(k, c, 0.5), c.fps, first)
        assert 0 <= i0 <= i1 <= last - first
        assert 0.0 <= w < 1.0


def test_frame_time_mapping_uses_the_recording_fps():
    assert mod.frame_time(1204, 40.02) == pytest.approx(30.0849575)
    assert mod.frame_pos(30.0849575, 40.02) == pytest.approx(1204.0)
    i0, i1, w = mod.blend_index(10.0125, 40.0, 0)              # 400.5 -> halfway 400/401
    assert (i0, i1) == (400, 401) and w == pytest.approx(0.5)
    i0, i1, w = mod.blend_index(10.0, 40.0, 100)               # exact frame, relative to first
    assert (i0, i1, w) == (300, 300, 0.0)


def test_trace_is_read_by_row_as_frame_not_by_the_csv_time_column():
    """The envelope CSV's time_s is row / 40 while the recording runs at 40.02 fps: the row is
    the video frame, so trace time = row / fps (otherwise the trace drifts ~45 ms by 90 s)."""
    n = 3606
    env = pd.DataFrame({"time_s": np.arange(n) / 40.0, "combined_pct": np.arange(n, dtype=float)})
    t, y = mod.trace_series(env, "combined_pct", fps=40.02, odor_on_s=30.08)
    assert t[3605] == pytest.approx(3605 / 40.02)
    assert y[3605] - y[0] == 3605.0


def test_px_traces_are_relative_to_the_pre_odor_rest():
    n = 2000
    fps = 40.0
    d = np.full(n, 50.0)
    d[1300:1400] = 110.0
    env = pd.DataFrame({"d_px": d})
    t, y = mod.trace_series(env, "d_px", fps=fps, odor_on_s=1200 / fps)
    assert np.nanmedian(y[:1200]) == pytest.approx(0.0)
    assert np.nanmax(y) == pytest.approx(60.0)
    assert mod.trace_kind("d_px") == "px" and mod.trace_kind("combined_pct") == "pct"
    assert mod.trace_kind("envelope_of_rms") == "pct"


def test_pct_traces_are_relative_to_the_pre_odor_rest_too():
    """The readout is the rise over rest, so the drawn PER % line must be too: 0 = the fly's own
    pre-odor level, and a retraction dips below 0."""
    fps, n = 40.0, 2400
    pct = np.full(n, 22.0)
    pct[1300:1500] = 2.0                                       # retraction to 20 points below rest
    env = pd.DataFrame({"combined_pct": pct})
    t, y = mod.trace_series(env, "combined_pct", fps=fps, odor_on_s=1200 / fps)
    assert np.nanmedian(y[:1200]) == pytest.approx(0.0)
    assert np.nanmin(y) == pytest.approx(-20.0)
    assert "rest" in mod.Y_LABEL["pct"]
    lo, hi = mod.shared_ylim([y], "pct")
    assert lo <= -20.0 and min(mod.y_ticks(lo, hi)) < 0       # the dip gets a labelled tick


def test_trace_value_interpolates_between_frames():
    t = np.array([0.0, 0.025, 0.05])
    y = np.array([0.0, 10.0, 20.0])
    assert mod.trace_at(0.0125, t, y) == pytest.approx(5.0)


def test_shared_y_scale_is_identical_for_every_segment_and_covers_all_peaks():
    lo, hi = mod.shared_ylim([np.array([1.0, 40.0]), np.array([3.0, 151.0])], "pct")
    assert lo < 0 <= 0 and hi >= 151.0
    lo2, hi2 = mod.shared_ylim([np.array([1.0, 20.0])], "pct")
    assert hi2 >= 100.0                                       # PER % axis always shows 0-100
    lo3, hi3 = mod.shared_ylim([np.array([0.0, 37.0]), np.array([0.0, 64.0])], "px")
    assert hi3 >= 64.0 and hi3 < 64.0 * 1.4
    ticks = mod.y_ticks(lo, hi)
    assert ticks[0] == 0 and all(lo <= v <= hi for v in ticks) and 3 <= len(ticks) <= 6


def test_peak_readout_text():
    c = mod.parse_spec([entry(peak_pct=86.8, peak_px_over_rest=68.2)])[0]
    big, small = mod.peak_text(c, "pct")
    assert big == "87 %" and small == "+68 px over rest"
    big, small = mod.peak_text(c, "px")
    assert big == "+68 px" and small == ""
    assert mod.peak_label(c) == mod.PEAK_LABEL


def test_retraction_readout_shows_the_dip():
    # A -1 clip carries its trough: the readout must say how far BELOW rest it went,
    # never "0 %" / "-0 px over rest".
    c = mod.parse_spec([entry(score=-1, peak_pct=-20.0, peak_px_over_rest=-33.8)])[0]
    big, small = mod.peak_text(c, "pct")
    assert big == "−20 %" and small == "34 px below rest"
    assert mod.peak_label(c) == mod.DIP_LABEL
    assert not FORBIDDEN.search(mod.DIP_LABEL)


def test_contrast_stretch_is_one_mapping_per_clip():
    rng = np.random.default_rng(0)
    frames = rng.integers(40, 90, size=(5, 20, 30)).astype(np.uint8)   # dim, low-contrast
    out = mod.stretch_contrast(frames)
    assert out.dtype == np.uint8 and out.shape == frames.shape
    assert out.max() >= 240 and out.min() <= 15                      # uses the full range
    # same input value -> same output value in every frame (no per-frame flicker)
    v = frames[0, 0, 0]
    assert len({int(o) for o in out[frames == v]}) == 1
    flat = np.full((3, 4, 4), 50, np.uint8)                          # no range: left alone
    assert np.array_equal(mod.stretch_contrast(flat), flat)


# ------------------------------------------------------------------ crop
def test_crop_is_fixed_size_led_toward_the_labellum_and_kept_in_frame():
    eye = (565.0, 620.0)
    lab = np.array([[565.0, 667.0], [560.0, 700.0]])
    x0, y0, w, h = mod.crop_for(eye, lab, (1080, 1080))
    assert (w, h) == mod.CROP_WH
    cx, cy = x0 + w / 2, y0 + h / 2
    assert cy > eye[1] and abs(cx - eye[0]) < 10               # shifted toward the labellum (down)
    # a fly at the frame's top edge: the crop is clamped inside the frame
    x0, y0, w, h = mod.crop_for((500.0, 40.0), np.array([[540.0, 95.0]]), (1080, 1080))
    assert y0 == 0 and 0 <= x0 <= 1080 - w
    # mirrored rig: labellum to the left of the eye -> crop leads left
    x0, y0, w, h = mod.crop_for((600.0, 500.0), np.array([[540.0, 500.0]]), (1080, 1080))
    assert x0 + w / 2 < 600.0


# ------------------------------------------------------------------ title / reading
def test_title_card_has_genotype_and_is_up_long_enough_to_read():
    els = mod.title_els()
    texts = [getattr(e, "text", "") for e in els]
    assert mod.TITLE_HEADLINE in texts
    assert mod.GENOTYPE in texts
    assert mod.GENOTYPE == "Gr5a-GAL4 > UAS-CsChrimson"
    dur = mod.title_duration(els)
    assert mod.style.card_reading_shortfall(mod.style.read_items(els), dur) == 0.0
    assert dur <= 12.0


def test_genotype_is_italic_on_the_title():
    assert mod.GENOTYPE_RUNS == [(mod.GENOTYPE, True)]


FORBIDDEN = re.compile(r"\bfly\s*#?\s*\d|batch|\brig\b|female|single fly|drosophila|melanogaster|"
                       r"representative|testing_|training_|Hex-|august", re.I)


def test_static_copy_has_no_provenance_or_boilerplate():
    for s in (mod.TITLE_HEADLINE, mod.TITLE_BODY, mod.HEADER_LABEL, mod.CLOSING_LABEL, mod.Y_LABEL["pct"],
              mod.Y_LABEL["px"], mod.SPEED_LABEL(0.5), mod.PEAK_LABEL, mod.ODOR_LABEL):
        assert not FORBIDDEN.search(s), s


def test_cli_default_output_is_the_hand_run_figures_folder():
    args = mod.build_parser().parse_args(["--spec", "x.json"])
    assert args.out == Path("/home/ramanlab/Documents/cole/Results/Figures/PER_score_scale.mp4")
    assert args.preview_dir is None and args.speed == 0.5


# ------------------------------------------------------------------ layout (no video decoded)
def _fake_data(c, kind="pct"):
    fps = c.fps
    n = int(round((c.clip_end_s + 2) * fps))
    t = np.arange(n) / fps
    y = 5 + 80 * np.exp(-((t - (c.odor_on_s + 3)) / 1.5) ** 2)
    return dict(t=t, y=y, light=(c.odor_on_s + 5, c.odor_off_s), kind=kind)


def _boxes_overlap(a, b, pad=2):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


LONG_1B = ("A brief, partial extension during the odor. Hard to say whether it is "
           "odor-evoked or a random, spontaneous extension.")


@pytest.mark.parametrize("over", [
    dict(score=1, variant="b", caption=LONG_1B),
    dict(score=-1, caption="The proboscis pulls in below rest."),
    dict(score=5, caption="Full extension, held for most of the odor window.", clip_start_s=29.5,
         clip_end_s=39.5),
])
def test_segment_texts_never_overlap_stay_in_frame_and_are_legible(over):
    clips = mod.parse_spec(full_spec())
    c = mod.parse_spec([entry(**over)])[0]
    idx = 3
    fig, hd = mod.segment_figure(c, idx, len(clips), (-6.0, 110.0), _fake_data(c), "Example 2 of 2",
                                 crop_wh=mod.CROP_WH, speed=0.5)
    try:
        mod.assert_text_floor(fig)
        boxes = mod.text_boxes(fig)
        names = [t for t, _ in boxes]
        assert any("odor" == t for t in names)
        assert any(t.startswith("Peak") for t in names)
        for t, (x0, y0, x1, y1) in boxes:
            assert 0 <= x0 and x1 <= mod.W and 0 <= y0 and y1 <= mod.H, t
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not _boxes_overlap(boxes[i][1], boxes[j][1]), (boxes[i], boxes[j])
        # header elements (PIL) do not collide with the matplotlib layer's texts
        for el in mod.header_els(idx, len(clips)):
            eb = mod.style.el_box(el)
            for t, b in boxes:
                assert not _boxes_overlap(eb, b), (t, b, eb)
        for t, _ in boxes:
            assert not FORBIDDEN.search(t), t
    finally:
        mod.plt.close(fig)


def test_video_panel_is_a_large_close_up():
    x, y, w, h = mod.VID
    assert w >= 900 and h >= 800 and y + h <= mod.H - 40
    assert abs(w / h - mod.CROP_WH[0] / mod.CROP_WH[1]) < 0.01


def test_frame_is_1920_by_1080():
    assert (mod.W, mod.H, mod.FPS) == (1920, 1080, 40)


def test_peak_is_searched_inside_the_odor_window_not_before_it():
    c = mod.parse_spec([entry(odor_on_s=30.0, odor_off_s=60.0, clip_start_s=28.0, clip_end_s=36.0)])[0]
    t = np.arange(0, 40, 0.025)
    y = np.full_like(t, 5.0)
    y[(t > 28.5) & (t < 29.0)] = 9.0                          # pre-odor wiggle, higher than the odor part
    y[(t > 33.0) & (t < 33.2)] = 7.0
    k = mod.peak_index(c, t, y)
    assert 33.0 < t[k] < 33.2
    # no odor overlap -> falls back to the clip window
    c2 = mod.parse_spec([entry(odor_on_s=30.0, odor_off_s=60.0, clip_start_s=20.0, clip_end_s=28.0)])[0]
    assert 28.5 > t[mod.peak_index(c2, t, y)] >= 20.0


@pytest.mark.parametrize("n, rows", [(3, [3]), (5, [3, 2]), (7, [4, 3]), (8, [4, 4])])
def test_closing_tiles_are_balanced_in_frame_and_never_overlap(n, rows):
    boxes = mod.tile_grid(n)
    assert len(boxes) == n
    ys = sorted({b[1] for b in boxes})
    assert [sum(b[1] == y for b in boxes) for y in ys] == rows
    for x, y, w, h in boxes:
        assert x >= mod.G - 1 and x + w <= mod.W - mod.G + 1 and y >= mod.CLOSE_TOP and y + h <= mod.CLOSE_BOTTOM
        assert abs(w / h - mod.CROP_WH[0] / mod.CROP_WH[1]) < 0.01
    for i in range(n):
        for j in range(i + 1, n):
            assert not _boxes_overlap(*[(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in (boxes[i], boxes[j])])


# ------------------------------------------------------------------ fixed camera, crisp frames (2026-11-02)


def test_default_is_a_fixed_crop_not_stabilised():
    """User 2026-11-02: the flies are glued and still -- re-centring every frame makes the
    proboscis movement harder to see, so the crop stays fixed unless --stabilize is given."""
    assert mod.STABILIZE is False
    assert mod.build_parser().parse_args(["--spec", "x.json"]).stabilize is False
    assert mod.build_parser().parse_args(["--spec", "x.json", "--stabilize"]).stabilize is True


def test_slow_motion_shows_real_frames_not_blends():
    """Blending neighbouring frames ghosts a moving proboscis; each output frame is the nearest
    recorded frame (at 0.5x every recorded frame is shown twice, motion never stops)."""
    frames = np.stack([np.full((2, 2), v, np.uint8) for v in (0, 100, 200)])
    assert mod.BLEND is False
    out = mod.frame_for(frames, 0, 1, 0.5)
    assert int(out[0, 0]) in (0, 100)                                  # never the 50 blend
    assert int(mod.frame_for(frames, 0, 1, 0.3)[0, 0]) == 0
    assert int(mod.frame_for(frames, 0, 1, 0.7)[0, 0]) == 100


def test_sharpen_lifts_edges_and_leaves_flat_regions():
    img = np.full((1, 21, 21), 60, np.uint8)
    img[:, :, 10:] = 160                                                # one vertical edge
    out = mod.sharpen(img)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert out[0, 10, 0] == 60 and out[0, 10, 20] == 160               # flat areas unchanged
    assert out[0, 10, 10] > 160 and out[0, 10, 9] < 60                 # edge contrast increased
