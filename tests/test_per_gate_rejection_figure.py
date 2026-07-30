"""Tests for the PER gate rejection methods figure.

The gate values asserted here are the ones the thesis text cites, and they live
in ``config/config_new.yaml`` -- NOT in the repo defaults (``config.yaml`` and
``src/fbpipe/config.py`` carry 150 / 180 / 250). The figure must read them at
runtime so it can never silently disagree with the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analysis.per_gate_rejection_figure import (
    CONFIG_PATH,
    GateSettings,
    load_gate_settings,
)


def test_gate_settings_match_config_new() -> None:
    """The gate values the figure reports come from config_new.yaml."""
    s = load_gate_settings()
    assert s.max_px == 160.0
    assert s.up_divisor == 4.0
    assert s.dorsal_px == 40.0
    assert s.max_jump_px == 80.0
    assert s.norm_min_px == 10.0
    assert s.norm_max_px == 160.0
    assert s.three_fly_max_px == 160.0


def test_config_path_points_at_config_new() -> None:
    assert CONFIG_PATH.name == "config_new.yaml"
    assert CONFIG_PATH.exists()


def test_gate_settings_are_not_hardcoded(tmp_path: Path) -> None:
    """Feeding a different config must change the values -- proving they are read,
    not baked into the script."""
    alt = tmp_path / "alt.yaml"
    alt.write_text(
        "proboscis_filter:\n"
        "  max_eye_prob_distance_px: 99.0\n"
        "  up_divisor: 3.0\n"
        "  max_jump_px: 55.0\n"
        "distance_limits:\n"
        "  class2_min: 5.0\n"
        "  class2_max: 99.0\n"
        "  three_fly_max_eye_prob_distance_px: 77.0\n",
        encoding="utf-8",
    )
    s = load_gate_settings(alt)
    assert s.max_px == 99.0
    assert s.up_divisor == 3.0
    assert s.dorsal_px == 33.0
    assert s.max_jump_px == 55.0
    assert s.norm_min_px == 5.0
    assert s.norm_max_px == 99.0
    # deliberately DIFFERENT from max_px (99.0) so a load_gate_settings that
    # conflated the two distinct config keys would fail this assertion.
    assert s.three_fly_max_px == 77.0


def test_gate_settings_rejects_missing_keys(tmp_path: Path) -> None:
    """A config without the gate blocks must fail loudly, not silently default."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_gate_settings(empty)


def test_gate_settings_rejects_missing_three_fly_key(tmp_path: Path) -> None:
    """distance_limits present but missing three_fly_max_eye_prob_distance_px
    must fail loudly too -- it is read exactly like the other gate values, not
    silently defaulted from max_eye_prob_distance_px."""
    partial = tmp_path / "partial.yaml"
    partial.write_text(
        "proboscis_filter:\n"
        "  max_eye_prob_distance_px: 99.0\n"
        "  up_divisor: 3.0\n"
        "  max_jump_px: 55.0\n"
        "distance_limits:\n"
        "  class2_min: 5.0\n"
        "  class2_max: 99.0\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        load_gate_settings(partial)


import numpy as np

from scripts.analysis.per_gate_rejection_figure import (
    SUBJECT,
    Offsets,
    _resolve_frame_numbers,
    load_subject_offsets,
    subject_parquet_path,
    subject_video_path,
)

requires_data = pytest.mark.skipif(
    not subject_parquet_path().exists(),
    reason="subject parquet not mounted on this machine",
)


def test_subject_identity() -> None:
    """The subject is pinned so the figure caption cannot drift from the data."""
    assert SUBJECT.dataset == "3Oct-Training-24-0.1"
    assert SUBJECT.trial_rel == "july_24_batch_2_rig_3/july_24_batch_2_testing_3"
    assert SUBJECT.slot == "fly3"
    assert SUBJECT.odor == "EthylButyrate"
    assert SUBJECT.eye_xy == (604.0, 830.0)
    assert SUBJECT.peak_frame == 1233


@requires_data
def test_subject_offsets_match_spec() -> None:
    """Golden numbers from the candidate scan. If the parquet is ever
    reprocessed, this fails loudly rather than the figure quietly changing."""
    off = load_subject_offsets()
    assert len(off.dx) == 3605, "subject was chosen for its perfect detection record"
    assert len(off.dx) == len(off.dy) == len(off.frames)

    ex, ey = off.eye_xy
    assert round(ex) == 604 and round(ey) == 830

    r = np.hypot(off.dx, off.dy)
    assert r.max() == pytest.approx(127.9, abs=0.1)

    peak_i = int(np.argmax(r))
    assert off.frames[peak_i] == SUBJECT.peak_frame
    assert off.dx[peak_i] == pytest.approx(96.1, abs=0.1)
    assert off.dy[peak_i] == pytest.approx(84.5, abs=0.1)


@requires_data
def test_subject_per_is_diagonal_never_dorsal() -> None:
    """The figure's argument for THIS fly: PER is a diagonal excursion --
    down AND laterally, not a narrow vertical column -- yet still never
    dorsal, which is why the gate is generous laterally/ventrally and tight
    dorsally regardless of a fly's particular angle of approach.

    Unlike a purely-ventral fly, dx here ranges widely (1.3 to 98.7 px), so
    "never dorsal" is asserted on dy alone, not on a narrow abs(dx) bound."""
    off = load_subject_offsets()
    assert off.dy.min() > 0, "this fly never goes dorsal"
    assert off.dx.max() > 90.0, "excursion is substantially lateral, not vertical"
    assert off.dy.max() > 80.0, "excursion is also substantially ventral"
    # diagonal, not axis-aligned: near the peak, dx and dy are comparable
    # magnitude rather than one dwarfing the other.
    r = np.hypot(off.dx, off.dy)
    peak_i = int(np.argmax(r))
    assert off.dx[peak_i] > 0.5 * off.dy[peak_i], "peak PER is diagonal, not near-vertical"


def test_subject_video_path_is_the_raw_recording() -> None:
    """The '*_distance_annotated.mp4' sibling is the pipeline's own overlay and
    must not be used -- we draw our own."""
    path = subject_video_path()
    assert path.name.startswith("output_")
    assert "distance_annotated" not in path.name


@requires_data
def test_subject_offsets_frames_match_parquet_frame_column() -> None:
    """`frames` must come from the parquet's own frame column, not row position.

    This asserts against the column itself (not against arange/contiguity), so
    it would fail if the implementation silently reverted to
    ``np.flatnonzero(ok)`` and the column ever diverged from row position --
    unlike an ``arange(3605)`` check, which both implementations satisfy today.
    """
    import pandas as pd

    from fbpipe.utils.columns import find_eye_xy_columns, find_proboscis_xy_columns
    from fbpipe.utils.tables import read_table

    df = read_table(subject_parquet_path())
    ex_col, ey_col = find_eye_xy_columns(df)
    px_col, py_col = find_proboscis_xy_columns(df)

    ex = pd.to_numeric(df[ex_col], errors="coerce").to_numpy(float)
    ey = pd.to_numeric(df[ey_col], errors="coerce").to_numpy(float)
    px = pd.to_numeric(df[px_col], errors="coerce").to_numpy(float)
    py = pd.to_numeric(df[py_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(px - ex) & np.isfinite(py - ey)
    expected_frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy()[ok].astype(int)

    off = load_subject_offsets()
    np.testing.assert_array_equal(off.frames, expected_frames)


def test_resolve_frame_numbers_prefers_frame_column_over_row_position() -> None:
    """Synthetic dataframe with non-contiguous frame numbers, so frame number
    and row position genuinely diverge. Under the OLD row-position
    implementation (``np.flatnonzero(ok)``) this would return [0, 2, 3], not
    [100, 105, 106] -- so this test fails on the old code, unlike checking
    against the real (contiguous) parquet where both implementations agree."""
    import pandas as pd

    df = pd.DataFrame({"frame": [100, 101, 105, 106, 110]})
    ok = np.array([True, False, True, True, False])
    result = _resolve_frame_numbers(df, ok)
    np.testing.assert_array_equal(result, np.array([100, 105, 106]))


def test_resolve_frame_numbers_falls_back_without_frame_column() -> None:
    """No frame/frame_number/frame_idx column at all -> row position."""
    import pandas as pd

    df = pd.DataFrame({"x_class0": [1, 2, 3, 4, 5]})
    ok = np.array([True, False, True, True, False])
    result = _resolve_frame_numbers(df, ok)
    np.testing.assert_array_equal(result, np.array([0, 2, 3]))


def test_resolve_frame_numbers_accepts_alias_columns() -> None:
    """"frame_number" and "frame_idx" are accepted aliases for "frame"."""
    import pandas as pd

    df = pd.DataFrame({"frame_number": [50, 51, 52]})
    ok = np.array([True, True, False])
    result = _resolve_frame_numbers(df, ok)
    np.testing.assert_array_equal(result, np.array([50, 51]))

    df2 = pd.DataFrame({"frame_idx": [7, 8, 9]})
    ok2 = np.array([False, True, True])
    result2 = _resolve_frame_numbers(df2, ok2)
    np.testing.assert_array_equal(result2, np.array([8, 9]))


def test_resolve_frame_numbers_degrades_to_row_position_on_nan() -> None:
    """A frame column with an unparsable/NaN value must not raise -- it
    degrades to row position rather than crashing on ``.astype(int)``."""
    import pandas as pd

    df = pd.DataFrame({"frame": [100, 101, np.nan, 106, 110]})
    ok = np.array([True, False, True, True, False])
    result = _resolve_frame_numbers(df, ok)
    np.testing.assert_array_equal(result, np.array([0, 2, 3]))


from scripts.analysis.per_gate_rejection_figure import (
    REJECTED_OFFSET,
    gate_boundary_offsets,
    gate_norm,
    offsets_survive_geometry_gate,
)


def test_boundary_extremes_match_the_gate() -> None:
    """160 px lateral, 160 px ventral, 40 px dorsal -- traced by the production
    function, not re-derived here."""
    s = load_gate_settings()
    pts = gate_boundary_offsets(s, n=720)
    dx, dy = pts[:, 0], pts[:, 1]

    assert dx.max() == pytest.approx(160.0, abs=0.5)
    assert dx.min() == pytest.approx(-160.0, abs=0.5)
    assert dy.max() == pytest.approx(160.0, abs=0.5), "ventral (dy > 0) is generous"
    assert dy.min() == pytest.approx(-40.0, abs=0.5), "dorsal (dy < 0) is tightened"


def test_boundary_comes_from_the_production_function(monkeypatch) -> None:
    """If someone re-implements the ellipse maths locally, this fails."""
    import scripts.analysis.per_gate_rejection_figure as mod

    called = {}

    def spy(max_px, up_divisor, n=72):
        called["args"] = (max_px, up_divisor, n)
        return [(0.0, 0.0)]

    monkeypatch.setattr(mod, "anisotropic_boundary_offsets", spy)
    gate_boundary_offsets(load_gate_settings(), n=123)
    assert called["args"] == (160.0, 4.0, 123)


def test_constructed_rejection_is_genuinely_rejected() -> None:
    """The invented X mark must be a rejection the real model would make.

    This is what keeps the figure honest: the point is fed through the actual
    production gate, not merely drawn outside a line we chose.
    """
    s = load_gate_settings()
    dx, dy = REJECTED_OFFSET

    assert gate_norm(np.array([dx]), np.array([dy]), s)[0] > 1.0

    survives = offsets_survive_geometry_gate(np.array([dx]), np.array([dy]), s)
    assert not survives[0], "the constructed bad detection must be blanked by the gate"

    r = float(np.hypot(dx, dy))
    assert r == pytest.approx(220.5, abs=0.2), "label reads '221 px'"


@requires_data
def test_every_plotted_accepted_point_survives_the_gate() -> None:
    """The blue cloud must contain nothing the gate would have removed."""
    s = load_gate_settings()
    off = load_subject_offsets()
    survives = offsets_survive_geometry_gate(off.dx, off.dy, s)
    assert survives.all()
    assert gate_norm(off.dx, off.dy, s).max() < 1.0


@requires_data
def test_peak_per_reads_well_inside_the_boundary() -> None:
    """This fly's peak PER sits comfortably inside the boundary (gate norm
    ~0.64, not near 1.0): its diagonal excursion is accommodated by the
    anisotropic gate without needing to hug the edge -- unlike the
    near-boundary framing of an earlier candidate subject."""
    s = load_gate_settings()
    off = load_subject_offsets()
    norms = gate_norm(off.dx, off.dy, s)
    assert 0.0 < norms.max() < 1.0, "accepted points must be inside the gate"
    assert norms.max() == pytest.approx(0.6392, abs=0.005)


from scripts.analysis.per_gate_rejection_figure import (
    CROP,
    FRAME_GAMMA,
    load_frame_crop,
)

requires_video = pytest.mark.skipif(
    not subject_video_path().exists(),
    reason="subject video not mounted on this machine",
)


def test_crop_contains_the_whole_gate() -> None:
    """The crop must show the full boundary plus margin, and stay inside the
    1080x1080 frame."""
    s = load_gate_settings()
    ex, ey = SUBJECT.eye_xy
    x0, y0, x1, y1 = CROP

    assert 0 <= x0 < x1 <= 1080
    assert 0 <= y0 < y1 <= 1080
    assert x0 <= ex - s.max_px and x1 >= ex + s.max_px
    assert y0 <= ey - s.dorsal_px and y1 >= ey + s.max_px
    # and room for the constructed rejection
    rdx, rdy = REJECTED_OFFSET
    assert x1 >= ex + rdx and y1 >= ey + rdy


@requires_video
def test_frame_crop_shape_and_type() -> None:
    x0, y0, x1, y1 = CROP
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert img.shape == (y1 - y0, x1 - x0, 3)
    assert img.dtype == np.uint8


@requires_video
def test_frame_crop_is_grayscale() -> None:
    """Ghosting is grayscale: the three channels must be identical."""
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert np.array_equal(img[:, :, 0], img[:, :, 1])
    assert np.array_equal(img[:, :, 1], img[:, :, 2])


@requires_video
def test_frame_crop_background_is_dark() -> None:
    """The frame keeps its natural polarity: a bright fly on a near-black
    background. Most of the crop IS background, so a low percentile must
    read genuinely dark -- draw_hero's white ink (HERO_INK) depends on this
    for contrast, the mirror image of the old light-surface guarantee."""
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert np.percentile(img, 20) < 50


@requires_video
def test_frame_crop_fly_is_bright() -> None:
    """The fly itself must read as genuinely bright, not merely 'less dark'
    -- the whole point of dropping the inversion/ghost treatment that
    produced a washed-out mean-gray blur."""
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert np.percentile(img, 90) > 190


@requires_video
def test_frame_crop_fly_is_still_visible() -> None:
    """The hero panel's entire purpose is that the audience sees a real fly.
    A flattened, low-contrast treatment would produce a small range here --
    this checks that real structure survives the stretch and gamma."""
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert np.percentile(img, 99) - np.percentile(img, 1) > 100


@requires_video
def test_gamma_lifts_shadow_detail_relative_to_a_raw_stretch() -> None:
    """Sanity check that the gamma parameter actually does the work: gamma
    < 1 (the default) must brighten the percentile-stretched frame relative
    to gamma=1.0 (no correction), since x**g >= x for x in [0, 1] when
    g < 1."""
    raw_stretch = load_frame_crop(subject_video_path(), SUBJECT.peak_frame, gamma=1.0)
    gamma_corrected = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert FRAME_GAMMA < 1.0
    assert gamma_corrected.mean() > raw_stretch.mean()


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.per_gate_rejection_figure import (  # noqa: E402
    ACCEPTED,
    EYE,
    HERO_INK,
    HERO_TEXTS,
    INK,
    REJECTED,
    draw_hero,
)


def test_palette_is_the_validated_one() -> None:
    """Validated all-pairs, light mode: worst CVD dE 13.0, normal-vision 16.3.
    Green/red was tested first and FAILED at deutan dE 4.1 -- never reinstate it.
    """
    assert ACCEPTED == "#2a78d6"
    assert REJECTED == "#eb6834"
    assert EYE == "#4a3aa7"
    for hexcode in (ACCEPTED, REJECTED, EYE):
        assert hexcode.lower() not in {"#0ca30c", "#d03b3b", "#008300", "#e34948"}


def test_hero_contains_no_sentences() -> None:
    """The figure carries panel titles, two direction words, gate values and
    short mark labels -- never prose. Longest permitted label is 3 words."""
    for text in HERO_TEXTS:
        assert len(text.split()) <= 3, f"too wordy for a slide: {text!r}"
    assert "dorsal" in HERO_TEXTS
    assert "ventral" in HERO_TEXTS


def _hero_axes(image=None):
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 10.0, 96.1]),
        dy=np.array([95.0, 110.0, 84.5]),
        frames=np.array([0, 1, 1233]),
        eye_xy=SUBJECT.eye_xy,
    )
    fig, ax = plt.subplots()
    draw_hero(ax, off, s, image)
    return fig, ax


def test_hero_draws_exactly_the_expected_text() -> None:
    fig, ax = _hero_axes()
    drawn = {t.get_text() for t in ax.texts if t.get_text()}
    assert drawn == set(HERO_TEXTS)
    plt.close(fig)


def test_hero_labels_the_acceptance_boundary_and_both_marks() -> None:
    fig, ax = _hero_axes()
    drawn = {t.get_text() for t in ax.texts}
    assert any("ACCEPTANCE" in t.upper() for t in drawn)
    assert "128 px" in drawn, "the accepted peak PER is direct-labelled"
    assert "221 px" in drawn, "the rejected example is direct-labelled"
    plt.close(fig)


def test_hero_gate_numbers_come_from_settings() -> None:
    """Change the config, and the numbers on the boundary change with it.

    A hardcoded-literal implementation (drawing "160"/"40" directly instead of
    ``str(int(settings.max_px))``/``str(int(settings.dorsal_px))``) would pass
    the old form of this test identically, because today's config happens to
    produce exactly those two strings and they are already baked into
    HERO_TEXTS. So this drives draw_hero with DIFFERENT synthetic settings and
    asserts the drawn numbers follow -- and separately re-checks the real
    settings values, to keep both the derivation and the production numbers
    pinned. Deliberately does not assert set-equality against HERO_TEXTS:
    under synthetic settings the drawn set legitimately differs from it.
    """
    synthetic = GateSettings(
        max_px=100.0, up_divisor=5.0, max_jump_px=80.0,
        norm_min_px=10.0, norm_max_px=100.0, three_fly_max_px=45.0,
    )
    assert synthetic.dorsal_px == 20.0

    fig, ax = _hero_axes()
    fig2, ax2 = plt.subplots()
    off = Offsets(
        dx=np.array([5.0, 10.0, 96.1]),
        dy=np.array([95.0, 110.0, 84.5]),
        frames=np.array([0, 1, 1233]),
        eye_xy=SUBJECT.eye_xy,
    )
    draw_hero(ax2, off, synthetic, None)
    drawn_synthetic = {t.get_text() for t in ax2.texts}
    assert "100" in drawn_synthetic
    assert "20" in drawn_synthetic
    assert "160" not in drawn_synthetic
    assert "40" not in drawn_synthetic
    plt.close(fig2)

    drawn = {t.get_text() for t in ax.texts}
    s = load_gate_settings()
    assert str(int(s.max_px)) in drawn
    assert str(int(s.dorsal_px)) in drawn
    assert "160" in drawn
    assert "40" in drawn
    plt.close(fig)


@requires_data
def test_hero_labels_are_pinned_to_the_underlying_data() -> None:
    """"128 px" and "221 px" are literal strings in draw_hero -- HERO_TEXTS is
    asserted as an exact set, so they cannot be computed dynamically inside
    draw_hero without changing that set. These assertions are what keep the
    literals honest: if the real data or the constructed rejection ever
    drifts, this fails loudly instead of the slide quietly lying.

    Both labels use the same rounding convention: round(). The rejection's
    true radius is 220.51 px, which rounds to 221, not 220 -- an earlier
    version of this test pinned the truncated ("220 px") value while the
    companion peak-PER assertion below already used round(), an
    inconsistency a reviewer flagged. The peak PER's true radius is
    ~127.92 px, which rounds to 128.
    """
    rdx, rdy = REJECTED_OFFSET
    assert round(float(np.hypot(rdx, rdy))) == 221, "label reads '221 px' (rounded)"

    off = load_subject_offsets()
    r = np.hypot(off.dx, off.dy)
    assert round(float(r.max())) == 128, "label reads '128 px' (rounded)"


def test_hero_rejected_marker_is_hollow() -> None:
    """Rejection is encoded by SHAPE first; colour is secondary, so nothing
    depends on hue alone. Uses all(), not any(): if there are several X
    markers, a solid-filled one hiding among hollow ones must fail this test,
    not slip through because at least one marker happened to be hollow."""
    fig, ax = _hero_axes()
    marks = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert marks, "the rejected example must be an X marker"
    assert all(m.get_markerfacecolor() in ("none", "None") for m in marks)
    plt.close(fig)


def test_hero_axis_is_equal_aspect() -> None:
    """Pixels are square; an unequal aspect would misrepresent the gate shape."""
    fig, ax = _hero_axes()
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_hero_accepted_cloud_matches_the_offsets() -> None:
    """The hero scatter is the figure's ONLY measured evidence. Replacing its
    data with fabricated points must be detectable: the scatter collection's
    offsets must equal Offsets.dx/dy exactly, in both count and coordinates."""
    off = Offsets(
        dx=np.array([5.0, 10.0, 96.1]),
        dy=np.array([95.0, 110.0, 84.5]),
        frames=np.array([0, 1, 1233]),
        eye_xy=SUBJECT.eye_xy,
    )
    s = load_gate_settings()
    fig, ax = plt.subplots()
    draw_hero(ax, off, s, None)

    assert len(ax.collections) == 1, "expected exactly one scatter collection"
    drawn = ax.collections[0].get_offsets()
    assert drawn.shape == (len(off.dx), 2)
    np.testing.assert_allclose(np.asarray(drawn)[:, 0], off.dx)
    np.testing.assert_allclose(np.asarray(drawn)[:, 1], off.dy)
    plt.close(fig)


def test_hero_peak_marker_sits_at_the_true_argmax() -> None:
    """pdx, pdy in draw_hero must come from argmax(hypot(dx, dy)), not from
    this subject's hardcoded golden values (96.1, 84.5). These synthetic
    offsets put the true peak at neither the last point nor near the golden
    values, so a hardcoded implementation draws the marker in the wrong
    place."""
    dx = np.array([5.0, -70.0, 15.0, 8.0])
    dy = np.array([40.0, 95.0, 200.0, 20.0])
    off = Offsets(dx=dx, dy=dy, frames=np.arange(4), eye_xy=SUBJECT.eye_xy)
    s = load_gate_settings()
    fig, ax = plt.subplots()
    draw_hero(ax, off, s, None)

    accepted_rgba = matplotlib.colors.to_rgba(ACCEPTED)
    peak_dots = [
        ln for ln in ax.lines
        if ln.get_marker() == "o"
        and matplotlib.colors.to_rgba(ln.get_markerfacecolor()) == accepted_rgba
    ]
    assert len(peak_dots) == 1, "expected exactly one peak-PER dot"
    px, py = peak_dots[0].get_xdata()[0], peak_dots[0].get_ydata()[0]
    assert (px, py) == pytest.approx((15.0, 200.0)), (
        "peak marker must sit at argmax(r), not at a hardcoded coordinate"
    )
    plt.close(fig)


def test_hero_draws_the_boundary_polyline_and_eye_anchor() -> None:
    """Drawing the acceptance boundary or the eye anchor as empty arrays
    (nothing rendered) currently passes every other test. This asserts the
    boundary ring's vertex count and extremes match the gate (160 lateral,
    160 ventral, -40 dorsal), and that a 'P' eye-anchor marker sits at
    (0, 0)."""
    off = Offsets(
        dx=np.array([5.0, 10.0, 96.1]),
        dy=np.array([95.0, 110.0, 84.5]),
        frames=np.array([0, 1, 1233]),
        eye_xy=SUBJECT.eye_xy,
    )
    s = load_gate_settings()
    fig, ax = plt.subplots()
    draw_hero(ax, off, s, None)

    # HERO_INK (white), not the figure-wide INK (near-black): the hero
    # background is now a dark video frame, so the boundary must be drawn in
    # the panel's own light-on-dark ink to stay legible.
    hero_ink_rgba = matplotlib.colors.to_rgba(HERO_INK)
    ink_rgba = matplotlib.colors.to_rgba(INK)
    boundary = [
        ln for ln in ax.lines
        if ln.get_marker() in ("", "None", None)
        and matplotlib.colors.to_rgba(ln.get_color()) == hero_ink_rgba
        and ln.get_linewidth() == pytest.approx(2.5)
    ]
    assert len(boundary) == 1, "expected exactly one boundary polyline"
    assert not [
        ln for ln in ax.lines
        if ln.get_marker() in ("", "None", None)
        and matplotlib.colors.to_rgba(ln.get_color()) == ink_rgba
    ], "the boundary must not use the figure-wide (near-black) INK on the dark hero panel"
    bx, by = boundary[0].get_xdata(), boundary[0].get_ydata()
    assert len(bx) == 361, "360 traced points plus the closing vertex"
    assert bx.max() == pytest.approx(160.0, abs=0.5)
    assert bx.min() == pytest.approx(-160.0, abs=0.5)
    assert by.max() == pytest.approx(160.0, abs=0.5), "ventral (dy > 0) is generous"
    assert by.min() == pytest.approx(-40.0, abs=0.5), "dorsal (dy < 0) is tightened"

    eye_rgba = matplotlib.colors.to_rgba(EYE)
    eye_marks = [
        ln for ln in ax.lines
        if ln.get_marker() == "P"
        and matplotlib.colors.to_rgba(ln.get_color()) == eye_rgba
    ]
    assert len(eye_marks) == 1, "expected exactly one eye-anchor marker"
    ex, ey = eye_marks[0].get_xdata()[0], eye_marks[0].get_ydata()[0]
    assert (ex, ey) == pytest.approx((0.0, 0.0))
    plt.close(fig)


def test_hero_marks_use_a_black_halo_not_white() -> None:
    """The hero background is now a bright fly on a dark surface, so relief
    strokes must be BLACK (to separate a mark from the bright fly), not the
    white halo that made sense on the old pale/ghosted background. Every
    text and line artist in draw_hero that carries a path effect must use a
    black -- not white -- stroke."""
    off = Offsets(
        dx=np.array([5.0, 10.0, 96.1]),
        dy=np.array([95.0, 110.0, 84.5]),
        frames=np.array([0, 1, 1233]),
        eye_xy=SUBJECT.eye_xy,
    )
    s = load_gate_settings()
    fig, ax = plt.subplots()
    draw_hero(ax, off, s, None)

    black_rgba = matplotlib.colors.to_rgba("black")
    white_rgba = matplotlib.colors.to_rgba("white")

    haloed_artists = [a for a in (*ax.lines, *ax.texts) if a.get_path_effects()]
    assert haloed_artists, "expected at least one haloed artist in draw_hero"
    for artist in haloed_artists:
        for effect in artist.get_path_effects():
            foreground = effect._gc.get("foreground")
            assert matplotlib.colors.to_rgba(foreground) == black_rgba, (
                f"{artist!r} carries a non-black halo: {foreground!r}"
            )
            assert matplotlib.colors.to_rgba(foreground) != white_rgba
    plt.close(fig)


from scripts.analysis.per_gate_rejection_figure import (
    MUTED,
    SCHEMATIC_PANELS,
    draw_cap_panel,
    draw_jump_panel,
    draw_norm_panel,
    draw_release_panel,
)


def test_schematic_panels_are_declared() -> None:
    """This fly has 2 flies and a max jump of 5.7 px, so the cap, release and
    jump panels are illustrations -- they must be flagged, not passed off as data."""
    assert SCHEMATIC_PANELS == frozenset({"cap", "release", "jump"})
    assert "norm" not in SCHEMATIC_PANELS, "panel E plots this fly's real spread"


def test_schematic_panels_actually_draw_a_dashed_border() -> None:
    """SCHEMATIC_PANELS is just a set of strings -- it says nothing about what
    ends up on the axes. Panels B (cap), C (release) and D (jump) are
    illustrations because this subject has only 2 flies (the >=3-fly release
    never fires) and a real max displacement of 5.7 px (the 80 px jump gate is
    never approached), so their marks must be visually flagged as invented,
    not measured. Panel E (normalize) plots this fly's REAL radius spread, so
    it must be left unmarked. If someone deleted the `_mark_schematic(ax)`
    call from a schematic panel, every other test would still pass -- this is
    the one that would catch it."""
    s = load_gate_settings()
    off = Offsets(dx=np.array([0.0]), dy=np.array([100.0]),
                  frames=np.array([0]), eye_xy=SUBJECT.eye_xy)
    muted_rgba = matplotlib.colors.to_rgba(MUTED)

    for draw in (
        lambda ax: draw_cap_panel(ax, 2),
        lambda ax: draw_release_panel(ax, s),
        lambda ax: draw_jump_panel(ax, s),
    ):
        fig, ax = plt.subplots()
        draw(ax)
        for spine in ax.spines.values():
            assert spine.get_visible()
            assert spine.get_linestyle() != "solid"
            assert matplotlib.colors.to_rgba(spine.get_edgecolor()) == muted_rgba
        plt.close(fig)

    fig, ax = plt.subplots()
    draw_norm_panel(ax, off, s)
    assert not any(spine.get_visible() for spine in ax.spines.values()), (
        "panel E plots real data and must not carry the schematic border"
    )
    plt.close(fig)


def test_jump_ring_is_centred_on_the_last_ACCEPTED_point() -> None:
    """The rule the panel exists to show: a rejected point never becomes the new
    reference, so the 80 px ring stays on the last accepted position."""
    s = load_gate_settings()
    fig, ax = plt.subplots()
    info = draw_jump_panel(ax, s)
    assert info["ring_radius"] == s.max_jump_px

    # The DRAWN circle must use the same value, or the test proves nothing about
    # the picture. Panel D works in px data coordinates for exactly this reason.
    circles = [p for p in ax.patches if isinstance(p, plt.Circle)]
    assert len(circles) == 1
    assert circles[0].get_radius() == pytest.approx(s.max_jump_px)
    assert circles[0].center == pytest.approx(info["ring_centre"])

    # The ring must sit on the LAST ACCEPTED point, never on the rejected one.
    # Asserting the exact coordinates is what makes this test able to fail.
    assert info["ring_centre"] == pytest.approx((130.0, 114.0))
    assert info["ring_centre"] != pytest.approx((250.0, 112.0))
    plt.close(fig)


def test_jump_ring_radius_tracks_settings_not_a_hardcoded_80() -> None:
    """config_new.yaml's max_jump_px happens to BE 80, so a circle built with a
    literal ``80`` would slip past the test above undetected. Feed a settings
    object with a different value and require the drawn circle to follow it --
    this is what makes "not hardcoded" an actual, falsifiable claim."""
    s = GateSettings(max_px=160.0, up_divisor=4.0, max_jump_px=55.0,
                      norm_min_px=10.0, norm_max_px=160.0, three_fly_max_px=99.0)
    fig, ax = plt.subplots()
    info = draw_jump_panel(ax, s)
    assert info["ring_radius"] == 55.0

    circles = [p for p in ax.patches if isinstance(p, plt.Circle)]
    assert len(circles) == 1
    assert circles[0].get_radius() == pytest.approx(55.0)
    assert circles[0].get_radius() != pytest.approx(80.0)
    plt.close(fig)


def test_norm_panel_band_edges_come_from_settings() -> None:
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([0.0, 0.0]), dy=np.array([94.4, 145.8]),
        frames=np.array([0, 1]), eye_xy=SUBJECT.eye_xy,
    )
    fig, ax = plt.subplots()
    info = draw_norm_panel(ax, off, s)
    assert info["band"] == (s.norm_min_px, s.norm_max_px) == (10.0, 160.0)
    drawn = {t.get_text() for t in ax.texts}
    assert "10" in drawn and "160" in drawn
    plt.close(fig)


def test_norm_panel_band_edges_track_settings_not_hardcoded_10_and_160() -> None:
    """config_new.yaml's class2_min/max happen to BE 10/160, so text literals
    "10"/"160" written straight into draw_norm_panel would slip past the test
    above undetected -- the same coincidence already caught on the jump ring.
    Feed settings with different band edges and require the drawn numbers,
    and the returned band, to follow them."""
    s = GateSettings(max_px=160.0, up_divisor=4.0, max_jump_px=80.0,
                      norm_min_px=25.0, norm_max_px=90.0, three_fly_max_px=99.0)
    off = Offsets(
        dx=np.array([0.0, 0.0]), dy=np.array([94.4, 145.8]),
        frames=np.array([0, 1]), eye_xy=SUBJECT.eye_xy,
    )
    fig, ax = plt.subplots()
    info = draw_norm_panel(ax, off, s)
    assert info["band"] == (25.0, 90.0)
    drawn = {t.get_text() for t in ax.texts}
    assert "25" in drawn and "90" in drawn
    assert "10" not in drawn and "160" not in drawn
    plt.close(fig)


def test_norm_panel_shaded_band_matches_the_gate_not_the_full_axis() -> None:
    """Widening the axhspan to the full axis (xmin=0, xmax=1) passes every
    other norm-panel test, since only the tick lines/labels were asserted.
    This inspects the Rectangle patch axhspan adds and converts its
    axes-fraction x-extent back to data coordinates via the axes' final
    xlim, then checks it lands on the settings band -- not on [0, axis_max]."""
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([0.0, 0.0]), dy=np.array([94.4, 145.8]),
        frames=np.array([0, 1]), eye_xy=SUBJECT.eye_xy,
    )
    fig, ax = plt.subplots()
    draw_norm_panel(ax, off, s)

    rects = [p for p in ax.patches if isinstance(p, plt.Rectangle)]
    assert len(rects) == 1, "expected exactly one shaded band patch"
    patch = rects[0]

    xlim0, xlim1 = ax.get_xlim()
    lo_frac = patch.get_x()
    hi_frac = lo_frac + patch.get_width()
    lo_data = xlim0 + lo_frac * (xlim1 - xlim0)
    hi_data = xlim0 + hi_frac * (xlim1 - xlim0)

    assert lo_data == pytest.approx(s.norm_min_px, abs=0.5)
    assert hi_data == pytest.approx(s.norm_max_px, abs=0.5)
    # and provably NOT the full axis
    assert lo_frac > 0.001
    assert hi_frac < 0.999
    plt.close(fig)


def test_norm_panel_spread_bar_matches_the_offsets_p1_p99() -> None:
    """Panel E is the one panel excluded from SCHEMATIC_PANELS because it
    plots this fly's REAL radius spread. Replacing that spread bar's x-data
    with a hardcoded constant like [0.0, 200.0], or the max dot with a
    hardcoded 175.0, passed every prior test -- none of them inspected this
    panel's actual line data. Uses offsets whose p1/p99/max are irregular
    floats no plausible hardcoded literal would match."""
    s = load_gate_settings()
    dx = np.zeros(20)
    dy = np.array([
        11.3, 108.4, 76.2, 43.9, 62.7, 91.5, 34.1, 121.8, 55.6, 68.0,
        99.9, 27.4, 84.3, 116.7, 47.2, 73.6, 59.8, 102.1, 38.5, 128.9,
    ])
    off = Offsets(dx=dx, dy=dy, frames=np.arange(20), eye_xy=SUBJECT.eye_xy)
    r = np.hypot(dx, dy)
    expected_lo, expected_hi = np.percentile(r, 1), np.percentile(r, 99)
    expected_max = r.max()

    fig, ax = plt.subplots()
    draw_norm_panel(ax, off, s)

    accepted_rgba = matplotlib.colors.to_rgba(ACCEPTED)
    spread_bars = [
        ln for ln in ax.lines
        if matplotlib.colors.to_rgba(ln.get_color()) == accepted_rgba
        and ln.get_linewidth() == pytest.approx(4.0)
    ]
    assert len(spread_bars) == 1, "expected exactly one spread bar"
    xdata = np.asarray(spread_bars[0].get_xdata())
    assert xdata == pytest.approx([expected_lo, expected_hi])

    max_dots = [
        ln for ln in ax.lines
        if ln.get_marker() == "o"
        and matplotlib.colors.to_rgba(ln.get_color()) == accepted_rgba
    ]
    assert len(max_dots) == 1, "expected exactly one max-displacement dot"
    assert max_dots[0].get_xdata()[0] == pytest.approx(expected_max)
    plt.close(fig)


def test_release_panel_shows_the_three_fly_limit() -> None:
    """The >=3-fly release panel must print settings.three_fly_max_px -- the
    value yolo_infer._max_valid_eye_prob_distance_px actually reads
    (distance_limits.three_fly_max_eye_prob_distance_px) -- NOT settings.max_px
    (proboscis_filter.max_eye_prob_distance_px), which gates a different rule.
    Both happen to be 160.0 in config_new.yaml today, so this alone cannot
    prove the panel reads the right key; see the synthetic-settings test below
    for that."""
    s = load_gate_settings()
    fig, ax = plt.subplots()
    draw_release_panel(ax, s)
    drawn = {t.get_text() for t in ax.texts}
    assert str(int(s.three_fly_max_px)) in drawn
    marks = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert marks, "the released binding terminates in an X"
    plt.close(fig)


def test_release_panel_limit_tracks_three_fly_max_px_not_max_px() -> None:
    """config_new.yaml's max_eye_prob_distance_px (max_px, the spatial gate)
    and three_fly_max_eye_prob_distance_px (three_fly_max_px, the >=3-fly
    release rule) happen to BOTH be 160.0 today -- the exact coincidence that
    let the panel cite the wrong config key (max_px) for seven prior fix
    rounds. Feed them DIFFERENT values here: if draw_release_panel ever reads
    settings.max_px instead of settings.three_fly_max_px again, "160" (from
    max_px) would appear instead of "140" (from three_fly_max_px), and this
    test would catch it -- a test where the two fields are equal cannot."""
    s = GateSettings(max_px=160.0, up_divisor=4.0, max_jump_px=80.0,
                      norm_min_px=10.0, norm_max_px=160.0, three_fly_max_px=140.0)
    fig, ax = plt.subplots()
    draw_release_panel(ax, s)
    drawn = {t.get_text() for t in ax.texts}
    assert "140" in drawn, "must draw three_fly_max_px, not max_px"
    assert "160" not in drawn, "must NOT draw max_px -- that is the wrong config key"
    plt.close(fig)


@pytest.mark.parametrize("n_flies", [1, 2, 3])
def test_cap_panel_keeps_the_highest_confidence_detections(n_flies: int) -> None:
    """n_flies resolved -> the n_flies highest-confidence proboscis detections
    are kept (filled circles) and the rest dropped (hollow X). Parametrised
    over n_flies: a hardcoded ``i < 2`` with a literal "2-fly cap" title would
    pass a single n_flies=2 case but fail here for 1 and 3, since neither the
    kept/dropped counts nor the title would track n_flies."""
    fig, ax = plt.subplots()
    draw_cap_panel(ax, n_flies=n_flies)
    drawn = {t.get_text() for t in ax.texts}
    assert f"{n_flies}-fly cap" in drawn
    kept = [ln for ln in ax.lines if ln.get_marker() == "o"
            and ln.get_markerfacecolor() not in ("none", "None")]
    dropped = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert len(kept) == n_flies
    assert len(dropped) == 4 - n_flies
    plt.close(fig)


def test_every_side_panel_title_is_short() -> None:
    """Minimal-text rule applies to the side strip too."""
    s = load_gate_settings()
    off = Offsets(dx=np.array([0.0]), dy=np.array([100.0]),
                  frames=np.array([0]), eye_xy=SUBJECT.eye_xy)
    for draw in (
        lambda ax: draw_cap_panel(ax, 2),
        lambda ax: draw_release_panel(ax, s),
        lambda ax: draw_jump_panel(ax, s),
        lambda ax: draw_norm_panel(ax, off, s),
    ):
        fig, ax = plt.subplots()
        draw(ax)
        for t in ax.texts:
            assert len(t.get_text().split()) <= 3, f"too wordy: {t.get_text()!r}"
        plt.close(fig)


from scripts.analysis.per_gate_rejection_figure import (
    CAPTION,
    FIGSIZE,
    make_figure,
    save_figure,
)


def test_default_canvas_is_a_16_by_9_slide() -> None:
    assert FIGSIZE == (13.33, 7.5)
    assert abs(FIGSIZE[0] / FIGSIZE[1] - 16 / 9) < 0.01


def test_caption_declares_what_is_measured_and_what_is_constructed() -> None:
    """The one honesty sentence. It belongs in the caption, not in the figure."""
    assert "constructed" in CAPTION.lower()
    assert "measured" in CAPTION.lower()
    assert "config_new.yaml" in CAPTION
    assert "not inverted" in CAPTION.lower(), (
        "load_frame_crop no longer inverts the frame -- the fly stays bright "
        "on a dark background -- and the caption must say so, not still claim "
        "the old inverted polarity"
    )
    assert "contrast" in CAPTION.lower() and "gamma" in CAPTION.lower(), (
        "load_frame_crop grayscales, percentile-stretches, then applies gamma -- "
        "the caption must describe that treatment"
    )


def test_caption_is_pure_ascii() -> None:
    """Arial (this deck's font) has no glyph for U+2717 BALLOT X ('X'): a
    stray non-ASCII character in CAPTION renders as a hollow missing-glyph
    box on the slide, as '✗' did before it was replaced with a plain
    'X'. This guards CAPTION only -- the '>=3 flies' panel title legitimately
    uses U+2265 (GREATER-THAN OR EQUAL TO), which Arial does render."""
    assert CAPTION.isascii(), "non-ASCII character would risk a missing glyph box"


def test_figure_has_five_panels() -> None:
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 96.1]), dy=np.array([95.0, 84.5]),
        frames=np.array([0, 1233]), eye_xy=SUBJECT.eye_xy,
    )
    fig = make_figure(off, s, image=None)
    assert len(fig.axes) == 5, "one hero + four gate panels"
    plt.close(fig)


def test_make_figure_pins_the_cap_panel_to_three_flies() -> None:
    """The trial has 3 flies (verified property of the new subject), so
    make_figure must call draw_cap_panel(n_flies=3) -- title "3-fly cap",
    3 kept (filled circles) and 1 dropped (hollow X). A make_figure that
    still hardcoded n_flies=2 would pass test_figure_has_five_panels
    identically, since that test only counts axes; this inspects the cap
    panel's own title and marks."""
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 96.1]), dy=np.array([95.0, 84.5]),
        frames=np.array([0, 1233]), eye_xy=SUBJECT.eye_xy,
    )
    fig = make_figure(off, s, image=None)

    cap_ax = fig.axes[1]
    drawn = {t.get_text() for t in cap_ax.texts}
    assert "3-fly cap" in drawn
    assert "2-fly cap" not in drawn

    kept = [ln for ln in cap_ax.lines if ln.get_marker() == "o"
            and ln.get_markerfacecolor() not in ("none", "None")]
    dropped = [ln for ln in cap_ax.lines if ln.get_marker() in {"x", "X"}]
    assert len(kept) == 3
    assert len(dropped) == 1
    plt.close(fig)


def test_save_writes_png_pdf_and_svg(tmp_path: Path) -> None:
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 96.1]), dy=np.array([95.0, 84.5]),
        frames=np.array([0, 1233]), eye_xy=SUBJECT.eye_xy,
    )
    fig = make_figure(off, s, image=None)
    paths = save_figure(fig, outdir=tmp_path)

    assert {p.suffix for p in paths} == {".png", ".pdf", ".svg"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_svg_keeps_text_editable(tmp_path: Path) -> None:
    """svg.fonttype='none' -- so the labels stay editable in Illustrator."""
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 96.1]), dy=np.array([95.0, 84.5]),
        frames=np.array([0, 1233]), eye_xy=SUBJECT.eye_xy,
    )
    fig = make_figure(off, s, image=None)
    paths = save_figure(fig, outdir=tmp_path)
    svg = next(p for p in paths if p.suffix == ".svg").read_text(encoding="utf-8")
    assert "<text" in svg, "text was converted to paths -- not editable"
    assert "ACCEPTANCE BOUNDARY" in svg


def test_pdf_and_ps_fonttype_keep_text_editable() -> None:
    """fonttype 42 (TrueType) embeds real, editable glyph outlines in PDF/PS
    output; fonttype 3 (Type 3 bitmap) does not -- labels become uneditable
    in Illustrator/Inkscape. Only SVG editability was previously guarded, so
    flipping pdf.fonttype 42->3 passed every other test in this file."""
    assert plt.rcParams["pdf.fonttype"] == 42, (
        "42 keeps PDF text editable in Illustrator/Inkscape; 3 does not"
    )
    assert plt.rcParams["ps.fonttype"] == 42, (
        "42 keeps PS/EPS text editable in Illustrator/Inkscape; 3 does not"
    )
