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
    """The five gate values the figure reports come from config_new.yaml."""
    s = load_gate_settings()
    assert s.max_px == 160.0
    assert s.up_divisor == 4.0
    assert s.dorsal_px == 40.0
    assert s.max_jump_px == 80.0
    assert s.norm_min_px == 10.0
    assert s.norm_max_px == 160.0


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
        "  class2_max: 99.0\n",
        encoding="utf-8",
    )
    s = load_gate_settings(alt)
    assert s.max_px == 99.0
    assert s.up_divisor == 3.0
    assert s.dorsal_px == 33.0
    assert s.max_jump_px == 55.0
    assert s.norm_min_px == 5.0
    assert s.norm_max_px == 99.0


def test_gate_settings_rejects_missing_keys(tmp_path: Path) -> None:
    """A config without the gate blocks must fail loudly, not silently default."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_gate_settings(empty)


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
    assert SUBJECT.dataset == "3Oct-Control-24-0.1"
    assert SUBJECT.trial_rel == "july_26_batch_1/july_26_batch_1_testing_1"
    assert SUBJECT.slot == "fly1"
    assert SUBJECT.odor == "3-Octonol"
    assert SUBJECT.eye_xy == (845.0, 183.0)
    assert SUBJECT.peak_frame == 1102


@requires_data
def test_subject_offsets_match_spec() -> None:
    """Golden numbers from the candidate scan. If the parquet is ever
    reprocessed, this fails loudly rather than the figure quietly changing."""
    off = load_subject_offsets()
    assert len(off.dx) == 3605, "subject was chosen for its perfect detection record"
    assert len(off.dx) == len(off.dy) == len(off.frames)

    ex, ey = off.eye_xy
    assert round(ex) == 845 and round(ey) == 183

    r = np.hypot(off.dx, off.dy)
    assert r.max() == pytest.approx(145.8, abs=0.1)

    peak_i = int(np.argmax(r))
    assert off.frames[peak_i] == SUBJECT.peak_frame
    assert off.dx[peak_i] == pytest.approx(28.9, abs=0.1)
    assert off.dy[peak_i] == pytest.approx(142.8, abs=0.1)


@requires_data
def test_subject_per_is_ventral() -> None:
    """The figure's argument: PER is a near-vertical ventral excursion, so the
    gate is generous ventrally and tight dorsally."""
    off = load_subject_offsets()
    assert off.dy.min() > 0, "this fly never goes dorsal"
    assert np.abs(off.dx).max() < 40.0
    assert off.dy.max() > 140.0


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
    assert r == pytest.approx(220.5, abs=0.2), "label reads '220 px'"


@requires_data
def test_every_plotted_accepted_point_survives_the_gate() -> None:
    """The blue cloud must contain nothing the gate would have removed."""
    s = load_gate_settings()
    off = load_subject_offsets()
    survives = offsets_survive_geometry_gate(off.dx, off.dy, s)
    assert survives.all()
    assert gate_norm(off.dx, off.dy, s).max() < 1.0


@requires_data
def test_peak_per_is_near_the_boundary_but_inside() -> None:
    """The whole point of this fly: it rides the edge without crossing it."""
    s = load_gate_settings()
    off = load_subject_offsets()
    norms = gate_norm(off.dx, off.dy, s)
    assert 0.80 < norms.max() < 1.0
    assert norms.max() == pytest.approx(0.830, abs=0.005)


from scripts.analysis.per_gate_rejection_figure import (
    CROP,
    GHOST_BLEND,
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
def test_frame_crop_surface_is_light() -> None:
    """The background the coloured overlay marks sit on must be genuinely
    light -- the palette validator WARNs at every mid-gray surface tested
    (orange falls to 1.61:1 on #b8b8b6), so a dim surface would void the
    overlay palette's contrast guarantees."""
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert np.percentile(img, 90) > 230


@requires_video
def test_frame_crop_fly_is_still_visible() -> None:
    """The hero panel's entire purpose is that the audience sees a real fly.
    The naive blend-toward-white approach (no contrast stretch, no invert)
    produced a range of 14 here -- a blank rectangle that passed every other
    assertion. This checks that real structure survives ghosting."""
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert np.percentile(img, 99) - np.percentile(img, 1) > 100


@requires_video
def test_ghost_blend_zero_returns_the_unghosted_frame() -> None:
    """Sanity check that the ghosting parameter actually does the work."""
    raw = load_frame_crop(subject_video_path(), SUBJECT.peak_frame, ghost=0.0)
    ghosted = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert GHOST_BLEND > 0
    assert ghosted.mean() > raw.mean()


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.per_gate_rejection_figure import (  # noqa: E402
    ACCEPTED,
    EYE,
    HERO_TEXTS,
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
        dx=np.array([5.0, 10.0, 28.9]),
        dy=np.array([95.0, 110.0, 142.8]),
        frames=np.array([0, 1, 1102]),
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
    assert "146 px" in drawn, "the accepted peak PER is direct-labelled"
    assert "220 px" in drawn, "the rejected example is direct-labelled"
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
        norm_min_px=10.0, norm_max_px=100.0,
    )
    assert synthetic.dorsal_px == 20.0

    fig, ax = _hero_axes()
    fig2, ax2 = plt.subplots()
    off = Offsets(
        dx=np.array([5.0, 10.0, 28.9]),
        dy=np.array([95.0, 110.0, 142.8]),
        frames=np.array([0, 1, 1102]),
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
    """"146 px" and "220 px" are literal strings in draw_hero -- HERO_TEXTS is
    asserted as an exact set, so they cannot be computed dynamically inside
    draw_hero without changing that set. These assertions are what keep the
    literals honest: if the real data or the constructed rejection ever
    drifts, this fails loudly instead of the slide quietly lying.

    The two labels use different rounding conventions on purpose, and the
    assertions follow the label they pin rather than a uniform rule:
    the rejection's true radius is 220.51 px, and "220 px" is that value
    TRUNCATED (matching the module's own "r = 220.5 px" comment next to
    REJECTED_OFFSET) -- round() would give 221 and wrongly fail here. The
    peak PER's true radius is ~145.75 px, and "146 px" is that value
    ROUNDED per the brief's ambiguity ruling -- int() would give 145 and
    wrongly fail here.
    """
    rdx, rdy = REJECTED_OFFSET
    assert int(np.hypot(rdx, rdy)) == 220, "label reads '220 px' (truncated)"

    off = load_subject_offsets()
    r = np.hypot(off.dx, off.dy)
    assert round(float(r.max())) == 146, "label reads '146 px' (rounded)"


def test_hero_rejected_marker_is_hollow() -> None:
    """Rejection is encoded by SHAPE first; colour is secondary, so nothing
    depends on hue alone."""
    fig, ax = _hero_axes()
    marks = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert marks, "the rejected example must be an X marker"
    assert any(m.get_markerfacecolor() in ("none", "None") for m in marks)
    plt.close(fig)


def test_hero_axis_is_equal_aspect() -> None:
    """Pixels are square; an unequal aspect would misrepresent the gate shape."""
    fig, ax = _hero_axes()
    assert ax.get_aspect() == 1.0
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
                      norm_min_px=10.0, norm_max_px=160.0)
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
                      norm_min_px=25.0, norm_max_px=90.0)
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


def test_release_panel_shows_the_three_fly_limit() -> None:
    s = load_gate_settings()
    fig, ax = plt.subplots()
    draw_release_panel(ax, s)
    drawn = {t.get_text() for t in ax.texts}
    assert str(int(s.max_px)) in drawn
    marks = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert marks, "the released binding terminates in an X"
    plt.close(fig)


def test_release_panel_limit_tracks_settings_not_a_hardcoded_160() -> None:
    """Same coincidence as the jump ring and the norm band: config_new.yaml's
    max_eye_prob_distance_px happens to BE 160, so a literal "160" written
    into draw_release_panel would pass the test above undetected. Feed a
    different max_px and require the drawn label to follow it."""
    s = GateSettings(max_px=140.0, up_divisor=4.0, max_jump_px=80.0,
                      norm_min_px=10.0, norm_max_px=160.0)
    fig, ax = plt.subplots()
    draw_release_panel(ax, s)
    drawn = {t.get_text() for t in ax.texts}
    assert "140" in drawn
    assert "160" not in drawn
    plt.close(fig)


def test_cap_panel_keeps_the_highest_confidence_detections() -> None:
    """Two flies resolved -> the two highest-confidence proboscis detections are
    kept and the rest dropped."""
    fig, ax = plt.subplots()
    draw_cap_panel(ax, n_flies=2)
    drawn = {t.get_text() for t in ax.texts}
    assert "2-fly cap" in drawn
    kept = [ln for ln in ax.lines if ln.get_marker() == "o"
            and ln.get_markerfacecolor() not in ("none", "None")]
    dropped = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert len(kept) == 2
    assert len(dropped) == 2
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
