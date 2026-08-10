"""Tests for the fly mounting workflow methods figure.

The figure is a 3x2 photo storyboard (panels A-F). Panels A-E are rebuilt from
the raw mounting photographs; panel F is the archived UV/IR frame preserved
from the original thesis figure and must not change.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "fly_mounting_workflow_figure.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fly_mounting_workflow_figure", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fm = _load_module()


# --------------------------------------------------------------------------
# square_crop_box
# --------------------------------------------------------------------------
def test_square_crop_box_is_square():
    left, upper, right, lower = fm.square_crop_box(1267, 1147, (0.0, 0.0, 1.0, 1.0))
    assert (right - left) == (lower - upper)


def test_square_crop_box_stays_inside_the_image():
    left, upper, right, lower = fm.square_crop_box(827, 1213, (0.0, 0.03, 1.0, 0.85))
    assert 0 <= left < right <= 827
    assert 0 <= upper < lower <= 1213


def test_square_crop_box_side_is_capped_by_the_short_edge():
    # ROI wants a 995 px square but the image is only 827 px wide.
    _, upper, _, lower = fm.square_crop_box(827, 1213, (0.0, 0.03, 1.0, 0.85))
    assert (lower - upper) == 827


def test_square_crop_box_centres_on_the_roi_when_it_fits():
    left, upper, right, lower = fm.square_crop_box(1000, 1000, (0.2, 0.2, 0.4, 0.4))
    assert pytest.approx((left + right) / 2, abs=1) == 300
    assert pytest.approx((upper + lower) / 2, abs=1) == 300


def test_square_crop_box_contains_the_roi_when_the_image_allows_it():
    roi = (0.1, 0.3, 0.5, 0.6)
    left, upper, right, lower = fm.square_crop_box(2000, 1600, roi)
    assert left <= roi[0] * 2000 and right >= roi[2] * 2000
    assert upper <= roi[1] * 1600 and lower >= roi[3] * 1600


def test_square_crop_box_shifts_instead_of_leaving_the_frame():
    # ROI hugging the right edge: the square must slide left, not overflow.
    left, _, right, _ = fm.square_crop_box(1000, 400, (0.9, 0.0, 1.0, 1.0))
    assert right == 1000
    assert left == 600


# --------------------------------------------------------------------------
# stretch_contrast
# --------------------------------------------------------------------------
def test_stretch_contrast_stays_in_unit_range():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.35, 0.55, size=(16, 16, 3))
    out = fm.stretch_contrast(img)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_stretch_contrast_expands_a_washed_out_image():
    rng = np.random.default_rng(1)
    img = rng.uniform(0.40, 0.60, size=(32, 32, 3))
    out = fm.stretch_contrast(img)
    assert out.ptp() > img.ptp()


def test_stretch_contrast_is_monotonic_within_a_channel():
    img = np.linspace(0.3, 0.7, 64).reshape(8, 8, 1).repeat(3, axis=2)
    out = fm.stretch_contrast(img)
    flat_in = img[..., 0].ravel()
    flat_out = out[..., 0].ravel()
    order = np.argsort(flat_in)
    assert np.all(np.diff(flat_out[order]) >= -1e-9)


def test_stretch_contrast_leaves_a_flat_image_alone():
    img = np.full((8, 8, 3), 0.5)
    out = fm.stretch_contrast(img)
    assert np.allclose(out, 0.5)


def test_stretch_contrast_drops_an_alpha_channel():
    img = np.dstack([np.full((8, 8, 3), 0.5), np.ones((8, 8, 1))])
    out = fm.stretch_contrast(img)
    assert out.shape[2] == 3


def test_stretch_contrast_strength_zero_is_a_no_op():
    rng = np.random.default_rng(2)
    img = rng.uniform(0.4, 0.6, size=(8, 8, 3))
    assert np.allclose(fm.stretch_contrast(img, strength=0.0), img)


def test_stretch_contrast_neutralises_a_colour_cast():
    """The stereoscope photos are blue-cyan: red tops out far below blue.

    A stretch shared across channels amplifies that cast; the per-channel
    stretch has to bring the channel ranges back together.
    """
    rng = np.random.default_rng(4)
    base = rng.uniform(0.2, 1.0, size=(64, 64))
    img = np.dstack([base * 0.65, base * 0.82, base * 0.90])  # blue-cast
    out = fm.stretch_contrast(img)
    tops = [np.percentile(out[..., c], 99.5) for c in range(3)]
    assert max(tops) - min(tops) < 0.05


def test_stretch_contrast_strength_blends_toward_the_full_stretch():
    rng = np.random.default_rng(3)
    img = rng.uniform(0.4, 0.6, size=(16, 16, 3))
    half = fm.stretch_contrast(img, strength=0.5)
    full = fm.stretch_contrast(img, strength=1.0)
    assert img.ptp() < half.ptp() < full.ptp()


# --------------------------------------------------------------------------
# panel table
# --------------------------------------------------------------------------
def test_six_panels_labelled_a_through_f():
    assert [p.letter for p in fm.PANELS] == list("ABCDEF")


def test_panels_a_to_e_use_photos_one_to_five_in_order():
    assert [p.source for p in fm.PANELS[:5]] == [
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
    ]


def test_panel_f_is_the_preserved_uv_frame():
    panel_f = fm.PANELS[5]
    assert panel_f.title == "Fly Mounted"
    assert panel_f.badge == "IR Illumination"
    assert panel_f.source == fm.PANEL_F_ASSET.name
    # The archived frame is already exposed correctly; re-stretching it would
    # change the published panel.
    assert panel_f.enhance is False


def test_panel_titles_match_the_published_figure():
    assert [p.title for p in fm.PANELS] == [
        "Cold Anesthesia On Abdomen",
        "UV Glue Application to Thorax",
        "Tethered & Inverted",
        "Tarsal Immobilization",
        "Head Fixation",
        "Fly Mounted",
    ]


def test_every_annotation_sits_inside_its_panel():
    for panel in fm.PANELS:
        for ann in panel.annotations:
            assert all(0.0 <= v <= 1.0 for v in ann.xy), (panel.letter, ann.text)
            assert all(0.0 <= v <= 1.0 for v in ann.xytext), (panel.letter, ann.text)


def test_annotations_use_the_two_accent_colours_only():
    used = {ann.color for panel in fm.PANELS for ann in panel.annotations}
    assert used <= {fm.AMBER, fm.CYAN}


def test_every_panel_carries_at_least_one_callout_except_f():
    for panel in fm.PANELS[:5]:
        assert panel.annotations, panel.letter


def test_crop_regions_are_well_formed():
    for panel in fm.PANELS:
        x0, y0, x1, y1 = panel.crop
        assert 0.0 <= x0 < x1 <= 1.0, panel.letter
        assert 0.0 <= y0 < y1 <= 1.0, panel.letter


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
def test_panel_grid_is_three_by_two_without_overlap():
    boxes = [fm.panel_rect(i) for i in range(6)]
    assert len(boxes) == 6
    row0_tops = {round(b[1] + b[3], 4) for b in boxes[:3]}
    row1_tops = {round(b[1] + b[3], 4) for b in boxes[3:]}
    assert len(row0_tops) == 1 and len(row1_tops) == 1
    assert row0_tops.pop() > row1_tops.pop()  # row 0 sits above row 1


def test_panels_do_not_run_off_the_canvas():
    for i in range(6):
        x, y, w, h = fm.panel_rect(i)
        assert x >= 0 and y >= 0
        assert x + w <= 1.0 + 1e-9
        assert y + h <= 1.0 + 1e-9


def test_panel_images_are_square_on_the_page():
    fig_w, fig_h = fm.FIG_SIZE_IN
    for i in range(6):
        _, _, w, h = fm.panel_rect(i)
        assert pytest.approx(w * fig_w, rel=1e-3) == h * fig_h


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
@pytest.fixture
def photo_dir(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(7)
    sizes = {
        "1.png": (1267, 1147),
        "2.png": (1996, 1637),
        "3.png": (827, 1213),
        "4.png": (1465, 1297),
        "5.png": (2045, 1513),
    }
    for name, (w, h) in sizes.items():
        # keep it small on disk but the right aspect ratio
        small = (max(w // 8, 8), max(h // 8, 8))
        arr = rng.integers(90, 180, size=(small[1], small[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / name)
    arr = rng.integers(0, 60, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / fm.PANEL_F_ASSET.name)
    return tmp_path


def test_build_figure_renders_all_six_panels(photo_dir):
    fig = fm.build_figure(photo_dir, panel_f_path=photo_dir / fm.PANEL_F_ASSET.name)
    try:
        titles = {
            t.get_text() for ax in fig.axes for t in ax.texts
        }
        for panel in fm.PANELS:
            assert panel.title in titles
            assert panel.letter in titles
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_build_figure_writes_png_pdf_and_svg(photo_dir, tmp_path):
    out = tmp_path / "out" / "fig_fly_mounting_workflow"
    written = fm.render(
        photo_dir,
        out,
        panel_f_path=photo_dir / fm.PANEL_F_ASSET.name,
        dpi=72,
    )
    assert {p.suffix for p in written} == {".png", ".pdf", ".svg"}
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_svg_keeps_labels_as_editable_text(photo_dir, tmp_path):
    """The published figure was an *editable* SVG -- labels must not be paths."""
    out = tmp_path / "editable"
    written = fm.render(
        photo_dir,
        out,
        panel_f_path=photo_dir / fm.PANEL_F_ASSET.name,
        dpi=72,
        formats=("svg",),
    )
    svg = written[0].read_text(encoding="utf-8")
    for label in ("Cold Anesthesia On Abdomen", "Tarsal Immobilization", "IR Illumination"):
        assert f">{label}<" in svg, label


def test_build_figure_reports_a_missing_photo_by_name(photo_dir):
    (photo_dir / "3.png").unlink()
    with pytest.raises(FileNotFoundError, match="3.png"):
        fm.build_figure(photo_dir, panel_f_path=photo_dir / fm.PANEL_F_ASSET.name)


def test_prepared_panel_image_is_square_and_unit_ranged(photo_dir):
    arr = fm.prepare_panel_image(photo_dir / "3.png", fm.PANELS[2])
    assert arr.shape[0] == arr.shape[1]
    assert arr.shape[2] == 3
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_prepared_panel_f_image_is_not_contrast_stretched(photo_dir):
    from PIL import Image

    path = photo_dir / fm.PANEL_F_ASSET.name
    raw = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0
    out = fm.prepare_panel_image(path, fm.PANELS[5])
    # panel F crop is the full frame, so the pixels must survive untouched
    assert np.allclose(out, raw)
