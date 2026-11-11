"""Image helpers behind the labellum rest-vs-extension pubfig."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/make_pubfig_labellum_rest_vs_extension.py"
spec = importlib.util.spec_from_file_location("labellum_pubfig", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_crop_stretch_shape_is_height_by_width():
    luma = np.zeros((40, 50), dtype=np.uint8)
    assert mod.crop_stretch(luma, (7, 3, 20, 12)).shape == (12, 20)


def test_crop_stretch_maps_range_to_full_scale():
    luma = np.full((10, 10), 40, dtype=np.uint8)
    luma[:, 5:] = 140
    out = mod.crop_stretch(luma, (0, 0, 10, 10), lo_pct=0, hi_pct=100)
    assert out[:, :5].max() == 0
    assert out[:, 5:].min() == 255


def test_crop_coords_shift_by_crop_origin():
    pts = mod.to_crop_coords([[110, 205], [120, 215]], (100, 200, 50, 40))
    np.testing.assert_allclose(pts, [[10, 5], [20, 15]])


def test_every_box_lies_inside_the_crop():
    _x0, _y0, w, h = mod.CROP
    for _title, eye, labellum in mod.PANELS.values():
        for box in (eye, labellum):
            p = mod.to_crop_coords(box, mod.CROP)
            assert (p[:, 0] >= 0).all() and (p[:, 0] <= w).all()
            assert (p[:, 1] >= 0).all() and (p[:, 1] <= h).all()


def test_shrink_polygon_keeps_centroid_and_scales_distances():
    box = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    out = mod.shrink_polygon(box, 0.5)
    np.testing.assert_allclose(out.mean(axis=0), box.mean(axis=0))
    np.testing.assert_allclose(np.linalg.norm(out - out.mean(0), axis=1),
                               0.5 * np.linalg.norm(box - box.mean(0), axis=1))


def test_shrink_polygon_keeps_orientation():
    box = np.array([[0.0, 0.0], [4.0, 1.0], [3.0, 3.0], [-1.0, 2.0]])
    a, b = box[1] - box[0], mod.shrink_polygon(box, 0.8)[1] - mod.shrink_polygon(box, 0.8)[0]
    np.testing.assert_allclose(b / np.linalg.norm(b), a / np.linalg.norm(a))


def test_unsharp_preserves_shape_and_flat_images():
    img = np.full((20, 30), 77.0)
    out = mod.unsharp(img)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, img)


def test_unsharp_boosts_an_edge():
    img = np.zeros((20, 20))
    img[:, 10:] = 100.0
    assert mod.unsharp(img)[10, 10] > 100.0  # overshoot on the bright side
