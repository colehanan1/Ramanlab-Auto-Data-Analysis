"""Per-frame (live) eye tracking in yolo_infer -- QC only.

Every frame records where each slot's eye is actually detected
(``eye_{i}_live_x/_y``) and how far that sits from the frozen anchor
(``eye_{i}_drift_px``); the per-fly tables carry them as ``eye_live_x/_y`` and
``eye_drift_px``. The frozen anchor (``eye_{i}_x/_y`` -> ``x_class0/y_class0``)
stays what every distance, angle and gate reads.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ultralytics")

from fbpipe.config import EyeQCSettings, Settings
from fbpipe.steps import yolo_infer
from fbpipe.utils import columns as column_aliases
from fbpipe.utils.multi_fly import EyeAnchorManager, StablePairing
from fbpipe.utils.tables import read_table

# Two flies stacked 350 px apart (Flybehavior2 framing); slot 0 is the top one.
ANCHORS = ((500.0, 100.0), (500.0, 450.0))
LIVE_SUFFIXES = ("live_x", "live_y", "drift_px")
PER_FLY_LIVE_COLS = ("eye_live_x", "eye_live_y", "eye_drift_px")
MARKER_BGR = (255, 255, 0)  # cyan: no red, so check_light_stimulus never sees it


def _box(cx, cy, half=5.0):
    return [cx - half, cy - half, cx + half, cy + half]


def _settings(**overrides) -> Settings:
    kwargs = dict(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=2,
        use_optical_flow=False,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


class _NoProboscis:
    def step(self, boxes, scores):
        return []


def _run_frame(monkeypatch, eyes_now, settings, *, scores=None, max_eyes=2):
    """One _process_frame call with anchors frozen on ANCHORS and this frame's
    eye detections centred on ``eyes_now``. Returns (annotated frame, row, eye_mgr)."""
    eye_mgr = EyeAnchorManager(max_eyes=max_eyes)
    eye_mgr.try_update_from_dets(
        np.array([_box(*c) for c in ANCHORS], dtype=np.float32),
        np.array([0.9, 0.8], dtype=np.float32),
    )
    boxes = np.array([_box(*c) for c in eyes_now], dtype=np.float32).reshape(-1, 4)
    if scores is None:
        scores = [0.9] * len(boxes)
    scores = np.asarray(scores, dtype=np.float32)
    monkeypatch.setattr(
        yolo_infer,
        "collect_detections",
        lambda result, classes: {
            0: {"boxes": boxes, "scores": scores},
            1: {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32)},
        },
    )
    frame, row, _ = yolo_infer._process_frame(
        frame=np.zeros((600, 1000, 3), dtype=np.uint8),
        frame_number=7,
        current_timestamp=0.175,
        single_trackers={},
        prev_gray=None,
        anchor=(0.0, 300.0),
        settings=settings,
        result=object(),
        eye_mgr=eye_mgr,
        cls8_tracker=_NoProboscis(),
        pairer=StablePairing(max_pairs=max_eyes),
        active_max_flies=max_eyes,
    )
    return frame, row, eye_mgr


def _live_keys(row):
    return [k for k in row if any(k.endswith(f"_{s}") for s in LIVE_SUFFIXES)]


def _same_row(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb):
            continue
        if va != vb:
            return False
    return True


# ---------------------------------------------------------------------------
# _process_frame rows
# ---------------------------------------------------------------------------


def test_shifted_eye_detection_records_live_position_and_drift(monkeypatch):
    _, row, _ = _run_frame(monkeypatch, [(503.0, 104.0), (500.0, 450.0)], _settings())

    assert (row["eye_0_live_x"], row["eye_0_live_y"]) == (503.0, 104.0)
    assert row["eye_0_drift_px"] == pytest.approx(5.0)
    assert (row["eye_1_live_x"], row["eye_1_live_y"], row["eye_1_drift_px"]) == (500.0, 450.0, 0.0)
    # The frozen anchor every distance reads does not follow the eye.
    assert (row["eye_0_x"], row["eye_0_y"]) == (500.0, 100.0)
    assert row["dist_eye_0_anchor"] == pytest.approx(np.hypot(500.0, 200.0))


def test_missing_eye_leaves_live_and_drift_nan_while_the_anchor_stays_frozen(monkeypatch):
    _, row, _ = _run_frame(monkeypatch, [(500.0, 100.0)], _settings())

    for suffix in LIVE_SUFFIXES:
        assert np.isnan(row[f"eye_1_{suffix}"])
    assert (row["eye_1_x"], row["eye_1_y"]) == (500.0, 450.0)
    assert row["eye_1_track_id"] == 1002
    assert row["eye_0_drift_px"] == 0.0


def test_every_slot_gets_the_live_keys_including_an_inactive_one(monkeypatch):
    _, row, _ = _run_frame(monkeypatch, list(ANCHORS), _settings(max_flies=3))

    assert np.isnan(row["eye_2_x"])  # no anchor in slot 2
    for suffix in LIVE_SUFFIXES:
        assert np.isnan(row[f"eye_2_{suffix}"])
    assert sorted(_live_keys(row)) == sorted(
        f"eye_{i}_{s}" for i in range(3) for s in LIVE_SUFFIXES
    )


def test_live_eye_is_tracked_before_the_anchors_confirm(monkeypatch):
    # Three slots expected, only two eyes seen so far: the anchors are partial
    # and unconfirmed, but their live eyes are still recorded, never blanked.
    _, row, eye_mgr = _run_frame(
        monkeypatch, [(502.0, 100.0), (500.0, 449.0)], _settings(max_flies=3), max_eyes=3
    )

    assert not eye_mgr.confirmed
    assert row["eye_0_drift_px"] == pytest.approx(2.0)
    assert row["eye_1_drift_px"] == pytest.approx(1.0)
    assert np.isnan(row["eye_2_drift_px"])


def test_live_eye_reads_the_raw_detections_that_top_k_evicts(monkeypatch):
    # A confident false positive outranks fly 1's real eye, so the k=2 top-k
    # keeps the false positive and drops the eye. The live match still finds it.
    _, row, _ = _run_frame(
        monkeypatch,
        [(900.0, 550.0), (500.0, 100.0), (501.0, 452.0)],
        _settings(),
        scores=[0.99, 0.95, 0.5],
    )

    assert (row["eye_1_live_x"], row["eye_1_live_y"]) == (501.0, 452.0)
    assert row["eye_1_drift_px"] == pytest.approx(np.hypot(1.0, 2.0))


def test_disabled_eye_qc_adds_no_live_keys_and_changes_nothing_else(monkeypatch):
    eyes_now = [(503.0, 104.0)]
    _, enabled, _ = _run_frame(monkeypatch, eyes_now, _settings())
    _, disabled, _ = _run_frame(
        monkeypatch, eyes_now, _settings(eye_qc=EyeQCSettings(enabled=False))
    )

    assert _live_keys(disabled) == []
    assert _live_keys(enabled)
    assert _same_row({k: v for k, v in enabled.items() if k not in _live_keys(enabled)}, disabled)


@pytest.mark.parametrize("radius", [0.0, -50.0, float("nan"), float("inf")])
def test_unusable_match_radius_falls_back_to_the_default(monkeypatch, radius):
    # load_settings passes odd numbers through; a radius that matched nothing
    # would leave every eye NaN for good (YOLO never re-runs a processed trial).
    settings = _settings(eye_qc=EyeQCSettings(match_radius_px=radius))
    _, row, _ = _run_frame(monkeypatch, [(500.0, 130.0), (500.0, 550.0)], settings)

    assert row["eye_0_drift_px"] == pytest.approx(30.0)  # inside the 50 px default
    # 100 px off: outside the default, though inside the 157 px spacing clamp
    # an infinite radius would leave.
    assert np.isnan(row["eye_1_drift_px"])


def test_configured_match_radius_is_used(monkeypatch):
    eyes_now = [(500.0, 130.0), (500.0, 450.0)]  # fly 0's eye 30 px off its anchor
    _, tight, _ = _run_frame(monkeypatch, eyes_now, _settings(eye_qc=EyeQCSettings(match_radius_px=20.0)))
    _, loose, _ = _run_frame(monkeypatch, eyes_now, _settings(eye_qc=EyeQCSettings(match_radius_px=40.0)))

    assert np.isnan(tight["eye_0_drift_px"])
    assert loose["eye_0_drift_px"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Annotated-video marker
# ---------------------------------------------------------------------------


def _marker_pixels(monkeypatch, eyes_now, **eye_qc):
    """(changed mask, annotated frame): pixels the live-eye marker changed."""
    on, _, _ = _run_frame(monkeypatch, eyes_now, _settings(eye_qc=EyeQCSettings(**eye_qc)))
    off, _, _ = _run_frame(
        monkeypatch, eyes_now, _settings(eye_qc=EyeQCSettings(**{**eye_qc, "draw_live_eye": False}))
    )
    return np.any(on != off, axis=2), on


def test_live_eye_marker_is_a_hollow_cyan_ring(monkeypatch):
    changed, frame = _marker_pixels(monkeypatch, [(501.0, 100.0)])

    ys, xs = np.nonzero(changed)
    assert len(ys) > 0
    assert (frame[ys, xs] == MARKER_BGR).all()  # never red/magenta, no other colour
    radii = np.hypot(xs - 501, ys - 100)
    assert radii.min() >= 5.0 and radii.max() <= 7.0  # a thin ring, no text, no fill
    assert not changed[100, 501]


def test_live_eye_marker_looks_the_same_at_any_drift(monkeypatch):
    # Nothing on the scored video may appear or change with drift state: a 12 px
    # drift (over the 5 px QC threshold) draws the very same ring as a 1 px one.
    def ring_offsets(cx):
        changed, _ = _marker_pixels(monkeypatch, [(float(cx), 100.0)])
        ys, xs = np.nonzero(changed)
        return sorted(zip((xs - cx).tolist(), (ys - 100).tolist()))

    small = ring_offsets(501)
    assert small
    assert small == ring_offsets(512)


def test_no_marker_when_eye_qc_or_draw_live_eye_is_off(monkeypatch):
    eyes_now = [(501.0, 100.0)]
    drawn, _, _ = _run_frame(monkeypatch, eyes_now, _settings())
    no_draw, _, _ = _run_frame(
        monkeypatch, eyes_now, _settings(eye_qc=EyeQCSettings(draw_live_eye=False))
    )
    disabled, _, _ = _run_frame(
        monkeypatch, eyes_now, _settings(eye_qc=EyeQCSettings(enabled=False, draw_live_eye=True))
    )

    assert not np.array_equal(drawn, no_draw)
    assert np.array_equal(no_draw, disabled)


# ---------------------------------------------------------------------------
# Per-fly export + column-resolver regressions
# ---------------------------------------------------------------------------


def _two_frame_rows(monkeypatch, settings):
    """Frame 0: both eyes on their anchors. Frame 1: fly 0's eye 5 px off, fly 1's missing."""
    _, still, _ = _run_frame(monkeypatch, list(ANCHORS), settings)
    _, moved, _ = _run_frame(monkeypatch, [(503.0, 104.0)], settings)
    still["frame"], still["timestamp"] = 0, 0.0
    moved["frame"], moved["timestamp"] = 1, 0.025
    return pd.DataFrame([still, moved])


def _export(monkeypatch, out_dir, settings=None):
    settings = settings or _settings()
    df = _two_frame_rows(monkeypatch, settings)
    paths = yolo_infer._export_per_fly_csvs(df, out_dir, "trial", settings, 2)
    assert [p.name for p in paths] == ["trial_fly1_distances.parquet", "trial_fly2_distances.parquet"]
    return [read_table(p) for p in paths]


def test_per_fly_export_carries_the_live_eye_columns_as_floats(monkeypatch, tmp_path):
    fly1, fly2 = _export(monkeypatch, tmp_path)

    for df in (fly1, fly2):
        assert all(df[c].dtype == np.float64 for c in PER_FLY_LIVE_COLS)
    assert fly1["eye_live_x"].tolist() == [500.0, 503.0]
    assert fly1["eye_live_y"].tolist() == [100.0, 104.0]
    assert fly1["eye_drift_px"].tolist() == pytest.approx([0.0, 5.0])
    assert fly1["x_class0"].tolist() == [500.0, 500.0]  # still the frozen anchor
    assert fly2["eye_drift_px"].iloc[0] == 0.0
    assert np.isnan(fly2["eye_live_x"].iloc[1]) and np.isnan(fly2["eye_drift_px"].iloc[1])


def test_per_fly_export_without_live_columns_writes_nan_floats(monkeypatch, tmp_path):
    fly1, _ = _export(monkeypatch, tmp_path, _settings(eye_qc=EyeQCSettings(enabled=False)))

    for c in PER_FLY_LIVE_COLS:
        assert fly1[c].dtype == np.float64 and fly1[c].isna().all()


def test_live_eye_names_never_resolve_as_eye_proboscis_or_distance_columns(monkeypatch, tmp_path):
    fly1, _ = _export(monkeypatch, tmp_path)
    assert column_aliases.find_eye_xy_columns(fly1) == ("x_class0", "y_class0")

    # With only the new columns present, no alias resolver may take any of them
    # for an eye, proboscis or distance column (per-fly and merged-table names).
    merged_names = [f"eye_{i}_{s}" for i in range(4) for s in LIVE_SUFFIXES]
    for names in (list(PER_FLY_LIVE_COLS), merged_names):
        live_only = pd.DataFrame(columns=["frame", "timestamp", *names])
        assert column_aliases.find_eye_xy_columns(live_only) == (None, None)
        assert column_aliases.find_proboscis_xy_columns(live_only) == (None, None)
        assert column_aliases.find_proboscis_distance_column(live_only) is None
        assert column_aliases.find_proboscis_distance_percentage_column(live_only) is None
        assert column_aliases.find_proboscis_min_distance_column(live_only) is None
        assert column_aliases.find_proboscis_max_distance_column(live_only) is None
        for name in names:
            assert not any(word in name for word in ("prob", "anchor", "distance"))


def test_proboscis_gate_still_measures_from_the_frozen_eye(monkeypatch, tmp_path):
    from fbpipe.utils.distance_sanity import sanitize_eye_prob_geometry_dataframe

    fly1, _ = _export(monkeypatch, tmp_path)
    # 36 px straight above the frozen eye (500, 100): inside a 150 px gate's
    # 37.5 px upward reach. From frame 1's live eye (503, 104) it is 40 px up.
    fly1["x_class1"], fly1["y_class1"] = 500.0, 64.0

    assert sanitize_eye_prob_geometry_dataframe(fly1, 150.0, 4.0)[1] == 0
    from_live_eye = fly1.assign(x_class0=fly1["eye_live_x"], y_class0=fly1["eye_live_y"])
    assert sanitize_eye_prob_geometry_dataframe(from_live_eye, 150.0, 4.0)[1] == 1


def test_angle_helpers_never_measure_from_the_live_eye(monkeypatch, tmp_path):
    from fbpipe.steps import compose_videos_rms
    from scripts.analysis import envelope_combined

    fly1, _ = _export(monkeypatch, tmp_path)
    no_frozen_eye = fly1.drop(columns=["x_class0", "y_class0"])
    for angle in (compose_videos_rms.compute_angle_deg_at_point2, envelope_combined._compute_angle_deg):
        angle(fly1, (0.0, 300.0))  # resolves x_class0/y_class0
        with pytest.raises(ValueError):
            angle(no_frozen_eye, (0.0, 300.0))  # never falls back to eye_live_x/_y


def test_rms_copy_filter_still_copies_only_the_frozen_eye_columns(monkeypatch, tmp_path):
    from fbpipe.steps import rms_copy_filter

    batch = tmp_path / "october_07_batch_1"
    _export(monkeypatch, batch / "trial")
    rms_copy_filter._process_fly_dir(batch, SimpleNamespace())

    out = read_table(batch / "RMS_calculations" / "updated_trial_fly1_distances.parquet")
    assert list(out.columns[:4]) == ["frame", "timestamp", "x_class0", "y_class0"]
    assert not set(PER_FLY_LIVE_COLS) & set(out.columns)
    assert out["x_class0"].tolist() == [500.0, 500.0]
