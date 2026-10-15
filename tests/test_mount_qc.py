import json
from pathlib import Path

import numpy as np
import pytest

from fbpipe.utils.mount_qc import (Detection, Thresholds, analyze, find_slots, in_gate,
                                   region_sharpness)

GOLDEN = Path(__file__).parent / "golden" / "mount_qc_pi2_20261014.json"


def eye(x, y, conf=0.95, w=44, h=56, r=0.0):
    return Detection("eye", x, y, w, h, r, conf)


def prob(x, y, conf=0.95, w=38, h=29, r=0.0):
    return Detection("proboscis", x, y, w, h, r, conf)


def good_frame(dx=0.0, dy=0.0):
    """Four well-placed flies, proboscis 90 px below each eye."""
    out = []
    for y in (150, 400, 650, 850):
        out += [eye(700 + dx, y + dy), prob(700 + dx, y + 90 + dy)]
    return out


def codes(report):
    return [(f.code, f.slot, f.part) for f in report.flags]


def test_golden_pi2_frame_flags_exactly_the_two_real_problems():
    g = json.loads(GOLDEN.read_text())
    dets = [Detection.from_dict(d) for d in g["detections"]]
    rep = analyze([dets], g["width"], g["height"], gate_px=280.0, up_divisor=4.0)
    assert [s.slot for s in rep.slots] == [1, 2, 3, 4]
    assert codes(rep) == [("edge_margin", 1, "eye"), ("bottom_room", 4, "proboscis")]
    assert "from top edge" in rep.flags[0].message
    assert "PER may leave frame" in rep.flags[1].message
    assert rep.flags[1].value < 40


def test_clean_mount_has_no_flags():
    rep = analyze([good_frame() for _ in range(10)], 1080, 1080, gate_px=160.0)
    assert rep.ok and rep.frames == 10
    assert all(s.eye_frac == 1.0 and s.proboscis_frac == 1.0 for s in rep.slots)
    assert rep.slots[0].eye_proboscis_px == pytest.approx(90.0)


def test_only_three_flies_visible():
    frames = [[d for d in good_frame() if not (d.y in (850, 940))] for _ in range(5)]
    rep = analyze(frames, 1080, 1080, gate_px=160.0)
    assert ("flies_visible", None, "eye") in codes(rep)
    assert "only 3 flies visible" in rep.flags[0].message


def test_fly_missing_in_some_frames_crosses_twenty_percent():
    frames = [good_frame() for _ in range(10)]
    for i in range(3):   # 30 % of frames lose fly 2's eye
        frames[i] = [d for d in frames[i] if not (d.cls == "eye" and d.y == 400)]
    rep = analyze(frames, 1080, 1080, gate_px=160.0)
    f = [x for x in rep.flags if x.code == "flies_visible"][0]
    assert f.slot == 2 and "missing in 30% of frames" in f.message
    frames[2] = good_frame()   # 20 % is not > 20 %
    assert "flies_visible" not in [x.code for x in analyze(frames, 1080, 1080, gate_px=160.0).flags]


def test_proboscis_missing_below_half_the_frames():
    frames = [good_frame() for _ in range(10)]
    for i in range(6):
        frames[i] = [d for d in frames[i] if not (d.cls == "proboscis" and d.y == 490)]
    rep = analyze(frames, 1080, 1080, gate_px=160.0)
    assert ("proboscis_missing", 2, "proboscis") in codes(rep)


def test_gate_is_anisotropic_upward():
    e = eye(700, 500)
    assert in_gate(e, prob(700, 500 + 150), 160.0, 4.0)        # 150 px down: in
    assert not in_gate(e, prob(700, 500 - 50), 160.0, 4.0)     # 50 px up > 160/4: out
    assert in_gate(e, prob(700, 500 - 35), 160.0, 4.0)
    assert not in_gate(e, prob(700, 500 + 170), 160.0, 4.0)    # beyond the gate


def test_each_proboscis_pairs_with_one_eye_only():
    frame = [eye(700, 400), eye(700, 470), prob(700, 500)]
    rep = analyze([frame], 1080, 1080, gate_px=160.0, th=Thresholds(expected_flies=2, slot_gap_px=30))
    assert sorted(s.proboscis_frac for s in rep.slots) == [0.0, 1.0]


def test_edge_margin_and_bottom_room_dedupe():
    frame = good_frame()
    frame[-1] = prob(700, 1040)   # fly 4 proboscis almost at the bottom
    rep = analyze([frame], 1080, 1080, gate_px=200.0)
    assert codes(rep) == [("bottom_room", 4, "proboscis")]


def test_bottom_room_threshold_boundary():
    frame = good_frame()
    frame[-1] = prob(700, 1080 - 120 - 14.5)   # bottom extent exactly 120 px from the edge
    assert analyze([frame], 1080, 1080, gate_px=250.0).ok
    frame[-1] = prob(700, 1080 - 119 - 14.5)
    assert ("bottom_room", 4, "proboscis") in codes(analyze([frame], 1080, 1080, gate_px=250.0))


def test_margins_use_actual_frame_size():
    frames = [good_frame(dx=1300) for _ in range(3)]          # eyes at x=2000 in a 2040-wide frame
    rep = analyze(frames, 2040, 1080, gate_px=160.0)
    assert ("edge_margin", 1, "eye") in codes(rep)
    assert analyze([good_frame(dx=1300)], 2600, 1080, gate_px=160.0).ok


def test_wobble_flags_a_moving_head():
    frames = [good_frame(dx=(30 if i % 2 else -30)) for i in range(10)]
    rep = analyze(frames, 1080, 1080, gate_px=160.0)
    assert {("wobble", k, "eye") for k in (1, 2, 3, 4)} <= set(codes(rep))


def test_low_confidence_is_image_quality():
    frames = [[Detection(d.cls, d.x, d.y, d.w, d.h, d.r, 0.4) for d in good_frame()]]
    rep = analyze(frames, 1080, 1080, gate_px=160.0)
    assert ("image_quality", None, None) in codes(rep)


def test_sharpness_against_reference():
    sharp = {1: 50.0, 2: 52.0, 3: 49.0, 4: 51.0}
    assert analyze([good_frame()], 1080, 1080, gate_px=160.0, sharpness=sharp, sharpness_reference=80.0).ok
    blurry = {k: 30.0 for k in sharp}
    rep = analyze([good_frame()], 1080, 1080, gate_px=160.0, sharpness=blurry, sharpness_reference=80.0)
    assert ("image_quality", None, None) in codes(rep)
    assert analyze([good_frame()], 1080, 1080, gate_px=160.0, sharpness=blurry).ok   # no reference yet


def test_no_frames_is_camera_flag():
    rep = analyze([], 1080, 1080, gate_px=160.0)
    assert codes(rep) == [("camera", None, None)] and rep.slots == []


def test_find_slots_ignores_rare_false_eyes():
    frames = [good_frame() for _ in range(40)]
    frames[0].append(eye(300, 50))      # one stray detection (1/40 < 5 %)
    assert len(find_slots(frames, Thresholds())) == 4


def test_region_sharpness_prefers_texture():
    rng = np.random.default_rng(0)
    flat = np.full((200, 200), 40, np.uint8)
    noisy = rng.integers(0, 255, (200, 200), dtype=np.uint8)
    assert region_sharpness(noisy, 100, 100) > region_sharpness(flat, 100, 100) == 0.0
    assert region_sharpness(noisy, 5, 5) > 0   # window clipped at the border, no crash


def test_report_to_dict_is_json_serialisable():
    json.dumps(analyze([good_frame()], 1080, 1080, gate_px=160.0).to_dict())
