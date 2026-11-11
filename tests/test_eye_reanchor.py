"""The drift-following eye anchor (fbpipe.utils.eye_reanchor).

The measuring point stays where the eye was anchored and moves only when
  * the eye drifts and stays drifted (the eye-QC rule: median over sustain_s
    above max_drift_px), or
  * every fly in the trial jumps the same way in the same frame (the camera
    or the whole image moved -- the proboscis moved with it).
A one-fly jump of a frame or two (a false eye match) never moves it.
"""
import numpy as np
import pandas as pd
import pytest

from fbpipe.utils import eye_reanchor as er

W = 20       # 0.5 s at 40 fps
THR = 5.0    # eye_qc.max_drift_px


def _still(n, x=500.0, y=300.0, jitter=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return x + rng.normal(0, jitter, n), y + rng.normal(0, jitter, n)


# ---------------------------------------------------------------- step anchor


def test_jitter_never_moves_the_anchor():
    lx, ly = _still(800)
    ox, oy, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert not events
    assert np.all(ox == 0) and np.all(oy == 0)


def test_one_fly_spike_of_two_frames_is_ignored():
    lx, ly = _still(800)
    lx[400:402] += 40.0                     # a false eye match, 2 frames
    ox, oy, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert not events
    assert np.all(ox == 0)


def test_sustained_shift_moves_anchor_from_its_onset():
    lx, ly = _still(800)
    lx[300:] += 9.0                         # the fly shifts 9 px and stays
    ox, oy, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert len(events) == 1
    assert events[0]["frame"] == 300
    assert np.all(ox[:300] == 0)
    assert ox[300:] == pytest.approx(9.0, abs=0.5)
    assert np.all(ox[300:] == ox[300])      # static after the move: one step, no tracking


def test_shift_below_threshold_is_not_followed():
    lx, ly = _still(800)
    lx[300:] += 3.0
    _, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert not events


def test_shift_and_return_moves_twice():
    lx, ly = _still(900)
    lx[200:500] += 10.0
    ox, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert [e["frame"] for e in events] == [200, 500]
    assert ox[350] == pytest.approx(10.0, abs=0.5)
    assert ox[700] == pytest.approx(0.0, abs=0.5)


def test_unmatched_frames_count_as_not_over():
    lx, ly = _still(800)
    lx[300:] += 9.0
    lx[300:] = np.where(np.arange(500) % 3 == 0, lx[300:], np.nan)   # eye seen 1 frame in 3
    ly[np.isnan(lx)] = np.nan
    _, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert not events                       # under half the window detected: not enough evidence


def test_initial_offset_is_the_starting_point():
    lx, ly = _still(400, x=506.0)
    ox, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR,
                                    initial=(6.0, 0.0))
    assert not events
    assert np.all(ox == 6.0)


# ---------------------------------------------------------------- shake & group


def test_symmetric_shake_does_not_move_the_anchor():
    # rig vibration: the eye swings +-15 px every frame for a second, centred
    # on where it was (IAA october_09_batch_2_rig_3, training_6, 18 s)
    lx, ly = _still(800)
    lx[300:340] += np.where(np.arange(40) % 2 == 0, 15.0, -15.0)
    ox, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert not events
    assert np.all(ox == 0)


def _flies(n=600, k=4, seed=0):
    xs, ys = [], []
    for i in range(k):
        x, y = _still(n, x=500.0, y=200.0 + 200 * i, seed=seed + i)
        xs.append(x)
        ys.append(y)
    return np.stack(xs, 1), np.stack(ys, 1)


def _group(lx, ly, **kw):
    return er.group_moves(lx - 500.0, ly - (200.0 + 200 * np.arange(lx.shape[1])), window=W,
                          min_px=2.0, agree_px=1.5, **kw)


def test_held_group_shift_is_a_group_move():
    lx, ly = _flies()
    lx[300:] += 3.0                         # every fly shifts 3 px and stays
    g = _group(lx, ly)
    assert g[340:].all() and not g[:300].any()


def test_brief_group_jump_is_not_a_group_move():
    lx, ly = _flies()
    lx[300:302] += 4.0                      # all flies, 2 frames (rig shake)
    assert not _group(lx, ly).any()


def test_one_fly_shifting_alone_is_not_a_group_move():
    lx, ly = _flies()
    lx[300:, 0] += 3.0
    assert not _group(lx, ly).any()


def test_flies_shifting_different_ways_is_not_a_group_move():
    lx, ly = _flies()
    lx[300:, :2] += 3.0
    lx[300:, 2:] -= 3.0
    assert not _group(lx, ly).any()


def test_single_fly_trial_has_no_group_moves():
    lx, ly = _flies(k=1)
    lx[300:] += 6.0
    assert not _group(lx, ly).any()


def test_group_move_needs_most_flies_seen():
    lx, ly = _flies()
    lx[300:, :2] += 3.0
    lx[300:, 2:] = np.nan                   # only 2 of 4 flies seen
    ly[300:, 2:] = np.nan
    assert not _group(lx, ly).any()


def test_lower_threshold_applies_only_where_given():
    lx, ly = _still(800)
    lx[300:] += 3.0
    thr = np.full(800, THR)
    _, _, ev = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=thr)
    assert not ev
    thr[300:] = 2.0
    ox, _, ev = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=thr)
    assert [e["frame"] for e in ev] == [300]
    assert ox[400] == pytest.approx(3.0, abs=0.5)


# ---------------------------------------------------------------- trial rewrite


def _table(n=800, eye=(500.0, 300.0), live=None, prob=(560.0, 300.0)):
    lx, ly = live if live is not None else _still(n, *eye)
    return pd.DataFrame({
        "frame": np.arange(n),
        "x_class0": eye[0], "y_class0": eye[1],
        "x_class1": prob[0], "y_class1": prob[1],
        "x_anchor": 0.0, "y_anchor": 540.0,
        "eye_live_x": lx, "eye_live_y": ly,
        "distance_0_1": np.hypot(prob[0] - eye[0], prob[1] - eye[1]),
    })


def test_trial_without_events_is_unchanged():
    tables = {1: _table(), 2: _table(eye=(500.0, 600.0), prob=(560.0, 600.0))}
    out, summary = er.reanchor_trial(tables, window=W, max_drift_px=THR)
    for k in tables:
        np.testing.assert_array_equal(out[k]["distance_0_1"], tables[k]["distance_0_1"])
        np.testing.assert_array_equal(out[k]["x_class0"], tables[k]["x_class0"])
    assert all(s["n_steps"] == 0 for s in summary)


def test_trial_drift_remeasures_distance_from_the_moved_anchor():
    lx, ly = _still(800)
    lx[300:] += 10.0                        # eye 10 px closer to the proboscis at x=560
    t = _table(live=(lx, ly))
    out, summary = er.reanchor_trial({1: t}, window=W, max_drift_px=THR)
    d = out[1]["distance_0_1"].to_numpy()
    assert d[100] == pytest.approx(60.0)
    assert d[500] == pytest.approx(50.0, abs=0.5)
    assert out[1]["x_class0_anchor"].iloc[500] == 500.0      # the old anchor is kept
    assert summary[0]["n_steps"] == 1


def test_trial_rewrite_is_idempotent_and_does_not_mutate_input():
    lx, ly = _still(800)
    lx[300:] += 10.0
    t = _table(live=(lx, ly))
    before = t.copy()
    once, _ = er.reanchor_trial({1: t}, window=W, max_drift_px=THR)
    pd.testing.assert_frame_equal(t, before)
    twice, _ = er.reanchor_trial({1: once[1]}, window=W, max_drift_px=THR)
    np.testing.assert_allclose(twice[1]["distance_0_1"], once[1]["distance_0_1"])


def _trial4(lx, ly):
    return {i + 1: _table(eye=(500.0, 200.0 + 200 * i), prob=(560.0, 200.0 + 200 * i),
                          live=(lx[:, i], ly[:, i])) for i in range(lx.shape[1])}


def test_trial_held_group_shift_moves_every_fly():
    lx, ly = _flies(n=800)
    lx[300:] += 3.0
    out, summary = er.reanchor_trial(_trial4(lx, ly), window=W, max_drift_px=THR, group_min_px=2.0)
    for k in out:
        assert out[k]["distance_0_1"].iloc[500] == pytest.approx(57.0, abs=0.5)
        assert out[k]["distance_0_1"].iloc[100] == 60.0
    assert all(s["n_steps"] == 1 for s in summary)


def test_trial_brief_group_jump_changes_nothing():
    lx, ly = _flies(n=800)
    lx[300:302] += 4.0
    t = _trial4(lx, ly)
    out, _ = er.reanchor_trial(t, window=W, max_drift_px=THR, group_min_px=2.0)
    for k in t:
        np.testing.assert_array_equal(out[k]["distance_0_1"], t[k]["distance_0_1"])


def test_initial_offset_applies_to_the_whole_trial():
    t = _table(live=_still(800, x=506.0))
    out, _ = er.reanchor_trial({1: t}, window=W, max_drift_px=THR, initial_offsets={1: (6.0, 0.0)})
    assert out[1]["x_class0"].iloc[0] == pytest.approx(506.0)
    assert out[1]["distance_0_1"].iloc[0] == pytest.approx(54.0)


# ---------------------------------------------------------------- next video


def test_eye_missing_from_warmup_borrows_the_next_videos_anchor():
    # trial order a, b, c; b's eye not seen in its first frames (anchor borrowed
    # from the template), so it takes c's anchor: the eye does not move between videos.
    anchors = {"a": (500.0, 300.0), "b": (497.0, 300.0), "c": (503.0, 300.0)}
    found = {"a": True, "b": False, "c": True}
    off = er.warmup_offsets(["a", "b", "c"], anchors, found)
    assert off == {"a": (0.0, 0.0), "b": (6.0, 0.0), "c": (0.0, 0.0)}


def test_last_video_missing_borrows_the_previous_one():
    anchors = {"a": (500.0, 300.0), "b": (497.0, 300.0)}
    off = er.warmup_offsets(["a", "b"], anchors, {"a": True, "b": False})
    assert off["b"] == (3.0, 0.0)


def test_never_found_keeps_its_anchor():
    off = er.warmup_offsets(["a"], {"a": (500.0, 300.0)}, {"a": False})
    assert off == {"a": (0.0, 0.0)}


def test_warmup_found_looks_only_at_the_first_frames():
    lx = np.full(100, np.nan)
    lx[10] = 1.0
    assert not er.warmup_found(lx, lx, scan_frames=5)
    lx[3] = 1.0
    assert er.warmup_found(lx, lx, scan_frames=5)


def test_slow_drift_is_followed_in_steps():
    lx, ly = _still(1200)
    lx[200:600] += np.linspace(0, 20, 400)  # the fly slides 20 px over 10 s and stays
    lx[600:] += 20.0
    ox, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert len(events) >= 3
    assert ox[-1] == pytest.approx(20.0, abs=THR)
    assert np.all(np.abs(ox[200:] - (lx[200:] - 500.0)) < 2 * THR)  # never far behind the eye


def test_still_false_spot_then_eye_lost_is_not_followed():
    # rig_3 light onset: a junk eye sits still 13 px off for 15 frames, then
    # the light washes the eye out for seconds -- nothing says the eye stayed there
    lx, ly = _still(900)
    lx[300:315] += 13.0
    lx[315:700] = np.nan
    ly[315:700] = np.nan
    ox, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert not events
    assert np.all(ox == 0)


def test_move_takes_the_spot_where_the_eye_stayed():
    lx, ly = _still(800)
    lx[300:] += 9.0
    lx[300:304] += 6.0                      # overshoot, then settles 9 px over
    ox, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR)
    assert len(events) == 1 and events[0]["frame"] == 300
    assert ox[600] == pytest.approx(9.0, abs=0.5)


def test_after_a_move_the_anchor_follows_where_the_eye_settles_next():
    # a loose fly: slips 9 px, then settles back 4 px -- under the 5 px bar for
    # a first move, but the anchor must not stay stuck 4 px off the eye
    lx, ly = _still(1000)
    lx[300:] += 9.0
    lx[600:] -= 4.0
    ox, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR, follow_px=2.0)
    assert [e["frame"] for e in events] == [300, 600]
    assert ox[800] == pytest.approx(5.0, abs=0.5)


def test_without_a_first_move_small_shifts_stay_ignored():
    lx, ly = _still(1000)
    lx[300:] += 4.0                          # 4 px held: never a first move
    _, _, events = er.step_offsets(lx - 500.0, ly - 300.0, window=W, max_drift_px=THR, follow_px=2.0)
    assert not events


# ---------------------------------------------------------------- light on


def test_eye_shift_while_the_light_is_on_is_ignored():
    # red light makes YOLO place the eye 5-10 px off while the head stays put
    # (3Oct october_11_batch_1 testing_9: shift 30.0-54.8 s, light 30.0-55.0 s)
    lx, ly = _still(900)
    lx[300:500] += 8.0
    t = _table(n=900, live=(lx, ly))
    moved, _ = er.reanchor_trial({1: t}, window=W, max_drift_px=THR)
    assert moved[1]["eye_moved"].any()                        # without the hold it would move
    held, summary = er.reanchor_trial({1: t}, window=W, max_drift_px=THR, hold=(300, 500))
    np.testing.assert_array_equal(held[1]["distance_0_1"], t["distance_0_1"])
    assert summary[0]["n_steps"] == 0


def test_real_slip_during_the_light_is_taken_up_after_it():
    lx, ly = _still(900)
    lx[400:] += 9.0                                           # slips at 400, light 300-500
    t = _table(n=900, live=(lx, ly))
    out, summary = er.reanchor_trial({1: t}, window=W, max_drift_px=THR, hold=(300, 500))
    assert summary[0]["first_step_frame"] == 501
    assert out[1]["distance_0_1"].iloc[700] == pytest.approx(51.0, abs=0.5)
    assert out[1]["distance_0_1"].iloc[450] == pytest.approx(60.0, abs=0.01)   # held: production value


def test_move_cannot_be_confirmed_with_frames_from_the_light():
    # the eye drifts 6 px just before light-on; the half-second that would
    # confirm it falls in the light, so the anchor keeps the original position
    # through the light (3Oct october_11_batch_2 testing_9 fly 1, 29.2 s)
    lx, ly = _still(900)
    lx[285:] += 6.0
    t = _table(n=900, live=(lx, ly))
    out, summary = er.reanchor_trial({1: t}, window=W, max_drift_px=THR, hold=(300, 600))
    d = out[1]["distance_0_1"].to_numpy()
    assert d[450] == pytest.approx(60.0, abs=0.01)           # light on: original eye
    assert summary[0]["first_step_frame"] >= 601             # taken up only after the light
    assert d[800] == pytest.approx(54.0, abs=0.5)
