"""Shared PER % oscilloscope helpers (scripts/analysis/lab_scope.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import lab_scope as s  # noqa: E402


def _trace(rest=40.0, sd=1.0, bump_t=None, bump=30.0, fps=40.0):
    t = np.arange(-12, 20, 1 / fps)
    rng = np.random.default_rng(0)
    v = rest + rng.normal(0, sd, t.size)
    if bump_t is not None:
        v[(t >= bump_t) & (t < bump_t + 1.0)] += bump
    return t, v


def test_threshold_is_rest_plus_floor_or_ten_sd():
    t, v = _trace(rest=40, sd=0.2)
    med, sd = s.rest_stats(v, t)
    assert s.per_threshold(v, t) == pytest.approx(med + 10.0)
    t, v = _trace(rest=40, sd=3.0)
    med, sd = s.rest_stats(v, t)
    assert s.per_threshold(v, t) == pytest.approx(med + 10 * sd)


def test_start_rule_odor_onset_without_early_movement():
    t, v = _trace(bump_t=5.0)
    assert not s.moves_early(v, t)
    assert s.clip_start_rel(v, t) == 0.0


def test_start_rule_one_second_early_with_movement_in_first_two_seconds():
    t, v = _trace(bump_t=1.2, bump=12.0)
    assert s.moves_early(v, t)
    assert s.clip_start_rel(v, t) == -1.0
    t, v = _trace(bump_t=2.5, bump=12.0)            # after the 2 s window: no
    assert s.clip_start_rel(v, t) == 0.0


def test_small_wiggle_is_not_movement():
    t, v = _trace(sd=0.2, bump_t=0.5, bump=3.0)      # 3 points < the 5-point floor
    assert not s.moves_early(v, t)


def test_per_pct_is_a_centred_three_frame_median():
    out = s.per_pct([0, 0, 90, 0, 0])
    assert list(out) == [0, 0, 0, 0, 0]


def test_stream_x_puts_the_head_at_the_right_edge():
    assert s.stream_x(3.0, 3.0, 800) == 800
    assert s.stream_x(2.0, 3.0, 800) == pytest.approx(800 - s.SPEED_PX_S)


def test_bands_never_draw_the_future():
    assert s.band_extent(0.0, 30.0, -0.5, 800) is None
    x0, x1 = s.band_extent(0.0, 30.0, 2.0, 800)
    assert x1 == 800 and x0 == pytest.approx(800 - 2 * s.SPEED_PX_S)


def test_split_at_threshold_inserts_the_crossing():
    xs, lo, hi = s.split_at_threshold([0, 1], [0, 10], 5)
    assert list(xs) == [0, 0.5, 1]
    assert lo[1] == 5 and hi[1] == 5 and np.isnan(lo[2]) and np.isnan(hi[0])


def test_readout_is_clamped_to_0_100_and_grows_above_threshold():
    assert s.format_readout(-3) == "0" and s.format_readout(104) == "100"
    assert s.readout_scale(10, 20) == 1.0
    assert s.readout_scale(20 + s.K_SPAN, 20) == pytest.approx(1 + s.READOUT_GAIN)


def test_x_ticks_run_back_from_now():
    secs = s.x_tick_seconds(840)
    assert secs[-1] == 0 and secs[0] == int(840 / s.SPEED_PX_S)
    assert s.x_tick_label(0) == "now" and s.x_tick_label(2) == "−2 s"


# ---------------------------------------------------------------- real data (skipped off-rig)
TRAINING = Path("/securedstorage/DATAsec/cole/Data-secured-New/Hex-Training-24-0.1/august_11_batch_1_rig_2")
needs_data = pytest.mark.skipif(not TRAINING.exists(), reason="secured data not mounted")


@needs_data
@pytest.mark.parametrize("trial", ["training_4", "testing_1", "testing_3"])
def test_training_video_trials_start_at_odor_onset(trial):
    import make_per_montage_video as m
    import pandas as pd
    import json
    td = m.trial_dir(trial)
    meta = json.loads((td / "_trial_meta.json").read_text())
    df = pd.read_parquet(td / f"{td.name}_{m.FLY}_distances.parquet")
    t = df["frame"].to_numpy(float) / float(meta["fps"]) - meta["odor_on_frame"] / float(meta["fps"])
    v = s.per_pct(df[s.PCT_COLUMN])
    assert s.clip_start_rel(v, t) == 0.0                  # none of the three moves in the first 2 s
