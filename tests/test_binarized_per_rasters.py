"""Tests for :mod:`scripts.analysis.binarized_per_rasters`.

The whole figure rests on one number per frame -- is the envelope above the red
threshold or not -- so the threshold is checked against the *project's own*
implementation (``envelope_visuals._baseline_theta``), not against a second copy
of the same arithmetic written here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis import binarized_per_rasters as mod  # noqa: E402
from scripts.analysis.envelope_visuals import _baseline_theta  # noqa: E402


# ---------------------------------------------------------------------------
# Threshold -- checked against the project's own function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_baseline_theta_matches_envelope_visuals(seed, k):
    rng = np.random.default_rng(seed)
    window = rng.gamma(2.0, 5.0, 400)
    assert mod.baseline_theta(window, k) == pytest.approx(_baseline_theta(window, k))


def test_baseline_theta_is_one_sided():
    """Downward dips are the opposite of an extension and must not raise theta."""
    base = np.full(200, 10.0)
    base[:20] = 12.0                       # a few upward samples set the scale
    theta_up_only = mod.baseline_theta(base, 2.0)

    dipped = base.copy()
    dipped[100:140] = -50.0                # huge downward excursion
    assert mod.baseline_theta(dipped, 2.0) == pytest.approx(theta_up_only)


def test_baseline_theta_flat_window_is_the_baseline():
    assert mod.baseline_theta(np.full(50, 7.5), 2.0) == pytest.approx(7.5)


def test_baseline_theta_empty_window_is_nan():
    assert np.isnan(mod.baseline_theta(np.array([]), 2.0))


def test_trial_theta_uses_only_the_pre_odor_baseline():
    """A big response after odor onset must not feed back into the threshold."""
    fps = 10.0
    trace = np.concatenate([np.full(100, 5.0), np.full(100, 500.0)])
    quiet = mod.trial_theta(trace, fps=fps, baseline_until_s=10.0, k=2.0)
    assert quiet == pytest.approx(5.0)


def test_trial_theta_nan_without_a_baseline():
    assert np.isnan(mod.trial_theta(np.arange(10.0), fps=10.0, baseline_until_s=0.0, k=2.0))
    assert np.isnan(mod.trial_theta(np.array([]), fps=10.0, baseline_until_s=5.0, k=2.0))


# ---------------------------------------------------------------------------
# Binarisation
# ---------------------------------------------------------------------------


def test_binarize_is_strictly_above_theta():
    """Matches envelope_visuals' `during > theta`; sitting exactly on the line is 0."""
    trace = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        mod.binarize(trace, 2.0), np.array([False, False, True, True])
    )


def test_binarize_treats_missing_samples_as_no_response():
    trace = np.array([1.0, np.nan, 9.0])
    np.testing.assert_array_equal(mod.binarize(trace, 2.0), np.array([False, False, True]))


def test_binarize_all_false_when_theta_is_nan():
    trace = np.array([1.0, 9.0, 100.0])
    assert not mod.binarize(trace, float("nan")).any()


def test_window_fraction_is_hand_computable():
    binary = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    # fps = 1, so seconds are indices: [2, 6) covers exactly the four True samples.
    assert mod.window_fraction(binary, fps=1.0, start_s=2.0, end_s=6.0) == pytest.approx(1.0)
    assert mod.window_fraction(binary, fps=1.0, start_s=0.0, end_s=10.0) == pytest.approx(0.4)
    assert mod.window_fraction(binary, fps=1.0, start_s=6.0, end_s=10.0) == pytest.approx(0.0)


def test_window_fraction_empty_window_is_nan():
    binary = np.ones(10, dtype=bool)
    assert np.isnan(mod.window_fraction(binary, fps=1.0, start_s=5.0, end_s=5.0))


def test_window_fraction_clips_to_the_trace():
    binary = np.ones(10, dtype=bool)
    assert mod.window_fraction(binary, fps=1.0, start_s=8.0, end_s=99.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fixtures: a miniature cohort
# ---------------------------------------------------------------------------

DATASET = "Hex-Control-24-0.1"
N_FRAMES = 12
FPS = 1.0
ODOR_ON, ODOR_OFF = 4.0, 8.0


def _row(fly, num, trial_type, label, during_ones, auc=None):
    """A trace whose pre-odor baseline is flat 1.0 with a single 3.0 spike.

    theta = 1 + 2 * 1.4826 * median(upward devs) -- the single 2.0 deviation makes
    it 1 + 2*1.4826*2 = 6.93, so a 10.0 sample counts and a 3.0 sample does not.
    """
    trace = np.full(N_FRAMES, 1.0)
    trace[1] = 3.0                                   # sets the baseline scale
    for i in range(during_ones):
        trace[int(ODOR_ON) + i] = 10.0
    row = {
        "dataset": DATASET, "fly": fly, "fly_number": num,
        "trial_type": trial_type, "trial_label": label,
        "trace_len": N_FRAMES, "fps": FPS,
        "trial_odor_on_s": ODOR_ON, "trial_odor_off_s": ODOR_OFF,
        "AUC-During": float(during_ones * 10.0 if auc is None else auc),
    }
    row.update({f"dir_val_{i}": float(v) for i, v in enumerate(trace)})
    return row


#: fly -> ones during odor on each of the 3 training trials
_TRAIN = {
    ("day_1", 1): [4, 4, 4],      # strongest
    ("day_1", 2): [2, 2, 2],
    ("day_2", 1): [0, 1, 2],
    ("day_2", 2): [0, 0, 0],      # weakest
}

#: Deliberately the REVERSE of the binarised ranking, so a test that claims to
#: sort by AUC cannot pass by accidentally reproducing the binary order.
_TRAIN_AUC = {
    ("day_1", 1): 10.0,
    ("day_1", 2): 20.0,
    ("day_2", 1): 30.0,
    ("day_2", 2): 40.0,
}

_TEST_ODORS = ["hexanol", "citral", "acv", "hexanol", "lightonly"]


def _training_frame():
    rows = []
    for (fly, num), ones in _TRAIN.items():
        for t, n_ones in enumerate(ones, start=1):
            rows.append(_row(fly, num, "training", f"training_{t}_hexanol", n_ones,
                             auc=_TRAIN_AUC[(fly, num)]))
    return pd.DataFrame(rows)


def _testing_frame():
    rows = []
    for (fly, num) in _TRAIN:
        for t, odor in enumerate(_TEST_ODORS, start=1):
            rows.append(_row(fly, num, "testing", f"testing_{t}_{odor}", t % 3))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trial table
# ---------------------------------------------------------------------------


def test_build_trials_parses_and_scores_each_trial():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    assert len(t) == 12
    assert set(t["odor"]) == {"hexanol"}
    assert sorted(t["trial_index"].unique()) == [1, 2, 3]

    row = t[(t.fly == "day_1") & (t.fly_number == 1) & (t.trial_index == 1)].iloc[0]
    # theta = 1 + 2*1.4826*2 = 6.9304 -> only the 10.0 samples clear it
    assert row["theta"] == pytest.approx(1.0 + 2 * 1.4826 * 2.0)
    assert row["n_ones_odor"] == 4
    assert row["odor_fraction"] == pytest.approx(1.0)      # 4 of 4 odor frames


def test_build_trials_respects_a_keep_list():
    keep = {("day_1", 1), ("day_2", 2)}
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training", keep=keep)
    assert set(zip(t.fly, t.fly_number)) == keep


def test_build_trials_ignores_other_datasets_and_trial_types():
    df = pd.concat([_training_frame(), _testing_frame()], ignore_index=True)
    t = mod.build_trials(df, dataset=DATASET, trial_type="training")
    assert set(t["trial_type"]) == {"training"}
    assert mod.build_trials(df, dataset="Nope-24-1", trial_type="training").empty


def test_build_trials_binary_length_matches_trace_len():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    for b in t["binary"]:
        assert len(b) == N_FRAMES


# ---------------------------------------------------------------------------
# Ordering -- the spine of both figures
# ---------------------------------------------------------------------------


def test_fly_order_is_by_mean_odor_fraction_descending():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    order = mod.fly_order(t)
    assert order == ["day_1#1", "day_1#2", "day_2#1", "day_2#2"]


def test_fly_order_uses_the_mean_across_all_training_trials():
    """day_2/1 ramps 0,1,2 -> mean 0.25; day_1/2 is flat 2,2,2 -> mean 0.5."""
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    scores = mod.fly_scores(t)
    assert scores["day_1#2"] == pytest.approx(0.5)
    assert scores["day_2#1"] == pytest.approx(0.25)


def test_fly_order_tie_break_is_deterministic():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    t = t.copy()
    t["odor_fraction"] = 0.5                            # force a total tie
    assert mod.fly_order(t) == sorted(mod.fly_order(t))  # falls back to name order


# ---------------------------------------------------------------------------
# Testing panels -- keyed on odor, not on trial index
# ---------------------------------------------------------------------------


def test_testing_panels_split_a_repeated_odor_by_presentation():
    t = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    panels = mod.testing_panels(t)
    labels = [p.label for p in panels]
    assert "Hexanol 1" in labels and "Hexanol 2" in labels
    assert labels.count("Hexanol 1") == 1


def test_repeated_odor_presentations_sit_next_to_each_other():
    """Hexanol 2 lands at testing_8 but must be drawn beside Hexanol 1."""
    t = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    labels = [p.label for p in mod.testing_panels(t)]
    assert labels.index("Hexanol 2") == labels.index("Hexanol 1") + 1


def test_testing_panels_lead_with_the_earliest_odor():
    t = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    labels = [p.label for p in mod.testing_panels(t)]
    assert labels[0] == "Hexanol 1"          # always testing_1
    assert labels.index("Citral") > labels.index("Hexanol 2")


def test_testing_panels_drop_light_only_by_default():
    t = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    assert "lightonly" in set(t["odor"])                 # it is in the data
    labels = [p.label for p in mod.testing_panels(t)]
    assert not any("Light" in l for l in labels)


def test_testing_panels_can_keep_light_only():
    t = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    labels = [p.label for p in mod.testing_panels(t, exclude_odors=())]
    assert labels[-1] == "Light only"


def test_testing_panels_cover_every_kept_trial_exactly_once():
    t = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    panels = mod.testing_panels(t, exclude_odors=())
    assert sum(len(p.trials) for p in panels) == len(t)
    seen = pd.concat([p.trials for p in panels])
    assert not seen.duplicated(subset=["fly", "fly_number", "trial_index"]).any()


# ---------------------------------------------------------------------------
# Odor labels come from the config's odor_remap, not from a hardcoded name
# ---------------------------------------------------------------------------


@pytest.fixture()
def restore_remap():
    from scripts.analysis import envelope_visuals as ev

    saved = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    yield
    ev._DATASET_ODOR_REMAP.clear()
    ev._DATASET_ODOR_REMAP.update(saved)


def test_odor_display_applies_the_registered_remap(restore_remap):
    """config_new.yaml says this cohort's ACV channel delivered isoamyl acetate."""
    from scripts.analysis.envelope_visuals import set_dataset_odor_remap

    set_dataset_odor_remap(
        {DATASET: {"Apple Cider Vinegar": "Isoamyl Acetate (1%)",
                   "Hexanol": "Hexanol (0.1%)"}}
    )
    assert mod.odor_display("acv", DATASET) == "Isoamyl Acetate (1%)"
    assert mod.odor_display("hexanol", DATASET) == "Hexanol (0.1%)"
    # An odor the remap does not mention keeps its plain display name.
    assert mod.odor_display("citral", DATASET) == "Citral"
    # A different dataset is untouched.
    assert mod.odor_display("acv", "Other-24-1") == "Apple Cider Vinegar"


def test_odor_display_without_a_remap_is_the_plain_name(restore_remap):
    from scripts.analysis.envelope_visuals import set_dataset_odor_remap

    set_dataset_odor_remap({})
    assert mod.odor_display("acv", DATASET) == "Apple Cider Vinegar"


def test_panel_labels_use_the_remapped_odor(restore_remap):
    from scripts.analysis.envelope_visuals import set_dataset_odor_remap

    set_dataset_odor_remap({DATASET: {"Apple Cider Vinegar": "Isoamyl Acetate (1%)"}})
    t = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    labels = [p.label for p in mod.testing_panels(t)]
    assert "Isoamyl Acetate (1%)" in labels
    assert "Apple Cider Vinegar" not in labels
    assert "ACV" not in labels


def test_load_config_remap_reads_the_real_config(restore_remap):
    """The shipped config already declares this dataset's ACV correction."""
    n = mod.load_config_remap(REPO_ROOT / "config" / "config_new.yaml")
    assert n > 0
    assert mod.odor_display("acv", "Hex-Control-24-0.1") == "Isoamyl Acetate (1%)"


# ---------------------------------------------------------------------------
# Raster assembly
# ---------------------------------------------------------------------------


def test_raster_rows_follow_the_given_order():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    order = ["day_2#2", "day_1#1", "day_2#1", "day_1#2"]
    grid, time_axis = mod.raster(t[t.trial_index == 1], order, pre_s=2.0, post_s=6.0, bin_s=1.0)
    assert grid.shape[0] == len(order)
    # day_2#2 has no response at all; day_1#1 responds through the odor window.
    assert np.nansum(grid[0]) == 0
    assert np.nansum(grid[1]) > 0


def test_raster_marks_a_missing_fly_as_nan_not_zero():
    """A blank row must read as 'not tested', never as 'tested and silent'."""
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    order = ["day_1#1", "ghost#9"]
    grid, _ = mod.raster(t[t.trial_index == 1], order, pre_s=2.0, post_s=6.0, bin_s=1.0)
    assert np.isnan(grid[1]).all()
    assert not np.isnan(grid[0]).all()


def test_raster_is_aligned_on_odor_onset():
    """Time zero is odor onset, so the pre-odor columns are the baseline."""
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    grid, time_axis = mod.raster(
        t[t.trial_index == 1], mod.fly_order(t), pre_s=4.0, post_s=4.0, bin_s=1.0
    )
    # time_axis carries bin CENTRES, so the first one sits half a bin inside -pre_s.
    assert time_axis[0] == pytest.approx(-4.0 + 1.0 / 2)
    assert time_axis[-1] == pytest.approx(4.0 - 1.0 / 2)
    zero = int(np.argmin(np.abs(time_axis)))
    # Every response in the fixture starts at odor onset, so nothing fires before it.
    assert np.nansum(grid[:, :zero]) == 0
    assert np.nansum(grid[:, zero:]) > 0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_training_figure_has_one_panel_per_trial():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    fig, meta = mod.figure_training(t, mod.fly_order(t), dataset=DATASET)
    try:
        assert meta["n_flies"] == 4
        assert [p["label"] for p in meta["panels"]] == ["Training 1", "Training 2", "Training 3"]
    finally:
        plt.close(fig)


def test_testing_figure_uses_the_training_order_unchanged():
    train = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    test = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    order = mod.fly_order(train)
    fig, meta = mod.figure_testing(test, order, dataset=DATASET)
    try:
        assert meta["fly_order"] == order
        labels = [p["label"] for p in meta["panels"]]
        assert labels[0] == "Hexanol 1"
        assert labels[1] == "Hexanol 2"          # the two presentations sit together
        assert not any("Light" in l for l in labels)
    finally:
        plt.close(fig)


def test_both_figures_report_the_same_fly_order():
    train = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    test = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    order = mod.fly_order(train)
    f1, m1 = mod.figure_training(train, order, dataset=DATASET)
    f2, m2 = mod.figure_testing(test, order, dataset=DATASET)
    try:
        assert m1["fly_order"] == m2["fly_order"] == order
    finally:
        plt.close(f1)
        plt.close(f2)


def test_binary_palette_reuses_the_project_reaction_green():
    from scripts.analysis.score_scale_figure import SCORE_COLORS

    assert mod.ON_COLOR == SCORE_COLORS[4]


# ---------------------------------------------------------------------------
# Continuous (envelope) variant
# ---------------------------------------------------------------------------


def test_build_trials_carries_the_continuous_trace_and_auc():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    row = t[(t.fly == "day_1") & (t.fly_number == 1) & (t.trial_index == 1)].iloc[0]
    assert len(row["trace"]) == N_FRAMES
    assert row["trace"][int(ODOR_ON)] == pytest.approx(10.0)
    assert row["auc_during"] == pytest.approx(10.0)


def test_fly_order_by_auc_uses_mean_auc_during():
    """Sorting by AUC must give the reverse of the binarised ranking here."""
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    assert mod.fly_order(t, by="auc") == ["day_2#2", "day_2#1", "day_1#2", "day_1#1"]
    assert mod.fly_order(t, by="binary") == ["day_1#1", "day_1#2", "day_2#1", "day_2#2"]


def test_fly_scores_by_auc_are_the_per_fly_means():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    scores = mod.fly_scores(t, by="auc")
    assert scores["day_2#2"] == pytest.approx(40.0)
    assert scores["day_1#1"] == pytest.approx(10.0)


def test_fly_order_rejects_an_unknown_key():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    with pytest.raises(ValueError):
        mod.fly_order(t, by="nonsense")


def test_raster_can_return_the_continuous_trace():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    order = mod.fly_order(t, by="auc")
    grid, _ = mod.raster(
        t[t.trial_index == 1], order, pre_s=4.0, post_s=4.0, bin_s=1.0, column="trace"
    )
    assert np.nanmax(grid) == pytest.approx(10.0)     # not 1.0 -- real units
    assert np.nanmin(grid) == pytest.approx(1.0)


def test_raster_continuous_still_marks_a_missing_fly_nan():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    grid, _ = mod.raster(
        t[t.trial_index == 1], ["day_1#1", "ghost#9"],
        pre_s=2.0, post_s=6.0, bin_s=1.0, column="trace",
    )
    assert np.isnan(grid[1]).all()


def test_envelope_mode_is_a_perceptually_uniform_sequential_map():
    """Cool-at-zero to warm-at-max, but never a rainbow: viridis family only."""
    assert mod.ENVELOPE_MODE.column == "trace"
    assert mod.ENVELOPE_MODE.cmap in {"viridis", "magma", "inferno", "plasma", "cividis"}
    assert (mod.ENVELOPE_MODE.vmin, mod.ENVELOPE_MODE.vmax) == (0.0, 100.0)
    assert mod.BINARY_MODE.column == "binary"


def test_heatmap_figure_renders_with_a_colorbar():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    order = mod.fly_order(t, by="auc")
    fig, meta = mod.figure_training(t, order, dataset=DATASET, mode=mod.ENVELOPE_MODE)
    try:
        assert meta["mode"] == "envelope"
        assert meta["fly_order"] == order
        assert [p["label"] for p in meta["panels"]] == [
            "Training 1", "Training 2", "Training 3"
        ]
        # One extra axes beyond the three panels: the colorbar.
        assert len(fig.axes) == 4
    finally:
        plt.close(fig)


def test_heatmap_testing_figure_keeps_the_auc_order():
    train = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    test = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    order = mod.fly_order(train, by="auc")
    fig, meta = mod.figure_testing(test, order, dataset=DATASET, mode=mod.ENVELOPE_MODE)
    try:
        assert meta["fly_order"] == order
        assert meta["mode"] == "envelope"
    finally:
        plt.close(fig)


def test_binary_figure_has_no_colorbar():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    fig, meta = mod.figure_training(t, mod.fly_order(t), dataset=DATASET)
    try:
        assert meta["mode"] == "binary"
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


def test_mode_vmax_override_is_respected():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    mode = mod.ENVELOPE_MODE.with_vmax(12.0)
    assert mode.vmax == pytest.approx(12.0)
    assert mode.vmin == pytest.approx(0.0)
    fig, meta = mod.figure_training(t, mod.fly_order(t, by="auc"),
                                    dataset=DATASET, mode=mode)
    try:
        assert meta["scale"]["vmax"] == pytest.approx(12.0)
    finally:
        plt.close(fig)


def test_robust_vmax_is_a_percentile_of_the_real_data():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    v = mod.robust_vmax(t, percentile=99.0)
    stacked = np.concatenate([np.asarray(x, dtype=float) for x in t["trace"]])
    assert v == pytest.approx(np.nanpercentile(stacked, 99.0))


# --------------------------------------------------------------------------- #
# Threshold knobs plumbed through build_trials.
#
# The raster must be able to render either baseline rule so the two can be put
# side by side. Defaults stay off, so an existing raster is byte-identical.
# --------------------------------------------------------------------------- #


def test_build_trials_defaults_reproduce_the_shipped_threshold():
    plain = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    explicit = mod.build_trials(
        _training_frame(), dataset=DATASET, trial_type="training", min_delta=0.0, anchor_s=None
    )
    assert plain["theta"].tolist() == explicit["theta"].tolist()


def test_build_trials_min_delta_raises_every_theta():
    plain = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    floored = mod.build_trials(
        _training_frame(), dataset=DATASET, trial_type="training", min_delta=50.0
    )
    assert (floored["theta"].to_numpy() > plain["theta"].to_numpy()).all()
    # A floor that clears the 10.0 response samples zeroes the odor fraction.
    assert floored["odor_fraction"].max() == 0.0




def _anchor_frame():
    """Baseline that RISES, so its last-seconds median differs from its overall one.

    _row()'s baseline is flat apart from one spike, which makes the anchored and
    whole-window locations coincide -- fine for the other tests, useless for
    proving the anchor is wired through. Here the baseline is [1, 1, 5, 5]:
    whole-window median 3, last-2 s median 5.
    """
    rows = []
    for t in (1, 2, 3):
        trace = np.array([1.0, 1.0, 5.0, 5.0, 10.0, 10.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0])
        row = {
            "dataset": DATASET, "fly": "day_1", "fly_number": 1,
            "trial_type": "training", "trial_label": f"training_{t}_hexanol",
            "trace_len": N_FRAMES, "fps": FPS,
            "trial_odor_on_s": ODOR_ON, "trial_odor_off_s": ODOR_OFF,
            "AUC-During": 20.0,
        }
        row.update({f"dir_val_{i}": float(v) for i, v in enumerate(trace)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_trials_anchor_changes_theta_and_rebinarises():
    """The anchor is a different estimator, so theta must actually move."""
    frame = _anchor_frame()
    anchored = mod.build_trials(
        frame, dataset=DATASET, trial_type="training", anchor_s=2.0, min_delta=0.0
    )
    plain = mod.build_trials(frame, dataset=DATASET, trial_type="training")
    # Baseline rises 1 -> 5, so the last-2 s median (5) sits above the
    # whole-window median (3) and the anchored threshold is the higher one.
    assert (anchored["theta"].to_numpy() > plain["theta"].to_numpy()).all()


def test_build_trials_theta_columns_are_finite():
    for kwargs in ({}, {"min_delta": 5.0}, {"anchor_s": 2.0, "min_delta": 5.0}):
        t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training", **kwargs)
        assert np.isfinite(t["theta"].to_numpy()).all()


def test_binary_raster_can_sort_by_auc():
    """--sort-by auc must use AUC-During, not the theta-dependent binary fraction.

    _TRAIN_AUC is deliberately the reverse of the binarised ranking, so an
    implementation that ignored the flag could not pass this by accident.
    """
    train = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    assert mod.fly_order(train, by="auc") != mod.fly_order(train, by="binary")
    assert mod.fly_order(train, by="auc") == list(reversed(mod.fly_order(train, by="binary")))


def test_auc_ordering_is_threshold_independent():
    """The point of sorting by AUC: the row order stops moving when theta moves."""
    plain = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    shifted = mod.build_trials(
        _training_frame(), dataset=DATASET, trial_type="training",
        min_delta=50.0, anchor_s=2.0,
    )
    assert mod.fly_order(plain, by="auc") == mod.fly_order(shifted, by="auc")
    assert mod.fly_scores(plain, by="auc") == mod.fly_scores(shifted, by="auc")
    # ...whereas the binary key is a function of theta, which is why the flag
    # matters: change the baseline rule and the sort key itself moves.
    assert mod.fly_scores(plain, by="binary") != mod.fly_scores(shifted, by="binary")


# --------------------------------------------------------------------------- #
# Graded mode: above-threshold cells shaded by how far above BASELINE they sit.
# --------------------------------------------------------------------------- #


def test_graded_column_is_zero_below_threshold():
    """Below θ carries no magnitude, so it can be painted with the 'under' colour."""
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    row = t.iloc[0]
    g = np.asarray(row["graded"], dtype=float)
    b = np.asarray(row["binary"], dtype=bool)
    assert (g[~b] == 0.0).all()


def test_graded_column_is_excess_over_baseline_not_over_theta():
    """The user asked for 'baseline subtracted %', which is not θ subtracted.

    θ sits above the resting median by k·σ; measuring from θ would report a
    response as smaller than it is by exactly that margin, and by a different
    margin on every trial.
    """
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    row = t.iloc[0]
    trace = np.asarray(row["trace"], dtype=float)
    g = np.asarray(row["graded"], dtype=float)
    b = np.asarray(row["binary"], dtype=bool)
    loc = float(row["baseline_loc"])
    assert loc == pytest.approx(1.0)          # the fixture's flat baseline
    assert g[b] == pytest.approx(trace[b] - loc)
    assert (g[b] > row["theta"] - loc).all()  # ...and strictly larger than θ-excess


def test_graded_mode_is_registered_with_a_sequential_green_ramp():
    m = mod.MODES["graded"]
    assert m.column == "graded"
    assert m.vmin > 0.0        # so 0 falls to set_under, i.e. the below-θ colour
    assert m.under_color == mod.OFF_COLOR


def test_graded_ramp_runs_light_to_the_pinned_green():
    """Sequential = one hue, light to dark, ending on the project's ON colour."""
    cm = mod.graded_cmap()
    light, dark = cm(0.0), cm(1.0)
    assert sum(light[:3]) > sum(dark[:3])          # light end really is lighter
    assert dark[:3] == pytest.approx(
        tuple(int(mod.ON_COLOR[i:i + 2], 16) / 255 for i in (1, 3, 5)), abs=0.02
    )


def test_robust_vmax_reads_the_graded_column():
    t = mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")
    v = mod.robust_vmax(t, column="graded")
    assert 0.0 < v <= float(np.nanmax(np.concatenate(t["graded"].tolist())))


def test_graded_baseline_loc_follows_the_threshold_rule():
    """Anchored θ moves the baseline location, so the excess must move with it."""
    frame = _anchor_frame()
    plain = mod.build_trials(frame, dataset=DATASET, trial_type="training")
    anchored = mod.build_trials(
        frame, dataset=DATASET, trial_type="training", anchor_s=2.0
    )
    assert plain["baseline_loc"].iloc[0] == pytest.approx(3.0)     # whole window
    assert anchored["baseline_loc"].iloc[0] == pytest.approx(5.0)  # last 2 s
