"""Tests for the cohort-average training PER trace driver.

The published per-fly figure (``*_training_envelope_trials_by_odor_30_shifted.png``)
stacks one panel per training trial for a *single* fly. This driver draws the
same six-panel layout from the cohort mean ± SEM across every fly in a dataset,
so a whole training cohort reads as one figure.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import envelope_visuals as ev
from scripts.analysis.avg_training_traces_dataset import (
    TRAINING_TRIALS,
    collect_training_traces,
    mean_sem,
    main,
    resample_traces,
    select_dataset_rows,
)

DATASET = "3Oct-Training-24-0.1"
EB_DATASET = "EB-Training-24-1"

REMAP = {
    "3-Octanol": "3-Octanol (0.1%)",
    "Ethyl Butyrate": "Ethyl Butyrate (1%)",
}

N_FRAMES = 120
FPS = 40.0
ODOR_ON_S = 0.5           # 20 baseline frames at 40 fps — keeps fixtures small
ODOR_OFF_S = 1.5
MAX_TIME_S = 2.5
BASELINE_FRAMES = int(round(ODOR_ON_S * FPS))
LIGHT_ON_S = 0.8


@pytest.fixture(autouse=True)
def _registered_remap():
    """Register/restore the odor remap the way run_workflows does."""
    saved_protocol = ev.get_protocol()
    saved_remap = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    ev.set_protocol("v2")
    ev.set_dataset_odor_remap({DATASET: dict(REMAP), EB_DATASET: dict(REMAP)})
    try:
        yield
    finally:
        ev.set_protocol(saved_protocol)
        ev.set_dataset_odor_remap(saved_remap)


def _rows_for_fly(dataset, fly, fly_number, odor, *, value_of_trial, trace_len=N_FRAMES):
    rows = []
    for trial in TRAINING_TRIALS:
        row = {
            "dataset": dataset,
            "fly": fly,
            "fly_number": fly_number,
            "trial_type": "training",
            "trial_label": f"training_{trial}_{odor}",
            "fps": FPS,
            "trial_odor_on_s": ODOR_ON_S,
            "trial_odor_off_s": ODOR_OFF_S,
            "trial_light_on_s": LIGHT_ON_S,
            "trace_len": trace_len,
        }
        vals = np.full(N_FRAMES, np.nan)
        vals[:trace_len] = float(value_of_trial(trial))
        row.update({f"dir_val_{i}": vals[i] for i in range(N_FRAMES)})
        rows.append(row)
    return rows


def _wide_frame():
    rows = []
    # 3Oct: two batch-1 flies and one batch-2 fly, constant traces 10*trial + fly
    for offset, (fly, number) in enumerate(
        [("july_20_batch_1_rig_2", 1), ("july_20_batch_1_rig_2", 2), ("july_21_batch_2", 1)]
    ):
        rows += _rows_for_fly(
            DATASET, fly, number, "3-octonol",
            value_of_trial=lambda t, o=offset: 10 * t + o,
        )
    # EB: one batch-1 fly, one batch-2 fly
    for offset, (fly, number) in enumerate(
        [("july_13_batch_1", 1), ("july_14_batch_2_rig_2", 1)]
    ):
        rows += _rows_for_fly(
            EB_DATASET, fly, number, "ethylbutyrate",
            value_of_trial=lambda t, o=offset: 100 + o,
        )
    return pd.DataFrame(rows)


def _dir_cols(df):
    return sorted(
        [c for c in df.columns if c.startswith("dir_val_")],
        key=lambda c: int(c.split("_")[-1]),
    )


# --------------------------------------------------------------------------
# select_dataset_rows


def test_select_dataset_rows_keeps_only_the_named_dataset():
    df = _wide_frame()
    out = select_dataset_rows(df, DATASET)
    assert set(out["dataset"]) == {DATASET}
    assert out.groupby(["fly", "fly_number"]).ngroups == 3


def test_select_dataset_rows_batch_filter_keeps_only_that_batch():
    df = _wide_frame()
    out = select_dataset_rows(df, DATASET, batch=1)
    assert set(out["fly"]) == {"july_20_batch_1_rig_2"}
    assert out.groupby(["fly", "fly_number"]).ngroups == 2


def test_select_dataset_rows_batch_filter_on_eb_drops_batch_2():
    df = _wide_frame()
    out = select_dataset_rows(df, EB_DATASET, batch=1)
    assert set(out["fly"]) == {"july_13_batch_1"}


def test_select_dataset_rows_drops_non_training_rows():
    df = _wide_frame()
    df.loc[df.index[0], "trial_type"] = "testing"
    out = select_dataset_rows(df, DATASET)
    assert set(out["trial_type"]) == {"training"}


def test_select_dataset_rows_unknown_dataset_is_empty():
    assert select_dataset_rows(_wide_frame(), "Nope-Training").empty


# --------------------------------------------------------------------------
# collect_training_traces


def test_collect_keys_traces_by_trial_then_fly():
    df = select_dataset_rows(_wide_frame(), DATASET)
    traces, _, _ = collect_training_traces(df, dir_cols=_dir_cols(df))
    assert sorted(traces) == list(TRAINING_TRIALS)
    for trial in TRAINING_TRIALS:
        assert sorted(traces[trial]) == [
            "july_20_batch_1_rig_2_fly1",
            "july_20_batch_1_rig_2_fly2",
            "july_21_batch_2_fly1",
        ]


def test_collect_returns_raw_values_without_baseline_frames():
    df = select_dataset_rows(_wide_frame(), DATASET)
    traces, _, _ = collect_training_traces(df, dir_cols=_dir_cols(df))
    trace = traces[3]["july_20_batch_1_rig_2_fly1"]
    assert np.allclose(trace, 30.0)


def test_collect_baseline_subtracts_when_baseline_frames_given():
    df = select_dataset_rows(_wide_frame(), DATASET)
    traces, _, _ = collect_training_traces(
        df, dir_cols=_dir_cols(df), baseline_frames=BASELINE_FRAMES
    )
    # Constant trace minus its own pre-odor mean is exactly zero.
    assert np.allclose(traces[3]["july_20_batch_1_rig_2_fly1"], 0.0)


def test_collect_trims_trailing_nan_padding():
    df = _wide_frame()
    short = 47
    for i in range(short, N_FRAMES):
        df.loc[df.index[0], f"dir_val_{i}"] = np.nan
    df = select_dataset_rows(df, DATASET)
    traces, _, _ = collect_training_traces(df, dir_cols=_dir_cols(df))
    assert len(traces[1]["july_20_batch_1_rig_2_fly1"]) == short


def test_collect_skips_all_nan_rows():
    df = _wide_frame()
    row = df.index[0]
    for i in range(N_FRAMES):
        df.loc[row, f"dir_val_{i}"] = np.nan
    df = select_dataset_rows(df, DATASET)
    traces, _, _ = collect_training_traces(df, dir_cols=_dir_cols(df))
    assert "july_20_batch_1_rig_2_fly1" not in traces[1]
    assert "july_20_batch_1_rig_2_fly1" in traces[2]


def test_collect_restricts_to_requested_trials():
    df = select_dataset_rows(_wide_frame(), DATASET)
    traces, _, _ = collect_training_traces(df, dir_cols=_dir_cols(df), trials=(1, 2))
    assert sorted(traces) == [1, 2]


def test_collect_ignores_trial_numbers_outside_the_schedule():
    df = _wide_frame()
    df.loc[df.index[0], "trial_label"] = "training_9_3-octonol"
    df = select_dataset_rows(df, DATASET)
    traces, _, _ = collect_training_traces(df, dir_cols=_dir_cols(df))
    assert 9 not in traces
    assert "july_20_batch_1_rig_2_fly1" not in traces[1]


def test_collect_gathers_measured_light_on_times():
    df = select_dataset_rows(_wide_frame(), DATASET)
    _, light, _ = collect_training_traces(df, dir_cols=_dir_cols(df))
    assert light[1] == pytest.approx([LIGHT_ON_S] * 3)


def test_collect_drops_nonpositive_light_on_times():
    df = _wide_frame()
    df.loc[df.index[0], "trial_light_on_s"] = 0.0
    df.loc[df.index[1], "trial_light_on_s"] = np.nan
    df = select_dataset_rows(df, DATASET)
    _, light, _ = collect_training_traces(df, dir_cols=_dir_cols(df))
    assert light[1] == pytest.approx([LIGHT_ON_S] * 2)
    assert light[2] == pytest.approx([LIGHT_ON_S] * 2)


def test_collect_returns_the_odor_display_label():
    df = select_dataset_rows(_wide_frame(), DATASET)
    _, _, odor = collect_training_traces(df, dir_cols=_dir_cols(df), dataset=DATASET)
    assert odor == "3-Octanol (0.1%)"


def test_collect_returns_eb_display_label():
    df = select_dataset_rows(_wide_frame(), EB_DATASET)
    _, _, odor = collect_training_traces(df, dir_cols=_dir_cols(df), dataset=EB_DATASET)
    assert odor == "Ethyl Butyrate (1%)"


def test_collect_keeps_one_trace_per_fly_and_trial_when_rows_duplicate():
    df = _wide_frame()
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)
    df = select_dataset_rows(df, DATASET)
    traces, _, _ = collect_training_traces(df, dir_cols=_dir_cols(df))
    assert len(traces[1]) == 3


# --------------------------------------------------------------------------
# resample_traces / mean_sem


def test_resample_puts_every_fly_on_a_common_time_grid():
    per_fly = {"a": np.full(80, 1.0), "b": np.full(100, 2.0)}
    t, matrix, ids = resample_traces(per_fly, fps=FPS, max_time_s=MAX_TIME_S)
    assert matrix.shape == (2, len(t))
    assert t[0] == 0.0 and t[-1] == pytest.approx(MAX_TIME_S)
    assert ids == ["a", "b"]


def test_resample_pads_short_traces_with_nan_beyond_their_end():
    t, matrix, ids = resample_traces(
        {"short": np.full(41, 5.0)}, fps=FPS, max_time_s=MAX_TIME_S
    )
    assert np.isfinite(matrix[0][t <= 1.0]).all()
    assert np.isnan(matrix[0][t > 1.0 + 1e-9]).all()


def test_resample_is_stable_for_an_empty_cohort():
    t, matrix, ids = resample_traces({}, fps=FPS, max_time_s=MAX_TIME_S)
    assert matrix.shape[0] == 0
    assert ids == []
    assert len(t) > 0


def test_mean_sem_uses_the_per_sample_finite_count():
    matrix = np.array([[1.0, 1.0], [3.0, np.nan], [5.0, 5.0]])
    mean, sem, n = mean_sem(matrix)
    assert mean == pytest.approx([3.0, 3.0])
    assert n.tolist() == [3, 2]
    assert sem[0] == pytest.approx(np.std([1.0, 3.0, 5.0]) / np.sqrt(3))
    assert sem[1] == pytest.approx(np.std([1.0, 5.0]) / np.sqrt(2))


def test_mean_sem_is_zero_for_a_single_fly():
    mean, sem, n = mean_sem(np.array([[2.0, 4.0]]))
    assert mean == pytest.approx([2.0, 4.0])
    assert sem == pytest.approx([0.0, 0.0])
    assert n.tolist() == [1, 1]


def test_mean_sem_reports_zero_coverage_for_all_nan_columns():
    mean, sem, n = mean_sem(np.array([[np.nan], [np.nan]]))
    assert n.tolist() == [0]
    assert np.isnan(mean[0])


# --------------------------------------------------------------------------
# main / end-to-end


def _run(tmp_path, wide, extra=()):
    csv = tmp_path / "wide_training.csv"
    wide.to_csv(csv, index=False)
    out = tmp_path / "figs"
    main([
        "--wide-csv", str(csv),
        "--dataset", DATASET,
        "--out-dir", str(out),
        "--fps", str(FPS),
        "--odor-on-s", str(ODOR_ON_S),
        "--odor-off-s", str(ODOR_OFF_S),
        "--max-time-s", str(MAX_TIME_S),
        *extra,
    ])
    return out


def test_main_writes_a_png_and_a_json_sidecar(tmp_path):
    out = _run(tmp_path, _wide_frame())
    assert (out / "avg_training_traces_3Oct-Training-24-0.1.png").exists()
    assert (out / "avg_training_traces_3Oct-Training-24-0.1.json").exists()


def test_main_sidecar_records_per_trial_fly_counts(tmp_path):
    out = _run(tmp_path, _wide_frame())
    meta = json.loads(
        (out / "avg_training_traces_3Oct-Training-24-0.1.json").read_text()
    )
    assert meta["dataset"] == DATASET
    assert meta["odor"] == "3-Octanol (0.1%)"
    assert [meta["per_trial"][str(t)]["n_flies"] for t in TRAINING_TRIALS] == [3] * 6
    assert meta["n_flies"] == 3
    assert meta["baseline_subtracted"] is False


def test_main_batch_filter_narrows_the_cohort(tmp_path):
    out = _run(tmp_path, _wide_frame(), extra=["--batch", "1"])
    meta = json.loads(
        (out / "avg_training_traces_3Oct-Training-24-0.1_batch1.json").read_text()
    )
    assert meta["batch"] == 1
    assert meta["n_flies"] == 2
    assert (out / "avg_training_traces_3Oct-Training-24-0.1_batch1.png").exists()


def test_main_baseline_subtract_tags_the_filename_and_sidecar(tmp_path):
    out = _run(tmp_path, _wide_frame(), extra=["--baseline-subtract"])
    meta_path = out / "avg_training_traces_3Oct-Training-24-0.1_baseline.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["baseline_subtracted"] is True
    # Constant fixture traces collapse to exactly zero once baseline-corrected.
    assert meta["per_trial"]["1"]["peak_mean"] == pytest.approx(0.0)


def test_main_records_the_mean_measured_light_onset(tmp_path):
    out = _run(tmp_path, _wide_frame())
    meta = json.loads(
        (out / "avg_training_traces_3Oct-Training-24-0.1.json").read_text()
    )
    assert meta["light_on_s"] == pytest.approx(LIGHT_ON_S)


def test_main_excludes_flagged_flies(tmp_path):
    flagged = tmp_path / "flagged.csv"
    pd.DataFrame([
        {
            "dataset": DATASET,
            "fly": "july_21_batch_2",
            "fly_number": 1,
            "FLY-State(1, 0, -1)": 0,
        }
    ]).to_csv(flagged, index=False)
    out = _run(tmp_path, _wide_frame(), extra=["--flagged-flies-csv", str(flagged)])
    meta = json.loads(
        (out / "avg_training_traces_3Oct-Training-24-0.1.json").read_text()
    )
    assert meta["n_flies"] == 2
    assert meta["flagged_flies_excluded"] == 1


def test_main_raises_when_the_dataset_has_no_training_rows(tmp_path):
    csv = tmp_path / "wide_training.csv"
    _wide_frame().to_csv(csv, index=False)
    with pytest.raises(SystemExit):
        main([
            "--wide-csv", str(csv),
            "--dataset", "Missing-Training",
            "--out-dir", str(tmp_path / "figs"),
        ])


def test_main_figure_has_one_panel_per_training_trial(tmp_path):
    import matplotlib.pyplot as plt

    from scripts.analysis.avg_training_traces_dataset import plot_training_means

    df = select_dataset_rows(_wide_frame(), DATASET)
    traces, light, odor = collect_training_traces(
        df, dir_cols=_dir_cols(df), dataset=DATASET
    )
    fig, _ = plot_training_means(
        traces,
        odor=odor,
        title="cohort",
        subtitle="n = 3 flies",
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        max_time_s=MAX_TIME_S,
        light_on_s=LIGHT_ON_S,
    )
    try:
        assert len(fig.axes) == len(TRAINING_TRIALS)
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any("Training 1" in t for t in texts)
        assert any(odor in t for t in texts)
        assert any("n = 3" in t or "n=3" in t for t in texts)
        assert fig.axes[-1].get_xlabel() == "Time (s)"
    finally:
        plt.close(fig)


def test_baseline_panels_share_one_y_limit(tmp_path):
    """Trial 1 and trial 6 must be read off the same scale.

    Autoscaling each panel independently makes a shrinking response look
    constant, which is the exact comparison this figure exists to support.
    """
    import matplotlib.pyplot as plt

    from scripts.analysis.avg_training_traces_dataset import plot_training_means

    df = _wide_frame()
    # Give trial 1 a big deflection and trial 6 a small one, so independent
    # autoscaling would produce visibly different limits.
    for trial, amp in ((1, 60.0), (6, 5.0)):
        mask = df["trial_label"].str.startswith(f"training_{trial}_3-octonol")
        for i in range(N_FRAMES):
            df.loc[mask, f"dir_val_{i}"] = 0.0 if i < BASELINE_FRAMES else amp
    df = select_dataset_rows(df, DATASET)
    traces, _, odor = collect_training_traces(
        df, dir_cols=_dir_cols(df), dataset=DATASET, baseline_frames=BASELINE_FRAMES
    )
    fig, _ = plot_training_means(
        traces,
        odor=odor,
        title="cohort",
        subtitle="n = 3 flies",
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        max_time_s=MAX_TIME_S,
        light_on_s=LIGHT_ON_S,
        baseline_subtracted=True,
    )
    try:
        limits = {ax.get_ylim() for ax in fig.axes}
        assert len(limits) == 1, f"panels disagree on y-limits: {limits}"
        low, high = limits.pop()
        assert high >= 60.0
    finally:
        plt.close(fig)


def test_raw_panels_use_the_fixed_zero_to_hundred_axis(tmp_path):
    """The raw variant must keep the per-fly figure's 0–100 axis."""
    import matplotlib.pyplot as plt

    from scripts.analysis.avg_training_traces_dataset import plot_training_means

    df = select_dataset_rows(_wide_frame(), DATASET)
    traces, _, odor = collect_training_traces(
        df, dir_cols=_dir_cols(df), dataset=DATASET
    )
    fig, _ = plot_training_means(
        traces, odor=odor, title="c", subtitle="s", fps=FPS,
        odor_on_s=ODOR_ON_S, odor_off_s=ODOR_OFF_S, max_time_s=MAX_TIME_S,
    )
    try:
        assert {ax.get_ylim() for ax in fig.axes} == {(0.0, 100.0)}
    finally:
        plt.close(fig)


def test_plot_marks_the_odor_window_and_the_light_onset(tmp_path):
    import matplotlib.pyplot as plt

    from scripts.analysis.avg_training_traces_dataset import plot_training_means

    df = select_dataset_rows(_wide_frame(), DATASET)
    traces, _, odor = collect_training_traces(
        df, dir_cols=_dir_cols(df), dataset=DATASET
    )
    fig, _ = plot_training_means(
        traces,
        odor=odor,
        title="cohort",
        subtitle="n = 3 flies",
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        max_time_s=MAX_TIME_S,
        light_on_s=LIGHT_ON_S,
    )
    try:
        ax = fig.axes[0]
        # Shaded odor window spans exactly odor_on..odor_off (data coords live
        # on the patch itself; the path is the unit rectangle).
        spans = [p for p in ax.patches if p.get_width() > 0]
        assert spans, "expected a shaded odor window"
        assert spans[0].get_x() == pytest.approx(ODOR_ON_S)
        assert spans[0].get_x() + spans[0].get_width() == pytest.approx(ODOR_OFF_S)
        light_lines = [ln for ln in ax.lines if ln.get_linestyle() == "-."]
        assert light_lines, "expected a light-onset marker"
        assert light_lines[0].get_xdata()[0] == pytest.approx(LIGHT_ON_S)
    finally:
        plt.close(fig)
