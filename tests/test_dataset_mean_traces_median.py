"""Opt-in median twins for the two-arm mean-trace figures.

``dataset_mean_traces_tvc.py --with-median`` writes, beside every
``<odor>_<tag>.png``:

* ``<odor>_<tag>_median.png`` -- the same figure with the per-frame MEDIAN as
  the line and the interquartile range (25th-75th percentile) as the band;
* ``<odor>_<tag>_mean_vs_median.png`` -- mean (solid) and median (dashed) of
  each arm on one axes, no bands.

The median is taken over exactly the traces the mean is: one baseline-corrected
trace per fly per presentation (``collect_presentations``), so a fly is one
unit in both statistics. The pipeline runs this script for every trained/control
pair WITHOUT the flag, so the default output must not change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis import dataset_mean_traces_tvc as tvc  # noqa: E402
from scripts.analysis.dataset_means_specific_flies import (  # noqa: E402
    _mean_sem_trace,
)

N_FRAMES = 120
FPS = 40.0
ODOR_ON_S = 0.5
BASELINE_FRAMES = int(round(ODOR_ON_S * FPS))
MAX_FRAMES = int(tvc.MAX_TIME_S * FPS)


@pytest.fixture(autouse=True)
def _protocol():
    saved_protocol = ev.get_protocol()
    saved_remap = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    ev.set_protocol("v2")
    ev.set_dataset_odor_remap({})
    try:
        yield
    finally:
        ev.set_protocol(saved_protocol)
        ev.set_dataset_odor_remap(saved_remap)
        plt.close("all")


# Per-fly response amplitude. One outlier fly per arm drags the mean far above
# the median, so a median that is secretly a mean fails loudly.
AMPLITUDE = {
    ("july_20_batch_1", 1): 1.0,
    ("july_20_batch_1", 2): 2.0,
    ("july_20_batch_2", 1): 3.0,
    ("july_20_batch_2", 2): 4.0,
    ("july_20_batch_3", 1): 100.0,   # the outlier
}


def _wide(datasets, labels, *, trial_type="testing"):
    rows = []
    for dataset in datasets:
        for (fly, fly_number), amp in AMPLITUDE.items():
            for i, label in enumerate(labels):
                row = {
                    "dataset": dataset,
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial_type": trial_type,
                    "trial_label": label,
                }
                row.update({
                    f"dir_val_{j}": amp * (i + 1) * max(0, j - BASELINE_FRAMES)
                    for j in range(N_FRAMES)
                })
                rows.append(row)
    return pd.DataFrame(rows)


LABELS = ["testing_1_hexanol", "testing_2_citral"]
ODOR_STEMS = ("Hexanol", "Citral")


def _run_tvc(tmp_path, *extra):
    wide = tmp_path / "wide.parquet"
    _wide(("A-Training-1", "A-Control-1"), LABELS).to_parquet(wide)
    out_dir = tmp_path / "figs"
    out_dir.mkdir(exist_ok=True)
    tvc.main([
        "--wide-csv", str(wide),
        "--train-dataset", "A-Training-1",
        "--control-dataset", "A-Control-1",
        "--out-dir", str(out_dir),
        "--fps", str(FPS), "--odor-on-s", str(ODOR_ON_S), "--odor-off-s", "1.0",
        *extra,
    ])
    return out_dir


def _per_fly(lengths_and_amps):
    """Synthetic per-fly dict with ragged lengths and a NaN gap."""
    out = {}
    for k, (n, amp) in enumerate(lengths_and_amps):
        t = amp * np.linspace(0.0, 1.0, n)
        if k == 1:
            t[5:9] = np.nan
        out[f"fly{k}"] = t
    return out


def _stack(per_fly, max_frames):
    width = max(len(v) for v in per_fly.values())
    arr = np.full((len(per_fly), width), np.nan)
    for i, v in enumerate(per_fly.values()):
        arr[i, : len(v)] = v
    return arr[:, :max_frames]


def _line(ax, label_part):
    hits = [ln for ln in ax.get_lines() if label_part in str(ln.get_label())]
    assert len(hits) == 1, [ln.get_label() for ln in ax.get_lines()]
    return hits[0]


# ---------------------------------------------------------------------------
# The CLI and the default output
# ---------------------------------------------------------------------------


def test_the_flag_is_off_by_default():
    a = tvc.build_parser(["--wide-csv", "/x/w.parquet", "--train-dataset", "A",
                          "--control-dataset", "B", "--out-dir", "/x/out"])
    assert a.with_median is False


def test_without_the_flag_no_median_figures_are_written(tmp_path):
    out_dir = _run_tvc(tmp_path)
    for stem in ODOR_STEMS:
        assert (out_dir / f"{stem}_training_vs_control.png").exists()
    assert not list(out_dir.glob("*_median.png"))
    assert not list(out_dir.glob("*_mean_vs_median.png"))


def test_with_the_flag_both_siblings_exist_for_every_odor(tmp_path):
    out_dir = _run_tvc(tmp_path, "--with-median")
    for stem in ODOR_STEMS:
        assert (out_dir / f"{stem}_training_vs_control.png").exists()
        assert (out_dir / f"{stem}_training_vs_control_median.png").exists()
        assert (out_dir / f"{stem}_training_vs_control_mean_vs_median.png").exists()


def test_a_rerun_with_the_flag_keeps_the_twins(tmp_path):
    _run_tvc(tmp_path, "--with-median")
    out_dir = _run_tvc(tmp_path, "--with-median")
    assert len(list(out_dir.glob("*_training_vs_control_median.png"))) == 2
    assert len(list(out_dir.glob("*_training_vs_control_mean_vs_median.png"))) == 2


def test_a_run_without_the_flag_prunes_earlier_twins(tmp_path):
    """Twins left from a --with-median run would otherwise sit beside mean
    figures they no longer describe."""
    _run_tvc(tmp_path, "--with-median")
    out_dir = _run_tvc(tmp_path)
    assert not list(out_dir.glob("*_median.png"))
    for stem in ODOR_STEMS:
        assert (out_dir / f"{stem}_training_vs_control.png").exists()


def test_a_stale_odor_twin_is_pruned_with_the_flag_on(tmp_path):
    out_dir = tmp_path / "figs"
    out_dir.mkdir()
    stale = [
        out_dir / "Linalool_training_vs_control_median.png",
        out_dir / "Linalool_training_vs_control_mean_vs_median.png",
    ]
    other = out_dir / "Linalool_some_other_driver_median.png"
    for p in (*stale, other):
        p.write_bytes(b"x")
    _run_tvc(tmp_path, "--with-median")
    assert not any(p.exists() for p in stale)
    assert other.exists()


def test_the_sidecar_is_unchanged_without_the_flag(tmp_path):
    import json

    out_dir = _run_tvc(tmp_path)
    side = json.loads((out_dir / "training_vs_control.json").read_text())
    assert "shared_median_ylim" not in side
    assert "shared_mean_vs_median_ylim" not in side


def test_phase_mode_writes_twins_under_its_own_tag(tmp_path):
    """The Sensitivity results folder runs the pre-test vs post-training mode."""
    testing = tmp_path / "testing.parquet"
    pretest = tmp_path / "pretest.parquet"
    _wide(("IAA-Sensitivity-1-1",), LABELS).to_parquet(testing)
    _wide(("IAA-Sensitivity-1-1",), LABELS, trial_type="pretest").to_parquet(pretest)
    out_dir = tmp_path / "figs"
    tvc.main([
        "--wide-csv", str(testing), "--pretest-wide-csv", str(pretest),
        "--train-dataset", "IAA-Sensitivity-1-1", "--out-dir", str(out_dir),
        "--fps", str(FPS), "--odor-on-s", str(ODOR_ON_S), "--odor-off-s", "1.0",
        "--with-median",
    ])
    for stem in ODOR_STEMS:
        for suffix in ("", "_median", "_mean_vs_median"):
            assert (out_dir / f"{stem}_{tvc.PHASE_TAG}{suffix}.png").exists()


def test_rig_split_writes_twins_too(tmp_path):
    wide = tmp_path / "wide.parquet"
    df = _wide(("A-Training-1", "A-Control-1"), LABELS)
    df["fly"] = df["fly"] + np.where(df["fly_number"] == 1, "_rig_1", "_rig_3")
    df.to_parquet(wide)
    out_dir = tmp_path / "figs"
    tvc.main([
        "--wide-csv", str(wide), "--train-dataset", "A-Training-1",
        "--control-dataset", "A-Control-1", "--out-dir", str(out_dir),
        "--fps", str(FPS), "--odor-on-s", str(ODOR_ON_S), "--odor-off-s", "1.0",
        "--rig-split", "1:3", "--with-median",
    ])
    tag = "train_rig_1_vs_rig_3"
    for stem in ODOR_STEMS:
        for suffix in ("", "_median", "_mean_vs_median"):
            assert (out_dir / f"{stem}_{tag}{suffix}.png").exists()


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


def test_the_median_helper_is_nanmedian_over_the_same_traces():
    per_fly = _per_fly([(30, 1.0), (40, 2.0), (25, 5.0), (40, 50.0)])
    med, q25, q75 = tvc.median_iqr_trace(per_fly, MAX_FRAMES)
    stacked = _stack(per_fly, MAX_FRAMES)
    np.testing.assert_allclose(med, np.nanmedian(stacked, axis=0), equal_nan=True)
    np.testing.assert_allclose(
        q25, np.nanpercentile(stacked, 25, axis=0), equal_nan=True
    )
    np.testing.assert_allclose(
        q75, np.nanpercentile(stacked, 75, axis=0), equal_nan=True
    )
    # Same length and alignment as the mean the existing figure draws.
    mean, _ = _mean_sem_trace(per_fly, MAX_FRAMES)
    assert med.shape == mean.shape


def test_the_median_helper_clips_to_max_frames():
    per_fly = {"a": np.arange(50.0), "b": np.arange(50.0) * 2}
    med, q25, q75 = tvc.median_iqr_trace(per_fly, 20)
    assert med.shape == q25.shape == q75.shape == (20,)


def test_one_outlier_fly_separates_median_from_mean():
    per_fly = {f"f{i}": np.full(10, v) for i, v in enumerate([1.0, 2.0, 3.0, 100.0])}
    med, q25, q75 = tvc.median_iqr_trace(per_fly, 10)
    mean, _ = _mean_sem_trace(per_fly, 10)
    np.testing.assert_allclose(med, 2.5)
    np.testing.assert_allclose(mean, 26.5)
    np.testing.assert_allclose(q25, np.percentile([1, 2, 3, 100], 25))
    np.testing.assert_allclose(q75, np.percentile([1, 2, 3, 100], 75))


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------


def _skewed_arms():
    t = np.linspace(0.0, 1.0, 60)
    train = {f"t{i}": a * t for i, a in enumerate([1.0, 2.0, 3.0, 4.0, 100.0])}
    ctrl = {f"c{i}": a * t for i, a in enumerate([0.5, 1.0, 1.5])}
    return train, ctrl


PLOT_KW = dict(fps=FPS, odor_on_s=ODOR_ON_S, odor_off_s=1.0,
               color_key="Hexanol", train_label="Trained", ctrl_label="Control")


def test_the_median_figure_draws_the_median_not_the_mean():
    train, ctrl = _skewed_arms()
    fig = tvc._plot_tvc_median_for_odor(
        odor="Hexanol", train_per_fly=train, ctrl_per_fly=ctrl, ylim=None,
        **PLOT_KW,
    )
    ax = fig.axes[0]
    tline = _line(ax, "Trained")
    med, _, _ = tvc.median_iqr_trace(train, MAX_FRAMES)
    mean, _ = _mean_sem_trace(train, MAX_FRAMES)
    np.testing.assert_allclose(tline.get_ydata(), med, equal_nan=True)
    assert not np.allclose(np.nan_to_num(tline.get_ydata()), np.nan_to_num(mean))


def test_the_median_figure_matches_the_mean_figure_layout():
    train, ctrl = _skewed_arms()
    from scripts.analysis.dataset_means_specific_flies import (
        _plot_training_vs_control_for_odor,
    )
    mean_fig = _plot_training_vs_control_for_odor(
        odor="Hexanol", train_per_fly=train, ctrl_per_fly=ctrl, ylim=(-10, 30),
        **PLOT_KW,
    )
    med_fig = tvc._plot_tvc_median_for_odor(
        odor="Hexanol", train_per_fly=train, ctrl_per_fly=ctrl, ylim=(-10, 30),
        **PLOT_KW,
    )
    m_ax, d_ax = mean_fig.axes[0], med_fig.axes[0]
    assert tuple(mean_fig.get_size_inches()) == tuple(med_fig.get_size_inches())
    assert m_ax.get_xlim() == d_ax.get_xlim()
    assert m_ax.get_ylim() == d_ax.get_ylim()
    assert m_ax.get_xlabel() == d_ax.get_xlabel()
    # Same arm colours.
    assert _line(m_ax, "Trained").get_color() == _line(d_ax, "Trained").get_color()
    assert _line(m_ax, "Control").get_color() == _line(d_ax, "Control").get_color()
    # Same odor window shading (axvspan) and the two odor lines.
    assert len(m_ax.patches) == len(d_ax.patches)


def test_the_median_figure_says_median_and_iqr():
    train, ctrl = _skewed_arms()
    fig = tvc._plot_tvc_median_for_odor(
        odor="Hexanol", train_per_fly=train, ctrl_per_fly=ctrl, ylim=None,
        **PLOT_KW,
    )
    ax = fig.axes[0]
    assert "median" in ax.get_ylabel().lower()
    assert "mean" not in ax.get_ylabel().lower()
    assert "median" in ax.get_title().lower()
    assert "iqr" in ax.get_title().lower()
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("n=5" in t for t in legend)      # n unchanged
    assert any("n=3" in t for t in legend)
    assert any("iqr" in t.lower() for t in legend)


def test_the_median_figure_band_is_the_iqr_and_is_not_clipped():
    train, ctrl = _skewed_arms()
    fig = tvc._plot_tvc_median_for_odor(
        odor="Hexanol", train_per_fly=train, ctrl_per_fly=ctrl, ylim=None,
        **PLOT_KW,
    )
    ax = fig.axes[0]
    _, q25, q75 = tvc.median_iqr_trace(train, MAX_FRAMES)
    bands = [c for c in ax.collections if c.get_paths()]
    ys = np.concatenate([p.vertices[:, 1] for c in bands for p in c.get_paths()])
    assert np.nanmax(ys) == pytest.approx(np.nanmax(q75))
    lo, hi = tvc.median_ylim([train, ctrl], fps=FPS)
    assert lo <= np.nanmin(q25) and hi >= np.nanmax(q75)


def test_the_comparison_figure_has_solid_means_and_dashed_medians():
    train, ctrl = _skewed_arms()
    fig = tvc._plot_tvc_mean_vs_median_for_odor(
        odor="Hexanol", train_per_fly=train, ctrl_per_fly=ctrl, ylim=None,
        **PLOT_KW,
    )
    # one panel per arm (control left, trained right) so the dashed median is never hidden
    # under the other arm's lines; the median is black, the mean the arm's colour
    assert len(fig.axes) == 2
    c_ax, ax = fig.axes
    assert ax.get_shared_y_axes().joined(c_ax, ax)
    assert not ax.collections and not c_ax.collections      # no bands
    t_mean = _line(ax, "Trained mean")
    t_med = _line(ax, "Trained median")
    c_mean = _line(c_ax, "Control mean")
    c_med = _line(c_ax, "Control median")
    for ln in (t_mean, c_mean):
        assert ln.get_linestyle() == "-"
    for ln in (t_med, c_med):
        assert ln.get_linestyle() == "--"
    assert t_med.get_color() == c_med.get_color() == "black"
    assert t_mean.get_color() != c_mean.get_color()
    med, _, _ = tvc.median_iqr_trace(train, MAX_FRAMES)
    mean, _ = _mean_sem_trace(train, MAX_FRAMES)
    np.testing.assert_allclose(t_med.get_ydata(), med, equal_nan=True)
    np.testing.assert_allclose(t_mean.get_ydata(), mean, equal_nan=True)
    text = " ".join(
        [fig.get_suptitle(), ax.get_title()]
        + [t.get_text() for t in ax.get_legend().get_texts()]
    ).lower()
    assert "solid" in text and "dashed" in text


def test_the_comparison_ylim_covers_both_lines():
    train, ctrl = _skewed_arms()
    lo, hi = tvc.mean_vs_median_ylim([train, ctrl], fps=FPS)
    for per_fly in (train, ctrl):
        mean, _ = _mean_sem_trace(per_fly, MAX_FRAMES)
        med, _, _ = tvc.median_iqr_trace(per_fly, MAX_FRAMES)
        for y in (mean, med):
            assert lo <= np.nanmin(y) and hi >= np.nanmax(y)
