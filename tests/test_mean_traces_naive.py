"""A naive arm on the dataset_mean_comparisons figures.

The trained-vs-control mean traces show what conditioning did, but not what the
odor looked like to a fly that had never met it. The random panels are exactly
that: flies that saw the odorant without ever being conditioned to it.

Three things this pins:

* **Matching is per ODOR, by concentration**, not per cohort. A cohort's panel
  spans several doses (Hexanol 0.1%, Citral 1%), and each dose has its own naive
  panel. An odor whose label carries no concentration has no naive baseline and
  is drawn trained-vs-control as before rather than matched to a nearby dose.
* **The naive mean pools TRIALS, not fly means.** A naive fly meets each odor
  twice and the trained/control arms are split by presentation, so there is no
  presentation to align on; pooling trials also avoids the mean-of-means
  reweighting the project rules out.
* **Its own subfolder**, so the existing two-arm figures keep their filenames
  and nothing downstream that globs them changes meaning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis import dataset_mean_traces_tvc as tvc  # noqa: E402
from scripts.analysis.dataset_means_specific_flies import (  # noqa: E402
    _plot_training_vs_control_for_odor,
)


# ── which naive panel does an odor match? ─────────────────────────────────


@pytest.mark.parametrize("label,expected", [
    ("Hexanol (0.1%)", "RandomPanel-24-0.1"),
    ("Citral (1%)", "RandomPanel-24-1"),
    ("3-Octanol (0.1%) 2", "RandomPanel-24-0.1"),
])
def test_odor_matches_its_concentration_panel(label, expected):
    assert tvc.naive_dataset_for_odor(label) == expected


def test_odor_without_a_concentration_has_no_naive_panel():
    """A bare label carries no dose. Matching it to a neighbouring one would
    silently compare against the wrong concentration."""
    assert tvc.naive_dataset_for_odor("Hexanol") is None
    assert tvc.naive_dataset_for_odor("") is None


def test_unmatched_concentration_has_no_naive_panel():
    """0.01% was never run as a random panel."""
    assert tvc.naive_dataset_for_odor("Hexanol (0.01%)") is None


# ── pooling ───────────────────────────────────────────────────────────────


def test_naive_mean_pools_trials_not_fly_means():
    """Two flies, unbalanced: 3 trials at 0 and 1 trial at 4. Pooling trials
    gives 1.0; a mean of fly means would give 2.0 by promoting the single
    trial to equal weight."""
    trials = np.array([[0.0], [0.0], [0.0], [4.0]])
    mean, sem, n = tvc.naive_mean_sem(trials)
    assert n == 4
    assert float(mean[0]) == pytest.approx(1.0)


def test_naive_sem_is_over_trials():
    trials = np.array([[0.0], [2.0], [4.0], [6.0]])
    mean, sem, n = tvc.naive_mean_sem(trials)
    expected = float(np.std([0, 2, 4, 6], ddof=1) / np.sqrt(4))
    assert float(sem[0]) == pytest.approx(expected)


def test_naive_mean_ignores_all_nan_columns():
    trials = np.array([[1.0, np.nan], [3.0, np.nan]])
    mean, sem, n = tvc.naive_mean_sem(trials)
    assert float(mean[0]) == pytest.approx(2.0)
    assert np.isnan(mean[1])


def test_empty_naive_returns_no_trace():
    mean, sem, n = tvc.naive_mean_sem(np.empty((0, 5)))
    assert n == 0
    assert mean.size == 0


# ── the drawing ───────────────────────────────────────────────────────────


def _per_fly(n=3, frames=40):
    return {f"f{i}": np.full(frames, float(i)) for i in range(n)}


def test_naive_trace_is_drawn_in_black():
    """Full black, distinct from the odor-coloured trained line and the grey
    control, so three arms stay tellable apart in every odor's palette."""
    fig = _plot_training_vs_control_for_odor(
        odor="Hexanol (0.1%)", train_per_fly=_per_fly(), ctrl_per_fly=_per_fly(),
        fps=40.0, odor_on_s=30.0, odor_off_s=60.0, ylim=None,
        naive_trials=np.full((5, 40), 2.0),
    )
    try:
        colors = [tuple(np.round(l.get_color(), 3)) if not isinstance(l.get_color(), str)
                  else l.get_color() for l in fig.axes[0].lines]
        assert any(c in ("black", "#000000", (0.0, 0.0, 0.0)) for c in colors)
    finally:
        plt.close(fig)


def test_naive_legend_counts_trials_not_flies():
    """The other two arms say n=<flies>; saying n=<flies> here would be a lie
    about what was averaged."""
    fig = _plot_training_vs_control_for_odor(
        odor="Hexanol (0.1%)", train_per_fly=_per_fly(), ctrl_per_fly=_per_fly(),
        fps=40.0, odor_on_s=30.0, odor_off_s=60.0, ylim=None,
        naive_trials=np.full((7, 40), 2.0),
    )
    try:
        labels = [l.get_label() for l in fig.axes[0].lines]
        assert any("7" in str(l) and "trial" in str(l).lower() for l in labels)
    finally:
        plt.close(fig)


def test_figure_without_naive_is_unchanged():
    """The two-arm figure must keep exactly the lines it had."""
    kw = dict(odor="Hexanol (0.1%)", train_per_fly=_per_fly(),
              ctrl_per_fly=_per_fly(), fps=40.0, odor_on_s=30.0,
              odor_off_s=60.0, ylim=None)
    a = _plot_training_vs_control_for_odor(**kw)
    b = _plot_training_vs_control_for_odor(**kw, naive_trials=None)
    try:
        assert len(a.axes[0].lines) == len(b.axes[0].lines)
    finally:
        plt.close(a)
        plt.close(b)


# ── where it lands ────────────────────────────────────────────────────────


def test_subfolder_name_is_self_explanatory():
    assert tvc.NAIVE_SUBDIR == "Trained_vs_Control_vs_Naive"


def test_cli_offers_the_naive_flag():
    args = tvc.build_parser([
        "--wide-csv", "w.parquet",
        "--train-dataset", "EB-Training-24-1",
        "--control-dataset", "EB-Control-24-1",
        "--out-dir", "/tmp/out",
        "--with-naive",
    ])
    assert args.with_naive is True


def test_naive_is_off_by_default():
    args = tvc.build_parser([
        "--wide-csv", "w.parquet",
        "--train-dataset", "EB-Training-24-1",
        "--control-dataset", "EB-Control-24-1",
        "--out-dir", "/tmp/out",
    ])
    assert args.with_naive is False


# ── pipeline wiring ───────────────────────────────────────────────────────


def test_pipeline_asks_for_the_naive_arm():
    import yaml
    import scripts.pipeline.run_workflows as rw
    from fbpipe.config import load_settings

    path = Path(rw.REPO_ROOT) / "config" / "config_new.yaml"
    if not path.is_file():
        pytest.skip("config_new.yaml not present")
    # Command SHAPE test: thaw everything so the shipped config's freeze state
    # (all non-sensitivity datasets are frozen) does not empty the command list.
    # Freeze behaviour is covered by tests/test_figure_freeze_adherence.py.
    settings = load_settings(path)
    settings._thaw_all = True
    cmds = rw._dataset_mean_traces_commands(
        (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("analysis"),
        settings, python_exec=sys.executable, config_path=path,
    )
    # The same script now also serves the pre-test vs post-training phase
    # comparison, which has no naive arm by construction: the fly's own
    # pre-test IS the baseline, so --with-naive would be meaningless there.
    # Scope this to the trained-vs-control commands it was written for.
    tvc_cmds = [c for c in cmds
                if any(str(p).endswith("dataset_mean_traces_tvc.py") for p in c)
                and "--pretest-wide-csv" not in c]
    assert tvc_cmds, "no trained-vs-control mean trace commands built"
    for cmd in tvc_cmds:
        assert "--with-naive" in cmd


def test_the_phase_comparison_does_not_ask_for_a_naive_arm():
    """Its baseline is the fly's own pre-test; a naive cohort would be a third
    arm answering a question the figure is not asking."""
    import yaml
    import scripts.pipeline.run_workflows as rw
    from fbpipe.config import load_settings

    path = Path(rw.REPO_ROOT) / "config" / "config_new.yaml"
    if not path.is_file():
        pytest.skip("config_new.yaml not present")
    settings = load_settings(path)
    settings._thaw_all = True
    cmds = rw._dataset_mean_traces_commands(
        (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("analysis"),
        settings, python_exec=sys.executable, config_path=path,
    )
    phase_cmds = [c for c in cmds if "--pretest-wide-csv" in c]
    assert phase_cmds, "no phase-comparison commands built"
    for cmd in phase_cmds:
        assert "--with-naive" not in cmd
        assert "--control-dataset" not in cmd


def test_three_arm_title_names_all_three_arms():
    """A three-arm figure captioned "Trained vs Control" misreports itself."""
    fig = _plot_training_vs_control_for_odor(
        odor="Hexanol (0.1%)", train_per_fly=_per_fly(), ctrl_per_fly=_per_fly(),
        fps=40.0, odor_on_s=30.0, odor_off_s=60.0, ylim=None,
        naive_trials=np.full((5, 40), 2.0),
    )
    try:
        assert "Naive" in fig.axes[0].get_title()
    finally:
        plt.close(fig)


def test_two_arm_title_is_unchanged():
    fig = _plot_training_vs_control_for_odor(
        odor="Hexanol (0.1%)", train_per_fly=_per_fly(), ctrl_per_fly=_per_fly(),
        fps=40.0, odor_on_s=30.0, odor_off_s=60.0, ylim=None,
    )
    try:
        assert fig.axes[0].get_title() == "Hexanol (0.1%) - Trained vs Control"
    finally:
        plt.close(fig)
