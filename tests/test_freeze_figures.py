"""A figure is skipped only when EVERY contributing dataset is frozen."""

from pathlib import Path

import pandas as pd

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev
import scripts.analysis.score_summary as ss
from fbpipe.config import DatasetOverride


class _Cfg:
    def __init__(self, overrides):
        self.dataset_overrides = overrides


F = DatasetOverride(freeze_figures=True)
U = DatasetOverride(freeze_figures=False)


def test_single_frozen_dataset_figure_is_skipped():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"]) is True


def test_single_live_dataset_figure_is_drawn():
    assert ev.should_skip_frozen_figure(_Cfg({"A": U}), ["A"]) is False


def test_all_frozen_aggregate_is_skipped():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F, "B": F}), ["A", "B"]) is True


def test_mixed_frozen_and_live_is_DRAWN():
    """The correctness case: frozen Control beside live Training must redraw, or
    adding flies to Training silently fails to appear."""
    assert ev.should_skip_frozen_figure(_Cfg({"A": F, "B": U}), ["A", "B"]) is False


def test_unknown_dataset_counts_as_live():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A", "UNKNOWN"]) is False


def test_empty_dataset_set_is_drawn():
    """Never skip on an empty contributor set -- that is 'unknown', not 'all frozen'.
    all([]) is True, which would silently skip every such figure."""
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), []) is False


def test_freeze_data_alone_does_not_skip_figures():
    """The flags are independent: data:true + figures:false still draws."""
    cfg = _Cfg({"A": DatasetOverride(freeze_data=True, freeze_figures=False)})
    assert ev.should_skip_frozen_figure(cfg, ["A"]) is False


def test_thaw_all_draws_everything():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"], thaw_all=True) is False


def test_thaw_named_dataset_draws():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"], thawed=["A"]) is False


def test_figures_only_does_not_bypass_freeze():
    """--figures-only forces figure steps on; it must not un-freeze them."""
    cfg = _Cfg({"A": F})
    assert ev.should_skip_frozen_figure(cfg, ["A"]) is True


# ---------------------------------------------------------------------------
# Emit-site verification: real figure generation, not just should_skip_frozen_figure
# in isolation. Each check constructs the exact contributing-dataset set the
# emit site resolves and confirms a live dataset in that set still forces a
# draw (the failure mode this task exists to prevent is a figure silently NOT
# drawn when it should be).
# ---------------------------------------------------------------------------


def _write_two_dataset_wide_csv(path: Path) -> None:
    """One fly in each of two datasets; two testing trials apiece."""
    rows = []
    for dataset, fly in (("Hex-Training", "frozen_fly"), ("Hex-Control", "live_fly")):
        for trial_num in (1, 2):
            rows.append(
                {
                    "dataset": dataset,
                    "fly": fly,
                    "fly_number": "1",
                    "trial_type": "testing",
                    "trial_label": f"testing_{trial_num}",
                    "fps": 40.0,
                    "global_min": 1.0,
                    "global_max": 25.0,
                    "trimmed_global_min": 1.0,
                    "trimmed_global_max": 25.0,
                    "trace_len": 4,
                    "dir_val_0": 0.0 + trial_num,
                    "dir_val_1": 8.0,
                    "dir_val_2": 16.0,
                    "dir_val_3": 4.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_envelope_plots_skips_frozen_dataset_draws_live(tmp_path):
    """generate_envelope_plots guards per fly, on that fly's own dataset
    (envelope_visuals.py, dataset_candidates at the top of the fly loop)."""
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    _write_two_dataset_wide_csv(wide_csv)
    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))
    out_dir = tmp_path / "plots"

    cfg = ev.EnvelopePlotConfig(
        matrix_npy=matrix_dir / "envelope_matrix_float16.npy",
        codes_json=matrix_dir / "code_maps.json",
        out_dir=out_dir,
        latency_sec=0.0,
        odor_latency_s=0.0,
        trial_type="testing",
        overwrite=True,
        dataset_overrides={"Hex-Training": F, "Hex-Control": U},
    )
    ev.generate_envelope_plots(cfg)

    frozen_dir = out_dir / "Hex-Training"
    live_dir = out_dir / "Hex-Control"
    assert not (frozen_dir.is_dir() and list(frozen_dir.glob("*.png"))), (
        "frozen dataset's per-fly figure must not be drawn"
    )
    assert live_dir.is_dir() and list(live_dir.glob("*.png")), (
        "live dataset's per-fly figure must still be drawn"
    )


def test_generate_reaction_matrices_skips_frozen_dataset_draws_live(tmp_path):
    """generate_reaction_matrices guards per dataset (the loop variable is
    confusingly named "odor" but is the canonical dataset -- see :1555)."""
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    rows = []
    for dataset, fly in (("Hex-Training", "frozen_fly"), ("Hex-Control", "live_fly")):
        for trial_num in (1, 2, 3):
            rows.append(
                {
                    "dataset": dataset,
                    "fly": fly,
                    "fly_number": "1",
                    "trial_type": "testing",
                    "trial_label": f"testing_{trial_num}",
                    "fps": 40.0,
                    "global_min": 1.0,
                    "global_max": 25.0,
                    "trimmed_global_min": 1.0,
                    "trimmed_global_max": 25.0,
                    "trace_len": 4,
                    "dir_val_0": 0.0 + trial_num,
                    "dir_val_1": 8.0,
                    "dir_val_2": 16.0,
                    "dir_val_3": 4.0,
                }
            )
    pd.DataFrame(rows).to_csv(wide_csv, index=False)
    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))
    out_dir = tmp_path / "plots"

    cfg = ev.MatrixPlotConfig(
        matrix_npy=matrix_dir / "envelope_matrix_float16.npy",
        codes_json=matrix_dir / "code_maps.json",
        out_dir=out_dir,
        latency_sec=0.0,
        overwrite=True,
        dataset_overrides={"Hex-Training": F, "Hex-Control": U},
    )
    ev.generate_reaction_matrices(cfg)

    frozen_dir = out_dir / "Hex-Training"
    live_dir = out_dir / "Hex-Control"
    assert not (frozen_dir.is_dir() and list(frozen_dir.glob("*.png"))), (
        "frozen dataset's reaction-matrix figure must not be drawn"
    )
    assert live_dir.is_dir() and list(live_dir.glob("*.png")), (
        "live dataset's reaction-matrix figure must still be drawn"
    )


def _make_train_control_rows() -> list[dict[str, object]]:
    train_patterns = [[0, 5, 4, 4, 5], [1, 4, 5, 5, 4]]
    ctrl_patterns = [[0, 0, 0, 1, 0], [1, 1, 0, 0, 1]]
    trial_labels = (
        "testing_1_hexanol",
        "testing_2_ethylbutyrate",
        "testing_3_hexanol",
        "testing_4_ethylbutyrate",
        "testing_5_ethylbutyrate",
    )
    rows: list[dict[str, object]] = []
    for idx, scores in enumerate(train_patterns, start=1):
        for trial_label, score in zip(trial_labels, scores):
            rows.append(
                {
                    "dataset": "EB-Training",
                    "fly": f"train_fly_{idx}",
                    "fly_number": str(idx),
                    "trial_label": trial_label,
                    "score": score,
                    "trial_type": "testing",
                }
            )
    for idx, scores in enumerate(ctrl_patterns, start=1):
        for trial_label, score in zip(trial_labels, scores):
            rows.append(
                {
                    "dataset": "EB-Control",
                    "fly": f"ctrl_fly_{idx}",
                    "fly_number": str(idx),
                    "trial_label": trial_label,
                    "score": score,
                    "trial_type": "testing",
                }
            )
    return rows


def test_score_summary_mixed_frozen_live_pair_still_draws(tmp_path):
    """_plot_bar_charts, _plot_heatmap, and _plot_training_vs_control_bars: a
    frozen EB-Training beside a live EB-Control must still redraw every figure
    that EB-Control contributes to, and must skip EB-Training's own solo
    bar chart."""
    ss.set_protocol("legacy")
    csv_path = tmp_path / "scores.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(_make_train_control_rows()).to_csv(csv_path, index=False)

    cfg = _Cfg({"EB-Training": F, "EB-Control": U})
    ss.generate_score_summary(csv_path=csv_path, out_dir=out_dir, overwrite=True, cfg=cfg)

    assert not (out_dir / "mean_score_EB-Training.png").exists(), (
        "EB-Training is the only contributor and is frozen -- must skip"
    )
    assert (out_dir / "mean_score_EB-Control.png").exists(), (
        "EB-Control is live -- must still draw"
    )
    assert (out_dir / "mean_score_heatmap.png").exists(), (
        "heatmap pools both datasets; EB-Control is live so it must redraw"
    )
    assert (out_dir / "mean_score_train_vs_ctrl_EB-Training.png").exists(), (
        "train-vs-control pairs frozen Training with live Control -- the "
        "canonical mixed case, must still draw"
    )


def test_score_summary_all_frozen_pair_is_skipped(tmp_path):
    """The same pair, both frozen: every figure touching either dataset skips."""
    ss.set_protocol("legacy")
    csv_path = tmp_path / "scores.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(_make_train_control_rows()).to_csv(csv_path, index=False)

    cfg = _Cfg({"EB-Training": F, "EB-Control": F})
    ss.generate_score_summary(csv_path=csv_path, out_dir=out_dir, overwrite=True, cfg=cfg)

    assert not (out_dir / "mean_score_EB-Training.png").exists()
    assert not (out_dir / "mean_score_EB-Control.png").exists()
    assert not (out_dir / "mean_score_heatmap.png").exists()
    assert not (out_dir / "mean_score_train_vs_ctrl_EB-Training.png").exists()
    # Non-figure outputs are untouched by freeze.figures -- only figures skip.
    assert (out_dir / "score_summary_by_odor_testing.csv").exists()


def test_plot_score_pair_mixed_frozen_live_still_draws(tmp_path):
    """_plot_score_pair (v2-only): frozen Training beside live Control must
    still draw the combined per-fly score-matrix pair figure."""
    ss.set_protocol("v2")
    csv_path = tmp_path / "scores.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(_make_train_control_rows()).to_csv(csv_path, index=False)

    cfg = _Cfg({"EB-Training": F, "EB-Control": U})
    ss.generate_score_summary(csv_path=csv_path, out_dir=out_dir, overwrite=True, cfg=cfg)

    assert (out_dir / "mean_score_pair_EB-Training.png").exists()

    cfg_frozen = _Cfg({"EB-Training": F, "EB-Control": F})
    out_dir_2 = tmp_path / "out2"
    ss.generate_score_summary(csv_path=csv_path, out_dir=out_dir_2, overwrite=True, cfg=cfg_frozen)
    assert not (out_dir_2 / "mean_score_pair_EB-Training.png").exists()
