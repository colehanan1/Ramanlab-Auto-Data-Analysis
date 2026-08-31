"""``Dataset_means`` figures must be redrawn, and must not collide.

Two compounding bugs, same family as the conditioning-score sidecar:

1. **Frozen on existence.** ``_save_figure`` skips whenever the PNG exists, with
   no staleness check, and ``run_workflows`` passed ``no_overwrite=True``
   unconditionally — no flag or config key could turn it off. The JSON sidecar
   written beside it has no such guard, so it refreshes every run and the two
   silently disagree. On disk, all 53 PNG/sidecar pairs had a PNG older than its
   sidecar; the oldest predate the 2026-08-25 threshold recalibration entirely.

2. **Dotted dataset names collide.** ``base_path.with_suffix(".png")`` replaces
   everything after the LAST dot. Every concentration-suffixed dataset contains
   one, so::

       3Oct-Training-24-0.1_testing_odors_mean   -> 3Oct-Training-24-0.png
       3Oct-Training-24-0.1_training_odors_mean  -> 3Oct-Training-24-0.png

   The testing and training figures write to the same file — and with bug 1,
   whichever landed first wins permanently.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import scripts.analysis.dataset_means as dm


# ── filenames must survive a dot in the dataset name ──────────────────────


def test_a_dotted_dataset_keeps_its_full_name(tmp_path):
    base = tmp_path / "3Oct-Training-24-0.1_testing_odors_mean"
    assert dm.figure_path(base, ".png").name == (
        "3Oct-Training-24-0.1_testing_odors_mean.png"
    )


def test_testing_and_training_do_not_collide(tmp_path):
    testing = dm.figure_path(tmp_path / "3Oct-Training-24-0.1_testing_odors_mean", ".png")
    training = dm.figure_path(tmp_path / "3Oct-Training-24-0.1_training_odors_mean", ".png")
    assert testing != training


def test_two_concentrations_do_not_collide(tmp_path):
    a = dm.figure_path(tmp_path / "Hex-Control-24-0.1_testing_odors_mean", ".png")
    b = dm.figure_path(tmp_path / "Hex-Control-24-0.2_testing_odors_mean", ".png")
    assert a != b


def test_an_undotted_dataset_is_unchanged(tmp_path):
    base = tmp_path / "EB-Control-24-1_testing_odors_mean"
    assert dm.figure_path(base, ".png").name == "EB-Control-24-1_testing_odors_mean.png"


def test_the_sidecar_sits_beside_its_figure(tmp_path):
    """They are read as a pair; a mismatch is how the disagreement went unseen."""
    base = tmp_path / "3Oct-Training-24-0.1_testing_odors_mean"
    png = dm.figure_path(base, ".png")
    js = dm.figure_path(base, ".json")
    assert png.stem == js.stem


# ── the figure must be redrawn when its input moves ───────────────────────


def _fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def _write(tmp_path, name="fig.png"):
    p = tmp_path / name
    p.write_bytes(b"old")
    return p


def test_a_figure_older_than_its_source_is_redrawn(tmp_path):
    import os

    base = tmp_path / "ds_testing_odors_mean"
    png = dm.figure_path(base, ".png")
    png.write_bytes(b"old")
    source = tmp_path / "wide.csv"
    source.write_text("x")
    os.utime(source, (png.stat().st_mtime + 10,) * 2)

    dm._save_figure(_fig(), base, overwrite=False, source_mtime=source.stat().st_mtime)
    assert png.read_bytes() != b"old", "stale figure was kept"


def test_a_figure_newer_than_its_source_is_kept(tmp_path):
    import os

    base = tmp_path / "ds_testing_odors_mean"
    png = dm.figure_path(base, ".png")
    png.write_bytes(b"old")
    source = tmp_path / "wide.csv"
    source.write_text("x")
    os.utime(png, (source.stat().st_mtime + 10,) * 2)

    dm._save_figure(_fig(), base, overwrite=False, source_mtime=source.stat().st_mtime)
    assert png.read_bytes() == b"old", "a fresh figure was needlessly redrawn"


def test_overwrite_still_always_redraws(tmp_path):
    base = tmp_path / "ds_testing_odors_mean"
    png = dm.figure_path(base, ".png")
    png.write_bytes(b"old")
    dm._save_figure(_fig(), base, overwrite=True)
    assert png.read_bytes() != b"old"


def test_without_a_source_mtime_the_skip_is_conservative(tmp_path):
    """Unknown freshness must mean redraw, not silently keep a stale figure."""
    base = tmp_path / "ds_testing_odors_mean"
    png = dm.figure_path(base, ".png")
    png.write_bytes(b"old")
    dm._save_figure(_fig(), base, overwrite=False, source_mtime=None)
    assert png.read_bytes() != b"old"


# ── the pipeline must not force the skip ──────────────────────────────────


def test_the_pipeline_does_not_hardcode_no_overwrite():
    """It passed no_overwrite=True unconditionally, so no flag or config key
    could ever redraw these figures."""
    src = Path("scripts/pipeline/run_workflows.py").read_text()
    assert "_run_dataset_means(config_path, no_overwrite=True" not in src
    assert "_run_dataset_means_training(config_path, no_overwrite=True" not in src


# ── the cache key must actually track the data ────────────────────────────


def test_the_expected_state_reads_the_real_wide_table():
    """It read analysis.combined.wide.output_csv, which does not exist in
    config_new (the table lives under combined.combined_base.wide). The key was
    therefore a constant {None, None}: the gate could never notice new data.

    Masked today only because force.dataset_means defaults True, so the step
    always runs — set that false to save time and the step freezes forever.
    """
    import yaml
    from scripts.pipeline.run_workflows import _dataset_means_expected

    path = Path("config/config_new.yaml")
    if not path.exists():
        pytest.skip("config/ is gitignored")
    analysis = yaml.safe_load(path.read_text())["analysis"]
    key = _dataset_means_expected(analysis)
    assert key["wide_csv_mtime"] is not None, (
        "cache key is a constant; it cannot detect that the wide table changed"
    )


def test_the_expected_state_still_reads_a_legacy_combined_wide_block():
    """Older configs put the table at analysis.combined.wide."""
    from scripts.pipeline.run_workflows import _dataset_means_expected

    key = _dataset_means_expected(
        {"combined": {"wide": {"output_csv": "/nonexistent/legacy.csv"}}}
    )
    assert "wide_csv_mtime" in key


def test_the_expected_state_tolerates_a_config_with_neither():
    from scripts.pipeline.run_workflows import _dataset_means_expected

    key = _dataset_means_expected({})
    assert key["wide_csv_mtime"] is None
