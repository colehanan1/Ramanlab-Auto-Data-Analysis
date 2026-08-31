#!/usr/bin/env python3
"""Tests for the light-only PER trace pipeline stage.

Covers:
  1. detect_light_datasets() finds only datasets that actually have light
     trials (trial_light_on_s populated) and returns them sorted.
  2. generate() writes one figure per fly into per-dataset subfolders
     (sorted by dataset) plus a summary CSV, and excludes odor-only datasets.
  3. run_workflows._run_light_only_traces() parses a config block and drives
     generate() end-to-end.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO), str(_REPO / "scripts" / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import light_trial_traces as lt  # noqa: E402

NCOLS = 1300  # dir_val columns (~32 s @ 40 fps — covers light on @ 30 s)
FPS = 40.0


def _row(dataset, fly, fly_number, trial_label, light_on):
    """Build one wide-CSV row dict with a synthetic dir_val trace."""
    rng = np.random.default_rng(abs(hash((dataset, fly, fly_number, trial_label))) % (2**32))
    trace = rng.uniform(2, 8, NCOLS)  # low baseline
    if light_on is not None:
        on_idx = int(light_on * FPS)
        trace[on_idx:] = rng.uniform(60, 95, NCOLS - on_idx)  # response after light
    row = {
        "dataset": dataset,
        "fly": fly,
        "fly_number": fly_number,
        "trial_label": trial_label,
        "trial_light_on_s": light_on if light_on is not None else np.nan,
        "trace_len": NCOLS,
        "fps": FPS,
    }
    for i, v in enumerate(trace):
        row[f"dir_val_{i}"] = v
    return row


def _synthetic_csv(path: Path) -> Path:
    rows = []
    # Two light datasets (each: 2 flies, odor trials 1-2 + light trials 15-16)
    for ds in ("DS-Light-A", "DS-Light-B"):
        for fly in (f"{ds}_sess1",):
            for fn in (1, 2):
                rows.append(_row(ds, fly, fn, "training_1_acv", None))
                rows.append(_row(ds, fly, fn, "training_2_hexanol", None))
                rows.append(_row(ds, fly, fn, "training_15", 30.0))
                rows.append(_row(ds, fly, fn, "training_16", 30.0))
    # One odor-only dataset (NO light trials) — must be excluded
    for fn in (1, 2):
        rows.append(_row("DS-OdorOnly", "odor_sess1", fn, "training_1_acv", None))
        rows.append(_row("DS-OdorOnly", "odor_sess1", fn, "training_2_hexanol", None))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_detect_light_datasets(tmp_path):
    csv = _synthetic_csv(tmp_path / "wide.csv")
    df = pd.read_csv(csv)
    got = lt.detect_light_datasets(df)
    assert got == ["DS-Light-A", "DS-Light-B"], got
    print("PASS: test_detect_light_datasets")


def test_generate_sorts_by_dataset(tmp_path):
    csv = _synthetic_csv(tmp_path / "wide.csv")
    out = tmp_path / "Light-Only"
    result = lt.generate(csv, out, datasets=None)  # auto-detect

    # Per-dataset subfolders exist; odor-only dataset excluded.
    assert (out / "DS-Light-A").is_dir()
    assert (out / "DS-Light-B").is_dir()
    assert not (out / "DS-OdorOnly").exists()

    # One figure per fly folder (1 folder per light dataset here).
    assert (out / "DS-Light-A" / "DS-Light-A_sess1.png").is_file()
    assert (out / "DS-Light-B" / "DS-Light-B_sess1.png").is_file()

    # Return contract.
    assert result["n_figures"] == 2
    assert set(result["datasets"]) == {"DS-Light-A", "DS-Light-B"}
    assert result["per_dataset"] == {"DS-Light-A": 1, "DS-Light-B": 1}

    # Summary CSV: only light trials (15,16) for the 2 light datasets × 2 flies.
    summ = pd.read_csv(out / "light_trial_summary.csv")
    assert set(summ["dataset"].unique()) == {"DS-Light-A", "DS-Light-B"}
    assert set(summ["trial"].unique()) == {15, 16}
    assert len(summ) == 2 * 2 * 2  # 2 ds × 2 flies × 2 light trials
    # Response is captured: peak after light exceeds pre-light peak.
    assert (summ["peak_during_after_light"] > summ["peak_pre_light"]).all()
    print("PASS: test_generate_sorts_by_dataset")


def test_generate_explicit_dataset_filter(tmp_path):
    csv = _synthetic_csv(tmp_path / "wide.csv")
    out = tmp_path / "Light-Only"
    result = lt.generate(csv, out, datasets=["DS-Light-A"])
    assert result["datasets"] == ["DS-Light-A"]
    assert (out / "DS-Light-A").is_dir()
    assert not (out / "DS-Light-B").exists()
    print("PASS: test_generate_explicit_dataset_filter")


def test_pipeline_stage_wiring(tmp_path):
    """run_workflows._run_light_only_traces parses cfg and produces figures."""
    from scripts.pipeline.run_workflows import _run_light_only_traces

    csv = _synthetic_csv(tmp_path / "wide.csv")
    out = tmp_path / "Light-Only"
    cfg = {"input_csv": str(csv), "out_dir": str(out)}
    _run_light_only_traces(cfg)
    assert (out / "DS-Light-A" / "DS-Light-A_sess1.png").is_file()
    assert (out / "light_trial_summary.csv").is_file()

    # Disabled block is a no-op.
    out2 = tmp_path / "Light-Only-Disabled"
    _run_light_only_traces({"input_csv": str(csv), "out_dir": str(out2), "enabled": False})
    assert not out2.exists()
    print("PASS: test_pipeline_stage_wiring")


# --- freeze awareness -------------------------------------------------


class _Ov:
    def __init__(self, data=False, figures=False):
        self.freeze_data = data
        self.freeze_figures = figures


class _Settings:
    """Minimal stand-in for fbpipe Settings: only what freeze_flags reads."""

    def __init__(self, overrides, thawed=(), thaw_all=False):
        self.dataset_overrides = overrides
        self._thawed = thawed
        self._thaw_all = thaw_all


def test_generate_skips_frozen_datasets(tmp_path):
    csv = _synthetic_csv(tmp_path / "wide.csv")
    out = tmp_path / "Light-Only"
    result = lt.generate(csv, out, datasets=None, skip_datasets=["DS-Light-B"])
    assert result["datasets"] == ["DS-Light-A"]
    assert result["skipped_datasets"] == ["DS-Light-B"]
    assert not (out / "DS-Light-B").exists()
    # The summary CSV must not carry the frozen dataset either.
    summ = pd.read_csv(out / "light_trial_summary.csv")
    assert set(summ["dataset"].unique()) == {"DS-Light-A"}
    print("PASS: test_generate_skips_frozen_datasets")


def test_pipeline_stage_skips_figure_frozen_dataset(tmp_path):
    from scripts.pipeline.run_workflows import _run_light_only_traces

    csv = _synthetic_csv(tmp_path / "wide.csv")
    out = tmp_path / "Light-Only"
    settings = _Settings({"DS-Light-B": _Ov(data=True, figures=True)})
    _run_light_only_traces({"input_csv": str(csv), "out_dir": str(out)}, settings=settings)
    assert (out / "DS-Light-A" / "DS-Light-A_sess1.png").is_file()
    assert not (out / "DS-Light-B").exists()
    print("PASS: test_pipeline_stage_skips_figure_frozen_dataset")


def test_pipeline_stage_data_freeze_alone_still_renders(tmp_path):
    """freeze.data without freeze.figures must NOT suppress the figures."""
    from scripts.pipeline.run_workflows import _run_light_only_traces

    csv = _synthetic_csv(tmp_path / "wide.csv")
    out = tmp_path / "Light-Only"
    settings = _Settings({"DS-Light-B": _Ov(data=True, figures=False)})
    _run_light_only_traces({"input_csv": str(csv), "out_dir": str(out)}, settings=settings)
    assert (out / "DS-Light-B" / "DS-Light-B_sess1.png").is_file()
    print("PASS: test_pipeline_stage_data_freeze_alone_still_renders")


def test_pipeline_stage_thaw_overrides_freeze(tmp_path):
    from scripts.pipeline.run_workflows import _run_light_only_traces

    csv = _synthetic_csv(tmp_path / "wide.csv")
    out = tmp_path / "Light-Only"
    settings = _Settings({"DS-Light-B": _Ov(data=True, figures=True)}, thawed=("DS-Light-B",))
    _run_light_only_traces({"input_csv": str(csv), "out_dir": str(out)}, settings=settings)
    assert (out / "DS-Light-B" / "DS-Light-B_sess1.png").is_file()
    print("PASS: test_pipeline_stage_thaw_overrides_freeze")


def test_explicit_allow_list_still_honors_freeze(tmp_path):
    """A config `datasets:` allow-list does not override a figure freeze."""
    from scripts.pipeline.run_workflows import _run_light_only_traces

    csv = _synthetic_csv(tmp_path / "wide.csv")
    out = tmp_path / "Light-Only"
    settings = _Settings({"DS-Light-B": _Ov(data=True, figures=True)})
    _run_light_only_traces(
        {"input_csv": str(csv), "out_dir": str(out),
         "datasets": ["DS-Light-A", "DS-Light-B"]},
        settings=settings,
    )
    assert not (out / "DS-Light-B").exists()
    print("PASS: test_explicit_allow_list_still_honors_freeze")


if __name__ == "__main__":
    import tempfile

    for fn in (
        test_detect_light_datasets,
        test_generate_sorts_by_dataset,
        test_generate_explicit_dataset_filter,
        test_pipeline_stage_wiring,
        test_generate_skips_frozen_datasets,
        test_pipeline_stage_skips_figure_frozen_dataset,
        test_pipeline_stage_data_freeze_alone_still_renders,
        test_pipeline_stage_thaw_overrides_freeze,
        test_explicit_allow_list_still_honors_freeze,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("\n=== ALL TESTS PASSED ===")
