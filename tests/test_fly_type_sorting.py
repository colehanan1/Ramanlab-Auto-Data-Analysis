"""Genotype (fly_type) sorting through the analysis pipeline.

Each Pi batch records a free-text ``Fly Type:`` in ``session_metadata.txt`` that
``fbpipe.utils.fly_type`` canonicalises (e.g. ``GR5a-OLD`` -> ``GR5a-Old``,
``GR5a-Retinol-Gcamp86`` -> ``GR5a-GCaMP8``). These tests pin that the canonical
genotype is threaded into the wide CSV / matrix (v2 only) and used to split RMS
and reaction-matrix figures by genotype when a dataset holds more than one
genotype. Legacy protocol must stay genotype-free (byte-for-byte v1 schema).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import envelope_combined as ec
from scripts.analysis import envelope_visuals as ev
from scripts.analysis import reaction_matrix_from_spreadsheet as rm
from scripts.analysis import reaction_matrix_training_vs_control as tvc
from scripts.analysis import score_summary as ss
from fbpipe.steps.predict_reactions import NON_REACTIVE_SPAN_PX, _augment_prediction_csv

# batch folder -> (raw "Fly Type:" string, expected canonical label)
GENOTYPE_BATCHES = {
    "june_01_batch_1": ("GR5a-OLD", "GR5a-Old"),
    "may_27_batch_2": ("GR5a-Retinol-Gcamp86", "GR5a-GCaMP8"),
}


def _build_two_genotype_dataset(root: Path, dataset_name: str = "Hex-Training") -> Path:
    """Two batches in one dataset, each a different genotype, v2-style on disk.

    Mirrors real layout: ``<Dataset>/<batch>/session_metadata.txt`` plus
    ``<batch>/angle_distance_rms_envelope/<stem>.csv`` testing trials.
    """
    dataset_root = root / dataset_name
    for batch, (raw_type, _canon) in GENOTYPE_BATCHES.items():
        batch_dir = dataset_root / batch
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "session_metadata.txt").write_text(f"Fly Type: {raw_type}\n")
        out_dir = batch_dir / "angle_distance_rms_envelope"
        out_dir.mkdir(parents=True, exist_ok=True)
        for trial_num in (1, 2, 3):
            stem = f"{batch}_testing_{trial_num}_angle_distance_rms_envelope"
            pd.DataFrame({"envelope_of_rms": np.linspace(0.0, 100.0, 64)}).to_csv(
                out_dir / f"{stem}.csv", index=False
            )
    return dataset_root


def test_build_wide_csv_v2_threads_fly_type(tmp_path):
    """v2 wide CSV gains a ``fly_type`` column carrying the canonical genotype."""
    ev.set_protocol("v2")
    dataset_root = _build_two_genotype_dataset(tmp_path)
    wide_csv = tmp_path / "wide.csv"
    ec.build_wide_csv([str(dataset_root)], str(wide_csv), measure_cols=["envelope_of_rms"])

    df = pd.read_csv(wide_csv)
    assert "fly_type" in df.columns
    cols = list(df.columns)
    # fly_type sits immediately after fly_number so code_maps column_order[2]
    # stays "fly_number" (positional contract in test_multi_fly_pipeline.py).
    assert cols.index("fly_type") == cols.index("fly_number") + 1
    by_fly = dict(zip(df["fly"].astype(str), df["fly_type"].astype(str)))
    assert by_fly["june_01_batch_1"] == "GR5a-Old"
    assert by_fly["may_27_batch_2"] == "GR5a-GCaMP8"


def test_build_wide_csv_legacy_omits_fly_type(tmp_path):
    """Legacy schema has no genotype concept; the column must not appear."""
    ev.set_protocol("legacy")
    dataset_root = _build_two_genotype_dataset(tmp_path)
    wide_csv = tmp_path / "wide.csv"
    ec.build_wide_csv([str(dataset_root)], str(wide_csv), measure_cols=["envelope_of_rms"])

    df = pd.read_csv(wide_csv)
    assert "fly_type" not in df.columns


def test_wide_to_matrix_encodes_fly_type(tmp_path):
    """fly_type survives the matrix encoding into code_maps.json (v2)."""
    ev.set_protocol("v2")
    dataset_root = _build_two_genotype_dataset(tmp_path)
    wide_csv = tmp_path / "wide.csv"
    ec.build_wide_csv([str(dataset_root)], str(wide_csv), measure_cols=["envelope_of_rms"])
    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))

    code_maps = json.loads((matrix_dir / "code_maps.json").read_text())
    assert "fly_type" in code_maps["code_maps"]
    assert code_maps["column_order"][2] == "fly_number"
    assert code_maps["column_order"][3] == "fly_type"
    assert set(code_maps["code_maps"]["fly_type"]) >= {"GR5a-Old", "GR5a-GCaMP8"}


# --- RMS envelope plots: split by genotype ---------------------------------

def _write_rms_wide_csv(path: Path, *, genotypes: dict) -> None:
    """Write a minimal testing-only wide CSV for envelope plotting.

    ``genotypes`` maps fly (batch) name -> canonical fly_type. Two testing
    trials per fly so the matrix has rows to render.
    """
    rows = []
    for fly, fly_type in genotypes.items():
        for trial_num in (1, 2):
            rows.append(
                {
                    "dataset": "Hex-Training",
                    "fly": fly,
                    "fly_number": "1",
                    "fly_type": fly_type,
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


def _envelope_cfg(matrix_dir: Path, out_dir: Path):
    return ev.EnvelopePlotConfig(
        matrix_npy=matrix_dir / "envelope_matrix_float16.npy",
        codes_json=matrix_dir / "code_maps.json",
        out_dir=out_dir,
        latency_sec=0.0,
        odor_latency_s=0.0,
        trial_type="testing",
        overwrite=True,
    )


def test_rms_plots_split_into_genotype_subfolders_when_multiple(tmp_path):
    """A dataset with two genotypes routes each fly's RMS figure into its own
    <dataset>/<genotype>/ subfolder."""
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    _write_rms_wide_csv(
        wide_csv,
        genotypes={"june_01_batch_1": "GR5a-Old", "may_27_batch_2": "GR5a-GCaMP8"},
    )
    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))
    out_dir = tmp_path / "plots"
    ev.generate_envelope_plots(_envelope_cfg(matrix_dir, out_dir))

    old_dir = out_dir / "Hex-Training" / "GR5a-Old"
    gc_dir = out_dir / "Hex-Training" / "GR5a-GCaMP8"
    assert old_dir.is_dir() and list(old_dir.glob("*.png")), "GR5a-Old subfolder missing"
    assert gc_dir.is_dir() and list(gc_dir.glob("*.png")), "GR5a-GCaMP8 subfolder missing"
    # No PNG should be dumped directly in the dataset folder when split.
    assert not list((out_dir / "Hex-Training").glob("*.png"))


def test_rms_plots_single_genotype_stays_flat(tmp_path):
    """A single-genotype dataset keeps the flat <dataset>/ layout (no split)."""
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    _write_rms_wide_csv(
        wide_csv,
        genotypes={"june_01_batch_1": "GR5a-Old", "june_02_batch_1": "GR5a-Old"},
    )
    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))
    out_dir = tmp_path / "plots"
    ev.generate_envelope_plots(_envelope_cfg(matrix_dir, out_dir))

    dataset_dir = out_dir / "Hex-Training"
    assert list(dataset_dir.glob("*.png")), "expected flat PNGs in dataset folder"
    assert not (dataset_dir / "GR5a-Old").exists()


# --- model_predictions.csv carries genotype --------------------------------

def test_augment_prediction_csv_carries_fly_type(tmp_path):
    """fly_type from the source wide CSV is merged into model_predictions.csv so
    the reaction-matrix scripts can split by genotype."""
    out_csv = tmp_path / "model_predictions.csv"
    pd.DataFrame(
        {
            "dataset": ["Hex-Training", "Hex-Training"],
            "fly": ["june_01_batch_1", "may_27_batch_2"],
            "fly_number": ["1", "1"],
            "trial_label": ["testing_1", "testing_1"],
            "prediction": [1, 0],
        }
    ).to_csv(out_csv, index=False)

    source_df = pd.DataFrame(
        {
            "dataset": ["Hex-Training", "Hex-Training"],
            "fly": ["june_01_batch_1", "may_27_batch_2"],
            "fly_number": ["1", "1"],
            "trial_label": ["testing_1", "testing_1"],
            "fly_type": ["GR5a-Old", "GR5a-GCaMP8"],
            "trial_type": ["testing", "testing"],
            "global_min": [1.0, 1.0],
            "global_max": [80.0, 80.0],
        }
    )

    _augment_prediction_csv(out_csv, source_df, threshold=NON_REACTIVE_SPAN_PX)

    merged = pd.read_csv(out_csv)
    assert "fly_type" in merged.columns
    by_fly = dict(zip(merged["fly"].astype(str), merged["fly_type"].astype(str)))
    assert by_fly["june_01_batch_1"] == "GR5a-Old"
    assert by_fly["may_27_batch_2"] == "GR5a-GCaMP8"


# --- reaction matrices (Matrix-PER-Reactions-Model): split by genotype ------

def _write_predictions_csv(path: Path, *, genotypes) -> None:
    """genotypes: list of (fly, fly_type). Plain testing_N labels (legacy odor
    schedule) with binary predictions so the matrix actually renders."""
    rows = []
    for fly, fly_type in genotypes:
        for trial_num in (1, 2, 3, 4):
            rows.append(
                {
                    "dataset": "Hex-Training",
                    "fly": fly,
                    "fly_number": "1",
                    "fly_type": fly_type,
                    "trial_label": f"testing_{trial_num}",
                    "trial_type": "testing",
                    "prediction": trial_num % 2,
                    "global_min": 1.0,
                    "global_max": 90.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _matrix_cfg(csv_path: Path, out_dir: Path):
    return rm.SpreadsheetMatrixConfig(
        csv_path=csv_path,
        out_dir=out_dir,
        latency_sec=0.0,
        trial_orders=("observed",),
    )


def test_reaction_matrices_split_by_genotype_when_multiple(tmp_path):
    """A dataset with two genotypes emits one reaction matrix per genotype into
    its own <dataset>/<genotype>/ subfolder (the user's '2 matrix plots')."""
    ev.set_protocol("legacy")
    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(
        csv_path,
        genotypes=[
            ("june_01_batch_1", "GR5a-Old"),
            ("june_02_batch_1", "GR5a-Old"),
            ("may_27_batch_2", "GR5a-GCaMP8"),
            ("may_28_batch_2", "GR5a-GCaMP8"),
        ],
    )
    out_dir = tmp_path / "matrix"
    rm.generate_reaction_matrices_from_csv(_matrix_cfg(csv_path, out_dir))

    old_pngs = list(out_dir.rglob("GR5a-Old/reaction_matrix_*.png"))
    gc_pngs = list(out_dir.rglob("GR5a-GCaMP8/reaction_matrix_*.png"))
    assert old_pngs, "expected a reaction matrix in the GR5a-Old subfolder"
    assert gc_pngs, "expected a reaction matrix in the GR5a-GCaMP8 subfolder"


def test_reaction_matrices_single_genotype_stays_flat(tmp_path):
    """A single-genotype dataset keeps one pooled matrix (no genotype subfolder)."""
    ev.set_protocol("legacy")
    csv_path = tmp_path / "model_predictions.csv"
    _write_predictions_csv(
        csv_path,
        genotypes=[
            ("june_01_batch_1", "GR5a-Old"),
            ("june_02_batch_1", "GR5a-Old"),
        ],
    )
    out_dir = tmp_path / "matrix"
    rm.generate_reaction_matrices_from_csv(_matrix_cfg(csv_path, out_dir))

    assert not list(out_dir.rglob("GR5a-*/reaction_matrix_*.png"))
    assert list(out_dir.rglob("reaction_matrix_*.png")), "expected a flat matrix"


# --- training-vs-control matrices: split by genotype -----------------------

def _write_train_control_predictions(path: Path, *, datasets) -> None:
    """datasets: dict dataset_name -> list of (fly, fly_type). testing_1..6."""
    rows = []
    for dataset, flies in datasets.items():
        for fly, fly_type in flies:
            for trial_num in range(1, 7):
                rows.append(
                    {
                        "dataset": dataset,
                        "fly": fly,
                        "fly_number": "1",
                        "fly_type": fly_type,
                        "trial_label": f"testing_{trial_num}",
                        "trial_type": "testing",
                        "prediction": (trial_num + (0 if "Old" in fly_type else 1)) % 2,
                        "global_min": 1.0,
                        "global_max": 90.0,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_training_vs_control_matrices_split_by_genotype(tmp_path):
    """When training + control both hold two genotypes, the train-vs-control
    comparison is emitted once per genotype into <Training>/<genotype>/, reading
    the per-genotype binary CSVs written by the reaction-matrix step."""
    ev.set_protocol("legacy")
    csv_path = tmp_path / "model_predictions.csv"
    flies = [
        ("june_01_batch_1", "GR5a-Old"),
        ("june_02_batch_1", "GR5a-Old"),
        ("may_27_batch_2", "GR5a-GCaMP8"),
        ("may_28_batch_2", "GR5a-GCaMP8"),
    ]
    _write_train_control_predictions(
        csv_path, datasets={"Hex-Training": flies, "Hex-Control": flies}
    )
    out_dir = tmp_path / "matrix"
    # 1) reaction matrices write per-genotype binary CSVs (trained-first order).
    rm.generate_reaction_matrices_from_csv(
        rm.SpreadsheetMatrixConfig(
            csv_path=csv_path, out_dir=out_dir, latency_sec=0.0,
            trial_orders=("trained-first",),
        )
    )
    # 2) training-vs-control reads them and emits per-genotype comparisons.
    tvc.generate_training_vs_control_matrices(
        rm.SpreadsheetMatrixConfig(
            csv_path=csv_path, out_dir=out_dir, latency_sec=0.0,
            trial_orders=("observed",),
        )
    )

    old = list(out_dir.rglob("GR5a-Old/reaction_matrix_train_vs_ctrl_*.png"))
    gc = list(out_dir.rglob("GR5a-GCaMP8/reaction_matrix_train_vs_ctrl_*.png"))
    assert old, "expected train-vs-ctrl matrix in GR5a-Old subfolder"
    assert gc, "expected train-vs-ctrl matrix in GR5a-GCaMP8 subfolder"


# --- score_summary plots: split by genotype --------------------------------

def _write_score_csv(path: Path, *, genotypes, dataset: str = "Hex-Training") -> None:
    """genotypes: list of (fly, fly_type). v2 odor-suffixed labels + ordinal score."""
    odors = ["hexanol", "benzaldehyde", "hexanol", "benzaldehyde"]
    rows = []
    for fly, fly_type in genotypes:
        for i, odor in enumerate(odors, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "fly": fly,
                    "fly_number": "1",
                    "fly_type": fly_type,
                    "trial_label": f"testing_{i}_{odor}",
                    "trial_type": "testing",
                    "score": (i + (0 if "Old" in fly_type else 1)) % 3,
                    "global_min": 1.0,
                    "global_max": 90.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_score_summary_splits_by_genotype_when_multiple(tmp_path):
    """When the scores span two genotypes, the score_summary set is emitted once
    per genotype into <out>/<genotype>/."""
    ev.set_protocol("v2")
    csv_path = tmp_path / "model_predictions.csv"
    _write_score_csv(
        csv_path,
        genotypes=[
            ("june_01_batch_1", "GR5a-Old"),
            ("june_02_batch_1", "GR5a-Old"),
            ("may_27_batch_2", "GR5a-GCaMP8"),
            ("may_28_batch_2", "GR5a-GCaMP8"),
        ],
    )
    out_dir = tmp_path / "score_summary"
    ss.generate_score_summary(csv_path, out_dir)

    for g in ("GR5a-Old", "GR5a-GCaMP8"):
        assert (out_dir / g / "score_summary_by_odor_testing.csv").exists(), f"{g} summary CSV missing"
        assert list((out_dir / g).glob("*.png")), f"{g} score plots missing"


def test_score_summary_single_genotype_stays_flat(tmp_path):
    """Single-genotype scores keep the flat layout (no genotype subfolder)."""
    ev.set_protocol("v2")
    csv_path = tmp_path / "model_predictions.csv"
    _write_score_csv(
        csv_path,
        genotypes=[
            ("june_01_batch_1", "GR5a-Old"),
            ("june_02_batch_1", "GR5a-Old"),
        ],
    )
    out_dir = tmp_path / "score_summary"
    ss.generate_score_summary(csv_path, out_dir)

    assert (out_dir / "score_summary_by_odor_testing.csv").exists()
    assert not (out_dir / "GR5a-Old").exists()
