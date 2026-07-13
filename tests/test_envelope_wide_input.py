"""Raw PER traces read the wide combined_base table directly (Parquet/CSV).

The Raw-Testing / Raw-Training per-fly trace figures used to read the float16
``.npy`` envelope matrix. They now read the wide table
(``all_envelope_rows_wide_combined_base.parquet``) via
``EnvelopePlotConfig.wide_input`` + ``envelope_visuals._load_wide_table`` so the
pipeline no longer needs to build that intermediate matrix.

These tests pin that:
  * the wide loader returns the same ``(df, env_cols)`` shape the matrix loader
    does (same id columns + ``dir_val_*`` traces, within float16 tolerance);
  * ``generate_envelope_plots`` renders identical figures from ``wide_input``
    without ever opening ``matrix_npy`` (which is kept only as a label hint);
  * the config guards against the ``use_per_trial_baseline`` regression that
    removing ``combined.wide`` would otherwise cause.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.analysis import envelope_combined as ec
from scripts.analysis import envelope_visuals as ev


def _write_wide_csv(path: Path) -> None:
    """Minimal testing-only wide CSV: two flies, two trials each, 4-sample traces."""
    rows = []
    for fly in ("june_01_batch_1", "june_02_batch_1"):
        for trial_num in (1, 2):
            rows.append(
                {
                    "dataset": "Hex-Training",
                    "fly": fly,
                    "fly_number": "1",
                    "fly_type": "GR5a-Old",
                    "trial_type": "testing",
                    "trial_label": f"testing_{trial_num}",
                    "fps": 40.0,
                    "global_min": 1.0,
                    "global_max": 25.0,
                    "trimmed_global_min": 1.0,
                    "trimmed_global_max": 25.0,
                    "trace_len": 4,
                    "dir_val_0": 0.0 + trial_num,
                    "dir_val_1": 8.5,
                    "dir_val_2": 16.25,
                    "dir_val_3": 4.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _wide_cfg(wide_input: Path, out_dir: Path, *, matrix_npy: Path) -> ev.EnvelopePlotConfig:
    return ev.EnvelopePlotConfig(
        # Intentionally bogus (never-created) matrix path: proves wide_input is
        # the data source and matrix_npy is only a label hint.
        matrix_npy=matrix_npy,
        codes_json=Path("/does/not/exist/code_maps.json"),
        out_dir=out_dir,
        latency_sec=0.0,
        odor_latency_s=0.0,
        trial_type="testing",
        wide_input=wide_input,
        overwrite=True,
    )


def test_wide_loader_matches_matrix_loader(tmp_path):
    """_load_wide_table returns the same ids + dir_val traces as _load_matrix."""
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    _write_wide_csv(wide_csv)

    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))

    m_df, m_env = ev._load_matrix(
        matrix_dir / "envelope_matrix_float16.npy", matrix_dir / "code_maps.json"
    )
    # wide_to_matrix writes a parquet sibling; read the wide table directly.
    w_df, w_env = ev._load_wide_table(wide_csv)

    assert m_env == w_env == ["dir_val_0", "dir_val_1", "dir_val_2", "dir_val_3"]

    # generate_envelope_plots runs both frames through _normalise_fly_columns
    # before use, so compare identity after that same normalisation (the matrix
    # loader decodes fly_number to str, the wide loader lets pandas infer it).
    key = ["dataset", "fly", "fly_number", "trial_label"]
    m = ev._normalise_fly_columns(m_df).set_index(key).sort_index()
    w = ev._normalise_fly_columns(w_df).set_index(key).sort_index()
    assert list(m.index) == list(w.index)

    # Values agree within float16 rounding (the wide table is full precision).
    np.testing.assert_allclose(
        w.loc[m.index, m_env].to_numpy(float),
        m[m_env].to_numpy(float),
        atol=0.05,
    )


def test_generate_envelope_plots_uses_wide_input_not_matrix(tmp_path):
    """Figures render from wide_input even when matrix_npy points nowhere."""
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    _write_wide_csv(wide_csv)
    wide_parquet = tmp_path / "wide.parquet"
    pd.read_csv(wide_csv).to_parquet(wide_parquet, engine="pyarrow", index=False)

    out_dir = tmp_path / "plots"
    ev.generate_envelope_plots(
        _wide_cfg(wide_parquet, out_dir, matrix_npy=tmp_path / "nope.npy")
    )

    pngs = list((out_dir / "Hex-Training").rglob("*.png"))
    assert pngs, "no figures rendered from wide_input"


def test_wide_input_render_matches_matrix_render(tmp_path):
    """The wide_input render produces the same set of figures as the matrix render."""
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    _write_wide_csv(wide_csv)
    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))

    # Matrix-backed render (baseline).
    matrix_out = tmp_path / "plots_matrix"
    ev.generate_envelope_plots(
        ev.EnvelopePlotConfig(
            matrix_npy=matrix_dir / "envelope_matrix_float16.npy",
            codes_json=matrix_dir / "code_maps.json",
            out_dir=matrix_out,
            latency_sec=0.0,
            odor_latency_s=0.0,
            trial_type="testing",
            overwrite=True,
        )
    )

    # Wide-backed render (new path); matrix_npy kept as the same label hint.
    wide_out = tmp_path / "plots_wide"
    ev.generate_envelope_plots(
        _wide_cfg(
            wide_csv, wide_out, matrix_npy=matrix_dir / "envelope_matrix_float16.npy"
        )
    )

    rel = lambda root: sorted(p.relative_to(root).as_posix() for p in root.rglob("*.png"))
    assert rel(matrix_out) == rel(wide_out) and rel(wide_out), (
        rel(matrix_out),
        rel(wide_out),
    )


def test_config_new_keeps_combined_base_baseline_and_drops_removed_blocks():
    """Guard the config surgery: combined_base keeps use_per_trial_baseline and
    the removed blocks stay removed (so the baseline-flag trap can't return)."""
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    combined = raw["analysis"]["combined"]
    combined_base = combined["combined_base"]

    assert combined_base["wide"]["use_per_trial_baseline"] is True
    assert "matrix" not in combined_base  # orphaned .npy build dropped
    for env in combined_base["envelopes"]:
        assert str(env["wide_input"]).endswith(".parquet")

    for removed in ("wide", "envelopes", "matrices", "matrix", "distance_base"):
        assert removed not in combined, f"combined.{removed} should be removed"
    assert "envelope_visuals" not in raw["analysis"]
    assert "training" not in raw["analysis"]
