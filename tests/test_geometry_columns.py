"""Column names, file naming and table I/O of the abdomen pipeline, plus the
isolation guarantee: production enumerators must never see these files."""
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.geometry import columns as C


def _df():
    return pd.DataFrame({"frame": [0, 1], "proboscis_x": [1.5, float("nan")], "geometry_status": ["ok", "ok"]})


def test_file_naming_uses_the_geometry_suffix(tmp_path):
    p = C.geometry_table_path(tmp_path, "september_02_batch_1_testing_3", 2, "csv")
    assert p == tmp_path / "september_02_batch_1_testing_3_fly2_geometry.csv"
    assert C.geometry_table_path(tmp_path, "t", 1, "parquet").suffix == ".parquet"
    assert C.meta_path(tmp_path, "t").name == "t_geometry_meta.json"
    assert C.annotated_video_path(tmp_path, "t").name == "t_geometry_annotated.mp4"
    assert C.stats_path(tmp_path, 3).name == "fly3_geometry_stats.json"
    assert C.fly_slot_from_geometry_name("x_fly12_geometry.parquet") == 12
    assert C.fly_slot_from_geometry_name("x_fly1_distances.csv") is None


def test_validate_table_format_rejects_unknown_values():
    assert C.validate_table_format("CSV") == "csv"
    assert C.validate_table_format("parquet") == "parquet"
    with pytest.raises(ValueError):
        C.validate_table_format("xlsx")


def test_write_csv_then_parquet_keeps_both_files(tmp_path):
    stem = tmp_path / "t_fly1_geometry.csv"
    out_csv = C.write_table_fmt(_df(), stem, "csv")
    out_pq = C.write_table_fmt(_df(), stem, "parquet")
    assert out_csv.exists() and out_csv.suffix == ".csv"
    assert out_pq.exists() and out_pq.suffix == ".parquet"
    assert out_csv.exists(), "switching to parquet must not delete the verified CSV"
    back = C.read_table_any(out_pq)
    assert list(back.columns) == ["frame", "proboscis_x", "geometry_status"]
    assert back["frame"].tolist() == [0, 1]


def test_iter_geometry_tables_prefers_parquet_and_sorts_by_slot(tmp_path):
    for name in ("t_fly2_geometry.csv", "t_fly1_geometry.csv", "t_fly1_geometry.parquet", "t_fly1_distances.csv"):
        _df().to_csv(tmp_path / name, index=False)
    found = C.iter_geometry_tables(tmp_path)
    assert [slot for slot, _ in found] == [1, 2]
    assert found[0][1].suffix == ".parquet"


def test_production_enumerators_never_see_geometry_files(tmp_path):
    """The production per-batch rglobs must not pick up anything the new pipeline writes."""
    from fbpipe.utils.fly_files import fly_slot_from_name, iter_fly_distance_csvs
    from fbpipe.steps.check_light_stimulus import TRIAL_DIR_RE

    batch = tmp_path / "september_02_batch_1_rig_3_copy"
    trial = batch / "geometry" / "september_02_batch_1_testing_3"
    trial.mkdir(parents=True)
    for name in ("september_02_batch_1_testing_3_fly1_geometry.csv",
                 "september_02_batch_1_testing_3_fly1_geometry.parquet",
                 "september_02_batch_1_testing_3_geometry_meta.json",
                 "september_02_batch_1_testing_3_geometry_annotated.mp4"):
        (trial / name).write_text("x")
    (batch / "geometry" / "fly1_geometry_stats.json").write_text("{}")
    assert list(iter_fly_distance_csvs(batch, recursive=True)) == []
    for name in ("september_02_batch_1_testing_3_fly1_geometry.csv", "fly1_geometry_stats.json"):
        assert fly_slot_from_name(name) is None
    assert TRIAL_DIR_RE.search("geometry") is None
    # the production trial regex does match our per-trial folder name -- but that folder sits
    # under geometry/, which check_light_stimulus only enumerates one level below the batch
    assert TRIAL_DIR_RE.search(trial.name) is not None
    assert [p.name for p in batch.iterdir()] == ["geometry"]
