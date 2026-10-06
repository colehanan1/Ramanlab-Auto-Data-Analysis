"""The per-trial steps scan fly dirs WITHOUT the derived RMS copies.

``iter_fly_distance_csvs`` recurses, and every ``RMS_calculations/updated_*_
flyN_distances.parquet`` matches the ``flyN_distances`` pattern. So reject,
distance_stats, distance_normalize and detect_dropped_frames used to re-read
every RMS copy (and detect_dropped_frames wrote a junk report next to each),
although rms_copy_filter overwrites those copies from their sources right after.

The helper takes an explicit ``exclude_dirs``. Its DEFAULT is unchanged
(recursive, nothing excluded -- ``tests/test_tables.py::
test_recursive_finds_parquet_in_subdir`` pins that), and only the per-trial
steps that walk a whole batch dir opt in.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.config import (
    ForceSettings,
    ParallelSettings,
    ProboscisFilterSettings,
    Settings,
)
from fbpipe.utils.fly_files import DERIVED_TABLE_DIRS, iter_fly_distance_csvs
from fbpipe.utils.tables import read_table, write_table

TRIAL = "october_05_batch_2_testing_1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _source_df(n: int = 30) -> pd.DataFrame:
    """A per-fly trial table: eye at (500, 500), proboscis 40 px right of it."""
    frames = list(range(n))
    df = pd.DataFrame(
        {
            "frame": frames,
            "timestamp": [f * 0.025 for f in frames],
            "track_id_class0": [1] * n,
            "x_class0": [500.0] * n,
            "y_class0": [500.0] * n,
            "track_id_class1": [2] * n,
            "x_class1": [540.0 + (i % 3) for i in range(n)],
            "y_class1": [500.0] * n,
            "distance_0_1": [40.0 + (i % 3) for i in range(n)],
            "angle_deg_c0_c1_vs_anchor": [0.0] * n,
        }
    )
    return df


def _rms_copy_df(n: int = 30) -> pd.DataFrame:
    """An RMS copy as rms_copy_filter writes it, plus one out-of-gate point.

    The proboscis at frame 7 sits 400 px from its eye, far outside every gate
    used below, so a reject pass that reaches this file WOULD rewrite it.
    """
    df = _source_df(n)[["frame", "timestamp", "x_class0", "y_class0", "x_class1", "y_class1"]].copy()
    df["distance_percentage_0_1"] = 25.0
    df.loc[7, "x_class1"] = 900.0
    return df


def _build_batch(root: Path) -> Path:
    """One batch dir: a trial table, its RMS copy, and a combine envelope CSV.

    The slot's stats JSON is there too, so distance_normalize has work to do.
    """
    batch = root / "october_05_batch_2"
    trial = batch / TRIAL
    trial.mkdir(parents=True)
    write_table(_source_df(), trial / f"{TRIAL}_fly1_distances.parquet")
    (batch / "fly1_global_distance_stats_class_0.json").write_text(
        json.dumps({"global_min": 40.0, "global_max": 42.0, "fly_max_distance": 42.0,
                    "effective_max_threshold": 95.0}),
        encoding="utf-8",
    )
    rms = batch / "RMS_calculations"
    write_table(_rms_copy_df(), rms / f"updated_{TRIAL}_fly1_distances.parquet")
    env = batch / "angle_distance_rms_envelope"
    env.mkdir()
    pd.DataFrame({"t": [0.0], "v": [1.0]}).to_csv(
        env / f"testing_1_hexanol_fly1_distances_fly1_angle_distance_rms_envelope.csv",
        index=False,
    )
    return batch


def _cfg(root: Path, *, force: bool = True) -> Settings:
    return Settings(
        model_path="",
        main_directories=[str(root)],
        auto_discover_flagged=False,
        allow_cpu=True,
        force=ForceSettings(pipeline=force),
        parallel=ParallelSettings(enabled=False),
        class2_min=10.0,
        class2_max=150.0,
        # The two gates are distinct keys; give them different values so a
        # step reading the wrong one cannot pass by coincidence.
        proboscis_filter=ProboscisFilterSettings(
            enabled=True, max_eye_prob_distance_px=150.0, up_divisor=4.0, max_jump_px=80.0
        ),
        three_fly_max_eye_prob_distance_px=170.0,
    )


def _spy_reads(monkeypatch, *modules) -> list[Path]:
    """Record every path passed to read_table / read_schema_columns in *modules*."""
    seen: list[Path] = []
    for mod in modules:
        for name in ("read_table", "read_schema_columns"):
            orig = getattr(mod, name, None)
            if orig is None:
                continue

            def wrapper(path, *args, _orig=orig, **kwargs):
                seen.append(Path(path))
                return _orig(path, *args, **kwargs)

            monkeypatch.setattr(mod, name, wrapper)
    return seen


def _in_derived_dir(path: Path) -> bool:
    return any(part in DERIVED_TABLE_DIRS for part in path.parts)


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_derived_table_dirs_are_the_rms_and_envelope_dirs():
    assert DERIVED_TABLE_DIRS == ("RMS_calculations", "angle_distance_rms_envelope")


def test_default_scan_still_recurses_into_rms_calculations(tmp_path):
    batch = _build_batch(tmp_path)
    found = [p.name for p, _, _ in iter_fly_distance_csvs(batch, recursive=True)]
    assert sorted(found) == sorted(
        [f"{TRIAL}_fly1_distances.parquet", f"updated_{TRIAL}_fly1_distances.parquet"]
    )


def test_exclude_dirs_drops_the_rms_copies(tmp_path):
    batch = _build_batch(tmp_path)
    found = list(iter_fly_distance_csvs(batch, recursive=True, exclude_dirs=DERIVED_TABLE_DIRS))
    assert [(p.name, token, slot) for p, token, slot in found] == [
        (f"{TRIAL}_fly1_distances.parquet", "fly1_distances", 1)
    ]


def test_exclude_dirs_keeps_the_default_order_of_what_remains(tmp_path):
    """Exclusion only removes entries; it never reorders the rest."""
    batch = _build_batch(tmp_path)
    for name in ("testing_2", "training_1", "pretest_1"):
        trial = batch / f"october_05_batch_2_{name}"
        trial.mkdir()
        for slot in (1, 2):
            write_table(_source_df(), trial / f"{trial.name}_fly{slot}_distances.parquet")
            write_table(_rms_copy_df(), batch / "RMS_calculations" / f"updated_{trial.name}_fly{slot}_distances.parquet")
    # A legacy CSV twin must still lose to its parquet.
    _source_df().to_csv(batch / "october_05_batch_2_training_1" / "october_05_batch_2_training_1_fly2_distances.csv", index=False)

    default = [p for p, _, _ in iter_fly_distance_csvs(batch, recursive=True)]
    excluded = [p for p, _, _ in iter_fly_distance_csvs(batch, recursive=True, exclude_dirs=DERIVED_TABLE_DIRS)]
    assert excluded == [p for p in default if not _in_derived_dir(p.relative_to(batch.resolve()))]
    assert len(excluded) == 7
    assert all(p.suffix == ".parquet" for p in excluded)


def test_exclude_dirs_applies_at_any_depth_below_base(tmp_path):
    batch = _build_batch(tmp_path)
    nested = batch / TRIAL / "RMS_calculations"
    write_table(_rms_copy_df(), nested / f"updated_{TRIAL}_fly2_distances.parquet")
    found = [p.name for p, _, _ in iter_fly_distance_csvs(batch, recursive=True, exclude_dirs=DERIVED_TABLE_DIRS)]
    assert found == [f"{TRIAL}_fly1_distances.parquet"]


def test_exclude_dirs_never_excludes_base_itself(tmp_path):
    """calculate_acceleration and compose_videos_rms scan RMS_calculations as
    the BASE; an exclusion that looked at the base's own name would empty them."""
    batch = _build_batch(tmp_path)
    rms = batch / "RMS_calculations"
    for recursive in (True, False):
        found = [p.name for p, _, _ in iter_fly_distance_csvs(rms, recursive=recursive, exclude_dirs=DERIVED_TABLE_DIRS)]
        assert found == [f"updated_{TRIAL}_fly1_distances.parquet"], recursive


def test_exclude_dirs_matches_whole_directory_names_only(tmp_path):
    batch = _build_batch(tmp_path)
    lookalike = batch / "RMS_calculations_old"
    write_table(_rms_copy_df(), lookalike / f"updated_{TRIAL}_fly3_distances.parquet")
    found = sorted(p.name for p, _, _ in iter_fly_distance_csvs(batch, recursive=True, exclude_dirs=DERIVED_TABLE_DIRS))
    assert found == [f"{TRIAL}_fly1_distances.parquet", f"updated_{TRIAL}_fly3_distances.parquet"]


def test_excluded_files_are_not_even_name_checked(tmp_path, capsys):
    """The envelope CSVs carry '_distances' but no slot suffix; the default scan
    logs a '[FLYFILES] ... missing the fly# token' line for each of them."""
    batch = _build_batch(tmp_path)
    list(iter_fly_distance_csvs(batch, recursive=True))
    assert "missing the fly# token" in capsys.readouterr().out
    list(iter_fly_distance_csvs(batch, recursive=True, exclude_dirs=DERIVED_TABLE_DIRS))
    out = capsys.readouterr().out
    assert "missing the fly# token" not in out
    assert "updated_" not in out


# ---------------------------------------------------------------------------
# The steps use it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "fbpipe.steps.reject_bad_proboscis",
        "fbpipe.steps.distance_stats",
        "fbpipe.steps.distance_normalize",
        "fbpipe.steps.distance_normalize_gpu",
        "fbpipe.steps.distance_normalize_ultra",
        "fbpipe.steps.detect_dropped_frames",
        "fbpipe.steps.rms_copy_filter",
    ],
)
def test_step_never_reads_an_rms_copy(module_name, tmp_path, monkeypatch):
    import fbpipe.utils.distance_sanity as distance_sanity

    module = importlib.import_module(module_name)
    root = tmp_path / "Hex-HexTest-Training-24-1"
    batch = _build_batch(root)
    seen = _spy_reads(monkeypatch, module, distance_sanity)

    module.main(_cfg(root))

    source = batch / TRIAL / f"{TRIAL}_fly1_distances.parquet"
    assert any(p.resolve() == source.resolve() for p in seen), "step never read its source table"
    leaked = [p for p in seen if _in_derived_dir(p)]
    assert leaked == [], f"{module_name} read derived copies: {leaked}"


@pytest.mark.parametrize("force", [True, False])
def test_reject_leaves_rms_copies_alone(tmp_path, force):
    """The RMS copy has a 400 px proboscis: the old recursive scan blanked it.
    rms_copy_filter re-derives every copy from its (sanitized) source, so the
    copy is not reject's to touch."""
    from fbpipe.steps import reject_bad_proboscis

    root = tmp_path / "Hex-HexTest-Training-24-1"
    batch = _build_batch(root)
    rms_copy = batch / "RMS_calculations" / f"updated_{TRIAL}_fly1_distances.parquet"
    before = rms_copy.read_bytes()

    reject_bad_proboscis.main(_cfg(root, force=force))

    assert rms_copy.read_bytes() == before
    assert read_table(rms_copy).loc[7, "x_class1"] == pytest.approx(900.0)


@pytest.mark.parametrize("force", [True, False])
def test_detect_dropped_frames_writes_no_rms_side_reports(tmp_path, force):
    from fbpipe.steps import detect_dropped_frames

    root = tmp_path / "Hex-HexTest-Training-24-1"
    batch = _build_batch(root)

    detect_dropped_frames.main(_cfg(root, force=force))

    assert (batch / TRIAL / f"{TRIAL}_fly1_distances_dropped_frames.txt").is_file()
    assert list((batch / "RMS_calculations").glob("*_dropped_frames.txt")) == []


def test_distance_stats_non_force_skip_is_no_longer_defeated_by_rms_copies(tmp_path):
    """RMS copies never carry ``distance_percentage``, so the recursive scan made
    ``_needs_stats_refresh`` true for every slot with a copy and the non-force
    skip never fired. Now a slot is skipped exactly when the step's own rule
    says so (stats exist, sources normalized, no 3-fly over-limit rows).

    NOTE for force.pipeline=false: this is also why a gate change no longer
    re-derives stats by accident -- a config change must force the re-derive.
    """
    from fbpipe.steps import distance_normalize, distance_stats

    root = tmp_path / "Hex-HexTest-Training-24-1"
    batch = _build_batch(root)
    cfg = _cfg(root, force=False)
    distance_stats.main(cfg)
    distance_normalize.main(cfg)
    stats_json = batch / "fly1_global_distance_stats_class_0.json"
    before = (stats_json.read_bytes(), stats_json.stat().st_mtime_ns)

    distance_stats.main(cfg)

    assert (stats_json.read_bytes(), stats_json.stat().st_mtime_ns) == before


def test_rms_copy_filter_still_copies_every_source(tmp_path):
    from fbpipe.steps import rms_copy_filter

    root = tmp_path / "Hex-HexTest-Training-24-1"
    batch = _build_batch(root)

    rms_copy_filter.main(_cfg(root, force=True))

    copies = sorted(p.name for p in (batch / "RMS_calculations").glob("updated_*.parquet"))
    assert copies == [f"updated_{TRIAL}_fly1_distances.parquet"]
    # Re-derived from the source: the 400 px planted point is gone.
    assert read_table(batch / "RMS_calculations" / copies[0]).loc[7, "x_class1"] == pytest.approx(541.0)
    # No second-generation copy of a copy.
    assert not list((batch / "RMS_calculations").rglob("updated_updated_*"))
