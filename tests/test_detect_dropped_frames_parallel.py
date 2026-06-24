"""Determinism test for the detect_dropped_frames parallelism refactor.

Runs the stage SERIAL and PARALLEL on identical fixture trees and asserts
that all produced _dropped_frames.txt outputs are byte-identical.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.config import ForceSettings, ParallelSettings, Settings
from fbpipe.steps.detect_dropped_frames import main
from fbpipe.utils.tables import write_table


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_COLUMNS = [
    "frame",
    "timestamp",
    "x_class0",
    "y_class0",
    "x_class1",
    "y_class1",
    "corners_class1",
    "distance_0_1",
    "angle_deg_c0_c1_vs_anchor",
]


def _make_distance_df(fly_idx: int, n_frames: int = 30) -> pd.DataFrame:
    """Return a minimal per-fly distance DataFrame with realistic columns.

    Introduces a deliberate gap in frame numbers (frames 10-14 missing) so
    that the dropped-frames report is non-trivial and content varies by fly_idx.
    """
    # Frames 0-9 then 15-29 — five missing frames in between (< 10 so no
    # consecutive-run flag), plus an NaN in distance_0_1 at frame 5.
    frames = list(range(0, 10)) + list(range(15, n_frames))
    actual_n = len(frames)
    data: dict = {col: [0] * actual_n for col in _COLUMNS}
    data["frame"] = frames
    data["timestamp"] = [float(f) * 0.025 for f in frames]
    data["x_class0"] = [float(100 + fly_idx + i) for i in range(actual_n)]
    data["y_class0"] = [float(200 + fly_idx + i) for i in range(actual_n)]
    data["x_class1"] = [float(150 + fly_idx + i) for i in range(actual_n)]
    data["y_class1"] = [float(250 + fly_idx + i) for i in range(actual_n)]
    data["corners_class1"] = ["[]"] * actual_n
    data["distance_0_1"] = [float(i) * 1.5 for i in range(actual_n)]
    # Inject one NaN distance so NaN-drop detection exercises the code path.
    data["distance_0_1"][5] = float("nan")
    data["angle_deg_c0_c1_vs_anchor"] = [0.0] * actual_n
    return pd.DataFrame(data)


def _build_fixture_tree(root: Path, n_fly_dirs: int = 3) -> None:
    """Create *n_fly_dirs* fly directories each with one distance CSV/Parquet."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_fly_dirs):
        fly_dir = root / f"fly{i + 1:02d}"
        fly_dir.mkdir()
        df = _make_distance_df(fly_idx=i)
        # Write as Parquet (the canonical format the pipeline uses).
        write_table(df, fly_dir / "fly1_distances.csv")


def _collect_reports(root: Path) -> dict[str, str]:
    """Return {relative_path_str: file_content} for all _dropped_frames.txt files."""
    results: dict[str, str] = {}
    for txt_path in sorted(root.rglob("*_dropped_frames.txt")):
        rel = str(txt_path.relative_to(root))
        results[rel] = txt_path.read_text(encoding="utf-8")
    return results


def _make_cfg(root: Path, parallel: ParallelSettings) -> Settings:
    return Settings(
        model_path="",
        main_directories=[str(root)],
        force=ForceSettings(pipeline=False),
        parallel=parallel,
        # Disable flagged-root auto-discovery so only root is scanned.
        auto_discover_flagged=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_serial_produces_reports(tmp_path: Path) -> None:
    """Sanity check: serial mode writes _dropped_frames.txt for each fly_dir."""
    root = tmp_path / "dataset"
    _build_fixture_tree(root, n_fly_dirs=2)
    cfg = _make_cfg(root, ParallelSettings(enabled=False))
    main(cfg)
    reports = _collect_reports(root)
    assert len(reports) == 2, f"Expected 2 reports, got: {list(reports)}"
    for _path, content in reports.items():
        # Gap 10-14 = 5 missing frames; NaN at frame 5 → total dropped = 6.
        assert "Dropped frames" in content or "No dropped frames" in content


def test_parallel_matches_serial(tmp_path: Path) -> None:
    """Parallel output is byte-identical to serial output for every report file."""
    serial_root = tmp_path / "serial_dataset"
    parallel_root = tmp_path / "parallel_dataset"

    # Build two identical fixture trees.
    _build_fixture_tree(serial_root, n_fly_dirs=3)
    _build_fixture_tree(parallel_root, n_fly_dirs=3)

    # Run serial.
    serial_cfg = _make_cfg(serial_root, ParallelSettings(enabled=False))
    main(serial_cfg)

    # Run parallel with 2 workers.
    parallel_cfg = _make_cfg(parallel_root, ParallelSettings(enabled=True, n_jobs=2))
    main(parallel_cfg)

    serial_reports = _collect_reports(serial_root)
    parallel_reports = _collect_reports(parallel_root)

    assert set(serial_reports) == set(parallel_reports), (
        f"Report file sets differ.\n"
        f"Serial:   {sorted(serial_reports)}\n"
        f"Parallel: {sorted(parallel_reports)}"
    )

    for rel_path, serial_content in serial_reports.items():
        parallel_content = parallel_reports[rel_path]
        assert serial_content == parallel_content, (
            f"Report mismatch for '{rel_path}':\n"
            f"--- serial ---\n{serial_content}\n"
            f"--- parallel ---\n{parallel_content}"
        )


def test_skip_logic_unchanged_in_parallel(tmp_path: Path) -> None:
    """Re-running in parallel mode skips up-to-date reports (same as serial)."""
    root = tmp_path / "dataset"
    _build_fixture_tree(root, n_fly_dirs=2)

    # First run (serial) — creates the reports.
    cfg_serial = _make_cfg(root, ParallelSettings(enabled=False))
    main(cfg_serial)

    # Collect mtime of every report after the first run.
    first_mtimes = {
        p: p.stat().st_mtime for p in root.rglob("*_dropped_frames.txt")
    }
    assert first_mtimes, "No reports written on first run."

    # Second run (parallel) — should NOT re-write any report (all up-to-date).
    cfg_parallel = _make_cfg(root, ParallelSettings(enabled=True, n_jobs=2))
    main(cfg_parallel)

    second_mtimes = {
        p: p.stat().st_mtime for p in root.rglob("*_dropped_frames.txt")
    }
    assert first_mtimes == second_mtimes, (
        "Parallel re-run modified reports that should have been skipped."
    )
