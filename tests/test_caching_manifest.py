#!/usr/bin/env python3
"""
Unit tests for file manifest-based caching system.

Tests the caching logic in scripts/run_workflows.py including:
- File tracking filters
- Manifest building
- Manifest comparison
- Cache validation decisions
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import pytest

# Import functions from run_workflows
import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "scripts"))

from run_workflows import (
    _should_track_file,
    _build_file_manifest,
    _compare_manifests,
)


class TestFileTracking:
    """Tests for _should_track_file() logic."""

    def test_tracks_csv_in_nested_structure(self, tmp_path: Path):
        """CSV files in nested directories should be tracked."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("data")

        assert _should_track_file(csv_file, dataset_root) is True

    def test_excludes_csv_in_root(self, tmp_path: Path):
        """CSV files directly in dataset root should NOT be tracked."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "metadata.csv"
        csv_file.write_text("data")

        assert _should_track_file(csv_file, dataset_root) is False

    def test_excludes_csv_one_level_deep(self, tmp_path: Path):
        """CSV files one level deep should NOT be tracked (need fly_dir/trial_dir/)."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "metadata.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("data")

        assert _should_track_file(csv_file, dataset_root) is False

    def test_excludes_sensors_files(self, tmp_path: Path):
        """Files starting with 'sensors_' should NOT be tracked."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "trial1" / "sensors_data.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("data")

        assert _should_track_file(csv_file, dataset_root) is False

    def test_excludes_hidden_files(self, tmp_path: Path):
        """Hidden files (starting with '.') should NOT be tracked."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "trial1" / ".hidden.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("data")

        assert _should_track_file(csv_file, dataset_root) is False

    def test_excludes_non_csv_files(self, tmp_path: Path):
        """Non-CSV files should NOT be tracked."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        txt_file = dataset_root / "fly1" / "trial1" / "data.txt"
        txt_file.parent.mkdir(parents=True)
        txt_file.write_text("data")

        assert _should_track_file(txt_file, dataset_root) is False


class TestManifestBuilding:
    """Tests for _build_file_manifest() logic."""

    def test_builds_empty_manifest_for_empty_dataset(self, tmp_path: Path):
        """Empty dataset should produce empty manifest."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        manifest = _build_file_manifest(dataset_root)

        assert manifest == {}

    def test_builds_manifest_with_single_file(self, tmp_path: Path):
        """Single trackable CSV should appear in manifest."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("test data")

        manifest = _build_file_manifest(dataset_root)

        assert len(manifest) == 1
        assert str(csv_file.absolute()) in manifest

        file_info = manifest[str(csv_file.absolute())]
        assert "mtime" in file_info
        assert "size" in file_info
        assert file_info["size"] == len("test data")

    def test_builds_manifest_with_multiple_files(self, tmp_path: Path):
        """Multiple trackable CSVs should all appear in manifest."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        # Create multiple CSV files
        files = [
            dataset_root / "fly1" / "trial1" / "coords.csv",
            dataset_root / "fly1" / "trial2" / "coords.csv",
            dataset_root / "fly2" / "trial1" / "distances.csv",
        ]

        for f in files:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"data for {f.name}")

        manifest = _build_file_manifest(dataset_root)

        assert len(manifest) == 3
        for f in files:
            assert str(f.absolute()) in manifest

    def test_manifest_excludes_untrackable_files(self, tmp_path: Path):
        """Files that shouldn't be tracked must be excluded from manifest."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        # Trackable
        trackable = dataset_root / "fly1" / "trial1" / "coords.csv"
        trackable.parent.mkdir(parents=True)
        trackable.write_text("data")

        # Not trackable (various reasons)
        (dataset_root / "metadata.csv").write_text("data")  # Root level
        (dataset_root / "fly1" / "sensors_temp.csv").write_text("data")  # sensors_
        (dataset_root / "fly1" / "trial1" / ".hidden.csv").write_text("data")  # Hidden

        manifest = _build_file_manifest(dataset_root)

        assert len(manifest) == 1  # Only trackable file
        assert str(trackable.absolute()) in manifest


class TestManifestComparison:
    """Tests for _compare_manifests() logic."""

    def test_identical_manifests_are_valid(self):
        """Identical manifests should return is_valid=True."""
        manifest = {
            "/path/to/file.csv": {"mtime": 1234567890.0, "size": 1000}
        }

        is_valid, changes = _compare_manifests(manifest, manifest)

        assert is_valid is True
        assert len(changes) == 1
        assert "unchanged" in changes[0].lower()

    def test_detects_new_files(self):
        """New files should invalidate cache."""
        current = {
            "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
            "/path/to/file2.csv": {"mtime": 1234567891.0, "size": 2000},
        }
        cached = {
            "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
        }

        is_valid, changes = _compare_manifests(current, cached)

        assert is_valid is False
        assert any("new" in c.lower() for c in changes)
        assert any("file2.csv" in c for c in changes)

    def test_detects_deleted_files(self):
        """Deleted files should invalidate cache."""
        current = {
            "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
        }
        cached = {
            "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
            "/path/to/file2.csv": {"mtime": 1234567891.0, "size": 2000},
        }

        is_valid, changes = _compare_manifests(current, cached)

        assert is_valid is False
        assert any("deleted" in c.lower() for c in changes)
        assert any("file2.csv" in c for c in changes)

    def test_detects_modified_files_by_mtime(self):
        """Files with changed mtime should invalidate cache."""
        current = {
            "/path/to/file.csv": {"mtime": 1234567900.0, "size": 1000},
        }
        cached = {
            "/path/to/file.csv": {"mtime": 1234567890.0, "size": 1000},
        }

        is_valid, changes = _compare_manifests(current, cached)

        assert is_valid is False
        assert any("modified" in c.lower() for c in changes)

    def test_detects_modified_files_by_size(self):
        """Files with changed size should invalidate cache."""
        current = {
            "/path/to/file.csv": {"mtime": 1234567890.0, "size": 2000},
        }
        cached = {
            "/path/to/file.csv": {"mtime": 1234567890.0, "size": 1000},
        }

        is_valid, changes = _compare_manifests(current, cached)

        assert is_valid is False
        assert any("modified" in c.lower() for c in changes)

    def test_reports_multiple_changes(self):
        """Multiple types of changes should all be reported."""
        current = {
            "/path/to/file1.csv": {"mtime": 1234567900.0, "size": 1000},  # Modified
            "/path/to/file3.csv": {"mtime": 1234567892.0, "size": 3000},  # New
        }
        cached = {
            "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
            "/path/to/file2.csv": {"mtime": 1234567891.0, "size": 2000},  # Deleted
        }

        is_valid, changes = _compare_manifests(current, cached)

        assert is_valid is False
        # Should report new, modified, and deleted
        changes_text = " ".join(changes).lower()
        assert "new" in changes_text
        assert "modified" in changes_text
        assert "deleted" in changes_text


class TestManifestIntegration:
    """Integration tests for full manifest workflow."""

    def test_manifest_detects_file_modification(self, tmp_path: Path):
        """End-to-end test: modifying a file should invalidate cache."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("original data")

        # Build initial manifest
        manifest1 = _build_file_manifest(dataset_root)

        # Simulate time passing
        time.sleep(0.01)

        # Modify the file
        csv_file.write_text("modified data with more content")

        # Build new manifest
        manifest2 = _build_file_manifest(dataset_root)

        # Compare
        is_valid, changes = _compare_manifests(manifest2, manifest1)

        assert is_valid is False
        assert any("modified" in c.lower() for c in changes)

    def test_manifest_detects_new_file_addition(self, tmp_path: Path):
        """End-to-end test: adding a file should invalidate cache."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file1 = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv_file1.parent.mkdir(parents=True)
        csv_file1.write_text("data")

        # Build initial manifest
        manifest1 = _build_file_manifest(dataset_root)
        assert len(manifest1) == 1

        # Add a new file
        csv_file2 = dataset_root / "fly1" / "trial2" / "coords.csv"
        csv_file2.parent.mkdir(parents=True)
        csv_file2.write_text("new data")

        # Build new manifest
        manifest2 = _build_file_manifest(dataset_root)
        assert len(manifest2) == 2

        # Compare
        is_valid, changes = _compare_manifests(manifest2, manifest1)

        assert is_valid is False
        assert any("new" in c.lower() for c in changes)

    def test_manifest_stable_when_no_changes(self, tmp_path: Path):
        """End-to-end test: no changes should result in valid cache."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("data")

        # Build manifest twice without changes
        manifest1 = _build_file_manifest(dataset_root)
        manifest2 = _build_file_manifest(dataset_root)

        # Compare
        is_valid, changes = _compare_manifests(manifest2, manifest1)

        assert is_valid is True
        assert "unchanged" in changes[0].lower()


class TestPerformance:
    """Performance characteristics tests."""

    def test_manifest_building_scales_reasonably(self, tmp_path: Path):
        """Manifest building should be fast even with many files."""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        # Create 100 CSV files
        for fly in range(5):
            for trial in range(20):
                csv_file = dataset_root / f"fly{fly}" / f"trial{trial}" / "coords.csv"
                csv_file.parent.mkdir(parents=True, exist_ok=True)
                csv_file.write_text(f"data for fly{fly} trial{trial}")

        # Time manifest building
        start = time.time()
        manifest = _build_file_manifest(dataset_root)
        elapsed = time.time() - start

        assert len(manifest) == 100
        assert elapsed < 1.0  # Should complete in under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
