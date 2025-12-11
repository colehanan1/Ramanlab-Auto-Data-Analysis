#!/usr/bin/env python3
"""
Standalone test for file manifest caching logic.

Run without pytest to avoid environment issues.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
import sys

# Import functions from run_workflows
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from run_workflows import (
    _should_track_file,
    _build_file_manifest,
    _compare_manifests,
)


def test_file_tracking():
    """Test that file tracking filters work correctly."""
    print("\n🧪 Testing file tracking filters...")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir) / "dataset"
        dataset_root.mkdir()

        # Should track: nested CSV
        tracked_csv = dataset_root / "fly1" / "trial1" / "coords.csv"
        tracked_csv.parent.mkdir(parents=True)
        tracked_csv.write_text("data")

        # Should NOT track: root CSV
        root_csv = dataset_root / "metadata.csv"
        root_csv.write_text("data")

        # Should NOT track: sensors file
        sensors_csv = dataset_root / "fly1" / "trial1" / "sensors_temp.csv"
        sensors_csv.write_text("data")

        # Test
        assert _should_track_file(tracked_csv, dataset_root) is True, "Should track nested CSV"
        assert _should_track_file(root_csv, dataset_root) is False, "Should NOT track root CSV"
        assert _should_track_file(sensors_csv, dataset_root) is False, "Should NOT track sensors CSV"

    print("✅ File tracking filters work correctly")


def test_manifest_building():
    """Test that manifest building captures file metadata."""
    print("\n🧪 Testing manifest building...")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir) / "dataset"
        dataset_root.mkdir()

        # Create test files
        csv1 = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv1.parent.mkdir(parents=True)
        csv1.write_text("data for file 1")

        csv2 = dataset_root / "fly2" / "trial1" / "distances.csv"
        csv2.parent.mkdir(parents=True)
        csv2.write_text("data for file 2")

        # Build manifest
        manifest = _build_file_manifest(dataset_root)

        # Verify
        assert len(manifest) == 2, f"Expected 2 files, got {len(manifest)}"
        assert str(csv1.absolute()) in manifest, "csv1 should be in manifest"
        assert str(csv2.absolute()) in manifest, "csv2 should be in manifest"

        # Check metadata
        info1 = manifest[str(csv1.absolute())]
        assert "mtime" in info1, "Should have mtime"
        assert "size" in info1, "Should have size"
        assert info1["size"] == len("data for file 1"), "Size should match"

    print("✅ Manifest building works correctly")


def test_manifest_comparison_unchanged():
    """Test that identical manifests are detected as valid."""
    print("\n🧪 Testing manifest comparison (no changes)...")

    manifest = {
        "/path/to/file.csv": {"mtime": 1234567890.0, "size": 1000}
    }

    is_valid, changes = _compare_manifests(manifest, manifest)

    assert is_valid is True, "Identical manifests should be valid"
    assert len(changes) == 0, f"Should have no changes, got: {changes}"

    print("✅ Manifest comparison detects unchanged files correctly")


def test_manifest_comparison_new_file():
    """Test that new files invalidate cache."""
    print("\n🧪 Testing manifest comparison (new file)...")

    cached = {
        "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
    }
    current = {
        "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
        "/path/to/file2.csv": {"mtime": 1234567891.0, "size": 2000},
    }

    is_valid, changes = _compare_manifests(current, cached)

    assert is_valid is False, "New file should invalidate cache"
    assert any("new" in c.lower() for c in changes), f"Should mention new file, got: {changes}"

    print("✅ Manifest comparison detects new files correctly")


def test_manifest_comparison_deleted_file():
    """Test that deleted files invalidate cache."""
    print("\n🧪 Testing manifest comparison (deleted file)...")

    cached = {
        "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
        "/path/to/file2.csv": {"mtime": 1234567891.0, "size": 2000},
    }
    current = {
        "/path/to/file1.csv": {"mtime": 1234567890.0, "size": 1000},
    }

    is_valid, changes = _compare_manifests(current, cached)

    assert is_valid is False, "Deleted file should invalidate cache"
    assert any("deleted" in c.lower() for c in changes), f"Should mention deleted file, got: {changes}"

    print("✅ Manifest comparison detects deleted files correctly")


def test_manifest_comparison_modified_file():
    """Test that modified files invalidate cache."""
    print("\n🧪 Testing manifest comparison (modified file)...")

    cached = {
        "/path/to/file.csv": {"mtime": 1234567890.0, "size": 1000},
    }
    current = {
        "/path/to/file.csv": {"mtime": 1234567900.0, "size": 1000},  # Changed mtime
    }

    is_valid, changes = _compare_manifests(current, cached)

    assert is_valid is False, "Modified file should invalidate cache"
    assert any("modified" in c.lower() for c in changes), f"Should mention modified file, got: {changes}"

    print("✅ Manifest comparison detects modified files correctly")


def test_end_to_end_file_modification():
    """Integration test: modifying a file should be detected."""
    print("\n🧪 Testing end-to-end file modification detection...")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir) / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("original data")

        # Build initial manifest
        manifest1 = _build_file_manifest(dataset_root)

        # Wait to ensure mtime changes
        time.sleep(0.01)

        # Modify file
        csv_file.write_text("modified data with more content")

        # Build new manifest
        manifest2 = _build_file_manifest(dataset_root)

        # Compare
        is_valid, changes = _compare_manifests(manifest2, manifest1)

        assert is_valid is False, "Modified file should invalidate cache"
        assert any("modified" in c.lower() for c in changes), f"Should detect modification, got: {changes}"

    print("✅ End-to-end modification detection works correctly")


def test_end_to_end_new_file():
    """Integration test: adding a file should be detected."""
    print("\n🧪 Testing end-to-end new file detection...")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir) / "dataset"
        dataset_root.mkdir()

        csv1 = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv1.parent.mkdir(parents=True)
        csv1.write_text("data")

        # Build initial manifest
        manifest1 = _build_file_manifest(dataset_root)
        assert len(manifest1) == 1

        # Add new file
        csv2 = dataset_root / "fly1" / "trial2" / "coords.csv"
        csv2.parent.mkdir(parents=True)
        csv2.write_text("new data")

        # Build new manifest
        manifest2 = _build_file_manifest(dataset_root)
        assert len(manifest2) == 2

        # Compare
        is_valid, changes = _compare_manifests(manifest2, manifest1)

        assert is_valid is False, "New file should invalidate cache"
        assert any("new" in c.lower() for c in changes), f"Should detect new file, got: {changes}"

    print("✅ End-to-end new file detection works correctly")


def test_end_to_end_no_changes():
    """Integration test: no changes should keep cache valid."""
    print("\n🧪 Testing end-to-end cache stability (no changes)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir) / "dataset"
        dataset_root.mkdir()

        csv_file = dataset_root / "fly1" / "trial1" / "coords.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("data")

        # Build manifest twice
        manifest1 = _build_file_manifest(dataset_root)
        manifest2 = _build_file_manifest(dataset_root)

        # Compare
        is_valid, changes = _compare_manifests(manifest2, manifest1)

        assert is_valid is True, "No changes should keep cache valid"
        assert len(changes) == 0, f"Should have no changes, got: {changes}"

    print("✅ End-to-end cache stability works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 Testing File Manifest-Based Caching System")
    print("=" * 60)

    tests = [
        test_file_tracking,
        test_manifest_building,
        test_manifest_comparison_unchanged,
        test_manifest_comparison_new_file,
        test_manifest_comparison_deleted_file,
        test_manifest_comparison_modified_file,
        test_end_to_end_file_modification,
        test_end_to_end_new_file,
        test_end_to_end_no_changes,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 ERROR: {test_func.__name__}")
            print(f"   Exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All tests passed! Caching system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
