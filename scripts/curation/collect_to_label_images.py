#!/usr/bin/env python3
"""
Collect all to-be-labeled images from yolo_curation/to_label directories
across all datasets in Data-secured and copy them to a central location.

Usage:
    python scripts/curation/collect_to_label_images.py
    python scripts/curation/collect_to_label_images.py --dry-run  # preview only
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fbpipe.config import load_raw_config


def collect_to_label_images(
    source_root: Path,
    dest_dir: Path,
    dry_run: bool = False,
    preserve_structure: bool = False,
) -> dict:
    """
    Collect all images from yolo_curation/to_label directories.

    Args:
        source_root: Root directory to search (e.g., /path/to/secure/storage)
        dest_dir: Destination directory for collected images
        dry_run: If True, only report what would be copied without actually copying
        preserve_structure: If True, preserve dataset/experiment folder structure

    Returns:
        Dictionary with statistics about the collection
    """
    stats = defaultdict(int)

    # Find all to_label directories
    to_label_dirs = list(source_root.glob("**/yolo_curation/to_label"))
    print(f"Found {len(to_label_dirs)} to_label directories")

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    skipped_files = []

    for to_label_dir in sorted(to_label_dirs):
        # Extract dataset and experiment names for context
        parts = to_label_dir.relative_to(source_root).parts
        dataset_name = parts[0] if len(parts) > 0 else "unknown"
        experiment_name = parts[1] if len(parts) > 1 else "unknown"

        # Find all PNG files in this directory
        png_files = list(to_label_dir.glob("*.png"))

        if png_files:
            print(f"\n{dataset_name}/{experiment_name}: {len(png_files)} images")
            stats[f"{dataset_name}/{experiment_name}"] = len(png_files)

        for png_file in png_files:
            # Create destination path
            if preserve_structure:
                # Create subdirectory structure: dataset/experiment/filename.png
                dest_path = dest_dir / dataset_name / experiment_name / png_file.name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Flat structure with prefixed filename
                # Format: dataset__experiment__original_filename.png
                prefix = f"{dataset_name}__{experiment_name}__"
                dest_path = dest_dir / (prefix + png_file.name)

            # Check if label file exists
            txt_file = png_file.with_suffix('.txt')
            has_label = txt_file.exists()

            if not dry_run:
                # Copy PNG file
                shutil.copy2(png_file, dest_path)
                copied_files.append(dest_path)

                # Copy label file if it exists
                if has_label:
                    dest_txt = dest_path.with_suffix('.txt')
                    shutil.copy2(txt_file, dest_txt)
                    stats['labeled_pairs'] += 1
                else:
                    stats['unlabeled'] += 1
            else:
                # Dry run - just report
                label_status = "✓ labeled" if has_label else "✗ unlabeled"
                print(f"  {png_file.name} -> {dest_path.name} ({label_status})")
                if has_label:
                    stats['labeled_pairs'] += 1
                else:
                    stats['unlabeled'] += 1

            stats['total_images'] += 1

    return stats, copied_files


def main():
    parser = argparse.ArgumentParser(
        description="Collect all to-be-labeled images from yolo_curation directories"
    )
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Root directory to search for to_label folders (default: config tools.collect_to_label_images.source_root)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory for collected images (default: config tools.collect_to_label_images.dest_dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be copied without actually copying",
    )
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve dataset/experiment folder structure (default: flat with prefixes)",
    )

    args = parser.parse_args()

    config_data = load_raw_config(args.config)
    tool_cfg = config_data.get("tools", {}).get("collect_to_label_images", {})
    if not isinstance(tool_cfg, dict):
        tool_cfg = {}

    source_root = args.source_root or Path(tool_cfg.get("source_root", ""))
    dest_dir = args.dest or Path(tool_cfg.get("dest_dir", ""))

    if not str(source_root):
        print("Error: Source root not configured. Provide --source-root or set tools.collect_to_label_images.source_root.")
        return 1
    if not str(dest_dir):
        print("Error: Destination not configured. Provide --dest or set tools.collect_to_label_images.dest_dir.")
        return 1

    args.source_root = source_root
    args.dest = dest_dir

    if not args.source_root.exists():
        print(f"Error: Source root does not exist: {args.source_root}")
        return 1

    print(f"Source: {args.source_root}")
    print(f"Destination: {args.dest}")
    if args.dry_run:
        print("DRY RUN - No files will be copied\n")
    print("=" * 80)

    stats, copied_files = collect_to_label_images(
        source_root=args.source_root,
        dest_dir=args.dest,
        dry_run=args.dry_run,
        preserve_structure=args.preserve_structure,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images: {stats['total_images']}")
    print(f"  Already labeled (PNG+TXT pairs): {stats['labeled_pairs']}")
    print(f"  Need labeling (PNG only): {stats['unlabeled']}")

    if not args.dry_run:
        print(f"\n✓ All images copied to: {args.dest}")
        print(f"\nNext steps:")
        print(f"  1. Label the {stats['unlabeled']} unlabeled images")
        print(f"  2. Save labels as .txt files next to each .png")
        print(f"  3. Copy labeled pairs back to their original to_label directories")
        print(f"  4. Re-run dataset curation to auto-organize them")
    else:
        print(f"\nRe-run without --dry-run to copy files")

    return 0


if __name__ == "__main__":
    exit(main())
