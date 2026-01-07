#!/usr/bin/env python3
"""
Filter X-AnyLabeling JSON Annotations to Specific Classes

This script filters X-AnyLabeling annotation JSON files to only include
specific target classes (eye and proboscis), removing all other class detections.

Usage:
    python filter_xanylabeling_annotations.py <input_json> <output_json>

Or run in batch mode on a directory:
    python filter_xanylabeling_annotations.py <input_dir> <output_dir>

Author: Auto-generated
Date: 2025-01-07
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def filter_annotations(data: Dict[str, Any], target_classes: List[str]) -> Dict[str, Any]:
    """
    Filter annotation data to only include target classes.

    Args:
        data: X-AnyLabeling JSON data
        target_classes: List of class names to keep

    Returns:
        Filtered JSON data
    """
    if 'shapes' not in data:
        return data

    original_count = len(data['shapes'])

    # Filter shapes to only include target classes
    data['shapes'] = [
        shape for shape in data['shapes']
        if shape.get('label') in target_classes
    ]

    filtered_count = len(data['shapes'])
    removed_count = original_count - filtered_count

    print(f"  Filtered: {original_count} → {filtered_count} shapes ({removed_count} removed)")

    return data


def process_file(input_path: Path, output_path: Path, target_classes: List[str]) -> bool:
    """
    Process a single JSON file.

    Args:
        input_path: Input JSON file path
        output_path: Output JSON file path
        target_classes: List of class names to keep

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Processing: {input_path.name}")

        # Read JSON
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Filter annotations
        filtered_data = filter_annotations(data, target_classes)

        # Write filtered JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)

        print(f"  Saved to: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def process_directory(input_dir: Path, output_dir: Path, target_classes: List[str]):
    """
    Process all JSON files in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        target_classes: List of class names to keep
    """
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {input_dir}")
        return

    print(f"\nFound {len(json_files)} JSON files to process\n")

    success_count = 0
    for json_file in json_files:
        output_file = output_dir / json_file.name
        if process_file(json_file, output_file, target_classes):
            success_count += 1
        print()

    print(f"Successfully processed: {success_count}/{len(json_files)} files")


def main():
    """Main execution function."""
    TARGET_CLASSES = ['eye', 'proboscis']

    if len(sys.argv) < 3:
        print("Usage:")
        print("  Single file: python filter_xanylabeling_annotations.py <input.json> <output.json>")
        print("  Directory:   python filter_xanylabeling_annotations.py <input_dir> <output_dir>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print("="*70)
    print("X-AnyLabeling Annotation Filter")
    print(f"Target Classes: {TARGET_CLASSES}")
    print("="*70 + "\n")

    try:
        # Check if input is a file or directory
        if input_path.is_file():
            # Single file mode
            if not input_path.suffix == '.json':
                print(f"Error: Input file must be a JSON file: {input_path}")
                return 1

            success = process_file(input_path, output_path, TARGET_CLASSES)
            return 0 if success else 1

        elif input_path.is_dir():
            # Directory mode
            process_directory(input_path, output_path, TARGET_CLASSES)
            return 0

        else:
            print(f"Error: Input path does not exist: {input_path}")
            return 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
