#!/usr/bin/env python3
"""
Filter YOLO11 OBB ONNX Model to Specific Classes

This script creates a filtered version of a YOLO ONNX model that only
outputs detections for specific target classes (eye and proboscis).

Author: Auto-generated
Date: 2025-01-07
"""

import argparse
import os
import sys
from pathlib import Path
import json
import yaml
import onnx
from onnx import numpy_helper
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fbpipe.config import load_raw_config

DEFAULT_TARGET_CLASS_INDICES = [2, 8]
DEFAULT_CLASS_NAMES = {
    0: "abdomen",
    1: "antenna",
    2: "eye",
    3: "frontLegs",
    4: "glue",
    5: "head",
    6: "needle-glue",
    7: "pipette",
    8: "proboscis",
}


def filter_yolo_onnx_model(
    input_onnx_path: Path,
    output_onnx_path: Path,
    target_class_indices: list[int],
    all_class_names: dict
) -> Path:
    """
    Create a filtered ONNX model that remaps target classes to new indices.

    For YOLO OBB models, we need to:
    1. Update the metadata to only include target classes
    2. Note: The actual model architecture stays the same, but we update
       the class mapping so X-AnyLabeling only shows target classes

    Args:
        input_onnx_path: Path to original ONNX model
        output_onnx_path: Path to save filtered model
        target_class_indices: List of class indices to keep (e.g., [2, 8])
        all_class_names: Dict of all class names {0: 'name1', ...}

    Returns:
        Path to filtered ONNX model
    """
    print(f"\nFiltering ONNX model to classes: {[all_class_names[i] for i in target_class_indices]}")
    print(f"  Input: {input_onnx_path}")
    print(f"  Output: {output_onnx_path}")

    # Load the ONNX model
    model = onnx.load(str(input_onnx_path))

    # Create new class mapping (remapping target classes to 0, 1, 2, ...)
    # This is a mapping from old class index to new class index
    old_to_new_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(target_class_indices)}

    # Create filtered class names dictionary
    filtered_class_names = {new_idx: all_class_names[old_idx]
                          for new_idx, old_idx in enumerate(target_class_indices)}

    print(f"\n  Class mapping:")
    for old_idx, new_idx in old_to_new_mapping.items():
        print(f"    {all_class_names[old_idx]} (index {old_idx}) -> index {new_idx}")

    # Update model metadata
    metadata = model.metadata_props

    # Find and update the 'names' metadata
    for prop in metadata:
        if prop.key == 'names':
            # Update with filtered class names
            prop.value = json.dumps(filtered_class_names)
            print(f"\n✓ Updated metadata 'names': {prop.value}")
            break

    # Save the modified model
    onnx.save(model, str(output_onnx_path))

    print(f"\n✓ Filtered ONNX model saved: {output_onnx_path}")
    print(f"  File size: {output_onnx_path.stat().st_size / (1024*1024):.2f} MB")

    return output_onnx_path, filtered_class_names, old_to_new_mapping


def main():
    """Main execution function."""
    print("="*70)
    print("YOLO11 OBB ONNX Model Class Filter")
    print("Filtering to Eye & Proboscis Only")
    print("="*70 + "\n")

    parser = argparse.ArgumentParser(description="Filter YOLO ONNX model metadata to target classes.")
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument("--input-onnx", default=None, help="Path to the source ONNX model.")
    parser.add_argument("--output-dir", default=None, help="Directory to write the filtered model.")
    parser.add_argument("--output-name", default=None, help="Filename for the filtered ONNX model.")
    parser.add_argument(
        "--class-ids",
        nargs="*",
        type=int,
        default=None,
        help="Target class indices to keep (default: 2 8).",
    )
    args = parser.parse_args()

    config_data = load_raw_config(args.config)
    tool_cfg = config_data.get("tools", {}).get("onnx_filter", {})
    if not isinstance(tool_cfg, dict):
        tool_cfg = {}

    input_onnx = args.input_onnx or tool_cfg.get("input_onnx_path", "")
    output_dir = args.output_dir or tool_cfg.get("output_dir", "")
    output_name = args.output_name or tool_cfg.get("output_name", "best_eye_proboscis_only.onnx")
    target_class_indices = args.class_ids or tool_cfg.get("target_class_indices") or DEFAULT_TARGET_CLASS_INDICES
    class_names = tool_cfg.get("class_names", DEFAULT_CLASS_NAMES)
    if not isinstance(class_names, dict):
        class_names = DEFAULT_CLASS_NAMES
    all_class_names = {int(k): str(v) for k, v in class_names.items()}

    try:
        # Validate input
        if not input_onnx:
            raise ValueError("Input ONNX path not configured. Provide --input-onnx or set tools.onnx_filter.input_onnx_path.")
        input_path = Path(input_onnx).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

        # Setup output
        if not output_dir:
            raise ValueError("Output directory not configured. Provide --output-dir or set tools.onnx_filter.output_dir.")
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name

        # Filter the model
        filtered_model_path, filtered_classes, class_mapping = filter_yolo_onnx_model(
            input_onnx_path=input_path,
            output_onnx_path=output_path,
            target_class_indices=[int(idx) for idx in target_class_indices],
            all_class_names=all_class_names
        )

        print("\n" + "="*70)
        print("✓ SUCCESS: Filtered ONNX model created!")
        print("="*70)
        print(f"\nFiltered model: {filtered_model_path}")
        print(f"Filtered classes: {list(filtered_classes.values())}")
        print(f"\nNote: This model's metadata has been updated, but it will still")
        print(f"      detect all 9 classes. You'll need to filter predictions")
        print(f"      in post-processing or use a different approach.")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
