#!/usr/bin/env python3
"""
Generate X-AnyLabeling YAML Configuration from YOLO11 OBB Model

This script creates an X-AnyLabeling configuration file from a trained YOLO11
Oriented Bounding Box (OBB) model, extracting only specific classes.

Author: Auto-generated
Date: 2025-01-07
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml


def validate_model_path(model_path: str) -> Path:
    """
    Validate that the model file exists and is readable.

    Args:
        model_path: Path to the YOLO model file

    Returns:
        Path object if valid

    Raises:
        FileNotFoundError: If model file doesn't exist
        PermissionError: If model file is not readable
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not model_path.is_file():
        raise ValueError(f"Model path is not a file: {model_path}")

    if not os.access(model_path, os.R_OK):
        raise PermissionError(f"Model file is not readable: {model_path}")

    print(f"✓ Model file validated: {model_path}")
    return model_path


def load_yolo_model(model_path: Path):
    """
    Load YOLO model and extract class information.

    Args:
        model_path: Path to the YOLO model file

    Returns:
        YOLO model object

    Raises:
        ImportError: If ultralytics is not installed
        Exception: If model loading fails
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics package not found. Install it with: pip install ultralytics"
        )

    try:
        print(f"Loading YOLO model from: {model_path}")
        model = YOLO(str(model_path))
        print(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        raise Exception(f"Failed to load YOLO model: {e}")


def extract_class_names(model, target_class_indices: list[int]) -> tuple[list[str], list[str]]:
    """
    Extract all class names from the model and identify target classes.

    Args:
        model: Loaded YOLO model
        target_class_indices: List of target class indices (for display naming)

    Returns:
        Tuple of (all_class_names_list, target_class_names_list)

    Raises:
        ValueError: If class indices are invalid
    """
    # Get class names from model
    if hasattr(model, 'names'):
        all_class_names = model.names
    elif hasattr(model, 'model') and hasattr(model.model, 'names'):
        all_class_names = model.model.names
    else:
        raise ValueError("Could not extract class names from model")

    print(f"\nTotal classes in model: {len(all_class_names)}")
    print(f"All class names: {all_class_names}")

    # Validate indices
    max_index = len(all_class_names) - 1
    for idx in target_class_indices:
        if idx < 0 or idx > max_index:
            raise ValueError(
                f"Class index {idx} is out of range [0, {max_index}]"
            )

    # Convert dict to ordered list (preserving class index order)
    if isinstance(all_class_names, dict):
        class_list = [all_class_names[i] for i in sorted(all_class_names.keys())]
    else:
        class_list = list(all_class_names)

    # Extract target classes for display naming
    target_classes = [all_class_names[idx] for idx in target_class_indices]

    print(f"\n✓ All classes (for YAML): {class_list}")
    print(f"✓ Target classes (for naming): {target_classes}")

    return class_list, target_classes


def export_model_to_onnx(model, output_dir: Path, model_name: str = "best") -> Path:
    """
    Export YOLO model to ONNX format for X-AnyLabeling compatibility.

    Args:
        model: Loaded YOLO model
        output_dir: Directory to save the ONNX file
        model_name: Name for the exported model

    Returns:
        Path to the exported ONNX file

    Raises:
        Exception: If export fails
    """
    print(f"\nExporting model to ONNX format...")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    try:
        # Export with OBB format
        onnx_path = output_dir / f"{model_name}.onnx"

        # Remove existing ONNX file if present
        if onnx_path.exists():
            print(f"  Removing existing ONNX file: {onnx_path}")
            onnx_path.unlink()

        # Export model
        print(f"  Exporting to: {onnx_path}")
        export_result = model.export(
            format='onnx',
            imgsz=640,  # Standard YOLO image size
            half=False,  # Use FP32 for better compatibility
            simplify=True,  # Simplify the ONNX model
            opset=12,  # ONNX opset version
        )

        # The export method returns the path to the exported file
        # It's typically saved in the same directory as the source model
        # We need to move it to our output directory if it's not already there
        if isinstance(export_result, str):
            exported_path = Path(export_result)
        else:
            # Try to find the exported file
            source_model_dir = Path(model.ckpt_path).parent
            exported_path = source_model_dir / f"{Path(model.ckpt_path).stem}.onnx"

        if exported_path != onnx_path and exported_path.exists():
            import shutil
            print(f"  Moving {exported_path} to {onnx_path}")
            shutil.move(str(exported_path), str(onnx_path))

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX export completed but file not found at {onnx_path}")

        print(f"✓ ONNX model exported successfully: {onnx_path}")
        print(f"  File size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")

        return onnx_path

    except Exception as e:
        raise Exception(f"Failed to export model to ONNX: {e}")


def generate_yaml_config(
    onnx_model_path: Path,
    all_class_names: list[str],
    target_class_names: list[str],
    output_dir: Path,
    iou_threshold: float = 0.45,
    conf_threshold: float = 0.25
) -> Path:
    """
    Generate X-AnyLabeling YAML configuration file.

    Args:
        onnx_model_path: Path to the ONNX model file
        all_class_names: List of ALL class names (must match model output indices)
        target_class_names: List of target class names (for config naming)
        output_dir: Directory to save the YAML file
        iou_threshold: IoU threshold for NMS
        conf_threshold: Confidence threshold for detections

    Returns:
        Path to the generated YAML file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory ready: {output_dir}")

    # Generate configuration name using target classes
    date_str = datetime.now().strftime("%Y%m%d")
    config_name = f"yolo11-obb-{'-'.join(target_class_names)}-r{date_str}"
    display_name = f"YOLO11 OBB - {' & '.join([c.capitalize() for c in target_class_names])}"

    # Create YAML configuration with ALL classes
    # This is critical: X-AnyLabeling needs all classes in order to map
    # model output class indices correctly
    # Note: Using 'yolov8_obb' type as X-AnyLabeling uses YOLOv8 handler for YOLO11 OBB
    config = {
        'type': 'yolov8_obb',
        'name': config_name,
        'provider': 'Ultralytics',
        'display_name': display_name,
        'model_path': str(onnx_model_path.absolute()),
        'iou_threshold': iou_threshold,
        'conf_threshold': conf_threshold,
        'filter_classes': target_class_names,  # Try to filter to only target classes
        'classes': all_class_names  # All classes, not just target ones
    }

    # Save YAML file
    yaml_path = output_dir / f"{config_name}.yaml"

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ YAML configuration generated: {yaml_path}")
    print(f"  Note: YAML contains all {len(all_class_names)} classes for proper index mapping")

    return yaml_path


def display_config_preview(yaml_path: Path):
    """
    Display the generated configuration file.

    Args:
        yaml_path: Path to the YAML file
    """
    print("\n" + "="*70)
    print("Generated Configuration Preview:")
    print("="*70)

    with open(yaml_path, 'r') as f:
        content = f.read()
        print(content)

    print("="*70)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fbpipe.config import load_raw_config


def main():
    """Main execution function."""
    print("="*70)
    print("X-AnyLabeling YAML Configuration Generator")
    print("YOLO11 OBB Model - Eye & Proboscis Classes")
    print("="*70 + "\n")

    parser = argparse.ArgumentParser(description="Generate X-AnyLabeling YAML config.")
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument("--model-path", default=None, help="Path to the YOLO model file.")
    parser.add_argument("--output-dir", default=None, help="Directory to save the ONNX+YAML outputs.")
    parser.add_argument(
        "--class-ids",
        nargs="*",
        type=int,
        default=None,
        help="Target class indices to include (default: 2 8).",
    )
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold for NMS.")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Confidence threshold.")
    args = parser.parse_args()

    config_data = load_raw_config(args.config)
    tool_cfg = config_data.get("tools", {}).get("xanylabeling", {})
    if not isinstance(tool_cfg, dict):
        tool_cfg = {}

    model_path_value = args.model_path or tool_cfg.get("model_path") or config_data.get("model_path", "")
    output_dir_value = args.output_dir or tool_cfg.get("output_dir", "")
    target_class_indices = args.class_ids or tool_cfg.get("target_class_indices", [2, 8])
    iou_threshold = args.iou_threshold or tool_cfg.get("iou_threshold", 0.45)
    conf_threshold = args.conf_threshold or tool_cfg.get("conf_threshold", 0.25)

    try:
        # Step 1: Validate model path
        print("Step 1: Validating model path...")
        if not model_path_value:
            raise ValueError(
                "Model path not configured. Provide --model-path or set tools.xanylabeling.model_path."
            )
        model_path = validate_model_path(model_path_value)

        # Step 2: Load YOLO model
        print("\nStep 2: Loading YOLO model...")
        model = load_yolo_model(model_path)

        # Step 3: Extract class names
        print("\nStep 3: Extracting class names...")
        all_class_names, target_class_names = extract_class_names(
            model,
            [int(idx) for idx in target_class_indices],
        )

        # Step 4: Export model to ONNX format
        print("\nStep 4: Exporting model to ONNX format...")
        if not output_dir_value:
            raise ValueError(
                "Output directory not configured. Provide --output-dir or set tools.xanylabeling.output_dir."
            )
        output_dir = Path(output_dir_value)
        onnx_model_path = export_model_to_onnx(
            model=model,
            output_dir=output_dir,
            model_name="best"
        )

        # Step 5: Generate YAML configuration
        print("\nStep 5: Generating YAML configuration...")
        yaml_path = generate_yaml_config(
            onnx_model_path=onnx_model_path,
            all_class_names=all_class_names,
            target_class_names=target_class_names,
            output_dir=output_dir,
            iou_threshold=float(iou_threshold),
            conf_threshold=float(conf_threshold),
        )

        # Step 6: Display preview
        display_config_preview(yaml_path)

        print("\n✓ SUCCESS: X-AnyLabeling configuration generated successfully!")
        print(f"\nGenerated files:")
        print(f"  ONNX Model: {onnx_model_path}")
        print(f"  YAML Config: {yaml_path}")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
