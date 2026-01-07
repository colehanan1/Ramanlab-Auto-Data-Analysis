#!/usr/bin/env python3
"""
Auto-Label Images Using YOLO Model

Collects images from yolo_curation/to_label directories, runs YOLO inference,
and saves auto-labeled images with annotations to a target directory.

Usage:
    python auto_label_images.py
"""

import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml


def find_curation_images(data_root: Path):
    """Find all images in yolo_curation/to_label directories."""
    image_paths = []

    # Find all PNG images in to_label folders
    pattern = "**/yolo_curation/to_label/*.png"
    for img_path in data_root.glob(pattern):
        image_paths.append(img_path)

    return image_paths


def auto_label_images(
    model_path: str,
    data_root: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Auto-label images using YOLO model.

    Args:
        model_path: Path to YOLO model (.pt file)
        data_root: Root directory containing fly data
        output_dir: Directory to save auto-labeled images
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    """
    print("="*70)
    print("YOLO Auto-Labeling")
    print("="*70)

    # Load YOLO model
    print(f"\nLoading YOLO model: {model_path}")
    model = YOLO(model_path)
    print(f"✓ Model loaded: {model.model.names}")

    # Find images to label
    print(f"\nSearching for images in: {data_root}")
    image_paths = find_curation_images(data_root)
    print(f"✓ Found {len(image_paths)} images to auto-label")

    if not image_paths:
        print("\n⚠ No images found in yolo_curation/to_label directories!")
        print("Run curation first: python -m src.fbpipe.pipeline --config config.yaml curate_yolo_dataset")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")

    # Process images
    print(f"\nAuto-labeling {len(image_paths)} images...")
    labeled_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        # Run inference
        results = model(
            img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # Get the first result
        result = results[0]

        # Copy image to output directory
        output_img = output_dir / img_path.name
        shutil.copy2(img_path, output_img)

        # Save YOLO OBB annotation
        output_txt = output_dir / f"{img_path.stem}.txt"

        # Convert detections to YOLO OBB format
        # Format: class_id x_center y_center width height rotation_angle
        with open(output_txt, 'w') as f:
            if hasattr(result, 'obb') and result.obb is not None:
                # OBB detections
                for box in result.obb:
                    class_id = int(box.cls[0])
                    # OBB format: [x1, y1, x2, y2, x3, y3, x4, y4]
                    obb_coords = box.xyxyxyxy[0].cpu().numpy()

                    # Convert to normalized center format with rotation
                    img_width = result.orig_shape[1]
                    img_height = result.orig_shape[0]

                    # Calculate center point
                    x_center = obb_coords[::2].mean() / img_width
                    y_center = obb_coords[1::2].mean() / img_height

                    # Calculate width and height (approximate from bounding points)
                    x_coords = obb_coords[::2]
                    y_coords = obb_coords[1::2]
                    width = (x_coords.max() - x_coords.min()) / img_width
                    height = (y_coords.max() - y_coords.min()) / img_height

                    # Get rotation angle if available
                    rotation = 0.0
                    if hasattr(box, 'rotation'):
                        rotation = float(box.rotation)

                    # Write annotation
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {rotation:.6f}\n")

            elif hasattr(result, 'boxes') and result.boxes is not None:
                # Regular bounding boxes (fallback)
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (box.xywh[0][0] / result.orig_shape[1]).item()
                    y_center = (box.xywh[0][1] / result.orig_shape[0]).item()
                    width = (box.xywh[0][2] / result.orig_shape[1]).item()
                    height = (box.xywh[0][3] / result.orig_shape[0]).item()
                    rotation = 0.0

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {rotation:.6f}\n")

        labeled_count += 1

        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(image_paths)} images...")

    print(f"\n✓ Auto-labeling complete!")
    print(f"  Total images labeled: {labeled_count}")
    print(f"  Output directory: {output_dir}")
    print(f"\nFiles saved:")
    print(f"  - {labeled_count} PNG images")
    print(f"  - {labeled_count} TXT annotations (YOLO OBB format)")


def main():
    """Main execution."""
    # Configuration
    MODEL_PATH = "/home/ramanlab/Documents/cole/sam2/notebooks/YOLOProjectProboscisLegs/runs/obb/train5/weights/best.pt"
    DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys")
    OUTPUT_DIR = Path("/home/ramanlab/Documents/cole/model/auto_labelled_images")
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45

    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"✗ ERROR: Model not found at {MODEL_PATH}")
        print("\nPlease update MODEL_PATH in the script to point to your trained YOLO model.")
        return 1

    # Run auto-labeling
    try:
        auto_label_images(
            model_path=MODEL_PATH,
            data_root=DATA_ROOT,
            output_dir=OUTPUT_DIR,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
