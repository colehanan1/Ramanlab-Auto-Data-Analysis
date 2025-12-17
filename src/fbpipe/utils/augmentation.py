"""
Data augmentation utilities for YOLO dataset curation.

Provides functions to augment labeled images and their corresponding
YOLO format annotations (bounding boxes).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger("fbpipe.augmentation")


def parse_yolo_annotation(txt_path: Path, img_width: int, img_height: int) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Parse YOLO format annotation file.

    YOLO format: <class_id> <x_center> <y_center> <width> <height> [<angle>]
    where coordinates are normalized to [0, 1]

    Args:
        txt_path: Path to YOLO .txt annotation file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of tuples: (class_id, x_center, y_center, width, height, angle)
        Coordinates are denormalized to pixel values
    """
    annotations = []

    if not txt_path.exists():
        return annotations

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            angle = float(parts[5]) if len(parts) > 5 else 0.0  # OBB format includes angle

            annotations.append((class_id, x_center, y_center, width, height, angle))

    return annotations


def save_yolo_annotation(
    txt_path: Path,
    annotations: List[Tuple[int, float, float, float, float, float]],
    img_width: int,
    img_height: int
) -> None:
    """
    Save annotations in YOLO format.

    Args:
        txt_path: Output path for .txt file
        annotations: List of (class_id, x_center, y_center, width, height, angle)
        img_width: Image width in pixels
        img_height: Image height in pixels
    """
    with open(txt_path, "w") as f:
        for class_id, x_center, y_center, width, height, angle in annotations:
            # Normalize to [0, 1]
            x_norm = x_center / img_width
            y_norm = y_center / img_height
            w_norm = width / img_width
            h_norm = height / img_height

            # Clamp to valid range
            x_norm = np.clip(x_norm, 0, 1)
            y_norm = np.clip(y_norm, 0, 1)
            w_norm = np.clip(w_norm, 0, 1)
            h_norm = np.clip(h_norm, 0, 1)

            if angle != 0.0:
                # OBB format with angle
                f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {angle:.6f}\n")
            else:
                # Standard bbox format
                f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def augment_horizontal_flip(
    image: np.ndarray,
    annotations: List[Tuple[int, float, float, float, float, float]]
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float, float]]]:
    """
    Apply horizontal flip augmentation.

    Args:
        image: Input image (H, W, C)
        annotations: List of (class_id, x_center, y_center, width, height, angle)

    Returns:
        Flipped image and adjusted annotations
    """
    h, w = image.shape[:2]
    flipped_image = cv2.flip(image, 1)  # Horizontal flip

    flipped_annotations = []
    for class_id, x_center, y_center, width, height, angle in annotations:
        # Flip x coordinate
        new_x = w - x_center

        # Flip angle (for OBB)
        new_angle = -angle if angle != 0.0 else 0.0

        flipped_annotations.append((class_id, new_x, y_center, width, height, new_angle))

    return flipped_image, flipped_annotations


def augment_brightness_contrast(
    image: np.ndarray,
    annotations: List[Tuple[int, float, float, float, float, float]],
    brightness_delta: float = 10.0,
    contrast_delta: float = 0.1
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float, float]]]:
    """
    Apply brightness and contrast jitter augmentation.

    Args:
        image: Input image (H, W, C)
        annotations: List of annotations (unchanged by this augmentation)
        brightness_delta: Maximum brightness change (±pixels)
        contrast_delta: Maximum contrast multiplier change (±factor)

    Returns:
        Augmented image and annotations (annotations unchanged)
    """
    # Random brightness and contrast adjustments
    rng = np.random.RandomState()

    brightness = rng.uniform(-brightness_delta, brightness_delta)
    contrast = 1.0 + rng.uniform(-contrast_delta, contrast_delta)

    # Apply adjustments
    augmented = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    return augmented, annotations


def augment_rotation(
    image: np.ndarray,
    annotations: List[Tuple[int, float, float, float, float, float]],
    max_angle: float = 5.0
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float, float]]]:
    """
    Apply minor rotation augmentation.

    Args:
        image: Input image (H, W, C)
        annotations: List of (class_id, x_center, y_center, width, height, angle)
        max_angle: Maximum rotation angle in degrees (±)

    Returns:
        Rotated image and adjusted annotations
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Random rotation angle
    rng = np.random.RandomState()
    angle_deg = rng.uniform(-max_angle, max_angle)
    angle_rad = np.radians(angle_deg)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Rotate annotations
    rotated_annotations = []
    for class_id, x_center, y_center, width, height, obb_angle in annotations:
        # Rotate center point
        point = np.array([[x_center, y_center]])
        rotated_point = cv2.transform(point.reshape(1, 1, 2), M).reshape(2)
        new_x, new_y = rotated_point

        # Update OBB angle
        new_obb_angle = obb_angle + angle_deg if obb_angle != 0.0 else 0.0

        rotated_annotations.append((class_id, new_x, new_y, width, height, new_obb_angle))

    return rotated_image, rotated_annotations


def augment_labeled_frames(
    labeled_dir: Path,
    augmented_dir: Path,
    strategies: List[str],
    multiplier: int = 2
) -> int:
    """
    Augment all labeled frames in a directory.

    Args:
        labeled_dir: Directory containing labeled PNG+TXT pairs
        augmented_dir: Output directory for augmented images
        strategies: List of augmentation strategies to apply
        multiplier: Target dataset size multiplier

    Returns:
        Number of augmented images created
    """
    augmented_dir.mkdir(parents=True, exist_ok=True)

    png_files = list(labeled_dir.glob("*.png"))
    augmented_count = 0

    strategy_functions = {
        "horizontal_flip": augment_horizontal_flip,
        "brightness_contrast_jitter": lambda img, ann: augment_brightness_contrast(img, ann, 10.0, 0.1),
        "minor_rotation": lambda img, ann: augment_rotation(img, ann, 5.0),
    }

    for png_file in png_files:
        txt_file = png_file.with_suffix(".txt")

        if not txt_file.exists():
            log.warning(f"No annotation file for {png_file.name}, skipping")
            continue

        # Read image
        image = cv2.imread(str(png_file))
        if image is None:
            log.warning(f"Failed to read {png_file.name}, skipping")
            continue

        h, w = image.shape[:2]

        # Parse annotations
        annotations = parse_yolo_annotation(txt_file, w, h)

        # Apply each augmentation strategy
        for strategy_name in strategies:
            if strategy_name not in strategy_functions:
                log.warning(f"Unknown augmentation strategy: {strategy_name}")
                continue

            aug_func = strategy_functions[strategy_name]
            aug_image, aug_annotations = aug_func(image, annotations)

            # Save augmented image
            aug_png_name = f"{png_file.stem}_{strategy_name}.png"
            aug_txt_name = f"{png_file.stem}_{strategy_name}.txt"

            aug_png_path = augmented_dir / aug_png_name
            aug_txt_path = augmented_dir / aug_txt_name

            cv2.imwrite(str(aug_png_path), aug_image)
            save_yolo_annotation(aug_txt_path, aug_annotations, w, h)

            augmented_count += 1

        # Stop if we've reached the target multiplier
        if augmented_count >= len(png_files) * (multiplier - 1):
            break

    log.info(f"Created {augmented_count} augmented images in {augmented_dir}")
    return augmented_count
