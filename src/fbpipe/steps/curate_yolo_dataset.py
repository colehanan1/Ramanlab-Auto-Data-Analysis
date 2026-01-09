"""
YOLO Dataset Curation Module

Identifies problematic tracking videos from YOLO inference results and extracts
frames for manual labeling to improve the training dataset.

This module:
1. Computes quality metrics from YOLO tracking data (jitter, missing frames)
2. Flags videos with poor tracking quality
3. Extracts frames stratified by quality regions
4. Manages labeling workflow and data augmentation
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ..config import Settings, get_main_directories
from ..utils.columns import PROBOSCIS_X_COL, PROBOSCIS_Y_COL
from ..utils.augmentation import augment_labeled_frames

log = logging.getLogger("fbpipe.curate_yolo")


def compute_jitter(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """
    Compute frame-to-frame spatial jitter (pixel movement).

    Args:
        df: DataFrame with tracking coordinates
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates

    Returns:
        Series of jitter values (Euclidean distance between consecutive frames)
    """
    x_vals = pd.to_numeric(df[x_col], errors="coerce")
    y_vals = pd.to_numeric(df[y_col], errors="coerce")

    dx = x_vals.diff().fillna(0)
    dy = y_vals.diff().fillna(0)

    jitter = np.sqrt(dx**2 + dy**2)
    return jitter


def compute_quality_metrics(
    df: pd.DataFrame,
    video_path: Path,
    quality_thresholds: Dict,
) -> Dict:
    """
    Compute per-video YOLO tracking quality metrics.

    Args:
        df: DataFrame with tracking data
        video_path: Path to video file
        quality_thresholds: Dict with threshold configuration

    Returns:
        Dictionary with quality metrics
    """
    # Get proboscis tracking columns
    prob_x = df.get(PROBOSCIS_X_COL, pd.Series(dtype=float))
    prob_y = df.get(PROBOSCIS_Y_COL, pd.Series(dtype=float))

    # Convert to numeric
    prob_x = pd.to_numeric(prob_x, errors="coerce")
    prob_y = pd.to_numeric(prob_y, errors="coerce")

    # Compute missing frame percentage
    total_frames = len(df)
    missing_frames = prob_x.isna().sum()
    pct_missing = missing_frames / total_frames if total_frames > 0 else 1.0

    # Compute spatial jitter
    jitter = compute_jitter(df, PROBOSCIS_X_COL, PROBOSCIS_Y_COL)
    valid_jitter = jitter[jitter > 0]  # Exclude zero values (no movement or missing frames)

    metrics = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "total_frames": total_frames,
        "missing_frames": int(missing_frames),
        "pct_missing": float(pct_missing),
        "median_jitter_px": float(valid_jitter.median()) if len(valid_jitter) > 0 else 0.0,
        "max_jitter_px": float(valid_jitter.max()) if len(valid_jitter) > 0 else 0.0,
        "mean_jitter_px": float(valid_jitter.mean()) if len(valid_jitter) > 0 else 0.0,
    }

    return metrics


def is_bad_tracking(metrics: Dict, quality_thresholds: Dict) -> bool:
    """
    Determine if video needs re-labeling based on quality metrics.

    Args:
        metrics: Quality metrics dictionary
        quality_thresholds: Threshold configuration

    Returns:
        True if video should be flagged for curation
    """
    # Check missing frame threshold
    if metrics["pct_missing"] > quality_thresholds["max_missing_pct"]:
        return True

    # Check jitter threshold
    if metrics["median_jitter_px"] > quality_thresholds["max_jitter_px"]:
        return True

    return False


def extract_frames_stratified(
    video_path: Path,
    df: pd.DataFrame,
    output_dir: Path,
    target_count: int = 10,
    stratification: Optional[Dict] = None,
) -> List[Path]:
    """
    Extract frames from video using stratified sampling based on tracking quality.

    Extracts frames from three regions:
    - High quality: frames with valid tracking and low jitter (seed data)
    - Low quality: frames with missing/poor tracking (problem cases)
    - Boundary: frames with moderate tracking quality (edge cases)

    Args:
        video_path: Path to video file
        df: DataFrame with tracking data
        output_dir: Directory to save extracted frames
        target_count: Number of frames to extract
        stratification: Dict with allocation ratios (high_quality, low_quality, boundary)

    Returns:
        List of paths to extracted frame images
    """
    if stratification is None:
        stratification = {
            "high_quality": 0.30,
            "low_quality": 0.50,
            "boundary": 0.20,
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute jitter for quality categorization
    jitter = compute_jitter(df, PROBOSCIS_X_COL, PROBOSCIS_Y_COL)
    prob_x = pd.to_numeric(df.get(PROBOSCIS_X_COL, pd.Series(dtype=float)), errors="coerce")

    # Categorize frames
    high_quality_mask = (~prob_x.isna()) & (jitter < 20.0)  # Valid tracking, low jitter
    low_quality_mask = prob_x.isna() | (jitter > 50.0)      # Missing or high jitter
    boundary_mask = (~high_quality_mask) & (~low_quality_mask)

    high_quality_frames = df[high_quality_mask]["frame"].values
    low_quality_frames = df[low_quality_mask]["frame"].values
    boundary_frames = df[boundary_mask]["frame"].values

    # Calculate sample counts
    high_count = int(target_count * stratification["high_quality"])
    low_count = int(target_count * stratification["low_quality"])
    boundary_count = int(target_count * stratification["boundary"])

    # Sample frames (deterministic for reproducibility)
    rng = np.random.RandomState(42)

    selected_frames = []

    if len(high_quality_frames) > 0:
        sample_high = rng.choice(
            high_quality_frames,
            size=min(high_count, len(high_quality_frames)),
            replace=False
        )
        selected_frames.extend([(int(f), "high_quality") for f in sample_high])

    if len(low_quality_frames) > 0:
        sample_low = rng.choice(
            low_quality_frames,
            size=min(low_count, len(low_quality_frames)),
            replace=False
        )
        selected_frames.extend([(int(f), "low_quality") for f in sample_low])

    if len(boundary_frames) > 0:
        sample_boundary = rng.choice(
            boundary_frames,
            size=min(boundary_count, len(boundary_frames)),
            replace=False
        )
        selected_frames.extend([(int(f), "boundary") for f in sample_boundary])

    # Extract frames from video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return []

    extracted_paths = []
    video_stem = video_path.stem

    for frame_idx, category in sorted(selected_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            log.warning(f"Failed to read frame {frame_idx} from {video_path.name}")
            continue

        # Save frame with metadata in filename
        frame_filename = f"{video_stem}_frame_{frame_idx:06d}_{category}.png"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        extracted_paths.append(frame_path)

    cap.release()

    log.info(f"Extracted {len(extracted_paths)} frames from {video_path.name}")
    return extracted_paths


def move_labeled_frames(curation_dir: Path) -> int:
    """
    Move frames with annotations from to_label/ to labeled/ folder.

    Detects PNG+TXT pairs and moves them together.

    Args:
        curation_dir: Root curation directory

    Returns:
        Number of frames moved
    """
    to_label_dir = curation_dir / "to_label"
    labeled_dir = curation_dir / "labeled"

    if not to_label_dir.exists():
        return 0

    labeled_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for png_file in to_label_dir.glob("*.png"):
        txt_file = png_file.with_suffix(".txt")

        if txt_file.exists():
            # Move both files
            shutil.move(str(png_file), str(labeled_dir / png_file.name))
            shutil.move(str(txt_file), str(labeled_dir / txt_file.name))
            moved_count += 1
            log.info(f"Labeled: {png_file.name}")

    return moved_count


def create_curation_manifest(
    fly_dir: Path,
    flagged_videos: List[Dict],
    extracted_frames: List[Path],
    config_summary: Dict,
) -> Dict:
    """
    Create manifest for curation state tracking.

    Args:
        fly_dir: Fly directory
        flagged_videos: List of flagged video metadata
        extracted_frames: List of extracted frame paths
        config_summary: Configuration summary

    Returns:
        Manifest dictionary
    """
    curation_dir = fly_dir / "yolo_curation"

    to_label_count = len(list((curation_dir / "to_label").glob("*.png"))) if (curation_dir / "to_label").exists() else 0
    labeled_count = len(list((curation_dir / "labeled").glob("*.png"))) if (curation_dir / "labeled").exists() else 0

    manifest = {
        "version": "1.0",
        "timestamp": time.time(),
        "fly_dir": str(fly_dir),
        "flagged_videos_count": len(flagged_videos),
        "flagged_videos": flagged_videos,
        "extraction_state": {
            "extracted_count": len(extracted_frames),
            "to_label_count": to_label_count,
            "labeled_count": labeled_count,
        },
        "config_summary": config_summary,
    }

    return manifest


def get_next_batch_number(dest_dir: Path) -> int:
    """
    Determine the next batch number by checking existing batch directories.

    Args:
        dest_dir: Destination directory containing batch folders

    Returns:
        Next available batch number
    """
    if not dest_dir.exists():
        return 1

    existing_batches = [
        d for d in dest_dir.iterdir()
        if d.is_dir() and d.name.startswith("batch_")
    ]

    if not existing_batches:
        return 1

    # Extract batch numbers and find max
    batch_numbers = []
    for batch_dir in existing_batches:
        try:
            num = int(batch_dir.name.replace("batch_", ""))
            batch_numbers.append(num)
        except ValueError:
            continue

    return max(batch_numbers) + 1 if batch_numbers else 1


def collect_to_label_images_batch(
    source_root: Path,
    dest_dir: Path,
    batch_number: Optional[int] = None,
) -> Dict:
    """
    Collect all images from yolo_curation/to_label directories into a new batch folder.

    Args:
        source_root: Root directory to search for to_label folders
        dest_dir: Base destination directory for batched images
        batch_number: Optional specific batch number, otherwise auto-increments

    Returns:
        Dictionary with collection statistics
    """
    from collections import defaultdict

    stats = defaultdict(int)

    # Determine batch number
    if batch_number is None:
        batch_number = get_next_batch_number(dest_dir)

    batch_dir = dest_dir / f"batch_{batch_number}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[BATCH COLLECTION] Collecting to_label images into: {batch_dir}")

    # Find all to_label directories
    to_label_dirs = list(source_root.glob("**/yolo_curation/to_label"))

    if not to_label_dirs:
        log.info("[BATCH COLLECTION] No to_label directories found")
        return stats

    log.info(f"[BATCH COLLECTION] Found {len(to_label_dirs)} to_label directories")

    copied_files = []

    for to_label_dir in sorted(to_label_dirs):
        # Extract dataset and experiment names for context
        parts = to_label_dir.relative_to(source_root).parts
        dataset_name = parts[0] if len(parts) > 0 else "unknown"
        experiment_name = parts[1] if len(parts) > 1 else "unknown"

        # Find all PNG files in this directory (unlabeled only)
        png_files = []
        for png_file in to_label_dir.glob("*.png"):
            txt_file = png_file.with_suffix('.txt')
            # Only copy unlabeled images
            if not txt_file.exists():
                png_files.append(png_file)

        if not png_files:
            continue

        log.info(f"[BATCH COLLECTION] {dataset_name}/{experiment_name}: {len(png_files)} unlabeled images")
        stats[f"{dataset_name}/{experiment_name}"] = len(png_files)

        for png_file in png_files:
            # Create prefixed filename: dataset__experiment__original_filename.png
            prefix = f"{dataset_name}__{experiment_name}__"
            dest_path = batch_dir / (prefix + png_file.name)

            # Copy PNG file
            shutil.copy2(png_file, dest_path)
            copied_files.append(dest_path)
            stats['total_images'] += 1

    # Write batch manifest
    manifest = {
        "batch_number": batch_number,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(source_root),
        "total_images": stats['total_images'],
        "source_breakdown": {k: v for k, v in stats.items() if k != 'total_images'},
    }

    manifest_path = batch_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"[BATCH COLLECTION] Collected {stats['total_images']} images → {batch_dir}")
    log.info(f"[BATCH COLLECTION] Batch manifest written → {manifest_path}")

    return stats


def main(cfg: Settings) -> None:
    """
    Main entry point for YOLO dataset curation.

    Scans YOLO inference outputs, identifies problematic videos,
    and extracts frames for manual labeling.

    Args:
        cfg: Pipeline configuration
    """
    # Check if curation is enabled
    if not hasattr(cfg, "yolo_curation") or not getattr(cfg.yolo_curation, "enabled", False):
        log.info("[CURATION] YOLO curation is disabled in config")
        return

    curation_cfg = cfg.yolo_curation
    quality_thresholds = {
        "max_jitter_px": curation_cfg.quality_thresholds.max_jitter_px,
        "max_missing_pct": curation_cfg.quality_thresholds.max_missing_pct,
    }

    target_frames_cfg = curation_cfg.target_frames

    log.info("[CURATION] Starting YOLO dataset curation")
    log.info(f"[CURATION] Quality thresholds: jitter<={quality_thresholds['max_jitter_px']}px, "
             f"missing<={quality_thresholds['max_missing_pct']*100}%")

    if curation_cfg.video_source_dirs:
        log.info(f"[CURATION] Searching for videos in {len(curation_cfg.video_source_dirs)} additional source directories")

    roots = get_main_directories(cfg)

    # Load global cache to track processed flies
    cache_root = (
        Path(cfg.cache_dir).expanduser()
        if getattr(cfg, "cache_dir", "")
        else Path(__file__).resolve().parents[3] / "cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / "yolo_curation_cache.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {"processed_flies": {}, "last_run": None}

    log.info(f"[CURATION] Cache loaded: {len(cache['processed_flies'])} flies previously processed")

    all_flagged_videos = []
    total_extracted = 0
    skipped_count = 0
    processed_count = 0

    for root in roots:
        log.info(f"[CURATION] Processing root directory: {root}")

        for fly_dir in [p for p in root.iterdir() if p.is_dir()]:
            # Check if this fly was already processed
            fly_key = str(fly_dir.resolve())
            if fly_key in cache["processed_flies"]:
                skipped_count += 1
                log.info(f"[CURATION] Skipping {fly_dir.name} (already processed on {cache['processed_flies'][fly_key]['timestamp']})")
                continue

            log.info(f"[CURATION] Inspecting fly directory: {fly_dir.name}")

            # Find YOLO output CSVs
            distance_csvs = list(fly_dir.glob("*/*_fly*_distances.csv"))

            if not distance_csvs:
                log.info(f"[CURATION] No YOLO distance CSVs found in {fly_dir.name}")
                continue

            fly_flagged = []
            fly_metrics = []

            for csv_path in distance_csvs:
                # Read tracking data
                try:
                    df = pd.read_csv(csv_path)
                except Exception as exc:
                    log.warning(f"Failed to read {csv_path.name}: {exc}")
                    continue

                # Find corresponding video
                # CSV pattern: {fly_dir}/RMS_calculations/{prefix}_{trial_folder}_fly{N}_distances.csv
                # Video pattern in secure storage: {fly_dir}/output_{trial_folder}_TIMESTAMP.mp4
                # Example CSV: updated_november_07_batch_1_testing_2_fly1_distances.csv
                # Example video: output_november_07_batch_1_testing_2_20251107_143551.mp4

                # Extract trial folder name from CSV filename
                csv_stem = csv_path.stem  # Remove .csv extension
                # Remove _fly{N}_distances suffix using regex
                match = re.match(r'(.+?)_fly\d+_distances', csv_stem)
                if match:
                    potential_name = match.group(1)
                    # Remove "updated_" prefix if present
                    folder_name = potential_name.replace("updated_", "", 1) if potential_name.startswith("updated_") else potential_name
                else:
                    # Fallback to parent directory name (old behavior)
                    folder_name = csv_path.parent.name

                # Build search patterns
                search_patterns = [
                    f"output_{folder_name}_*.mp4",  # With output_ prefix and timestamp
                    f"{folder_name}.mp4",            # Without prefix
                    f"{folder_name}_preprocessed.mp4",
                    f"pre_{folder_name}.mp4",
                ]

                video_path = None
                search_locations = []

                # First try the same directory as CSV
                for pattern in search_patterns:
                    matches = list(fly_dir.glob(pattern))
                    if matches:
                        video_path = matches[0]
                        break

                # If not found, search in configured video source directories
                if video_path is None:
                    for source_dir in curation_cfg.video_source_dirs:
                        source_path = Path(source_dir).expanduser().resolve()

                        # Map fly directory: extract just the fly name (e.g., "september_16_fly_1")
                        fly_name = fly_dir.name
                        mapped_fly_dir = source_path / fly_name

                        search_locations.append(mapped_fly_dir)

                        # Try each pattern with glob
                        for pattern in search_patterns:
                            matches = list(mapped_fly_dir.glob(pattern))
                            if matches:
                                video_path = matches[0]
                                log.info(f"[CURATION] Found video: {video_path}")
                                break

                        if video_path:
                            break

                if video_path is None:
                    log.warning(f"Cannot find video for {csv_path.name}")
                    log.warning(f"  Folder name: {folder_name}")
                    log.warning(f"  Expected pattern: output_{folder_name}_*.mp4")
                    log.warning(f"  Searched {len(search_locations)} locations: {[str(loc) for loc in search_locations[:3]]}")
                    continue

                # Compute quality metrics
                metrics = compute_quality_metrics(df, video_path, quality_thresholds)
                fly_metrics.append(metrics)

                # Check if video should be flagged
                if is_bad_tracking(metrics, quality_thresholds):
                    log.info(f"[CURATION] Flagged: {video_path.name} "
                            f"(missing={metrics['pct_missing']*100:.1f}%, "
                            f"jitter={metrics['median_jitter_px']:.1f}px)")
                    fly_flagged.append({
                        "video_path": str(video_path),
                        "csv_path": str(csv_path),
                        "metrics": metrics,
                    })

            if not fly_flagged:
                log.info(f"[CURATION] No problematic videos found in {fly_dir.name}")
                continue

            # Create curation directory
            curation_dir = fly_dir / curation_cfg.output_dir
            curation_dir.mkdir(parents=True, exist_ok=True)
            to_label_dir = curation_dir / "to_label"
            to_label_dir.mkdir(parents=True, exist_ok=True)

            # Extract frames from flagged videos
            extracted_frames = []
            for flagged in fly_flagged[:5]:  # Limit to 5 videos per fly to avoid explosion
                video_path = Path(flagged["video_path"])
                csv_path = Path(flagged["csv_path"])

                df = pd.read_csv(csv_path)

                frames_per_video = target_frames_cfg.per_video
                frames = extract_frames_stratified(
                    video_path,
                    df,
                    to_label_dir,
                    target_count=frames_per_video,
                    stratification=target_frames_cfg.stratification,
                )
                extracted_frames.extend(frames)

            total_extracted += len(extracted_frames)
            all_flagged_videos.extend(fly_flagged)

            # Save quality metrics
            metrics_path = curation_dir / "quality_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(fly_metrics, f, indent=2)
            log.info(f"[CURATION] Wrote quality metrics → {metrics_path}")

            # Save flagged videos list
            flagged_path = curation_dir / "flagged_videos.json"
            with open(flagged_path, "w", encoding="utf-8") as f:
                json.dump(fly_flagged, f, indent=2)
            log.info(f"[CURATION] Wrote flagged videos → {flagged_path}")

            # Create manifest
            config_summary = {
                "quality_thresholds": quality_thresholds,
                "target_frames": {
                    "per_video": target_frames_cfg.per_video,
                    "stratification": target_frames_cfg.stratification,
                },
            }
            manifest = create_curation_manifest(
                fly_dir,
                fly_flagged,
                extracted_frames,
                config_summary
            )
            manifest_path = curation_dir / "curation_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            log.info(f"[CURATION] Wrote manifest → {manifest_path}")

            # Mark this fly as processed in cache
            cache["processed_flies"][fly_key] = {
                "fly_name": fly_dir.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "flagged_count": len(fly_flagged),
                "extracted_frames": len(extracted_frames)
            }
            processed_count += 1

            # Move any already-labeled frames
            moved = move_labeled_frames(curation_dir)
            if moved > 0:
                log.info(f"[CURATION] Moved {moved} labeled frames to labeled/")

            # Apply augmentation if enabled
            if curation_cfg.augmentation.enabled:
                labeled_dir = curation_dir / "labeled"
                augmented_dir = curation_dir / "augmented"

                if labeled_dir.exists() and len(list(labeled_dir.glob("*.png"))) > 0:
                    log.info(f"[CURATION] Applying data augmentation to labeled frames")
                    aug_count = augment_labeled_frames(
                        labeled_dir,
                        augmented_dir,
                        strategies=list(curation_cfg.augmentation.strategies),
                        multiplier=curation_cfg.augmentation.multiplier
                    )
                    log.info(f"[CURATION] Created {aug_count} augmented frames → {augmented_dir}")

    # Save updated cache
    cache["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
    log.info(f"[CURATION] Cache saved: {len(cache['processed_flies'])} flies total")

    # Summary
    log.info(f"[CURATION] === Summary ===")
    log.info(f"[CURATION] Flies skipped (already processed): {skipped_count}")
    log.info(f"[CURATION] Flies newly processed: {processed_count}")
    log.info(f"[CURATION] Total flagged videos: {len(all_flagged_videos)}")
    log.info(f"[CURATION] Total frames extracted: {total_extracted}")
    log.info(f"[CURATION] Frames are ready for labeling in */yolo_curation/to_label/")
    log.info(f"[CURATION] After labeling, place .txt annotation files next to .png files")
    log.info(f"[CURATION] Run the pipeline again to auto-move labeled frames and apply augmentation")
    log.info(f"[CURATION] To reset cache and re-process all flies: rm {cache_file}")

    # Collect all to_label images into batched directory for centralized labeling
    if curation_cfg.video_source_dirs:
        repo_root = Path(__file__).resolve().parents[3]
        batch_dest = Path(
            os.getenv("CURATION_BATCH_DEST", str(repo_root / "data" / "to-be-labelled"))
        ).expanduser().resolve()

        # Find common parent directory to search once instead of per-subdirectory
        source_paths = [Path(d).expanduser().resolve() for d in curation_cfg.video_source_dirs if Path(d).exists()]

        if source_paths:
            # Try to find common parent (e.g., /securedstorage/DATAsec/cole/Data-secured)
            common_parent = None
            first_path = source_paths[0]

            # Check if all paths share a common parent
            for parent in first_path.parents:
                if all(parent in p.parents or p == parent for p in source_paths):
                    common_parent = parent
                    break

            if common_parent and common_parent.exists():
                log.info(f"[BATCH COLLECTION] Searching common parent: {common_parent}")
                collect_to_label_images_batch(common_parent, batch_dest)
            else:
                # Fallback: search each directory individually
                for source_path in source_paths:
                    log.info(f"[BATCH COLLECTION] Searching: {source_path}")
                    collect_to_label_images_batch(source_path, batch_dest)
