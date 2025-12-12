#!/usr/bin/env python3
"""
Compress raw H.264 videos with ffmpeg before YOLO processing.

This script:
1. Finds all .mp4/.avi videos in configured directories
2. Compresses them with H.264 CRF 30
3. Saves compressed version with _compressed suffix
4. Optionally deletes raw videos to save space

Usage:
    python scripts/compress_videos.py [--delete-raw] [--dry-run]
"""
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# From config.yaml
MAIN_DIRECTORIES = [
    "/home/ramanlab/Documents/cole/Data/flys/opto_EB(6-training)/",
    "/home/ramanlab/Documents/cole/Data/flys/opto_benz_1/",
    "/home/ramanlab/Documents/cole/Data/flys/opto_hex/",
    "/home/ramanlab/Documents/cole/Data/flys/opto_ACV/",
    "/home/ramanlab/Documents/cole/Data/flys/hex_control/",
    "/home/ramanlab/Documents/cole/Data/flys/EB_control/",
    "/home/ramanlab/Documents/cole/Data/flys/Benz_control/",
    "/home/ramanlab/Documents/cole/Data/flys/opto_AIR/",
    "/home/ramanlab/Documents/cole/Data/flys/opto_3-oct/",
]

VIDEO_EXTENSIONS = {'.mp4', '.avi'}
CRF = 30  # Compression quality (28-32 recommended)
PRESET = 'medium'  # Encoding speed (ultrafast/fast/medium/slow/veryslow)

def find_videos(directories: List[str]) -> List[Path]:
    """Find all video files in configured directories."""
    videos = []
    for dir_path in directories:
        root = Path(dir_path).expanduser().resolve()
        if not root.exists():
            log.warning(f"Directory not found: {root}")
            continue

        for fly_dir in root.iterdir():
            if not fly_dir.is_dir():
                continue

            for video in fly_dir.iterdir():
                if video.suffix.lower() in VIDEO_EXTENSIONS:
                    # Skip already compressed videos
                    if '_compressed' in video.stem:
                        continue
                    videos.append(video)

    return videos

def compress_video(input_path: Path, output_path: Path, dry_run: bool = False) -> bool:
    """Compress video using ffmpeg with H.264 CRF encoding."""
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-crf', str(CRF),
        '-preset', PRESET,
        '-c:a', 'copy',  # Copy audio without re-encoding
        '-y',  # Overwrite output
        str(output_path)
    ]

    if dry_run:
        log.info(f"[DRY RUN] Would compress: {input_path.name}")
        return True

    log.info(f"Compressing: {input_path.name}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Check file size reduction
        original_size = input_path.stat().st_size / (1024**2)  # MB
        compressed_size = output_path.stat().st_size / (1024**2)  # MB
        reduction = (1 - compressed_size / original_size) * 100

        log.info(f"  ✓ {input_path.name}: {original_size:.1f}MB → {compressed_size:.1f}MB ({reduction:.1f}% reduction)")
        return True

    except subprocess.CalledProcessError as e:
        log.error(f"  ✗ Failed to compress {input_path.name}: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Compress videos before YOLO processing')
    parser.add_argument('--delete-raw', action='store_true',
                       help='Delete raw videos after successful compression')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually compressing')
    args = parser.parse_args()

    log.info("Finding videos to compress...")
    videos = find_videos(MAIN_DIRECTORIES)

    if not videos:
        log.info("No videos found to compress")
        return

    log.info(f"Found {len(videos)} videos to compress")

    if args.dry_run:
        log.info("DRY RUN MODE - no files will be modified")

    compressed_count = 0
    failed_count = 0

    for video in videos:
        output_path = video.parent / f"{video.stem}_compressed{video.suffix}"

        # Skip if compressed version already exists
        if output_path.exists():
            log.info(f"Skipping (already compressed): {video.name}")
            continue

        success = compress_video(video, output_path, dry_run=args.dry_run)

        if success:
            compressed_count += 1

            if args.delete_raw and not args.dry_run:
                log.info(f"  Deleting raw: {video.name}")
                video.unlink()
        else:
            failed_count += 1

    log.info("")
    log.info("="*60)
    log.info(f"Compression complete!")
    log.info(f"  Compressed: {compressed_count} videos")
    log.info(f"  Failed: {failed_count} videos")
    if args.delete_raw and not args.dry_run:
        log.info(f"  Deleted: {compressed_count} raw videos")
    log.info("="*60)

if __name__ == "__main__":
    main()
