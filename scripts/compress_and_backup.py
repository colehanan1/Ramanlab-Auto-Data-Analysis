#!/usr/bin/env python3
"""
Compression and backup script for emergency data archival.
Compresses critical data and backs up to SMB share and Box.
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import zipfile
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compress_backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompressionBackup:
    def __init__(self):
        """Initialize compression backup handler."""
        self.base_path = Path("/home/ramanlab/Documents/cole")
        self.archive_dir = self.base_path / "backups_compressed"
        self.smb_location = "smb://ramanfile.local/ramanfiles/cole/flyTrackingData"
        self.box_remote = "Box-Folder"
        self.box_folder = "Ramanlab-Backups"

        # Create archive directory if it doesn't exist
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def compress_csvs(self, dry_run=False):
        """Compress all CSV files into a single zip."""
        logger.info("Starting CSV compression...")

        csv_files = [
            "Data/Opto/Combined/all_envelope_rows_wide.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_training.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_combined_base.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_combined_base_training.csv",
            "Data/Opto/Combined/model_predictions.csv",
            "Data/Opto/all_eye_prob_coords_wide.csv",
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = self.archive_dir / f"CSVs_archive_{timestamp}.zip"

        files_to_compress = []
        for csv_file in csv_files:
            src = self.base_path / csv_file
            if src.exists():
                files_to_compress.append((src, src.name))
            else:
                logger.warning(f"CSV not found: {src}")

        if not files_to_compress:
            logger.warning("No CSV files found to compress")
            return None

        if dry_run:
            logger.info(f"[DRY-RUN] Would create: {zip_path}")
            for src, name in files_to_compress:
                logger.info(f"[DRY-RUN]   + {name} ({self._get_size(src)})")
            return zip_path

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
                for src, name in files_to_compress:
                    zipf.write(src, arcname=name)
                    logger.info(f"  + Added: {name}")

            original_size = sum(f[0].stat().st_size for f in files_to_compress)
            compressed_size = zip_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            logger.info(f"✓ Created: {zip_path.name}")
            logger.info(f"  Original: {self._format_size(original_size)}")
            logger.info(f"  Compressed: {self._format_size(compressed_size)}")
            logger.info(f"  Ratio: {ratio:.1f}% reduction")
            return zip_path
        except Exception as e:
            logger.error(f"Error compressing CSVs: {e}")
            return None

    def compress_results(self, dry_run=False):
        """Compress all results/figures into a single zip."""
        logger.info("Starting Results/Figures compression...")

        results_dirs = [
            "Results/Opto/Reaction_Matrices",
            "Results/Opto/Reaction_Matrices/Overlay",
            "Results/Opto/PER-Envelopes",
            "Results/Opto/Training-PER-Envelopes",
            "Results/Opto/Reaction_Predictions(Strictest)",
            "Results/Opto/CombinedBase-PER-Envelopes",
            "Results/Opto/CombinedBase-Training-PER-Envelopes",
            "Results/Opto/Weekly-Training-Envelopes",
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = self.archive_dir / f"Results_archive_{timestamp}.zip"

        dirs_to_compress = []
        for results_dir in results_dirs:
            src = self.base_path / results_dir
            if src.exists():
                dirs_to_compress.append(src)
            else:
                logger.warning(f"Results directory not found: {src}")

        if not dirs_to_compress:
            logger.warning("No results directories found to compress")
            return None

        if dry_run:
            logger.info(f"[DRY-RUN] Would create: {zip_path}")
            total_size = 0
            for src in dirs_to_compress:
                size = self._get_dir_size(src)
                total_size += size
                logger.info(f"[DRY-RUN]   + {src.name}/ ({self._format_size(size)})")
            logger.info(f"[DRY-RUN] Total size: {self._format_size(total_size)}")
            return zip_path

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
                for src in dirs_to_compress:
                    for file_path in src.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(self.base_path)
                            zipf.write(file_path, arcname=arcname)
                    logger.info(f"  + Added: {src.name}/")

            original_size = sum(self._get_dir_size(d) for d in dirs_to_compress)
            compressed_size = zip_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            logger.info(f"✓ Created: {zip_path.name}")
            logger.info(f"  Original: {self._format_size(original_size)}")
            logger.info(f"  Compressed: {self._format_size(compressed_size)}")
            logger.info(f"  Ratio: {ratio:.1f}% reduction")
            return zip_path
        except Exception as e:
            logger.error(f"Error compressing results: {e}")
            return None

    def backup_to_smb(self, zip_files, dry_run=False):
        """Backup compressed zip files to SMB share."""
        logger.info("Starting SMB backup of compressed files...")

        for zip_file in zip_files:
            if not zip_file or not zip_file.exists():
                logger.warning(f"Zip file not found: {zip_file}")
                continue

            cmd = ["cp", str(zip_file), self.smb_location + "/"]

            if dry_run:
                logger.info(f"[DRY-RUN] Would copy: {zip_file.name} to SMB")
                continue

            try:
                logger.info(f"Copying to SMB: {zip_file.name}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode == 0:
                    logger.info(f"✓ Successfully backed up to SMB: {zip_file.name}")
                else:
                    logger.error(f"✗ Failed to copy to SMB: {result.stderr}")
            except Exception as e:
                logger.error(f"Error copying to SMB: {e}")

    def backup_to_box(self, zip_files, dry_run=False):
        """Backup compressed zip files to Box."""
        logger.info("Starting Box backup of compressed files...")

        for zip_file in zip_files:
            if not zip_file or not zip_file.exists():
                logger.warning(f"Zip file not found: {zip_file}")
                continue

            # Determine Box destination folder based on filename
            if "CSVs" in zip_file.name:
                dest = f"{self.box_remote}:{self.box_folder}/CSVs/{zip_file.name}"
            elif "Results" in zip_file.name:
                dest = f"{self.box_remote}:{self.box_folder}/Results/{zip_file.name}"
            else:
                dest = f"{self.box_remote}:{self.box_folder}/{zip_file.name}"

            cmd = ["rclone", "copy", str(zip_file), dest, "-v"]

            if dry_run:
                logger.info(f"[DRY-RUN] Would upload to Box: {zip_file.name}")
                continue

            try:
                logger.info(f"Uploading to Box: {zip_file.name}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode == 0:
                    logger.info(f"✓ Successfully uploaded to Box: {zip_file.name}")
                else:
                    logger.error(f"✗ Failed to upload to Box: {result.stderr}")
            except Exception as e:
                logger.error(f"Error uploading to Box: {e}")

    def cleanup_old_archives(self, days_old=30, dry_run=False):
        """Delete archive files older than specified days."""
        logger.info(f"Cleaning up archives older than {days_old} days...")

        cutoff_time = datetime.now().timestamp() - (days_old * 86400)
        deleted_count = 0

        for zip_file in self.archive_dir.glob("*.zip"):
            if zip_file.stat().st_mtime < cutoff_time:
                if dry_run:
                    logger.info(f"[DRY-RUN] Would delete: {zip_file.name}")
                else:
                    try:
                        zip_file.unlink()
                        logger.info(f"✓ Deleted old archive: {zip_file.name}")
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting {zip_file}: {e}")

        logger.info(f"Cleanup complete. Deleted {deleted_count} old archives.")

    def _get_size(self, path):
        """Get file size in bytes."""
        return path.stat().st_size if path.exists() else 0

    def _get_dir_size(self, path):
        """Get total size of directory in bytes."""
        total = 0
        if path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total += file_path.stat().st_size
        return total

    def _format_size(self, bytes_size):
        """Format bytes to human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"

    def run_full_compression_backup(self, compress_csvs=True, compress_results=True,
                                    backup_smb=True, backup_box=True, cleanup=True,
                                    dry_run=False):
        """Run full compression and backup workflow."""
        logger.info("=" * 60)
        logger.info(f"Starting Compression Backup - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Dry-run mode: {dry_run}")
        logger.info("=" * 60)

        zip_files = []

        try:
            if compress_csvs:
                csv_zip = self.compress_csvs(dry_run)
                if csv_zip:
                    zip_files.append(csv_zip)

            if compress_results:
                results_zip = self.compress_results(dry_run)
                if results_zip:
                    zip_files.append(results_zip)

            if backup_smb:
                self.backup_to_smb(zip_files, dry_run)

            if backup_box:
                self.backup_to_box(zip_files, dry_run)

            if cleanup:
                self.cleanup_old_archives(dry_run=dry_run)

            logger.info("=" * 60)
            logger.info("Compression backup completed successfully!")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Compression backup failed: {e}")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compress and backup critical data to SMB and Box"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backed up without actually doing it"
    )
    parser.add_argument(
        "--csvs-only",
        action="store_true",
        help="Only compress and backup CSV files"
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Only compress and backup results/figures"
    )
    parser.add_argument(
        "--smb-only",
        action="store_true",
        help="Only backup to SMB (skip Box)"
    )
    parser.add_argument(
        "--box-only",
        action="store_true",
        help="Only backup to Box (skip SMB)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't delete old archive files"
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run cleanup of old archives"
    )

    args = parser.parse_args()

    backup = CompressionBackup()

    if args.cleanup_only:
        backup.cleanup_old_archives(dry_run=args.dry_run)
        return

    # Determine what to backup
    compress_csvs = not args.results_only or args.csvs_only
    compress_results = not args.csvs_only or args.results_only
    backup_smb = not args.box_only
    backup_box = not args.smb_only
    cleanup = not args.no_cleanup

    success = backup.run_full_compression_backup(
        compress_csvs=compress_csvs,
        compress_results=compress_results,
        backup_smb=backup_smb,
        backup_box=backup_box,
        cleanup=cleanup,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
