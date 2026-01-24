#!/usr/bin/env python3
"""
Unified backup system for project data.
Supports direct syncing (default) and optional compression.
Uses rsync for SMB and rclone for Box/cloud storage.
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import zipfile
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BackupSystem:
    def __init__(self, config_path=None):
        """Initialize backup system with config."""
        self.base_path = Path("/home/ramanlab/Documents/cole")

        # Load config
        if config_path is None:
            config_path = Path("/home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis/config/config.yaml")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.backup_config = self.config.get('backups', {})
        self.compression_enabled = self.backup_config.get('compression', {}).get('enabled', False)
        self.archive_dir = self.base_path / "backups_compressed"

        if self.compression_enabled:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

    def should_backup(self, destination):
        """Check if a backup destination is enabled."""
        return self.backup_config.get('destinations', {}).get(destination, {}).get('enabled', True)

    def should_use_compression(self, destination):
        """Check if compression is enabled for a destination."""
        dest_config = self.backup_config.get('destinations', {}).get(destination, {})
        # Use destination-specific override if set, otherwise use global setting
        if 'use_compression' in dest_config:
            return dest_config['use_compression']
        return self.compression_enabled

    def get_csv_files(self):
        """Get list of CSV files to backup."""
        csv_files = [
            "Data/Opto/Combined/all_envelope_rows_wide.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_training.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_combined_base.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_combined_base_training.csv",
            "Data/Opto/Combined/model_predictions.csv",
            "Data/Opto/all_eye_prob_coords_wide.csv",
        ]
        return [self.base_path / f for f in csv_files if (self.base_path / f).exists()]

    def get_results_dirs(self):
        """Get list of results directories to backup."""
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
        return [self.base_path / d for d in results_dirs if (self.base_path / d).exists()]

    def compress_csvs(self):
        """Compress CSV files. Returns path to zip file."""
        if not self.compression_enabled:
            return None

        logger.info("Compressing CSV files...")
        csv_files = self.get_csv_files()

        if not csv_files:
            logger.warning("No CSV files found to compress")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = self.archive_dir / f"CSVs_archive_{timestamp}.zip"

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
                for src in csv_files:
                    zipf.write(src, arcname=src.name)
                    logger.info(f"  + Compressed: {src.name}")

            original_size = sum(f.stat().st_size for f in csv_files)
            compressed_size = zip_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            logger.info(f"✓ Created: {zip_path.name} ({ratio:.1f}% reduction)")
            return zip_path
        except Exception as e:
            logger.error(f"Error compressing CSVs: {e}")
            return None

    def compress_results(self):
        """Compress results directories. Returns path to zip file."""
        if not self.compression_enabled:
            return None

        logger.info("Compressing Results/Figures...")
        results_dirs = self.get_results_dirs()

        if not results_dirs:
            logger.warning("No results directories found to compress")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = self.archive_dir / f"Results_archive_{timestamp}.zip"

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
                for src in results_dirs:
                    for file_path in src.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(self.base_path)
                            zipf.write(file_path, arcname=arcname)
                    logger.info(f"  + Compressed: {src.name}/")

            original_size = sum(self._get_dir_size(d) for d in results_dirs)
            compressed_size = zip_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            logger.info(f"✓ Created: {zip_path.name} ({ratio:.1f}% reduction)")
            return zip_path
        except Exception as e:
            logger.error(f"Error compressing results: {e}")
            return None

    def backup_to_smb(self, dry_run=False):
        """Backup to SMB using rsync."""
        if not self.should_backup('smb'):
            logger.info("SMB backup disabled, skipping")
            return True

        use_compression = self.should_use_compression('smb')
        smb_config = self.backup_config.get('destinations', {}).get('smb', {})
        smb_base = smb_config.get('base_path', 'smb://ramanfile.local/ramanfiles/cole')

        logger.info("Starting SMB backup via rsync...")

        success = True

        # Backup CSVs
        csv_dest = smb_config.get('csvs_path', f"{smb_base}/flyTrackingData")
        self._rsync_csvs_to_smb(csv_dest, use_compression, dry_run)

        # Backup Results
        results_dest = smb_config.get('results_path', f"{smb_base}/Figures")
        self._rsync_results_to_smb(results_dest, use_compression, dry_run)

        return success

    def _rsync_csvs_to_smb(self, dest, use_compression, dry_run):
        """Rsync CSVs to SMB."""
        csv_files = self.get_csv_files()

        if not csv_files:
            logger.info("No CSV files to backup to SMB")
            return

        if use_compression:
            zip_file = self.compress_csvs()
            if zip_file:
                self._rsync_file(zip_file, dest, dry_run)
        else:
            for csv_file in csv_files:
                # Copy individual CSV files via rsync
                cmd = ["rsync", "-av", "--update"]
                if dry_run:
                    cmd.append("--dry-run")
                cmd.extend([str(csv_file), f"{dest}/"])

                try:
                    if dry_run:
                        logger.info(f"[DRY-RUN] rsync {csv_file.name} to {dest}")
                    else:
                        logger.info(f"Rsyncing: {csv_file.name} -> {dest}")
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            logger.info(f"✓ Backed up to SMB: {csv_file.name}")
                        else:
                            logger.warning(f"⚠ SMB backup warning: {result.stderr}")
                except Exception as e:
                    logger.error(f"Error rsyncing to SMB: {e}")

    def _rsync_results_to_smb(self, dest, use_compression, dry_run):
        """Rsync Results to SMB."""
        results_dirs = self.get_results_dirs()

        if not results_dirs:
            logger.info("No results directories to backup to SMB")
            return

        if use_compression:
            zip_file = self.compress_results()
            if zip_file:
                self._rsync_file(zip_file, dest, dry_run)
        else:
            for results_dir in results_dirs:
                # Rsync directory recursively
                cmd = ["rsync", "-av", "--update"]
                if dry_run:
                    cmd.append("--dry-run")
                cmd.extend([str(results_dir) + "/", f"{dest}/"])

                try:
                    if dry_run:
                        logger.info(f"[DRY-RUN] rsync {results_dir.name}/ to {dest}")
                    else:
                        logger.info(f"Rsyncing: {results_dir.name}/ -> {dest}")
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                        if result.returncode == 0:
                            logger.info(f"✓ Backed up to SMB: {results_dir.name}/")
                        else:
                            logger.warning(f"⚠ SMB backup warning: {result.stderr}")
                except Exception as e:
                    logger.error(f"Error rsyncing to SMB: {e}")

    def _rsync_file(self, src_file, dest, dry_run):
        """Helper to rsync a single file."""
        cmd = ["rsync", "-av", "--update"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend([str(src_file), f"{dest}/"])

        try:
            if dry_run:
                logger.info(f"[DRY-RUN] rsync {src_file.name} to {dest}")
            else:
                logger.info(f"Rsyncing: {src_file.name} -> {dest}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    logger.info(f"✓ Backed up to SMB: {src_file.name}")
                else:
                    logger.warning(f"⚠ SMB backup warning: {result.stderr}")
        except Exception as e:
            logger.error(f"Error rsyncing file: {e}")

    def backup_to_box(self, dry_run=False):
        """Backup to Box using rclone."""
        if not self.should_backup('box'):
            logger.info("Box backup disabled, skipping")
            return True

        use_compression = self.should_use_compression('box')
        box_config = self.backup_config.get('destinations', {}).get('box', {})
        box_remote = box_config.get('remote', 'Box-Folder')
        box_folder = box_config.get('folder', 'Ramanlab-Backups')

        logger.info("Starting Box backup via rclone...")

        # Backup CSVs
        csv_files = self.get_csv_files()
        if use_compression:
            zip_file = self.compress_csvs()
            if zip_file:
                self._rclone_copy(zip_file, f"{box_remote}:{box_folder}/CSVs/", dry_run)
        else:
            for csv_file in csv_files:
                self._rclone_copy(csv_file, f"{box_remote}:{box_folder}/CSVs/", dry_run)

        # Backup Results
        results_dirs = self.get_results_dirs()
        if use_compression:
            zip_file = self.compress_results()
            if zip_file:
                self._rclone_copy(zip_file, f"{box_remote}:{box_folder}/Results/", dry_run)
        else:
            for results_dir in results_dirs:
                self._rclone_sync(results_dir, f"{box_remote}:{box_folder}/Results/{results_dir.name}", dry_run)

    def backup_to_secured(self, dry_run=False):
        """Backup to secured storage via rsync."""
        if not self.should_backup('secured'):
            logger.info("Secured storage backup disabled, skipping")
            return True

        use_compression = self.should_use_compression('secured')
        secured_config = self.backup_config.get('destinations', {}).get('secured', {})
        secured_base = secured_config.get('base_path', '/securedstorage/DATAsec/cole/Data-secured')

        logger.info("Starting Secured storage backup via rsync...")

        # Backup CSVs
        csv_files = self.get_csv_files()
        if use_compression:
            zip_file = self.compress_csvs()
            if zip_file:
                self._rsync_file(zip_file, f"{secured_base}/backups", dry_run)
        else:
            for csv_file in csv_files:
                cmd = ["rsync", "-av", "--update"]
                if dry_run:
                    cmd.append("--dry-run")
                cmd.extend([str(csv_file), f"{secured_base}/CSVs/"])

                try:
                    if dry_run:
                        logger.info(f"[DRY-RUN] rsync {csv_file.name} to {secured_base}/CSVs/")
                    else:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            logger.info(f"✓ Backed up to secured: {csv_file.name}")
                except Exception as e:
                    logger.error(f"Error rsyncing to secured: {e}")

    def _rclone_copy(self, src, dest, dry_run=False):
        """Copy file to Box via rclone."""
        cmd = ["rclone", "copy"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend([str(src), dest, "-v", "--update"])

        try:
            logger.info(f"Uploading to Box: {Path(src).name} -> {dest}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.info(f"✓ Backed up to Box: {Path(src).name}")
            else:
                logger.warning(f"⚠ Box backup warning: {result.stderr}")
        except Exception as e:
            logger.error(f"Error uploading to Box: {e}")

    def _rclone_sync(self, src, dest, dry_run=False):
        """Sync directory to Box via rclone."""
        cmd = ["rclone", "sync"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend([str(src), dest, "-v", "--update"])

        try:
            logger.info(f"Syncing to Box: {Path(src).name}/ -> {dest}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.info(f"✓ Backed up to Box: {Path(src).name}/")
            else:
                logger.warning(f"⚠ Box backup warning: {result.stderr}")
        except Exception as e:
            logger.error(f"Error syncing to Box: {e}")

    def _get_dir_size(self, path):
        """Get total size of directory."""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
        return total

    def run_all_backups(self, dry_run=False):
        """Run all enabled backup destinations."""
        logger.info("=" * 70)
        logger.info(f"Starting Backup System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Dry-run: {dry_run}")
        logger.info(f"Compression enabled: {self.compression_enabled}")
        logger.info("=" * 70)

        self.backup_to_smb(dry_run)
        self.backup_to_box(dry_run)
        self.backup_to_secured(dry_run)

        logger.info("=" * 70)
        logger.info("✓ Backup system completed")
        logger.info("=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Backup system for project data')
    parser.add_argument('--dry-run', action='store_true', help='Run without making changes')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--smb-only', action='store_true', help='Backup to SMB only')
    parser.add_argument('--box-only', action='store_true', help='Backup to Box only')
    parser.add_argument('--secured-only', action='store_true', help='Backup to secured storage only')

    args = parser.parse_args()

    backup = BackupSystem(args.config)
    backup.run_all_backups(args.dry_run)


if __name__ == "__main__":
    main()
