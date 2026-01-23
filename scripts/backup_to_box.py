#!/usr/bin/env python3
"""
Backup script to sync data to Box cloud storage.
Supports backing up CSV files, results/figures, and model files.
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

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


class BoxBackup:
    def __init__(self, config_path=None):
        """Initialize backup handler with optional config file."""
        self.base_path = Path("/home/ramanlab/Documents/cole")
        self.rclone_remote = "Box-Folder"  # Your configured rclone remote
        self.rclone_folder = "Ramanlab-Backups"

    def check_rclone_installed(self):
        """Verify rclone is installed and configured."""
        try:
            result = subprocess.run(
                ["rclone", "version"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error("rclone not found. Install with: pip install rclone")
                return False
            logger.info("rclone is installed")
            return True
        except FileNotFoundError:
            logger.error("rclone command not found. Install with: pip install rclone")
            return False

    def check_box_remote_configured(self):
        """Check if Box remote is configured in rclone."""
        try:
            result = subprocess.run(
                ["rclone", "listremotes"],
                capture_output=True,
                text=True
            )
            remotes = result.stdout.strip().split('\n')
            # Remove colons from remote names for comparison
            remotes = [r.rstrip(':') for r in remotes if r]
            if self.rclone_remote in remotes:
                logger.info(f"Box remote '{self.rclone_remote}' is configured")
                return True
            else:
                logger.error(
                    f"Box remote '{self.rclone_remote}' not configured. "
                    f"Run: rclone config to set it up"
                )
                logger.error(f"Available remotes: {remotes}")
                return False
        except Exception as e:
            logger.error(f"Error checking rclone remotes: {e}")
            return False

    def backup_csvs(self, dry_run=False):
        """Backup all CSV files."""
        logger.info("Starting CSV backup...")

        csv_sources = [
            "Data/Opto/Combined/all_envelope_rows_wide.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_training.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_combined_base.csv",
            "Data/Opto/Combined/all_envelope_rows_wide_combined_base_training.csv",
            "Data/Opto/Combined/model_predictions.csv",
            "Data/Opto/all_eye_prob_coords_wide.csv",
        ]

        for csv_file in csv_sources:
            src = self.base_path / csv_file
            if not src.exists():
                logger.warning(f"CSV not found (may not exist yet): {src}")
                continue

            dest = f"{self.rclone_remote}:{self.rclone_folder}/CSVs/{src.name}"
            self._sync_file(src, dest, dry_run)

    def backup_results(self, dry_run=False):
        """Backup all results/figures."""
        logger.info("Starting Results/Figures backup...")

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

        for results_dir in results_dirs:
            src = self.base_path / results_dir
            if not src.exists():
                logger.warning(f"Results directory not found: {src}")
                continue

            dest = f"{self.rclone_remote}:{self.rclone_folder}/Results/{src.name}"
            self._sync_directory(src, dest, dry_run)

    def backup_data_folders(self, dry_run=False):
        """Backup complete Data folders."""
        logger.info("Starting Data folders backup...")

        data_dirs = [
            "Data/flys",
            "Data/Opto/Combined",
        ]

        for data_dir in data_dirs:
            src = self.base_path / data_dir
            if not src.exists():
                logger.warning(f"Data directory not found: {src}")
                continue

            dest = f"{self.rclone_remote}:{self.rclone_folder}/Data/{src.name}"
            self._sync_directory(src, dest, dry_run)

    def backup_models(self, dry_run=False):
        """Backup model files."""
        logger.info("Starting Model files backup...")

        model_dirs = [
            "model/YOLOProjectProboscisLegs/runs/obb/train10/weights",
        ]

        for model_dir in model_dirs:
            src = self.base_path / model_dir
            if not src.exists():
                logger.warning(f"Model directory not found: {src}")
                continue

            dest = f"{self.rclone_remote}:{self.rclone_folder}/Models/{src.name}"
            self._sync_directory(src, dest, dry_run)

    def _sync_file(self, src, dest, dry_run=False):
        """Sync a single file to Box (only new/modified files)."""
        # Use 'copy' for files with --update flag (only newer files)
        cmd = ["rclone", "copy"]
        if dry_run:
            cmd.append("--dry-run")
        # --update: only copy if source is newer than destination
        # --no-traverse: skip checking destination (faster)
        cmd.extend([str(src), dest, "-v", "--update", "--no-traverse"])

        try:
            logger.info(f"Syncing file: {src.name} -> {dest}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Successfully backed up: {src.name}")
            else:
                logger.error(f"✗ Failed to backup {src.name}: {result.stderr}")
        except Exception as e:
            logger.error(f"Error syncing {src}: {e}")

    def _sync_directory(self, src, dest, dry_run=False):
        """Sync a directory to Box (only new/modified files)."""
        # Use 'sync' which intelligently syncs only changed files
        cmd = ["rclone", "sync"]
        if dry_run:
            cmd.append("--dry-run")
        # --update: skip files that are newer on remote
        # --delete-excluded: delete files on remote not in source
        # --no-traverse: skip checking destination for speed
        cmd.extend([str(src), dest, "-v", "--update", "--ignore-errors"])

        try:
            logger.info(f"Syncing directory: {src.name} -> {dest}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Successfully backed up: {src.name}")
            else:
                logger.warning(f"⚠ Partial backup of {src.name}: {result.stderr}")
        except Exception as e:
            logger.error(f"Error syncing {src}: {e}")

    def run_full_backup(self, backup_csvs=True, backup_results=True,
                       backup_data=True, backup_models=True, dry_run=False):
        """Run selected backup operations."""
        logger.info("=" * 60)
        logger.info(f"Starting Box Backup - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Dry-run mode: {dry_run}")
        logger.info("=" * 60)

        # Check prerequisites
        if not self.check_rclone_installed():
            logger.error("Cannot proceed without rclone")
            return False

        if not self.check_box_remote_configured():
            logger.error("Cannot proceed without Box remote configured")
            return False

        try:
            if backup_csvs:
                self.backup_csvs(dry_run)
            if backup_results:
                self.backup_results(dry_run)
            if backup_data:
                self.backup_data_folders(dry_run)
            if backup_models:
                self.backup_models(dry_run)

            logger.info("=" * 60)
            logger.info("Backup completed successfully!")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Backup failed with error: {e}")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Backup Ramanlab data to Box cloud storage"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backed up without actually backing up"
    )
    parser.add_argument(
        "--csvs-only",
        action="store_true",
        help="Only backup CSV files"
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Only backup results/figures"
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only backup data folders"
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only backup model files"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup rclone Box remote (interactive)"
    )

    args = parser.parse_args()

    backup = BoxBackup()

    if args.setup:
        logger.info("Starting rclone setup for Box...")
        subprocess.run(["rclone", "config"])
        return

    # Determine what to backup
    backup_csvs = not any([args.results_only, args.data_only, args.models_only]) or args.csvs_only
    backup_results = not any([args.csvs_only, args.data_only, args.models_only]) or args.results_only
    backup_data = not any([args.csvs_only, args.results_only, args.models_only]) or args.data_only
    backup_models = not any([args.csvs_only, args.results_only, args.data_only]) or args.models_only

    success = backup.run_full_backup(
        backup_csvs=backup_csvs,
        backup_results=backup_results,
        backup_data=backup_data,
        backup_models=backup_models,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
