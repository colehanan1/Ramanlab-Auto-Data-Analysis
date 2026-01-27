"""Utility for copying files to SMB shares using rclone."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SMBCopier:
    """Handle copying files and directories to SMB shares using rclone."""

    def __init__(self, rclone_remote: str = "SMB-Ramanfile"):
        """
        Initialize SMB copier.

        Args:
            rclone_remote: Name of the rclone remote to use for SMB transfers.
                          Must be configured in ~/.config/rclone/rclone.conf
        """
        self.rclone_remote = rclone_remote
        self._verify_remote_exists()

    def _verify_remote_exists(self) -> bool:
        """Verify that the rclone remote is configured."""
        try:
            result = subprocess.run(
                ["rclone", "listremotes"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if self.rclone_remote in result.stdout:
                logger.debug(f"✓ rclone remote '{self.rclone_remote}' is configured")
                return True
            else:
                logger.warning(
                    f"⚠ rclone remote '{self.rclone_remote}' not found. "
                    f"Configure it with: rclone config"
                )
                return False
        except Exception as e:
            logger.error(f"Error verifying rclone remote: {e}")
            return False

    def copy_file(
        self,
        src: Path | str,
        dest_path: str,
        skip_same_size: bool = True,
        verbose: bool = False,
        dry_run: bool = False
    ) -> bool:
        """
        Copy a single file to SMB share.

        Args:
            src: Source file path
            dest_path: Destination path on SMB share (e.g., "ramanfiles/cole/Figures/")
            skip_same_size: Skip files with same size (default: True)
            verbose: Enable verbose logging
            dry_run: Preview action without copying

        Returns:
            True if successful, False otherwise
        """
        src = Path(src)
        if not src.exists():
            logger.error(f"Source file not found: {src}")
            return False

        cmd = ["rclone", "copy", str(src), f"{self.rclone_remote}:{dest_path}"]

        if skip_same_size:
            cmd.append("--size-only")
        if verbose:
            cmd.append("-v")
        if dry_run:
            cmd.append("--dry-run")

        try:
            logger.info(f"Copying file: {src.name} -> {self.rclone_remote}:{dest_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info(f"✓ Successfully copied: {src.name}")
                return True
            else:
                logger.error(f"✗ Failed to copy {src.name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout copying file: {src.name}")
            return False
        except Exception as e:
            logger.error(f"Error copying file {src.name}: {e}")
            return False

    def sync_directory(
        self,
        src: Path | str,
        dest_path: str,
        skip_same_size: bool = True,
        verbose: bool = False,
        dry_run: bool = False,
        delete_extra: bool = False
    ) -> bool:
        """
        Sync a directory to SMB share.

        Args:
            src: Source directory path
            dest_path: Destination path on SMB share
            skip_same_size: Skip files with same size (default: True)
            verbose: Enable verbose logging
            dry_run: Preview action without syncing
            delete_extra: Delete files on destination that aren't in source

        Returns:
            True if successful, False otherwise
        """
        src = Path(src)
        if not src.exists() or not src.is_dir():
            logger.error(f"Source directory not found: {src}")
            return False

        cmd = ["rclone", "sync", str(src), f"{self.rclone_remote}:{dest_path}"]

        if skip_same_size:
            cmd.append("--size-only")
        if verbose:
            cmd.append("-v")
        if dry_run:
            cmd.append("--dry-run")
        if delete_extra:
            cmd.append("--delete-during")

        try:
            logger.info(f"Syncing directory: {src.name}/ -> {self.rclone_remote}:{dest_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                logger.info(f"✓ Successfully synced: {src.name}/")
                return True
            else:
                logger.error(f"✗ Failed to sync {src.name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout syncing directory: {src.name}")
            return False
        except Exception as e:
            logger.error(f"Error syncing directory {src.name}: {e}")
            return False

    def copy_to_csv_path(self, src: Path | str, dry_run: bool = False) -> bool:
        """
        Copy CSV file to the standard CSV location on SMB.

        Args:
            src: Source CSV file
            dry_run: Preview without copying

        Returns:
            True if successful
        """
        return self.copy_file(src, "ramanfiles/cole/flyTrackingData/", dry_run=dry_run)

    def copy_to_figures_path(self, src: Path | str, dry_run: bool = False) -> bool:
        """
        Copy file/directory to the standard Figures location on SMB.

        Args:
            src: Source file or directory
            dry_run: Preview without copying

        Returns:
            True if successful
        """
        src = Path(src)
        if src.is_dir():
            return self.sync_directory(
                src,
                f"ramanfiles/cole/Figures/{src.name}/",
                dry_run=dry_run
            )
        else:
            return self.copy_file(src, "ramanfiles/cole/Figures/", dry_run=dry_run)

    def test_connection(self) -> bool:
        """Test connection to SMB share."""
        try:
            logger.info(f"Testing connection to {self.rclone_remote}...")
            result = subprocess.run(
                ["rclone", "lsd", f"{self.rclone_remote}:"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info(f"✓ Successfully connected to {self.rclone_remote}")
                return True
            else:
                logger.error(f"✗ Failed to connect: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            return False


# Singleton instance for use throughout the codebase
_smb_copier: Optional[SMBCopier] = None


def get_smb_copier() -> SMBCopier:
    """Get or create the SMB copier instance."""
    global _smb_copier
    if _smb_copier is None:
        _smb_copier = SMBCopier()
    return _smb_copier


def copy_csv_to_smb(csv_path: Path | str, dry_run: bool = False) -> bool:
    """Convenience function to copy CSV file to SMB."""
    return get_smb_copier().copy_to_csv_path(csv_path, dry_run=dry_run)


def copy_to_smb(src: Path | str, dest_path: str, dry_run: bool = False) -> bool:
    """Convenience function to copy file/directory to SMB."""
    src = Path(src)
    copier = get_smb_copier()
    if src.is_dir():
        return copier.sync_directory(src, dest_path, dry_run=dry_run)
    else:
        return copier.copy_file(src, dest_path, dry_run=dry_run)
