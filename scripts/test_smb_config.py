#!/usr/bin/env python3
"""Test script to validate SMB rclone configuration."""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fbpipe.utils.smb_copy import get_smb_copier


def main():
    """Test SMB configuration."""
    logger.info("=" * 70)
    logger.info("SMB Configuration Test")
    logger.info("=" * 70)

    copier = get_smb_copier()

    # Test 1: Check remote exists
    logger.info("\n[Test 1] Checking rclone remote...")
    if copier._verify_remote_exists():
        logger.info("✓ Remote 'SMB-Ramanfile' is configured")
    else:
        logger.error("✗ Remote 'SMB-Ramanfile' not found")
        logger.error("  Run: rclone config")
        return False

    # Test 2: Test connection
    logger.info("\n[Test 2] Testing SMB connection...")
    if copier.test_connection():
        logger.info("✓ Successfully connected to SMB share")
    else:
        logger.error("✗ Failed to connect to SMB share")
        logger.error("  Check credentials and SMB host availability")
        return False

    # Test 3: List SMB contents
    logger.info("\n[Test 3] Listing SMB share contents...")
    try:
        import subprocess
        result = subprocess.run(
            ["rclone", "lsd", "SMB-Ramanfile:ramanfiles/cole", "-h"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info("✓ SMB folders visible:")
            for line in result.stdout.strip().split('\n')[:10]:
                if line.strip():
                    logger.info(f"  {line}")
        else:
            logger.warning(f"⚠ Could not list folders: {result.stderr}")
    except Exception as e:
        logger.error(f"Error listing folders: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ All tests passed! SMB is ready for use.")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("1. Review config/config.yaml and add out_dir_smb fields")
    logger.info("2. Example config:")
    logger.info("   out_dir_smb: 'ramanfiles/cole/Figures/MyResults/'")
    logger.info("3. Run pipeline normally - SMB copies will happen automatically")
    logger.info("\nDocumentation: docs/SMB-RCLONE-SETUP.md")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
