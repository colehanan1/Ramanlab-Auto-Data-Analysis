"""Test video search with secure storage directories."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fbpipe.config import load_settings

def test_video_source_dirs():
    """Test that video_source_dirs loads correctly."""
    print("Testing video source directories configuration...")

    cfg = load_settings("config.yaml")

    print(f"\n✓ Configuration loaded successfully")
    print(f"  - Curation enabled: {cfg.yolo_curation.enabled}")
    print(f"  - Video source dirs: {len(cfg.yolo_curation.video_source_dirs)}")

    if cfg.yolo_curation.video_source_dirs:
        print(f"\n  Configured video source directories:")
        for i, dir_path in enumerate(cfg.yolo_curation.video_source_dirs, 1):
            exists = Path(dir_path).exists()
            status = "✓" if exists else "✗ (not found)"
            print(f"    {i}. {status} {dir_path}")
    else:
        print(f"  ℹ No additional video source directories configured")

    print("\n✓ Video source directory configuration is valid")

if __name__ == "__main__":
    print("=" * 70)
    print("YOLO Curation - Video Source Directory Test")
    print("=" * 70)

    try:
        test_video_source_dirs()
        print("\n" + "=" * 70)
        print("✓ TEST PASSED")
        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
