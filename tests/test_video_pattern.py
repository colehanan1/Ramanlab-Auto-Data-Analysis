"""Test video search pattern matching for secure storage."""

from pathlib import Path
import sys

def test_video_pattern_matching():
    """Test that glob patterns work with real video naming."""
    print("Testing video pattern matching...")

    # Example from user:
    # /securedstorage/DATAsec/cole/Data-secured/opto_EB/september_16_fly_1/
    #   output_september_16_fly_1_testing_3_20250916_143551.mp4

    secure_storage = Path("/securedstorage/DATAsec/cole/Data-secured/opto_EB")

    if not secure_storage.exists():
        print(f"⚠ Secure storage not found: {secure_storage}")
        print("Cannot test real video patterns")
        return

    print(f"\n✓ Secure storage accessible: {secure_storage}")

    # Find fly directories
    fly_dirs = [d for d in secure_storage.iterdir() if d.is_dir()]
    print(f"\n  Found {len(fly_dirs)} fly directories")

    # Test pattern matching on first few directories
    for fly_dir in fly_dirs[:3]:
        print(f"\n  Checking: {fly_dir.name}")

        # List all MP4 files
        videos = list(fly_dir.glob("*.mp4"))
        if videos:
            print(f"    Found {len(videos)} video(s):")
            for video in videos[:3]:  # Show first 3
                print(f"      - {video.name}")

            # Test if pattern matching would work
            # Simulate folder_name from CSV like "september_16_fly_1_testing_3"
            video_name = videos[0].stem  # Remove .mp4
            if video_name.startswith("output_"):
                # Extract the base name without output_ and timestamp
                parts = video_name.split("_")
                # Find where trial type starts (testing, training)
                trial_idx = None
                for i, part in enumerate(parts):
                    if part in ("testing", "training"):
                        trial_idx = i
                        break

                if trial_idx and trial_idx + 1 < len(parts):
                    # Reconstruct folder_name: everything from after "output_" to trial_type + number
                    folder_name = "_".join(parts[1:trial_idx+2])
                    print(f"    Inferred folder name: {folder_name}")

                    # Test if glob pattern works
                    pattern = f"output_{folder_name}_*.mp4"
                    matches = list(fly_dir.glob(pattern))
                    print(f"    Pattern '{pattern}' matches: {len(matches)} video(s)")
                    if matches:
                        print(f"      ✓ Would find: {matches[0].name}")
        else:
            print(f"    No videos found")

    print("\n✓ Pattern matching test complete")

if __name__ == "__main__":
    print("=" * 70)
    print("YOLO Curation - Video Pattern Test")
    print("=" * 70)

    try:
        test_video_pattern_matching()
        print("\n" + "=" * 70)
        print("✓ TEST COMPLETE")
        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
