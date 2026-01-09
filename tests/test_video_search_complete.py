"""Complete test of video search with real secure storage pattern."""

from pathlib import Path
import sys

def test_complete_video_search():
    """Test complete video search workflow."""
    print("Testing complete video search workflow...\n")

    # Simulate real scenario
    fly_name = "september_16_fly_1"
    folder_name = "september_16_fly_1_testing_3"

    # Build search pattern (what the module does)
    search_patterns = [
        f"output_{folder_name}_*.mp4",  # With output_ prefix and timestamp
        f"{folder_name}.mp4",            # Without prefix
        f"{folder_name}_preprocessed.mp4",
        f"pre_{folder_name}.mp4",
    ]

    print(f"Fly name: {fly_name}")
    print(f"Folder name: {folder_name}")
    print(f"\nSearch patterns:")
    for i, pattern in enumerate(search_patterns, 1):
        print(f"  {i}. {pattern}")

    # Check secure storage
    secure_base = Path("/securedstorage/DATAsec/cole/Data-secured/opto_EB")
    fly_dir = secure_base / fly_name

    if not fly_dir.exists():
        print(f"\n⚠ Fly directory not found: {fly_dir}")
        print("Cannot complete test with real files")
        return False

    print(f"\n✓ Fly directory exists: {fly_dir}")

    # Try each pattern
    video_found = None
    for pattern in search_patterns:
        matches = list(fly_dir.glob(pattern))
        if matches:
            video_found = matches[0]
            print(f"\n✓ Pattern '{pattern}' matched!")
            print(f"  Found: {video_found.name}")
            break
        else:
            print(f"  ✗ Pattern '{pattern}' - no match")

    if video_found:
        print(f"\n✅ SUCCESS: Video found at {video_found}")
        return True
    else:
        print(f"\n❌ FAILED: No video found for folder '{folder_name}'")
        print(f"\nAvailable videos in {fly_dir.name}:")
        videos = list(fly_dir.glob("*.mp4"))
        for v in videos[:5]:
            print(f"  - {v.name}")
        return False


def test_multiple_flies():
    """Test with multiple flies to verify consistency."""
    print("\n" + "=" * 70)
    print("Testing multiple fly directories...")
    print("=" * 70 + "\n")

    secure_base = Path("/securedstorage/DATAsec/cole/Data-secured/opto_EB")

    # Find actual fly directories (not plot/summary dirs)
    fly_dirs = [
        d for d in secure_base.iterdir()
        if d.is_dir() and not any(skip in d.name for skip in ["plot", "summary", "threshold"])
    ]

    success_count = 0
    total_count = 0

    for fly_dir in fly_dirs[:5]:  # Test first 5 flies
        print(f"\nChecking {fly_dir.name}:")

        videos = list(fly_dir.glob("output_*_testing_*.mp4"))
        if not videos:
            videos = list(fly_dir.glob("output_*_training_*.mp4"))

        if videos:
            video = videos[0]
            print(f"  Video: {video.name}")

            # Extract folder name from video filename
            # output_september_16_fly_1_testing_3_20250916_143551.mp4
            name_parts = video.stem.split("_")

            # Find trial type position
            trial_idx = None
            for i, part in enumerate(name_parts):
                if part in ("testing", "training"):
                    trial_idx = i
                    break

            if trial_idx and trial_idx + 1 < len(name_parts):
                # Reconstruct folder name: everything except "output" and timestamp
                folder_name = "_".join(name_parts[1:trial_idx + 2])
                print(f"  Folder name: {folder_name}")

                # Test if pattern would match
                pattern = f"output_{folder_name}_*.mp4"
                matches = list(fly_dir.glob(pattern))

                if matches:
                    print(f"  ✓ Pattern matches: {len(matches)} video(s)")
                    success_count += 1
                else:
                    print(f"  ✗ Pattern failed")

                total_count += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {success_count}/{total_count} successful pattern matches")
    print(f"{'=' * 70}")

    return success_count == total_count


if __name__ == "__main__":
    print("=" * 70)
    print("YOLO Curation - Complete Video Search Test")
    print("=" * 70 + "\n")

    try:
        # Test 1: Single fly with known pattern
        test1_passed = test_complete_video_search()

        # Test 2: Multiple flies
        test2_passed = test_multiple_flies()

        if test1_passed and test2_passed:
            print("\n" + "=" * 70)
            print("✅ ALL TESTS PASSED")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("⚠ SOME TESTS FAILED")
            print("=" * 70)
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
