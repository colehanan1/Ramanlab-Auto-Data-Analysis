"""secure_copy_and_cleanup must prune local .mp4 AND .h264 videos once they have
been copied to the secured location."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import envelope_combined as ec  # noqa: E402


def test_cleanup_prunes_h264_and_mp4_after_secured_copy(tmp_path):
    source = tmp_path / "Hex-Training"
    dest = tmp_path / "secured"
    fly = source / "may_01_batch_1"  # month-named -> eligible for cleanup
    fly.mkdir(parents=True)

    mp4 = fly / "trial_1.mp4"
    h264 = fly / "trial_1.h264"
    keep = fly / "trial_1_distances.csv"  # non-video, must be preserved
    for f in (mp4, h264, keep):
        f.write_bytes(b"x")

    ec.secure_copy_and_cleanup([str(source)], str(dest), perform_cleanup=True)

    secured_fly = dest / "Hex-Training" / "may_01_batch_1"
    # Both videos copied to secured...
    assert (secured_fly / "trial_1.mp4").is_file()
    assert (secured_fly / "trial_1.h264").is_file()
    # ...and pruned from local.
    assert not mp4.exists()
    assert not h264.exists()         # the new behaviour
    # Non-video data is never pruned.
    assert keep.exists()
