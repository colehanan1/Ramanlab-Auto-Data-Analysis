"""combine_distance_angle skips frozen experiment folders.

``combined.combine`` regenerates each batch's ``angle_distance_rms_envelope/``
CSVs -- the very files ``build_wide_csv`` then reads. Recomputing them for a
retired folder is pure waste, and leaving them untouched is what "frozen"
means: the rows in the wide CSV stay exactly as they were last derived.

run_workflows already skips this step for a whole data-frozen dataset
(``[FROZEN] combined.combine -> skipping recompute``); this is the per-folder
version inside a dataset that is still live.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev


@pytest.fixture(autouse=True)
def _v2():
    ev.set_protocol("v2")


def _batch_with_distances(root: Path, name: str) -> Path:
    """A batch dir holding one per-trial distances CSV, the combine input.

    Columns are the ones ``_locate_trials`` actually looks for (DIST_COLS /
    ANGLE_COLS); without them the batch is skipped as "no distance trials" and
    the test would pass for the wrong reason.
    """
    trial = root / name / f"{name}_testing_1"
    trial.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame": range(40),
            "angle_centered_pct": np.linspace(0, 100, 40),
            "distance_percentage": np.linspace(0, 100, 40),
            "eye_proboscis_distance": np.linspace(10, 50, 40),
        }
    ).to_csv(trial / f"{name}_testing_1_fly1_distances.csv", index=False)
    return root / name


def test_combine_skips_a_frozen_folder(tmp_path):
    root = tmp_path / "Hex-Training-24-0.1"
    frozen = _batch_with_distances(root, "may_22_batch_1_rig_2")
    live = _batch_with_distances(root, "august_10_batch_1_rig_2")

    ec.combine_distance_angle(
        ec.CombineConfig(root=root, skip_folders=frozenset({"may_22_batch_1_rig_2"}))
    )

    assert not (frozen / "angle_distance_rms_envelope").exists(), (
        "frozen folder was recomputed"
    )
    assert (live / "angle_distance_rms_envelope").exists(), (
        "live folder was not processed"
    )


def test_combine_processes_everything_by_default(tmp_path):
    """An absent skip_folders must reproduce today's behavior exactly."""
    root = tmp_path / "Hex-Training-24-0.1"
    a = _batch_with_distances(root, "may_22_batch_1_rig_2")
    b = _batch_with_distances(root, "august_10_batch_1_rig_2")

    ec.combine_distance_angle(ec.CombineConfig(root=root))

    assert (a / "angle_distance_rms_envelope").exists()
    assert (b / "angle_distance_rms_envelope").exists()


def test_combine_leaves_an_existing_frozen_output_untouched(tmp_path):
    """The point of skipping: previously derived rows survive verbatim, so the
    wide CSV keeps reporting what the folder said when it was retired."""
    root = tmp_path / "Hex-Training-24-0.1"
    frozen = _batch_with_distances(root, "may_22_batch_1_rig_2")
    out = frozen / "angle_distance_rms_envelope"
    out.mkdir(parents=True)
    stale = out / "may_22_batch_1_rig_2_testing_1_angle_distance_rms_envelope.csv"
    stale.write_text("envelope_of_rms\n42.0\n", encoding="utf-8")

    ec.combine_distance_angle(
        ec.CombineConfig(root=root, skip_folders=frozenset({"may_22_batch_1_rig_2"}))
    )

    assert stale.read_text(encoding="utf-8") == "envelope_of_rms\n42.0\n"
