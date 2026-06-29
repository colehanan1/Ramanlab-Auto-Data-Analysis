"""Shared, fully-deterministic fixtures for the protocol-switch test suite.

Both the golden-capture step and the protocol tests build the SAME on-disk
fixture through :func:`build_envelope_fixture`, so byte-for-byte comparisons are
meaningful. The layout mirrors real data on disk
(``<Dataset>/<batch>/angle_distance_rms_envelope/<stem>.csv``) and deliberately
contains NO Pi sidecars / ``session_metadata.txt`` — exactly the sidecar-less
case that distinguishes legacy (1260/2460 hardcoded windows) from v2 (config
fallback windows).

Values are fixed (no RNG) so results are reproducible across machines. Traces
are 3000 frames long so the odor windows (legacy 1260/2460 vs v2 1200/2400) sit
in the interior and actually change the integrated metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical dataset folder name so _canon_dataset resolves deterministically.
DATASET_NAME = "Hex-Control"
TRACE_LEN = 3000
MEASURE_COL = "envelope_of_rms"

# Two batches ("flies") x a handful of trials. Fixed per-(fly, trial) ramps whose
# slope/offset differ so rows are distinguishable but deterministic.
_FLIES = ("october_01_fly1", "october_02_fly1")
_TRIALS = (
    ("testing", 1),
    ("testing", 2),
    ("testing", 3),
    ("training", 1),
    ("training", 2),
)


def _trace(fly_idx: int, trial_type: str, trial_num: int) -> np.ndarray:
    """Deterministic 3000-frame ramp unique to (fly, trial)."""
    base = np.linspace(0.0, 100.0, TRACE_LEN, dtype=float)
    type_bump = 7.0 if trial_type == "training" else 0.0
    offset = float(fly_idx * 3 + trial_num) + type_bump
    return np.clip(base + offset, 0.0, 100.0)


def build_envelope_fixture(root: Path) -> Path:
    """Create the deterministic dataset under ``root`` and return the dataset dir.

    Returns the dataset root to pass to ``build_wide_csv`` (i.e. the directory
    named ``Hex-Control``).
    """
    dataset_root = Path(root) / DATASET_NAME
    for fly_idx, fly in enumerate(_FLIES):
        out_dir = dataset_root / fly / "angle_distance_rms_envelope"
        out_dir.mkdir(parents=True, exist_ok=True)
        for trial_type, trial_num in _TRIALS:
            stem = f"{fly}_{trial_type}_{trial_num}_angle_distance_rms_envelope"
            pd.DataFrame({MEASURE_COL: _trace(fly_idx, trial_type, trial_num)}).to_csv(
                out_dir / f"{stem}.csv", index=False
            )
    return dataset_root
