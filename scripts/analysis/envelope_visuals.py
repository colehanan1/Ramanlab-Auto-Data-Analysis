"""Visualisation utilities for Hilbert envelope outputs.

This script extends the notebook-style plotting snippets shared by the lab
into efficient, configurable CLI commands.  It can generate reaction matrix
figures (with optional alternate trial ordering) and produce per-fly envelope
plots using the float16 matrix emitted by :mod:`scripts.analysis.envelope_exports`.

Usage examples::

    # Build both matrix variants (testing order + trained-first)
    python scripts/analysis/envelope_visuals.py matrices \
        --matrix-npy /path/to/envelope_matrix_float16.npy \
        --codes-json /path/to/code_maps.json \
        --latency-sec 2.75

    # Produce envelope traces for every fly, grouped by odor
    python scripts/analysis/envelope_visuals.py envelopes \
        --matrix-npy /path/to/envelope_matrix_float16.npy \
        --codes-json /path/to/code_maps.json \
        --latency-sec 2.75

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec, transforms
from matplotlib.colors import BoundaryNorm, ListedColormap

from fbpipe.odor_constants import (  # noqa: F401 — re-exported for backward compat
    ODOR_CANON as _SHARED_ODOR_CANON,
    DATASET_ALIAS as _SHARED_DATASET_ALIAS,
)
from fbpipe.plot_style import apply_lab_style

apply_lab_style()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol selector – set once at pipeline start via set_protocol()
# ---------------------------------------------------------------------------
_ACTIVE_PROTOCOL: str = "legacy"


def set_protocol(protocol: str) -> None:
    """Set the active experiment protocol ('legacy' or 'v2')."""
    global _ACTIVE_PROTOCOL
    _ACTIVE_PROTOCOL = protocol


# ---------------------------------------------------------------------------
# Canonical mappings – imported from fbpipe.odor_constants (single source of truth).
# ODOR_CANON, DISPLAY_LABEL, ODOR_ORDER, TESTING_DATASET_ALIAS, DATASET_ALIAS,
# _canon_dataset, _odor_dataset_key, resolve_dataset_label are all imported
# at the top of this file.
# ---------------------------------------------------------------------------

ODOR_CANON: Mapping[str, str] = {
    "acv": "ACV",
    "apple cider vinegar": "ACV",
    "apple-cider-vinegar": "ACV",
    "3-octonol": "3-octonol",
    "3 octonol": "3-octonol",
    "3-octanol": "3-octonol",
    "3 octanol": "3-octonol",
    "benz": "Benz",
    "benzaldehyde": "Benz",
    "benz-ald": "Benz",
    "benzadhyde": "Benz",
    "benz-training": "Benz-Training",
    "benz training": "Benz-Training",
    "benz-training-24": "Benz-Training-24",
    "benz-training-24-2": "Benz-Training-24-2",
    "benz-training-24-02": "Benz-Training-24-02",
    "benz training 24": "Benz-Training-24",
    "benz-control": "Benz-Control",
    "benz-control-24-2": "Benz-Control-24-2",
    "benz-control-24-02": "Benz-Control-24-02",
    "benz_control": "Benz-Control",
    "benz control": "Benz-Control",
    "ethyl butyrate": "EB",
    "eb_control": "EB-Control",
    "eb control": "EB-Control",
    "eb-control": "EB-Control",
    "eb-training": "EB-Training",
    "eb-training(no-operant)": "EB-Training(No-Operant)",
    "eb-training-no-operant": "EB-Training(No-Operant)",
    "hex_control": "Hex-Control",
    "hex_control_24": "Hex-Control-24",
    "hex_control_36": "Hex-Control-36",
    "hexanol control": "Hex-Control",
    "hex-control": "Hex-Control",
    "hex-control-24": "Hex-Control-24",
    "hex-control-24-2": "Hex-Control-24-2",
    "hex-control-24-02": "Hex-Control-24-02",
    "hex-control-24-002": "Hex-Control-24-002",
    "hex-control-36": "Hex-Control-36",
    "hex-training": "Hex-Training",
    "hex-training-24": "Hex-Training-24",
    "hex-training-24-2": "Hex-Training-24-2",
    "hex-training-24-02": "Hex-Training-24-02",
    "hex-training-36": "Hex-Training-36",
    "hex control 24": "Hex-Control-24",
    "hex control 36": "Hex-Control-36",
    "hex training 24": "Hex-Training-24",
    "hex training 36": "Hex-Training-36",
    "acv-training": "ACV-Training",
    "air-training": "AIR-Training",
    "3oct-training": "3OCT-Training",
    "3oct training": "3OCT-Training",
    "3oct-training-24-2": "3OCT-Training-24-2",
    "3oct training 24 2": "3OCT-Training-24-2",
    "optogenetics benzaldehyde": "Benz-Training",
    "optogenetics benzaldehyde 1": "Benz-Training",
    "optogenetics ethyl butyrate": "EB-Training",
    "opto_eb": "EB-Training",
    "opto_eb(6-training)": "EB-Training(No-Operant)",
    "optogenetics apple cider vinegar": "ACV-Training",
    "optogenetics acv": "ACV-Training",
    "optogenetics hexanol": "Hex-Training",
    "optogenetics hex": "Hex-Training",
    "hexanol": "Hex-Training",
    "opto_hex": "Hex-Training",
    "opto_acv": "ACV-Training",
    "optogenetics air": "AIR-Training",
    "opto_air": "AIR-Training",
    "optogenetics 3-octanol": "3OCT-Training",
    "opto_3-oct": "3OCT-Training",
    # Legacy aliases – keep old opto_* names pointing to new canonical names
    "opto_benz": "Benz-Training",
    "opto_benz_1": "Benz-Training",
    "opto_eb_6_training": "EB-Training(No-Operant)",
    "10s_odor_benz": "10s_Odor_Benz",
    # v2 new datasets
    "cit-training": "Cit-Training",
    "cit_training": "Cit-Training",
    "cit training": "Cit-Training",
    "cit-control": "Cit-Control",
    "cit_control": "Cit-Control",
    "cit control": "Cit-Control",
    "citral": "Cit-Training",
    "lin-training": "Lin-Training",
    "lin_training": "Lin-Training",
    "lin training": "Lin-Training",
    "lin-control": "Lin-Control",
    "lin_control": "Lin-Control",
    "lin control": "Lin-Control",
    "linalool": "Lin-Training",
    "acv-control": "ACV-Control",
    "acv_control": "ACV-Control",
    "acv control": "ACV-Control",
    "3oct-control": "3OCT-Control",
    "3oct_control": "3OCT-Control",
    "3oct control": "3OCT-Control",
    "3oct-control-24-2": "3OCT-Control-24-2",
    "3oct_control_24_2": "3OCT-Control-24-2",
    "3oct control 24 2": "3OCT-Control-24-2",
    "3oct_training": "3OCT-Training",
}

DISPLAY_LABEL = {
    "ACV": "Apple Cider Vinegar",
    "3-octonol": "3-Octonol",
    "Benz": "Benzaldehyde",
    "10s_Odor_Benz": "Benzaldehyde",
    "EB": "Ethyl Butyrate",
    "EB-Control": "Ethyl Butyrate",
    "Hex-Control": "Hexanol",
    "Hex-Control-24": "Hexanol",
    "Hex-Control-24-2": "Hexanol",
    "Hex-Control-24-02": "Hexanol",
    "Hex-Control-24-002": "Hexanol",
    "Hex-Control-36": "Hexanol",
    "Benz-Control": "Benzaldehyde",
    "Benz-Control-24-2": "Benzaldehyde",
    "Benz-Control-24-02": "Benzaldehyde",
    "Benz-Training": "Benzaldehyde",
    "Benz-Training-24": "Benzaldehyde",
    "Benz-Training-24-2": "Benzaldehyde",
    "Benz-Training-24-02": "Benzaldehyde",
    "EB-Training": "Ethyl Butyrate",
    "EB-Training(No-Operant)": "Ethyl Butyrate (6-Training)",
    "ACV-Training": "Apple Cider Vinegar",
    "Hex-Training": "Hexanol",
    "Hex-Training-24": "Hexanol",
    "Hex-Training-24-2": "Hexanol",
    "Hex-Training-24-02": "Hexanol",
    "Hex-Training-36": "Hexanol",
    "AIR-Training": "AIR",
    "3OCT-Training": "3-Octonol",
    "3OCT-Training-24-2": "3-Octonol",
    # v2 new datasets
    "Cit-Training": "Citral",
    "Cit-Control": "Citral",
    "Lin-Training": "Linalool",
    "Lin-Control": "Linalool",
    "ACV-Control": "Apple Cider Vinegar",
    "3OCT-Control": "3-Octonol",
    "3OCT-Control-24-2": "3-Octonol",
}

ODOR_ORDER = [
    "ACV",
    "ACV-Training",
    "3-octonol",
    "Benz",
    "EB",
    "10s_Odor_Benz",
    "Benz-Training",
    "EB-Training",
    "EB-Training(No-Operant)",
    "EB-Control",
    "Benz-Training-24",
    "Benz-Training-24-2",
    "Benz-Control",
    "Benz-Control-24-2",
    "Hex-Training",
    "Hex-Training-24",
    "Hex-Training-24-2",
    "Hex-Training-36",
    "Hex-Control",
    "Hex-Control-24",
    "Hex-Control-24-2",
    "Hex-Control-24-002",
    "Hex-Control-36",
    "AIR-Training",
    "3OCT-Training",
    "3OCT-Training-24-2",
    # v2 new datasets
    "Cit-Training",
    "Cit-Control",
    "Lin-Training",
    "Lin-Control",
    "ACV-Control",
    "3OCT-Control",
    "3OCT-Control-24-2",
]

REACTION_RATE_ODOR_ORDER = [
    "3-Octonol",
    "Apple Cider Vinegar",
    "Benzaldehyde",
    "Citral",
    "Ethyl Butyrate",
    "Hexanol",
    "Linalool",
]
REACTION_RATE_ODOR_INDEX = {
    label.casefold(): idx for idx, label in enumerate(REACTION_RATE_ODOR_ORDER)
}

TRAINED_FIRST_ORDER = (2, 4, 5, 1, 3, 6, 7, 8, 9)
HEXANOL_LABEL = "Hexanol"

PRIMARY_ODOR_LABEL = {
    "EB-Control": "Ethyl Butyrate",
    "Hex-Control": HEXANOL_LABEL,
    "Hex-Control-24": HEXANOL_LABEL,
    "Hex-Control-24-2": HEXANOL_LABEL,
    "Hex-Control-24-02": HEXANOL_LABEL,
    "Hex-Control-24-002": HEXANOL_LABEL,
    "Hex-Control-36": HEXANOL_LABEL,
    "Benz-Control": "Benzaldehyde",
    "Benz-Control-24-2": "Benzaldehyde",
    "Benz-Control-24-02": "Benzaldehyde",
    # v2 datasets
    "Cit-Control": "Citral",
    "Lin-Control": "Linalool",
    "ACV-Control": "Apple Cider Vinegar",
    "3OCT-Control": "3-Octonol",
    "3OCT-Control-24-2": "3-Octonol",
}

TRAINING_ODOR_SCHEDULE_ACV = {
    1: "Apple Cider Vinegar",
    2: "Apple Cider Vinegar",
    3: "Apple Cider Vinegar",
    4: "Apple Cider Vinegar",
    5: HEXANOL_LABEL,
    6: "Apple Cider Vinegar",
    7: HEXANOL_LABEL,
    8: "Apple Cider Vinegar",
}

TRAINING_ODOR_SCHEDULE = {
    1: "Benzaldehyde",
    2: "Benzaldehyde",
    3: "Benzaldehyde",
    4: "Benzaldehyde",
    5: HEXANOL_LABEL,
    6: "Benzaldehyde",
    7: HEXANOL_LABEL,
    8: "Benzaldehyde",
}

TRAINING_ODOR_SCHEDULE_AIR = {
    1: "AIR",
    2: "AIR",
    3: "AIR",
    4: "AIR",
    5: HEXANOL_LABEL,
    6: "AIR",
    7: HEXANOL_LABEL,
    8: "AIR",
}

TRAINING_ODOR_SCHEDULE_EB = {
    1: "Ethyl Butyrate",
    2: "Ethyl Butyrate",
    3: "Ethyl Butyrate",
    4: "Ethyl Butyrate",
    5: HEXANOL_LABEL,
    6: "Ethyl Butyrate",
    7: HEXANOL_LABEL,
    8: "Ethyl Butyrate",
}

TRAINING_ODOR_SCHEDULE_EB_6TRAINING = {
    1: "Ethyl Butyrate",
    2: "Ethyl Butyrate",
    3: "Ethyl Butyrate",
    4: "Ethyl Butyrate",
    5: "Ethyl Butyrate",
    6: "Ethyl Butyrate",
}

TRAINING_ODOR_SCHEDULE_HEX = {
    1: HEXANOL_LABEL,
    2: HEXANOL_LABEL,
    3: HEXANOL_LABEL,
    4: HEXANOL_LABEL,
    5: "Apple Cider Vinegar",
    6: HEXANOL_LABEL,
    7: "Apple Cider Vinegar",
    8: HEXANOL_LABEL,
}

TRAINING_ODOR_SCHEDULE_3OCT = {
    1: "3-Octonol",
    2: "3-Octonol",
    3: "3-Octonol",
    4: "3-Octonol",
    5: HEXANOL_LABEL,
    6: "3-Octonol",
    7: HEXANOL_LABEL,
    8: "3-Octonol",
}

TESTING_DATASET_ALIAS = {
    "Hex-Training": "Hex-Control",
    "Hex-Training-24": "Hex-Control",
    "Hex-Training-24-2": "Hex-Control",
    "Hex-Training-24-02": "Hex-Control",
    "Hex-Training-36": "Hex-Control",
    "Hex-Control-24": "Hex-Control",
    "Hex-Control-24-2": "Hex-Control",
    "Hex-Control-24-02": "Hex-Control",
    "Hex-Control-24-002": "Hex-Control",
    "Hex-Control-36": "Hex-Control",
    "EB-Training": "EB-Control",
    "EB-Training(No-Operant)": "EB-Control",
    "Benz-Training": "Benz-Control",
    "Benz-Training-24": "Benz-Control",
    "Benz-Training-24-2": "Benz-Control",
    "Benz-Training-24-02": "Benz-Control",
    "Benz-Control-24-2": "Benz-Control",
    "Benz-Control-24-02": "Benz-Control",
    "ACV-Training": "ACV",
    "3OCT-Training": "3OCT-Training",
    "3OCT-Training-24-2": "3OCT-Training",
    "3OCT-Control-24-2": "3OCT-Control-24-2",
    # v2 datasets – training aliases to their control counterpart
    "Cit-Training": "Cit-Control",
    "Lin-Training": "Lin-Control",
}
NON_REACTIVE_SPAN_PX = 7.5
TRAINING_EXTENDED_ODOR_TRIALS = frozenset({4, 6, 8})
TRAINING_EXTENDED_ODOR_OFF_S = 65.0
TRAINING_DISCRIMINATE_ODOR_TRIALS = frozenset({5, 7})
LIGHT_START_EARLY_TRIALS = frozenset({1, 2, 3})
LIGHT_START_LATE_TRIALS = frozenset({4, 6, 8})
LIGHT_START_EARLY_S = 35.0
LIGHT_START_LATE_S = 40.0
ODOR_PLUS_LIGHT_COLOR = "#9e9e9e"
ODOR_PLUS_LIGHT_ALPHA = 0.20
ODOR_PLUS_LIGHT_LINGER_ALPHA = 0.12
DISCRIMINATE_ODOR_COLOR = "#4d4d4d"
DISCRIMINATE_ODOR_ALPHA = 0.28
DISCRIMINATE_ODOR_LINGER_ALPHA = 0.18
DISCRIMINATE_ODOR_LABEL = "Discriminate odor"
ODOR_PLUS_LIGHT_LABEL = "Odor + light"


# ---------------------------------------------------------------------------
# Utility helpers


def _canon_dataset(value: str) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    stripped = value.strip()
    key = stripped.lower()
    if key.endswith("-flagged"):
        base_key = key[: -len("-flagged")]
        canon = ODOR_CANON.get(base_key)
        if canon is not None:
            return f"{canon}-flagged"
        return stripped
    canon = ODOR_CANON.get(key)
    if canon is not None:
        return canon
    return stripped


def _odor_dataset_key(dataset_canon: str) -> str:
    dataset_text = str(dataset_canon).strip() if isinstance(dataset_canon, str) else "UNKNOWN"
    if not dataset_text:
        return "UNKNOWN"
    lower = dataset_text.lower()
    if lower.endswith("-flagged"):
        base = dataset_text[: -len("-flagged")].strip()
        return ODOR_CANON.get(base.lower(), base)
    return ODOR_CANON.get(lower, dataset_text)


def _safe_dirname(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "export"


def canonical_dataset(value: str) -> str:
    """Return the canonical ODOR identifier for *value*."""

    return _canon_dataset(value)


def resolve_dataset_label(values: Sequence[str] | str) -> str:
    """Return a human-readable label for one or more dataset identifiers."""

    if isinstance(values, str):
        candidates = {_canon_dataset(values)} if values else set()
    else:
        candidates = {_canon_dataset(val) for val in values if isinstance(val, str) and val}

    candidates = {val for val in candidates if val}
    if not candidates:
        return "UNKNOWN"

    if len(candidates) == 1:
        key = next(iter(candidates))
        return DISPLAY_LABEL.get(key, key)

    pretty = [DISPLAY_LABEL.get(key, key) for key in sorted(candidates)]
    return f"Mixed ({'+'.join(pretty)})"


def resolve_dataset_output_dir(base: Path, values: Sequence[str] | str) -> Path:
    """
    Return the output directory for the provided dataset identifiers.

    Creates separate folders for each unique dataset to avoid mixing results:
    - Hex-Training → "Hex-Training" folder
    - Hex-Control → "Hex-Control" folder
    - EB-Training → "EB-Training" folder
    - EB-Control → "EB-Control" folder
    - etc.
    """
    if isinstance(values, str):
        candidates = {_canon_dataset(values)} if values else set()
    else:
        candidates = {_canon_dataset(val) for val in values if isinstance(val, str) and val}

    candidates = {val for val in candidates if val}
    if not candidates:
        return base / "UNKNOWN"

    if len(candidates) == 1:
        # Single dataset: use the canonical name directly (preserves Hex-Training, Hex-Control, etc.)
        dataset_name = next(iter(candidates))
        return base / _safe_dirname(dataset_name)

    # Multiple datasets: create combined folder name
    sorted_names = sorted(candidates)
    combined_name = "_".join(sorted_names)
    return base / _safe_dirname(combined_name)


def should_write(path: Path, overwrite: bool) -> bool:
    """Return ``True`` if *path* should be written, honouring overwrite policy."""

    lower_path = str(path).lower()
    reaction_artifact = "reaction_matrix" in lower_path or "reaction_prediction" in lower_path

    if path.exists():
        if reaction_artifact:
            return True
        if not overwrite:
            return False
    elif path.parent is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def _trial_num(label: str) -> int:
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else -1


def _trial_odor_window_seconds(
    *,
    trial_label: str,
    trial_type: str,
    odor_on_s: float,
    odor_off_s: float,
    odor_latency_s: float,
) -> tuple[float, float]:
    """Return latency-shifted odor on/off times (seconds) for one trial."""

    on_cmd = float(odor_on_s)
    off_cmd = float(odor_off_s)
    # v2 protocol: all trials use standard 30s odor, no extended trials
    if _ACTIVE_PROTOCOL == "legacy":
        if trial_type == "training" and _trial_num(trial_label) in TRAINING_EXTENDED_ODOR_TRIALS:
            off_cmd = TRAINING_EXTENDED_ODOR_OFF_S

    off_cmd = max(off_cmd, on_cmd)
    latency = max(float(odor_latency_s), 0.0)
    return on_cmd + latency, off_cmd + latency


def _trial_light_start_seconds(*, trial_label: str, trial_type: str) -> float | None:
    """Return light pulsing start time (seconds) for training trials."""

    if trial_type != "training":
        return None
    # v2 protocol: no variable light start times (light schedule is in config)
    if _ACTIVE_PROTOCOL == "v2":
        return None
    trial_num = _trial_num(trial_label)
    if trial_num in LIGHT_START_EARLY_TRIALS:
        return LIGHT_START_EARLY_S
    if trial_num in LIGHT_START_LATE_TRIALS:
        return LIGHT_START_LATE_S
    return None


def _is_discriminate_odor_trial(*, trial_label: str, trial_type: str) -> bool:
    """Return True for training trials using the discriminate-only odor condition."""

    # v2 protocol: no discriminate odor trials (all 6 training trials are same odor)
    if _ACTIVE_PROTOCOL == "v2":
        return False
    if trial_type != "training":
        return False
    return _trial_num(trial_label) in TRAINING_DISCRIMINATE_ODOR_TRIALS


def _trained_label(dataset_canon: str) -> str:
    dataset_key = _odor_dataset_key(dataset_canon)
    return PRIMARY_ODOR_LABEL.get(
        dataset_key, DISPLAY_LABEL.get(dataset_key, dataset_key)
    )


def _display_odor_v2(dataset_canon: str, trial_label: str) -> str:
    """V2 protocol: extract odor name from trial label suffix or fall back to dataset label."""
    label_str = str(trial_label)
    # Try to extract odor suffix from label like "testing_2_Hexanol" or "training_1_ACV"
    match = re.match(r"(?:testing|training)_\d+_(.+)", label_str, re.IGNORECASE)
    if match:
        odor_suffix = match.group(1)
        if odor_suffix.lower() == "lightonly":
            return "Light Only"
        # Map through DISPLAY_LABEL for consistency
        return DISPLAY_LABEL.get(odor_suffix, odor_suffix)

    # No suffix: for training trials, return the trained odor
    dataset_key = _odor_dataset_key(dataset_canon)
    label_lower = label_str.lower()
    if "training" in label_lower:
        return DISPLAY_LABEL.get(dataset_key, dataset_key)

    # testing_9 without suffix → Light Only
    number = _trial_num(label_str)
    if number == 9 and "testing" in label_lower:
        return "Light Only"

    return DISPLAY_LABEL.get(dataset_key, dataset_key)


def _display_odor(dataset_canon: str, trial_label: str) -> str:
    if _ACTIVE_PROTOCOL == "v2":
        return _display_odor_v2(dataset_canon, trial_label)
    dataset_key = _odor_dataset_key(dataset_canon)
    number = _trial_num(trial_label)
    label_lower = str(trial_label).lower()

    if "training" in label_lower:
        # Select the appropriate training schedule based on dataset
        if dataset_key == "AIR-Training":
            odor_name = TRAINING_ODOR_SCHEDULE_AIR.get(number)
            if odor_name:
                return odor_name
        elif dataset_key in ("3OCT-Training", "3OCT-Training-24-2", "3OCT-Control-24-2"):
            odor_name = TRAINING_ODOR_SCHEDULE_3OCT.get(number)
            if odor_name:
                return odor_name
        elif dataset_key in ("EB-Training", "EB-Control"):
            odor_name = TRAINING_ODOR_SCHEDULE_EB.get(number)
            if odor_name:
                return odor_name
        elif dataset_key == "EB-Training(No-Operant)":
            odor_name = TRAINING_ODOR_SCHEDULE_EB_6TRAINING.get(number)
            if odor_name:
                return odor_name
        elif dataset_key in (
            "Hex-Training",
            "Hex-Training-24",
            "Hex-Training-24-2",
            "Hex-Training-24-02",
            "Hex-Training-36",
            "Hex-Control",
            "Hex-Control-24",
            "Hex-Control-24-2",
            "Hex-Control-24-02",
            "Hex-Control-24-002",
            "Hex-Control-36",
        ):
            odor_name = TRAINING_ODOR_SCHEDULE_HEX.get(number)
            if odor_name:
                return odor_name
        elif dataset_key in ("ACV-Training", "ACV"):
            odor_name = TRAINING_ODOR_SCHEDULE_ACV.get(number)
            if odor_name:
                return odor_name
        else:
            odor_name = TRAINING_ODOR_SCHEDULE.get(number)
            if odor_name:
                return odor_name
        return DISPLAY_LABEL.get(dataset_key, dataset_key)

    # Handle AIR-Training testing trials
    if dataset_key == "AIR-Training":
        if number in (1, 3):
            return HEXANOL_LABEL
        if number in (2, 4, 5):
            return "AIR"
        if number == 6:
            return "Apple Cider Vinegar"
        if number == 7:
            return "Ethyl Butyrate"
        if number == 8:
            return "Benzaldehyde"
        if number == 9:
            return "Citral"
        if number == 10:
            return "3-Octonol"

    # Handle 3OCT-Training testing trials
    if dataset_key in ("3OCT-Training", "3OCT-Training-24-2", "3OCT-Control-24-2"):
        if number in (1, 3):
            return HEXANOL_LABEL
        if number in (2, 4, 5):
            return "3-Octonol"
        if number == 6:
            return "Apple Cider Vinegar"
        if number == 7:
            return "Ethyl Butyrate"
        if number == 8:
            return "Benzaldehyde"
        if number == 9:
            return "Citral"
        if number == 10:
            return "Linalool"

    dataset_for_testing = TESTING_DATASET_ALIAS.get(dataset_key, dataset_key)

    if dataset_for_testing == "Hex-Control":
        if number in (1, 3):
            return "Apple Cider Vinegar"
        if number in (2, 4, 5):
            return HEXANOL_LABEL
    else:
        if number in (1, 3):
            return HEXANOL_LABEL
    if number in (2, 4, 5):
        return DISPLAY_LABEL.get(
            dataset_for_testing, DISPLAY_LABEL.get(dataset_key, dataset_key)
        )

    mapping = {
        "ACV": {
            6: "3-Octonol",
            7: "Ethyl Butyrate",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "3-octonol": {6: "Benzaldehyde", 7: "Citral", 8: "Linalool"},
        "Benz": {6: "Citral", 7: "Linalool"},
        "Benz-Control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
        "EB": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "EB-Control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "EB-Training(No-Operant)": {
            # Testing trials (6-10): same as EB-Training
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "Hex-Control": {
            6: "Benzaldehyde",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
        "ACV-Training": {
            6: "3-Octonol",
            7: "Ethyl Butyrate",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "10s_Odor_Benz": {6: "Benzaldehyde", 7: "Benzaldehyde"},
    }

    if dataset_for_testing in mapping:
        return mapping[dataset_for_testing].get(number, trial_label)
    return mapping.get(dataset_key, {}).get(number, trial_label)


def _normalise_fly_label(value: object) -> str:
    text = "UNKNOWN" if value is None else str(value).strip()
    return text or "UNKNOWN"


def _normalise_fly_number(value: object) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    if not text:
        return "UNKNOWN"
    lowered = text.lower()
    if lowered in {"nan", "none", "unknown"}:
        return "UNKNOWN"
    return text


def _normalise_fly_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "fly" not in df.columns:
        df["fly"] = "UNKNOWN"
    else:
        df["fly"] = df["fly"].apply(_normalise_fly_label)

    if "fly_number" not in df.columns:
        df["fly_number"] = "UNKNOWN"
    df["fly_number"] = df["fly_number"].apply(_normalise_fly_number)
    return df


def _fly_sort_key(fly: str, fly_number: str) -> Tuple[str, int, object]:
    fly_norm = _normalise_fly_label(fly)
    number_norm = _normalise_fly_number(fly_number)
    try:
        number_val = int(float(number_norm))
        return (fly_norm.lower(), 0, number_val)
    except ValueError:
        return (fly_norm.lower(), 1, number_norm.lower())


def _fly_row_label(fly: str, fly_number: str) -> str:
    fly_norm = _normalise_fly_label(fly)
    number_norm = _normalise_fly_number(fly_number)
    if number_norm == "UNKNOWN":
        return fly_norm
    return f"{fly_norm} — Fly {number_norm}"


def _is_trained_odor(dataset_canon: str, odor_name: str) -> bool:
    trained = _trained_label(dataset_canon)
    return str(odor_name).strip().lower() == str(trained).strip().lower()


def _resolve_trace_len(trace_len: object, max_len: int) -> int | None:
    try:
        value = float(trace_len)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return max(0, min(int(round(value)), max_len))


def _extract_env_row(env_row: np.ndarray, trace_len: object = None) -> np.ndarray:
    env = env_row.astype(float, copy=False)
    resolved_len = _resolve_trace_len(trace_len, env.size)
    if resolved_len is not None:
        env = env[:resolved_len]
    elif np.isnan(env).any():
        finite_idx = np.flatnonzero(np.isfinite(env))
        if finite_idx.size == 0:
            return np.empty(0, dtype=float)
        env = env[: finite_idx[-1] + 1]

    if env.size == 0 or not np.isfinite(env).any():
        return np.empty(0, dtype=float)
    return env


def is_non_reactive_span(global_min: object, global_max: object, *, threshold: float = NON_REACTIVE_SPAN_PX) -> bool:
    """Return ``True`` when the provided span suggests no training reaction."""

    try:
        gmin = float(global_min)
    except (TypeError, ValueError):
        return False
    try:
        gmax = float(global_max)
    except (TypeError, ValueError):
        return False

    if not (math.isfinite(gmin) and math.isfinite(gmax)):
        return False
    return abs(gmax - gmin) <= float(threshold)


def _span_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return span-defining columns, preferring trimmed values when available."""

    if {"trimmed_global_min", "trimmed_global_max"}.issubset(df.columns):
        trimmed_min = pd.to_numeric(df["trimmed_global_min"], errors="coerce")
        trimmed_max = pd.to_numeric(df["trimmed_global_max"], errors="coerce")
        base_min = pd.to_numeric(df.get("global_min"), errors="coerce")
        base_max = pd.to_numeric(df.get("global_max"), errors="coerce")
        gmin = trimmed_min.where(trimmed_min.notna(), base_min)
        gmax = trimmed_max.where(trimmed_max.notna(), base_max)
    else:
        gmin = pd.to_numeric(df.get("global_min"), errors="coerce")
        gmax = pd.to_numeric(df.get("global_max"), errors="coerce")
    if gmin is None:
        gmin = pd.Series(np.nan, index=df.index, dtype=float)
    if gmax is None:
        gmax = pd.Series(np.nan, index=df.index, dtype=float)
    if not isinstance(gmin, pd.Series):
        gmin = pd.Series(gmin, index=df.index)
    if not isinstance(gmax, pd.Series):
        gmax = pd.Series(gmax, index=df.index)
    return gmin, gmax


def compute_non_reactive_flags(
    df: pd.DataFrame,
    *,
    threshold: float = NON_REACTIVE_SPAN_PX,
    include_tracking: bool = True,
    flagged_flies_csv: str = "",
) -> pd.Series:
    """
    Return a boolean mask flagging flies as non-reactive.

    When *flagged_flies_csv* is provided and the file exists, flies are flagged
    based on the CSV truth table: flies with ``FLY-State != 1`` are excluded.
    Otherwise falls back to the original span-threshold + tracking criteria.

    Args:
        df: DataFrame with fly data
        threshold: Motion span threshold (pixels) – used only as fallback
        include_tracking: If True, also flag flies with poor tracking quality (fallback only)
        flagged_flies_csv: Path to the flagged-flies truth CSV

    Returns:
        Boolean Series where True = non-reactive / excluded fly
    """
    # --- CSV-based flagging (preferred when available) ---
    if flagged_flies_csv:
        try:
            _src = str(Path(__file__).resolve().parents[2] / "src")
            if _src not in sys.path:
                sys.path.insert(0, _src)
            from fbpipe.config import load_flagged_fly_exclusions
            exclusions = load_flagged_fly_exclusions(flagged_flies_csv)
            if exclusions:
                ds = df["dataset"].astype(str).str.strip() if "dataset" in df.columns else pd.Series("", index=df.index)
                fl = df["fly"].astype(str).str.strip() if "fly" in df.columns else pd.Series("", index=df.index)
                fn = df["fly_number"].astype(str).str.strip() if "fly_number" in df.columns else pd.Series("", index=df.index)
                keys = list(zip(ds, fl, fn))
                mask = pd.Series([k in exclusions for k in keys], index=df.index)
                n_flagged = mask.sum()
                if n_flagged > 0:
                    logger.info(f"Flagged {n_flagged} rows via CSV truth table ({len(exclusions)} excluded flies)")
                return mask
            return pd.Series(False, index=df.index, dtype=bool)
        except Exception as exc:
            logger.warning(f"Failed to load flagged flies CSV, falling back to threshold: {exc}")

    # --- Fallback: span-threshold + tracking criteria ---
    gmin, gmax = _span_columns(df)
    span = pd.Series(np.nan, index=df.index, dtype=float)
    if isinstance(gmax, pd.Series) and isinstance(gmin, pd.Series):
        span = (gmax - gmin).abs()
    non_reactive_motion = gmin.notna() & gmax.notna() & span.le(float(threshold))
    non_reactive_motion = non_reactive_motion.fillna(False)

    non_reactive_tracking = pd.Series(False, index=df.index)
    if include_tracking and "tracking_flagged" in df.columns:
        non_reactive_tracking = df["tracking_flagged"].fillna(False).astype(bool)

    combined = non_reactive_motion | non_reactive_tracking

    n_motion_only = (non_reactive_motion & ~non_reactive_tracking).sum()
    n_tracking_only = (~non_reactive_motion & non_reactive_tracking).sum()
    n_both = (non_reactive_motion & non_reactive_tracking).sum()
    n_total = combined.sum()

    if n_total > 0:
        logger.info(
            f"Flagged {n_total} flies as non-reactive: "
            f"{n_motion_only} low motion only, "
            f"{n_tracking_only} poor tracking only, "
            f"{n_both} both criteria"
        )

    return combined


def non_reactive_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series for the ``_non_reactive`` column, if present."""

    series = df.get("_non_reactive")
    if series is None:
        return pd.Series(False, index=df.index, dtype=bool)

    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=df.index)

    mask = series.astype(bool)
    if not mask.index.equals(df.index):
        mask = mask.reindex(df.index, fill_value=False)
    return mask.fillna(False)


def _compute_theta(
    env: np.ndarray, fps: float, baseline_until_s: float, std_mult: float
) -> float:
    """Compute the response threshold using only the pre-command baseline."""

    if env.size == 0 or fps <= 0:
        return math.nan

    before_end = min(int(round(baseline_until_s * fps)), env.size)
    if before_end <= 0:
        return math.nan

    window = env[:before_end]
    baseline = float(np.nanmedian(window))
    mad = float(np.nanmedian(np.abs(window - baseline)))
    sigma = 1.4826 * mad
    return float(baseline + std_mult * sigma)


def filter_and_validate_trial_type(
    df: pd.DataFrame,
    target_type: str = "testing",
    fallback_to_all: bool = False
) -> tuple[pd.DataFrame, bool, str]:
    """
    Robustly filter dataframe to specified trial type with diagnostics.

    This function handles missing or malformed trial_type columns gracefully,
    normalizes whitespace and casing, and provides clear diagnostic messages.

    Args:
        df: Input dataframe with potential trial_type column
        target_type: Target trial type to filter for (e.g., "testing", "training")
        fallback_to_all: If True, return all rows when target type not found

    Returns:
        Tuple of (filtered_df, success, reason):
        - filtered_df: DataFrame filtered to target type (or empty if failed)
        - success: Boolean indicating if filtering succeeded
        - reason: String describing the result (for logging/debugging)
    """
    if "trial_type" not in df.columns:
        logger.warning(
            "Column 'trial_type' not found in matrix; "
            "proceeding with all available rows"
        )
        return df, True, "no_trial_type_column"

    # Normalize: strip whitespace, lowercase
    df = df.copy()
    df["trial_type_clean"] = (
        df["trial_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Count available trial types
    available_types = df["trial_type_clean"].unique()
    available_counts = df["trial_type_clean"].value_counts().to_dict()
    logger.info(
        f"Available trial types: {available_counts}"
    )

    # Filter to target type
    mask = df["trial_type_clean"] == target_type.lower()
    filtered = df.loc[mask].copy()

    if not filtered.empty:
        logger.info(
            f"Filtered to '{target_type}' trials: "
            f"{len(filtered)} rows remain (from {len(df)} total)"
        )
        return filtered, True, f"filtered_to_{target_type}"

    # Target type not found
    logger.warning(
        f"No '{target_type}' trials found in matrix. "
        f"Available types: {list(available_types)}"
    )

    if fallback_to_all:
        logger.warning(
            f"Falling back to ALL trials ({len(df)} rows) "
            f"for visualization"
        )
        return df, True, "fallback_to_all"
    else:
        logger.error(
            f"Cannot proceed without '{target_type}' trials"
        )
        return pd.DataFrame(), False, "no_target_trials"


def _load_matrix(matrix_path: Path, codes_json: Path) -> tuple[pd.DataFrame, list[str]]:
    matrix = np.load(matrix_path, allow_pickle=False)
    with codes_json.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    ordered_cols: list[str] = list(meta["column_order"])
    code_maps: Mapping[str, Mapping[str, int]] = meta["code_maps"]
    metric_cols_meta = set(meta.get("metric_columns", []) or [])
    env_cols_meta = list(meta.get("env_columns", []) or [])
    df = pd.DataFrame(matrix, columns=ordered_cols)

    decode_candidates = ["dataset", "fly", "fly_number", "trial_type", "trial_label"]
    rev_maps = {
        col: {int(code): label for label, code in mapping.items()}
        for col, mapping in code_maps.items()
        if col in decode_candidates or col == "fps"
    }

    for col in decode_candidates:
        if col not in df.columns:
            continue
        values = np.rint(df[col].to_numpy(np.float32, copy=False)).astype(np.int32, copy=False)
        mapped = pd.Series(values).map(rev_maps.get(col, {})).fillna("UNKNOWN")
        df[col] = mapped

    if "fps" in df.columns:
        fps_codes = np.rint(df["fps"].to_numpy(np.float32, copy=False)).astype(np.int32, copy=False)
        if "fps" in rev_maps:
            fps_strings = pd.Series(fps_codes).map(rev_maps["fps"]).astype(str)
            df["fps"] = pd.to_numeric(fps_strings, errors="coerce")
        else:
            df["fps"] = pd.to_numeric(df["fps"], errors="coerce")
    else:
        df["fps"] = np.nan

    if env_cols_meta:
        env_cols = [col for col in env_cols_meta if col in df.columns]
    else:
        env_cols = [
            col
            for col in ordered_cols
            if col
            not in {
                "fps",
                "dataset",
                "fly",
                "fly_number",
                "trial_type",
                "trial_label",
            }
            and col not in metric_cols_meta
        ]
    prefixed = [col for col in env_cols if col.startswith("dir_val_")]
    if prefixed:
        env_cols = prefixed
    return df, env_cols


# ---------------------------------------------------------------------------
# Reaction matrix generation


@dataclass
class MatrixPlotConfig:
    matrix_npy: Path
    codes_json: Path
    out_dir: Path
    latency_sec: float
    fps_default: float = 40.0
    before_sec: float = 30.0
    during_sec: float = 30.0
    after_window_sec: float = 30.0
    threshold_std_mult: float = 3.0
    min_samples_over: int = 20
    row_gap: float = 0.6
    height_per_gap_in: float = 3.0
    bottom_shift_in: float = 0.5
    trial_orders: Sequence[str] = field(default_factory=lambda: ("observed", "trained-first"))
    include_hexanol: bool = True
    overwrite: bool = True


def _score_trial(env: np.ndarray, fps: float, cfg: MatrixPlotConfig) -> tuple[int, int]:
    if env.size == 0:
        return (0, 0)

    fps = fps if math.isfinite(fps) and fps > 0 else cfg.fps_default
    before_end = int(round(cfg.before_sec * fps))
    shift = int(round(cfg.latency_sec * fps))
    during_start = before_end + shift
    during_end = during_start + int(round(cfg.during_sec * fps))
    after_end = during_end + int(round(cfg.after_window_sec * fps))

    total = env.size
    before_end = max(0, min(before_end, total))
    during_start = max(before_end, min(during_start, total))
    during_end = max(during_start, min(during_end, total))
    after_end = max(during_end, min(after_end, total))

    before = env[:before_end]
    during = env[during_start:during_end]
    after = env[during_end:after_end]

    if before.size == 0:
        return (0, 0)

    baseline = float(np.nanmedian(before))
    mad = float(np.nanmedian(np.abs(before - baseline)))
    sigma = 1.4826 * mad
    theta = float(baseline + cfg.threshold_std_mult * sigma)
    during_hit = int(np.sum(during > theta) >= cfg.min_samples_over) if during.size else 0
    after_hit = int(np.sum(after > theta) >= cfg.min_samples_over) if after.size else 0
    return during_hit, after_hit


def _normalise_odor_label(value: object) -> str:
    text = "UNKNOWN" if value is None else str(value).strip()
    return text or "UNKNOWN"


def reaction_rate_stats_from_rows(
    df: pd.DataFrame,
    dataset_canon: str,
    *,
    include_hexanol: bool,
    context: str,
    trial_col: str = "trial",
    reaction_col: str = "during_hit",
    separate_presentations: bool = False,
) -> pd.DataFrame:
    """Aggregate per-odor reaction rates from row-wise trial data."""

    if df.empty:
        raise ValueError("No rows available to compute reaction-rate statistics.")

    missing = [col for col in (trial_col, reaction_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for reaction stats: {', '.join(missing)}")

    working = df[[trial_col, reaction_col]].copy()
    working["trial_num"] = working[trial_col].map(_trial_num)
    working["odor"] = working[trial_col].map(lambda trial: _display_odor(dataset_canon, trial))

    if not include_hexanol:
        mask = working["odor"].astype(str).str.strip().str.casefold() != HEXANOL_LABEL.casefold()
        working = working.loc[mask].copy()
        if working.empty:
            raise RuntimeError("All odors were removed after excluding Hexanol.")

    working["reaction_flag"] = pd.to_numeric(working[reaction_col], errors="coerce").fillna(0).astype(int)
    group_cols = ["trial_num", "odor"] if separate_presentations else ["odor"]
    stats_df = (
        working.groupby(group_cols, dropna=False)["reaction_flag"]
        .agg(num_reactions="sum", num_trials="size")
        .reset_index()
    )
    stats_df["rate"] = np.where(
        stats_df["num_trials"] > 0,
        stats_df["num_reactions"] / stats_df["num_trials"],
        0.0,
    )

    zero_trial_mask = stats_df["num_trials"] == 0
    if zero_trial_mask.any():
        missing_odors = stats_df.loc[zero_trial_mask, "odor"].tolist()
        logger.warning("Odors with zero trials encountered in %s: %s", context, missing_odors)

    # Use PRIMARY_ODOR_LABEL for control datasets to get the correct trained odor
    highlight_label = _trained_label(dataset_canon)
    stats_df["is_trained"] = stats_df["odor"].astype(str).str.casefold() == highlight_label.casefold()

    if separate_presentations:
        stats_df = stats_df.assign(
            _order_trial=pd.to_numeric(stats_df["trial_num"], errors="coerce").fillna(10**9).astype(int),
            _order_label=stats_df["odor"].astype(str).str.casefold(),
        )
        stats_df = stats_df.sort_values(
            ["_order_trial", "_order_label"], ascending=[True, True], kind="mergesort"
        ).drop(columns=["_order_trial", "_order_label"])
    else:
        stats_df = stats_df.assign(
            _order_key=stats_df["odor"].map(
                lambda value: REACTION_RATE_ODOR_INDEX.get(str(value).casefold(), len(REACTION_RATE_ODOR_INDEX))
            ),
            _order_label=stats_df["odor"].astype(str).str.casefold(),
        )
        stats_df = stats_df.sort_values(
            ["_order_key", "_order_label"], ascending=[True, True], kind="mergesort"
        ).drop(columns=["_order_key", "_order_label"])
    stats_df = stats_df.reset_index(drop=True)

    logger.debug(
        "Per-odor reaction stats for %s:\n%s",
        context,
        stats_df.to_string(index=False),
    )
    logger.info(
        "Reaction-rate bar order for %s: %s",
        context,
        stats_df["odor"].astype(str).tolist(),
    )
    return stats_df


def plot_reaction_rate_bars(
    ax: plt.Axes,
    stats_df: pd.DataFrame,
    *,
    title: str,
) -> None:
    """Plot a reaction-rate bar chart into the provided axis."""

    if stats_df.empty:
        ax.set_visible(False)
        return

    if "trial_num" in stats_df.columns:
        stats_df = stats_df.sort_values(["trial_num", "odor"], kind="mergesort").reset_index(drop=True)

    x = np.arange(len(stats_df))
    colors = [
        "tab:blue" if bool(is_trained) else "0.6" for is_trained in stats_df["is_trained"]
    ]

    bars = ax.bar(
        x,
        stats_df["rate"].to_numpy(float),
        color=colors,
        edgecolor="black",
        linewidth=0.75,
    )
    ax.set_xticks(x)

    # Display trained odor in ALL CAPS and blue
    labels = [
        str(odor).upper() if bool(is_trained) else str(odor)
        for odor, is_trained in zip(stats_df["odor"], stats_df["is_trained"])
    ]
    ax.set_xticklabels(labels, rotation=35, ha="right")

    # Color trained odor labels blue
    for tick, is_trained in zip(ax.get_xticklabels(), stats_df["is_trained"]):
        if bool(is_trained):
            tick.set_color("tab:blue")
            tick.set_weight("bold")
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("PER %")
    ax.set_xlabel("Presented Odor")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.margins(x=0.02)

    for bar, (_, row) in zip(bars, stats_df.iterrows()):
        rate = float(row["rate"])
        trials = int(row["num_trials"])
        text_y = min(rate + 0.05, 1.02)
        annotation = f"{rate:.0%}\n(n={trials})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _style_trained_xticks(ax, labels: Sequence[str], trained_display: str, fontsize: int) -> None:
    ax.set_xticks(np.arange(len(labels)))
    styled = []
    for label in labels:
        if label.strip().lower() == trained_display.lower():
            styled.append(label.upper())
        else:
            styled.append(label)
    ax.set_xticklabels(styled, rotation=35, ha="right", va="top", fontsize=fontsize)
    for tick, label in zip(ax.get_xticklabels(), styled):
        if label.upper() == trained_display.upper():
            tick.set_color("tab:blue")
    ax.tick_params(axis="x", pad=2)


def _trial_order_for(dataset_trials: Sequence[str], order: str) -> list[str]:
    if order == "observed":
        return sorted(dataset_trials, key=_trial_num)
    if order == "trained-first":
        mapping = {trial: _trial_num(trial) for trial in dataset_trials}
        ordered: list[str] = []
        for number in TRAINED_FIRST_ORDER:
            for trial, tnum in mapping.items():
                if tnum == number and trial not in ordered:
                    ordered.append(trial)
        extras = [trial for trial in dataset_trials if trial not in ordered]
        ordered.extend(sorted(extras, key=_trial_num))
        return ordered
    raise ValueError(f"Unsupported trial order: {order}")


def _is_testing_11_label(label: object) -> bool:
    """Return True when the label looks like testing trial 11."""
    return "testing" in str(label).lower() and _trial_num(label) == 11


def _is_testing_9_label(label: object) -> bool:
    """Return True when the label looks like testing trial 9 (v2 light-only)."""
    return "testing" in str(label).lower() and _trial_num(label) == 9


def _is_light_only_label(label: object) -> bool:
    """Return True for light-only trials: testing_11 (legacy) or testing_9 (v2)."""
    if _ACTIVE_PROTOCOL == "v2":
        return _is_testing_9_label(label)
    return _is_testing_11_label(label)


def _drop_testing_11(trials: Sequence[str]) -> list[str]:
    """Remove testing trial 11 (legacy) or testing trial 9 (v2) from reaction prediction plots."""
    cleaned = [trial for trial in trials if not _is_light_only_label(trial)]
    return cleaned


def _order_suffix(order: str) -> str:
    return "unordered" if order == "trained-first" else order.replace("_", "-")


def _matrix_title(dataset_canon: str) -> str:
    """Return plot title text based on dataset origin (Training vs Control)."""

    dataset_key = _odor_dataset_key(dataset_canon)
    base = DISPLAY_LABEL.get(dataset_key, dataset_key)
    is_conditioning = "Training" in dataset_key or dataset_key in TESTING_DATASET_ALIAS
    suffix = "Conditioning Results" if is_conditioning else "Control Results"
    return f"{base} {suffix}"


def generate_reaction_matrices(cfg: MatrixPlotConfig) -> None:
    df, env_cols = _load_matrix(cfg.matrix_npy, cfg.codes_json)

    # Robust trial type filtering with diagnostics
    df_filtered, success, reason = filter_and_validate_trial_type(
        df,
        target_type="testing",
        fallback_to_all=False  # Strict: don't mix testing and training
    )

    if not success or df_filtered.empty:
        available_types = df.get("trial_type", pd.Series()).unique() if "trial_type" in df.columns else "unknown"
        logger.error(
            f"Cannot build reaction matrices: {reason}. "
            f"Ensure matrix contains testing trials. "
            f"Available trial types: {available_types}"
        )
        raise RuntimeError(
            f"No testing trials found in matrix; cannot build reaction matrices. "
            f"Reason: {reason}. "
            f"If this is a training-only dataset, "
            f"use generate_envelope_plots() instead."
        )

    df = df_filtered

    df["fps"] = df["fps"].replace([np.inf, -np.inf], np.nan).fillna(cfg.fps_default)
    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df = _normalise_fly_columns(df)
    df["_non_reactive"] = compute_non_reactive_flags(df)

    flagged_mask = df["_non_reactive"].astype(bool)
    if flagged_mask.any():
        flagged = df.loc[flagged_mask, ["dataset_canon", "fly", "fly_number"]].drop_duplicates()
        summaries = ", ".join(
            f"{row.dataset_canon}::{row.fly}::{row.fly_number}"
            for row in flagged.itertuples(index=False)
        )
        print(f"[analysis] reaction_matrices: excluding non-reactive flies: {summaries}")
        df = df.loc[~flagged_mask].copy()
        if df.empty:
            raise RuntimeError("All flies were flagged non-reactive; nothing to plot.")

    env_data = df[env_cols].to_numpy(np.float32, copy=False)
    dataset_vals = df["dataset_canon"].to_numpy(str)
    fly_vals = df["fly"].to_numpy(str)
    fly_number_vals = df["fly_number"].to_numpy(str)
    trial_vals = df["trial_label"].to_numpy(str)
    fps_vals = df["fps"].to_numpy(float)
    non_reactive_vals = df["_non_reactive"].to_numpy(bool)
    if "trace_len" in df.columns:
        trace_len_vals = pd.to_numeric(df["trace_len"], errors="coerce").to_numpy(float)
    else:
        trace_len_vals = np.full(len(df), np.nan, dtype=float)

    scores = []
    for (
        env_row,
        dataset_val,
        fly_val,
        fly_number_val,
        trial_val,
        fps_val,
        non_reactive_val,
        trace_len_val,
    ) in zip(
        env_data,
        dataset_vals,
        fly_vals,
        fly_number_vals,
        trial_vals,
        fps_vals,
        non_reactive_vals,
        trace_len_vals,
        strict=False,
    ):
        env = _extract_env_row(env_row, trace_len=trace_len_val)
        during_hit, after_hit = _score_trial(env, float(fps_val), cfg)
        scores.append(
            {
                "dataset": dataset_val,
                "fly": fly_val,
                "fly_number": fly_number_val,
                "trial": trial_val,
                "trial_num": _trial_num(trial_val),
                "during_hit": during_hit,
                "after_hit": after_hit,
                "_non_reactive": bool(non_reactive_val),
            }
        )

    scores_df = pd.DataFrame(scores)
    if scores_df.empty:
        raise RuntimeError("Scoring yielded no results; verify the matrix inputs.")

    cmap = ListedColormap(["0.7", "1.0", "0.0"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    present = scores_df["dataset"].unique().tolist()
    ordered_present = [odor for odor in ODOR_ORDER if odor in present]
    extras = sorted(odor for odor in present if odor not in ODOR_ORDER)

    # Collect reaction rate statistics for CSV export
    all_rate_stats = []

    for order in cfg.trial_orders:
        order_suffix = _order_suffix(order)
        for odor in ordered_present + extras:
            subset = scores_df[scores_df["dataset"] == odor]
            if subset.empty:
                continue

            subset = subset.copy()
            subset = _normalise_fly_columns(subset)
            drop_mask = subset["trial"].apply(_is_testing_11_label)
            if drop_mask.any():
                subset = subset.loc[~drop_mask].copy()
                if subset.empty:
                    print(
                        "[INFO] reaction_matrices: skipping",
                        odor,
                        "because only testing_11 trials remained after filtering.",
                    )
                    continue
            flagged_mask = non_reactive_mask(subset)
            flagged_pairs = {
                (row.fly, row.fly_number)
                for row in subset[flagged_mask][["fly", "fly_number"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
            if flagged_pairs:
                fly_pair_series = subset[["fly", "fly_number"]].apply(tuple, axis=1)
                keep_mask = ~fly_pair_series.isin(flagged_pairs)
                subset = subset.loc[keep_mask]
                if subset.empty:
                    print(
                        "[INFO] reaction_matrices: skipping", odor, "because all flies were non-reactive."
                    )
                    continue
            fly_pairs = [
                (row.fly, row.fly_number)
                for row in subset[["fly", "fly_number"]].drop_duplicates().itertuples(index=False)
            ]
            fly_pairs.sort(key=lambda pair: _fly_sort_key(*pair))
            trial_list = _trial_order_for(list(subset["trial"].unique()), order)
            trial_list = _drop_testing_11(trial_list)
            pretty_labels = [_display_odor(odor, trial) for trial in trial_list]

            during_matrix = np.full((len(fly_pairs), len(trial_list)), -1, dtype=int)

            fly_map = {pair: idx for idx, pair in enumerate(fly_pairs)}
            trial_map = {trial: idx for idx, trial in enumerate(trial_list)}
            for _, row in subset.iterrows():
                key = (row["fly"], row["fly_number"])
                i = fly_map[key]
                j = trial_map[row["trial"]]
                during_matrix[i, j] = int(row["during_hit"])

            odor_label = DISPLAY_LABEL.get(odor, odor)
            trained_display = DISPLAY_LABEL.get(odor, odor)
            n_flies = len(fly_pairs)
            n_trials = len(trial_list)

            base_w = max(10.0, 0.70 * n_trials + 6.0)
            base_h = max(5.0, n_flies * 0.26 + 3.8)
            fig_w = base_w
            gap_scale = 0.6
            fig_h = base_h + cfg.row_gap * cfg.height_per_gap_in * gap_scale + cfg.bottom_shift_in

            xtick_fs = 9 if n_trials <= 10 else (8 if n_trials <= 16 else 7)

            plt.rcParams.update(
                {
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "font.family": "Arial",
                    "font.sans-serif": ["Arial"],
                }
            )
            fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
            gs = gridspec.GridSpec(
                2,
                1,
                height_ratios=[3.0, 1.25],
                hspace=cfg.row_gap * gap_scale,
            )

            ax_during = fig.add_subplot(gs[0, 0])
            ax_dc = fig.add_subplot(gs[1, 0])

            ax_during.imshow(during_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            ax_during.set_title(
                f"{_matrix_title(odor)}\n(DURING shifted by +{cfg.latency_sec:.2f} s)",
                fontsize=14,
                weight="bold",
                linespacing=1.1,
            )
            _style_trained_xticks(ax_during, pretty_labels, trained_display, xtick_fs)
            ax_during.set_yticks([])
            ax_during.set_ylabel(f"{n_flies} Flies", fontsize=11)
            for idx, pair in enumerate(fly_pairs):
                if pair in flagged_pairs:
                    ax_during.text(
                        -0.35,
                        idx,
                        "*",
                        ha="right",
                        va="center",
                        color="red",
                        fontsize=12,
                        fontweight="bold",
                        clip_on=False,
                    )

            rate_context = f"{odor_label} ({order_suffix})"
            try:
                rate_stats = reaction_rate_stats_from_rows(
                    subset,
                    odor,
                    include_hexanol=cfg.include_hexanol,
                    context=rate_context,
                    trial_col="trial",
                    reaction_col="during_hit",
                    separate_presentations=True,
                )
            except RuntimeError:
                ax_dc.text(
                    0.5,
                    0.5,
                    "No odors available for rate summary",
                    ha="center",
                    va="center",
                    fontsize=11,
                    transform=ax_dc.transAxes,
                )
                ax_dc.set_axis_off()
            else:
                plot_reaction_rate_bars(
                    ax_dc,
                    rate_stats,
                    title="Reaction Rates by Odor",
                )
                # Collect rate stats for CSV export
                for _, row in rate_stats.iterrows():
                    all_rate_stats.append({
                        "dataset": odor,
                        "trial_order": order,
                        "trial_num": int(row["trial_num"]) if "trial_num" in row else np.nan,
                        "odor_sent": str(row["odor"]),
                        "reaction_rate": float(row["rate"]),
                        "num_reactions": int(row["num_reactions"]),
                        "num_trials": int(row["num_trials"]),
                    })

            shift_frac = cfg.bottom_shift_in / fig_h if fig_h else 0.0
            for axis in (ax_dc,):
                pos = axis.get_position()
                new_y0 = max(0.05, pos.y0 - shift_frac)
                axis.set_position([pos.x0, new_y0, pos.width, pos.height])

            odor_dir = resolve_dataset_output_dir(cfg.out_dir, odor)

            png_name = (
                f"reaction_matrix_{odor.replace(' ', '_')}_{int(cfg.after_window_sec)}"
                f"_latency_{cfg.latency_sec:.3f}s"
            )
            if order_suffix != "observed":
                png_name += f"_{order_suffix}"
            png_path = odor_dir / f"{png_name}.png"
            if should_write(png_path, cfg.overwrite):
                fig.savefig(png_path, dpi=300, bbox_inches="tight")

            row_key_name = f"row_key_{odor.replace(' ', '_')}_{int(cfg.after_window_sec)}"
            if order_suffix != "observed":
                row_key_name += f"_{order_suffix}"
            row_key_path = odor_dir / f"{row_key_name}.txt"
            if should_write(row_key_path, cfg.overwrite):
                with row_key_path.open("w", encoding="utf-8") as fh:
                    for idx, (fly, fly_number) in enumerate(fly_pairs):
                        label = _fly_row_label(fly, fly_number)
                        if (fly, fly_number) in flagged_pairs:
                            label = f"* {label}"
                        fh.write(f"Row {idx}: {label}\n")

            if order == "trained-first":
                export = subset.copy()
                export["odor_sent"] = export["trial"].apply(lambda t: _display_odor(odor, t))
                order_map = {trial: idx for idx, trial in enumerate(trial_list)}
                export["trial_ord"] = export["trial"].map(order_map).fillna(10**9).astype(int)
                export = export.sort_values([
                    "fly",
                    "fly_number",
                    "trial_ord",
                    "trial_num",
                    "trial",
                ])
                export_cols = [
                    "dataset",
                    "fly",
                    "fly_number",
                    "trial_num",
                    "odor_sent",
                    "during_hit",
                    "after_hit",
                ]
                export_path = odor_dir / f"binary_reactions_{odor.replace(' ', '_')}_{order_suffix}.csv"
                if should_write(export_path, cfg.overwrite):
                    export.to_csv(export_path, columns=export_cols, index=False)

            plt.close(fig)

    # Export aggregated reaction rate statistics to CSV
    if all_rate_stats:
        stats_df = pd.DataFrame(all_rate_stats)

        # For each trial order, create a separate summary CSV
        for order in cfg.trial_orders:
            order_stats = stats_df[stats_df["trial_order"] == order].copy()
            if order_stats.empty:
                continue

            if "trial_num" in order_stats.columns:
                order_stats["presentation"] = order_stats.apply(
                    lambda row: (
                        f"T{int(row['trial_num'])}_{row['odor_sent']}"
                        if pd.notna(row["trial_num"])
                        else str(row["odor_sent"])
                    ),
                    axis=1,
                )
                presentation_order = (
                    order_stats[["presentation", "trial_num", "odor_sent"]]
                    .drop_duplicates()
                    .sort_values(["trial_num", "odor_sent"], kind="mergesort")["presentation"]
                    .tolist()
                )
                pivot_columns = "presentation"
            else:
                presentation_order = (
                    order_stats["odor_sent"].drop_duplicates().astype(str).tolist()
                )
                pivot_columns = "odor_sent"

            # Create pivot table: rows = datasets, columns = presentation, values = reaction_rate
            pivot = order_stats.pivot_table(
                index="dataset",
                columns=pivot_columns,
                values="reaction_rate",
                aggfunc="first"  # Should only be one value per dataset-odor pair
            )
            if presentation_order:
                pivot = pivot.reindex(columns=presentation_order)

            # Sort datasets by ODOR_ORDER
            ordered_datasets = [d for d in ODOR_ORDER if d in pivot.index]
            extra_datasets = sorted(d for d in pivot.index if d not in ODOR_ORDER)
            all_datasets = ordered_datasets + extra_datasets
            pivot = pivot.loc[all_datasets]

            # Reset index to make dataset a column
            pivot = pivot.reset_index()

            # Save to CSV
            order_suffix = _order_suffix(order)
            csv_filename = f"reaction_rates_summary_{order_suffix}.csv"
            csv_path = cfg.out_dir / csv_filename

            if should_write(csv_path, cfg.overwrite):
                pivot.to_csv(csv_path, index=False, float_format="%.4f")
                logger.info("Exported reaction rate summary to %s", csv_path)


# ---------------------------------------------------------------------------
# Envelope traces per fly


@dataclass
class EnvelopePlotConfig:
    matrix_npy: Path
    codes_json: Path
    out_dir: Path
    latency_sec: float
    fps_default: float = 40.0
    odor_on_s: float = 30.0
    odor_off_s: float = 60.0
    odor_latency_s: float = 0.0
    after_show_sec: float = 30.0
    threshold_std_mult: float = 3.0
    trial_type: str = "testing"
    light_annotation_mode: str = "none"
    max_flies: int | None = None
    overwrite: bool = False
    fly_filter: str | None = None
    fly_number_filter: str | None = None
    style_scale: float = 1.0
    trace_linewidth_scale: float = 1.0
    panel_title_scale: float = 1.0
    figure_title_scale: float = 1.0
    figure_subtitle_scale: float = 1.0
    legend_scale: float = 1.0
    plot_size_scale: float = 1.0
    single_ylabel_trial_num: int | None = None
    fixed_y_max: float = 100.0
    y_label_override: str | None = None
    show_legend: bool = True
    show_figure_title: bool = True
    show_figure_subtitle: bool = True
    legend_anchor_x: float = 0.98
    legend_anchor_y: float = 0.97
    figure_title_x: float = 0.5
    figure_title_y: float = 0.995
    figure_title_ha: str = "center"
    figure_subtitle_x: float = 0.5
    figure_subtitle_y: float | None = None
    figure_subtitle_ha: str = "center"
    panel_title_x: float = 0.0
    panel_title_y: float = 1.02
    panel_title_va: str = "bottom"
    panel_title_use_data_y: bool = False
    odor_on_label_trial_num: int | None = None
    odor_on_label_text: str | None = None
    odor_on_label_scale: float = 1.0
    odor_on_label_y: float | None = None
    tight_h_pad: float | None = None


def _envelope_ylabel(cfg: EnvelopePlotConfig) -> str:
    if cfg.y_label_override:
        return str(cfg.y_label_override)
    matrix_label = str(cfg.matrix_npy).lower()
    if "combined_base" in matrix_label or "angle_distance" in matrix_label:
        return "Max Distance x Angle %"
    return "RMS (a.u.)"


def _bounded_scale(value: float | int | None, *, default: float = 1.0) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(scale) or scale <= 0.0:
        return default
    return scale


def _trial_selection_rank(trial_label: str) -> tuple[int, int]:
    """Prefer canonical distance-labelled trial rows when duplicate aliases exist."""

    label = str(trial_label).strip().lower()
    prefers_distances = 1 if "_distances_" in label else 0
    # Shorter labels are typically the canonical export when preference ties.
    return (prefers_distances, -len(label))


def _select_trial_rows(fly_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate raw aliases so each fly renders one subplot per trial number."""

    chosen: dict[int, tuple[tuple[int, int], int]] = {}
    passthrough: list[int] = []

    for idx, row in fly_df.iterrows():
        trial_label = str(row.get("trial_label", ""))
        trial_num = _trial_num(trial_label)
        if trial_num < 0:
            passthrough.append(idx)
            continue
        rank = _trial_selection_rank(trial_label)
        current = chosen.get(trial_num)
        if current is None or rank > current[0]:
            chosen[trial_num] = (rank, idx)

    selected = [idx for _, idx in sorted(chosen.values(), key=lambda item: _trial_num(str(fly_df.loc[item[1], "trial_label"])))]
    selected.extend(passthrough)
    return fly_df.loc[selected].copy()


def generate_envelope_plots(cfg: EnvelopePlotConfig) -> None:
    df, env_cols = _load_matrix(cfg.matrix_npy, cfg.codes_json)
    trial_type = cfg.trial_type.strip().lower()
    if trial_type not in {"testing", "training"}:
        raise ValueError(f"Unsupported trial type: {cfg.trial_type!r}")
    light_annotation_mode = str(cfg.light_annotation_mode).strip().lower()
    if light_annotation_mode not in {"none", "line", "paired-span"}:
        raise ValueError(
            f"Unsupported light_annotation_mode: {cfg.light_annotation_mode!r}; "
            "expected one of {'none', 'line', 'paired-span'}."
        )
    max_flies = cfg.max_flies if cfg.max_flies is None else max(int(cfg.max_flies), 1)

    # Robust trial type filtering with diagnostics
    df_filtered, success, reason = filter_and_validate_trial_type(
        df,
        target_type=trial_type,
        fallback_to_all=False
    )

    if not success or df_filtered.empty:
        available_types = df.get("trial_type", pd.Series()).unique() if "trial_type" in df.columns else "unknown"
        logger.error(
            f"Cannot build {trial_type} envelope plots: {reason}. "
            f"Available trial types: {available_types}"
        )
        raise RuntimeError(
            f"No {trial_type} trials found in matrix; cannot build envelope plots. "
            f"Reason: {reason}. "
            f"Available trial types: {available_types}"
        )

    df = df_filtered

    df = _normalise_fly_columns(df)

    df["fps"] = df["fps"].replace([np.inf, -np.inf], np.nan).fillna(cfg.fps_default)
    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["_non_reactive"] = compute_non_reactive_flags(df)

    fly_filter = _normalise_fly_label(cfg.fly_filter) if cfg.fly_filter else None
    fly_number_filter = _normalise_fly_number(cfg.fly_number_filter) if cfg.fly_number_filter else None
    if fly_filter is not None:
        df = df.loc[df["fly"] == fly_filter].copy()
    if fly_number_filter is not None:
        df = df.loc[df["fly_number"] == fly_number_filter].copy()
    if df.empty:
        raise RuntimeError(
            "No envelope rows matched the requested fly filters. "
            f"fly={cfg.fly_filter!r}, fly_number={cfg.fly_number_filter!r}"
        )

    ui_scale = _bounded_scale(cfg.style_scale)
    trace_scale = _bounded_scale(cfg.trace_linewidth_scale)
    line_scale = max(ui_scale, trace_scale)
    axes_linewidth = 0.8 * ui_scale
    tick_width = 0.8 * ui_scale
    tick_length = 3.5 * ui_scale
    base_font = 10 * ui_scale
    ylabel_font = 10 * ui_scale
    xlabel_font = 11 * ui_scale
    panel_title_font = 11 * ui_scale * _bounded_scale(cfg.panel_title_scale)
    legend_font = 9 * ui_scale * _bounded_scale(cfg.legend_scale)
    figure_title_font = 14 * ui_scale * _bounded_scale(cfg.figure_title_scale)
    figure_subtitle_font = 12 * ui_scale * _bounded_scale(cfg.figure_subtitle_scale)
    shared_ylabel_font = 10 * ui_scale
    odor_on_label_font = 12 * ui_scale * _bounded_scale(cfg.odor_on_label_scale)
    trace_lw = 1.2 * trace_scale
    odor_marker_lw = 1.0 * line_scale
    light_marker_lw = 1.3 * line_scale
    threshold_lw = 1.0 * line_scale
    plot_size_scale = _bounded_scale(cfg.plot_size_scale)
    single_ylabel_trial_num = (
        int(cfg.single_ylabel_trial_num)
        if cfg.single_ylabel_trial_num is not None
        else None
    )
    fixed_y_max = max(float(cfg.fixed_y_max), 1.0)

    env_data = df[env_cols].to_numpy(np.float32, copy=False)
    fps_values = df["fps"].to_numpy(float)
    dataset_values = df["dataset_canon"].to_numpy(str)
    trial_values = df["trial_label"].to_numpy(str)

    env_lookup = {idx: env_row for idx, env_row in zip(df.index, env_data, strict=False)}
    fps_lookup = {idx: fps for idx, fps in zip(df.index, fps_values, strict=False)}
    dataset_lookup = {idx: ds for idx, ds in zip(df.index, dataset_values, strict=False)}
    trial_lookup = {idx: tr for idx, tr in zip(df.index, trial_values, strict=False)}
    if "trace_len" in df.columns:
        trace_len_values = pd.to_numeric(df["trace_len"], errors="coerce").to_numpy(float)
        trace_len_lookup = {
            idx: trace_len
            for idx, trace_len in zip(df.index, trace_len_values, strict=False)
        }
    else:
        trace_len_lookup = {}

    odor_latency = max(cfg.odor_latency_s, 0.0)
    odor_on_cmd = cfg.odor_on_s
    odor_off_cmd = cfg.odor_off_s
    odor_off_effective = odor_off_cmd + odor_latency
    linger = max(cfg.latency_sec, 0.0)
    x_max_limit = odor_off_effective + linger + cfg.after_show_sec
    y_label = _envelope_ylabel(cfg)
    matrix_label = str(cfg.matrix_npy).lower()
    share_ylabel = "combined_base" in matrix_label or "angle_distance" in matrix_label

    flies_rendered = 0
    for (fly, fly_number), fly_df in df.groupby(["fly", "fly_number"], sort=False):
        if max_flies is not None and flies_rendered >= max_flies:
            break
        fly_df = _select_trial_rows(fly_df)
        fly_df = fly_df.sort_values("trial_label", key=lambda s: s.map(_trial_num))
        indices = fly_df.index.to_numpy()
        trial_curves: list[
            tuple[str, np.ndarray, np.ndarray, float, bool, float, float, float | None, bool, int]
        ] = []
        y_max = 0.0

        dataset_candidates = [dataset_lookup[idx] for idx in indices if dataset_lookup[idx]]
        folder_dir = resolve_dataset_output_dir(cfg.out_dir, dataset_candidates or ("UNKNOWN",))
        fly_number_label = str(fly_number)
        suffix = "" if fly_number_label.upper() == "UNKNOWN" else f"_fly{fly_number_label}"
        out_path = folder_dir / (
            f"{fly}{suffix}_{trial_type}_envelope_trials_by_odor_"
            f"{int(cfg.after_show_sec)}_shifted.png"
        )
        print(
            "[DEBUG] envelope_plots: generating",
            f"fly={fly}",
            f"fly_number={fly_number_label}",
            f"trials={len(indices)}",
            f"output={out_path}",
        )
        if out_path.exists() and not cfg.overwrite:
            continue

        for idx in indices:
            env = _extract_env_row(
                env_lookup[idx],
                trace_len=trace_len_lookup.get(idx),
            )
            if env.size == 0:
                continue

            fps = float(fps_lookup[idx]) if math.isfinite(fps_lookup[idx]) else cfg.fps_default
            t_full = np.arange(env.size, dtype=float) / max(fps, 1e-9)
            mask = t_full <= x_max_limit + 1e-9
            env = env[mask]
            t_full = t_full[mask]
            if env.size == 0:
                continue

            theta = _compute_theta(env, fps, odor_on_cmd, cfg.threshold_std_mult)
            dataset_canon = dataset_lookup[idx]
            trial_label = trial_lookup[idx]
            odor_name = _display_odor(dataset_canon, trial_label)
            is_trained = _is_trained_odor(dataset_canon, odor_name)
            trial_on_effective, trial_off_effective = _trial_odor_window_seconds(
                trial_label=trial_label,
                trial_type=trial_type,
                odor_on_s=odor_on_cmd,
                odor_off_s=odor_off_cmd,
                odor_latency_s=odor_latency,
            )
            light_start_s = _trial_light_start_seconds(trial_label=trial_label, trial_type=trial_type)
            is_discriminate_odor = _is_discriminate_odor_trial(
                trial_label=trial_label,
                trial_type=trial_type,
            )

            max_local = float(np.nanmax(env)) if np.isfinite(env).any() else 0.0
            if math.isfinite(theta):
                max_local = max(max_local, theta)
            y_max = max(y_max, max_local)

            trial_curves.append(
                (
                    odor_name,
                    t_full,
                    env,
                    theta,
                    is_trained,
                    trial_on_effective,
                    trial_off_effective,
                    light_start_s,
                    is_discriminate_odor,
                    _trial_num(trial_label),
                )
            )

        if not trial_curves:
            continue

        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": axes_linewidth,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "font.family": "Arial",
                "font.sans-serif": ["Arial"],
                "font.size": base_font,
                "xtick.major.width": tick_width,
                "ytick.major.width": tick_width,
                "xtick.minor.width": tick_width,
                "ytick.minor.width": tick_width,
                "xtick.major.size": tick_length,
                "ytick.major.size": tick_length,
                "xtick.minor.size": tick_length * 0.7,
                "ytick.minor.size": tick_length * 0.7,
            }
        )

        n_rows = len(trial_curves)
        fig_h = max(3.0, n_rows * 1.6 * plot_size_scale + 1.5)
        fig_w = 10 * plot_size_scale
        fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), sharex=True)
        if n_rows == 1:
            axes = [axes]
        if share_ylabel:
            fig.text(
                0.02,
                0.5,
                y_label,
                va="center",
                rotation="vertical",
                fontsize=shared_ylabel_font,
            )

        for ax, (
            odor_name,
            t,
            env,
            theta,
            is_trained,
            trial_on_effective,
            trial_off_effective,
            light_start_s,
            is_discriminate_odor,
            trial_num,
        ) in zip(axes, trial_curves):
            ax.plot(t, env, linewidth=trace_lw, color="black")
            ax.axvline(trial_on_effective, linestyle="--", linewidth=odor_marker_lw, color="black")
            ax.axvline(trial_off_effective, linestyle="--", linewidth=odor_marker_lw, color="black")

            transit_on_end = min(trial_on_effective, x_max_limit)
            steady_off_end = min(trial_off_effective, x_max_limit)
            linger_off_end = min(trial_off_effective + linger, x_max_limit)
            odor_bar_color = DISCRIMINATE_ODOR_COLOR if is_discriminate_odor else ODOR_PLUS_LIGHT_COLOR
            odor_bar_alpha = DISCRIMINATE_ODOR_ALPHA if is_discriminate_odor else ODOR_PLUS_LIGHT_ALPHA
            odor_linger_alpha = (
                DISCRIMINATE_ODOR_LINGER_ALPHA
                if is_discriminate_odor
                else ODOR_PLUS_LIGHT_LINGER_ALPHA
            )

            if light_annotation_mode == "paired-span" and light_start_s is not None:
                light_start_eff = min(max(light_start_s, transit_on_end), steady_off_end)
                if light_start_eff > transit_on_end:
                    ax.axvspan(
                        transit_on_end,
                        light_start_eff,
                        alpha=odor_bar_alpha,
                        color=odor_bar_color,
                    )
                if steady_off_end > light_start_eff:
                    ax.axvspan(
                        light_start_eff,
                        steady_off_end,
                        alpha=0.22,
                        color="#f4a261",
                    )
                if linger_off_end > steady_off_end:
                    ax.axvspan(
                        steady_off_end,
                        linger_off_end,
                        alpha=odor_linger_alpha,
                        color=odor_bar_color,
                    )
            elif linger_off_end > transit_on_end:
                ax.axvspan(
                    transit_on_end,
                    linger_off_end,
                    alpha=odor_bar_alpha,
                    color=odor_bar_color,
                )

            if light_annotation_mode == "line" and light_start_s is not None and light_start_s <= x_max_limit:
                ax.axvline(
                    light_start_s,
                    linestyle="-.",
                    linewidth=light_marker_lw,
                    color="tab:green",
                )

            if math.isfinite(theta):
                ax.axhline(theta, linestyle="-", linewidth=threshold_lw, color="tab:red", alpha=0.9)

            ax.set_ylim(0, fixed_y_max)
            ax.set_xlim(0, x_max_limit)
            ax.margins(x=0, y=0.02)
            ax.tick_params(axis="both", which="both", labelsize=base_font, width=tick_width, length=tick_length)
            show_ylabel = (
                not share_ylabel
                and (
                    single_ylabel_trial_num is None
                    or trial_num == single_ylabel_trial_num
                )
            )
            if show_ylabel:
                ax.set_ylabel(y_label, fontsize=ylabel_font)
            elif not share_ylabel:
                ax.set_ylabel("")

            panel_title_transform = (
                transforms.blended_transform_factory(ax.transAxes, ax.transData)
                if cfg.panel_title_use_data_y
                else ax.transAxes
            )
            panel_title_y = float(cfg.panel_title_y)
            panel_title_va = str(cfg.panel_title_va)

            if is_trained:
                ax.text(
                    float(cfg.panel_title_x),
                    panel_title_y,
                    odor_name.upper(),
                    transform=panel_title_transform,
                    ha="left",
                    va=panel_title_va,
                    fontsize=panel_title_font,
                    weight="bold",
                    color="tab:blue",
                    clip_on=False,
                )
            else:
                ax.text(
                    float(cfg.panel_title_x),
                    panel_title_y,
                    odor_name,
                    transform=panel_title_transform,
                    ha="left",
                    va=panel_title_va,
                    fontsize=panel_title_font,
                    weight="bold",
                    color="black",
                    clip_on=False,
                )

            odor_on_label_trial_num = (
                int(cfg.odor_on_label_trial_num)
                if cfg.odor_on_label_trial_num is not None
                else None
            )
            if (
                cfg.odor_on_label_text
                and odor_on_label_trial_num is not None
                and trial_num == odor_on_label_trial_num
                and trial_on_effective < trial_off_effective
            ):
                odor_on_label_y = (
                    float(cfg.odor_on_label_y)
                    if cfg.odor_on_label_y is not None
                    else fixed_y_max * 0.82
                )
                ax.text(
                    (trial_on_effective + trial_off_effective) / 2.0,
                    odor_on_label_y,
                    str(cfg.odor_on_label_text),
                    ha="center",
                    va="center",
                    fontsize=odor_on_label_font,
                    weight="bold",
                    color="black",
                )

        axes[-1].set_xlabel("Time (s)", fontsize=xlabel_font)

        legend_handles = [
            plt.Line2D([0], [0], linestyle="--", linewidth=odor_marker_lw, color="black", label="Odor at fly"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                alpha=ODOR_PLUS_LIGHT_ALPHA,
                color=ODOR_PLUS_LIGHT_COLOR,
                label=ODOR_PLUS_LIGHT_LABEL,
            ),
        ]
        if trial_type == "training":
            legend_handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    alpha=DISCRIMINATE_ODOR_ALPHA,
                    color=DISCRIMINATE_ODOR_COLOR,
                label=DISCRIMINATE_ODOR_LABEL,
                )
            )
        if light_annotation_mode == "line":
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    linestyle="-.",
                    linewidth=light_marker_lw,
                    color="tab:green",
                    label="Light pulsing starts",
                )
            )
        elif light_annotation_mode == "paired-span":
            legend_handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    alpha=0.22,
                    color="#f4a261",
                    label="Light + Odor Paired",
                )
            )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                linestyle="-",
                linewidth=threshold_lw,
                color="tab:red",
                label=r"$\theta = \mathrm{median}_{before} + k\cdot\mathrm{MAD}_{before}$",
            )
        )
        if cfg.show_legend:
            fig.legend(
                handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(float(cfg.legend_anchor_x), float(cfg.legend_anchor_y)),
                frameon=True,
                fontsize=legend_font,
                title=f"Threshold: k = {cfg.threshold_std_mult:g}",
                title_fontsize=legend_font,
            )

        fly_caption = fly
        if fly_number_label.upper() != "UNKNOWN":
            fly_caption = f"{fly} - Fly {fly_number_label}"
        fly_flagged = bool(non_reactive_mask(fly_df).any())
        title_y = float(cfg.figure_title_y)
        subtitle_y = (
            float(cfg.figure_subtitle_y)
            if cfg.figure_subtitle_y is not None
            else title_y - 0.035
        )
        phase_label = "Training" if trial_type == "training" else "Testing"
        if cfg.show_figure_title:
            fig.text(
                float(cfg.figure_title_x),
                title_y,
                f"Proboscis Distance Across {phase_label} Trials",
                ha=str(cfg.figure_title_ha),
                va="center",
                fontsize=figure_title_font,
                weight="bold",
                color="black",
            )
        if cfg.show_figure_subtitle:
            fig.text(
                float(cfg.figure_subtitle_x),
                subtitle_y,
                fly_caption,
                ha=str(cfg.figure_subtitle_ha),
                va="center",
                fontsize=figure_subtitle_font,
                weight="bold",
                color="tab:red" if fly_flagged else "black",
            )
        left_pad = 0.04 if share_ylabel else 0.0
        has_top_decoration = cfg.show_legend or cfg.show_figure_title or cfg.show_figure_subtitle
        layout_top = (
            max(0.84, 0.93 - 0.03 * max(ui_scale - 1.0, 0.0))
            if has_top_decoration
            else 0.985
        )
        tight_layout_kwargs = {"rect": [left_pad, 0, 1, layout_top]}
        if cfg.tight_h_pad is not None:
            tight_layout_kwargs["h_pad"] = float(cfg.tight_h_pad)
        fig.tight_layout(**tight_layout_kwargs)

        if should_write(out_path, cfg.overwrite):
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            flies_rendered += 1

        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI plumbing


def _parse_matrices_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--matrix-npy", type=Path, required=True, help="Float16 matrix produced by the convert step.")
    subparser.add_argument("--codes-json", type=Path, required=True, help="JSON metadata file from the convert step.")
    subparser.add_argument("--out-dir", type=Path, required=True, help="Directory for exported figures.")
    subparser.add_argument("--latency-sec", type=float, default=0.0, help="Mean odor transit latency in seconds.")
    subparser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when decoding rows.")
    subparser.add_argument("--before-sec", type=float, default=30.0, help="Duration of the baseline window (seconds).")
    subparser.add_argument("--during-sec", type=float, default=30.0, help="Duration of the DURING window (seconds).")
    subparser.add_argument("--after-window-sec", type=float, default=30.0, help="Duration of the AFTER window (seconds).")
    subparser.add_argument(
        "--threshold-std-mult",
        type=float,
        default=3.0,
        help="Threshold multiplier applied to baseline MAD (scaled to sigma).",
    )
    subparser.add_argument("--min-samples-over", type=int, default=20, help="Minimum samples over threshold to count a hit.")
    subparser.add_argument("--row-gap", type=float, default=0.6, help="Vertical gap between matrix and bar charts.")
    subparser.add_argument("--height-per-gap-in", type=float, default=3.0, help="Figure height added per 1.0 of row gap (inches).")
    subparser.add_argument("--bottom-shift-in", type=float, default=0.5, help="Downward shift applied to bar charts (inches).")
    subparser.add_argument(
        "--trial-order",
        action="append",
        choices=("observed", "trained-first"),
        help="Trial ordering strategy. Repeat to request multiple variants.",
    )
    subparser.add_argument(
        "--exclude-hexanol",
        action="store_true",
        help="Exclude Hexanol from 'other' reaction counts.",
    )
    subparser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")


def _parse_envelopes_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--matrix-npy", type=Path, required=True, help="Float16 matrix produced by the convert step.")
    subparser.add_argument("--codes-json", type=Path, required=True, help="JSON metadata file from the convert step.")
    subparser.add_argument("--out-dir", type=Path, required=True, help="Directory for per-fly envelope plots.")
    subparser.add_argument("--latency-sec", type=float, default=0.0, help="Mean odor transit latency in seconds.")
    subparser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when decoding rows.")
    subparser.add_argument("--odor-on-s", type=float, default=30.0, help="Commanded odor ON timestamp (seconds).")
    subparser.add_argument("--odor-off-s", type=float, default=60.0, help="Commanded odor OFF timestamp (seconds).")
    subparser.add_argument(
        "--odor-latency-s",
        type=float,
        default=0.0,
        help="Transit delay between valve command and odor at the fly (seconds).",
    )
    subparser.add_argument("--after-show-sec", type=float, default=30.0, help="Duration to display after odor off (seconds).")
    subparser.add_argument(
        "--threshold-std-mult",
        type=float,
        default=3.0,
        help="Threshold multiplier applied to baseline MAD (scaled to sigma).",
    )
    subparser.add_argument(
        "--trial-type",
        choices=("testing", "training"),
        default="testing",
        help="Trial type to visualise.",
    )
    subparser.add_argument(
        "--light-annotation-mode",
        choices=("none", "line", "paired-span"),
        default="none",
        help=(
            "Training-only light annotation style: none, a vertical line at light onset, "
            "or a paired-color span from light onset to odor OFF."
        ),
    )
    subparser.add_argument(
        "--max-flies",
        type=int,
        default=None,
        help="Optional cap for number of fly figures to render (useful for quick samples).",
    )
    subparser.add_argument("--fly", type=str, default=None, help="Optional exact fly label to render.")
    subparser.add_argument("--fly-number", type=str, default=None, help="Optional exact fly number to render.")
    subparser.add_argument(
        "--style-scale",
        type=float,
        default=1.0,
        help="Multiplier for fonts, ticks, and axis line widths.",
    )
    subparser.add_argument(
        "--trace-linewidth-scale",
        type=float,
        default=1.0,
        help="Multiplier for envelope and annotation line widths.",
    )
    subparser.add_argument(
        "--panel-title-scale",
        type=float,
        default=1.0,
        help="Multiplier for per-trial odor label font size.",
    )
    subparser.add_argument(
        "--figure-title-scale",
        type=float,
        default=1.0,
        help="Multiplier for the main figure title font size.",
    )
    subparser.add_argument(
        "--figure-subtitle-scale",
        type=float,
        default=1.0,
        help="Multiplier for the fly subtitle font size.",
    )
    subparser.add_argument(
        "--legend-scale",
        type=float,
        default=1.0,
        help="Multiplier for legend font sizes.",
    )
    subparser.add_argument(
        "--plot-size-scale",
        type=float,
        default=1.0,
        help="Multiplier for the rendered figure width and per-row height.",
    )
    subparser.add_argument(
        "--single-ylabel-trial-num",
        type=int,
        default=None,
        help="If set, only show the y-axis label on the subplot for this trial number.",
    )
    subparser.add_argument(
        "--fixed-y-max",
        type=float,
        default=100.0,
        help="Upper y-axis limit shared across all subplots.",
    )
    subparser.add_argument(
        "--y-label-override",
        type=str,
        default=None,
        help="Optional replacement text for the y-axis label.",
    )
    subparser.add_argument(
        "--legend-anchor-x",
        type=float,
        default=0.98,
        help="Legend bbox anchor x position in figure coordinates.",
    )
    subparser.add_argument(
        "--legend-anchor-y",
        type=float,
        default=0.97,
        help="Legend bbox anchor y position in figure coordinates.",
    )
    subparser.add_argument(
        "--figure-title-x",
        type=float,
        default=0.5,
        help="Main title x position in figure coordinates.",
    )
    subparser.add_argument(
        "--figure-title-y",
        type=float,
        default=0.995,
        help="Main title y position in figure coordinates.",
    )
    subparser.add_argument(
        "--figure-title-ha",
        choices=("left", "center", "right"),
        default="center",
        help="Horizontal alignment for the main title.",
    )
    subparser.add_argument(
        "--figure-subtitle-x",
        type=float,
        default=0.5,
        help="Subtitle x position in figure coordinates.",
    )
    subparser.add_argument(
        "--figure-subtitle-y",
        type=float,
        default=None,
        help="Optional subtitle y position in figure coordinates.",
    )
    subparser.add_argument(
        "--figure-subtitle-ha",
        choices=("left", "center", "right"),
        default="center",
        help="Horizontal alignment for the subtitle.",
    )
    subparser.add_argument(
        "--panel-title-x",
        type=float,
        default=0.0,
        help="Per-panel odor label x position in axes coordinates.",
    )
    subparser.add_argument(
        "--panel-title-y",
        type=float,
        default=1.02,
        help="Per-panel odor label y position in axes coordinates unless --panel-title-use-data-y is set.",
    )
    subparser.add_argument(
        "--panel-title-va",
        choices=("top", "center", "bottom", "baseline", "center_baseline"),
        default="bottom",
        help="Vertical alignment for the per-panel odor label.",
    )
    subparser.add_argument(
        "--panel-title-use-data-y",
        action="store_true",
        help="Interpret --panel-title-y in data coordinates while keeping x in axes coordinates.",
    )
    subparser.add_argument(
        "--tight-h-pad",
        type=float,
        default=None,
        help="Optional vertical padding passed to tight_layout between stacked subplots.",
    )
    subparser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrices_parser = subparsers.add_parser("matrices", help="Generate reaction matrix figures.")
    _parse_matrices_args(matrices_parser)

    envelopes_parser = subparsers.add_parser("envelopes", help="Generate per-fly envelope plots.")
    _parse_envelopes_args(envelopes_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "matrices":
        trial_orders: Sequence[str] = args.trial_order or ("observed", "trained-first")
        cfg = MatrixPlotConfig(
            matrix_npy=args.matrix_npy,
            codes_json=args.codes_json,
            out_dir=args.out_dir,
            latency_sec=args.latency_sec,
            fps_default=args.fps_default,
            before_sec=args.before_sec,
            during_sec=args.during_sec,
            after_window_sec=args.after_window_sec,
            threshold_std_mult=args.threshold_std_mult,
            min_samples_over=args.min_samples_over,
            row_gap=args.row_gap,
            height_per_gap_in=args.height_per_gap_in,
            bottom_shift_in=args.bottom_shift_in,
            trial_orders=trial_orders,
            include_hexanol=not args.exclude_hexanol,
            overwrite=args.overwrite,
        )
        generate_reaction_matrices(cfg)
        return

    if args.command == "envelopes":
        cfg = EnvelopePlotConfig(
            matrix_npy=args.matrix_npy,
            codes_json=args.codes_json,
            out_dir=args.out_dir,
            latency_sec=args.latency_sec,
            fps_default=args.fps_default,
            odor_on_s=args.odor_on_s,
            odor_off_s=args.odor_off_s,
            odor_latency_s=args.odor_latency_s,
            after_show_sec=args.after_show_sec,
            threshold_std_mult=args.threshold_std_mult,
            trial_type=args.trial_type,
            light_annotation_mode=args.light_annotation_mode,
            max_flies=args.max_flies,
            overwrite=args.overwrite,
            fly_filter=args.fly,
            fly_number_filter=args.fly_number,
            style_scale=args.style_scale,
            trace_linewidth_scale=args.trace_linewidth_scale,
            panel_title_scale=args.panel_title_scale,
            figure_title_scale=args.figure_title_scale,
            figure_subtitle_scale=args.figure_subtitle_scale,
            legend_scale=args.legend_scale,
            plot_size_scale=args.plot_size_scale,
            single_ylabel_trial_num=args.single_ylabel_trial_num,
            fixed_y_max=args.fixed_y_max,
            y_label_override=args.y_label_override,
            legend_anchor_x=args.legend_anchor_x,
            legend_anchor_y=args.legend_anchor_y,
            figure_title_x=args.figure_title_x,
            figure_title_y=args.figure_title_y,
            figure_title_ha=args.figure_title_ha,
            figure_subtitle_x=args.figure_subtitle_x,
            figure_subtitle_y=args.figure_subtitle_y,
            figure_subtitle_ha=args.figure_subtitle_ha,
            panel_title_x=args.panel_title_x,
            panel_title_y=args.panel_title_y,
            panel_title_va=args.panel_title_va,
            panel_title_use_data_y=args.panel_title_use_data_y,
            tight_h_pad=args.tight_h_pad,
        )
        generate_envelope_plots(cfg)
        return

    parser.error(f"Unknown command: {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    main()
