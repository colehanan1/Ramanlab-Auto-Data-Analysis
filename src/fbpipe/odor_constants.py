"""Single source of truth for odor canonical names, display labels, and dataset aliases.

Every script that needs to resolve odor/dataset names should import from here
instead of maintaining its own copy.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

# ---------------------------------------------------------------------------
# Canonical odor name mapping  (lowercase key → display-ready canonical name)
# Merged from envelope_visuals.py and envelope_combined.py.
# ---------------------------------------------------------------------------

ODOR_CANON: Mapping[str, str] = {
    # ── Base odor names ──
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
    "ethyl butyrate": "EB",
    "10s_odor_benz": "10s_Odor_Benz",
    "hexanol": "Hex-Training",
    "citral": "Cit-Training",
    "linalool": "Lin-Training",
    # ── Training variants ──
    "benz-training": "Benz-Training",
    "benz training": "Benz-Training",
    "benz-training-24": "Benz-Training-24",
    "benz training 24": "Benz-Training-24",
    "benz-training-24-2": "Benz-Training-24-2",
    "benz-training-24-02": "Benz-Training-24-02",
    "eb-training": "EB-Training",
    "eb-training(no-operant)": "EB-Training(No-Operant)",
    "eb-training-no-operant": "EB-Training(No-Operant)",
    "hex-training": "Hex-Training",
    "hex-training-24": "Hex-Training-24",
    "hex-training-24-2": "Hex-Training-24-2",
    "hex-training-24-02": "Hex-Training-24-02",
    "hex-training-36": "Hex-Training-36",
    "hex-training-24-002": "Hex-Training-24-002",
    "hex training 24": "Hex-Training-24",
    "hex training 36": "Hex-Training-36",
    "acv-training": "ACV-Training",
    "air-training": "AIR-Training",
    "3oct-training": "3OCT-Training",
    "3oct training": "3OCT-Training",
    "3oct-training-24-2": "3OCT-Training-24-2",
    "3oct training 24 2": "3OCT-Training-24-2",
    "cit-training": "Cit-Training",
    "cit_training": "Cit-Training",
    "cit training": "Cit-Training",
    "lin-training": "Lin-Training",
    "lin_training": "Lin-Training",
    "lin training": "Lin-Training",
    # ── Control variants ──
    "benz-control": "Benz-Control",
    "benz_control": "Benz-Control",
    "benz control": "Benz-Control",
    "benz-control-24-2": "Benz-Control-24-2",
    "benz-control-24-02": "Benz-Control-24-02",
    "eb-control": "EB-Control",
    "eb_control": "EB-Control",
    "eb control": "EB-Control",
    "hex-control": "Hex-Control",
    "hex_control": "Hex-Control",
    "hexanol control": "Hex-Control",
    "hex control": "Hex-Control",
    "hex-control-24": "Hex-Control-24",
    "hex_control_24": "Hex-Control-24",
    "hex control 24": "Hex-Control-24",
    "hex-control-24-2": "Hex-Control-24-2",
    "hex-control-24-02": "Hex-Control-24-02",
    "hex-control-24-002": "Hex-Control-24-002",
    "hex_control_24_002": "Hex-Control-24-002",
    "hex control 24 002": "Hex-Control-24-002",
    "hex-control-24-0002": "Hex-Control-24-0002",
    "hex_control_24_0002": "Hex-Control-24-0002",
    "hex control 24 0002": "Hex-Control-24-0002",
    "hex-control-36": "Hex-Control-36",
    "hex_control_36": "Hex-Control-36",
    "hex control 36": "Hex-Control-36",
    "cit-control": "Cit-Control",
    "cit_control": "Cit-Control",
    "cit control": "Cit-Control",
    "lin-control": "Lin-Control",
    "lin_control": "Lin-Control",
    "lin control": "Lin-Control",
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
    # ── Legacy optogenetics folder names ──
    "optogenetics benzaldehyde": "Benz-Training",
    "optogenetics benzaldehyde 1": "Benz-Training",
    "optogenetics ethyl butyrate": "EB-Training",
    "optogenetics apple cider vinegar": "ACV-Training",
    "optogenetics acv": "ACV-Training",
    "optogenetics hexanol": "Hex-Training",
    "optogenetics hex": "Hex-Training",
    "optogenetics air": "AIR-Training",
    "optogenetics 3-octanol": "3OCT-Training",
    "opto_eb": "EB-Training",
    "opto_eb(6-training)": "EB-Training(No-Operant)",
    "opto_eb_6_training": "EB-Training(No-Operant)",
    "opto_hex": "Hex-Training",
    "opto_acv": "ACV-Training",
    "opto_air": "AIR-Training",
    "opto_3-oct": "3OCT-Training",
    "opto_benz": "Benz-Training",
    "opto_benz_1": "Benz-Training",
    "opto_ACV": "ACV-Training",
    "opto_AIR": "AIR-Training",
    "opto_EB": "EB-Training",
    "opto_EB_6_training": "EB-Training(No-Operant)",
    "opto_EB(6-training)": "EB-Training(No-Operant)",
    # ── Manual / sucrose-trained fly datasets ──
    "manual_3-octonol": "manual_3-octonol",
    "manual_10s_odor_benz": "manual_10s_Odor_Benz",
    "manual_acv": "manual_ACV",
    "manual_benz": "manual_Benz",
    "manual_eb": "manual_EB",
    "manual_ret_eb": "manual_ret_EB",
    "ret_eb": "manual_ret_EB",
}


# ---------------------------------------------------------------------------
# Human-readable display labels for canonical odor names.
# ---------------------------------------------------------------------------

DISPLAY_LABEL: dict[str, str] = {
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
    "Hex-Control-24-0002": "Hexanol",
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
    "Hex-Training-24-002": "Hexanol",
    "AIR-Training": "AIR",
    "3OCT-Training": "3-Octonol",
    "3OCT-Training-24-2": "3-Octonol",
    "Cit-Training": "Citral",
    "Cit-Control": "Citral",
    "Lin-Training": "Linalool",
    "Lin-Control": "Linalool",
    "ACV-Control": "Apple Cider Vinegar",
    "3OCT-Control": "3-Octonol",
    "3OCT-Control-24-2": "3-Octonol",
    # Manual / sucrose-trained fly datasets
    "manual_3-octonol": "3-Octonol",
    "manual_10s_Odor_Benz": "Benzaldehyde",
    "manual_ACV": "Apple Cider Vinegar",
    "manual_Benz": "Benzaldehyde",
    "manual_EB": "Ethyl Butyrate",
    "manual_ret_EB": "Ethyl Butyrate",
}


# ---------------------------------------------------------------------------
# Preferred ordering for figures / axes.
# ---------------------------------------------------------------------------

ODOR_ORDER: list[str] = [
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
    "Hex-Training-24-002",
    "Hex-Control",
    "Hex-Control-24",
    "Hex-Control-24-2",
    "Hex-Control-24-002",
    "Hex-Control-24-0002",
    "Hex-Control-36",
    "AIR-Training",
    "3OCT-Training",
    "3OCT-Training-24-2",
    "Cit-Training",
    "Cit-Control",
    "Lin-Training",
    "Lin-Control",
    "ACV-Control",
    "3OCT-Control",
    "3OCT-Control-24-2",
]


# ---------------------------------------------------------------------------
# Testing dataset alias – maps training/variant datasets to the control
# dataset they should be grouped with for testing analysis.
# ---------------------------------------------------------------------------

TESTING_DATASET_ALIAS: dict[str, str] = {
    "Hex-Training": "Hex-Control",
    "Hex-Training-24": "Hex-Control",
    "Hex-Training-24-2": "Hex-Control",
    "Hex-Training-24-02": "Hex-Control",
    "Hex-Training-36": "Hex-Control",
    "Hex-Training-24-002": "Hex-Control",
    "Hex-Control-24": "Hex-Control",
    "Hex-Control-24-2": "Hex-Control",
    "Hex-Control-24-02": "Hex-Control",
    "Hex-Control-24-002": "Hex-Control",
    "Hex-Control-24-0002": "Hex-Control",
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
    "AIR-Training": "AIR-Training",
    "Cit-Training": "Cit-Control",
    "Lin-Training": "Lin-Control",
    # Manual / sucrose-trained fly datasets (self-mappings)
    "manual_3-octonol": "manual_3-octonol",
    "manual_10s_Odor_Benz": "manual_10s_Odor_Benz",
    "manual_ACV": "manual_ACV",
    "manual_Benz": "manual_Benz",
    "manual_EB": "manual_EB",
    "manual_ret_EB": "manual_ret_EB",
}


# ---------------------------------------------------------------------------
# Dataset alias – maps legacy opto_* and variant folder names to canonical.
# Used by per-folder scripts to normalize directory names.
# ---------------------------------------------------------------------------

DATASET_ALIAS: dict[str, str] = {
    # legacy opto_* / *_control names
    "opto_3-oct": "3OCT-Training",
    "opto_ACV": "ACV-Training",
    "opto_AIR": "AIR-Training",
    "Benz_control": "Benz-Control",
    "benz_control": "Benz-Control",
    "opto_benz": "Benz-Training",
    "opto_benz_1": "Benz-Training",
    "EB_control": "EB-Control",
    "eb_control": "EB-Control",
    "opto_EB": "EB-Training",
    "opto_EB_6_training": "EB-Training(No-Operant)",
    "opto_EB(6-training)": "EB-Training(No-Operant)",
    "hex_control": "Hex-Control",
    "opto_hex": "Hex-Training",
    # lower-case folder-style aliases
    "3oct-training": "3OCT-Training",
    "3oct-training-24-2": "3OCT-Training-24-2",
    "3oct-control-24-2": "3OCT-Control-24-2",
    "acv-training": "ACV-Training",
    "air-training": "AIR-Training",
    "benz-control": "Benz-Control",
    "benz-control-24-2": "Benz-Control-24-2",
    "benz-control-24-02": "Benz-Control-24-02",
    "benz-training": "Benz-Training",
    "benz-training-24": "Benz-Training-24",
    "benz-training-24-2": "Benz-Training-24-2",
    "benz-training-24-02": "Benz-Training-24-02",
    "eb-control": "EB-Control",
    "eb-training": "EB-Training",
    "eb-training(no-operant)": "EB-Training(No-Operant)",
    "hex-control": "Hex-Control",
    "hex-control-24-2": "Hex-Control-24-2",
    "hex-control-24-02": "Hex-Control-24-02",
    "hex-control-24-002": "Hex-Control-24-002",
    "hex-training": "Hex-Training",
    "hex-training-24": "Hex-Training-24",
    "hex-training-24-2": "Hex-Training-24-2",
    "hex-training-24-02": "Hex-Training-24-02",
    "hex-training-24-002": "Hex-Training-24-002",
    "hex-control-24-0002": "Hex-Control-24-0002",
}


# ---------------------------------------------------------------------------
# Pure lookup helpers
# ---------------------------------------------------------------------------

def canon_dataset(value: str) -> str:
    """Return the canonical ODOR identifier for *value*."""
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


def odor_dataset_key(dataset_canon: str) -> str:
    """Extract the core odor key, stripping '-flagged' suffix."""
    dataset_text = str(dataset_canon).strip() if isinstance(dataset_canon, str) else "UNKNOWN"
    if not dataset_text:
        return "UNKNOWN"
    lower = dataset_text.lower()
    if lower.endswith("-flagged"):
        base = dataset_text[: -len("-flagged")].strip()
        return ODOR_CANON.get(base.lower(), base)
    return ODOR_CANON.get(lower, dataset_text)


def resolve_dataset_label(values: Sequence[str] | str) -> str:
    """Return a human-readable label for one or more dataset identifiers."""
    if isinstance(values, str):
        candidates = {canon_dataset(values)} if values else set()
    else:
        candidates = {canon_dataset(val) for val in values if isinstance(val, str) and val}

    candidates = {val for val in candidates if val}
    if not candidates:
        return "UNKNOWN"
    if len(candidates) == 1:
        key = next(iter(candidates))
        return DISPLAY_LABEL.get(key, key)
    pretty = [DISPLAY_LABEL.get(key, key) for key in sorted(candidates)]
    return f"Mixed ({'+'.join(pretty)})"
