"""Pure (pandas/numpy-only) data logic for the config_new blinded scoring GUI.

Kept free of tkinter/cv2/matplotlib so it can be unit-tested headlessly and
imported by the GUI script.
"""

from __future__ import annotations

import re as _re

import numpy as np
import pandas as pd

# Datasets that are light-only (no odor) — excluded from scoring.
LIGHT_ONLY_DATASETS: set[str] = {"LightSweep-Control-24-0.01"}


def compute_theta(
    env: np.ndarray, fps: float, odor_on_s: float = 30.0, std_mult: float = 3.0
) -> float:
    """Response threshold = resting median + k * one-sided (upper) robust spread.

    Only deviations ABOVE the baseline median feed the dispersion, so downward
    dips below resting do not raise the threshold. Mirrors ``_baseline_theta`` in
    ``scripts/analysis/envelope_visuals.py``.
    """
    n_before = int(round(odor_on_s * fps))
    before = env[:n_before]
    before = before[np.isfinite(before)]
    if before.size < 3:
        return float("nan")
    baseline = float(np.nanmedian(before))
    dev = before - baseline
    up = dev[dev > 0.0]
    mad_up = float(np.nanmedian(up)) if up.size else 0.0
    sigma = 1.4826 * mad_up
    return float(baseline + std_mult * sigma)


def filter_trials(df: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame:
    """Testing trials eligible for scoring.

    Keeps ``trial_type == 'testing'`` with a well-formed ``trial_label`` of the
    form ``testing_<N>`` OR ``training_<N>`` — override datasets (e.g. RandomPanel)
    are scored as testing yet keep ``training_<N>`` labels, so both prefixes must
    survive. Drops light-only datasets. With ``dataset`` set, keeps only rows
    whose ``dataset`` column EXACTLY equals it. Training trials are already absent
    from the combined_base CSV; ``testing_11`` is intentionally retained.
    """
    out = df[df["trial_type"].astype(str).str.strip().str.lower() == "testing"].copy()
    out = out[
        out["trial_label"].astype(str).str.match(
            r"(?:testing|training)_\d+", case=False, na=False
        )
    ]
    out = out[~out["dataset"].astype(str).isin(LIGHT_ONLY_DATASETS)]
    if dataset is not None:
        out = out[out["dataset"].astype(str) == dataset]
    return out.copy()


def available_datasets(df: pd.DataFrame) -> list[str]:
    return sorted(df["dataset"].astype(str).unique().tolist())


def apply_exclusions(df: pd.DataFrame, exclude: set[tuple[str, int]]) -> pd.DataFrame:
    if not exclude:
        return df
    mask = pd.Series(
        [
            (str(r["fly"]).strip(), int(r["fly_number"])) in exclude
            for _, r in df.iterrows()
        ],
        index=df.index,
    )
    return df[~mask].copy()


def _core_label(trial_label: str) -> str:
    m = _re.match(r"(testing_\d+)", str(trial_label).strip())
    return m.group(1) if m else str(trial_label).strip()


def summarize_datasets(
    df: pd.DataFrame, scored_keys: set[tuple[str, str, int, str]]
) -> list[dict]:
    rows: list[dict] = []
    for dataset, grp in df.groupby(df["dataset"].astype(str)):
        scored = 0
        for _, r in grp.iterrows():
            key = (
                str(r["dataset"]).strip(),
                str(r["fly"]).strip(),
                int(r["fly_number"]),
                _core_label(r["trial_label"]),
            )
            if key in scored_keys:
                scored += 1
        rows.append(
            {
                "dataset": dataset,
                "trials": int(len(grp)),
                "flies": int(grp[["fly", "fly_number"]].drop_duplicates().shape[0]),
                "scored": scored,
            }
        )
    return sorted(rows, key=lambda d: d["dataset"])
