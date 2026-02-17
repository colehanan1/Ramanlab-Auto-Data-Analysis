"""NaN-aware array stacking and mean/SEM aggregation utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def nan_pad_stack(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Right-pad 1-D arrays with NaN to equal length and stack into a 2-D array.

    Parameters
    ----------
    arrays : sequence of 1-D ndarrays
        Input traces of possibly unequal length.

    Returns
    -------
    np.ndarray
        Shape ``(len(arrays), max_length)`` with dtype ``float64``.
        Shorter arrays are padded with ``NaN`` on the right.

    Raises
    ------
    ValueError
        If *arrays* is empty.
    """
    if not arrays:
        raise ValueError("nan_pad_stack requires at least one array")

    max_len = max(len(a) for a in arrays)
    out = np.full((len(arrays), max_len), np.nan, dtype=np.float64)
    for i, a in enumerate(arrays):
        arr = np.asarray(a, dtype=np.float64)
        out[i, : len(arr)] = arr
    return out


def nanmean_sem(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute pointwise NaN-aware mean and SEM across axis 0.

    Parameters
    ----------
    stacked : np.ndarray
        2-D array of shape ``(n_subjects, n_timepoints)``.

    Returns
    -------
    mean : np.ndarray
        Shape ``(n_timepoints,)``.
    sem : np.ndarray
        Shape ``(n_timepoints,)``.  Where *n* <= 1 the SEM is set to 0.
    """
    with np.errstate(all="ignore"):
        mean = np.nanmean(stacked, axis=0)
        n = np.sum(np.isfinite(stacked), axis=0)
        std = np.nanstd(stacked, axis=0, ddof=0)
        sem = np.where(n > 1, std / np.sqrt(n), 0.0)
    return mean, sem


def count_finite_contributors(stacked: np.ndarray) -> int:
    """Count subjects (rows) that have at least one finite value.

    This is the *n* reported in the legend of dataset-mean plots.
    """
    return int(np.sum(np.any(np.isfinite(stacked), axis=1)))
