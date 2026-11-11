"""Deterministic inputs for the window-rescore tests.

Shared by ``tests/test_window_rescore_delay.py`` and
``tests/_capture_window_rescore_golden.py``: a small wide table whose rows hit
every edge of the window layout, the pipeline's AUC threshold rule, and a tiny
self-contained ordinal model (optionally with an odor-arrival delay and a
two-guard ensemble) so the guard can be tested without the production model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FPS = 40.0
#: The widest trial (90 s at 40 fps). Shorter trials are NaN-padded to it, as in
#: the pipeline's wide tables.
N_COLS = 3600

#: ``analysis.combined.combined_base.wide`` threshold keys, as config_new.yaml
#: carried them on 2026-11-10. Any values would do: the tests only need the
#: rule the wide tables were built with to be a real, non-default one.
PIPELINE_WIDE_KEYS = {
    "threshold_std_mult": 3.0,
    "threshold_min_delta": 5.0,
    "threshold_anchor_s": 5.0,
    "threshold_noise_block_s": 5.0,
    "threshold_noise_pctl": 25.0,
}

#: (name, on_s, off_s, duration_s, fps, events, global_max, trimmed_global_min)
#: ``events`` are (start_s, stop_s, height) bumps added to a 10 ± 0.5 baseline;
#: height None blanks that span (a tracking gap).
_ROWS = [
    ("standard_odor_response", 30.0, 60.0, 90.0, 40.0, [(33.0, 40.0, 60.0)], 80.0, 9.0),
    ("short_odor_late_response", 30.0, 35.0, 65.0, 40.0, [(36.0, 39.0, 60.0)], 80.0, 9.0),
    # Entirely inside the 1 s transit after valve-on.
    ("transit_spike", 30.0, 35.0, 65.0, 40.0, [(30.2, 30.8, 80.0)], 95.0, 9.0),
    # Between valve-off and valve-off + 1 s: the odor window of a 1 s-delay
    # model, the "after" window of a delay-free one.
    ("valve_off_second", 30.0, 35.0, 65.0, 40.0, [(35.1, 35.9, 60.0)], 80.0, 9.0),
    # No timing at all: the 30 / 60 s protocol at 40 fps over every column.
    ("missing_timing", np.nan, np.nan, np.nan, np.nan, [(45.0, 50.0, 40.0)], 60.0, 9.0),
    # Tracking gaps in the baseline and the odor; a zero trimmed min (ratios NaN).
    ("nan_gaps_zero_tmin", 30.0, 60.0, 90.0, 40.0,
     [(10.0, 12.0, None), (50.0, 51.0, None), (40.0, 44.0, 30.0)], 50.0, 0.0),
    # Constant trace: zero MAD, zero AUCs; no global max (NaN feature).
    ("flat", 30.0, 35.0, 65.0, 40.0, "flat", np.nan, 10.0),
    ("fps20", 10.0, 20.0, 40.0, 20.0, [(22.0, 25.0, 50.0)], 70.0, 9.0),
    # Ends 2 s after valve-off: every later window is clipped to the recording.
    ("truncated", 30.0, 35.0, 37.0, 40.0, [(34.0, 36.5, 40.0)], 60.0, 9.0),
]

ROW_NAMES = [r[0] for r in _ROWS]


def trace(i: int, dur: float, fps: float, events) -> np.ndarray:
    """A seeded 10 ± 0.5 trace of ``dur`` s plus ``events``, NaN-padded to ``N_COLS``."""
    rng = np.random.default_rng(100 + i)
    fps = FPS if not np.isfinite(fps) else fps
    n = N_COLS if not np.isfinite(dur) else int(round(dur * fps))
    if events == "flat":
        x = np.full(n, 10.0)
    else:
        x = 10.0 + rng.normal(0.0, 0.5, n)
        for start, stop, height in events:
            a, b = int(round(start * fps)), int(round(stop * fps))
            if height is None:
                x[a:b] = np.nan
            else:
                x[a:b] += height
    out = np.full(N_COLS, np.nan)
    out[:n] = x
    return out


def wide_rows(dataset: str = "DS-A", trial_type: str = "testing") -> pd.DataFrame:
    """One wide-table row per entry of ``_ROWS`` (keys, timing, per-fly values, trace)."""
    meta, traces = [], []
    for i, (name, on, off, dur, fps, events, gmax, tmin) in enumerate(_ROWS):
        meta.append({
            "dataset": dataset, "fly": f"fly_{name}", "fly_number": 1,
            "trial_label": f"{trial_type}_{i + 1}_{name}", "trial_type": trial_type,
            "fps": fps, "trial_odor_on_s": on, "trial_odor_off_s": off,
            "trial_duration_s": dur, "global_max": gmax, "trimmed_global_min": tmin,
            "frozen": False,
        })
        traces.append(trace(i, dur, fps, events))
    trace_df = pd.DataFrame(np.vstack(traces), columns=[f"dir_val_{j}" for j in range(N_COLS)])
    return pd.concat([pd.DataFrame(meta), trace_df], axis=1)


def pipeline_rule():
    """The pipeline's AUC threshold rule for ``PIPELINE_WIDE_KEYS``."""
    from fbpipe.analysis.threshold import ThresholdRule

    return ThresholdRule.from_mapping(PIPELINE_WIDE_KEYS)


# ---------------------------------------------------------------------------
# A tiny ordinal model with the production model's stored settings
# ---------------------------------------------------------------------------

#: Features of the tiny model: one AUC log ratio (needs ``fbr_auc_eps``), one
#: signal feature and one whole-trial value.
TINY_FEATURES = ["AUC-During-Before-LogRatio", "peak_z", "local_max"]
#: Member k of the guard ensemble votes 3 when peak_z exceeds this, else 0.
MEMBER_PEAK_Z = (5.0, 20.0, 50.0, 100.0, 150.0)


def _fit(X: pd.DataFrame, scores: np.ndarray, seed: int):
    import xgboost as xgb
    from flybehavior_response import xgb_ordinal as xo

    labels = np.searchsorted(xo.ORIGINAL_CLASSES, np.asarray(scores))
    params = {"objective": "multi:softprob", "num_class": xo.N_CLASSES, "max_depth": 2,
              "eta": 0.6, "seed": seed, "nthread": 1}
    return xgb.train(params, xgb.DMatrix(X, label=labels), 6)


def build_tiny_model(root: Path, *, odor_delay_s: float | None = 1.0,
                     guard_min_agree: float | None = 0.6) -> Path:
    """Write a model that scores 2 on every trial, plus (optionally) a two-guard.

    With the guard, a 2 stands only where enough members see peak_z above
    their own cut (``MEMBER_PEAK_Z``): responding traces keep their 2, flat
    ones are demoted to 1. Returns the model file's path.
    """
    from flybehavior_response import xgb_ordinal as xo
    from flybehavior_response.ensemble_uncertainty import save_ensemble

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    X = pd.DataFrame({
        "AUC-During-Before-LogRatio": rng.uniform(-3.0, 8.0, 400),
        "peak_z": rng.uniform(0.0, 200.0, 400),
        "local_max": rng.uniform(5.0, 120.0, 400),
    })[TINY_FEATURES]
    base = _fit(X, np.full(len(X), 2), seed=0)
    xo.attach_scoring_config(base, recipe="tiny-test", fill_values={}, auc_eps=1.0,
                             odor_delay_s=odor_delay_s)
    if guard_min_agree is not None:
        members = [_fit(X, np.where(X["peak_z"] > cut, 3, 0), seed=k)
                   for k, cut in enumerate(MEMBER_PEAK_Z)]
        save_ensemble(members, root / "ens")
        xo.attach_two_guard(base, "ens", guard_min_agree)   # relative to the model file
    path = root / "model.json"
    base.save_model(str(path))
    return path
