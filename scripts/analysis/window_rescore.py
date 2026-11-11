"""Rescore trials with the production ordinal model on a different time window.

The ordinal scorer (``reaction_prediction.model_path``) sees a trial through
three windows: baseline, "during" and "after" (30 s from the end of "during").
This module recomputes every window feature from the ``dir_val_*`` trace with
the "during" slot moved, then scores the trial with the same model, the way
production (``flybehavior-response predict-ordinal``) scores it. ``on`` /
``off`` are the valve times:

    odor             [on, off)              the production layout
    after            [off, off + after_s)   only the time after odor-off
    odor_plus_after  [on, off + after_s)    the odor plus the time after it

Baseline is always ``[0, on)``; "after" always follows the end of "during".

Odor-arrival delay. Production scores a model saved with ``fbr_odor_delay_s``
= d (any d, 0 included; recipe ``v2-auc-logratio-delay1s``, d = 1 s, deployed
2026-11-09) from the trace alone, on FlyBehaviorScoring's ``window_features``
layout: baseline [0, on), a transit [on, on + d) left out of every value, odor
[on + d, off + d) and post from off + d. Here both valve references move by d
the same way:

    odor             [on + d, off + d)              production's own windows
    after            [off + d, off + d + after_s)
    odor_plus_after  [on + d, off + d + after_s)

The baseline does not move, and the transit is left out on every window,
whole-trial ``local_min`` / ``local_max`` included (those stay production's
values whatever the window). Every value comes from the scorer's own functions
(``compute_window_features`` / ``compute_signal_features``), the AUCs too: for
a delay model production recomputes them with the scorer's threshold (baseline
median + 3 MAD sigma) and never reads the wide table's AUC-* columns, so the
pipeline rule from ``--config`` is not used. Only ``global_max`` /
``trimmed_global_min`` come from the table. The ``score_window`` label names
the shift: "odor + 5 s after (+1 s arrival delay)". With the 2026-11-09 guard60
model, the ``odor`` window reproduced model_predictions.csv on 100 % of its
11,942 trials (all 39 datasets, the guard's 113 demoted 2s included), checked
2026-11-10.

Delay-free models (no ``fbr_odor_delay_s``): whole-trial values cover the full
trial, and the AUC values use the PIPELINE's threshold rule (fbpipe
ThresholdRule from ``analysis.combined.combined_base.wide`` in the config), the
rule that built the wide tables' AUC-* columns those models read, not the
scorer repo's own MAD rule, which agreed with production on only 92.5 % of
scores. With it, the ``odor`` window reproduced model_predictions.csv on
99.9 % of IAA-Sensitivity-24-1 trials (100 % on responded / not), checked
2026-10-16.

Two-guard. A model saved with ``fbr_two_guard_ensemble`` (the deployed guard60:
20 members, ``fbr_two_guard_min_agree`` 0.6) scores a 2 as 1 unless that share
of its ensemble also says >= 2. ``xgb_ordinal.predict_ordinal_scores`` applies
it, here exactly as in production, on the features of the window scored.

Caveat: the model was trained on 30 s "during" windows. Scores on any other
length are the model's reading of an input it never saw in training.

CLI: write a copy of a predictions CSV with ``score`` / ``prediction`` taken
from a different window, for the figure scripts that read that CSV.

    python3 scripts/analysis/window_rescore.py \
        --predictions-csv .../model_predictions.csv \
        --wide-csv .../all_envelope_rows_wide_combined_base_pretest.parquet \
                   .../all_envelope_rows_wide_combined_base.parquet \
        --model-path .../model_ordinal_xgb_v2_both.json --config config/config_new.yaml \
        --window odor_plus_after --after-s 5 --paired-only \
        --output-csv .../model_predictions_odor_plus_5s.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from flybehavior_response import xgb_ordinal as xo  # noqa: E402
from flybehavior_response.window_features import (  # noqa: E402
    WINDOW_FEATURES,
    Windows,
    compute_window_features,
)

from fbpipe.analysis.threshold import ThresholdRule  # noqa: E402
from fbpipe.utils.frozen_folders import drop_frozen  # noqa: E402

#: Upstream's after-window length in frames (envelope_combined.AFTER_FRAMES).
AFTER_FRAMES = 1200

DEFAULT_FPS = 40.0
PROTOCOL_ODOR_ON_S = 30.0
PROTOCOL_ODOR_OFF_S = 60.0
KEY = ["dataset", "fly", "fly_number", "trial_label"]
_TIMING = ["fps", "trial_odor_on_s", "trial_odor_off_s", "trial_duration_s"]


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSpec:
    """Where the "during" slot starts and ends, relative to valve on / off."""

    name: str
    label: str
    start_ref: str      # "on" | "off"
    start_offset_s: float
    end_ref: str
    end_offset_s: float


def window_spec(name: str, after_s: float = 5.0) -> WindowSpec:
    a = float(after_s)
    tag = f"{a:g} s"
    specs = {
        "odor": WindowSpec("odor", "odor", "on", 0.0, "off", 0.0),
        "after": WindowSpec("after", f"{tag} after odor", "off", 0.0, "off", a),
        "odor_plus_after": WindowSpec(
            "odor_plus_after", f"odor + {tag} after", "on", 0.0, "off", a
        ),
    }
    if name not in specs:
        raise ValueError(f"window must be one of {sorted(specs)}: {name!r}")
    return specs[name]


def window_label(spec: WindowSpec, delay_s: float | None) -> str:
    """``spec.label``, naming the model's odor-arrival shift when it has one."""
    if not delay_s:
        return spec.label
    return f"{spec.label} (+{float(delay_s):g} s arrival delay)"


def _num(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def trial_windows(
    ctx: Mapping,
    n_cols: int,
    spec: WindowSpec,
    default_fps: float = DEFAULT_FPS,
    *,
    delay_s: float | None = None,
) -> tuple[Windows, float]:
    """Frame indices of one trial's windows under ``spec``, and the fps used.

    ``delay_s`` (a model's ``fbr_odor_delay_s``) moves both valve references,
    on and off, by the delay, as the scorer's ``window_features.trial_windows``
    moves its odor window; the baseline still ends at valve-on. ``None`` is
    the delay-free layout.
    """
    fps = _num(ctx.get("fps"))
    fps = fps if fps is not None and fps > 0 else float(default_fps)
    dur = _num(ctx.get("trial_duration_s"))
    n = n_cols if dur is None or dur <= 0 else min(n_cols, int(round(dur * fps)))
    if n < 3:
        raise ValueError(f"trace too short for odor windows: {n} frames")
    on = _num(ctx.get("trial_odor_on_s"))
    off = _num(ctx.get("trial_odor_off_s"))
    on = PROTOCOL_ODOR_ON_S if on is None else on
    off = PROTOCOL_ODOR_OFF_S if off is None else off
    if delay_s is None:
        ref = {"on": on, "off": off}
    else:
        ref = {"on": on + float(delay_s), "off": off + float(delay_s)}
    start_s = ref[spec.start_ref] + spec.start_offset_s
    end_s = ref[spec.end_ref] + spec.end_offset_s
    b = min(max(int(round(on * fps)), 1), n - 2)
    s = min(max(int(round(start_s * fps)), b), n - 1)
    e = min(max(int(round(end_s * fps)), s + 1), n)
    return Windows(b, s, e, n), fps


# ---------------------------------------------------------------------------
# Features and scores
# ---------------------------------------------------------------------------


def _dir_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in df.columns if str(c).startswith("dir_val_")),
        key=lambda c: int(str(c).rsplit("_", 1)[-1]),
    )


def threshold_rule_from_config(config_path: Path | str) -> ThresholdRule:
    """The AUC threshold rule the pipeline built the wide tables with."""
    import yaml

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    wide = (((cfg.get("analysis") or {}).get("combined") or {})
            .get("combined_base") or {}).get("wide") or {}
    return ThresholdRule.from_mapping(wide)


def _segment_auc(values: np.ndarray, threshold: float, dt: float) -> float:
    """Σ (x − θ)⁺ · dt over finite samples (envelope_combined._segment_auc)."""
    finite = values[np.isfinite(values)]
    diff = finite - threshold
    return float(diff[diff > 0].sum() * dt) if finite.size and dt > 0 else 0.0


def _row_features(
    trace: np.ndarray, ctx: Mapping, spec: WindowSpec, rule: ThresholdRule | None
) -> dict:
    w, fps = trial_windows(ctx, len(trace), spec)
    gmin = _num(ctx.get("trimmed_global_min"))
    gmin = float("nan") if gmin is None else gmin
    gmax = _num(ctx.get("global_max"))
    out = {"global_max": float("nan") if gmax is None else gmax, "trimmed_global_min": gmin}
    out.update(compute_window_features(trace, fps, w, trimmed_global_min=gmin))
    # compute_window_features drops [baseline_end, odor_start) from the
    # whole-trial extremes (it is the transit gap there). Here that gap is the
    # odor itself for the "after" window, so restore the full-trial values.
    whole = np.asarray(trace, dtype=float)[: w.n_frames]
    whole = whole[np.isfinite(whole)]
    if whole.size:
        out["local_min"] = float(whole.min())
        out["local_max"] = float(whole.max())
        out["local_max_over_global_min"] = (
            out["local_max"] / gmin if math.isfinite(gmin) and gmin != 0 else float("nan")
        )
    if rule is not None:
        # The pipeline's AUC rule (see module docstring), on the moved windows.
        x = np.asarray(trace, dtype=float)[: w.n_frames]
        before = x[: w.baseline_end]
        theta = rule.theta_from_baseline(before, fps=fps)
        theta = theta if np.isfinite(theta) else 0.0
        dt = 1.0 / fps
        out["AUC-Before"] = _segment_auc(before, theta, dt)
        out["AUC-During"] = _segment_auc(x[w.odor_start: w.odor_end], theta, dt)
        out["AUC-After"] = _segment_auc(x[w.odor_end: w.odor_end + AFTER_FRAMES], theta, dt)
    out.update(xo.compute_signal_features(
        trace, fps=fps, odor_on_idx=w.odor_start, odor_off_idx=w.odor_end,
        n_frames=w.n_frames, baseline_end_idx=w.baseline_end,
    ))
    return out


#: The production "during" slot, for a delay model's whole-trial values.
_ODOR = WindowSpec("odor", "odor", "on", 0.0, "off", 0.0)
#: Values over the whole trial: the same whatever the "during" slot.
_WHOLE_TRIAL = ("local_min", "local_max", "local_max_over_global_min")


def _delay_row_features(
    trace: np.ndarray, ctx: Mapping, spec: WindowSpec, delay_s: float
) -> dict:
    """One row on the odor-delay layout, valued the way production values it.

    The scorer's own functions on the moved windows (``xgb_ordinal._delayed_raw``
    on its odor window). ``compute_window_features`` drops
    ``[baseline_end, odor_start)`` from the whole-trial extremes: the transit
    second on production's layout, but the odor too once the "during" slot
    moves past it, so those come from production's windows of the trial.
    """
    w, fps = trial_windows(ctx, len(trace), spec, delay_s=delay_s)
    gmin = _num(ctx.get("trimmed_global_min"))
    gmin = float("nan") if gmin is None else gmin
    gmax = _num(ctx.get("global_max"))
    out = {"global_max": float("nan") if gmax is None else gmax, "trimmed_global_min": gmin}
    out.update(compute_window_features(trace, fps, w, trimmed_global_min=gmin))
    odor_w, _ = trial_windows(ctx, len(trace), _ODOR, delay_s=delay_s)
    if w != odor_w:
        whole = compute_window_features(trace, fps, odor_w, trimmed_global_min=gmin)
        out.update({key: whole[key] for key in _WHOLE_TRIAL})
    out.update(xo.compute_signal_features(
        trace, fps=fps, odor_on_idx=w.odor_start, odor_off_idx=w.odor_end,
        n_frames=w.n_frames, baseline_end_idx=w.baseline_end,
    ))
    return out


def raw_features(
    df: pd.DataFrame,
    spec: WindowSpec,
    *,
    rule: ThresholdRule | None = None,
    odor_delay_s: float | None = None,
) -> pd.DataFrame:
    """Every window value and signal feature for each row, under ``spec``.

    ``odor_delay_s`` is the model's ``fbr_odor_delay_s``. Not None: the delay
    layout, every value (AUCs included) the scorer's own, as production
    computes them for that model; ``rule`` is then unused. None: the
    delay-free layout, where ``rule`` is the pipeline's AUC threshold; None
    there falls back to the scorer repo's MAD rule, which does NOT reproduce
    a delay-free model's production scores.
    """
    cols = _dir_cols(df)
    if not cols:
        raise ValueError("No dir_val_* trace columns in the table.")
    traces = df[cols].to_numpy(dtype=float)
    ctx_cols = [c for c in _TIMING + ["global_max", "trimmed_global_min"] if c in df.columns]
    ctx = df[ctx_cols].reset_index(drop=True)
    if odor_delay_s is None:
        rows = [_row_features(traces[i], ctx.iloc[i], spec, rule) for i in range(len(df))]
    else:
        delay = float(odor_delay_s)
        rows = [_delay_row_features(traces[i], ctx.iloc[i], spec, delay) for i in range(len(df))]
    return pd.DataFrame(rows, columns=["global_max", "trimmed_global_min",
                                       *WINDOW_FEATURES, *xo.SIGNAL_FEATURES])


def model_features(
    df: pd.DataFrame,
    spec: WindowSpec,
    cfg: "xo.ScoringConfig",
    *,
    rule: ThresholdRule | None = None,
) -> pd.DataFrame:
    """The matrix a model with scoring config ``cfg`` scores, on window ``spec``.

    The model's stored delay picks the layout (``raw_features``); its feature
    list, fill values and AUC epsilon finish the matrix, as in production.
    For a delay model's ``odor`` window this IS production's
    ``features_for_model``.
    """
    raw = raw_features(df, spec, rule=rule, odor_delay_s=cfg.odor_delay_s)
    return xo._finish(raw, cfg.features, cfg.fill_values, cfg.auc_eps)


def model_odor_delay_s(model_path: Path | str) -> float | None:
    """The odor-arrival delay a model file was saved with, or None."""
    return xo.scoring_config(xo.load_xgb_model(model_path)).odor_delay_s


def _scores(booster, cfg, df: pd.DataFrame, spec: WindowSpec,
            rule: ThresholdRule | None) -> np.ndarray:
    # predict_ordinal_scores applies the model's stored two-guard, if any.
    feats = model_features(df, spec, cfg, rule=rule)
    return np.asarray(xo.predict_ordinal_scores(booster, feats)).astype(int)


def rescore(
    df: pd.DataFrame,
    spec: WindowSpec,
    model_path: Path | str,
    *,
    rule: ThresholdRule | None = None,
) -> np.ndarray:
    """Ordinal scores of every row of ``df`` on window ``spec``.

    Scored as production scores: the model's stored delay, features, fill
    values and AUC epsilon, then its stored two-guard.
    """
    booster = xo.load_xgb_model(model_path)
    return _scores(booster, xo.scoring_config(booster), df, spec, rule)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in KEY:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_wide_rows(
    wide_paths: Sequence[Path | str], datasets: Sequence[str]
) -> pd.DataFrame:
    """Rows of ``datasets`` from each wide table, frozen folders dropped."""
    wanted = sorted({str(d) for d in datasets})
    frames = []
    for path in wide_paths:
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            df = pq.read_table(path, filters=[("dataset", "in", wanted)]).to_pandas()
        else:
            df = pd.read_csv(path)
            df = df.loc[df["dataset"].astype(str).isin(wanted)]
        frames.append(drop_frozen(df))
    if not frames:
        return pd.DataFrame(columns=KEY)
    out = _norm_keys(pd.concat(frames, ignore_index=True))
    return out.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)


def apply_scores(
    template: pd.DataFrame,
    rescored: pd.DataFrame,
    *,
    binary_threshold: int,
    window_label: str,
) -> pd.DataFrame:
    """``template`` with ``score`` / ``prediction`` from ``rescored``.

    Rows with no rescored counterpart are dropped, not left at their old score:
    a CSV mixing two windows would be worse than a shorter one.
    """
    key_t = _norm_keys(template[KEY])
    new = _norm_keys(rescored[KEY + ["score"]]).rename(columns={"score": "_new"})
    merged = template.reset_index(drop=True).join(
        key_t.merge(new, on=KEY, how="left")[["_new"]]
    )
    merged = merged.loc[merged["_new"].notna()].copy()
    merged["score"] = merged.pop("_new").astype(int)
    merged["prediction"] = (merged["score"] >= int(binary_threshold)).astype(int)
    merged["score_window"] = window_label
    return merged.reset_index(drop=True)


def rescore_predictions(
    template: pd.DataFrame,
    wide_paths: Sequence[Path | str],
    model_path: Path | str,
    spec: WindowSpec,
    *,
    binary_threshold: int,
    rule: ThresholdRule | None = None,
) -> pd.DataFrame:
    """Every trial of ``template`` rescored on ``spec``, as a predictions table.

    ``score_window`` is ``spec.label``, plus the model's arrival shift if any.
    """
    booster = xo.load_xgb_model(model_path)
    cfg = xo.scoring_config(booster)
    label = window_label(spec, cfg.odor_delay_s)
    wide = load_wide_rows(wide_paths, template["dataset"].astype(str).unique())
    wide = wide.merge(_norm_keys(template[KEY]).drop_duplicates(), on=KEY, how="inner")
    if wide.empty:
        return template.iloc[0:0].assign(score_window=label)
    scores = _scores(booster, cfg, wide, spec, rule)
    rescored = wide[KEY].assign(score=scores)
    return apply_scores(template, rescored, binary_threshold=binary_threshold,
                        window_label=label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--predictions-csv", required=True, type=Path,
                   help="Production predictions; its rows are the trials rescored.")
    p.add_argument("--wide-csv", required=True, nargs="+", type=Path,
                   help="Wide trace tables (pretest and testing).")
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path,
                   help="Pipeline config; its wide-table AUC threshold rule is used "
                        "for delay-free models (a delay model keeps its own).")
    p.add_argument("--window", default="odor_plus_after",
                   choices=["odor", "after", "odor_plus_after"])
    p.add_argument("--after-s", type=float, default=5.0)
    p.add_argument("--binary-threshold", type=int, default=2)
    p.add_argument("--cohorts", nargs="*", default=None)
    p.add_argument("--paired-only", action="store_true",
                   help="Only datasets that have pretest rows (the pre-vs-post cohorts).")
    p.add_argument("--output-csv", required=True, type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = window_spec(args.window, args.after_s)
    delay = model_odor_delay_s(args.model_path)
    if delay is not None:
        print(f"[window-rescore] model's odor-arrival delay is {delay:g} s: windows "
              f"shifted +{delay:g} s and AUCs on the model's own threshold, as production "
              "scores it (the config's AUC rule applies to delay-free models only)")
    template = pd.read_csv(args.predictions_csv)
    if args.cohorts:
        template = template.loc[template["dataset"].astype(str).isin(args.cohorts)]
    if args.paired_only and "trial_type" in template.columns:
        is_pre = template["trial_type"].astype(str).str.strip().str.lower() == "pretest"
        paired_sets = set(template.loc[is_pre, "dataset"].astype(str))
        template = template.loc[template["dataset"].astype(str).isin(paired_sets)]
    out = rescore_predictions(
        template.reset_index(drop=True), args.wide_csv, args.model_path, spec,
        binary_threshold=args.binary_threshold,
        rule=threshold_rule_from_config(args.config),
    )
    dropped = len(template) - len(out)
    if dropped:
        print(f"[window-rescore] {dropped} trial(s) had no trace row and were dropped")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"[window-rescore] {len(out)} trial(s) scored on '{window_label(spec, delay)}' "
          f"→ {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
