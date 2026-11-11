#!/usr/bin/env python3
"""Run the post-YOLO pipeline on the live-eye comparison trees.

Two arms, each a real copy of the four sensitivity datasets (no mp4s):

* ``anchor`` -- ``Data/flys_AnchorEye``: production tables, untouched.
* ``live``   -- ``Data/flys_LiveEye``: eye measured per frame
  (``live_eye_shadow.py``); november_07_batch_1 re-run on brightened video.

Both arms go through the same stages with the same code and config, so the
only difference between their outputs is the eye (and Nov 7's brightening).
The stages mirror ``run_workflows.py`` call for call but write only into the
comparison folders: no driver, so no freeze cache, secure cleanup, SMB/Box
copies, eye/timing/light QC reports or ntfy.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

COLE = Path("/home/ramanlab/Documents/cole")
DATA = COLE / "Data"
CONFIG = REPO / "config" / "config_new.yaml"
ARMS = {"anchor": DATA / "flys_AnchorEye", "live": DATA / "flys_LiveEye", "drift": DATA / "flys_DriftEye",
        "livehold": DATA / "flys_LiveEyeOdorHold", "livelight": DATA / "flys_LiveEyeLightHold",
        "nolightscale": DATA / "flys_NoLightMinMax", "nolightmin": DATA / "flys_NoLightMin",
        "final": DATA / "flys_FinalMethod"}
#: Arms whose per-trial steps run with config_new.yaml plus these overrides
#: (written to a copy beside the arm's CSVs; config_new.yaml is never edited).
ARM_CONFIG_OVERRIDES = {
    "nolightscale": {"distance_limits": {"exclude_light_frames_from_minmax": True}},
    "nolightmin": {"distance_limits": {"exclude_light_frames_from_minmax": True,
                                       "light_frames_left_out_of": "min"}},
    # what goes into config_new.yaml for the Sensitivity window datasets
    "final": {"distance_limits": {"exclude_light_frames_from_minmax": True,
                                  "light_frames_left_out_of": "min",
                                  "light_exclusion_datasets": ["IAA-Sensitivity-1-1"]},
              "eye_measure": {"mode": "live_light_hold", "datasets": ["IAA-Sensitivity-1-1"]}},
}
CSV_ROOT = DATA / "CSVs-LiveEye"
RESULTS_ROOT = COLE / "Results" / "Figures" / "LiveEye-Comparison"
DATASETS = (
    "3Oct-Sensitivity-1-0.1",
    "Hex-Sensitivity-1-0.1",
    "IAA-Sensitivity-1-1",
    "Paraffin-Sensitivity-1-100",
)
PER_TRIAL_STEPS = (
    "reject_bad_proboscis",
    "distance_stats",
    "distance_normalize",
    "detect_dropped_frames",
    "rms_copy_filter",
    "update_ofm_state",
)
STEM = "all_envelope_rows_wide_combined_base"
_ALLOWED = (*ARMS.values(), CSV_ROOT, RESULTS_ROOT)
_SCRUB_ENV = ("MAIN_DIRECTORIES", "FLAGGED_ROOT", "CLASS2_MIN", "CLASS2_MAX", "FBPIPE_FREEZE_PLAN")


def _guard(path: Path) -> Path:
    path = Path(path).resolve()
    if not any(path == a or a in path.parents for a in _ALLOWED):
        raise SystemExit(f"refusing to write {path}: outside the comparison folders")
    return path


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB_ENV}
    env.update(
        FBPIPE_DISABLE_NTFY="1",
        PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO)]),
        OPENBLAS_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    env.update(extra)
    return env


def _settings(arm: str | None = None):
    """Runtime settings for an arm: its own config (overrides merged) when it has one."""
    from fbpipe.config import load_settings
    from scripts.analysis import envelope_combined as ec
    from scripts.analysis.envelope_visuals import set_protocol

    settings = load_settings(_arm_config(arm) if arm else CONFIG)
    set_protocol(settings.protocol)
    ec.set_runtime_settings(settings)
    return settings


def _arm_config(arm: str) -> Path:
    """config_new.yaml, or a copy of it with the arm's overrides merged in."""
    if arm not in ARM_CONFIG_OVERRIDES:
        return CONFIG
    import yaml

    data = yaml.safe_load(CONFIG.read_text())
    for block, values in ARM_CONFIG_OVERRIDES[arm].items():
        data.setdefault(block, {}).update(values)
    out = _guard(CSV_ROOT / arm / f"config_{arm}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def stage_steps(arm: str) -> None:
    config = _arm_config(arm)
    for ds in DATASETS:
        root = _guard(ARMS[arm] / ds)
        qc = _guard(RESULTS_ROOT / arm / "_qc_sink")
        qc.mkdir(parents=True, exist_ok=True)
        steps = PER_TRIAL_STEPS
        if "eye_measure" in ARM_CONFIG_OVERRIDES.get(arm, {}):
            steps = ("eye_live_measure", *PER_TRIAL_STEPS)   # the production step, as the driver runs it
        cmd = [sys.executable, "-m", "fbpipe.pipeline", "--config", str(config), *steps]
        print(f"[{arm}] per-trial steps → {root}", flush=True)
        subprocess.run(
            cmd, check=True, cwd=REPO,
            env=_env(MAIN_DIRECTORY=str(root), EYE_QC_REPORT_DIR=str(qc),
                     LIGHT_CHECK_CSV=str(qc / "light_stimulus_flags.csv")),
        )


def stage_combine(arm: str) -> None:
    from fbpipe.utils.frozen_folders import frozen_folders_for_root
    from scripts.analysis.envelope_combined import CombineConfig, combine_distance_angle

    settings = _settings(arm)
    for ds in DATASETS:
        root = _guard(ARMS[arm] / ds)
        skip = frozenset(frozen_folders_for_root(settings, root) or ())
        print(f"[{arm}] envelopes → {root} (frozen folders left as is: {len(skip)})", flush=True)
        combine_distance_angle(CombineConfig(
            root=root, skip_folders=skip, fps_default=40.0, window_sec=0.25,
            odor_on_s=30.0, odor_off_s=60.0, odor_latency_s=2.15, save_trial_plots=False,
        ))


def stage_wide(arm: str) -> None:
    from fbpipe.analysis.threshold import ThresholdRule
    from fbpipe.config import load_raw_config
    from fbpipe.utils.frozen_folders import frozen_folders_for_root
    from scripts.analysis import envelope_combined as ec

    settings = _settings(arm)
    wide = load_raw_config(CONFIG)["analysis"]["combined"]["combined_base"]["wide"]
    out = _guard(CSV_ROOT / arm)
    out.mkdir(parents=True, exist_ok=True)
    roots = [str(_guard(ARMS[arm] / ds)) for ds in DATASETS]
    frozen = {Path(r).name: n for r in roots if (n := frozen_folders_for_root(settings, r))}
    print(f"[{arm}] wide CSV → {out}", flush=True)
    ec.build_wide_csv(
        roots, str(out / f"{STEM}.csv"),
        measure_cols=wide["measure_cols"], fps_fallback=float(wide["fps_fallback"]),
        exclude_roots=[], distance_limits=(settings.class2_min, settings.class2_max),
        config_path=CONFIG, trial_type_filter=wide.get("trial_type_filter"),
        extra_trial_exports={"training": str(out / f"{STEM}_training.csv"),
                             "pretest": str(out / f"{STEM}_pretest.csv")},
        non_reactive_threshold=settings.non_reactive_span_px,
        use_per_trial_baseline=bool(wide["use_per_trial_baseline"]),
        threshold_rule=ThresholdRule.from_mapping(wide),
        frozen_slices=None, frozen_folders=frozen or None,
    )


def stage_score(arm: str) -> None:
    from fbpipe.config import load_raw_config, load_settings
    from fbpipe.steps import predict_reactions

    out = _guard(CSV_ROOT / arm)
    settings = load_settings(CONFIG)
    rp = dataclasses.replace(
        settings.reaction_prediction,
        data_csv=str(out / f"{STEM}.csv"),
        extra_data_csvs=(str(out / f"{STEM}_pretest.csv"),),
        output_csv=str(_guard(out / "model_predictions.csv")),
    )
    settings = dataclasses.replace(settings, reaction_prediction=rp)
    print(f"[{arm}] scoring testing + pretest → {rp.output_csv}", flush=True)
    predict_reactions.main(settings)

    ts = load_raw_config(CONFIG)["analysis"]["training_scores"]
    cmd = [
        sys.executable, str(REPO / "scripts" / "analysis" / "naive_vs_trial_score_bars.py"),
        "--score-training-only",
        "--training-wide-csv", str(out / f"{STEM}_training.parquet"),
        "--training-predictions-csv", str(_guard(out / "model_predictions_training.csv")),
        "--model-path", str(ts["model_path"]),
        "--rescore",
    ]
    print(f"[{arm}] scoring training trials", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=_env())


def stage_score_allflies(arm: str) -> None:
    """Score every fly, the hand-flagged ones included, into *_allflies.csv.

    The regular ``score`` stage drops flies listed in flagged-flys-truth.csv
    (badtrack, moved, badMount, ...), exactly like production. Those are the
    flies an eye-anchor change is most likely to affect, so the comparison
    scores them too -- into separate files, never the regular ones.
    """
    from fbpipe.config import load_settings
    from fbpipe.steps import predict_reactions

    import pandas as pd

    out = _guard(CSV_ROOT / arm)
    base = load_settings(CONFIG)
    # An existing, header-only list: "" would switch predict_reactions to its
    # older span-threshold exclusion instead of excluding nobody. And with
    # nobody excluded, predict_reactions hands the scorer the main table only
    # and silently skips extra_data_csvs -- so testing and pretest are scored
    # in two calls and joined.
    no_flags = _guard(CSV_ROOT / "_no_flagged_flies.csv")
    no_flags.write_text('dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment\n')
    parts = []
    for table in (f"{STEM}.csv", f"{STEM}_pretest.csv"):
        part = _guard(out / f"_allflies_{Path(table).stem}.csv")
        rp = dataclasses.replace(base.reaction_prediction, data_csv=str(out / table),
                                 extra_data_csvs=(), output_csv=str(part))
        settings = dataclasses.replace(base, reaction_prediction=rp, flagged_flies_csv=str(no_flags))
        print(f"[{arm}] scoring {table}, flagged flies included", flush=True)
        predict_reactions.main(settings)
        parts.append(pd.read_csv(part))
    pd.concat(parts, ignore_index=True).to_csv(_guard(out / "model_predictions_allflies.csv"), index=False)

    from fbpipe.config import load_raw_config

    ts = load_raw_config(CONFIG)["analysis"]["training_scores"]
    script = REPO / "scripts" / "analysis" / "naive_vs_trial_score_bars.py"
    code = (
        "import importlib.util, sys; from pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('nvt', {str(script)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['nvt'] = m; spec.loader.exec_module(m)\n"
        "m.FLAGGED_FLIES_CSV = Path('/nonexistent/flagged-flys-truth.csv')\n"
        "sys.exit(m.main(sys.argv[1:]))\n"
    )
    cmd = [
        sys.executable, "-c", code,
        "--score-training-only",
        "--training-wide-csv", str(out / f"{STEM}_training.parquet"),
        "--training-predictions-csv", str(_guard(out / "model_predictions_training_allflies.csv")),
        "--model-path", str(ts["model_path"]),
        "--rescore",
    ]
    print(f"[{arm}] scoring training trials, flagged flies included", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=_env())


def stage_window(arm: str) -> None:
    from fbpipe.config import load_raw_config, load_settings

    cfg = load_raw_config(CONFIG)["analysis"]["sensitivity_window_results"]
    settings = load_settings(CONFIG)
    out_dir = _guard(RESULTS_ROOT / arm / "Sensitivity-Results")
    no_figs = _guard(RESULTS_ROOT / arm / "_no_production_figures")
    no_figs.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts" / "analysis" / "sensitivity_window_results.py"),
        "--out-dir", str(out_dir), "--start", str(cfg["start"]), "--end", str(cfg["end"]),
    ]
    for ds in cfg["datasets"]:
        if ds in DATASETS:
            cmd += ["--dataset", str(ds)]
    cmd += ["--bin-hours", str(cfg["bin_hours"]), "--split-date", str(cfg["split_date"])]
    for ds in cfg.get("split_cohorts") or []:
        if ds in DATASETS:
            cmd += ["--split-cohort", str(ds)]
    for ds in cfg.get("trial_series_cohorts") or []:
        if ds in DATASETS:
            cmd += ["--trial-series-cohort", str(ds)]
    cmd += [
        "--csv-dir", str(_guard(CSV_ROOT / arm)),
        "--flagged-flies-csv", str(cfg.get("flagged_flies_csv") or settings.flagged_flies_csv),
        "--data-root", str(_guard(ARMS[arm])),
        "--figures-root", str(no_figs),
        "--config", str(CONFIG),
    ]
    print(f"[{arm}] sensitivity results → {out_dir}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=_env())


STAGES = {
    "steps": stage_steps,
    "combine": stage_combine,
    "wide": stage_wide,
    "score": stage_score,
    "window": stage_window,
    "score_allflies": stage_score_allflies,
}


def main(argv=None):
    global DATASETS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), action="append", required=True)
    ap.add_argument("--stage", choices=list(STAGES), action="append")
    ap.add_argument("--dataset", choices=DATASETS, action="append",
                    help="limit every stage to these datasets (default: all four)")
    args = ap.parse_args(argv)
    if args.dataset:
        DATASETS = tuple(args.dataset)
    for arm in args.arm:
        for name in args.stage or [s for s in STAGES if s != "score_allflies"]:
            t0 = time.time()
            STAGES[name](arm)
            print(f"[{arm}] {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
