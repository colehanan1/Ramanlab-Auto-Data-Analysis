#!/usr/bin/env python3
"""Standard figure set for the flies whose experiment started inside a clock window.

Picks every batch folder of ``--dataset`` whose first pre-test trial started in
the half-open window [``--start``, ``--end``) -- Pi local clock time, from the
raw trial file names (``experiment_start_times.start_table``) -- and builds,
for exactly those folders:

    _inputs/                                filtered copies of the pipeline tables
    <dataset>/                              pre vs post score figures (pretest_vs_test_comparison)
    Mean-Traces_Pre-Test_vs_Post-Training/  dataset_mean_traces_tvc
    Cumulative-Responders_Pre-Test_vs_Post-Training/
                                            cumulative_responders: % of flies responding
                                            since odor onset, one figure per odor + all odors
    Responder-Mean-Traces_Pre-Test_vs_Post-Training/
                                            responder_mean_traces: the mean trace of flies
                                            scored as responding vs the rest, per odor + all
    Every-Fly-Heatmaps_Pre-Test_vs_Post-Training/
                                            every_fly_heatmaps: every fly, pre | post, per odor
    Above-Threshold-Traces_Pre-Test_vs_Post-Training/
                                            above_threshold_traces: mean extension above the
                                            red line and % of flies above it, per odor + all
    Graded-PER-Rasters/                     binarized_per_rasters --mode graded
    Training-AUC-vs-Testing-Response/       training_auc_vs_control_response
    Pre-vs-Post-PER-Traces/                 pretest_vs_test_traces (one figure per fly)
    AUC-Before-vs-During/, PeakZ-Before-vs-During/, AUC-LogRatio-Before-vs-During/
                                            auc_before_during --metric auc | peakz | logratio
    Model-Feature-Drivers/                  model_feature_drivers
    FINDINGS.md                             the flies, and the key numbers of the above
    _build_log.txt                          every command and its output

Folders are matched by exact name, never by prefix: ``october_14_batch_1``
would otherwise pull in ``october_14_batch_1_rig_2``, a different rig that may
have started outside the window. Flagged flies (``flagged-flys-truth.csv``
FLY-State 0 or -1) are dropped from every table and trial labels lose their
``-<conc>`` suffix, as in ``day_vs_day_report.py``. The pipeline's own tables
are only read.

Exit status: 0 built, 2 built but a figure step failed, 3 nothing to plot (no
folder started in the window, or every fly in it is flagged). 1 is Python's own,
for a crash.

Usage
-----
    python3 scripts/analysis/start_window_report.py --dataset IAA-Sensitivity-1-1 \\
        --start 13:00 --end 14:00 \\
        --out-dir /home/ramanlab/Documents/cole/Results/Show/IAA-Sensitivity-1-1_by-start-time_2026-10-27/1pm-2pm
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scripts.analysis.starvation_pre_post_comparison as spc  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.experiment_start_times import DATA_ROOT, start_table  # noqa: E402

CONFIG = REPO_ROOT / "config" / "config_new.yaml"
CSV = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
FLAGGED = CSV / "flagged-flys-truth.csv"
PY = sys.executable
#: Exit code for "nothing to plot", apart from the 1 of an uncaught exception:
#: sensitivity_window_results drops a cohort's previous figure set on this one only.
NOTHING_TO_PLOT = 3
#: Model score counted as responding: the score figures' % responding and the
#: cumulative curves must use the same call, so both get it from here.
RESPONDING_SCORE = 2
#: Frame rate and odor onset of the mean traces: each trial loses the mean of its first
#: ODOR_ON_S seconds. The responder split must draw the very same traces.
TRACE_FPS = 40
ODOR_ON_S = 30

_TABLES = (("testing", "all_envelope_rows_wide_combined_base.parquet"),
           ("pretest", "all_envelope_rows_wide_combined_base_pretest.parquet"),
           ("training", "all_envelope_rows_wide_combined_base_training.parquet"))
_PREDICTIONS = (("predictions", "model_predictions.csv"),
                ("training_predictions", "model_predictions_training.csv"))


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

_CLOCK_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", re.IGNORECASE)


def parse_clock(text: str) -> dt.time:
    """``"13:00"``, ``"9:05"``, ``"1pm"``, ``"11:30am"`` -> time of day."""
    m = _CLOCK_RE.match(str(text))
    if not m:
        raise ValueError(f"not a clock time: {text!r}")
    hour, minute, half = int(m[1]), int(m[2] or 0), (m[3] or "").lower()
    if half:
        if not 1 <= hour <= 12:
            raise ValueError(f"not a 12-hour clock time: {text!r}")
        hour = hour % 12 + (12 if half == "pm" else 0)
    return dt.time(hour, minute)


def _clock_text(t: dt.time) -> str:
    hour = t.hour % 12 or 12
    half = "am" if t.hour < 12 else "pm"
    return f"{hour}{half}" if t.minute == 0 else f"{hour}:{t.minute:02d}{half}"


def window_label(lo: dt.time, hi: dt.time) -> str:
    """``1pm-2pm``: the folder-name convention of the by-start-time sets."""
    return f"{_clock_text(lo)}-{_clock_text(hi)}"


def folders_in_window(starts: pd.DataFrame, lo: dt.time, hi: dt.time) -> pd.DataFrame:
    """Rows of ``start_table`` whose first pre-test started in [lo, hi)."""
    clock = pd.to_datetime(starts["first_pretest"]).dt.time
    keep = [(lo <= c < hi) for c in clock]
    return starts.loc[keep].sort_values("first_pretest").reset_index(drop=True)


def drop_folders(window: pd.DataFrame, excluded: Sequence[str]) -> pd.DataFrame:
    """``window`` without the folders named in ``excluded`` (exact names).

    A name that is not in the window is an error, so a typo cannot silently
    leave the folder in."""
    missing = sorted(set(excluded) - set(window["folder"]))
    if missing:
        raise ValueError(f"excluded folder(s) not in the window: {', '.join(missing)}")
    return window.loc[~window["folder"].isin(set(excluded))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def subset_rows(df: pd.DataFrame, dataset: str, folders: Sequence[str],
                flagged_csv: Path | str) -> pd.DataFrame:
    """Rows of exactly these folders of ``dataset``, flagged flies dropped,
    trial labels normalised (``testing_2_hexanol-0.1`` -> ``testing_2_hexanol``)."""
    mask = (df["dataset"].astype(str).str.strip() == dataset) & (
        df["fly"].astype(str).str.strip().isin(set(folders)))
    out = df.loc[mask].copy()
    out["trial_label"] = out["trial_label"].map(spc.normalize_trial_label)
    return spc.drop_flagged(out, flagged_csv)


def write_inputs(dataset: str, folders: Sequence[str], inputs: Path, *,
                 csv_dir: Path = CSV, flagged: Path = FLAGGED) -> dict[str, Path]:
    inputs.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, name in _TABLES:
        df = subset_rows(pd.read_parquet(csv_dir / name), dataset, folders, flagged)
        paths[key] = inputs / f"wide_{key}.parquet"
        df.to_parquet(paths[key], index=False)
        print(f"  {key:8s} wide rows: {len(df)}  flies: "
              f"{df.groupby(['fly', 'fly_number']).ngroups}")
    for key, name in _PREDICTIONS:
        df = subset_rows(pd.read_csv(csv_dir / name), dataset, folders, flagged)
        paths[key] = inputs / f"{key}.csv"
        df.to_csv(paths[key], index=False)
        print(f"  {key:20s} rows: {len(df)}")
    return paths


def fly_table(window: pd.DataFrame, testing: pd.DataFrame, dataset: str,
              flagged: Path) -> pd.DataFrame:
    """Per folder: start time, flies kept, flies dropped as flagged."""
    from fbpipe.config import load_flagged_fly_exclusions

    excluded = load_flagged_fly_exclusions(flagged) if Path(flagged).exists() else set()
    kept = testing.groupby("fly")["fly_number"].nunique()
    rows = []
    for _, r in window.iterrows():
        n_flagged = sum(1 for ds, fly, _num in excluded if ds == dataset and fly == r["folder"])
        rows.append({"folder": r["folder"], "start": pd.Timestamp(r["first_pretest"]),
                     "flies": int(kept.get(r["folder"], 0)), "flagged_dropped": n_flagged})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _run(cmd: list, log) -> int:
    cmd = [str(c) for c in cmd]
    print("  $", " ".join(cmd)[:160])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    log.write(f"\n$ {' '.join(cmd)}\n[exit {r.returncode}]\n{r.stdout[-4000:]}\n"
              f"{r.stderr[-4000:]}\n")
    log.flush()
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        print(f"  !! exit {r.returncode}: {tail}")
    return r.returncode


def run_figures(out: Path, p: dict[str, Path], dataset: str, *, config: Path,
                label: str, log, rate_test: str = "fisher",
                flagged: Path = FLAGGED) -> dict[str, int]:
    """Every figure script on the filtered tables; returns exit code per step.

    ``flagged`` is the truth table the inputs were filtered with, so a step that
    drops flagged flies itself drops the same ones."""
    a = "scripts/analysis/"
    steps = {
        "score figures": [PY, a + "pretest_vs_test_comparison.py",
                          "--predictions-csv", p["predictions"], "--out-root", out,
                          "--binary-threshold", RESPONDING_SCORE, "--rate-test", rate_test,
                          "--flagged-flies-csv", flagged],
        "mean traces": [PY, a + "dataset_mean_traces_tvc.py", "--wide-csv", p["testing"],
                        "--pretest-wide-csv", p["pretest"], "--train-dataset", dataset,
                        "--out-dir", out / "Mean-Traces_Pre-Test_vs_Post-Training",
                        "--fps", TRACE_FPS, "--odor-on-s", ODOR_ON_S,
                        "--flagged-flies-csv", flagged, "--config", config],
        "cumulative responders": [
            PY, a + "cumulative_responders.py", "--pretest-wide-csv", p["pretest"],
            "--testing-wide-csv", p["testing"], "--predictions-csv", p["predictions"],
            "--config", config, "--dataset", dataset, "--flagged-flies-csv", flagged,
            "--binary-threshold", RESPONDING_SCORE, "--label", label,
            "--out-dir", out / "Cumulative-Responders_Pre-Test_vs_Post-Training"],
        "responder traces": [
            PY, a + "responder_mean_traces.py", "--pretest-wide-csv", p["pretest"],
            "--testing-wide-csv", p["testing"], "--predictions-csv", p["predictions"],
            "--dataset", dataset, "--flagged-flies-csv", flagged,
            "--binary-threshold", RESPONDING_SCORE, "--fps", TRACE_FPS,
            "--odor-on-s", ODOR_ON_S, "--label", label,
            "--out-dir", out / "Responder-Mean-Traces_Pre-Test_vs_Post-Training"],
        "every-fly heatmaps": [
            PY, a + "every_fly_heatmaps.py", "--pretest-wide-csv", p["pretest"],
            "--testing-wide-csv", p["testing"], "--predictions-csv", p["predictions"],
            "--config", config, "--dataset", dataset, "--flagged-flies-csv", flagged,
            "--binary-threshold", RESPONDING_SCORE, "--fps", TRACE_FPS, "--label", label,
            "--out-dir", out / "Every-Fly-Heatmaps_Pre-Test_vs_Post-Training"],
        "above-threshold traces": [
            PY, a + "above_threshold_traces.py", "--pretest-wide-csv", p["pretest"],
            "--testing-wide-csv", p["testing"], "--predictions-csv", p["predictions"],
            "--config", config, "--dataset", dataset, "--flagged-flies-csv", flagged,
            "--binary-threshold", RESPONDING_SCORE, "--fps", TRACE_FPS, "--label", label,
            "--out-dir", out / "Above-Threshold-Traces_Pre-Test_vs_Post-Training"],
        "graded rasters": [PY, a + "binarized_per_rasters.py",
                           "--training-wide-csv", p["training"], "--wide-csv", p["testing"],
                           "--pretest-wide-csv", p["pretest"],
                           "--predictions-csv", p["predictions"], "--config", config,
                           "--dataset", dataset, "--out-dir", out / "Graded-PER-Rasters",
                           "--mode", "graded", "--sort-by", "ratio"],
        "training AUC": [PY, a + "training_auc_vs_control_response.py",
                         "--training-wide-csv", p["training"],
                         "--testing-wide-csv", p["testing"],
                         "--predictions-csv", p["predictions"], "--dataset", dataset,
                         "--out-dir", out / "Training-AUC-vs-Testing-Response"],
        "per-fly traces": [PY, a + "pretest_vs_test_traces.py",
                           "--pretest-wide-csv", p["pretest"],
                           "--testing-wide-csv", p["testing"],
                           "--out-dir", out / "Pre-vs-Post-PER-Traces",
                           "--predictions-csv", p["predictions"], "--config", config],
    }
    common = ["--pretest-wide-csv", p["pretest"], "--testing-wide-csv", p["testing"],
              "--predictions-csv", p["predictions"], "--config", config, "--label", label]
    for metric, folder in (("auc", "AUC-Before-vs-During"), ("peakz", "PeakZ-Before-vs-During"),
                           ("logratio", "AUC-LogRatio-Before-vs-During")):
        steps[f"{metric} figures"] = [PY, a + "auc_before_during.py", "--metric", metric,
                                      *common, "--out-dir", out / folder]
    steps["model drivers"] = [PY, a + "model_feature_drivers.py", *common,
                              "--out-dir", out / "Model-Feature-Drivers"]
    return {name: _run(cmd, log) for name, cmd in steps.items()}


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

#: (row label, folder, table key, measure) for the findings tables.
KEY_MEASURES = (
    ("Signed AUC Δ (% × s)", "AUC-Before-vs-During", "auc", "delta_signed"),
    ("Unsigned AUC Δ (% × s)", "AUC-Before-vs-During", "auc", "delta_unsigned"),
    ("Raw-area ratio", "AUC-Before-vs-During", "auc", "ratio_raw"),
    ("Signed peak z, during (SD)", "PeakZ-Before-vs-During", "peakz", "zpeak_during_signed"),
    ("Signed peak z Δ (SD)", "PeakZ-Before-vs-During", "peakz", "zpeak_delta_signed"),
    ("Unsigned peak z, during (SD)", "PeakZ-Before-vs-During", "peakz",
     "zpeak_during_unsigned"),
    ("Signed peak-z ratio, during / size of before", "PeakZ-Before-vs-During", "peakz",
     "zpeak_ratio_signed"),
    ("Thresholded AUC, during (% × s)", "AUC-LogRatio-Before-vs-During", "logratio",
     "thr_auc_during"),
    ("Log ratio (model input)", "AUC-LogRatio-Before-vs-During", "logratio",
     "thr_auc_log_ratio"),
)
_PRE, _POST = "Pre-test", "Post-training"


def count_significant(summary: pd.DataFrame, alpha: float = 0.05) -> tuple[int, int]:
    """``(p < alpha, tests)`` over the per-odor pre-vs-post tests of a summary CSV.

    Each test sits on two rows (one per phase); the pooled group has none.
    """
    rows = summary.loc[(summary["phase"] == _PRE) & (summary["odor"] != "All odors")]
    p = rows["p_pre_vs_post"].to_numpy(dtype=float)
    p = p[np.isfinite(p)]
    return int((p < alpha).sum()), int(p.size)


def _fmt(v: float, digits: int) -> str:
    return "–" if not np.isfinite(v) else f"{v:.{digits}f}"


def _p(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    return "**<0.001**" if v < 0.001 else (f"**{v:.3f}**" if v < 0.05 else f"{v:.2f}")


def _read(out: Path, folder: str, name: str) -> pd.DataFrame | None:
    path = out / folder / name
    return pd.read_csv(path) if path.exists() else None


def write_findings(out: Path, dataset: str, label: str, flies: pd.DataFrame,
                   exits: dict[str, int], excluded: Sequence[str] = ()) -> Path:
    pre = phase_terms.pre_label(dataset, "Pre-test")
    post = phase_terms.post_label(dataset, "Post-training")
    tag = phase_terms.odor_tag(dataset)
    per_trial = _read(out, "AUC-Before-vs-During", "auc_per_trial.csv")
    trained = ""
    if per_trial is not None and per_trial["trained"].any():
        trained = str(per_trial.loc[per_trial["trained"], "odor"].iloc[0])
    n_flies = int(flies["flies"].sum())
    n_trials = 0 if per_trial is None else len(per_trial)

    lines = [f"# {dataset}, start {label}: trace AUC, peak z and what drives the model score",
             "",
             f"Built {dt.date.today().isoformat()} by `scripts/analysis/start_window_report.py` "
             "(every command and its output: `_build_log.txt`). Flies: every batch folder whose "
             f"first {pre.lower()} trial started in the window **{label}** (Pi clock, "
             "half-open), flagged flies removed. "
             f"**{n_flies} flies, {n_trials} odor trials.** {tag[:1].upper() + tag[1:]}: "
             f"{trained or 'not identified'}.",
             "", "## Flies", "", "| Folder | First pre-test | Flies | Flagged, dropped |",
             "|---|---|---|---|"]
    for _, r in flies.iterrows():
        lines.append(f"| {r['folder']} | {r['start']:%Y-%m-%d %H:%M} | {r['flies']} "
                     f"| {r['flagged_dropped']} |")
    if excluded:
        lines += ["", f"Excluded by hand: {', '.join(excluded)} (started in the window, "
                      "left out with `--exclude-folder`)."]

    failed = [k for k, v in exits.items() if v != 0]
    if failed:
        lines += ["", f"**Steps that failed (see `_build_log.txt`): {', '.join(failed)}.**"]

    summaries = {key: _read(out, folder, f"{key}_summary.csv")
                 for _, folder, key, _ in KEY_MEASURES}
    corrs = {key: _read(out, folder, f"{key}_vs_score_correlation.csv")
             for _, folder, key, _ in KEY_MEASURES}

    lines += ["", f"## 1. {pre} vs {post}: the {tag} ({trained})", "",
              f"| Measure | {pre} mean (median) | {post} mean (median) | Paired p |",
              "|---|---|---|---|"]
    for row_label, _folder, key, measure in KEY_MEASURES:
        s = summaries.get(key)
        if s is None:
            continue
        s = s.loc[(s["odor"] == trained) & (s["measure"] == measure)].set_index("phase")
        if s.empty:
            continue
        digits = 2 if "ratio" in measure else 1
        cells = [f"{_fmt(s.loc[ph, 'mean'], digits)} ({_fmt(s.loc[ph, 'median'], digits)})"
                 for ph in (_PRE, _POST)]
        lines.append(f"| {row_label} | {cells[0]} | {cells[1]} | "
                     f"{_p(float(s.loc[_PRE, 'p_pre_vs_post']))} |")
    n_sig = n_tests = 0
    for s in summaries.values():                  # one summary per metric family
        if s is not None:
            k, n = count_significant(s)
            n_sig, n_tests = n_sig + k, n_tests + n
    lines += ["", f"Wilcoxon signed-rank, {post.lower()} vs {pre.lower()} paired by fly, "
              f"uncorrected. Across every odor and measure, **{n_sig} of {n_tests}** "
              f"per-odor comparisons have p < 0.05; about {0.05 * n_tests:.0f} would by "
              "chance."]

    lines += ["", "## 2. How each measure tracks the model score (Spearman ρ, per trial)", "",
              f"| Measure | {pre} | {post} | Combined |", "|---|---|---|---|"]
    for row_label, _folder, key, measure in KEY_MEASURES:
        c = corrs.get(key)
        if c is None:
            continue
        c = c.loc[c["measure"] == measure].set_index("phase")["rho"]
        if c.empty:
            continue
        lines.append(f"| {row_label} | {_fmt(c.get('Pre-test', np.nan), 2)} | "
                     f"{_fmt(c.get('Post-training', np.nan), 2)} | "
                     f"{_fmt(c.get('Pre + post combined', np.nan), 2)} |")
    lines += ["", "The ρ p-values in the figures are descriptive only: trials are treated "
              "as independent, but each fly gives seven per phase."]

    drivers = _read(out, "Model-Feature-Drivers", "model_feature_drivers.csv")
    feats = _read(out, "Model-Feature-Drivers", "model_features_per_trial.csv")
    if drivers is not None:
        agree = ""
        if feats is not None:
            ok = feats["score"].notna()
            agree = (f" Rebuilt inputs reproduce the stored score on "
                     f"**{int((feats.loc[ok, 'recomputed_score'] == feats.loc[ok, 'score']).sum())}"
                     f" of {int(ok.sum())}** trials.")
        lines += ["", "## 3. What drives the model score", "", agree.strip(), "",
                  "| Input | Share of training gain | Share of attribution here | ρ with score |",
                  "|---|---|---|---|"]
        for _, r in drivers.head(6).iterrows():
            lines.append(f"| `{r['feature']}` | {r['gain_share']:.0%} | {r['shap_share']:.0%} "
                         f"| {_fmt(r['rho_all'], 2)} |")
        best = drivers.loc[drivers["rho_all"].idxmax()]
        lines += ["", f"Best single correlate of the score: `{best['feature']}` "
                  f"(ρ = {best['rho_all']:.2f}). Correlations with the model's own score are "
                  "partly circular; agreement with hand scores is the independent check."]

    lines += ["", "## Definitions", "",
              "- *During* = each trial's odor window (~30 s); *before* = the window of the "
              "same length just ahead of odor onset. Δ = during − before on the same trial.",
              "- **AUC** (no threshold): area of (trace − median of the before window). "
              "Signed counts dips negative; unsigned counts |deviation|. Raw-area ratio = "
              "area under the raw trace during / before (the raster CSVs' `auc_ratio`).",
              "- **Peak z**: z = (trace − before mean) / before SD. Signed = the largest |z| "
              "with its sign; unsigned = its size.",
              "- **Thresholded AUC**: the wide table's `AUC-Before`/`AUC-During`, area above "
              "the response threshold θ (the red line on the trace figures). Log ratio = "
              "log((During + ε)/(Before + ε)), ε stored on the model: the model's "
              "`AUC-During-Before-LogRatio` input.",
              "- Figure footnotes carry the full definitions; the 1pm-2pm set of 2026-10-26 "
              "has the long-form write-up."]
    path = out / "FINDINGS.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--start", required=True, help="window start, e.g. 13:00 or 1pm")
    p.add_argument("--end", required=True, help="window end (exclusive), e.g. 14:00 or 2pm")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--csv-dir", type=Path, default=CSV)
    p.add_argument("--flagged-flies-csv", type=Path, default=FLAGGED)
    p.add_argument("--exclude-folder", action="append", default=[],
                   help="leave this folder out by exact name (repeatable)")
    p.add_argument("--rate-test", choices=("fisher", "mcnemar"), default="fisher",
                   help="pre-vs-post response-rate test on the score figures; mcnemar "
                        "pairs each fly with itself, as the pipeline's figures do")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lo, hi = parse_clock(args.start), parse_clock(args.end)
    label = window_label(lo, hi)
    window = folders_in_window(start_table(args.data_root, args.dataset), lo, hi)
    if window.empty:
        print(f"[start-window] no {args.dataset} folder started in {label}")
        return NOTHING_TO_PLOT
    window = drop_folders(window, args.exclude_folder)
    folders = window["folder"].tolist()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[start-window] {args.dataset} {label}: {len(folders)} folders -> {out}")
    p = write_inputs(args.dataset, folders, out / "_inputs", csv_dir=args.csv_dir,
                     flagged=args.flagged_flies_csv)
    flies = fly_table(window, pd.read_parquet(p["testing"], columns=["fly", "fly_number"]),
                      args.dataset, args.flagged_flies_csv)
    if flies["flies"].sum() == 0:
        print("[start-window] every fly in the window is flagged; nothing to plot")
        return NOTHING_TO_PLOT
    with open(out / "_build_log.txt", "w") as log:
        log.write("$ " + " ".join([Path(sys.argv[0]).name, *map(str, argv or sys.argv[1:])])
                  + "\n")
        exits = run_figures(out, p, args.dataset, config=args.config,
                            label=f"start {label}", log=log, rate_test=args.rate_test,
                            flagged=args.flagged_flies_csv)
    findings = write_findings(out, args.dataset, label, flies, exits,
                              excluded=args.exclude_folder)
    failed = [k for k, v in exits.items() if v != 0]
    print(f"[start-window] {findings}" + (f"  FAILED: {failed}" if failed else ""))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
