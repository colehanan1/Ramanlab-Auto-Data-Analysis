#!/usr/bin/env python3
"""Recording-day-vs-recording-day figure set for a *-Sensitivity-* dataset.

Auto-detects the recording days present in a dataset (from its batch folder
names), picks the two most recent days that have at least one unflagged fly,
and builds the full figure set for each day plus their direct comparison --
the same layout as a hand-built ``<dataset>_<DayA>-vs-<DayB>/`` folder:

    <out-root>/<dataset>_<DayA>-vs-<DayB>/
        <DayA>/                  every standard sensitivity figure set
        <DayB>/                  every standard sensitivity figure set
        <DayA>-vs-<DayB>/        the direct comparison (starvation_pre_post_comparison.py)
        README.md

Each day's input tables are filtered copies of the pipeline's own wide/
prediction tables (flagged flies dropped, odor_pins "<Odor>-<conc>" labels
normalised), so nothing here changes the pipeline's own output.

Usage
-----
    # one dataset, auto-picks its two most recent days
    python3 scripts/analysis/day_vs_day_report.py --dataset IAA-Sensitivity-18-1

    # several datasets in one run; datasets with <2 usable days are skipped
    python3 scripts/analysis/day_vs_day_report.py \\
        --dataset IAA-Sensitivity-18-1 --dataset IAA-Sensitivity-6-1 \\
        --dataset IAA-Sensitivity-1-1 --dataset IAA-Sensitivity-12-1 \\
        --dataset IAA-Sensitivity-30-1 --dataset IAA-Sensitivity-24-1

    # force a specific pair instead of the two most recent
    python3 scripts/analysis/day_vs_day_report.py --dataset IAA-Sensitivity-18-1 \\
        --days october_07,october_08

    # publication score-bar figures too (PNG/PDF/SVG, --ymax auto-raised if a
    # bar + SEM would be clipped)
    python3 scripts/analysis/day_vs_day_report.py --dataset IAA-Sensitivity-24-1 --pubfig

    # BATCH vs batch instead of day vs day (a batch = october_11_batch_2 on all
    # rigs); --all compares every usable batch (up to 5) side by side
    python3 scripts/analysis/day_vs_day_report.py --dataset 3Oct-Sensitivity-1-0.1 \\
        --by batch --all --hours 0 --pubfig

    # pick exact units (days or batches) instead of the two most recent
    python3 scripts/analysis/day_vs_day_report.py --dataset 3Oct-Sensitivity-1-0.1 \\
        --by batch --units october_11_batch_1,october_11_batch_3

--hours overrides the starvation time read from the dataset name (the
*-Sensitivity-1-* datasets are 0 h). Dataset names match case-insensitively.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analysis.starvation_pre_post_comparison as spc  # noqa: E402

CONFIG = REPO_ROOT / "config" / "config_new.yaml"
CSV = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
FIGS = Path("/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures")
FLAGGED = CSV / "flagged-flys-truth.csv"
DEFAULT_OUT_ROOT = Path("/home/ramanlab/Documents/cole/Results/Figures")
PY = sys.executable

_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
          "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
          "december": 12}
_FOLDER_DATE_RE = re.compile(r"^([a-z]+)_(\d{2})")
_FOLDER_BATCH_RE = re.compile(r"^[a-z]+_\d{2}_batch_(\d+)")
MAX_UNITS = 5  # starvation_pre_post_comparison draws at most five groups
_DATASET_RE = re.compile(r"^[A-Za-z0-9]+-Sensitivity-(?P<hours>[\d.]+)-(?P<conc>[\d.]+)$")


# ── pure helpers ─────────────────────────────────────────────────────────


def folder_date(folder_name: str, *, year: int | None = None) -> dt.date:
    """``"october_08_batch_2_rig_2"`` -> ``date(2026, 10, 8)``.

    Batch folder names carry no year, so the pipeline's assumption applies
    here too: the current year, since these are current-year recordings.
    """
    m = _FOLDER_DATE_RE.match(folder_name.strip().lower())
    if not m or m.group(1) not in _MONTHS:
        raise ValueError(f"not a recording-day folder name: {folder_name!r}")
    return dt.date(year or dt.date.today().year, _MONTHS[m.group(1)], int(m.group(2)))


def folder_batch(folder_name: str) -> int:
    """``"october_11_batch_2_rig_3"`` -> ``2``."""
    m = _FOLDER_BATCH_RE.match(folder_name.strip().lower())
    if not m:
        raise ValueError(f"no batch number in folder name: {folder_name!r}")
    return int(m.group(1))


def day_prefix(date: dt.date) -> str:
    month = next(name for name, num in _MONTHS.items() if num == date.month)
    return f"{month}_{date.day:02d}"


def day_label(date: dt.date) -> str:
    return date.strftime("%b %-d") if sys.platform != "win32" else date.strftime("%b %#d")


def starvation_hours(dataset: str) -> tuple[float, str]:
    """``"IAA-Sensitivity-18-1"`` -> ``(18.0, "1")`` (hours, trained-odor conc)."""
    m = _DATASET_RE.match(dataset)
    if not m:
        raise ValueError(f"dataset name doesn't match <Odor>-Sensitivity-<h>-<conc>: {dataset!r}")
    return float(m["hours"]), m["conc"]


@dataclasses.dataclass(frozen=True)
class DayInfo:
    """One comparison unit: a recording day, or (``batch`` set) one batch of a day.

    A batch spans its rigs: ``october_11_batch_2``, ``_rig_2`` and ``_rig_3`` are
    one unit. Its prefix also matches ``october_11_batch_2X`` -- a day with ten or
    more batches would need explicit ``--units``.
    """
    date: dt.date
    n_flies: int
    batch: int | None = None

    @property
    def prefix(self) -> str:
        base = day_prefix(self.date)
        return base if self.batch is None else f"{base}_batch_{self.batch}"

    @property
    def label(self) -> str:
        base = day_label(self.date)
        return base if self.batch is None else f"{base} batch {self.batch}"

    @property
    def tag(self) -> str:
        base = self.date.strftime("%b%d")
        return base if self.batch is None else f"{base}b{self.batch}"

    @property
    def sort_key(self) -> tuple:
        return (self.date, self.batch or 0)


def usable_units(wide: pd.DataFrame, *, flagged_flies_csv: Path | str,
                 by: str = "day") -> list[DayInfo]:
    """One unit per day (``by="day"``) or per day+batch (``by="batch"``) with
    >=1 unflagged fly in *wide*, in time order."""
    if by not in ("day", "batch"):
        raise ValueError(f"by must be 'day' or 'batch': {by!r}")
    from fbpipe.config import canon_fly_number, load_flagged_fly_exclusions

    excluded = load_flagged_fly_exclusions(flagged_flies_csv) if Path(flagged_flies_csv).exists() else set()
    flies = wide[["dataset", "fly", "fly_number"]].drop_duplicates()
    if excluded:
        keys = [(str(d).strip(), str(f).strip(), canon_fly_number(n))
                for d, f, n in zip(flies["dataset"], flies["fly"], flies["fly_number"])]
        flies = flies.loc[[k not in excluded for k in keys]]

    counts: dict[tuple, int] = {}
    for fly in flies["fly"].astype(str):
        try:
            key = (folder_date(fly), folder_batch(fly) if by == "batch" else None)
        except ValueError:
            continue
        counts[key] = counts.get(key, 0) + 1
    units = (DayInfo(d, n, batch=b) for (d, b), n in counts.items())
    return sorted(units, key=lambda u: u.sort_key)


def usable_days(wide: pd.DataFrame, *, flagged_flies_csv: Path | str) -> list[DayInfo]:
    """One :class:`DayInfo` per calendar day with >=1 unflagged fly in *wide*."""
    return usable_units(wide, flagged_flies_csv=flagged_flies_csv, by="day")




def pick_days(days: Sequence[DayInfo], *, explicit: tuple[str, str] | None = None
             ) -> tuple[DayInfo, DayInfo] | None:
    """The two most recent usable days, or an explicit ``(prefix_a, prefix_b)`` pair."""
    if explicit:
        by_prefix = {d.prefix: d for d in days}
        missing = [p for p in explicit if p not in by_prefix]
        if missing:
            raise ValueError(f"no usable flies for {missing} (available: {sorted(by_prefix)})")
        a, b = by_prefix[explicit[0]], by_prefix[explicit[1]]
        return (a, b) if a.date <= b.date else (b, a)
    if len(days) < 2:
        return None
    ordered = sorted(days, key=lambda d: d.date)
    return ordered[-2], ordered[-1]


def pick_units(units: Sequence[DayInfo], *, explicit: Sequence[str] | None = None,
               all_units: bool = False) -> list[DayInfo] | None:
    """Units to compare, in time order: an explicit prefix list, every unit
    (``all_units``, at most :data:`MAX_UNITS`), or the two most recent. None when
    fewer than two are usable."""
    by_prefix = {u.prefix: u for u in units}
    if explicit:
        missing = [p for p in explicit if p not in by_prefix]
        if missing:
            raise ValueError(f"no usable flies for {missing} (available: {sorted(by_prefix)})")
        chosen = [by_prefix[p] for p in explicit]
    elif all_units:
        chosen = list(units)
    else:
        chosen = sorted(units, key=lambda u: u.sort_key)[-2:]
    if len(chosen) < 2:
        return None
    if len(chosen) > MAX_UNITS:
        raise ValueError(f"{len(chosen)} units to compare; at most {MAX_UNITS} -- "
                         f"pass --units to choose ({sorted(by_prefix)})")
    return sorted(chosen, key=lambda u: u.sort_key)


def out_dir_name(*units: DayInfo) -> str:
    return "-vs-".join(u.tag for u in units)


def _same_day_batches(units: Sequence[DayInfo]) -> bool:
    return all(u.batch is not None for u in units) and len({u.date for u in units}) == 1


def display_labels(units: Sequence[DayInfo]) -> list[str]:
    """Group names drawn under every bar pair, so kept short: "Batch 2" when
    every unit is a batch of the same day, "Oct 11 b2" across days."""
    if _same_day_batches(units):
        return [f"Batch {u.batch}" for u in units]
    if all(u.batch is not None for u in units):
        return [f"{day_label(u.date)} b{u.batch}" for u in units]
    return [u.label for u in units]


def comparison_text(*units: DayInfo, hours: float | None) -> str:
    text = " vs ".join(display_labels(units))
    if _same_day_batches(units):
        text = f"{day_label(units[0].date)}: {text}"
    if hours is None:
        return text
    return f"{text} ({'both' if len(units) == 2 else 'all'} {hours:g} h starved)"


def resolve_dataset(name: str, known: Sequence[str]) -> str:
    """Exact or case-insensitive match against *known*; otherwise stop with suggestions."""
    import difflib

    if name in known:
        return name
    folded = {k.casefold(): k for k in known}
    if name.casefold() in folded:
        return folded[name.casefold()]
    close = difflib.get_close_matches(name.casefold(), list(folded), n=3, cutoff=0.6)
    hint = ", ".join(folded[c] for c in close) or "none close"
    raise SystemExit(f"unknown dataset {name!r}; did you mean: {hint}")


def trained_odor(dataset: str) -> str:
    """The CS+ display name for a dataset, e.g. 3Oct-* -> '3-Octanol'."""
    from scripts.analysis.envelope_visuals import _trained_label

    return _trained_label(dataset) or "Isoamyl Acetate"


# ── driver (shells out to the already-tested analysis scripts) ────────────


def _run(cmd: list, log) -> int:
    cmd = [str(c) for c in cmd]
    print("  $", " ".join(cmd)[:200])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    log.write(f"\n$ {' '.join(cmd)}\n[exit {r.returncode}]\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}\n")
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        print(f"  !! exit {r.returncode}: {tail}")
    return r.returncode


def _model_path() -> Path:
    cfg = yaml.safe_load(CONFIG.read_text())
    return Path(cfg["analysis"]["pretest_score_bars"].get("model_path")
               or cfg["reaction_prediction"]["model_path"])


def _write_inputs(group: spc.Group, inputs: Path) -> dict[str, Path]:
    inputs.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in (("testing", "all_envelope_rows_wide_combined_base.parquet"),
                      ("pretest", "all_envelope_rows_wide_combined_base_pretest.parquet"),
                      ("training", "all_envelope_rows_wide_combined_base_training.parquet")):
        df = spc.drop_flagged(spc.subset_and_relabel(pd.read_parquet(CSV / name), [group]), FLAGGED)
        paths[key] = inputs / f"wide_{key}.parquet"
        df.to_parquet(paths[key], index=False)
        print(f"  {key:8s} wide rows: {len(df)}  flies: {df.groupby(['fly','fly_number']).ngroups}")
    for key, name in (("predictions", "model_predictions.csv"),
                      ("training_predictions", "model_predictions_training.csv")):
        df = spc.drop_flagged(spc.subset_and_relabel(pd.read_csv(CSV / name), [group]), FLAGGED)
        paths[key] = inputs / f"{key}.csv"
        df.to_csv(paths[key], index=False)
        print(f"  {key:20s} rows: {len(df)}")
    return paths


def _build_day(out: Path, group: spc.Group, log, *, pubfig: bool) -> None:
    print(f"\n=== {out.name}: {group}")
    p = _write_inputs(group, out / "_inputs")
    ds, model = group.dataset, _model_path()

    _run([PY, "scripts/analysis/pretest_vs_test_comparison.py", "--predictions-csv", p["predictions"],
          "--out-root", out / "Pre-vs-Post-Sensitivity", "--binary-threshold", 2,
          "--rate-test", "fisher", "--flagged-flies-csv", FLAGGED], log)
    if pubfig:
        summary = out / "Pre-vs-Post-Sensitivity" / ds / f"{ds}_summary.csv"
        if summary.exists():
            s = pd.read_csv(summary)
            ymax = max(5.0, (s["mean_post"] + s["sem_post"]).max() + 0.2,
                      (s["mean_pre"] + s["sem_pre"]).max() + 0.2)
            _run([PY, "scripts/analysis/pubfig_pre_vs_post_score_bars.py", "--summary-csv", summary,
                  "--ymax", ymax], log)

    _run([PY, "scripts/analysis/dataset_mean_traces_tvc.py", "--wide-csv", p["testing"],
          "--pretest-wide-csv", p["pretest"], "--train-dataset", ds,
          "--out-dir", out / "Mean-Traces_Pre-Test_vs_Post-Training", "--fps", 40, "--odor-on-s", 30,
          "--flagged-flies-csv", FLAGGED, "--config", CONFIG], log)

    # Pre-test-baselined trial score bars: borrow the matching PRETEST_COHORTS
    # spec (same odor family) and re-key it to this dataset.
    hours, conc = starvation_hours(ds)
    code = (
        "import sys, dataclasses; from pathlib import Path; "
        f"sys.path.insert(0, {str(REPO_ROOT)!r}); sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
        "import scripts.analysis.naive_vs_trial_score_bars as n; "
        f"base = next((s for s in n.PRETEST_COHORTS.values() "
        f"if s.dataset.split('-Sensitivity-')[0] == {ds!r}.split('-Sensitivity-')[0]), None); "
        f"spec = dataclasses.replace(base, dataset={ds!r}) if base else None; "
        f"n.run([{ds!r}], predictions_csv=Path({str(p['predictions'])!r}), "
        f"training_wide=Path({str(p['training'])!r}), "
        f"training_predictions_csv=Path({str(p['training_predictions'])!r}), "
        f"model_path=Path({str(model)!r}), out_dir=Path({str(out / 'Pre-Test-vs-Trial-Score-Bars')!r}), "
        f"registry={{{ds!r}: spec}} if spec else {{}})"
    )
    _run([PY, "-c", code], log)

    for mode, folder in (("binary", "Binarized-PER-Rasters"), ("graded", "Graded-PER-Rasters")):
        _run([PY, "scripts/analysis/binarized_per_rasters.py", "--training-wide-csv", p["training"],
              "--wide-csv", p["testing"], "--pretest-wide-csv", p["pretest"],
              "--predictions-csv", p["predictions"], "--config", CONFIG, "--dataset", ds,
              "--out-dir", out / folder, "--mode", mode, "--sort-by", "ratio"], log)

    _run([PY, "scripts/analysis/training_auc_vs_control_response.py",
          "--training-wide-csv", p["training"], "--testing-wide-csv", p["testing"],
          "--predictions-csv", p["predictions"], "--dataset", ds,
          "--out-dir", out / "Training-AUC-vs-Testing-Response"], log)

    _run([PY, "scripts/analysis/reaction_matrix_from_spreadsheet.py", "--csv-path", p["predictions"],
          "--out-dir", out / "Matrix-PER-Reactions-Model", "--latency-sec", 2.15,
          "--after-window-sec", 30.0, "--row-gap", 0.6, "--height-per-gap-in", 3.0,
          "--bottom-shift-in", 0.5, "--non-reactive-threshold", 20,
          "--flagged-flies-csv", FLAGGED, "--trial-order", "trained-first", "--overwrite",
          "--config", CONFIG, "--protocol", "v2"], log)

    from fbpipe.config import load_flagged_fly_exclusions
    excluded = load_flagged_fly_exclusions(FLAGGED)
    for kind in ("Raw-Pre-Testing-PER-Traces", "Raw-Training-PER-Traces", "Raw-Testing-PER-Traces"):
        src = FIGS / kind / ds
        n = 0
        if src.exists():
            for f in sorted(src.glob(f"{group.fly_prefix}*")):
                if any(f.name.startswith(f"{fly}_fly{num}_") for ds_, fly, num in excluded if ds_ == ds):
                    continue
                dst = out / "Raw-PER-Traces-per-fly" / kind / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
        print(f"  copied {n} {kind} figures")


def _build_comparison(out: Path, units: Sequence[DayInfo], dataset: str, *,
                      hours: float | None, by: str, log) -> None:
    print(f"\n=== comparison → {out.name}")
    cmd = [PY, "scripts/analysis/starvation_pre_post_comparison.py",
           "--predictions-csv", CSV / "model_predictions.csv",
           "--pretest-wide-csv", CSV / "all_envelope_rows_wide_combined_base_pretest.parquet",
           "--testing-wide-csv", CSV / "all_envelope_rows_wide_combined_base.parquet"]
    for label, u in zip(display_labels(units), units):
        cmd += ["--group", f"{label}={dataset}:{u.prefix}"]
    cmd += ["--out-dir", out, "--flagged-flies-csv", FLAGGED, "--binary-threshold", 2,
            "--trained-odor", trained_odor(dataset), "--title", dataset, "--file-tag", by,
            "--comparison", comparison_text(*units, hours=hours)]
    _run(cmd, log)


_README = """\
# {dataset} — {title}

Hand-run figure set (not produced by the pipeline), built {date} by
`scripts/analysis/day_vs_day_report.py` (repo tool, not a one-off script).
The exact command is the first line of `_build_log.txt`.

## {unit_word}s

| {unit_word} | Flies (flagged removed) |
|---|---|
{rows}

{hours_line}Trained odor (CS+): {cs}. Scores: current model
(`{model}`). Tables as of the pipeline run this was built against.

## Folders

- {unit_dirs}: every standard sensitivity figure set for that {unit_lower}'s flies
  (Pre-vs-Post-Sensitivity, mean traces, pre-test-baselined trial score bars,
  binarized/graded rasters, Training-AUC vs testing, reaction matrix, per-fly raw traces).
- `{out_dir}/`: the direct comparison (score: Wilcoxon paired within fly / Mann-Whitney
  between {unit_lower}s; response %: Fisher's exact throughout). CSVs: `summary_by_group`,
  `stats_between_groups`, `paired_scores_per_fly`{omnibus}.

Colors: blue = pre-test, orange = post-training; lighter = earlier {unit_lower}
({order}).
n counts flies; p-values are uncorrected. Trial labels since 2026-10-01 carry the vial
concentration (`<Odor>-<conc>`); inputs are filtered copies with that suffix stripped so
odor names and the CS+ resolve.
"""


def run_dataset(dataset: str, *, out_root: Path, explicit: Sequence[str] | None,
                pubfig: bool, by: str = "day", all_units: bool = False,
                hours: float | None = None) -> Path | None:
    name_hours, _conc = starvation_hours(dataset)
    hours = name_hours if hours is None else hours
    wide = pd.concat([
        pd.read_parquet(CSV / "all_envelope_rows_wide_combined_base_pretest.parquet",
                        columns=["dataset", "fly", "fly_number"]),
        pd.read_parquet(CSV / "all_envelope_rows_wide_combined_base.parquet",
                        columns=["dataset", "fly", "fly_number"]),
    ])
    wide = wide[wide["dataset"].astype(str).str.strip() == dataset]
    units = usable_units(wide, flagged_flies_csv=FLAGGED, by=by)
    print(f"\n########## {dataset} by {by}: {[(u.prefix, u.n_flies) for u in units]}")
    chosen = pick_units(units, explicit=explicit, all_units=all_units)
    if chosen is None:
        print(f"  skipped: fewer than 2 {by}s with unflagged flies "
              f"({len(units)} found: {[u.prefix for u in units]})")
        return None
    out = out_root / f"{dataset}_{out_dir_name(*chosen)}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "_build_log.txt", "w") as log:
        log.write(f"$ day_vs_day_report.py --dataset {dataset} --by {by}"
                  + (f" --units {','.join(explicit)}" if explicit else "")
                  + (" --all" if all_units else "") + f" --hours {hours:g}"
                  + (" --pubfig" if pubfig else "") + "\n")
        for u in chosen:
            _build_day(out / u.tag, spc.Group(u.label, dataset, u.prefix), log, pubfig=pubfig)
        _build_comparison(out / out_dir_name(*chosen), chosen, dataset, hours=hours, by=by, log=log)
    unit_word = "Day" if by == "day" else "Batch"
    (out / "README.md").write_text(_README.format(
        dataset=dataset, title=" vs ".join(u.label for u in chosen),
        date=dt.date.today().isoformat(), unit_word=unit_word, unit_lower=unit_word.lower(),
        rows="\n".join(f"| {u.label} | {u.n_flies} |" for u in chosen),
        hours_line=f"{hours:g} h starved. ", cs=trained_odor(dataset), model=_model_path(),
        unit_dirs=", ".join(f"`{u.tag}/`" for u in chosen), out_dir=out_dir_name(*chosen),
        omnibus=", `stats_omnibus` (all groups at once)" if len(chosen) > 2 else "",
        order=" → ".join(u.label for u in chosen)))
    text = (out / "_build_log.txt").read_text().splitlines()
    ok = sum(1 for line in text if "[exit 0]" in line)
    bad = sum(1 for line in text if "[exit " in line) - ok
    print(f"  -> {out}  ({ok} ok, {bad} failed)")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", action="append", required=True,
                    help="A *-Sensitivity-* dataset name (case-insensitive); repeatable.")
    ap.add_argument("--by", choices=("day", "batch"), default="day",
                    help="Compare recording days (default) or batches (a batch = "
                         "<month>_<dd>_batch_<n> across its rigs).")
    ap.add_argument("--units", "--days", dest="units", default=None,
                    help="Comma-separated prefixes to compare, e.g. october_07,october_08 or "
                         "october_11_batch_1,october_11_batch_3. Applies to every --dataset in "
                         "this call. Default: the two most recent usable units.")
    ap.add_argument("--all", dest="all_units", action="store_true",
                    help=f"Compare every usable unit side by side (at most {MAX_UNITS}).")
    ap.add_argument("--hours", type=float, default=None,
                    help="Starvation hours for titles/README; default read from the dataset "
                         "name (the *-Sensitivity-1-* datasets are really 0 h).")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--pubfig", action="store_true",
                    help="Also render the publication score-bar figure for each unit.")
    args = ap.parse_args(argv)
    explicit = [u.strip() for u in args.units.split(",") if u.strip()] if args.units else None
    if explicit is not None and len(explicit) < 2:
        raise SystemExit("--units needs at least two comma-separated prefixes")

    known = sorted(set(pd.read_parquet(CSV / "all_envelope_rows_wide_combined_base_pretest.parquet",
                                       columns=["dataset"])["dataset"].astype(str).str.strip()))
    built, skipped = [], []
    for name in args.dataset:
        dataset = resolve_dataset(name, known)
        out = run_dataset(dataset, out_root=args.out_root, explicit=explicit, pubfig=args.pubfig,
                          by=args.by, all_units=args.all_units, hours=args.hours)
        (built if out else skipped).append(dataset)

    print(f"\n{len(built)} built, {len(skipped)} skipped" + (f": {skipped}" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
