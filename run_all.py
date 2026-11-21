#!/usr/bin/env python3
"""Single entry point of the PER analysis (``src/per_analysis``).

    python3 run_all.py                      # everything, into ./results
    python3 run_all.py --only exp1 --only cross
    python3 run_all.py --results /tmp/res --log-level DEBUG

Order:

1. ``io.load_trials()`` (cached in ``results/01_data_qc``), then ``qc.run_qc`` writes
   ``01_data_qc`` (QC report, column mapping, protocol facts, b2b counts check).
2. Each module's ``run(trials, results, tables=results/tables)``: exp1, exp2, exp3,
   cross, power, daynight (``--only`` picks some).
3. The sensitivity run, always: every QC exclusion rule on, exp1/exp2/exp3/cross re-run
   into ``results/07_sensitivity_with_exclusions/`` (same folders, own ``tables/``).
   ``--only`` limits it too.
4. ``report.write_readmes`` / ``write_summary`` / ``write_index``.

A module that raises does not stop the others: its traceback is logged and a finding
with ``analysis="failed"`` lands in SUMMARY.md. Exit code 0 only when every requested
module succeeded in every run (and QC ran).

``--report-only``: no analysis; rewrite READMEs, SUMMARY.md and index.html from the last
run's ``findings.json`` + ``run_manifest.json`` (without a manifest: module status from
``run_all.log``, QC summary recomputed).

``--exclude-qc``: the MAIN run drops QC-flagged flies too (every rule on). Without it the
main run drops only what the ``config.EXCLUDE_*`` switches turn on (all off by default).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import random
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from per_analysis import config as C  # noqa: E402
from per_analysis import io, qc, report  # noqa: E402

log = logging.getLogger("run_all")

MODULES = ("exp1", "exp2", "exp3", "cross", "power", "daynight")
SENSITIVITY_MODULES = report.SENSITIVITY_MODULES
ALL_RULES = {k: True for k in qc.EXCLUSION_RULES}
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


# ── setup ─────────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", action="append", choices=MODULES, metavar="MODULE",
                   help=f"run only this module (repeatable): {', '.join(MODULES)}. "
                        "Limits the main and the exclusion run.")
    p.add_argument("--exclude-qc", action="store_true",
                   help="also drop QC-flagged flies (every rule) from the MAIN run; the "
                        "07 exclusion run happens either way")
    p.add_argument("--results", type=Path, default=C.RESULTS_DIR,
                   help="results root (default: ./results at the repo root)")
    p.add_argument("--report-only", action="store_true",
                   help="run no analysis: rewrite READMEs, SUMMARY.md and index.html from "
                        "findings.json and run_manifest.json of the last run")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="console log level; run_all.log gets INFO or finer")
    return p.parse_args(argv)


def setup_logging(results_dir: Path, level: str, mode: str = "w") -> list[logging.Handler]:
    """Console at ``level`` + ``results/run_all.log`` (fresh each run; appended to by
    ``--report-only``) at INFO or finer.
    Returns the handlers added (removed again by :func:`teardown_logging`)."""
    lvl = logging.getLevelName(level)
    file_lvl = min(lvl, logging.INFO)
    fmt = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(lvl)
    console.setFormatter(fmt)
    fileh = logging.FileHandler(results_dir / "run_all.log", mode=mode, encoding="utf-8")
    fileh.setLevel(file_lvl)
    fileh.setFormatter(fmt)
    root = logging.getLogger()
    root._run_all_prev_level = root.level          # type: ignore[attr-defined]
    root.setLevel(min(lvl, file_lvl))
    root.addHandler(console)
    root.addHandler(fileh)
    logging.captureWarnings(True)
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return [console, fileh]


def teardown_logging(handlers: list[logging.Handler]) -> None:
    root = logging.getLogger()
    for h in handlers:
        root.removeHandler(h)
        h.close()
    logging.captureWarnings(False)
    root.setLevel(getattr(root, "_run_all_prev_level", logging.WARNING))


def seed_everything(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ── inputs and QC ─────────────────────────────────────────────────────────────
def load_inputs(cfg: C.AnalysisConfig) -> dict:
    """The tidy table plus the QC inputs beside it (raw-file listing, protocol facts,
    the b2b counts check). Only the tidy table is required."""
    trials = io.load_trials(cfg)
    out = dict(trials=trials, raw=None, facts=None, b2b=None)
    for key, fn in (("raw", io.load_raw_trials), ("facts", io.load_protocol_facts)):
        try:
            out[key] = fn(cfg)
        except Exception:
            log.exception("step=load_inputs input=%s status=failed (QC runs without it)", key)
    try:
        b2b, ok = io.b2b_counts_check(trials)
        b2b.to_csv(cfg.qc_dir / "b2b_counts_check.csv", index=False)
        out["b2b"], out["b2b_ok"] = b2b, ok
    except Exception:
        log.exception("step=load_inputs input=b2b_counts_check status=failed")
    return out


def write_column_mapping(qc_dir: Path) -> Path:
    path = qc_dir / "column_mapping.yaml"
    path.write_text(yaml.safe_dump(io.column_mapping(), sort_keys=False, allow_unicode=True,
                                   width=100))
    return path


def exclusion_counts(per_fly: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Flies each rule flags, the union, and one row per excluded fly with its rules."""
    n = len(per_fly)
    counts, rules_of = {}, {}
    for key, (col, _switch) in qc.EXCLUSION_RULES.items():
        ids = per_fly.loc[per_fly[col].astype(bool), "fly_id"]
        counts[key] = int(len(ids))
        for fid in ids:
            rules_of.setdefault(fid, []).append(key)
        log.info("exclusion rule=%s flies=%d of=%d", key, len(ids), n)
    counts["union"] = len(rules_of)
    counts["kept"] = n - len(rules_of)
    log.info("exclusion rule=ALL flies=%d of=%d kept=%d", len(rules_of), n, counts["kept"])
    ident = [c for c in ("fly_id", "experiment", "cohort", "batch", "window") if c in per_fly]
    table = per_fly[per_fly["fly_id"].isin(list(rules_of))][ident].copy()
    table["rules"] = table["fly_id"].map(lambda f: "; ".join(rules_of[f]))
    if "cohort" in table:
        for cohort, g in table.groupby("cohort"):
            log.info("exclusion cohort=%s flies=%d", cohort, len(g))
    return counts, table


# ── modules ───────────────────────────────────────────────────────────────────
def load_module(name: str):
    """Import ``per_analysis.<name>`` lazily (it may not exist yet)."""
    return importlib.import_module(f"per_analysis.{name}")


def _where(exc: BaseException) -> str:
    """' at file.py:LINE in func' of the deepest traceback frame in this repo (run_all
    itself excluded); '' when there is none."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if not f.filename.startswith("<")]
    ours = [f for f in frames if Path(f.filename).resolve().is_relative_to(REPO)
            and Path(f.filename).resolve() != Path(__file__).resolve()]
    f = (ours or [None])[-1]
    return f" at {Path(f.filename).name}:{f.lineno} in {f.name}" if f else ""


def _failed(name: str, run: str, exc: BaseException, stage: str) -> dict:
    where = _where(exc)
    verb = "could not be imported" if stage == "import" else "raised"
    text = f"`{name}.run` {verb}: {type(exc).__name__}: {exc}{where}"
    if stage == "import":
        text = f"`per_analysis.{name}` {verb}: {type(exc).__name__}: {exc}{where}"
    return dict(section=report.MODULE_DIRS.get(name, name), analysis="failed", primary=False,
                text=text, numbers={}, module=name, run=run,
                error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def _reset_plot_state() -> None:
    try:
        from per_analysis import plots
        plots.reset_ylims()
    except Exception:                       # pragma: no cover - plots is foundation
        log.debug("plots.reset_ylims unavailable", exc_info=True)


def run_module(name: str, trials: pd.DataFrame, out: Path, tables: Path, *,
               run: str) -> tuple[list[dict], bool, float]:
    """-> (findings, ok, seconds). Never raises (except KeyboardInterrupt)."""
    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    tables.mkdir(parents=True, exist_ok=True)
    _reset_plot_state()                     # each run registers its own shared y-limits
    seed_everything()
    log.info("module=%s run=%s status=start flies=%d trials=%d out=%s", name, run,
             trials["fly_id"].nunique(), len(trials), out)
    try:
        mod = load_module(name)
    except (Exception, SystemExit) as exc:
        log.exception("module=%s run=%s status=failed stage=import", name, run)
        return [_failed(name, run, exc, "import")], False, time.perf_counter() - t0
    try:
        result = mod.run(trials, out, tables=tables)
    except (Exception, SystemExit) as exc:
        secs = time.perf_counter() - t0
        log.exception("module=%s run=%s status=failed stage=run seconds=%.1f", name, run, secs)
        return [_failed(name, run, exc, "run")], False, secs
    finally:
        plt.close("all")
    secs = time.perf_counter() - t0
    if result is None:
        log.warning("module=%s run=%s returned None, expected list[dict]", name, run)
        result = []
    if not isinstance(result, (list, tuple)):
        exc = TypeError(f"run() returned {type(result).__name__}, expected list[dict]")
        log.error("module=%s run=%s status=failed stage=findings: %s", name, run, exc)
        return [_failed(name, run, exc, "run")], False, secs
    findings = report.normalize_findings(result, module=name, run=run)
    log.info("module=%s run=%s status=ok findings=%d primary=%d skipped=%d seconds=%.1f",
             name, run, len(findings), sum(f["primary"] for f in findings),
             sum(f["analysis"] == "skipped" for f in findings), secs)
    return findings, True, secs


# ── main ──────────────────────────────────────────────────────────────────────
MANIFEST = "run_manifest.json"
FINDINGS = "findings.json"
_STATUS_LINE = re.compile(r"module=(\S+) run=(\S+) status=(ok|failed)\b(?:.*?seconds=([\d.]+))?")
_DONE_LINE = re.compile(r"run_all done .*?seconds=([\d.]+)")


def _write_json(path: Path, obj) -> None:
    try:
        path.write_text(json.dumps(obj, indent=1, default=str))
    except Exception:
        log.exception("could not write %s", path)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()
    results = Path(args.results).expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)
    handlers = setup_logging(results, args.log_level, mode="a" if args.report_only else "w")
    try:
        if args.report_only:
            return _report_only(args, results)
        return _main(args, results, t_start)
    finally:
        teardown_logging(handlers)


def qc_step(trials: pd.DataFrame, inputs: dict, cfg: C.AnalysisConfig, *,
            write: bool = True) -> tuple[pd.DataFrame, dict]:
    """run_qc (+ its 01_data_qc outputs when ``write``) -> (per-fly flags, QC summary)."""
    qc_dir = Path(cfg.qc_dir)
    if write:
        write_column_mapping(qc_dir)
    per_fly, md = qc.run_qc(trials, inputs.get("raw"), threshold=cfg.threshold,
                            facts=inputs.get("facts"), b2b=inputs.get("b2b"))
    if write:
        qc.write_qc_outputs(per_fly, md, qc_dir)
    summary = dict(threshold=cfg.threshold, n_trials=len(trials),
                   n_flies=int(trials["fly_id"].nunique()),
                   n_cohorts=int(trials["cohort"].nunique()),
                   flags={col: int(per_fly[col].astype(bool).sum())
                          for col, _sw in qc.EXCLUSION_RULES.values()})
    checks = []
    facts = inputs.get("facts")
    if facts is not None and len(facts):
        bad = facts[~facts["ok"].astype(bool)]
        checks.append(f"Protocol facts: {len(facts) - len(bad)} of {len(facts)} as expected"
                      + (" (changed: " + "; ".join(bad["fact"].astype(str)) + ")"
                         if len(bad) else "")
                      + f" (`{report.QC_DIR}/protocol_facts.csv`).")
    if "b2b_ok" in inputs:
        checks.append("Pair-trial counts vs the b2b step: "
                      + ("agree" if inputs["b2b_ok"] else "MISMATCH")
                      + f" (`{report.QC_DIR}/b2b_counts_check.csv`).")
    summary["checks"] = checks
    summary["status"] = "ok"
    log.info("step=qc status=ok flies=%d flags=%s", len(per_fly), summary["flags"])
    return per_fly, summary


def main_run_trials(trials: pd.DataFrame, per_fly: pd.DataFrame | None,
                    exclude_qc: bool) -> tuple[pd.DataFrame, str]:
    """The main run's trials and a sentence saying what was excluded from them."""
    if per_fly is None:
        if exclude_qc:
            log.error("--exclude-qc requested but QC failed: the main run keeps every fly")
            return trials, "none (--exclude-qc requested, but QC failed)"
        return trials, "none"
    n = trials["fly_id"].nunique()
    if exclude_qc:
        kept = qc.apply_exclusions(trials, per_fly, ALL_RULES)
        return kept, f"every QC rule (--exclude-qc): {n - kept['fly_id'].nunique()} flies dropped"
    kept = qc.apply_exclusions(trials, per_fly)                  # config switches
    on = [k for k, (_c, sw) in qc.EXCLUSION_RULES.items() if getattr(C, sw)]
    if not on:
        return kept, "none"
    return kept, (f"config switches {', '.join(on)}: {n - kept['fly_id'].nunique()} "
                  "flies dropped")


def _main(args: argparse.Namespace, results: Path, t_start: float) -> int:
    selected = [m for m in MODULES if not args.only or m in args.only]
    sens = [m for m in SENSITIVITY_MODULES if m in selected]
    cfg = dataclasses.replace(C.DEFAULT_CONFIG, results_dir=results)
    tables = results / report.TABLES_DIR
    s07 = results / report.SENSITIVITY_DIR
    log.info("run_all start results=%s modules=%s exclusion_run=%s exclude_qc=%s "
             "threshold=%s windows=%s seed=%d", results, ",".join(selected), ",".join(sens),
             args.exclude_qc, cfg.threshold,
             ",".join(f"{w.name}:{w.start:%H:%M}-{w.end:%H:%M}" for w in cfg.windows), C.SEED)
    report.create_tree(results)
    findings: list[dict] = []
    status = {m: {"main": "not run", "exclusions": None, "seconds": 0.0} for m in selected}
    for m in sens:
        status[m]["exclusions"] = "not run"
    ok_all = True
    qc_summary: dict = {"threshold": cfg.threshold, "main_exclusions": "none"}

    # 1. data
    t0 = time.perf_counter()
    try:
        inputs = load_inputs(cfg)
        trials = inputs["trials"]
        log.info("step=load_trials status=ok rows=%d flies=%d cohorts=%d seconds=%.1f",
                 len(trials), trials["fly_id"].nunique(), trials["cohort"].nunique(),
                 time.perf_counter() - t0)
    except Exception as exc:
        log.exception("step=load_trials status=failed: nothing can run")
        findings.append(dict(_failed("io.load_trials", "main", exc, "run"),
                             section=report.QC_DIR))
        _finish(results, findings, qc_summary, status, time.perf_counter() - t_start)
        return 2
    qc_summary.update(n_trials=len(trials), n_flies=int(trials["fly_id"].nunique()),
                      n_cohorts=int(trials["cohort"].nunique()))

    # 2. QC
    per_fly = None
    try:
        per_fly, summary = qc_step(trials, inputs, cfg)
        qc_summary.update(summary)
    except Exception as exc:
        log.exception("step=qc status=failed")
        ok_all = False
        qc_summary["status"] = f"failed: {type(exc).__name__}: {exc}"
        findings.append(dict(_failed("qc", "main", exc, "run"), section=report.QC_DIR))

    # 3. main run
    main_trials, qc_summary["main_exclusions"] = main_run_trials(trials, per_fly,
                                                                 args.exclude_qc)
    for name in selected:
        f, ok, secs = run_module(name, main_trials, results, tables, run="main")
        findings += f
        ok_all &= ok
        status[name]["main"] = "ok" if ok else "FAILED"
        status[name]["seconds"] += secs

    # 4. sensitivity run with every exclusion on
    if sens:
        if per_fly is None:
            ok_all = False
            for name in sens:
                status[name]["exclusions"] = "FAILED"
                findings.append(dict(section=report.SENSITIVITY_DIR, analysis="failed",
                                     primary=False, module=name, run="exclusions", numbers={},
                                     text=f"`{name}` not re-run with exclusions: QC failed, "
                                          "so no fly is flagged."))
        else:
            counts, excluded = exclusion_counts(per_fly)
            qc_summary["excluded"] = counts
            s07.mkdir(parents=True, exist_ok=True)
            excluded.to_csv(s07 / "excluded_flies.csv", index=False)
            excl_trials = qc.apply_exclusions(trials, per_fly, ALL_RULES)
            for name in sens:
                f, ok, secs = run_module(name, excl_trials, s07, s07 / report.TABLES_DIR,
                                         run="exclusions")
                findings += f
                ok_all &= ok
                status[name]["exclusions"] = "ok" if ok else "FAILED"
                status[name]["seconds"] += secs

    ok_all &= _finish(results, findings, qc_summary, status, time.perf_counter() - t_start)
    failed = [f"{f.get('module')}({f.get('run')})" for f in findings
              if f.get("analysis") == "failed"]
    log.info("run_all done status=%s seconds=%.1f figures=%d tables=%d failed=%s",
             "ok" if ok_all else "FAILED", time.perf_counter() - t_start,
             sum(1 for _ in results.rglob("*.png")),
             sum(1 for p in tables.rglob("*") if p.is_file() and p.name != "README.md"),
             ",".join(failed) or "none")
    return 0 if ok_all else 1


def status_from_log(path: Path) -> tuple[dict, float | None]:
    """Module status and total seconds from run_all.log's key=value lines (the last
    full run's lines win)."""
    status: dict = {}
    total = None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return status, total
    for line in lines:
        m = _STATUS_LINE.search(line)
        if m:
            name, run, st, secs = m.groups()
            s = status.setdefault(name, {"main": "not run", "exclusions": None, "seconds": 0.0})
            if run == "main" and s["main"] != "not run":          # a newer run starts
                s.update(exclusions=None, seconds=0.0)
            s[run] = "ok" if st == "ok" else "FAILED"
            s["seconds"] += float(secs or 0.0)
        d = _DONE_LINE.search(line)
        if d:
            total = float(d.group(1))
    order = [m for m in MODULES if m in status] + [m for m in status if m not in MODULES]
    return {m: status[m] for m in order}, total


def _report_only(args: argparse.Namespace, results: Path) -> int:
    """Rewrite READMEs, SUMMARY.md and index.html from findings.json (+ run_manifest.json,
    else run_all.log and a fresh QC pass) without running any analysis."""
    fpath = results / FINDINGS
    if not fpath.exists():
        log.error("step=report_only status=failed: %s missing; run the analysis first", fpath)
        return 2
    findings = json.loads(fpath.read_text())
    mpath = results / MANIFEST
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        qc_summary = manifest.get("qc_summary") or {}
        status = (manifest.get("run_info") or {}).get("modules") or {}
        seconds = (manifest.get("run_info") or {}).get("seconds")
        log.info("step=report_only source=%s findings=%d", mpath.name, len(findings))
    else:
        status, seconds = status_from_log(results / "run_all.log")
        cfg = dataclasses.replace(C.DEFAULT_CONFIG, results_dir=results)
        qc_summary = {"threshold": cfg.threshold}
        try:
            inputs = load_inputs(cfg)
            trials = inputs["trials"]
            per_fly, summary = qc_step(trials, inputs, cfg, write=False)
            qc_summary.update(summary)
            _kept, qc_summary["main_exclusions"] = main_run_trials(trials, per_fly,
                                                                   args.exclude_qc)
            qc_summary["excluded"] = exclusion_counts(per_fly)[0]
        except Exception as exc:
            log.exception("step=report_only qc=failed")
            qc_summary["status"] = f"failed: {type(exc).__name__}: {exc}"
        log.info("step=report_only source=run_all.log+qc findings=%d", len(findings))
    ok = _finish(results, findings, qc_summary, status, seconds, write_findings=False)
    failed = any(f.get("analysis") == "failed" for f in findings)
    return 0 if ok and not failed else 1


def _finish(results: Path, findings: list[dict], qc_summary: dict, status: dict,
            seconds: float | None, *, write_findings: bool = True) -> bool:
    """findings.json + run_manifest.json, READMEs, SUMMARY.md, index.html. False if one
    of the report steps failed."""
    ok = True
    run_info = dict(modules=status, seconds=seconds)
    if write_findings:
        _write_json(results / FINDINGS, findings)
    _write_json(results / MANIFEST, dict(qc_summary=qc_summary, run_info=run_info))
    for step, fn in (("write_readmes", lambda: report.write_readmes(results)),
                     ("write_summary", lambda: report.write_summary(
                         results, findings, qc_summary, run_info=run_info)),
                     ("write_index", lambda: report.write_index(results))):
        try:
            fn()
            log.info("step=%s status=ok", step)
        except Exception:
            log.exception("step=%s status=failed", step)
            ok = False
    return ok


if __name__ == "__main__":
    sys.exit(main())
