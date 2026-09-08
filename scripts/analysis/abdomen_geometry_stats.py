#!/usr/bin/env python3
"""Statistics over the abdomen geometry pipeline outputs (PER vs abdomen movement, odors,
alive/weak/dead flies, blinded 'abdomen' comments, box size vs angle).

    conda run -n yolo-env python scripts/analysis/abdomen_geometry_stats.py --config config/config_abdomen_all.yaml

Reads every tracked batch under the config's dataset roots, joins the production model scores,
the blinded human scores and the flagged-fly states, and writes tables, figures and report.md
to <geometry.figures_out_dir>/stats (or --out-dir).
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fbpipe.geometry import stats as ST  # noqa: E402
from fbpipe.geometry.config import load_geometry_config  # noqa: E402

CSV_DIR = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--predictions", default=str(CSV_DIR / "model_predictions.csv"))
    p.add_argument("--blinded", default=str(CSV_DIR / "blinded_video_scores.csv"))
    p.add_argument("--flagged", default=str(CSV_DIR / "flagged-flys-truth.csv"))
    p.add_argument("--include-frozen", action="store_true")
    p.add_argument("--metrics-csv", default=None, help="reuse a previously written fly_trial_metrics.csv instead of re-reading every table")
    a = p.parse_args(argv)
    settings, gcfg = load_geometry_config(a.config)
    out_dir = Path(a.out_dir) if a.out_dir else Path(gcfg.figures_out_dir) / "stats"
    if a.metrics_csv:
        import pandas as pd
        df = pd.read_csv(a.metrics_csv)
    else:
        df = ST.collect_fly_trial_metrics(settings, gcfg, include_frozen=a.include_frozen)
    print(f"[STATS] {len(df)} fly-trials collected")
    df = ST.attach_labels(df, predictions_csv=Path(a.predictions), blinded_csv=Path(a.blinded), flagged_csv=Path(a.flagged))
    res = ST.run_analysis(df, out_dir)
    print(f"[STATS] {res['n_fly_trials']} fly-trials, {res['n_flies']} flies; pooled PER-vs-angle rho {res['pooled_rho']:.3f}; "
          f"{len(res['figures'])} figure(s) -> {res['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
