# Multi-fly trial post-processing CLI

This command walks the YOLO output tree for every dataset root you supply, normalises each fly's trials, writes per-trial CSVs with the derived metrics, and finishes by exporting a fly-wide and an all-fly summary table.

## When to run it

Run the tool after the standard pipeline (`make run` or `python -m fbpipe.steps...`) finishes producing the per-trial `flyN_distances.csv` files. Point the CLI at the directories that contain the fly folders (for example, `opto_benz_1/September_9_fly_1/...`).

## Basic usage

```bash
# activate the same environment you used for the pipeline
python scripts/multi_fly_processing.py /path/to/opto_benz_1 /path/to/opto_benz_2
```

The positional arguments are dataset roots. Each root should contain one directory per fly; the script will recurse into those folders and locate every `fly*_distances.csv` trial automatically.【F:scripts/multi_fly_processing.py†L39-L112】【F:scripts/multi_fly_processing.py†L547-L588】

Use `--help` to see all options and their defaults:

```bash
python scripts/multi_fly_processing.py --help
```

## What the command writes

For every fly slot discovered under a root, the CLI will:

1. Derive global min/max distance bounds across all of the slot's trials so the normalisation is consistent per fly.【F:scripts/multi_fly_processing.py†L344-L378】【F:scripts/multi_fly_processing.py†L400-L442】
2. Save an augmented CSV for each trial in `multi_fly_processed/<slot_name>/`. These files contain the raw distances, percentage-normalised distances, RMS, and Hilbert envelope columns alongside the original YOLO output.【F:scripts/multi_fly_processing.py†L443-L515】
3. Assemble a fly-level wide CSV (`<slot_name>_combined.csv`) containing metadata, summary metrics (AUC, peak, latency), and the envelope time-series for every trial.【F:scripts/multi_fly_processing.py†L516-L566】
4. After all roots finish, emit a single `all_flies_combined.csv` under the directory named by `--output` (defaults to `multi_fly_outputs/`).【F:scripts/multi_fly_processing.py†L590-L641】

You will see `[INFO]`, `[OK]`, or `[WARN]` log lines for each step so you can monitor progress.【F:scripts/multi_fly_processing.py†L436-L519】【F:scripts/multi_fly_processing.py†L598-L641】

## Tuning knobs

The defaults mirror the existing pipeline configuration, but you can override them per run:

* `--distance-min` / `--distance-max` – clip the acceptable proboscis distances before computing the global bounds.【F:scripts/multi_fly_processing.py†L600-L623】
* `--fps-default` – fallback frame rate when timestamps are missing; used in envelope calculations.【F:scripts/multi_fly_processing.py†L458-L514】【F:scripts/multi_fly_processing.py†L600-L623】
* `--window-sec` – width (seconds) of the RMS/Hilbert window applied to the normalised distance percentage.【F:scripts/multi_fly_processing.py†L468-L514】【F:scripts/multi_fly_processing.py†L618-L623】

Adjust these flags to match any deviations in your acquisition settings.
