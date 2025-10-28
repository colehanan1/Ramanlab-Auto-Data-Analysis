
# Fly Behavior Pipeline (YOLO → Distances/Angles → RMS/Envelope → Videos)

End-to-end, reproducible pipeline that:
1) Runs YOLO (OBB or axis-aligned) on videos with a Kalman tracker + optional optical flow.
2) Writes per-frame CSVs (coordinates, distances, angles).
3) Computes global min/max for distances; normalizes to percent.
4) Flags dropped frames, stages RMS-ready CSVs, and annotates OFM state.
5) Organizes trial videos and renders line-panel videos with RMS overlays.
6) Produces consistent, versioned outputs under each *fly* folder and hands matrix-ready data to the analysis scripts.
7) Scores proboscis reactions with the `flybehavior-response` model CLI.
8) Builds black/white reaction matrices directly from the model predictions.

> **Minimum**: Linux + CUDA GPU. Set `model_path` and `main_directory` in `config.yaml` or `.env`.

---

## Quickstart

```bash
# 0) Prepare venv + deps (or use your own Conda env; see below)
make setup

# 1) Install/refresh dependencies inside the active environment
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# (optional but recommended if you are developing):
python -m pip install -e .

# The requirements pin `numpy<2` so the packaged reaction model, which was
# trained against NumPy 1.x, deserialises correctly. If your environment already
# has NumPy 2.x installed, rerun the install command above to downgrade before
# launching the pipeline. The bundled `sitecustomize.py` only activates its
# MT19937 shim when it detects NumPy 2.x, so it stays inert once the downgrade
# completes but future-proofs the workflow if you ever upgrade again.

# 2) Configure paths
cp .env.example .env
# edit MODEL_PATH and MAIN_DIRECTORY, or edit config.yaml directly

# 3) Run everything (env must stay active)
make run   # or: python scripts/run_workflows.py --config config.yaml
```

### Using an existing Conda environment

If you already have a GPU-capable Conda environment (e.g., `yolo-env`), activate it and install this project's dependencies into that environment instead of creating a new virtualenv:

```bash
conda activate yolo-env
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# (optional but recommended if you are developing):
python -m pip install -e .

# As above, make sure the install step downgrades any pre-existing NumPy 2.x
# builds; the reaction-scoring artifacts require NumPy 1.x semantics. The
# bundled `sitecustomize.py` now short-circuits under NumPy 1.x and only patches
# MT19937 handling if a future dependency pulls in NumPy 2.x, so leave it in
# place even after the downgrade.
```

The `make setup` target simply automates those steps against a fresh `venv`. Skipping it is safe as long as the active environment satisfies `requirements.txt` and provides CUDA-enabled builds of PyTorch/Ultralytics.

### Sharing the environment with collaborators

To let others reproduce your Conda setup:

1. Export a lock file that records the packages you are using:

   ```bash
   conda env export --name yolo-env --from-history > environment.yml
   ```

   The `--from-history` flag keeps the file focused on top-level dependencies rather than every transitive build detail, which reduces cross-platform friction. Drop the flag if you need an exact replica including build strings.

2. Commit or distribute `environment.yml` alongside the code. Collaborators can recreate the environment with:

   ```bash
   conda env create -f environment.yml -n yolo-env
   conda activate yolo-env
   ```

3. When you need to hand off a ready-to-run binary snapshot (for offline systems), package it as an archive:

   ```bash
   conda pack -n yolo-env -o yolo-env.tar.gz
   ```

   Recipients can unpack the archive and run `conda-unpack` in place.

Remember to keep CUDA drivers and GPU libraries aligned across machines; the exported environment only covers the user-space Python dependencies.

## Structure

```
fbpipe/
  config.py                  # load config (YAML + .env + env vars)
  pipeline.py                # simple CLI orchestrator
  utils/                     # helpers (timestamps, tracking, geometry)
  steps/                     # individual steps (you can run separately)
    yolo_infer.py
    distance_stats.py
    distance_normalize.py
    detect_dropped_frames.py
    rms_copy_filter.py
    update_ofm_state.py
    move_videos.py
    compose_videos_rms.py
    predict_reactions.py
    reaction_matrix.py
```

Each step can also be run independently:
```bash
python -m fbpipe.steps.yolo_infer --config config.yaml
python -m fbpipe.steps.distance_stats --config config.yaml
...
```

The two reaction-analysis steps wrap the new automation (run automatically at
the end of `make run`):

```bash
python -m fbpipe.steps.predict_reactions --config config.yaml
python -m fbpipe.steps.reaction_matrix --config config.yaml
```

The first command invokes the packaged `flybehavior-response predict` CLI to
write a spreadsheet of binary responses. The second command feeds that
spreadsheet into `scripts/reaction_matrix_from_spreadsheet.py`, reproducing the
figure layout without manual intervention.

## Geometric feature extraction CLI

The repository ships with `geom_features.py`, a standalone tool for deriving
per-frame and per-trial geometry from the eye/proboscis coordinate exports. Run
it after the YOLO step has written the post-processed CSVs:

```bash
# Basic invocation: scan one or more dataset roots and write outputs under outdir
python geom_features.py --roots /data/opto_EB /data/opto_PB --outdir /data/geom

# Preview the work without writing files
python geom_features.py --roots /data/opto_EB --outdir /tmp/geom --dry-run

# Smoke-test a subset of flies/trials
python geom_features.py --roots /data/opto_EB --outdir /tmp/geom --limit-flies 1 --limit-trials 2

# Append behaviour windows to a single enriched CSV
python geom_features.py --input /data/opto_EB/.../testing_6_fly1_distances_geom.csv
```

> **Note:** `geom_features.py` creates directories under `--outdir`. Pick a
> location that already exists or that you have permission to create (for
> example, somewhere under your home directory or within the dataset roots).
> Using protected locations such as `/data` without elevated privileges will
> now raise a clear permission error before any processing begins.

Key behaviours:

* The tool recursively discovers trial CSVs in directories whose names contain
  `training` or `testing` tokens beneath each fly folder (for example,
  `.../september_10_fly_2/september_10_fly_2_testing_1/september_10_fly_2_testing_1_fly1_distances.csv`).
* Each trial produces an enriched `<trial>_geom.csv` alongside the source file
  unless `--outdir` is specified, in which case the enriched CSVs are written to
  the mirrored folder structure under that directory. Every enriched dataframe
  now also gains a behavioural companion `<trial>_with_behavior.csv` saved right
  next to the raw YOLO output containing the same per-frame geometry plus the
  new window masks and summary scalars described below. The `_geom` export
  retains the full legacy schema for backwards compatibility.
* A consolidated `geom_features_all_flies.csv` is always written inside
  `--outdir` with one summary row per trial, including per-fly scaling metrics
  and per-trial statistics.
* All enriched frames from *testing* trials are streamed into
  `geom_features_testing_all_frames.csv` within `--outdir`. Each row begins with
  the fly identifiers (`dataset`, `fly`, `fly_number`, `trial_type`,
  `trial_label`)—for example, `opto_EB,september_09_fly_1,1,testing,testing_10`—
  followed by the per-fly scale metrics, the per-trial summaries (including the
  new baseline/during/early behavioural scalars), and finally the per-frame
  geometry. The geometry block now includes three mask columns
  (`is_before`, `is_during`, `is_after`) so you can slice specific windows
  without recomputing logic downstream. Only frames whose `frame` index falls
  between 0 and 3600 (inclusive) are written so that the combined CSV remains
  tractable while still covering the first minute of a 60 fps recording. A
  companion
  `geom_features_testing_all_frames.schema.json` lists the column groups,
  ordering, and dtypes so you can load the massive table predictably with
  `pandas`/`polars`/Spark. Expect this file to grow large (tens of millions of
  rows) when processing entire cohorts.
* You can enrich a previously generated CSV in isolation with the new window
  metrics via `python geom_features.py --input path/to/trial_geom.csv`. This
  mode writes `path/to/trial_with_behavior.csv` (or prints the intended action
  when combined with `--dry-run`) and logs the key diagnostics: frame counts per
  window, baseline/during means, fraction of high extensions during odor, rise
  speed, and directional consistency.
* Coordinate schemas in either long or wide format are handled automatically;
  if the initial trial CSV lacks coordinates, the CLI searches for the
  corresponding `*_coords*.csv` helper in the same or parent directory.

The command depends on `pandas` and `numpy`, which are already pinned in
`requirements.txt`. Activate the same environment you use for the main pipeline
before invoking the CLI.

### Behavioural window columns

The per-frame block now ends with three integer mask columns:

| Column | Meaning |
| --- | --- |
| `is_before` | 1 when `frame < 1260` (baseline), otherwise 0 |
| `is_during` | 1 when `1260 ≤ frame < 2460` (odor presentation), otherwise 0 |
| `is_after` | 1 when `frame ≥ 2460` (post-odor), otherwise 0 |

Every row also includes the behavioural scalars computed from those windows and
anchored to each trial:

| Column | Description |
| --- | --- |
| `r_before_mean`, `r_before_std` | Mean and standard deviation of `r_pct_robust_fly` during the baseline window |
| `r_during_mean`, `r_during_std` | Mean and standard deviation of `r_pct_robust_fly` during odor |
| `r_during_minus_before_mean` | Difference between odor and baseline means |
| `cos_theta_during_mean`, `sin_theta_during_mean` | Mean pointing direction components during odor |
| `direction_consistency` | Magnitude of the mean direction vector (`√(cos²+sin²)`) |
| `frac_high_ext_during` | Fraction of odor frames where `r_pct_robust_fly ≥ 75%` |
| `rise_speed` | Percent-extension per second across the first 1 s after odor onset |

These scalars propagate to the consolidated trial summary and the aggregated
testing CSV so downstream ML jobs can leverage both frame-level and trial-level
behaviour without recomputing window logic.

## Multi-fly YOLO inference

The pipeline now exports up to four concurrent flies (class-2 eyes paired with
class-8 proboscis tracks) from a single video. No extra entry point is required:
once configured, `make run` or `python -m fbpipe.steps.yolo_infer --config
config.yaml` automatically fans the merged CSV into per-fly files.

1. **Configure limits** – edit `config.yaml` (or corresponding environment
   variables) under the `yolo` section:

   | Key | Purpose |
   | --- | --- |
   | `max_flies` | Number of eye anchors (slots) to track; defaults to 4. |
   | `max_proboscis_tracks` | Number of proboscis tracks kept after confidence filtering. |
   | `pair_rebind_ratio` | Fraction of the frame diagonal used as the rebinding radius when an eye temporarily loses its paired proboscis. |
   | `zero_iou_epsilon` | Numeric tolerance for enforcing non-overlapping eye anchors. |

   Environment overrides are available (e.g., `MAX_FLIES=3 make run`).【F:config.yaml†L11-L33】【F:src/fbpipe/config.py†L11-L72】

2. **Run inference** – execute `make run` for the entire pipeline or call the
   YOLO step directly if you only need detections:

   ```bash
   python -m fbpipe.steps.yolo_infer --config config.yaml
   ```

   The step emits:

   - `*_distances_merged.csv`: merged view with all fly slots and metadata.
   - `*_fly{N}_distances.csv`: one file per populated slot with the legacy
     single-fly schema.【F:src/fbpipe/steps/yolo_infer.py†L234-L318】

3. **Inspect outputs** – annotated videos and CSVs land in the processed folder
   created beside each source video. Downstream steps (distance stats,
   normalization, RMS, etc.) consume the merged CSVs without additional
   configuration.

When you add more flies per video, confirm that your YOLO model reliably
separates the class-2 detections. Raising `zero_iou_epsilon` loosens the
non-overlap constraint if detections sit close together, while reducing
`pair_rebind_ratio` tightens how far a proboscis track can drift before it is
considered unpaired.

## GPU requirements

The pipeline expects a CUDA-capable GPU for production workloads. Setting `allow_cpu: true` (or `ALLOW_CPU=true` via environment) now **forces** the YOLO step to run entirely on CPU, which is useful for smoke tests or when the local CUDA stack is unstable. If CUDA initialisation or inference fails while GPU mode is requested and `allow_cpu` is enabled, the step logs a warning and permanently switches to CPU for the rest of the run. CPU mode is significantly slower and should only be used for debugging.

## Notes

- The pipeline is resilient to missing files/columns; steps skip gracefully when inputs are absent.
- RMS/Envelope calculations ignore values outside `[0, 100]` and NaNs.
- Video overlay deletes source trial videos after composing by default; toggle in `config.yaml`.
- See `docs/pipeline_overview.md` for a deeper look at how the steps are orchestrated.

## dir_val heatmap utility

Generate per-fly odor heatmaps, combined-condition views, and mean ± SEM traces from a wide CSV. The utility now caps every heatmap to the first 3,600 frames and produces dataset-wide aggregates that stack all `testing_6`–`testing_10` trials across flies alongside the per-fly combined figure. A second dataset overview renders a single heatmap where each row is the across-fly average trace for `TRAIN-COMBINED`, `TEST-COMBINED`, and `testing_6`–`testing_10`, making it easy to spot cross-fly consensus dynamics at a glance:

```bash
python scripts/plot_dirval_heatmaps.py \
  --csv /path/to/all_envelope_rows_wide.csv \
  --dataset opto_EB \
  --outdir results/heatmaps \
  --labels-train 2 4 5 \
  --labels-test 1 3 \
  --colprefix dir_val_ \
  --normalize zscore \
  --sort-by peak \
  --grid-cols 2 \
  --dpi 300 \
  --dry-run 0 \
  --log-level INFO
```

### Colour scaling and normalisation

Historically the plots relied on z-score normalisation and percentile clipping to highlight relative structure within each trial. With the latest update every heatmap defaults to the physical `dir_val` scale: the colour bar spans `0` (dark purple) to `200` (bright yellow) whenever `--normalize none` is active, ensuring consistent interpretation across flies and datasets. Opt into `--normalize zscore` when you explicitly want per-trial standardisation; in that mode the code falls back to the robust percentile-driven limits so the colour bar reflects standard deviations rather than raw millimetre values. Direction values now apply a unity floor to the angle-derived multiplier so low-angle periods no longer attenuate the distance percentage—`dir_val` only scales up from the base distance trace.

## Standalone analysis scripts

### Reaction matrices from spreadsheet predictions

Generate the black/white reaction matrices directly from the manual scoring
spreadsheet used during the review process:

```bash
python scripts/reaction_matrix_from_spreadsheet.py \
    --csv-path /home/ramanlab/PycharmProjects/FlyBehaviorScoring/artifacts/predictions_envelope_t2.csv \
    --out-dir artifacts/reaction_matrices_spreadsheet \
    --latency-sec 0.0
```

The command mirrors the original figure naming and will emit PNGs plus helper
CSV/row-key exports under the chosen output directory.

## Nightly automation

To run the full pipeline every night at midnight, use the bundled cron helpers:

1. Ensure the Conda environment you use for the project (default: `yolo-env`) is available in non-interactive shells. If `conda`
   is not on cron's `PATH`, export `CONDA_BASE=/path/to/miniconda3` (or similar) before installing the job so the nightly
   wrapper can source the correct `conda.sh`.
2. From the repository root, install/update the cron job:

   ```bash
   ./scripts/install_midnight_cron.sh
   ```

   The installer writes a `0 0 * * *` entry that launches `make run` via
   `scripts/nightly_make_run.sh`, appending logs to `logs/nightly_make_run.log`.

3. Adjust behaviour via environment variables when invoking the installer:

   ```bash
   YOLO_ENV_NAME=my-env ./scripts/install_midnight_cron.sh
   ```

   This overrides the Conda environment activated before `make run` executes.

4. Inspect or prune the cron entry with standard tooling:

   ```bash
   crontab -l        # list entries (should include the "Ramanlab Auto Data Analysis" marker)
   crontab -e        # edit manually if needed
   ```

To remove the scheduled run entirely, delete the two lines marked with the
`Ramanlab Auto Data Analysis nightly make run` comment from your crontab.
