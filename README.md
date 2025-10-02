
# Fly Behavior Pipeline (YOLO → Distances/Angles → RMS/Envelope → Videos)

End-to-end, reproducible pipeline that:
1) Runs YOLO (OBB or axis-aligned) on videos with a Kalman tracker + optional optical flow.
2) Writes per-frame CSVs (coordinates, distances, angles).
3) Computes global min/max for distances; normalizes to percent.
4) Generates plots (time-series, histograms, envelopes).
5) Computes angles and centered-angle-% heatmaps.
6) Organizes trial videos and renders line-panel videos with RMS overlay.
7) Produces consistent, versioned outputs under each *fly* folder.

> **Minimum**: Linux + CUDA GPU. Set `model_path` and `main_directory` in `config.yaml` or `.env`.

---

## Quickstart

```bash
# 0) Activate your Conda environment (no virtualenv is created by the repo)
conda activate yolo-env

# 1) Install/refresh dependencies inside the active environment
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# (optional but recommended if you are developing):
python -m pip install -e .

# 2) Configure paths
cp .env.example .env
# edit MODEL_PATH and MAIN_DIRECTORY, or edit config.yaml directly

# 3) Run everything (env must stay active)
make run   # or: python -m fbpipe.pipeline --config config.yaml all
```

### Nightly automation

If you want the pipeline to ingest new data every night, install the bundled cron job helper after activating `yolo-env` once:

```bash
./scripts/install_midnight_cron.sh
```

This command registers a cron entry that executes `make run` at midnight server time, logs output to `logs/nightly_make_run.log`, and ensures the `yolo-env` Conda environment is activated via `scripts/nightly_make_run.sh`. Adjust the `YOLO_ENV_NAME` environment variable before scheduling if you rely on a differently named Conda environment. Review or edit the schedule at any time with `crontab -l` / `crontab -e`.

### Envelope exports for optogenetics datasets

To reproduce the Jupyter utilities that generate wide CSVs of distance envelopes and the downstream float16 matrix, use the scripted workflow:

```bash
python scripts/envelope_exports.py collect \
  --roots /home/ramanlab/Documents/cole/Data/flys/opto_benz/ \
         /home/ramanlab/Documents/cole/Data/flys/opto_EB/ \
         /home/ramanlab/Documents/cole/Data/flys/opto_benz_1/ \
  --out-csv /home/ramanlab/Documents/cole/Data/single_matrix_opto/all_envelope_rows_wide.csv

python scripts/envelope_exports.py convert \
  --input-csv /home/ramanlab/Documents/cole/Data/single_matrix_opto/all_envelope_rows_wide.csv \
  --out-dir /home/ramanlab/Documents/cole/Data/single_matrix_opto/
```

All arguments are optional; the defaults mirror the notebook cells (measurement column priorities, FPS fallbacks, rolling window length, and output paths). Supply additional roots or override the output destinations when your data lives elsewhere.

### Envelope visualisation commands

After converting the wide CSV into a float16 matrix, the new `scripts/envelope_visuals.py` helper renders the reaction matrices and per-fly envelope traces without touching notebooks.

Generate both the "testing order" and "trained-first" matrix variants (each odor ends up in its own subfolder) and drop per-odor CSV summaries for the trained-first ordering:

```bash
python scripts/envelope_visuals.py matrices \
  --matrix-npy /home/ramanlab/Documents/cole/Data/single_matrix_opto/envelope_matrix_float16.npy \
  --codes-json /home/ramanlab/Documents/cole/Data/single_matrix_opto/code_maps.json \
  --out-dir /home/ramanlab/Documents/cole/Results/Opto/Matrixs_DIST \
  --latency-sec 2.75
```

Render per-fly envelope traces, highlighting the trained odor and shading the latency-adjusted delivery window. Each fly's figure is replicated into every odor folder that appears in its trials:

```bash
python scripts/envelope_visuals.py envelopes \
  --matrix-npy /home/ramanlab/Documents/cole/Data/single_matrix_opto/envelope_matrix_float16.npy \
  --codes-json /home/ramanlab/Documents/cole/Data/single_matrix_opto/code_maps.json \
  --out-dir /home/ramanlab/Documents/cole/Results/Opto/Envlope_DIST \
  --latency-sec 2.75
```

Both commands expose flags for FPS fallbacks, baseline/during/after windows, bar-chart spacing, and trial-order selection, so you can adapt the exports if acquisition parameters change.

### Training envelopes and latency summaries

`scripts/envelope_training.py` covers the training-trial notebooks. It reads the same CSV folders and float16 matrix, but it focuses exclusively on `training_*` files.

Render per-fly Hilbert envelopes with global peaks annotated and the odor window shaded:

```bash
python scripts/envelope_training.py plots \
  --root /home/ramanlab/Documents/cole/Data/flys/opto_hex/ \
  --fps-default 40 --window-sec 0.25 --odor-on 30 --odor-off 60
```

Gather the resulting PNGs into a single folder (defaults to `all_training_envelope_plots/` under the root):

```bash
python scripts/envelope_training.py collect \
  --root /home/ramanlab/Documents/cole/Data/flys/opto_hex/ \
  --dest-folder all_training_envelope_plots
```

Compute latency-to-threshold summaries directly from the float16 matrix. The command emits per-fly bars, per-odor means (with SEM), and a grand-mean CSV + figure:

```bash
python scripts/envelope_training.py latency \
  --matrix-npy /home/ramanlab/Documents/cole/Data/single_matrix_opto/envelope_matrix_float16.npy \
  --codes-json /home/ramanlab/Documents/cole/Data/single_matrix_opto/code_maps.json \
  --out-dir /home/ramanlab/Documents/cole/Results/Opto/Training_RESP_Time_DIST \
  --before-sec 30 --during-sec 35 --threshold-mult 4 --latency-ceiling 9.5 --trials 4 5 6
```

### Using an existing Conda environment

If you already have a GPU-capable Conda environment (e.g., `yolo-env`), the commands above install the required packages directly into it—no extra virtualenv will be created by default.

If you prefer a single command to refresh dependencies after updating `requirements.txt`, `make setup` simply runs the two `pip` commands shown in the Quickstart while respecting the currently active environment.

The repository no longer bootstraps its own `.venv`; it always relies on whichever environment (Conda, mamba, virtualenv, etc.) you have already activated before running the commands.

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
    plot_distance_time.py
    detect_dropped_frames.py
    rms_copy_filter.py
    update_ofm_state.py
    histograms.py
    envelope_over_time.py
    collect_envelope_plots.py
    angle_compute_and_plots.py
    angle_heatmaps.py
    move_videos.py
    compose_videos_rms.py
```

Each step can also be run independently:
```bash
python -m fbpipe.steps.yolo_infer --config config.yaml
python -m fbpipe.steps.distance_stats --config config.yaml
...
```

## GPU requirements

The pipeline expects a CUDA-capable GPU. If CUDA initialisation fails or the driver is unstable, the YOLO step will automatically fall back to CPU inference (with a warning) so long as `allow_cpu` is enabled. CPU mode is significantly slower and should be reserved for debugging or smoke tests.

## Notes

- The pipeline is resilient to missing files/columns; steps skip gracefully when inputs are absent.
- Angle centering uses the first frame where `distance_percentage == 0`, or otherwise the minimal absolute value across a fly.
- RMS/Envelope calculations ignore values outside `[0, 100]` and NaNs.
- Video overlay deletes source trial videos after composing by default; toggle in `config.yaml`.
- See `docs/pipeline_overview.md` for a deeper look at how the steps are orchestrated.
