
# Fly Behavior Pipeline (YOLO → Distances/Angles → RMS/Envelope → Videos)

End-to-end, reproducible pipeline that:
1) Runs YOLO (OBB or axis-aligned) on videos with a Kalman tracker + optional optical flow.
2) Writes per-frame CSVs (coordinates, distances, angles).
3) Computes global min/max for distances; normalizes to percent.
4) Flags dropped frames, stages RMS-ready CSVs, and annotates OFM state.
5) Organizes trial videos and renders line-panel videos with RMS overlays.
6) Produces consistent, versioned outputs under each *fly* folder and hands matrix-ready data to the analysis scripts.

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

# 2) Configure paths
cp .env.example .env
# edit MODEL_PATH and MAIN_DIRECTORY, or edit config.yaml directly

# 3) Run everything (env must stay active)
make run   # or: python -m fbpipe.pipeline --config config.yaml all
```

### Using an existing Conda environment

If you already have a GPU-capable Conda environment (e.g., `yolo-env`), activate it and install this project's dependencies into that environment instead of creating a new virtualenv:

```bash
conda activate yolo-env
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# (optional but recommended if you are developing):
python -m pip install -e .
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
```

Each step can also be run independently:
```bash
python -m fbpipe.steps.yolo_infer --config config.yaml
python -m fbpipe.steps.distance_stats --config config.yaml
...
```

## GPU requirements

The pipeline expects a CUDA-capable GPU for production workloads. Setting `allow_cpu: true` (or `ALLOW_CPU=true` via environment) now **forces** the YOLO step to run entirely on CPU, which is useful for smoke tests or when the local CUDA stack is unstable. If CUDA initialisation or inference fails while GPU mode is requested and `allow_cpu` is enabled, the step logs a warning and permanently switches to CPU for the rest of the run. CPU mode is significantly slower and should only be used for debugging.

## Notes

- The pipeline is resilient to missing files/columns; steps skip gracefully when inputs are absent.
- RMS/Envelope calculations ignore values outside `[0, 100]` and NaNs.
- Video overlay deletes source trial videos after composing by default; toggle in `config.yaml`.
- See `docs/pipeline_overview.md` for a deeper look at how the steps are orchestrated.

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
