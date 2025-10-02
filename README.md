
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
