# Project Instructions

Fly-behaviour recording + analysis repo. Three Raspberry Pi rigs record PER
(proboscis extension response) trials; `fbpipe` turns the videos into per-trial
parquet/CSV and figures; an ESPHome enclosure logs environment telemetry to Home
Assistant.

Deeper "why" for most rules below lives in the per-topic memory files indexed by
`MEMORY.md`. This file is the always-loaded summary — keep it to facts that
apply on nearly every task.

## Repo map

| Path | What |
|---|---|
| `src/fbpipe/steps/` | pipeline stages: `yolo_infer`, `reject_bad_proboscis`, `distance_stats`, `distance_normalize`, `compose_videos_rms`, … |
| `src/fbpipe/analysis/` | table/trace readers (`traces.read_wide_table`), stats |
| `src/fbpipe/utils/` | `rig_gates.py`, `rig_anchor.py`, `trial_metadata.py`, `light_stimulus.py` |
| `scripts/pipeline/run_workflows.py` | the pipeline driver — every step is orchestrated here |
| `scripts/analysis/` | figure scripts; **many are hand-run, not wired into the pipeline** |
| `config/` | gitignored except `example.*` |
| `PiCode/` | code pushed to the Pis; gitignored |
| `tests/` | plain pytest, no pytest config |

## Commands

```bash
# full pipeline (operative config — NOT the Makefile default)
python3 scripts/pipeline/run_workflows.py --config config/config_new.yaml

# restyle figures from existing CSVs (see the freeze caveat below)
python3 scripts/pipeline/run_workflows.py --config config/config_new.yaml --figures-only

# tests
python3 -m pytest tests/ -q
```

`make run` / `make figures` default to `CONFIG=config/config.yaml`, which is
**not** the live config — pass `CONFIG=config/config_new.yaml` or call the driver
directly.

## Config

- **`config/config_new.yaml` is the operative config.** `config.yaml`,
  `config_manual.yaml`, `example.yaml` and the `src/fbpipe/config.py` dataclass
  defaults carry *different* gate values (150 / 180 / 250 vs 160 / 160 / 160).
- `config/*` is **gitignored** — code and tests get committed, the config half of
  any change lives only on disk. Say so explicitly when handing off work.
- Never hardcode a gate value in an analysis or figure. Load it from
  `config_new.yaml` at runtime, and read the key the production code actually
  reads: `proboscis_filter.max_eye_prob_distance_px` (spatial gate, every fly
  count) and `distance_limits.three_fly_max_eye_prob_distance_px` (≥3-fly
  pairing) are both 160.0 here, so reading the wrong one is correct only by
  coincidence. In a test, give the two keys *different* values.
- YAML keeps the **last** duplicate key. A dataset that already has a `freeze:`
  block silently loses a second one, parsing fine while freezing nothing. Verify
  by resolving real folder names, not by reading the config.

## Where output goes

- `Results/New-Opto-Fly-Figures/` — everything the `config_new` pipeline writes.
  Default any new pipeline step's `out_root`/`out_dir` here.
- `Results/Figures/` — hand-run one-off sets (`*_mean_traces_new`,
  `*_rig_batch_breakdowns_new`, pubfigs). A pipeline step must never write here,
  even when the reference figures were hand-made there.

## Rigs

| Host | Tailscale IP | Script | LIGHT_PIN | Light drive | PiCode dir on Pi | venv |
|---|---|---|---|---|---|---|
| `behaviorlocust` (Pi 1) | `100.121.64.36` | `combinedv2_1_pi1.py` | 25 | Thorlabs TTL, digital only — all PWM stripped 2026-07-29 | `~/Documents/Cole/` (capital C) | `~/cam-env/bin/python` |
| `flybehavior2` (Pi 2) | `100.93.23.70` | `combinedv2_1.py` + `phase_status.py` (must ship together) | 13 | LDD-L PWM, `--light-brightness` 15 | `~/Documents/cole/` (lowercase c) | `~/myenv/bin/python` |
| `flybehavior3` (Pi 3) | `100.97.155.65` | `combinedv2_1_pi3.py` | — | LDD-L PWM at **30** — edited on the Pi; a plain `PiCode/` sync reverts it to 15 | — | — |

`ssh <host>` is keyless via `~/.ssh/pi_backup` (do not "fix" it back to
`id_ed25519` — that breaks the backup cron). Address rigs by **tailnet IP**;
Pi 3's LAN IP is DHCP and has moved subnets. `100.94.160.60` is `behavior4080`,
the user's desktop — not a Pi.

- **Never** use `--delete`, `--delete-excluded`, `--delete-after` or any rsync
  delete flag on a Pi push. Additive sync only. If a mirror sync is genuinely
  wanted, list the files that would be deleted and get an explicit go-ahead.
- Ctrl-C on a local `ssh` kills only the ssh client. The remote python keeps
  running and keeps the GPIO claimed, so the next launch dies with `GPIO busy`.
  `pkill -TERM -f <script>` on the Pi first, or invoke with `ssh -t`.

## Analysis invariants

- **Never average an average.** A cohort mean is the sum of all N×M trial scores
  divided by N×M — not the mean of per-fly means. When n therefore counts trials,
  label the axis that way (`n=40 trials`).
- **`SCORE_COLORS` in `scripts/analysis/score_summary.py` is fixed.** It is a
  CVD-validated PRGn purple→green ramp diverging at 1.5 (`binary_threshold=2`),
  pinned character-for-character by a test. Red→green was measured and rejected
  (protanope worst-pair ΔE 4.1). Reuse `_score_cmap()`; never define a second ramp.
- `flybehavior2` batches **strictly after 2026-07-25** use the mirrored left-edge
  anchor (`MIRRORED_HOSTS_AFTER`, `utils/rig_anchor.py`) **and** 280 px gates
  (`rig_gate_overrides`, `utils/rig_gates.py`). Same host and cutoff — the two
  must stay in sync. Any fly whose stats max sits flush against its gate
  (e.g. 159.9x) is being clipped; raw pre-reject values are unrecoverable from the
  parquet, and `yolo` only re-runs if the trial `out_dir` is deleted first.
- **Folder freeze**: frozen rows stay in the wide CSV marked `frozen: true` and
  are kept out of every plot. `--figures-only` **cannot apply a changed freeze
  rule** — it forces `skip_combined`, so the `frozen` column keeps its old values
  and the figures silently retain folders you just retired. Change a freeze or
  cohort key → full run.
- Retirement globs must be **date-bounded** (`{match: "*_rig_3", before: …}`) and
  identical across both arms of a trained/control pair. rig_3 is good again from
  2026-08-11; an unbounded glob silently swallows every later batch.
- RandomPanel datasets are **testing-only** (`trial_type_override: testing`).
  Training-only steps must skip gracefully (`NoTargetTrialsError`,
  `_csv_has_training_rows`), never abort the run.
- Regenerating anything under `Opto-Fly-Figures-OctNov` needs `--include-flagged`:
  11 of Hex-Control's 15 flies carry `dataset == "Hex-Control-flagged"`, so the
  figure silently rebuilds at n=4 instead of n=15.
- The current `ordinal_scorer` model *does* predict score 1; the older
  `combined_two_datasets` model never did (0 of ~5000). Check which one
  `config_new.yaml` points at before claiming a score distribution.
- Matplotlib traps that fail silently: `fig.colorbar(..., ax=ax)` shrinks that one
  axes and breaks matrix↔bar alignment (use a dedicated GridSpec column and
  `cax=`); `sharex` shares the tick *formatter*, so `set_xticklabels` on one axes
  relabels the other (use `secondary_xaxis`, which lives in `ax.child_axes`, not
  `fig.axes`).

## Known-stale / known-failing

- `tests/test_reaction_rate_bars_only.py` — 4 RandomPanel cases fail on a clean
  HEAD (verified 2026-08-26). The expectations are July percentages typed into
  the test; the threshold rework moved the numbers. Don't chase them in an
  unrelated run.
- Hand-run scripts whose PNGs sit beside pipeline-owned ones and drift silently:
  `reaction_rate_bars_only.py`, `rig_batch_breakdowns.py`,
  `pubfig_naive_vs_trained.py`. Regenerating `rig_batch_breakdowns` for EB-24-1
  requires its `--batch-override` flags — ask before regenerating.
- Aborted/restarted sessions leave short orphan `*_envelope.csv` files that get
  scored as real trials. Detect by **duplicate trial index within a fly**, never
  by absolute length (some protocols are genuinely short). ~105 such rows across
  26 batches are still outstanding.

## Enclosure telemetry (ESPHome / Home Assistant)

- Fly enclosure: ESP32 + BME680 sensors, LED light cycle, fan, heater. ESPHome
  config `esphomeflynursery2.yaml`.
- Home Assistant `10.229.137.171:8123`; InfluxDB `10.229.137.171:8086`, database
  `homeassistant`, user `homeassistant`. Credentials in `.env` — never commit
  secrets.
- Entities: `flynursery2_{temperature,humidity,pressure}_{fly_2,room}`,
  `flynursery2_fly_sun_brightness`, `flynursery2_heat_pad_watts`,
  `flynursery2_average_rpm_2`. Measurements are keyed by unit: `°C`, `%`, `hPa`,
  `RPM`, `W`.
- HA's InfluxDB block uses an explicit entity include list in
  `configuration.yaml` — a new sensor must be added there or it never reaches
  Influx.

## Workflow

- **Test first, then change.** Write and run a test pinning current behaviour and
  validating assumptions before editing implementation.
- New dependency on an external service (InfluxDB, MQTT, an API): write a
  standalone connectivity script and confirm the data exists before wiring it
  into the pipeline.
- Re-run the suite before quoting any pass/fail count — concurrent sessions
  rewriting shared files have produced dozens of phantom failures.
