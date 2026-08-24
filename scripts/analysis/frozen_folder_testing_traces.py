#!/usr/bin/env python3
"""Raw testing PER traces for the folder-FROZEN flies a dataset dropped from its figures.

``3Oct-Training-24-0.1`` freezes ``*_rig_3`` recorded before 2026-08-11 (the rig
was miscalibrated until its repair). Those rows stay in the wide table marked
``frozen: true`` — the CSV is a complete record — but ``_load_wide_table`` drops
them, so no figure in ``Raw-Testing-PER-Traces/3OCT-Training-24-0.1/`` shows
them. This script renders exactly those removed flies, in the same style as the
pipeline's raw-testing figures, so the excluded data can be inspected.

Flies the flagged-flies truth CSV excludes (``FLY-State != 1``) are left out —
"non-flagged flies only" — and reported in the manifest so the drop is visible
rather than silent.

Outputs, under ``--out-dir``:

* ``<fly>_fly<N>_testing_envelope_trials_by_odor_30_shifted.png`` (+ ``.svg``)
  — one figure per fly, one stacked panel per odor trial, identical style to the
  pipeline's Raw-Testing-PER-Traces figures. The renderer drops the ``testing_9``
  light-only trial from the panels (it is not an odor trial), exactly as the
  pipeline figures do; that trial is still in the manifest and raw CSV.
* ``manifest.csv`` — one row per rendered trial (fly, fly number, trial, odor,
  figure, per-trial windows, AUC/peak metrics already in the wide table).
* ``excluded_flagged_flies.csv`` — the frozen flies skipped as flagged.
* ``traces_raw.csv[.gz]`` — the raw ``dir_val_*`` envelope samples behind the
  figures, one row per trial (written unless ``--no-raw-csv``).
* ``README.md`` — what the folder is, how it was made, how to reproduce it.

Usage::

    python scripts/analysis/frozen_folder_testing_traces.py
    python scripts/analysis/frozen_folder_testing_traces.py \
        --dataset 3Oct-Control-24-0.1 --out-dir /tmp/rig3-control
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_DATASET = "3Oct-Training-24-0.1"
DEFAULT_PATTERN = "*_rig_3"
DEFAULT_CONFIG = _REPO / "config" / "config_new.yaml"
DEFAULT_WIDE = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_combined_base.parquet"
)
DEFAULT_OUT_DIR = Path(
    "/home/ramanlab/Documents/cole/Results/"
    "3Oct-Training-24-0.1_Removed-rig_3_Raw-Testing-Traces"
)

FIGURE_SUFFIX = "_testing_envelope_trials_by_odor_30_shifted.png"


class NoFrozenRowsError(RuntimeError):
    """No frozen, non-flagged testing rows matched the requested selection."""


# ── selection ───────────────────────────────────────────────────────────────
def _canon_fly_number(value: object) -> str:
    """``1.0`` -> ``"1"``. The truth CSV and the wide table disagree on this."""
    from fbpipe.config import canon_fly_number

    return canon_fly_number(value)


def select_rows(
    df: pd.DataFrame,
    *,
    dataset: str = DEFAULT_DATASET,
    pattern: str = DEFAULT_PATTERN,
    flagged_flies_csv: str = "",
    trial_type: str = "testing",
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Return the frozen, non-flagged rows to render, plus what was dropped.

    The returned frame has ``frozen`` cleared: ``_load_wide_table`` calls
    ``drop_frozen`` unconditionally, so a still-frozen slice would render empty.

    Raises :class:`NoFrozenRowsError` when nothing survives — an empty run that
    quietly writes a README and no figures is the failure mode to avoid.
    """
    if "frozen" not in df.columns:
        raise NoFrozenRowsError(
            "wide table has no `frozen` column; it predates folder freeze "
            "(rebuild it with the current build_wide_csv)."
        )

    ds = df["dataset"].astype(str).str.strip()
    frozen = df["frozen"].fillna(False).astype(bool)
    fly = df["fly"].astype(str).str.strip()
    matches_pattern = fly.map(lambda name: fnmatchcase(name, pattern))
    is_type = df["trial_type"].astype(str).str.strip().str.lower() == trial_type.lower()

    sel = df.loc[(ds == dataset) & frozen & matches_pattern & is_type].copy()

    dropped: dict[str, list] = {"flagged": []}
    if flagged_flies_csv and Path(flagged_flies_csv).exists():
        from fbpipe.config import load_flagged_fly_exclusions

        exclusions = load_flagged_fly_exclusions(flagged_flies_csv)
        keys = list(
            zip(
                sel["dataset"].astype(str).str.strip(),
                sel["fly"].astype(str).str.strip(),
                sel["fly_number"].map(_canon_fly_number),
                strict=False,
            )
        )
        flagged = pd.Series([k in exclusions for k in keys], index=sel.index)
        dropped["flagged"] = sorted({k for k, f in zip(keys, flagged, strict=False) if f})
        sel = sel.loc[~flagged].copy()

    if sel.empty:
        raise NoFrozenRowsError(
            f"No frozen non-flagged {trial_type} rows for dataset={dataset!r} "
            f"pattern={pattern!r}. Nothing to render."
        )

    # Un-freeze the slice so the figure loader keeps it (see docstring).
    sel["frozen"] = False
    return sel, dropped


def build_manifest(kept: pd.DataFrame) -> pd.DataFrame:
    """One row per rendered trial: identity, odor, figure name, trial metrics."""
    from fbpipe.odor_constants import canon_dataset
    from scripts.analysis.envelope_visuals import _display_odor

    fly_number = kept["fly_number"].map(_canon_fly_number)
    dataset_canon = kept["dataset"].astype(str).map(canon_dataset)
    out = pd.DataFrame(
        {
            "dataset": kept["dataset"].astype(str),
            "fly": kept["fly"].astype(str),
            "fly_number": fly_number,
            "trial_label": kept["trial_label"].astype(str),
            "odor": [
                _display_odor(dc, tl)
                for dc, tl in zip(
                    dataset_canon, kept["trial_label"].astype(str), strict=False
                )
            ],
            "figure": [
                f"{f}_fly{n}{FIGURE_SUFFIX}"
                for f, n in zip(kept["fly"].astype(str), fly_number, strict=False)
            ],
        },
        index=kept.index,
    )
    passthrough = [
        "fly_type",
        "fps",
        "trace_len",
        "trial_odor_on_s",
        "trial_odor_off_s",
        "trial_duration_s",
        "trial_light_on_s",
        "global_min",
        "global_max",
        "AUC-Before",
        "AUC-During",
        "AUC-After",
        "TimeToPeak-During",
        "Peak-Value",
    ]
    for col in passthrough:
        if col in kept.columns:
            out[col] = kept[col]
    return out.sort_values(["fly", "fly_number", "trial_label"]).reset_index(drop=True)


def odor_panel_count(manifest: pd.DataFrame) -> int:
    """Panels per fly = ODOR trials per fly, not trials per fly.

    ``generate_envelope_plots`` drops the ``testing_9`` light-only trial from the
    panels (it is not an odor trial), matching the pipeline's own Raw-Testing
    figures. Counting trials here would overstate the figure by one panel.
    """
    is_odor = ~manifest["trial_label"].astype(str).str.lower().str.contains("light")
    if not is_odor.any():
        return 0
    return int(manifest.loc[is_odor].groupby(["fly", "fly_number"]).size().max())


# ── pipeline-identical rendering ────────────────────────────────────────────
def _register_pipeline_context(config_path: Path) -> Mapping[str, Any]:
    """Mirror run_workflows' figure-time registries so labels match the pipeline.

    Without this the panels lose the v2 odor-suffix parsing and the dataset's
    ``odor_remap`` (ACV -> "Isoamyl Acetate (1%)", concentrations in every
    name), and the figures would disagree with every other figure in the repo.
    """
    from fbpipe.config import load_settings
    from scripts.analysis.envelope_visuals import (
        set_dataset_light_windows,
        set_dataset_odor_remap,
        set_light_check_fractions,
        set_model_scores,
        set_protocol,
    )

    settings = load_settings(config_path)
    set_protocol(settings.protocol)

    light_windows: dict[str, tuple[float, float]] = {}
    remap: dict[str, dict[str, str]] = {}
    for ds_name, ov in settings.dataset_overrides.items():
        mapping = getattr(ov, "odor_remap", None)
        if mapping:
            remap[str(ds_name)] = dict(mapping)
        if getattr(ov, "light_only", False):
            start, duration = ov.light_start_s, ov.light_duration_s
            if start is not None and duration is not None:
                light_windows[str(ds_name)] = (
                    float(start),
                    float(start) + float(duration),
                )
    set_dataset_odor_remap(remap)
    set_dataset_light_windows(light_windows)

    # Per-trial model score annotations (upper-right "Score: N"), same source
    # the pipeline uses. Absent CSV -> no annotation, which is the pipeline's
    # own fallback.
    scores: dict[tuple[str, str, str, str], int] = {}
    pred_csv = getattr(settings.reaction_prediction, "output_csv", "") or ""
    if pred_csv and Path(pred_csv).exists():
        pred = pd.read_csv(
            pred_csv, usecols=["dataset", "fly", "fly_number", "trial_label", "score"]
        )
        for row in pred.itertuples(index=False):
            try:
                scores[
                    (
                        str(row.dataset),
                        str(row.fly),
                        str(row.fly_number),
                        str(row.trial_label),
                    )
                ] = int(row.score)
            except (TypeError, ValueError):
                continue
    set_model_scores(scores)

    # Light-stimulus QC check marks on the light-only panel.
    from fbpipe.steps.check_light_stimulus import DEFAULT_CSV_PATH
    from scripts.analysis.envelope_visuals import load_light_check_fractions

    light_csv = Path(os.getenv("LIGHT_CHECK_CSV", str(DEFAULT_CSV_PATH)))
    set_light_check_fractions(load_light_check_fractions(light_csv))

    return {
        "protocol": settings.protocol,
        "flagged_flies_csv": str(getattr(settings, "flagged_flies_csv", "") or ""),
        "n_model_scores": len(scores),
        "odor_remap": remap.get(DEFAULT_DATASET, {}),
    }


def _envelope_config(slice_path: Path, out_dir: Path, *, overwrite: bool):
    """EnvelopePlotConfig with the pipeline's own style + window defaults.

    Values come from ``config_new.yaml``'s ``combined.combined_base.envelopes``
    entry and ``run_workflows._PIPELINE_ENVELOPE_STYLE_DEFAULTS``, so the output
    is stylistically identical to the Raw-Testing-PER-Traces figures.
    """
    from scripts.analysis.envelope_visuals import EnvelopePlotConfig
    from scripts.pipeline.run_workflows import _PIPELINE_ENVELOPE_STYLE_DEFAULTS

    return EnvelopePlotConfig(
        # Never opened while wide_input is set — kept only as a label hint.
        matrix_npy=Path(
            "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/matrix/raw/"
            "envelope_matrix_float16.npy"
        ),
        codes_json=Path(
            "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/matrix/raw/"
            "code_maps.json"
        ),
        wide_input=slice_path,
        out_dir=out_dir,
        latency_sec=0.0,
        fps_default=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
        odor_latency_s=2.15,
        after_show_sec=30.0,
        threshold_std_mult=2.0,
        trial_type="testing",
        overwrite=overwrite,
        **_PIPELINE_ENVELOPE_STYLE_DEFAULTS,
    )


def _flatten_dataset_subdir(out_dir: Path) -> int:
    """Move figures out of the per-dataset subfolder the renderer creates.

    ``generate_envelope_plots`` routes each fly into ``<out_dir>/<dataset>/``.
    This folder is already dataset-specific, so the extra level is noise.
    """
    moved = 0
    for sub in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        for fig in sorted(sub.rglob("*")):
            if fig.is_file():
                target = out_dir / fig.name
                shutil.move(str(fig), str(target))
                moved += 1
        shutil.rmtree(sub, ignore_errors=True)
    return moved


def _write_raw_traces(kept: pd.DataFrame, out_dir: Path) -> Path:
    """Write the raw dir_val_* envelope samples behind the figures."""
    env_cols = sorted(
        (c for c in kept.columns if str(c).startswith("dir_val_")),
        key=lambda c: int(str(c).split("_")[-1]),
    )
    id_cols = [
        c
        for c in ("dataset", "fly", "fly_number", "trial_label", "fps", "trace_len")
        if c in kept.columns
    ]
    # Trim to the longest real trace so ~12k mostly-empty columns don't ship.
    max_len = int(pd.to_numeric(kept.get("trace_len"), errors="coerce").max() or 0)
    if max_len > 0:
        env_cols = env_cols[:max_len]
    path = out_dir / "traces_raw.csv.gz"
    kept[id_cols + env_cols].to_csv(path, index=False, compression="gzip")
    return path


README = """# Removed rig_3 raw testing traces — {dataset}

Raw testing PER traces for the **{n_flies} non-flagged flies** in
`{dataset}` that the folder freeze removed from every figure.

## Why these flies are missing from the normal figures

`config/config_new.yaml` freezes `{pattern}` for this dataset:

```yaml
{dataset}:
  freeze:
    folders:
      - match: "{pattern}"
        before: 2026-08-11
```

rig_3 was miscalibrated until its 2026-08-11 repair, so every rig_3 batch
recorded before that date is excluded from the plots. The rows stay in the wide
CSV marked `frozen: true` — the CSV remains a complete record — but
`envelope_visuals._load_wide_table` drops them, so they appear in no figure
under `Results/New-Opto-Fly-Figures/Raw-Testing-PER-Traces/`.

This folder renders them anyway, in the pipeline's own raw-testing style, so the
removed data can be inspected.

## What is here

| File | Contents |
|---|---|
| `*_testing_envelope_trials_by_odor_30_shifted.png` / `.svg` | One figure per fly, {n_panels} stacked odor panels |
| `manifest.csv` | One row per trial: fly, fly number, trial, odor, figure, per-trial windows, AUC/peak |
| `excluded_flagged_flies.csv` | Frozen flies skipped because the flagged truth CSV marks them `FLY-State != 1` |
| `traces_raw.csv.gz` | The raw `dir_val_*` envelope samples behind the figures, one row per trial |

## Selection

* dataset `{dataset}`, folder pattern `{pattern}`, `frozen == true`
* `trial_type == testing` only
* flies **not** listed in `{flagged_csv}`
  ({n_flagged_flies} frozen fly/flies excluded on that basis)

**{n_flies} flies across {n_batches} batches, {n_trials} testing trials**
({n_panels} odor panels per fly plus the `testing_9` light-only trial, which the
renderer excludes from the panels — same as the pipeline figures — but which is
present in `manifest.csv` and `traces_raw.csv.gz`).

Batches: {batches}

## Reading the figures

Same conventions as `Raw-Testing-PER-Traces`: y is max distance × angle
(%, 0–105); the shaded band is the odor window, taken per trial from the rig
sidecar's ActiveOFM transitions (commanded 30–60 s); the red line is the
per-trial reaction threshold θ = median_before + 2.0 × MAD_before over the 30 s
baseline. Panels run in delivery order — `testing_1` and `testing_8` are the
trained odor (3-Octanol), `testing_2`–`testing_7` the randomised panel.

Odor display names carry this dataset's `odor_remap`: the rig's "ACV" channel
actually delivered isoamyl acetate, and every odor is labelled with its
concentration.

## Reproduce

```bash
python scripts/analysis/frozen_folder_testing_traces.py \\
    --dataset {dataset} --pattern '{pattern}' \\
    --out-dir {out_dir}
```

Generated {stamp} from `{wide}` (protocol {protocol}).
"""


def generate(
    *,
    wide_path: Path = DEFAULT_WIDE,
    out_dir: Path = DEFAULT_OUT_DIR,
    dataset: str = DEFAULT_DATASET,
    pattern: str = DEFAULT_PATTERN,
    config_path: Path = DEFAULT_CONFIG,
    overwrite: bool = True,
    svg: bool = True,
    raw_csv: bool = True,
) -> dict[str, Any]:
    """Render the frozen flies' raw testing traces into *out_dir*."""
    from fbpipe.utils.tables import read_table

    context = _register_pipeline_context(config_path)
    df = read_table(Path(wide_path))
    kept, dropped = select_rows(
        df,
        dataset=dataset,
        pattern=pattern,
        flagged_flies_csv=context["flagged_flies_csv"],
    )

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if svg:
        from fbpipe.figure_export import install_svg_sidecar

        install_svg_sidecar()

    # The renderer reads a wide table off disk; hand it only the selected rows
    # so no live fly can leak into this folder.
    tmp_dir = Path(tempfile.mkdtemp(prefix="frozen-traces-"))
    try:
        slice_path = tmp_dir / "frozen_slice.parquet"
        kept.to_parquet(slice_path, index=False)

        from scripts.analysis.envelope_visuals import generate_envelope_plots

        generate_envelope_plots(_envelope_config(slice_path, out_dir, overwrite=overwrite))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    n_moved = _flatten_dataset_subdir(out_dir)

    manifest = build_manifest(kept)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    flagged_rows = [
        {"dataset": d, "fly": f, "fly_number": n} for d, f, n in dropped["flagged"]
    ]
    pd.DataFrame(
        flagged_rows or [], columns=["dataset", "fly", "fly_number"]
    ).to_csv(out_dir / "excluded_flagged_flies.csv", index=False)

    if raw_csv:
        _write_raw_traces(kept, out_dir)

    n_flies = manifest.groupby(["fly", "fly_number"]).ngroups
    batches = sorted(manifest["fly"].unique())
    n_panels = odor_panel_count(manifest)
    (out_dir / "README.md").write_text(
        README.format(
            dataset=dataset,
            pattern=pattern,
            n_flies=n_flies,
            n_batches=len(batches),
            n_trials=len(manifest),
            n_panels=n_panels,
            n_flagged_flies=len(flagged_rows),
            flagged_csv=context["flagged_flies_csv"] or "(none)",
            batches=", ".join(f"`{b}`" for b in batches),
            out_dir=out_dir,
            wide=wide_path,
            protocol=context["protocol"],
            stamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        ),
        encoding="utf-8",
    )

    n_figs = len(list(out_dir.glob(f"*{FIGURE_SUFFIX}")))
    return {
        "out_dir": out_dir,
        "n_flies": n_flies,
        "n_trials": len(manifest),
        "n_figures": n_figs,
        "n_moved": n_moved,
        "flagged_excluded": flagged_rows,
        "batches": batches,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wide", type=Path, default=DEFAULT_WIDE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--pattern", default=DEFAULT_PATTERN)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--no-svg", action="store_true")
    p.add_argument("--no-raw-csv", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate(
        wide_path=args.wide,
        out_dir=args.out_dir,
        dataset=args.dataset,
        pattern=args.pattern,
        config_path=args.config,
        overwrite=not args.no_overwrite,
        svg=not args.no_svg,
        raw_csv=not args.no_raw_csv,
    )
    print(
        f"[frozen-traces] {result['n_figures']} figure(s), {result['n_flies']} fly/flies, "
        f"{result['n_trials']} trial(s) → {result['out_dir']}"
    )
    for row in result["flagged_excluded"]:
        print(f"[frozen-traces] flagged, excluded: {row['fly']} fly{row['fly_number']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
