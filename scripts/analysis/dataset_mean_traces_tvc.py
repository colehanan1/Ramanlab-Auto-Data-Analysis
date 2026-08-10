"""Trained-vs-control mean traces, one figure per odor *presentation*.

Same figure as ``Matrix-PER-Reactions-Model/Training_vs_Control/`` — trained
and control cohort means with SEM bands, baseline-subtracted, odor window
shaded — but the two presentations of a repeated odor get their own figures
instead of being collapsed to the first.

That matters for the v2 testing schedule, where the trained odor is presented
twice (``testing_1`` and ``testing_8``) with every other odor in between: the
second presentation is the one that shows whether the response survived the
rest of the panel. ``dataset_means_specific_flies`` keeps only the first, so
it cannot draw it.

Odor names come from the same pipeline the reaction matrices use — per-dataset
``odor_remap`` from the config, so ACV reads "Isoamyl Acetate (1%)" — and a
repeated odor is suffixed " 1" / " 2" in presentation order, matching the
reaction-matrix column labels exactly.

Usage::

    python scripts/analysis/dataset_mean_traces_tvc.py \
        --wide-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/all_envelope_rows_wide_combined_base.parquet \
        --train-dataset 3Oct-Training-24-0.1 \
        --control-dataset 3Oct-Control-24-0.1 \
        --config config/config_new.yaml \
        --flagged-flies-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/flagged-flys-truth.csv \
        --out-dir /home/ramanlab/Documents/cole/Results/Figures/3Oct-24-0.1_mean_traces/Training_vs_Control
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.analysis.traces import baseline_correct, read_wide_table  # noqa: E402
from scripts.analysis.dataset_means_specific_flies import (  # noqa: E402
    DPI,
    MAX_TIME_S,
    _mean_trace,
    _plot_training_vs_control_for_odor,
    _safe_odor_filename,
    _shared_ylim_from_means,
)
from scripts.analysis.rig_batch_breakdowns import batch_of  # noqa: E402
from scripts.analysis.mean_trace_score import (  # noqa: E402
    annotate_mean_scores,
    model_settings_from_config,
    score_group_means,
)
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _canon_dataset,
    _display_label_ci,
    _extract_odor_from_label,
    _is_light_only_label,
    _normalise_fly_columns,
    _trial_num,
    apply_dataset_odor_remap,
    compute_non_reactive_flags,
    set_protocol,
)

LOGGER = logging.getLogger("dataset_mean_traces_tvc")

# "3-Octanol (0.1%) 2" -> drop " 2", then drop " (0.1%)".
_PRESENTATION_SUFFIX = re.compile(r"\s+\d+$")
_CONCENTRATION_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def base_odor_key(label: str) -> str:
    """Bare odor name behind a display label, for colour lookup.

    ``ODOR_COLOURS`` is keyed on plain names ("Hexanol"), so a remapped,
    concentration-tagged, presentation-numbered label would miss it and fall
    through to the grey default — and the two 3-octanol figures would not even
    match each other.
    """
    stripped = _PRESENTATION_SUFFIX.sub("", str(label).strip())
    return _CONCENTRATION_SUFFIX.sub("", stripped).strip()


def collect_per_presentation_traces(
    ds_df: pd.DataFrame,
    dataset_canon: str,
    *,
    baseline_frames: int,
    dir_cols: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Per-fly baseline-corrected traces keyed by odor **and presentation**.

    Presentation order is resolved per fly from the trial number, so a shuffled
    input frame still calls ``testing_1`` presentation 1. Only odors that
    actually repeat get a numeric suffix; a once-presented odor keeps its plain
    display name, as in the published figures.
    """
    return collect_presentations(
        ds_df, dataset_canon, baseline_frames=baseline_frames, dir_cols=dir_cols
    )[0]


def collect_presentations(
    ds_df: pd.DataFrame,
    dataset_canon: str,
    *,
    baseline_frames: int,
    dir_cols: Sequence[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list]]:
    """``(traces, source_row_index)`` keyed by odor-and-presentation.

    The second mapping gives the wide-table rows behind each group's mean, so
    the same rows can be averaged into a synthetic row and scored by the model.
    """
    # (fly, fly_number) -> odor -> {occurrence: (trace, row_index)}
    per_fly_odor: dict[tuple[str, int], dict[str, dict[int, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    if ds_df.empty:
        return {}, {}

    dir_cols = list(dir_cols)
    frame = ds_df.copy()
    frame["_trial_num"] = frame["trial_label"].astype(str).map(_trial_num)

    for fly_key, fly_rows in frame.groupby(["fly", "fly_number"], sort=False):
        seen: dict[str, int] = {}
        for idx, row in fly_rows.sort_values("_trial_num").iterrows():
            trial_label = str(row["trial_label"])
            if _is_light_only_label(trial_label):
                continue
            raw_odor = _extract_odor_from_label(trial_label)
            if raw_odor == trial_label:      # no parseable odor token
                continue
            if int(row["_trial_num"]) < 0:
                continue
            odor = apply_dataset_odor_remap(
                dataset_canon, _display_label_ci(raw_odor)
            )
            trace = row[dir_cols].to_numpy(dtype=np.float64)
            finite = np.isfinite(trace)
            if not finite.any():
                continue
            trace = baseline_correct(
                trace[: np.where(finite)[0][-1] + 1], baseline_frames
            )
            occurrence = seen.get(odor, 0) + 1
            seen[odor] = occurrence
            per_fly_odor[(str(fly_key[0]), fly_key[1])][odor].setdefault(
                occurrence, (trace, idx)
            )

    max_occurrence: dict[str, int] = defaultdict(int)
    for odor_map in per_fly_odor.values():
        for odor, occ_map in odor_map.items():
            max_occurrence[odor] = max(max_occurrence[odor], max(occ_map))

    per_key: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    rows_by_key: dict[str, list] = defaultdict(list)
    for (fly, fly_number), odor_map in per_fly_odor.items():
        fly_id = f"{fly}_fly{fly_number}"
        for odor, occ_map in odor_map.items():
            numbered = max_occurrence[odor] > 1
            for occurrence, (trace, idx) in occ_map.items():
                key = f"{odor} {occurrence}" if numbered else odor
                per_key[key][fly_id] = trace
                rows_by_key[key].append(idx)
    return dict(per_key), dict(rows_by_key)


def _sort_key(label: str) -> tuple[str, int]:
    m = _PRESENTATION_SUFFIX.search(label)
    return (base_odor_key(label).casefold(), int(m.group().strip()) if m else 0)


def _prepare(
    wide_df: pd.DataFrame,
    dataset: str,
    *,
    fps: float,
    odor_on_s: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list], pd.DataFrame]:
    ds_df = wide_df[wide_df["dataset"].astype(str).str.strip() == dataset]
    if "trial_type" in ds_df.columns:
        ds_df = ds_df[ds_df["trial_type"].astype(str).str.strip() == "testing"]
    if ds_df.empty:
        LOGGER.warning("No testing rows for %s", dataset)
        return {}, {}, ds_df
    dir_cols = sorted(
        [c for c in ds_df.columns if c.startswith("dir_val_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    per_odor, rows_by_key = collect_presentations(
        ds_df,
        _canon_dataset(dataset),
        baseline_frames=max(1, int(round(odor_on_s * fps))),
        dir_cols=dir_cols,
    )
    for odor in sorted(per_odor, key=_sort_key):
        LOGGER.info("  %-28s n=%d", odor, len(per_odor[odor]))
    return per_odor, rows_by_key, ds_df


def _apply_config_remap(config_path: str) -> None:
    from fbpipe.config import load_settings

    from scripts.analysis.envelope_visuals import set_dataset_odor_remap

    settings = load_settings(config_path)
    remap = {
        str(ds): dict(ov.odor_remap)
        for ds, ov in settings.dataset_overrides.items()
        if getattr(ov, "odor_remap", None)
    }
    if remap:
        set_dataset_odor_remap(remap)
        LOGGER.info("Loaded odor_remap for %d datasets from %s", len(remap), config_path)


def build_parser(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--wide-csv", type=Path, required=True,
                   help="all_envelope_rows_wide_combined_base CSV or Parquet.")
    p.add_argument("--train-dataset", required=True)
    p.add_argument("--control-dataset", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--config", type=str, default="",
                   help="Pipeline config YAML; loads dataset_overrides.odor_remap.")
    p.add_argument("--flagged-flies-csv", type=str, default="",
                   help="flagged-flys-truth CSV (FLY-State != 1 excluded).")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--odor-on-s", type=float, default=30.0)
    p.add_argument("--odor-off-s", type=float, default=60.0)
    p.add_argument("--batch", type=int, default=None,
                   help="Restrict both arms to one starvation batch "
                        "(the batch_N token in the fly folder name).")
    p.add_argument("--protocol", default="v2", choices=["v2", "legacy"])
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--score-mean-trace", action="store_true", default=False,
                   help="Score each cohort's mean trace with the ordinal PER "
                        "model and print it on the figure. Needs --model-path "
                        "or a --config that sets reaction_prediction.model_path.")
    p.add_argument("--model-path", type=str, default="",
                   help="Ordinal model JSON; overrides the config's model_path.")
    p.add_argument("--binary-threshold", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _resolve_model(args) -> tuple[Path, int] | None:
    """``(model_path, binary_threshold)`` for mean-trace scoring, or None."""
    if not args.score_mean_trace:
        return None
    model_path, threshold = None, args.binary_threshold
    if args.model_path:
        model_path = Path(args.model_path)
    elif args.config:
        model_path, cfg_threshold = model_settings_from_config(args.config)
        if threshold is None:
            threshold = cfg_threshold
    if model_path is None:
        raise SystemExit(
            "--score-mean-trace needs --model-path or a --config carrying "
            "reaction_prediction.model_path"
        )
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    return model_path, int(threshold if threshold is not None else 2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    set_protocol(args.protocol)
    if args.config:
        _apply_config_remap(args.config)

    wide_df = read_wide_table(args.wide_csv)
    LOGGER.info("Loaded %d rows from %s", len(wide_df), args.wide_csv)
    wide_df = _normalise_fly_columns(wide_df)

    if args.flagged_flies_csv:
        flagged = compute_non_reactive_flags(
            wide_df, flagged_flies_csv=args.flagged_flies_csv
        )
        if flagged.any():
            dropped = wide_df.loc[flagged, ["dataset", "fly", "fly_number"]].drop_duplicates()
            in_scope = dropped[dropped["dataset"].isin(
                [args.train_dataset, args.control_dataset]
            )]
            LOGGER.info(
                "Excluding %d flagged flies (%d in this pair)", len(dropped), len(in_scope)
            )
            wide_df = wide_df.loc[~flagged].copy()

    if args.batch is not None:
        keep = wide_df["fly"].map(batch_of) == args.batch
        LOGGER.info(
            "Batch %d: keeping %d of %d rows", args.batch, int(keep.sum()), len(wide_df)
        )
        wide_df = wide_df.loc[keep].copy()
        if wide_df.empty:
            raise RuntimeError(f"No flies in batch {args.batch}")

    LOGGER.info("=== %s", args.train_dataset)
    train, train_rows, train_df = _prepare(
        wide_df, args.train_dataset, fps=args.fps, odor_on_s=args.odor_on_s
    )
    LOGGER.info("=== %s", args.control_dataset)
    control, ctrl_rows, ctrl_df = _prepare(
        wide_df, args.control_dataset, fps=args.fps, odor_on_s=args.odor_on_s
    )
    if not train or not control:
        raise RuntimeError(
            "No usable traces for "
            f"{args.train_dataset if not train else args.control_dataset}"
        )

    # --- optional: score each cohort mean trace with the ordinal model -------
    mean_scores: dict[str, dict[str, float]] = {}
    model = _resolve_model(args)
    if model is not None:
        model_path, binary_threshold = model
        groups: dict[str, pd.DataFrame] = {}
        for arm, rows_by_key, source in (
            ("Trained", train_rows, train_df), ("Control", ctrl_rows, ctrl_df)
        ):
            for odor, idx in rows_by_key.items():
                groups[f"{arm}|{odor}"] = source.loc[idx]
        LOGGER.info("Scoring %d cohort-mean traces with %s", len(groups), model_path)
        mean_scores = score_group_means(
            groups, model_path=model_path, binary_threshold=binary_threshold
        )

    max_frames = int(MAX_TIME_S * args.fps)
    shared_ylim = _shared_ylim_from_means(
        [_mean_trace(pf, max_frames) for pf in (*train.values(), *control.values())],
        fps=args.fps,
    )
    LOGGER.info("Shared mean ylim: %s", shared_ylim)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_odor_meta: dict[str, dict[str, int]] = {}
    skipped: list[str] = []

    for odor in sorted(set(train) | set(control), key=_sort_key):
        train_per_fly = train.get(odor, {})
        ctrl_per_fly = control.get(odor, {})
        if not train_per_fly or not ctrl_per_fly:
            LOGGER.warning(
                "Skipping %s — trained n=%d, control n=%d",
                odor, len(train_per_fly), len(ctrl_per_fly),
            )
            skipped.append(odor)
            continue
        fig = _plot_training_vs_control_for_odor(
            odor=odor,
            train_per_fly=train_per_fly,
            ctrl_per_fly=ctrl_per_fly,
            fps=args.fps,
            odor_on_s=args.odor_on_s,
            odor_off_s=args.odor_off_s,
            ylim=shared_ylim,
            color_key=base_odor_key(odor),
        )
        entry = {
            "n_training_flies": len(train_per_fly),
            "n_control_flies": len(ctrl_per_fly),
        }
        train_score = mean_scores.get(f"Trained|{odor}")
        ctrl_score = mean_scores.get(f"Control|{odor}")
        if train_score is not None and ctrl_score is not None:
            annotate_mean_scores(fig.axes[0], train_score, ctrl_score)
            entry["mean_trace_score_training"] = train_score["score"]
            entry["mean_trace_score_control"] = ctrl_score["score"]
            entry["mean_trace_reacted_training"] = train_score["prediction"]
            entry["mean_trace_reacted_control"] = ctrl_score["prediction"]

        out_png = args.out_dir / f"{_safe_odor_filename(odor)}_training_vs_control.png"
        if args.overwrite or not out_png.exists():
            fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
            LOGGER.info("Saved %s", out_png)
        plt.close(fig)
        per_odor_meta[odor] = entry

    sidecar = {
        "training_dataset": args.train_dataset,
        "control_dataset": args.control_dataset,
        "fps": args.fps,
        "odor_on_s": args.odor_on_s,
        "odor_off_s": args.odor_off_s,
        "shared_mean_ylim": list(shared_ylim),
        "batch": args.batch,
        "flagged_flies_csv": args.flagged_flies_csv,
        "per_odor": per_odor_meta,
        "skipped_odors": skipped,
    }
    sidecar_path = args.out_dir / "training_vs_control.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", sidecar_path)


if __name__ == "__main__":
    main()
