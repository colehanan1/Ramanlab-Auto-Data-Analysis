"""Reaction-matrix comparison for a hand-picked subset of training flies vs.
the full control cohort, OR the full cohort minus excluded flies/rigs.

Same visual output as ``reaction_matrix_training_vs_control.py`` (bar chart of
PER% per odor with Fisher's exact test brackets, plus a control|training
during-reaction heatmap pair).

Two independent, combinable selection mechanisms:
  - ``--fly`` restricts the TRAINING side to a caller-supplied allow list of
    ``(fly, fly_number)`` pairs; control always uses the full control cohort.
  - ``--exclude-fly`` / ``--exclude-pattern`` drop matching flies from BOTH
    training and control (e.g. a bad recording rig affects every dataset it
    touches, not just the training arm).

Rates/Fisher stats are computed directly from the (fly-filtered) predictions
CSV rather than from the pipeline's precomputed ``binary_reactions_*.csv``,
so the bar chart and the heatmap panel are guaranteed consistent with each
other for the selected subset.

Usage (hand-picked training subset)::

    python scripts/analysis/reaction_matrix_specific_flies_vs_control.py \
        --csv-path /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv \
        --train-dataset EB-Training-24-1 \
        --control-dataset EB-Control-24-1 \
        --fly july_13_batch_1:1 --fly july_13_batch_1:2 ... \
        --out-dir /home/ramanlab/Documents/cole/Results/Figures/EB-Training-24-1_specific15_vs_control

Usage (exclude a bad rig/batch from both sides)::

    python scripts/analysis/reaction_matrix_specific_flies_vs_control.py \
        --csv-path /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv \
        --train-dataset EB-Training-24-1 \
        --control-dataset EB-Control-24-1 \
        --exclude-pattern rig_3 --exclude-pattern july_17_batch_2_rig_2 \
        --out-dir /home/ramanlab/Documents/cole/Results/Figures/EB-Training-24-1_excl_rig3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbpipe.utils.tables import read_table
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import (
    DISPLAY_LABEL,
    get_protocol,
    set_protocol,
    _canon_dataset,
    _matrix_title,
    _normalise_fly_columns,
    _style_trained_xticks,
    _trained_label,
    _trial_num,
    compute_non_reactive_flags,
)
from scripts.analysis.reaction_matrix_from_spreadsheet import (
    _filter_trial_types,
    _normalise_trial_label,
)
from scripts.analysis.reaction_matrix_training_vs_control import (
    _RC_CONTEXT,
    _build_during_matrix,
    _draw_significance_brackets,
    plot_training_vs_control_bars,
)


def _parse_fly_arg(raw: str) -> tuple[str, str]:
    fly, sep, fly_number = raw.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"--fly value {raw!r} must be 'fly_folder_name:fly_number'"
        )
    return fly.strip(), fly_number.strip()


def _rates_from_matrix(matrix: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Per-column reaction rate/trial-count directly off a during-hit matrix."""
    rows = []
    for j, col in enumerate(columns):
        colvals = matrix[:, j] if matrix.size else np.empty(0)
        finite = colvals[~np.isnan(colvals)]
        total = int(finite.size)
        reactions = int(np.nansum(finite)) if total else 0
        rate = (reactions / total * 100.0) if total else 0.0
        rows.append(
            {"odor": col, "num_trials": total, "num_reactions": reactions, "rate": rate}
        )
    return pd.DataFrame(rows)


def _resolve_non_reactive_mask(df: pd.DataFrame, flagged_flies_csv: str) -> pd.Series:
    """Non-reactive/excluded-fly mask.

    A ``flagged_flies_csv`` truth table, when given, is authoritative and is
    always recomputed — even if the predictions CSV already carries a
    ``_non_reactive`` column, since that column may have been computed
    without the truth table (e.g. all False) and would otherwise silently
    shadow it. The existing column is only trusted as a shortcut when no
    truth table is supplied.
    """
    if flagged_flies_csv:
        return compute_non_reactive_flags(df, flagged_flies_csv=flagged_flies_csv)
    if "_non_reactive" in df.columns:
        return df["_non_reactive"].fillna(False).astype(bool)
    return compute_non_reactive_flags(df)


def _exclusion_mask(
    df: pd.DataFrame,
    exclude_flies: set[tuple[str, str]],
    exclude_patterns: Sequence[str],
) -> pd.Series:
    """Rows whose ``(fly, fly_number)`` is in ``exclude_flies``, or whose
    ``fly`` folder name contains any of ``exclude_patterns`` (case-insensitive
    substring). Applies to every dataset in ``df`` — unlike ``--fly``
    (training-only allow list), exclusions drop matching flies everywhere,
    since a bad rig/batch affects both arms of a training-vs-control figure.
    """
    mask = pd.Series(False, index=df.index)
    if df.empty:
        return mask
    fly_cf = df["fly"].astype(str).str.casefold()
    for pat in exclude_patterns:
        mask |= fly_cf.str.contains(pat.casefold(), regex=False)
    if exclude_flies:
        fly_key = pd.Series(list(zip(df["fly"], df["fly_number"])), index=df.index)
        mask |= fly_key.isin(exclude_flies)
    return mask


def _fisher_per_column(
    train_matrix: np.ndarray, ctrl_matrix: np.ndarray, columns: list[str]
) -> dict[tuple[int, str], float]:
    p_values: dict[tuple[int, str], float] = {}
    for j, col in enumerate(columns):
        t = train_matrix[:, j] if train_matrix.size else np.empty(0)
        c = ctrl_matrix[:, j] if ctrl_matrix.size else np.empty(0)
        t_finite = t[~np.isnan(t)]
        c_finite = c[~np.isnan(c)]
        if t_finite.size == 0 or c_finite.size == 0:
            p_values[(0, col)] = np.nan
            continue
        t_react, t_total = int(np.nansum(t_finite)), int(t_finite.size)
        c_react, c_total = int(np.nansum(c_finite)), int(c_finite.size)
        table = [[t_react, t_total - t_react], [c_react, c_total - c_react]]
        _, p = fisher_exact(table)
        p_values[(0, col)] = p
    return p_values


def build_parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv-path", type=Path, required=True,
                    help="Path to model_predictions.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-dataset", required=True,
                    help="Training dataset name as it appears in the predictions CSV")
    p.add_argument("--control-dataset", required=True,
                    help="Control dataset name as it appears in the predictions CSV")
    p.add_argument("--fly", action="append", dest="flies", default=None,
                    type=_parse_fly_arg,
                    help="fly_folder_name:fly_number (repeatable) — restrict TRAINING "
                         "to only these flies (control is untouched). Omit when using "
                         "--exclude-fly/--exclude-pattern instead.")
    p.add_argument("--exclude-fly", action="append", dest="exclude_flies", default=[],
                    type=_parse_fly_arg,
                    help="fly_folder_name:fly_number (repeatable) — drop this exact fly "
                         "from BOTH training and control.")
    p.add_argument("--exclude-pattern", action="append", dest="exclude_patterns", default=[],
                    help="Case-insensitive substring matched against the fly folder name "
                         "(repeatable) — drop any matching fly from BOTH training and "
                         "control, e.g. 'rig_3'.")
    p.add_argument("--flagged-flies-csv", type=str, default="",
                    help="Path to the flagged-flies truth CSV (FLY-State != 1 excluded). "
                         "Authoritative when given — overrides any stale _non_reactive "
                         "column already in the predictions CSV. Same convention as "
                         "reaction_matrix_training_vs_control.py --flagged-flies-csv.")
    p.add_argument("--latency-sec", type=float, default=2.15)
    p.add_argument("--after-window-sec", type=float, default=30.0)
    p.add_argument("--protocol", default="v2", choices=["v2", "legacy"])
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser(argv)
    set_protocol(args.protocol)

    specific_flies = set(args.flies) if args.flies else set()
    exclude_flies = set(args.exclude_flies)
    exclude_patterns = list(args.exclude_patterns)
    if not specific_flies and not exclude_flies and not exclude_patterns:
        raise ValueError(
            "Nothing to select: pass --fly (training allow list) and/or "
            "--exclude-fly/--exclude-pattern (drop from both sides)."
        )

    df = read_table(args.csv_path)
    required = {"dataset", "fly", "fly_number", "trial_label", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = _filter_trial_types(df, allowed=("testing",))
    if df.empty:
        raise RuntimeError("No testing trials found in predictions CSV.")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["trial_label"] = df["trial_label"].astype(str).str.strip()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = _normalise_fly_columns(df)

    flagged_mask = _resolve_non_reactive_mask(df, args.flagged_flies_csv)
    if flagged_mask.any():
        flagged = df.loc[flagged_mask, ["dataset", "fly", "fly_number"]].drop_duplicates()
        summaries = ", ".join(
            f"{r.dataset}::{r.fly}::{r.fly_number}" for r in flagged.itertuples(index=False)
        )
        print(f"[INFO] Excluding non-reactive/flagged flies: {summaries}")
        df = df.loc[~flagged_mask].copy()

    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)
    df["during_hit"] = df["prediction"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["dataset", "fly", "fly_number", "trial"], keep="first")
    df = _normalise_fly_columns(df)

    train_canon = _canon_dataset(args.train_dataset)
    ctrl_canon = _canon_dataset(args.control_dataset)

    n_dropped_flies = 0
    if exclude_flies or exclude_patterns:
        excluded_mask = _exclusion_mask(df, exclude_flies, exclude_patterns)
        if excluded_mask.any():
            dropped = df.loc[excluded_mask, ["dataset_canon", "fly", "fly_number"]].drop_duplicates()
            n_dropped_flies = len(dropped)
            summaries = ", ".join(
                f"{r.dataset_canon}::{r.fly}::{r.fly_number}" for r in dropped.itertuples(index=False)
            )
            print(f"[INFO] Excluding {n_dropped_flies} fly rows (rig/pattern exclusion): {summaries}")
        df = df.loc[~excluded_mask].copy()

    is_train_row = df["dataset_canon"] == train_canon
    fly_key = pd.Series(list(zip(df["fly"], df["fly_number"])), index=df.index)

    if specific_flies:
        in_selection = fly_key.isin(specific_flies)
        available = set(fly_key[is_train_row])
        missing_flies = specific_flies - available
        if missing_flies:
            print(
                f"[WARN] {len(missing_flies)} requested flies not found in "
                f"{args.train_dataset} (after non-reactive/exclusion filtering): {sorted(missing_flies)}"
            )
        df = df.loc[(~is_train_row) | in_selection].copy()
    else:
        missing_flies = set()

    n_train_flies = (
        df.loc[df["dataset_canon"] == train_canon, ["fly", "fly_number"]]
        .drop_duplicates()
        .shape[0]
    )
    n_ctrl_flies_resolved = (
        df.loc[df["dataset_canon"] == ctrl_canon, ["fly", "fly_number"]]
        .drop_duplicates()
        .shape[0]
    )
    if specific_flies:
        print(f"[INFO] Training subset resolved to {n_train_flies}/{len(specific_flies)} flies")
    else:
        print(f"[INFO] Training resolved to {n_train_flies} flies, control to {n_ctrl_flies_resolved} flies")

    during_matrix, fly_pairs, odor_columns, flagged_pairs = _build_during_matrix(
        df, train_canon, None, remap_from=train_canon, order="observed"
    )
    if not len(fly_pairs):
        raise RuntimeError("No usable training rows after filtering to the requested flies.")
    ctrl_matrix, ctrl_fly_pairs, _ctrl_cols, ctrl_flagged_pairs = _build_during_matrix(
        df, ctrl_canon, None, remap_from=train_canon, columns=odor_columns, order="observed"
    )

    trained_display = _trained_label(train_canon)
    train_rate = _rates_from_matrix(during_matrix, odor_columns)
    train_rate["is_trained"] = (
        train_rate["odor"].astype(str).str.casefold().str.startswith(trained_display.casefold())
    )
    ctrl_rate = _rates_from_matrix(ctrl_matrix, odor_columns)

    p_values = _fisher_per_column(during_matrix, ctrl_matrix, odor_columns)
    for (_, odor), p in p_values.items():
        if pd.isna(p):
            continue
        print(f"  Fisher's exact: {odor:25s} p={p:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    odor_label = DISPLAY_LABEL.get(train_canon, train_canon)
    n_trials = len(odor_columns)
    base_w = max(10.0, 0.70 * n_trials + 6.0)
    xtick_fs = 9 if n_trials <= 10 else (8 if n_trials <= 16 else 7)

    if specific_flies:
        subset_tag = f"specific{len(specific_flies)}"
        bar_title = f"Reaction Rates – {odor_label} ({len(specific_flies)} selected flies vs Control)"
        train_panel_title = f"Selected Training ({during_matrix.shape[0]} Flies)"
    else:
        subset_tag = f"excl{n_dropped_flies}"
        bar_title = f"Reaction Rates – {odor_label} (Training vs Control, {n_dropped_flies} flies excluded)"
        train_panel_title = f"Training ({during_matrix.shape[0]} Flies)"

    png_stub = (
        f"reaction_matrix_train_vs_ctrl_{args.train_dataset.replace(' ', '_')}"
        f"_{subset_tag}_{int(args.after_window_sec)}_latency_"
        f"{args.latency_sec:.3f}s"
    )

    with plt.rc_context(_RC_CONTEXT):
        # --- Figure A: bar chart, training-subset vs full control ---
        fig_bar, ax_bar = plt.subplots(figsize=(base_w, 5.0))
        plot_training_vs_control_bars(
            ax_bar, train_rate, ctrl_rate,
            title=bar_title,
            p_values=p_values,
        )
        bar_path = args.out_dir / f"{png_stub}.png"
        fig_bar.savefig(bar_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {bar_path}")
        plt.close(fig_bar)

        # --- Figure B: control (left) | training-subset (right) matrix pair ---
        cmap = ListedColormap(["white", "black"])
        cmap.set_bad(color="0.7")
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        n_ctrl = max(1, ctrl_matrix.shape[0])
        n_train = max(1, during_matrix.shape[0])
        cell_h = 0.26
        pair_h = max(4.0, max(n_ctrl, n_train) * cell_h + 3.0)
        fig_pair = plt.figure(figsize=(base_w * 1.15, pair_h))
        gs_pair = gridspec.GridSpec(1, 2, wspace=0.12)
        ax_c = fig_pair.add_subplot(gs_pair[0, 0])
        ax_t = fig_pair.add_subplot(gs_pair[0, 1])
        for ax, mat, title, pairs, flagged in (
            (ax_c, ctrl_matrix, f"Control ({ctrl_matrix.shape[0]} Flies)",
             ctrl_fly_pairs, ctrl_flagged_pairs),
            (ax_t, during_matrix, train_panel_title,
             fly_pairs, flagged_pairs),
        ):
            if mat.size:
                ax.imshow(
                    mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
                    extent=(-0.5, len(odor_columns) - 0.5, mat.shape[0] - 0.5, -0.5),
                )
                ax.set_ylim(max(n_ctrl, n_train) - 0.5, -0.5)
            _style_trained_xticks(ax, list(odor_columns), trained_display, xtick_fs)
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels([f"{f} fly{n}" for f, n in pairs], fontsize=6)
            ax.set_title(title, fontsize=12, weight="bold")
            for idx, pair in enumerate(pairs):
                if pair in flagged:
                    ax.text(-0.35, idx, "*", ha="right", va="center", color="red",
                            fontsize=12, fontweight="bold", clip_on=False)
        fig_pair.suptitle(_matrix_title(train_canon), fontsize=14, weight="bold")
        pair_stub = png_stub.replace(
            "reaction_matrix_train_vs_ctrl_", "reaction_matrix_pair_"
        )
        pair_path = args.out_dir / f"{pair_stub}.png"
        fig_pair.savefig(pair_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {pair_path}")
        plt.close(fig_pair)

    sidecar = {
        "train_dataset": args.train_dataset,
        "control_dataset": args.control_dataset,
        "requested_flies": [f"{f}:{n}" for f, n in sorted(specific_flies)],
        "missing_flies": [f"{f}:{n}" for f, n in sorted(missing_flies)],
        "exclude_flies": [f"{f}:{n}" for f, n in sorted(exclude_flies)],
        "exclude_patterns": exclude_patterns,
        "n_dropped_flies": n_dropped_flies,
        "n_training_flies_used": n_train_flies,
        "n_control_flies": ctrl_matrix.shape[0],
        "odor_columns": list(odor_columns),
        "latency_sec": args.latency_sec,
        "after_window_sec": args.after_window_sec,
    }
    sidecar_path = args.out_dir / f"{png_stub}.json"
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"[SAVED] {sidecar_path}")


if __name__ == "__main__":
    main()
