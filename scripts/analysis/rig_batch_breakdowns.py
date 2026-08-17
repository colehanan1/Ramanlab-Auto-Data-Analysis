"""Rig- and batch-wise breakdowns of one training/control pair.

Splits a training/control dataset pair along the two nuisance factors that are
baked into the fly folder name — which recording rig, and which starvation
batch — and draws the same four figures for every split:

* ``reaction_matrix_<tag>_<window>_latency_<lat>s.png``  PER% bars + Fisher's exact
* ``reaction_matrix_pair_<tag>_...png``                  per-fly binary reaction panels
* ``mean_score_<tag>.png``                               mean ordinal score bars + Mann-Whitney
* ``mean_score_pair_<tag>.png``                          per-fly score panels

plus a JSON sidecar per statistic carrying the group sizes and per-odor p-values.

Three families of comparison are emitted:

``tvc_rig_2``               training vs control, restricted to rig 2
``train_rig_1_vs_rig_3``    training arm only, rig 1 vs rig 3
``ctrl_batch_1_vs_batch_2`` control arm only, batch 1 vs batch 2

The rig lives in the fly folder name as a ``_rig_N`` suffix and **rig 1 is
implicit** (``july_20_batch_1`` is a rig-1 fly, ``july_20_batch_1_rig_2`` is a
rig-2 fly). The batch is the ``batch_N`` token. Both arms are always filtered
by the same rule, since a bad rig affects training and control alike.

Usage::

    python scripts/analysis/rig_batch_breakdowns.py \
        --csv-path /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv \
        --train-dataset 3Oct-Training-24-0.1 \
        --control-dataset 3Oct-Control-24-0.1 \
        --config config/config_new.yaml \
        --flagged-flies-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/flagged-flys-truth.csv \
        --out-dir /home/ramanlab/Documents/cole/Results/Figures/3Oct-24-0.1_rig_batch_breakdowns
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fbpipe.utils.tables import read_table  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import (  # noqa: E402
    DISPLAY_LABEL,
    _canon_dataset,
    _matrix_title,
    _normalise_fly_columns,
    _style_trained_xticks,
    _trained_label,
    _trial_num,
    set_protocol,
)
from scripts.analysis.reaction_matrix_from_spreadsheet import (  # noqa: E402
    _filter_trial_types,
    _normalise_trial_label,
)
from scripts.analysis.reaction_matrix_specific_flies_vs_control import (  # noqa: E402
    _fisher_per_column,
    _rates_from_matrix,
    _resolve_non_reactive_mask,
)
from scripts.analysis.reaction_matrix_training_vs_control import (  # noqa: E402
    _RC_CONTEXT,
    _build_during_matrix,
    _format_p_value,
    plot_training_vs_control_bars,
)
from scripts.analysis.score_summary import (  # noqa: E402
    REACTION_BOUNDARY_Y,
    SCORES,
    _load_scores,
    _per_fly_score_matrix,
    _score_cmap,
)

# Rig 1 has no suffix. The `_rig_` prefix in the pattern is what keeps the
# `batch_N` token from being read as a rig number.
_RIG_RE = re.compile(r"_rig_(\d+)", re.IGNORECASE)
_BATCH_RE = re.compile(r"batch_(\d+)", re.IGNORECASE)

# The batch token in a folder name records which starvation schedule the fly
# was *meant* to be on; when the actual starvation time disagrees (e.g. the
# august_09 batch-2 flies were starved on the batch-1 schedule), an explicit
# override reassigns just that fly. Labels rename the displayed batch group
# (titles/legends) without touching tags or filenames.
_BATCH_OVERRIDES: dict[str, int] = {}
_BATCH_LABELS: dict[int, str] = {}


def set_batch_overrides(overrides: dict[str, int]) -> None:
    _BATCH_OVERRIDES.clear()
    _BATCH_OVERRIDES.update({str(k): int(v) for k, v in overrides.items()})


def set_batch_labels(labels: dict[int, str]) -> None:
    _BATCH_LABELS.clear()
    _BATCH_LABELS.update({int(k): str(v) for k, v in labels.items()})


def batch_label(value: int) -> str:
    return _BATCH_LABELS.get(int(value), f"Batch {value}")

TRAIN_COLOR_TRAINED = "#1a3a6b"
TRAIN_COLOR_OTHER = "#7bafd4"
CTRL_COLOR_TRAINED = "#808080"
CTRL_COLOR_OTHER = "#c8c8c8"


# ---------------------------------------------------------------------------
# Fly folder parsing / group definitions
# ---------------------------------------------------------------------------


def rig_of(fly: str) -> int:
    """Recording rig for a fly folder name; rig 1 is the unsuffixed default."""
    m = _RIG_RE.search(str(fly))
    return int(m.group(1)) if m else 1


def batch_of(fly: str) -> int | None:
    """Starvation batch for a fly folder name, or None when it carries none."""
    key = str(fly)
    if key in _BATCH_OVERRIDES:
        return _BATCH_OVERRIDES[key]
    m = _BATCH_RE.search(key)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class Group:
    """One side of a comparison: a dataset, optionally narrowed by rig/batch.

    ``value`` is a single level, or a tuple of levels to pool (rigs 1+2 as one
    group).
    """

    label: str
    dataset: str
    kind: str | None      # "rig" | "batch" | None (whole dataset)
    value: int | tuple[int, ...] | None


@dataclass(frozen=True)
class Comparison:
    tag: str
    title_suffix: str
    a: Group
    b: Group


def _factor_series(df: pd.DataFrame, kind: str) -> pd.Series:
    fn = rig_of if kind == "rig" else batch_of
    return df["fly"].map(fn)


def select_group(df: pd.DataFrame, group: Group) -> pd.DataFrame:
    """Rows of ``df`` belonging to ``group``."""
    sub = df[df["dataset_canon"] == group.dataset]
    if group.kind is None:
        return sub.copy()
    factor = _factor_series(sub, group.kind)
    if isinstance(group.value, tuple):
        return sub[factor.isin(group.value)].copy()
    return sub[factor == group.value].copy()


def _values_present(df: pd.DataFrame, dataset: str, kind: str) -> list[int]:
    sub = df[df["dataset_canon"] == dataset]
    vals = _factor_series(sub, kind).dropna()
    return sorted({int(v) for v in vals})


def build_comparisons(
    df: pd.DataFrame, train_canon: str, ctrl_canon: str
) -> list[Comparison]:
    """Enumerate every rig/batch split that both sides can actually support.

    A level present on only one arm gets no training-vs-control figure — it
    would draw one arm's bars against an empty cohort — but it still takes part
    in the within-arm comparisons for the arm that has it.
    """
    comparisons: list[Comparison] = []

    def level_label(kind: str, v: int) -> str:
        if kind == "batch":
            return batch_label(v)
        return f"{kind.capitalize()} {v}"

    for kind in ("rig", "batch"):
        train_vals = _values_present(df, train_canon, kind)
        ctrl_vals = _values_present(df, ctrl_canon, kind)
        for v in sorted(set(train_vals) & set(ctrl_vals)):
            comparisons.append(
                Comparison(
                    tag=f"tvc_{kind}_{v}",
                    title_suffix=f"{level_label(kind, v)}: Training vs Control",
                    a=Group(f"Training", train_canon, kind, v),
                    b=Group(f"Control", ctrl_canon, kind, v),
                )
            )

    for arm, dataset, arm_label in (
        ("train", train_canon, "Training"),
        ("ctrl", ctrl_canon, "Control"),
    ):
        for kind in ("rig", "batch"):
            vals = _values_present(df, dataset, kind)
            if len(vals) < 2:
                continue
            base = vals[0]
            for v in vals[1:]:
                comparisons.append(
                    Comparison(
                        tag=f"{arm}_{kind}_{base}_vs_{kind}_{v}",
                        title_suffix=(
                            f"{arm_label}: {level_label(kind, base)}"
                            f" vs {level_label(kind, v)}"
                        ),
                        a=Group(level_label(kind, base), dataset, kind, base),
                        b=Group(level_label(kind, v), dataset, kind, v),
                    )
                )

    return comparisons


# ---------------------------------------------------------------------------
# Score statistics
# ---------------------------------------------------------------------------


def group_score_samples(
    df: pd.DataFrame, columns: Sequence[str]
) -> dict[str, np.ndarray]:
    """Per-odor arrays of *fly-level* mean scores — one value per fly.

    Averaging within fly first is what makes the fly the unit of replication;
    pooling raw trials would let a fly that saw an odor twice count twice.
    """
    samples: dict[str, np.ndarray] = {c: np.array([], dtype=float) for c in columns}
    if df.empty:
        return samples
    fly_level = (
        df.groupby(["odor_col", "fly", "fly_number"])["score"].mean().reset_index()
    )
    for odor, grp in fly_level.groupby("odor_col"):
        if odor in samples:
            samples[odor] = grp["score"].to_numpy(float)
    return samples


def group_score_stats(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Mean / SEM / n over fly-level means, one row per column in order."""
    samples = group_score_samples(df, columns)
    rows = []
    for odor in columns:
        vals = samples[odor]
        n = int(vals.size)
        if n:
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        else:
            mean, sem = np.nan, 0.0
        rows.append(
            {"odor": odor, "mean_score": mean, "sem_score": sem, "n_flies": n}
        )
    return pd.DataFrame(rows)


def mannwhitney_per_column(
    samples_a: dict[str, np.ndarray],
    samples_b: dict[str, np.ndarray],
    columns: Sequence[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for odor in columns:
        a = samples_a.get(odor, np.array([], dtype=float))
        b = samples_b.get(odor, np.array([], dtype=float))
        if a.size == 0 or b.size == 0:
            out[odor] = float("nan")
            continue
        out[odor] = float(
            mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue
        )
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _fly_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df[["fly", "fly_number"]].drop_duplicates().shape[0])


def _relabel_legend(ax: plt.Axes, label_a: str, label_b: str) -> None:
    """Rename the Training/Control legend for within-arm comparisons."""
    handles, labels = ax.get_legend_handles_labels()
    renamed = []
    for lab in labels:
        if lab == "Training":
            renamed.append(label_a)
        elif lab == "Control":
            renamed.append(label_b)
        else:
            renamed.append(lab)
    ax.legend(handles, renamed, loc="upper right", fontsize=9, framealpha=0.8)


def _write_sidecar(
    path: Path,
    comp: Comparison,
    n_a: int,
    n_b: int,
    columns: Sequence[str],
    p_key: str,
    p_values: dict[str, float],
) -> None:
    payload = {
        "tag": comp.tag,
        "title_suffix": comp.title_suffix,
        "group_a": {"label": comp.a.label, "dataset": comp.a.dataset, "n_flies": n_a},
        "group_b": {"label": comp.b.label, "dataset": comp.b.dataset, "n_flies": n_b},
        "odor_columns": list(columns),
        p_key: {k: (None if pd.isna(v) else float(v)) for k, v in p_values.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVED] {path}")


def reaction_figures(
    df: pd.DataFrame,
    comp: Comparison,
    *,
    train_canon: str,
    columns: list[str],
    out_dir: Path,
    latency_sec: float,
    after_window_sec: float,
) -> bool:
    """PER% bars + per-fly reaction panels for one comparison."""
    df_a = select_group(df, comp.a)
    df_b = select_group(df, comp.b)

    mat_a, flies_a, _cols, flagged_a = _build_during_matrix(
        df_a, comp.a.dataset, None, remap_from=train_canon, columns=columns,
        order="observed",
    )
    mat_b, flies_b, _cols_b, flagged_b = _build_during_matrix(
        df_b, comp.b.dataset, None, remap_from=train_canon, columns=columns,
        order="observed",
    )
    if not len(flies_a) or not len(flies_b):
        print(f"[SKIP] {comp.tag}: one side has no usable flies")
        return False

    trained_display = _trained_label(train_canon)
    rate_a = _rates_from_matrix(mat_a, columns)
    rate_a["is_trained"] = (
        rate_a["odor"].astype(str).str.casefold().str.startswith(trained_display.casefold())
    )
    rate_b = _rates_from_matrix(mat_b, columns)
    p_values = _fisher_per_column(mat_a, mat_b, columns)

    odor_label = DISPLAY_LABEL.get(train_canon, train_canon)
    n_cols = len(columns)
    base_w = max(10.0, 0.70 * n_cols + 6.0)
    xtick_fs = 9 if n_cols <= 10 else (8 if n_cols <= 16 else 7)
    stub = (
        f"{comp.tag}_{int(after_window_sec)}_latency_{latency_sec:.3f}s"
    )

    with plt.rc_context(_RC_CONTEXT):
        fig_bar, ax_bar = plt.subplots(figsize=(base_w, 5.0))
        plot_training_vs_control_bars(
            ax_bar, rate_a, rate_b,
            title=f"Reaction Rates – {odor_label} ({comp.title_suffix})",
            p_values=p_values,
        )
        _relabel_legend(ax_bar, comp.a.label, comp.b.label)
        bar_path = out_dir / f"reaction_matrix_{stub}.png"
        fig_bar.savefig(bar_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {bar_path}")
        plt.close(fig_bar)

        cmap = ListedColormap(["white", "black"])
        cmap.set_bad(color="0.7")
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        n_max = max(1, mat_a.shape[0], mat_b.shape[0])
        pair_h = max(4.0, n_max * 0.26 + 3.0)
        fig_pair = plt.figure(figsize=(base_w * 1.15, pair_h))
        gs_pair = gridspec.GridSpec(1, 2, wspace=0.12)
        ax_left = fig_pair.add_subplot(gs_pair[0, 0])
        ax_right = fig_pair.add_subplot(gs_pair[0, 1])
        # Group B sits left / group A right, mirroring the published
        # "Control | Training" reading order.
        for ax, mat, pairs, flagged, label in (
            (ax_left, mat_b, flies_b, flagged_b, comp.b.label),
            (ax_right, mat_a, flies_a, flagged_a, comp.a.label),
        ):
            if mat.size:
                ax.imshow(
                    mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
                    extent=(-0.5, n_cols - 0.5, mat.shape[0] - 0.5, -0.5),
                )
                ax.set_ylim(n_max - 0.5, -0.5)
            _style_trained_xticks(ax, list(columns), trained_display, xtick_fs)
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels([f"{f} fly{n}" for f, n in pairs], fontsize=6)
            ax.set_title(f"{label} ({mat.shape[0]} Flies)", fontsize=12, weight="bold")
            for idx, pair in enumerate(pairs):
                if pair in flagged:
                    ax.text(-0.35, idx, "*", ha="right", va="center", color="red",
                            fontsize=12, fontweight="bold", clip_on=False)
        fig_pair.suptitle(
            f"{_matrix_title(train_canon)} – {comp.title_suffix}",
            fontsize=14, weight="bold",
        )
        pair_path = out_dir / f"reaction_matrix_pair_{stub}.png"
        fig_pair.savefig(pair_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {pair_path}")
        plt.close(fig_pair)

    _write_sidecar(
        out_dir / f"reaction_matrix_{stub}.json",
        comp, mat_a.shape[0], mat_b.shape[0], columns,
        "fisher_p", {odor: p for (_, odor), p in p_values.items()},
    )
    return True


def _draw_score_brackets(
    ax: plt.Axes,
    x: np.ndarray,
    bar_w: float,
    stats_a: pd.DataFrame,
    stats_b: pd.DataFrame,
    p_values: dict[str, float],
) -> None:
    for i, odor in enumerate(stats_a["odor"]):
        p = p_values.get(odor, np.nan)
        p_str = _format_p_value(p)
        if not p_str:
            continue
        a = stats_a.iloc[i]
        b = stats_b.iloc[i]
        top = max(
            0.0,
            float(np.nan_to_num(a["mean_score"])) + float(a["sem_score"]),
            float(np.nan_to_num(b["mean_score"])) + float(b["sem_score"]),
        )
        bracket_y = top + 0.75
        tip_y = bracket_y - 0.08
        ax.plot(
            [x[i] - bar_w / 2, x[i] - bar_w / 2, x[i] + bar_w / 2, x[i] + bar_w / 2],
            [tip_y, bracket_y, bracket_y, tip_y],
            color="black", linewidth=0.9, clip_on=False,
        )
        ax.text(x[i], bracket_y + 0.05, p_str, ha="center", va="bottom", fontsize=7)


def score_figures(
    scores: pd.DataFrame,
    comp: Comparison,
    *,
    train_canon: str,
    out_dir: Path,
) -> bool:
    """Mean-score bars + per-fly score panels for one comparison."""
    df_a = select_group(scores, comp.a)
    df_b = select_group(scores, comp.b)
    if df_a.empty or df_b.empty:
        print(f"[SKIP] {comp.tag}: one side has no scored trials")
        return False

    columns = sorted(
        set(df_a["odor_col"]).union(df_b["odor_col"]), key=str.casefold
    )
    stats_a = group_score_stats(df_a, columns)
    stats_b = group_score_stats(df_b, columns)
    p_values = mannwhitney_per_column(
        group_score_samples(df_a, columns), group_score_samples(df_b, columns), columns
    )

    trained = _trained_label(train_canon)
    is_trained = [str(c).casefold().startswith(trained.casefold()) for c in columns]
    label = DISPLAY_LABEL.get(train_canon, train_canon)

    x = np.arange(len(columns))
    bar_w = 0.35
    vals_a = stats_a["mean_score"].fillna(0.0).to_numpy(float)
    vals_b = stats_b["mean_score"].fillna(0.0).to_numpy(float)
    err_a = stats_a["sem_score"].fillna(0.0).to_numpy(float)
    err_b = stats_b["sem_score"].fillna(0.0).to_numpy(float)
    y_top = float(
        np.max(np.concatenate([np.maximum(vals_a + err_a, 0.0),
                               np.maximum(vals_b + err_b, 0.0)]))
    ) if len(columns) else 0.0

    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(max(7, len(columns) * 1.0 + 2), 5.5))
        ax.bar(
            x - bar_w / 2, vals_a, width=bar_w, yerr=err_a, capsize=4,
            color=[TRAIN_COLOR_TRAINED if t else TRAIN_COLOR_OTHER for t in is_trained],
            edgecolor="black", linewidth=0.75, label=comp.a.label,
        )
        ax.bar(
            x + bar_w / 2, vals_b, width=bar_w, yerr=err_b, capsize=4,
            color=[CTRL_COLOR_TRAINED if t else CTRL_COLOR_OTHER for t in is_trained],
            edgecolor="black", linewidth=0.75, label=comp.b.label,
        )
        for xi, mean_v, sem_v in zip(x - bar_w / 2, vals_a, err_a):
            ax.text(xi, mean_v + sem_v + 0.12, f"{mean_v:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
        for xi, mean_v, sem_v in zip(x + bar_w / 2, vals_b, err_b):
            ax.text(xi, mean_v + sem_v + 0.12, f"{mean_v:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

        labels = [
            f"{str(o).upper() if t else str(o)}\n(n={na}/{nb})"
            for o, t, na, nb in zip(
                columns, is_trained, stats_a["n_flies"], stats_b["n_flies"]
            )
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        for tick, t in zip(ax.get_xticklabels(), is_trained):
            if t:
                tick.set_color(TRAIN_COLOR_TRAINED)
                tick.set_weight("bold")

        ax.set_ylabel("Mean Score")
        ax.set_xlabel("Presented Odor")
        ax.set_title(
            f"Mean Model Score – {label} ({comp.title_suffix})",
            fontsize=13, weight="bold",
        )
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.axhline(y=REACTION_BOUNDARY_Y, color="red", linewidth=0.8, linestyle=":",
                   alpha=0.6, label="Reaction Boundary")
        ax.set_ylim(-1.5, max(6.1, y_top + 1.0))
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        _draw_score_brackets(ax, x, bar_w, stats_a, stats_b, p_values)
        plt.tight_layout()
        bar_path = out_dir / f"mean_score_{comp.tag}.png"
        fig.savefig(bar_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {bar_path}")
        plt.close(fig)

        # --- per-fly score panels -------------------------------------------
        mat_a, flies_a = _per_fly_score_matrix(
            df_a.assign(dataset_canon=comp.a.dataset), comp.a.dataset, columns
        )
        mat_b, flies_b = _per_fly_score_matrix(
            df_b.assign(dataset_canon=comp.b.dataset), comp.b.dataset, columns
        )
        n_max = max(len(flies_a), len(flies_b))
        cmap, norm = _score_cmap()
        fig = plt.figure(
            figsize=(max(9, len(columns) * 1.5 + 3), n_max * 0.26 + 4.0)
        )
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 0.10, 0.035], wspace=0.10)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 3])
        for ax, mat, flies, panel_label in (
            (ax_left, mat_b, flies_b, comp.b.label),
            (ax_right, mat_a, flies_a, comp.a.label),
        ):
            # extent must match THIS matrix's row count, never n_max: imshow
            # stretches to fill the extent, so a shared extent would render the
            # shorter panel's cells too tall. Equal cell height comes from the
            # shared set_ylim below.
            ax.imshow(
                np.ma.masked_invalid(mat), cmap=cmap, norm=norm, aspect="auto",
                interpolation="nearest",
                extent=(-0.5, len(columns) - 0.5, mat.shape[0] - 0.5, -0.5),
            )
            ax.set_ylim(n_max - 0.5, -0.5)
            for j in range(len(columns) + 1):
                ax.axvline(j - 0.5, color="white", lw=1.6)
            for i in range(n_max + 1):
                ax.axhline(i - 0.5, color="white", lw=1.6)
            ax.set_yticks([])
            ax.set_xticks(np.arange(len(columns)))
            ax.set_xticklabels(
                [str(c).upper() if t else str(c) for c, t in zip(columns, is_trained)],
                rotation=35, ha="right", fontsize=8,
            )
            for tick, t in zip(ax.get_xticklabels(), is_trained):
                if t:
                    tick.set_color(TRAIN_COLOR_TRAINED)
                    tick.set_weight("bold")
            ax.set_title(f"{panel_label} ({mat.shape[0]} Flies)",
                         fontsize=12, weight="bold")
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=SCORES
        )
        cb.set_label("Odor Response Score", fontsize=9, labelpad=4)
        cb.ax.tick_params(labelsize=8, pad=2)
        cb.ax.axhline(REACTION_BOUNDARY_Y, color="black", lw=2.2)
        fig.suptitle(
            f"Per-Fly Odor Response – {label} ({comp.title_suffix})",
            fontsize=14, weight="bold",
        )
        pair_path = out_dir / f"mean_score_pair_{comp.tag}.png"
        fig.savefig(pair_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {pair_path}")
        plt.close(fig)

    _write_sidecar(
        out_dir / f"mean_score_{comp.tag}.json",
        comp, _fly_count(df_a), _fly_count(df_b), columns,
        "mannwhitney_p", p_values,
    )
    return True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_reaction_frame(
    csv_path: Path, *, flagged_flies_csv: str = ""
) -> pd.DataFrame:
    """Predictions CSV prepared exactly as the reaction-matrix scripts do."""
    df = read_table(csv_path)
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

    flagged = _resolve_non_reactive_mask(df, flagged_flies_csv)
    if flagged.any():
        dropped = df.loc[flagged, ["dataset", "fly", "fly_number"]].drop_duplicates()
        print(
            f"[INFO] Excluding {len(dropped)} flagged/non-reactive flies: "
            + ", ".join(f"{r.dataset}::{r.fly}::{r.fly_number}"
                        for r in dropped.itertuples(index=False))
        )
        df = df.loc[~flagged].copy()

    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)
    df["during_hit"] = df["prediction"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["dataset", "fly", "fly_number", "trial"], keep="first")
    return _normalise_fly_columns(df)


def load_score_frame(csv_path: Path, *, flagged_flies_csv: str = "") -> pd.DataFrame:
    scores = _load_scores(csv_path)
    flagged = _resolve_non_reactive_mask(scores, flagged_flies_csv)
    if flagged.any():
        scores = scores.loc[~flagged].copy()
    return scores


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
        print(f"[INFO] Loaded odor_remap for {len(remap)} datasets from {config_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_batch_overrides(pairs: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for pair in pairs:
        fly, sep, batch = str(pair).partition("=")
        if not sep or not fly:
            raise ValueError(f"--batch-override expects FLY=N, got {pair!r}")
        try:
            out[fly] = int(batch)
        except ValueError:
            raise ValueError(
                f"--batch-override batch must be an integer, got {pair!r}"
            ) from None
    return out


def _parse_batch_labels(pairs: Sequence[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for pair in pairs:
        batch, sep, label = str(pair).partition("=")
        if not sep or not label:
            raise ValueError(f"--batch-label expects N=LABEL, got {pair!r}")
        try:
            out[int(batch)] = label
        except ValueError:
            raise ValueError(
                f"--batch-label batch must be an integer, got {pair!r}"
            ) from None
    return out


def _parse_rig_compares(
    specs: Sequence[str],
) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Parse ``ARM:RIGS_A:RIGS_B`` specs, e.g. ``train:1,2:3``."""
    out: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for spec in specs:
        parts = str(spec).split(":")
        if len(parts) != 3:
            raise ValueError(f"--rig-compare expects ARM:RIGS:RIGS, got {spec!r}")
        arm, side_a, side_b = parts
        if arm not in ("train", "ctrl"):
            raise ValueError(f"--rig-compare arm must be train|ctrl, got {spec!r}")
        try:
            rigs_a = tuple(int(v) for v in side_a.split(",") if v != "")
            rigs_b = tuple(int(v) for v in side_b.split(",") if v != "")
        except ValueError:
            raise ValueError(
                f"--rig-compare rigs must be integers, got {spec!r}"
            ) from None
        if not rigs_a or not rigs_b:
            raise ValueError(f"--rig-compare expects rigs on both sides, got {spec!r}")
        out.append((arm, rigs_a, rigs_b))
    return out


def _rig_compare_comparison(
    arm: str,
    rigs_a: tuple[int, ...],
    rigs_b: tuple[int, ...],
    train_canon: str,
    ctrl_canon: str,
) -> Comparison:
    dataset = train_canon if arm == "train" else ctrl_canon
    arm_label = "Training" if arm == "train" else "Control"

    def label(rigs: tuple[int, ...]) -> str:
        return "Rig " + "+".join(str(r) for r in rigs)

    def tag_part(rigs: tuple[int, ...]) -> str:
        return "_".join(str(r) for r in rigs)

    return Comparison(
        tag=f"{arm}_rig_{tag_part(rigs_a)}_vs_rig_{tag_part(rigs_b)}",
        title_suffix=f"{arm_label}: {label(rigs_a)} vs {label(rigs_b)}",
        a=Group(label(rigs_a), dataset, "rig", rigs_a),
        b=Group(label(rigs_b), dataset, "rig", rigs_b),
    )


def _parse_tvc_rigs(specs: Sequence[str]) -> list[tuple[int, ...]]:
    """Parse ``--tvc-rig`` comma-separated rig lists, e.g. ``1,2``."""
    out: list[tuple[int, ...]] = []
    for spec in specs:
        try:
            rigs = tuple(int(v) for v in str(spec).split(",") if v != "")
        except ValueError:
            raise ValueError(f"--tvc-rig rigs must be integers, got {spec!r}") from None
        if not rigs:
            raise ValueError(f"--tvc-rig expects at least one rig, got {spec!r}")
        out.append(rigs)
    return out


def _tvc_rig_comparison(
    rigs: tuple[int, ...], train_canon: str, ctrl_canon: str
) -> Comparison:
    label = "Rig " + "+".join(str(r) for r in rigs)
    tag_part = "_".join(str(r) for r in rigs)
    return Comparison(
        tag=f"tvc_rig_{tag_part}",
        title_suffix=f"{label}: Training vs Control",
        a=Group("Training", train_canon, "rig", rigs),
        b=Group("Control", ctrl_canon, "rig", rigs),
    )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-dataset", required=True)
    p.add_argument("--control-dataset", required=True)
    p.add_argument("--config", type=str, default="",
                   help="Pipeline config YAML; loads dataset_overrides.odor_remap.")
    p.add_argument("--flagged-flies-csv", type=str, default="",
                   help="flagged-flys-truth CSV (FLY-State != 1 excluded). "
                        "Authoritative when given.")
    p.add_argument("--latency-sec", type=float, default=2.15)
    p.add_argument("--after-window-sec", type=float, default=30.0)
    p.add_argument("--protocol", default="v2", choices=["v2", "legacy"])
    p.add_argument("--batch-override", action="append", default=[],
                   metavar="FLY=N",
                   help="Reassign one fly folder to batch N (repeatable), for "
                        "flies whose folder token disagrees with the actual "
                        "starvation schedule.")
    p.add_argument("--batch-label", action="append", default=[],
                   metavar="N=LABEL",
                   help='Display label for batch group N, e.g. '
                        '"1=Starved 24±3 h". Tags/filenames keep batch numbers.')
    p.add_argument("--rig-compare", action="append", default=[],
                   metavar="ARM:RIGS:RIGS",
                   help="Extra within-arm rig comparison, pooling comma-"
                        "separated rigs per side (repeatable). E.g. "
                        "train:1,2:3 compares rigs 1+2 combined vs rig 3 "
                        "on the training arm.")
    p.add_argument("--tvc-rig", action="append", default=[], metavar="RIGS",
                   help="Extra training-vs-control comparison restricted to "
                        "these comma-separated rigs pooled (repeatable). E.g. "
                        "1,2 compares the arms over rigs 1+2 combined.")
    p.add_argument("--only", action="append", default=[], metavar="TAG",
                   help="Render only comparisons with these tags (repeatable) "
                        "— for adding a figure to a published folder without "
                        "regenerating the rest.")
    p.add_argument("--restrict-batch", type=int, default=None, metavar="N",
                   help="Narrow both arms to flies of batch N (after "
                        "--batch-override) before building any comparison — "
                        "for a figure set over one starvation group only.")
    p.add_argument("--fly-type", default="", metavar="TYPE",
                   help="Keep only flies whose fly_type matches TYPE "
                        "(case-insensitive), e.g. GR5a-Old.")
    p.add_argument("--fly-prefix", default="", metavar="PREFIX",
                   help="Keep only fly folders starting with PREFIX "
                        "(case-insensitive), e.g. august for one month.")
    args = p.parse_args(argv)

    set_protocol(args.protocol)
    overrides = _parse_batch_overrides(args.batch_override)
    set_batch_overrides(overrides)
    set_batch_labels(_parse_batch_labels(args.batch_label))
    if overrides:
        print(f"[INFO] Batch overrides: {overrides}")
    if args.config:
        _apply_config_remap(args.config)

    train_canon = _canon_dataset(args.train_dataset)
    ctrl_canon = _canon_dataset(args.control_dataset)

    reactions = load_reaction_frame(
        args.csv_path, flagged_flies_csv=args.flagged_flies_csv
    )
    scores = load_score_frame(
        args.csv_path, flagged_flies_csv=args.flagged_flies_csv
    )

    if args.fly_type:
        wanted_type = args.fly_type.strip().casefold()
        for name, frame in (("reactions", reactions), ("scores", scores)):
            if "fly_type" not in frame.columns:
                raise ValueError(
                    f"--fly-type given but the {name} frame has no fly_type column"
                )
        n_before = _fly_count(reactions)
        type_mask = reactions["fly_type"].astype(str).str.strip().str.casefold()
        reactions = reactions[type_mask == wanted_type].copy()
        score_mask = scores["fly_type"].astype(str).str.strip().str.casefold()
        scores = scores[score_mask == wanted_type].copy()
        print(
            f"[INFO] Restricted to fly_type {args.fly_type!r}: "
            f"{_fly_count(reactions)} of {n_before} flies kept"
        )

    if args.fly_prefix:
        prefix = args.fly_prefix.strip().casefold()
        n_before = _fly_count(reactions)
        reactions = reactions[
            reactions["fly"].str.casefold().str.startswith(prefix)
        ].copy()
        scores = scores[scores["fly"].str.casefold().str.startswith(prefix)].copy()
        print(
            f"[INFO] Restricted to fly prefix {args.fly_prefix!r}: "
            f"{_fly_count(reactions)} of {n_before} flies kept"
        )

    if args.restrict_batch is not None:
        n_before = _fly_count(reactions)
        reactions = reactions[
            reactions["fly"].map(batch_of) == args.restrict_batch
        ].copy()
        scores = scores[scores["fly"].map(batch_of) == args.restrict_batch].copy()
        print(
            f"[INFO] Restricted to batch {args.restrict_batch}: "
            f"{_fly_count(reactions)} of {n_before} flies kept"
        )

    for canon in (train_canon, ctrl_canon):
        if canon not in set(reactions["dataset_canon"]):
            raise ValueError(
                f"{canon} not present in {args.csv_path}; found "
                f"{sorted(set(reactions['dataset_canon']))}"
            )

    # One reference column list for every reaction figure, taken from the full
    # training cohort: per-comparison columns would let two figures disagree on
    # column order, and an odor a single rig never saw would shift the layout
    # instead of showing up as an empty column.
    _m, _f, columns, _flag = _build_during_matrix(
        reactions, train_canon, None, remap_from=train_canon, order="observed"
    )
    if not columns:
        raise RuntimeError(f"No odor columns resolved for {train_canon}")

    comparisons = build_comparisons(reactions, train_canon, ctrl_canon)
    for arm, rigs_a, rigs_b in _parse_rig_compares(args.rig_compare):
        comparisons.append(
            _rig_compare_comparison(arm, rigs_a, rigs_b, train_canon, ctrl_canon)
        )
    for rigs in _parse_tvc_rigs(args.tvc_rig):
        comparisons.append(_tvc_rig_comparison(rigs, train_canon, ctrl_canon))
    if args.only:
        wanted = set(args.only)
        unknown = wanted.difference(c.tag for c in comparisons)
        if unknown:
            raise ValueError(
                f"--only tags not among the comparisons: {sorted(unknown)}; "
                f"available: {[c.tag for c in comparisons]}"
            )
        comparisons = [c for c in comparisons if c.tag in wanted]
    print(f"[INFO] {len(comparisons)} comparisons: {[c.tag for c in comparisons]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for comp in comparisons:
        print(f"\n=== {comp.tag} — {comp.title_suffix}")
        reaction_figures(
            reactions, comp,
            train_canon=train_canon, columns=list(columns), out_dir=args.out_dir,
            latency_sec=args.latency_sec, after_window_sec=args.after_window_sec,
        )
        score_figures(scores, comp, train_canon=train_canon, out_dir=args.out_dir)

    print(f"\n[DONE] {len(comparisons)} comparisons written to {args.out_dir}")


if __name__ == "__main__":
    main()
