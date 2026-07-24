"""Random-split null distribution of control reaction-rate gaps (EB-24-1).

How different do two random subgroups of the EB-Control-24-1 pool look, per odor,
just by chance? We repeatedly split the 27 control flies into two random groups of
`--group-size` (default 10, with the remainder held out), and for each split record
the per-odor absolute reaction-rate gap ``|rate_A - rate_B|``. Over `--n-splits`
(default 100) random splits this gives the chance/noise floor: the typical gap two
random control groups show by luck.

The held-out flies are re-drawn every split (they rotate; they are not a fixed set).

Two figures are written (300 dpi PNG + a JSON sidecar):
  * ``*_chance_only.png``       mean ``|A-B|`` bar + 5-95th percentile whisker per odor.
  * ``*_with_real_overlay.png`` same, plus a red diamond at the OBSERVED batch-1-vs-
    batch-2 ``|gap|`` per odor and its approximate empirical p-value.

CAVEAT (labeled on the overlay figure + in the sidecar): the chance band uses
equal-size groups (default 10 vs 10) while the real batch split is 9 vs 18. Smaller
groups swing more by chance, so the overlay is an *approximate* reference, not a
size-matched permutation test.

Reuses the exact tested helpers behind the reference reaction-matrix figures
(``_build_during_matrix`` / ``_rates_from_matrix`` and the odor-remap / column
styling), so the reaction-rate definition matches the pipeline: for each odor
``rate = (# flies with prediction==1) / (# flies presented that odor) * 100``.

Example:
    python scripts/analysis/reaction_rate_random_split_null.py \
        --csv-path /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv \
        --flagged-flies-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/flagged-flys-truth.csv \
        --config config/config_new.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fbpipe.utils.tables import read_table  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    DISPLAY_LABEL,
    _canon_dataset,
    _normalise_fly_columns,
    _style_trained_xticks,
    _trained_label,
    _trial_num,
    set_dataset_odor_remap,
    set_protocol,
)
from scripts.analysis.reaction_matrix_from_spreadsheet import (  # noqa: E402
    _filter_trial_types,
    _normalise_trial_label,
)
from scripts.analysis.reaction_matrix_training_vs_control import (  # noqa: E402
    _RC_CONTEXT,
    _build_during_matrix,
)
from scripts.analysis.reaction_matrix_specific_flies_vs_control import (  # noqa: E402
    _rates_from_matrix,
    _resolve_non_reactive_mask,
)

# --- defaults --------------------------------------------------------------
DEF_CSV = "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv"
DEF_FLAG = "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/flagged-flys-truth.csv"
DEF_CONFIG = REPO / "config" / "config_new.yaml"
DEF_OUT = "/home/ramanlab/Documents/cole/Results/Figures/EB-24-1_random_split_null"
DEF_CONTROL = "EB-Control-24-1"

# CVD-validated (dataviz skill: run scripts/validate_palette.js) blue/red pair.
CHANCE_BLUE = "#2166ac"  # chance-gap bars
REAL_RED = "#d6604d"  # observed batch-1-vs-batch-2 overlay
WHISKER = "#555555"  # recessive 5-95% whisker
INK = "#222222"  # text (never the series color)


# --- data loading (mirrors the reference reaction-matrix driver's prep) -----
def _apply_config_odor_remap(config_path: Path, control_dataset: str) -> None:
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    overrides = (cfg or {}).get("dataset_overrides", {}) or {}
    remap: dict[str, dict[str, str]] = {}
    for ds_name, ov in overrides.items():
        mapping = (ov or {}).get("odor_remap") if isinstance(ov, dict) else None
        if mapping:
            remap[str(ds_name)] = dict(mapping)
    set_dataset_odor_remap(remap)
    if control_dataset in remap:
        print(f"[remap] {control_dataset}: {remap[control_dataset]}")


def prep(csv: str, flag: str, config: Path, control_dataset: str, protocol: str) -> pd.DataFrame:
    set_protocol(protocol)
    _apply_config_odor_remap(config, control_dataset)
    df = read_table(csv)
    df = _filter_trial_types(df, allowed=("testing",))
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["trial_label"] = df["trial_label"].astype(str).str.strip()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = _normalise_fly_columns(df)
    mask = _resolve_non_reactive_mask(df, flag)
    df = df.loc[~mask].copy()
    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)
    df["during_hit"] = df["prediction"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["dataset", "fly", "fly_number", "trial"], keep="first")
    df = _normalise_fly_columns(df)
    return df


# --- reaction-rate gaps -----------------------------------------------------
def _rates_for_flies(ctrl: pd.DataFrame, ctrl_c: str, pair_set: set, cols) -> np.ndarray:
    """Per-odor reaction rate (%) for the given (fly, fly_number) flies, on fixed cols."""
    key = ctrl[["fly", "fly_number"]].apply(tuple, axis=1)
    sub = ctrl[key.isin(pair_set)].copy()
    mat, pairs, _, _ = _build_during_matrix(
        sub, ctrl_c, None, remap_from=ctrl_c, columns=cols, order="observed"
    )
    if not len(pairs):
        return np.full(len(cols), np.nan)
    return _rates_from_matrix(mat, cols)["rate"].to_numpy()


def _rates_for_substr(ctrl: pd.DataFrame, ctrl_c: str, token: str, cols) -> tuple[np.ndarray, int]:
    """Per-odor reaction rate (%) for control flies whose folder name contains `token`."""
    keep = ctrl["fly"].astype(str).str.lower().str.contains(token)
    sub = ctrl[keep].copy()
    n = sub[["fly", "fly_number"]].drop_duplicates().shape[0]
    mat, pairs, _, _ = _build_during_matrix(
        sub, ctrl_c, None, remap_from=ctrl_c, columns=cols, order="observed"
    )
    if not len(pairs):
        return np.full(len(cols), np.nan), n
    return _rates_from_matrix(mat, cols)["rate"].to_numpy(), n


def sample_chance_gaps(
    ctrl: pd.DataFrame, ctrl_c: str, cols, *, n_splits: int, group_size: int, seed: int
) -> np.ndarray:
    """Return an (n_splits x n_cols) array of |rate_A - rate_B| over random splits.

    Every split independently reshuffles the whole pool and draws A = first
    `group_size`, B = next `group_size`; the remaining flies are held out (a fresh
    hold-out set each split).
    """
    pool = list(map(tuple, ctrl[["fly", "fly_number"]].drop_duplicates().to_numpy()))
    if len(pool) < 2 * group_size:
        raise ValueError(
            f"pool has {len(pool)} flies; need >= {2 * group_size} for two groups of {group_size}"
        )
    rng = np.random.default_rng(seed)
    gaps = np.empty((n_splits, len(cols)), dtype=float)
    for i in range(n_splits):
        order = rng.permutation(len(pool))
        a_set = {pool[j] for j in order[:group_size]}
        b_set = {pool[j] for j in order[group_size : 2 * group_size]}
        rate_a = _rates_for_flies(ctrl, ctrl_c, a_set, cols)
        rate_b = _rates_for_flies(ctrl, ctrl_c, b_set, cols)
        gaps[i] = np.abs(rate_a - rate_b)
    return gaps


# --- plotting ---------------------------------------------------------------
def plot_chance_gap(
    cols,
    trained: str,
    mean_gap: np.ndarray,
    p05: np.ndarray,
    p95: np.ndarray,
    *,
    title: str,
    n_splits: int,
    group_size: int,
    real_gap: np.ndarray | None = None,
    emp_p: np.ndarray | None = None,
    real_sizes: tuple[int, int] | None = None,
    show_values: bool = True,
    out_path: Path,
) -> None:
    x = np.arange(len(cols))
    top = float(np.nanmax(p95))
    if real_gap is not None:
        top = max(top, float(np.nanmax(real_gap)))
    ymax = max(10.0, top * 1.12)

    base_w = max(9.0, 0.85 * len(cols) + 4.0)
    xtick_fs = 9 if len(cols) <= 10 else (8 if len(cols) <= 16 else 7)

    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(base_w, 5.0))

        # mean chance gap bars (2px surface gap via width < 1)
        bars_label = f"Chance gap: mean |A−B|, {group_size}v{group_size} random splits"
        ax.bar(x, mean_gap, width=0.62, color=CHANCE_BLUE, zorder=2, label=bars_label)

        # 5-95th percentile whisker (percentiles => p05 <= mean <= p95)
        yerr = np.vstack([mean_gap - p05, p95 - mean_gap])
        ax.errorbar(
            x, mean_gap, yerr=yerr, fmt="none", ecolor=WHISKER,
            elinewidth=1.5, capsize=3, capthick=1.5, zorder=3,
            label="5–95th percentile of splits",
        )

        # mean value labels seated at the bar top (white bbox masks the whisker line
        # behind them). Suppressed on the overlay figure to keep the comparison clean.
        if show_values:
            for xi, m in zip(x, mean_gap):
                ax.text(
                    xi, m, f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
                    zorder=4,
                )

        # optional observed batch-1-vs-batch-2 overlay
        if real_gap is not None:
            ax.scatter(
                x, real_gap, marker="D", s=72, color=REAL_RED, edgecolor="white",
                linewidth=1.0, zorder=5,
                label=(
                    f"Observed batch 1 vs 2 |gap|"
                    + (f" ({real_sizes[0]}v{real_sizes[1]} flies)" if real_sizes else "")
                ),
            )
            if emp_p is not None:
                for xi, rv, p in zip(x, real_gap, emp_p):
                    if np.isnan(rv):
                        continue
                    ax.annotate(
                        f"p={p:.2f}", (xi, rv), textcoords="offset points",
                        xytext=(0, 8), ha="center", va="bottom", fontsize=7,
                        color=REAL_RED, fontweight="bold",
                    )

        ax.set_ylim(0, ymax)
        ax.set_ylabel("Reaction-rate gap  |A − B|  (%)", fontsize=11, color=INK)
        ax.set_title(title, fontsize=13, weight="bold", color=INK, pad=42)
        ax.margins(x=0.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="0.88", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        _style_trained_xticks(ax, list(cols), trained, xtick_fs)

        # legend as a horizontal strip above the axes (below the title) so it never
        # collides with bars, whiskers, or the red overlay markers/p-labels.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels, fontsize=8, frameon=False, loc="lower center",
            bbox_to_anchor=(0.5, 1.0), ncol=len(handles), columnspacing=1.6,
            handletextpad=0.5,
        )

        # footnote, dropped well below the rotated x-tick labels
        note = f"{n_splits} random splits of the control pool; held-out flies re-drawn each split."
        if real_gap is not None and real_sizes is not None and real_sizes != (group_size, group_size):
            note += (
                f"\nCaveat: chance groups are {group_size}v{group_size} but the observed"
                f" split is {real_sizes[0]}v{real_sizes[1]}; overlay is an approximate"
                f" reference, not a size-matched permutation test."
            )
        ax.annotate(
            note, xy=(0.0, -0.42), xycoords="axes fraction", ha="left", va="top",
            fontsize=7, color="0.35", annotation_clip=False,
        )

        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[SAVED] {out_path}")


# --- main -------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-path", default=DEF_CSV)
    ap.add_argument("--flagged-flies-csv", default=DEF_FLAG)
    ap.add_argument("--config", default=str(DEF_CONFIG))
    ap.add_argument("--control-dataset", default=DEF_CONTROL)
    ap.add_argument("--out-dir", default=DEF_OUT)
    ap.add_argument("--protocol", default="v2")
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ctrl_c = _canon_dataset(args.control_dataset)
    df = prep(args.csv_path, args.flagged_flies_csv, Path(args.config), args.control_dataset, args.protocol)
    ctrl = df[df["dataset_canon"] == ctrl_c].copy()
    n_pool = ctrl[["fly", "fly_number"]].drop_duplicates().shape[0]
    print(f"[pool] {args.control_dataset}: {n_pool} flies")

    # fixed odor columns from the full control pool -> all splits share axes
    _mat, _pairs, cols, _flagged = _build_during_matrix(
        ctrl, ctrl_c, None, remap_from=ctrl_c, order="observed"
    )
    trained = _trained_label(ctrl_c)
    odor_label = DISPLAY_LABEL.get(ctrl_c, ctrl_c)
    print(f"[cols] {len(cols)} odors; trained={trained!r}")

    # chance distribution
    gaps = sample_chance_gaps(
        ctrl, ctrl_c, cols, n_splits=args.n_splits, group_size=args.group_size, seed=args.seed
    )
    mean_gap = np.nanmean(gaps, axis=0)
    p05 = np.nanpercentile(gaps, 5, axis=0)
    p95 = np.nanpercentile(gaps, 95, axis=0)
    std = np.nanstd(gaps, axis=0)

    # observed batch-1-vs-batch-2 gap (folder-substring split, fixed cols)
    rate_b1, n_b1 = _rates_for_substr(ctrl, ctrl_c, "batch_1", cols)
    rate_b2, n_b2 = _rates_for_substr(ctrl, ctrl_c, "batch_2", cols)
    real_gap = np.abs(rate_b1 - rate_b2)
    # approximate empirical p: fraction of chance splits with gap >= observed
    emp_p = np.array([
        float(np.nanmean(gaps[:, j] >= real_gap[j])) if not np.isnan(real_gap[j]) else np.nan
        for j in range(len(cols))
    ])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stub = (
        f"reaction_rate_chance_gap_ctrl_random_"
        f"{args.group_size}v{args.group_size}_{args.n_splits}splits_seed{args.seed}"
    )

    plot_chance_gap(
        cols, trained, mean_gap, p05, p95,
        title=f"Chance reaction-rate gap between random control groups – {odor_label}",
        n_splits=args.n_splits, group_size=args.group_size,
        out_path=out / f"{stub}_chance_only.png",
    )
    plot_chance_gap(
        cols, trained, mean_gap, p05, p95,
        title=f"Random-group chance gap vs observed batch split – {odor_label}",
        n_splits=args.n_splits, group_size=args.group_size,
        real_gap=real_gap, emp_p=emp_p, real_sizes=(n_b1, n_b2),
        show_values=False,
        out_path=out / f"{stub}_with_real_overlay.png",
    )

    sidecar = {
        "control_dataset": args.control_dataset,
        "n_pool_flies": int(n_pool),
        "n_splits": args.n_splits,
        "group_size": args.group_size,
        "seed": args.seed,
        "held_out_per_split": int(n_pool - 2 * args.group_size),
        "observed_batch_split": {"batch_1_flies": int(n_b1), "batch_2_flies": int(n_b2)},
        "size_mismatch_caveat": (
            f"chance groups are {args.group_size}v{args.group_size} but observed split is "
            f"{n_b1}v{n_b2}; empirical p is approximate, not size-matched."
        ),
        "per_odor": [
            {
                "odor": str(c),
                "mean_abs_gap": float(mean_gap[j]),
                "p05": float(p05[j]),
                "p95": float(p95[j]),
                "std": float(std[j]),
                "observed_batch_gap": (None if np.isnan(real_gap[j]) else float(real_gap[j])),
                "empirical_p": (None if np.isnan(emp_p[j]) else float(emp_p[j])),
            }
            for j, c in enumerate(cols)
        ],
    }
    with open(out / f"{stub}.json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"[SAVED] {out / f'{stub}.json'}")


if __name__ == "__main__":
    main()
