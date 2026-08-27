"""RandomPanel responses by the *trial position* an odor is presented in.

Companion to ``randompanel_conc_comparison`` that reframes the same three
RandomPanel concentration datasets (10, 1, 0.1) around *when* in the session a
trial happens, rather than which odor it is:

* **Figure 1 — response vs trial position (1-14), all odors pooled.** Three
  stacked panels (concentration high -> low), mean ordinal score and % reaction
  as a function of trial number. Reveals session-position effects (habituation /
  sensitization; the block-1 vs block-2 step at trial 8). A per-panel Spearman
  correlation (score vs trial) quantifies any monotonic drift.

* **Figure 2 — per-odor response by trial position.** For each concentration, a
  grid of one small subplot per odorant showing the metric at every trial
  position that odorant landed in. Because the RandomPanel order is randomised
  across rigs, each odorant appears in many positions, so this answers "does
  benzaldehyde score differently in train-1 vs train-4?". A per-odorant Spearman
  rho / p marks whether the response depends on position.

  * **2a** uses the raw trial number 1-14 (as requested).
  * **2b** collapses to within-block position 1-7 (``(trial-1) % 7 + 1``),
    pooling each odorant's two exposures for ~2x the n per cell.

Metrics everywhere: mean ordinal score (+/- SEM) and % reacting (score >= 2,
+/- Wilson 95% CI). The unit of analysis is one animal (fly, fly_number) x one
trial -- exactly one presentation per animal per trial, so no exposure pooling.

Usage::

    python scripts/analysis/randompanel_trial_position.py \
        --csv-path /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv \
        --out-dir  /home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Matrix-PER-Reactions-Model/analysis_random_panel \
        --config   config/config_new.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import set_protocol, should_write
from scripts.analysis.randompanel_conc_comparison import (
    CONC_BY_DATASET,
    CONC_COLOR,
    CONC_PLOT_ORDER,
    REACTION_BOUNDARY,
    _RC_CONTEXT,
    _sig_stars,
    _wilson_ci,
)
from scripts.analysis.score_summary import _load_scores
from scripts.analysis.per_axis_labels import PERCENT_Y_LABEL, SCORE_Y_LABEL  # noqa: E402

# Odor trials run 1..14 (7 odors x 2 exposure blocks); 15-20 are light-only and
# already dropped by ``_load_scores`` (no odor token in the label).
N_ODOR_TRIALS = 14
BLOCK_SIZE = 7  # odors per exposure block

# Odor display names come from the per-dataset ``odor_remap`` in the config
# (for the RandomPanel datasets: Linalool -> "Isoamyl Acetate"), exactly like
# ``randompanel_conc_comparison``. Pass --config so the remap is applied.


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_trials(csv_path: Path, *, fly_type: str = "GR5a-Old", config: str = "") -> pd.DataFrame:
    """Load the three RandomPanel concentrations, one row per animal x trial.

    Unlike ``randompanel_conc_comparison._load_panel`` this keeps every trial
    separate (no per-fly exposure pooling) and carries ``trial_num`` so we can
    analyse response by session position.

    Odor display names come from the per-dataset ``odor_remap`` in ``config``
    (RandomPanel: Linalool -> "Isoamyl Acetate"), same as the sibling
    ``randompanel_conc_comparison._load_panel``.
    """
    set_protocol("v2")
    if config:
        try:
            from fbpipe.config import load_settings
            from scripts.analysis.envelope_visuals import set_dataset_odor_remap

            settings = load_settings(config)
            remap = {
                str(ds): dict(ov.odor_remap)
                for ds, ov in settings.dataset_overrides.items()
                if getattr(ov, "odor_remap", None)
            }
            if remap:
                set_dataset_odor_remap(remap)
        except Exception as exc:  # noqa: BLE001 -- defensive, matches conc script
            print(f"[WARN] Failed to load odor_remap from {config}: {exc}")

    df = _load_scores(csv_path)
    if "fly_type" in df.columns and fly_type:
        df = df[df["fly_type"].astype(str).str.strip() == fly_type].copy()
    df = df[df["dataset_canon"].isin(CONC_BY_DATASET)].copy()
    if df.empty:
        return df

    df["conc"] = df["dataset_canon"].map(CONC_BY_DATASET)
    df["odor"] = df["odor_display"]
    df["trial_num"] = pd.to_numeric(df["trial_num"], errors="coerce")
    # Keep only the odor trials (defensive: light-only should already be gone).
    df = df[df["trial_num"].between(1, N_ODOR_TRIALS)].copy()
    df["trial_num"] = df["trial_num"].astype(int)
    return add_position_columns(
        df[["dataset_canon", "conc", "fly", "fly_number", "trial_num", "odor", "score"]]
    )


def add_position_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``block`` (1/2), ``blockpos`` (1..7) and ``reacted`` (score>=2)."""
    out = df.copy()
    tn = out["trial_num"].astype(int)
    out["block"] = np.where(tn <= BLOCK_SIZE, 1, 2)
    out["blockpos"] = ((tn - 1) % BLOCK_SIZE) + 1
    out["reacted"] = (out["score"] >= REACTION_BOUNDARY).astype(int)
    return out


# ---------------------------------------------------------------------------
# Aggregation / statistics
# ---------------------------------------------------------------------------


def aggregate(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Per group: n, mean/SEM score, reaction count/rate and Wilson 95% CI."""
    recs: list[dict] = []
    for key, grp in df.groupby(list(group_cols)):
        n = len(grp)
        k = int(grp["reacted"].sum())
        lo, hi = _wilson_ci(k, n)
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec.update(
            {
                "n": n,
                "k_react": k,
                "mean_score": float(grp["score"].mean()),
                "sem_score": float(grp["score"].sem(ddof=1)) if n > 1 else 0.0,
                "pct_react": k / n if n else np.nan,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
        recs.append(rec)
    return pd.DataFrame(recs)


def spearman_by_group(
    df: pd.DataFrame, group_cols: Sequence[str], pos_col: str
) -> pd.DataFrame:
    """Per group: Spearman correlation of raw per-trial score vs ``pos_col``."""
    recs: list[dict] = []
    for key, grp in df.groupby(list(group_cols)):
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        n = len(grp)
        if grp[pos_col].nunique() >= 2 and n >= 3:
            rho, p = spearmanr(grp[pos_col].to_numpy(float), grp["score"].to_numpy(float))
        else:
            rho, p = np.nan, np.nan
        rec.update({"pos_col": pos_col, "n": n, "rho": float(rho), "p": float(p)})
        recs.append(rec)
    return pd.DataFrame(recs)


def _spearman_label(rho: float, p: float) -> str:
    if rho is None or np.isnan(rho):
        return "Spearman: n/a"
    ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    stars = _sig_stars(p)
    tail = f" {stars}" if stars and stars != "ns" else ""
    return rf"$\rho$={rho:+.2f}, {ptxt}{tail}"


def _odor_sort_key(odor: str) -> str:
    # Alphabetical by display name -- matches the odor order in the existing
    # RandomPanel / EB-24-1 figures.
    return odor


# ---------------------------------------------------------------------------
# Figure 1 -- response vs trial position, all odors pooled
# ---------------------------------------------------------------------------


def _err_arrays(cells: pd.DataFrame, value: str, err: tuple[str, str] | str):
    v = cells[value].to_numpy(float)
    if isinstance(err, tuple):
        lo = np.clip(v - cells[err[0]].to_numpy(float), 0, None)
        hi = np.clip(cells[err[1]].to_numpy(float) - v, 0, None)
    else:
        e = cells[err].to_numpy(float)
        lo = hi = e
    return v, np.vstack([lo, hi])


def plot_trial_position(
    summary: pd.DataFrame,
    spear: pd.DataFrame,
    *,
    value: str,
    err: tuple[str, str] | str,
    ylabel: str,
    title: str,
    png_path: Path,
    as_pct: bool = False,
    reaction_line: bool = False,
) -> None:
    """Figure 1: three stacked panels (conc high->low), metric vs trial 1..14."""
    concs = [c for c in CONC_PLOT_ORDER if c in set(summary["conc"])]
    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(
            len(concs), 1, figsize=(9.5, 2.7 * len(concs) + 0.6), sharex=True
        )
        axes = np.atleast_1d(axes)
        spear_by_conc = {r["conc"]: r for _, r in spear.iterrows()}

        for ax, conc in zip(axes, concs):
            cells = summary[summary["conc"] == conc].sort_values("trial_num")
            x = cells["trial_num"].to_numpy(int)
            v, yerr = _err_arrays(cells, value, err)
            ax.errorbar(
                x, v, yerr=yerr, marker="o", ms=6, lw=1.8, capsize=3,
                color=CONC_COLOR[conc], ecolor=CONC_COLOR[conc], mfc=CONC_COLOR[conc],
                mec="white", mew=0.8, clip_on=False, zorder=3,
            )
            # n under each point.
            ytxt = 0.0 if as_pct else -0.15
            for xi, ni in zip(x, cells["n"].to_numpy(int)):
                ax.annotate(
                    str(ni), (xi, ytxt), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6, color="#888888",
                )
            # Block boundary between trial 7 and 8.
            ax.axvline(BLOCK_SIZE + 0.5, color="#bbbbbb", lw=0.9, ls="--", zorder=1)
            if reaction_line:
                ax.axhline(
                    REACTION_BOUNDARY, color="red", lw=0.8, ls=":", alpha=0.6, zorder=1
                )
            sr = spear_by_conc.get(conc)
            ax.set_title(
                f"conc {conc:g}    ({_spearman_label(sr['rho'], sr['p']) if sr is not None else ''})",
                fontsize=10, loc="left",
            )
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
            if as_pct:
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t*100:.0f}%"))

        axes[-1].set_xlabel("Trial position (odor sent 1..14)")
        axes[-1].set_xticks(range(1, N_ODOR_TRIALS + 1))
        # Block annotation on the top panel.
        axes[0].annotate(
            "block 1", (BLOCK_SIZE / 2 + 0.5, 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=7, color="#999999",
        )
        axes[0].annotate(
            "block 2", (BLOCK_SIZE * 1.5 + 0.5, 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=7, color="#999999",
        )
        fig.suptitle(title, fontsize=13, weight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[trial_pos] Wrote {png_path}")


# ---------------------------------------------------------------------------
# Figure 2 -- per-odor response by trial position
# ---------------------------------------------------------------------------


def plot_odor_grid(
    summary: pd.DataFrame,
    spear: pd.DataFrame,
    *,
    conc: float,
    pos_col: str,
    value: str,
    err: tuple[str, str] | str,
    ylabel: str,
    xlabel: str,
    title: str,
    png_path: Path,
    as_pct: bool = False,
    reaction_line: bool = False,
) -> None:
    """Figure 2: one small subplot per odorant, metric vs trial position."""
    odors = sorted(summary["odor"].unique(), key=_odor_sort_key)
    n = len(odors)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    spear_by_odor = {r["odor"]: r for _, r in spear.iterrows()}
    xmax = N_ODOR_TRIALS if pos_col == "trial_num" else BLOCK_SIZE

    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).ravel()
        color = CONC_COLOR[conc]

        for ax, odor in zip(axes, odors):
            cells = summary[summary["odor"] == odor].sort_values(pos_col)
            x = cells[pos_col].to_numpy(int)
            v, yerr = _err_arrays(cells, value, err)
            ax.errorbar(
                x, v, yerr=yerr, marker="o", ms=5.5, lw=1.6, capsize=2.5,
                color=color, ecolor=color, mfc=color, mec="white", mew=0.7,
                clip_on=False, zorder=3,
            )
            for xi, vi, ni in zip(x, v, cells["n"].to_numpy(int)):
                ax.annotate(
                    f"{ni}", (xi, vi), textcoords="offset points", xytext=(0, 5),
                    ha="center", va="bottom", fontsize=5.5, color="#888888",
                )
            if reaction_line:
                ax.axhline(REACTION_BOUNDARY, color="red", lw=0.7, ls=":", alpha=0.6)
            sr = spear_by_odor.get(odor)
            sub = _spearman_label(sr["rho"], sr["p"]) if sr is not None else ""
            ax.set_title(f"{odor}\n{sub}", fontsize=8.5)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.set_xlim(0.5, xmax + 0.5)
            if as_pct:
                ax.set_ylim(-0.02, 1.05)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t*100:.0f}%"))

        for ax in axes[n:]:
            ax.set_visible(False)

        # Shared axis labels.
        fig.supxlabel(xlabel, fontsize=10)
        fig.supylabel(ylabel, fontsize=10)
        fig.suptitle(title, fontsize=13, weight="bold")
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[trial_pos] Wrote {png_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate(
    csv_path: Path,
    out_dir: Path,
    *,
    fly_type: str = "GR5a-Old",
    config: str = "",
    overwrite: bool = True,
) -> None:
    df = load_trials(csv_path, fly_type=fly_type, config=config)
    if df.empty:
        print(
            f"[trial_pos] No RandomPanel trials for fly_type={fly_type!r} in "
            f"{csv_path.name}; skipping."
        )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    geno = f" ({fly_type})" if fly_type else ""

    # --- Figure 1: response vs trial position (all odors pooled) -----------
    trial_summary = aggregate(df, ["conc", "trial_num"]).sort_values(["conc", "trial_num"])
    trial_spear = spearman_by_group(df, ["conc"], "trial_num")

    summ_csv = out_dir / "trial_position_summary.csv"
    if should_write(summ_csv, overwrite):
        trial_summary.merge(trial_spear[["conc", "rho", "p"]], on="conc", how="left").to_csv(
            summ_csv, index=False, float_format="%.4f"
        )
        print(f"[trial_pos] Wrote {summ_csv}")

    f1_score = out_dir / "trial_position_mean_score.png"
    if should_write(f1_score, overwrite):
        plot_trial_position(
            trial_summary, trial_spear,
            value="mean_score", err="sem_score",
            ylabel=SCORE_Y_LABEL,
            title=f"RandomPanel: mean score by trial position{geno}",
            png_path=f1_score, reaction_line=True,
        )
    f1_react = out_dir / "trial_position_pct_react.png"
    if should_write(f1_react, overwrite):
        plot_trial_position(
            trial_summary, trial_spear,
            value="pct_react", err=("ci_lo", "ci_hi"),
            ylabel=PERCENT_Y_LABEL,
            title=f"RandomPanel: % reaction (score ≥ 2) by trial position{geno}",
            png_path=f1_react, as_pct=True,
        )

    # --- Figure 2: per-odor response by trial position ---------------------
    odor_rows: list[pd.DataFrame] = []
    for pos_col, tag, xlabel in (
        ("trial_num", "trial", "Trial position (1..14)"),
        ("blockpos", "blockpos", "Within-block position (1..7, exposures pooled)"),
    ):
        odor_summary = aggregate(df, ["conc", "odor", pos_col])
        odor_spear = spearman_by_group(df, ["conc", "odor"], pos_col)
        merged = odor_summary.merge(
            odor_spear[["conc", "odor", "rho", "p"]], on=["conc", "odor"], how="left"
        )
        merged["pos_col"] = pos_col
        odor_rows.append(merged.rename(columns={pos_col: "position"}))

        for conc in [c for c in CONC_PLOT_ORDER if c in set(df["conc"])]:
            sub = odor_summary[odor_summary["conc"] == conc]
            sp = odor_spear[odor_spear["conc"] == conc]
            cs = f"{conc:g}".replace(".", "p")
            score_png = out_dir / f"by_odor_{tag}_meanscore_conc-{cs}.png"
            if should_write(score_png, overwrite):
                plot_odor_grid(
                    sub, sp, conc=conc, pos_col=pos_col,
                    value="mean_score", err="sem_score",
                    ylabel=SCORE_Y_LABEL, xlabel=xlabel,
                    title=f"RandomPanel conc {conc:g}: mean score by {xlabel.split(' (')[0].lower()} per odorant{geno}",
                    png_path=score_png, reaction_line=True,
                )
            react_png = out_dir / f"by_odor_{tag}_pctreact_conc-{cs}.png"
            if should_write(react_png, overwrite):
                plot_odor_grid(
                    sub, sp, conc=conc, pos_col=pos_col,
                    value="pct_react", err=("ci_lo", "ci_hi"),
                    ylabel=PERCENT_Y_LABEL, xlabel=xlabel,
                    title=f"RandomPanel conc {conc:g}: % reaction by {xlabel.split(' (')[0].lower()} per odorant{geno}",
                    png_path=react_png, as_pct=True,
                )

    odor_csv = out_dir / "by_odor_trial_position_summary.csv"
    if should_write(odor_csv, overwrite):
        pd.concat(odor_rows, ignore_index=True).sort_values(
            ["pos_col", "conc", "odor", "position"]
        ).to_csv(odor_csv, index=False, float_format="%.4f")
        print(f"[trial_pos] Wrote {odor_csv}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fly-type", type=str, default="GR5a-Old")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    generate(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        fly_type=args.fly_type,
        config=args.config,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
