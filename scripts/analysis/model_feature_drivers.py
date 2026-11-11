"""Which of the scoring model's inputs drive its scores on one cohort?

The ordinal scorer (``reaction_prediction.model_path``) is an XGBoost softmax
model over 24 per-trial inputs. For the pre-test and post-training odor trials
of the given wide tables this rebuilds those inputs with the scorer's own
``features_for_model`` (fill values, epsilon and recipe all read off the
model), checks that they reproduce the stored scores, then ranks the inputs
three ways:

    gain_share   the model's own total split gain over all its trees: what it
                 leaned on when it was TRAINED (global, cohort-independent)
    shap_share   mean |SHAP contribution| of each input to the margin of the
                 class each trial was given: what decided THESE trials' scores
    rho_*        Spearman correlation of each input with the stored score, per
                 phase and pooled. Partly circular -- the score is computed
                 from these inputs -- so it says what tracks the model, not what
                 the fly did; agreement with hand scores is the independent check.

Outputs (``--out-dir``):
    model_feature_drivers.csv       one row per input, sorted by shap_share
    model_features_per_trial.csv    every trial's 24 inputs, stored + recomputed score
    Model_feature_drivers.png       the three rankings side by side

Usage
-----
    python3 scripts/analysis/model_feature_drivers.py \
        --pretest-wide-csv <folder>/_inputs/wide_pretest.parquet \
        --testing-wide-csv <folder>/_inputs/wide_testing.parquet \
        --predictions-csv <folder>/_inputs/predictions_raster.csv \
        --config config/config_new.yaml \
        --out-dir <folder>/Model-Feature-Drivers --label "start 1pm-2pm"
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from flybehavior_response import xgb_ordinal as xo  # noqa: E402

from fbpipe.utils.frozen_folders import drop_frozen  # noqa: E402
from fbpipe.utils.tables import read_table  # noqa: E402
from scripts.analysis import auc_before_during as abd  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis import pretest_vs_test_traces as pvt  # noqa: E402

PHASE_NAMES = {"pretest": "Pre-test", "testing": "Post-training"}
BAR_COLOR = "#4a5f78"
INK = abd.INK
INK_MUTED = abd.INK_MUTED


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------


def predict_classes(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """The class index (0..k-1) a softmax booster gives each row."""
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    return booster.predict(dm).astype(int)


def attribution(contrib: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Mean |contribution| per feature to the margin of each row's own class.

    ``contrib`` is XGBoost's ``pred_contribs`` output for a multiclass model,
    shaped (rows, classes, features + bias); the bias column is dropped.
    """
    own = contrib[np.arange(len(classes)), classes, :-1]
    return np.abs(own).mean(axis=0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x[ok], y[ok]).statistic)


def driver_table(
    booster: xgb.Booster, X: pd.DataFrame, score: np.ndarray, phase: np.ndarray
) -> pd.DataFrame:
    """One row per input: training gain share, attribution share, rho per phase."""
    feats = list(X.columns)
    gain = pd.Series(booster.get_score(importance_type="total_gain")).reindex(feats).fillna(0.0)
    dm = xgb.DMatrix(X, feature_names=feats)
    contrib = booster.predict(dm, pred_contribs=True)
    shap = attribution(contrib, predict_classes(booster, X))
    score = np.asarray(score, dtype=float)
    phase = np.asarray(phase)
    rows = []
    for i, f in enumerate(feats):
        x = X[f].to_numpy(float)
        row = {"feature": f,
               "gain_share": float(gain[f] / gain.sum()) if gain.sum() else float("nan"),
               "mean_abs_shap": float(shap[i]),
               "shap_share": float(shap[i] / shap.sum()) if shap.sum() else float("nan")}
        for ph in PHASE_NAMES:
            m = phase == ph
            row[f"rho_{ph}"] = _spearman(x[m], score[m])
        row["rho_all"] = _spearman(x, score)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("shap_share", ascending=False,
                                          ignore_index=True)


def score_agreement(predicted: np.ndarray, stored: np.ndarray) -> tuple[int, int]:
    """``(matching, compared)``: trials whose recomputed score equals the stored one."""
    predicted = np.asarray(predicted, dtype=float)
    stored = np.asarray(stored, dtype=float)
    ok = np.isfinite(stored)
    return int((predicted[ok] == stored[ok]).sum()), int(ok.sum())


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _model_path(args) -> Path:
    if args.model_path is not None:
        return Path(args.model_path)
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    return Path(cfg["reaction_prediction"]["model_path"])


def _odor_trials(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    df = drop_frozen(df.copy())
    df = df.loc[df["trial_type"].astype(str).str.strip().str.lower() == phase]
    df = ev._normalise_fly_columns(df)
    df["dataset"] = df["dataset"].map(ev._canon_dataset)
    keep = [pvt.odor_name(ds, label) is not None
            for ds, label in zip(df["dataset"], df["trial_label"])]
    return df.loc[keep].reset_index(drop=True)


def build_trials(
    tables: dict[str, pd.DataFrame], booster: xgb.Booster, predictions_csv: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Every odor trial's model inputs, recomputed score and stored score."""
    cfg = xo.scoring_config(booster)
    frames = []
    for phase, df in tables.items():
        trials = _odor_trials(df, phase)
        X = xo.features_for_model(trials, cfg).reset_index(drop=True)
        meta = trials[["dataset", "fly", "fly_number", "trial_label"]].astype(str)
        meta = meta.assign(phase=phase,
                           odor=[pvt.odor_name(d, lab) for d, lab in
                                 zip(meta["dataset"], meta["trial_label"])])
        meta["recomputed_score"] = xo.predict_ordinal_scores(booster, X)
        frames.append(pd.concat([meta, X], axis=1))
    t = pd.concat(frames, ignore_index=True)
    t = abd.attach_scores(t, predictions_csv)
    return t, list(cfg.features)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot_drivers(table: pd.DataFrame, title: str, note: str, *, cohorts: Sequence[str] = ()):
    """``cohorts``: the datasets drawn; all *-Sensitivity-* -> pre-/post-exposure legend."""
    n = len(table)
    y = np.arange(n)[::-1]                       # top-ranked input at the top
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 0.33 * n + 2.9),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    strongest = {table.loc[table["gain_share"].idxmax(), "feature"],
                 table.loc[table["rho_all"].idxmax(), "feature"], table.loc[0, "feature"]}
    for ax, col, head in (
        (axes[0], "gain_share", "Model's training importance\n(share of total split gain)"),
        (axes[1], "shap_share", "Attribution on these trials\n(share of mean |SHAP|)"),
    ):
        v = table[col].to_numpy(float) * 100
        ax.barh(y, v, height=0.68, color=BAR_COLOR, zorder=2)
        for yi, vi in list(zip(y, v))[:5]:                # label the top five only
            ax.text(vi + v.max() * 0.015, yi, f"{vi:.0f}%", va="center", fontsize=8,
                    color=INK)
        ax.set_xlim(0, v.max() * 1.15)
        ax.set_xlabel("% of total")
        ax.set_title(head, fontsize=10.5, color=INK, loc="left")
    ax = axes[2]
    ax.axvline(0, color=INK_MUTED, lw=0.8, zorder=1)
    cohort = abd.wording_cohort(cohorts)
    for col, color, marker, label in (
        ("rho_pretest", abd.PHASE_COLOR["Pre-test"], "o", abd.phase_label("Pre-test", cohort)),
        ("rho_testing", abd.PHASE_COLOR["Post-training"], "o",
         abd.phase_label("Post-training", cohort)),
        ("rho_all", INK, "D", "Pre + post"),
    ):
        ax.scatter(table[col], y, s=30 if marker == "o" else 26, color=color,
                   marker=marker, edgecolors="white", linewidths=0.5, zorder=3,
                   label=label)
    ax.set_xlim(-0.45, 1.0)
    ax.set_xlabel("Spearman rho with the model score")
    ax.set_title("Correlation with the score\n(per trial)", fontsize=10.5, color=INK,
                 loc="left")
    # The longer pre-/post-exposure labels push a lower-right legend onto the dots
    ax.legend(loc="best" if cohort else "lower right", frameon=False, fontsize=8.5)
    for i, a in enumerate(axes):
        a.set_ylim(-0.7, n - 0.3)
        a.set_yticks(y)
        if i == 0:
            a.set_yticklabels(table["feature"], fontsize=8.5)
            for tick, f in zip(a.get_yticklabels(), table["feature"]):
                if f in strongest:
                    tick.set_fontweight("bold")
        else:
            a.tick_params(labelleft=False)
        a.grid(axis="x", color="#e6e6e3", lw=0.6, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=12)
    fig.text(0.01, 0.005, note, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom",
             wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pretest-wide-csv", required=True, type=Path)
    p.add_argument("--testing-wide-csv", required=True, type=Path)
    p.add_argument("--predictions-csv", required=True, type=Path,
                   help="The stored scores the ranking is checked against.")
    p.add_argument("--out-dir", required=True, type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path,
                     help="Pipeline config; its reaction_prediction.model_path is used.")
    src.add_argument("--model-path", type=Path)
    p.add_argument("--label", default="", help="Appended to the figure title.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = _model_path(args)
    booster = xo.load_xgb_model(model)
    tables = {"pretest": read_table(args.pretest_wide_csv),
              "testing": read_table(args.testing_wide_csv)}
    t, feats = build_trials(tables, booster, args.predictions_csv)
    match, compared = score_agreement(t["recomputed_score"], t["score"].astype(float))
    print(f"[model-drivers] {len(t)} odor trials; recomputed score == stored for "
          f"{match} of {compared}")
    if compared and match < compared:
        print("[model-drivers] WARNING: the model no longer reproduces every stored "
              "score -- predictions were made with different inputs or another model")
    score = t["score"].astype(float).to_numpy()
    score = np.where(np.isfinite(score), score, t["recomputed_score"].to_numpy(float))
    table = driver_table(booster, t[feats], score, t["phase"].to_numpy())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "model_feature_drivers.csv", index=False)
    t.to_csv(out / "model_features_per_trial.csv", index=False)

    cfg = xo.scoring_config(booster)
    n_flies = t[["dataset", "fly", "fly_number"]].drop_duplicates().shape[0]
    datasets = ", ".join(sorted(t["dataset"].unique()))
    label = f" ({args.label})" if args.label else ""
    title = (f"{datasets}{label}: which scorer input drives the scores "
             f"({n_flies} flies, {len(t)} odor trials)")
    note = (
        f"Model: {model.name} (recipe {cfg.recipe}). Inputs rebuilt from the wide tables "
        f"with flybehavior_response.xgb_ordinal.features_for_model; the recomputed score "
        f"equals the stored score for {match} of {compared} trials. Training importance = "
        "each input's share of the model's total split gain over all trees (global, fixed "
        "at training). Attribution = mean |SHAP contribution| of each input to the margin "
        "of the class each trial was given, over these trials. rho = Spearman correlation "
        "of each input with the stored score; partly circular, since the score is computed "
        "from these inputs. Rows sorted by attribution; bold = top by any of the three."
    )
    fig = plot_drivers(table, title, note, cohorts=t["dataset"].unique())
    fig.savefig(out / "Model_feature_drivers.png", dpi=200)
    plt.close(fig)
    print(f"[model-drivers] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
