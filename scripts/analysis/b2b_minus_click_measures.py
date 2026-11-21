"""B2B IAA odor pairs: each partner's change minus the paraffin click's change, on the
scorer's log AUC ratio and on the PER-trace peak-z ratio.

User request 2026-11-18: "want to make these [the Sensitivity cross-cohort log AUC and
peak-z ratio change-minus-control figures] but comparing the same way we are doing for
the b2b for the paraffin oil for both datasets". The control is within the fly: the
"Paraffin Oil" partner is a master-valve click with no odor (since 2026-11-10), in the
same odor-2 slot as every other partner, so it is a time-matched control.

* IAA-PairFirst: the scored odor is partner X as odor 2 right after IAA; naive = the
  fly's pre-test trial of X.
* IAA-PairSecond: the scored odor is IAA as odor 2 right after X; naive = the fly's IAA
  pre-test trial.

Per fly f and partner X: Delta_X = (mean of the fly's in-pair odor-2 trials of X, both
rounds) - (its naive value), and V_f = Delta_X - Delta_click. A bar pools trials over the
flies with both X and the click: (sum of in-pair X trials / count - sum of naive X trials
/ count) - (the same for the click), never a mean of per-fly means. The click's own
change (in-pair - naive, every fly with it) is a hatched column on the right.

Tests (scripts/analysis/b2b_stats.py): the headline p flips whole batches (batch folder =
the ``fly`` column) of V together, ``batch_sign_flip``, and its 95 % CI inverts that test
(the CI of the per-fly mean of V, so it clears 0 exactly when p < 0.05); ``p_fly`` is the
fly-level exact sign-flip, CSV only. Holm runs over every bar of a panel (both datasets'
partners and both click bars); stars come only from the Holm-adjusted batch p; the raw
batch p is printed on every bar (user rule: p on every odor).

Measures, one value per trial (the naive pre-test rows and production's split odor-2
rows, ``fbpipe.utils.odor_pairs.split_pair_rows``: the pre-odor-1 block, then everything
from odor 2's valve onset, timed as a normal trial):

  log_auc      the scoring model's input ``AUC-During-Before-LogRatio`` as production
               computes it: ``window_rescore.model_features`` on the odor window with the
               model at ``reaction_prediction.model_path`` (its arrival delay, its own AUC
               threshold, its fill values for a missing value, its eps). If a model's
               feature list lacks the column, ln(AUC-During + eps) - ln(AUC-Before + eps)
               is computed from ``window_rescore.raw_features`` with the model's eps.
  peakz_ratio  ``auc_before_during.trial_peak_z`` on the row's ``dir_val_*`` trace (cut at
               ``trace_len``, the Sensitivity folder's trace) with the row's own fps and
               odor window: ``zpeak_ratio_signed`` (peak z during, sign kept, / |peak z
               before|) and ``zpeak_ratio_unsigned`` (|during| / |before|). Before = the
               odor window's length just ahead of onset (odor 2: before odor 1).

Carry-over: odor 2 starts 11 s after odor 1's onset but is scored against the block
before odor 1. In PairFirst, X and the click both follow IAA, which largely cancels IAA's
tail; in PairSecond, "IAA after X" vs "IAA after the click" still carries X's own late
response.

Outputs (``<out>/Change-Minus-Click/``):

  b2b_log_auc_change_minus_click.{png,pdf,csv}
  b2b_peakz_ratio_change_minus_click.{png,pdf,csv}
  b2b_measures_per_trial.csv     every trial measured, naive and in-pair

and, per group (``<out>/<group>/``, user 2026-11-20: "just the naive compared to the test",
laid out like ``<group>_scores_pooled``): ``<group>_peakz_pooled.{png,pdf,csv}`` and
``_stats.csv``, each odor's naive peak-z ratio vs in the pair (rounds 1 + 2), no click
subtracted, exact paired sign-flip per fly, Holm over the panel's odors (secondary: not
controlled for time in the rig).

Entry point: :func:`write`. Called by b2b_pair_figures' ``build()``; this module imports
b2b_pair_figures only inside functions, so there is no import cycle.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fbpipe.analysis import fly_stats as fs  # noqa: E402
from fbpipe.utils import odor_pairs as op  # noqa: E402
from scripts.analysis import b2b_stats as bs  # noqa: E402

OUT_DIR = "Change-Minus-Click"
PER_TRIAL_CSV = "b2b_measures_per_trial.csv"
CLICK = "Paraffin Oil"
SCORED_SECOND = "IAA"
NAIVE, IN_PAIR = "naive", "in_pair"
KINDS = ("partner minus click", "click's own change")
LOG_AUC_FEATURE = "AUC-During-Before-LogRatio"
META = ["dataset", "fly", "fly_number", "trial_label", "role", "scored_odor", "partner",
        "round"]
PEAKZ_COLUMNS = ("zpeak_before_unsigned", "zpeak_during_signed", "zpeak_during_unsigned",
                 "zpeak_ratio_signed", "zpeak_ratio_unsigned")

#: measure -> panels (trial column, y label), the title phrase, the measure's definition.
MEASURES: dict[str, dict] = {
    "log_auc": {
        "panels": (("log_auc", "Change minus the click's change in\nln(AUC during / AUC "
                               "before)\n(the scorer's input)"),),
        "phrase": "log AUC during / before",
    },
    "peakz_ratio": {
        "panels": (("zpeak_ratio_signed", "Change minus the click's change in\nsigned "
                                          "peak-z ratio (during / |before|)"),
                   ("zpeak_ratio_unsigned", "Change minus the click's change in\nunsigned "
                                            "peak-z ratio (|during| / |before|)")),
        "phrase": "the PER-trace peak z-score, during / before",
    },
}
TEST_TEXT = ("whole batches sign-flipped together on one value per fly (its change minus its "
             "click change; click bar: its own change), two-sided, 95% CI by test inversion; "
             "Holm over every bar of the panel")

INK, INK_2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f4f3f0"


def _b2b():
    """b2b_pair_figures, imported on use (it imports this module inside build())."""
    from scripts.analysis import b2b_pair_figures as b

    return b


# ── which trials ─────────────────────────────────────────────────────────────
def _label_odors(label: str):
    """(phase, number, [odor names]) of a wide-table label, the odor-2 suffix dropped."""
    s = str(label)
    if s.endswith(op.ODOR2_SUFFIX):
        s = s[: -len(op.ODOR2_SUFFIX)]
    return _b2b().parse_label(s)


def naive_and_pair_rows(pre_df: pd.DataFrame, pair_df: pd.DataFrame,
                        odor2_window: Callable) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(naive rows, odor-2 rows): wide rows of the two datasets, each with ``role``,
    ``scored_odor``, ``partner`` and ``round`` added.

    Odor-2 rows are production's (``op.split_pair_rows``), one per pair trial whose odor-2
    window ``odor2_window(row)`` finds; ``round`` counts a partner's pair trials in trial
    order (1, 2), as ``b2b_pair_figures.build_records`` does. Naive rows are the single-odor
    pre-test trials that serve as a baseline: every odor's in PairFirst, IAA's in
    PairSecond (``partner`` blank: the same trial is every partner's naive value)."""
    b = _b2b()

    def ours(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "dataset" not in df.columns:
            return pd.DataFrame(columns=[*([] if df is None else df.columns)])
        return df[df["dataset"].isin(b.DATASETS)].reset_index(drop=True)

    pre, pair = ours(pre_df), ours(pair_df)
    meta_new = ["role", "scored_odor", "partner", "round"]

    keep = [(r.name, odor) for (ds, _fly, _fn, odor), r in b._pretest_by(pre).items()
            if ds == b.FIRST or odor == SCORED_SECOND]
    naive_df = pre.loc[[i for i, _ in keep]].assign(
        role=NAIVE, scored_odor=[o for _, o in keep], partner="", round=0)

    if pair.empty:
        return naive_df.reset_index(drop=True), pd.DataFrame(columns=[*pair.columns, *meta_new])
    split, _ = op.split_pair_rows(pair, odor2_window)
    o2 = split[split["trial_label"].astype(str).str.endswith(op.ODOR2_SUFFIX)]
    where = {(r.dataset, r.fly, r.fly_number, r.trial_label): i
             for i, r in zip(o2.index, o2[["dataset", "fly", "fly_number", "trial_label"]]
                             .itertuples(index=False))}
    seen: dict[tuple, int] = {}
    picked, info = [], []
    for _num, (o1, o2_odor), r in b._pair_rows(pair):
        ds, fly, fn = r["dataset"], r["fly"], r["fly_number"]
        partner = o2_odor if ds == b.FIRST else o1
        k = (ds, fly, fn, partner)
        seen[k] = seen.get(k, 0) + 1
        i = where.get((ds, fly, fn, f"{r['trial_label']}{op.ODOR2_SUFFIX}"))
        if i is None:
            continue
        picked.append(i)
        info.append((IN_PAIR, o2_odor, partner, seen[k]))
    o2 = o2.loc[picked].copy()
    for col, vals in zip(meta_new, zip(*info) if info else ([],) * 4):
        o2[col] = list(vals)
    return naive_df.reset_index(drop=True), o2.reset_index(drop=True)


# ── measures ─────────────────────────────────────────────────────────────────
def _dir_cols(df: pd.DataFrame) -> list[str]:
    return sorted((c for c in df.columns if str(c).startswith("dir_val_")),
                  key=lambda c: int(str(c)[8:]))


def peak_z_values(rows: pd.DataFrame) -> pd.DataFrame:
    """``auc_before_during.trial_peak_z`` of every row, on the trace the Sensitivity
    folder uses (``pretest_vs_test_traces._to_trace``: ``dir_val_*`` cut at
    ``trace_len``; the odor window only when off > on > 0)."""
    from scripts.analysis import auc_before_during as abd
    from scripts.analysis import envelope_visuals as ev

    cols = _dir_cols(rows)
    out = []
    for _, r in rows.iterrows():
        env = ev._extract_env_row(r[cols].to_numpy(dtype=float), trace_len=r.get("trace_len"))
        fps = pd.to_numeric(r.get("fps"), errors="coerce")
        on = pd.to_numeric(r.get("trial_odor_on_s"), errors="coerce")
        off = pd.to_numeric(r.get("trial_odor_off_s"), errors="coerce")
        ok = np.isfinite(on) and np.isfinite(off) and off > on > 0
        if env.size == 0:
            out.append({c: np.nan for c in PEAKZ_COLUMNS})
            continue
        z = abd.trial_peak_z(env, fps=float(fps) if np.isfinite(fps) and fps > 0 else np.nan,
                             odor_on_s=float(on) if ok else np.nan,
                             odor_off_s=float(off) if ok else np.nan)
        out.append({c: z[c] for c in PEAKZ_COLUMNS})
    return pd.DataFrame(out, columns=list(PEAKZ_COLUMNS), index=rows.index)


def _log_auc(rows: pd.DataFrame, cfg, rule, wr) -> np.ndarray:
    """The model input ``AUC-During-Before-LogRatio`` of every row on the odor window
    (``wr.model_features``: production's matrix); when the model's feature list lacks it,
    ln(AUC-During + eps) - ln(AUC-Before + eps) from ``wr.raw_features`` (areas clipped
    at 0, as ``xgb_ordinal.auc_log_ratios`` does) with the model's eps."""
    spec = wr.window_spec("odor")
    if LOG_AUC_FEATURE in list(cfg.features):
        feats = wr.model_features(rows, spec, cfg, rule=rule)
        if LOG_AUC_FEATURE in feats.columns:
            return feats[LOG_AUC_FEATURE].to_numpy(dtype=float)
    if cfg.auc_eps is None:
        raise ValueError(f"the model has no {LOG_AUC_FEATURE} and no auc_eps")
    raw = wr.raw_features(rows, spec, rule=rule, odor_delay_s=cfg.odor_delay_s)
    eps = float(cfg.auc_eps)
    during = raw["AUC-During"].astype(float).clip(lower=0)
    before = raw["AUC-Before"].astype(float).clip(lower=0)
    return (np.log(during + eps) - np.log(before + eps)).to_numpy(dtype=float)


def _log_auc_scorer(model_path: Path, config: Path) -> Callable[[pd.DataFrame], np.ndarray]:
    """``rows -> log AUC`` with the model loaded once."""
    from scripts.analysis import window_rescore as wr

    cfg = wr.xo.scoring_config(wr.xo.load_xgb_model(model_path))
    rule = wr.threshold_rule_from_config(config)
    return lambda rows: _log_auc(rows, cfg, rule, wr)


def model_eps(config: Path | None) -> tuple[float | None, float | None]:
    """(auc_eps, odor delay) stored on the config's model, for the figure note."""
    model = _model_path(config)
    if model is None:
        return None, None
    try:
        from scripts.analysis import window_rescore as wr

        cfg = wr.xo.scoring_config(wr.xo.load_xgb_model(model))
        return cfg.auc_eps, cfg.odor_delay_s
    except Exception:  # noqa: BLE001 - only the note loses the numbers
        return None, None


def _model_path(config: Path | None) -> Path | None:
    if config is None or not Path(config).is_file():
        return None
    return _b2b().plus_after_model(Path(config))


def trial_measures(frames: Sequence[pd.DataFrame], *, config: Path | None) -> pd.DataFrame:
    """One row per trial of ``frames`` (naive rows, odor-2 rows): ``META``, timing, the
    peak-z values and ``log_auc``. ``log_auc`` is left out (with a ``[b2b][WARN]``) when
    the config names no model or the model cannot score; each frame keeps its own
    columns, as production scored each table."""
    scorer = None
    model = _model_path(config)
    if model is None:
        print("[b2b][WARN] Change-Minus-Click: no reaction_prediction.model_path in the "
              "config: log AUC figure skipped")
    else:
        try:
            scorer = _log_auc_scorer(model, Path(config))
        except Exception as exc:  # noqa: BLE001 - a scorer problem must not cost peak z
            print(f"[b2b][WARN] Change-Minus-Click: model {model} unusable, log AUC figure "
                  f"skipped: {exc}")
    parts = []
    for rows in frames:
        if rows is None or rows.empty:
            continue
        rows = rows.reset_index(drop=True)
        t = rows[META].copy()
        t["fps"] = pd.to_numeric(rows["fps"], errors="coerce")
        t["odor_on_s"] = pd.to_numeric(rows["trial_odor_on_s"], errors="coerce")
        t["odor_off_s"] = pd.to_numeric(rows["trial_odor_off_s"], errors="coerce")
        t = pd.concat([t, peak_z_values(rows)], axis=1)
        if scorer is not None:
            try:
                t["log_auc"] = scorer(rows)
            except Exception as exc:  # noqa: BLE001
                print(f"[b2b][WARN] Change-Minus-Click: log AUC failed, figure skipped: {exc}")
                scorer = None
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=META)
    out = pd.concat(parts, ignore_index=True)
    if scorer is None and "log_auc" in out.columns:
        out = out.drop(columns=["log_auc"])
    return out


# ── contrasts ────────────────────────────────────────────────────────────────
def _change(trials: pd.DataFrame, dataset: str, partner: str, col: str, flies=None):
    """(pooled change, per-fly change, n in-pair trials, n naive trials) from naive to
    in-pair for ``partner`` over the flies with both (and in ``flies`` when given). Pooled:
    trial sum / trial count of each side; per fly: the mean of its in-pair trials minus
    its naive value. Per-fly index: (fly, fly_number); the fly column is the batch."""
    key = ["fly", "fly_number"]
    d = trials[trials["dataset"] == dataset]
    d = d.assign(_v=pd.to_numeric(d[col], errors="coerce").astype(float))
    d = d[np.isfinite(d["_v"].to_numpy(dtype=float))]
    pair = d[(d["role"] == IN_PAIR) & (d["partner"] == partner)]
    empty = pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=key))
    if pair.empty:
        return np.nan, empty, 0, 0
    naive = d[(d["role"] == NAIVE) & d["scored_odor"].isin(set(pair["scored_odor"]))]
    pm, nm = pair.groupby(key)["_v"].mean(), naive.groupby(key)["_v"].mean()
    both = pm.index.intersection(nm.index)
    if flies is not None:
        both = both.intersection(flies)
    if not len(both):
        return np.nan, empty, 0, 0
    pair = pair[pd.MultiIndex.from_frame(pair[key]).isin(both)]
    naive = naive[pd.MultiIndex.from_frame(naive[key]).isin(both)]
    pooled = float(pair["_v"].mean() - naive["_v"].mean())
    return pooled, (pm.loc[both] - nm.loc[both]).sort_index(), int(len(pair)), int(len(naive))


def partner_order(trials: pd.DataFrame) -> list[str]:
    """The partners present, in ``b2b_pair_figures.ORDER``, the click last."""
    present = set(trials.loc[trials["role"] == IN_PAIR, "partner"])
    return [p for p in _b2b().ORDER if p in present and p != CLICK] + (
        [CLICK] if CLICK in present else [])


def _test_row(values: pd.Series) -> dict:
    v = values.to_numpy(dtype=float)
    batches = values.index.get_level_values("fly").astype(str)
    r = bs.batch_sign_flip(v, batches, ci=True)
    return {"n_flies": int(r["n_flies"]), "n_batches": int(r["n_batches"]),
            "mean_per_fly": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            "p_batch": r["p"], "p_batch_exact": bool(r["exact"]),
            "p_fly": bs.sign_flip_p(v) if v.size else np.nan,
            "batches": ";".join(sorted(set(batches)))}


def panel_rows(trials: pd.DataFrame, col: str) -> list[dict]:
    """Every bar of one panel (column ``col``), datasets in b2b order: each partner's
    change minus the click's over the flies with both, then the click's own change."""
    b = _b2b()
    rows = []
    for ds in [d for d in b.DATASETS if d in set(trials["dataset"])]:
        _, c_all, *_ = _change(trials, ds, CLICK, col)
        if c_all.empty:
            print(f"[b2b][WARN] Change-Minus-Click: {ds} has no {CLICK} click trials with "
                  f"{col}: no bars")
            continue
        sub = trials[trials["dataset"] == ds]
        for partner in [p for p in partner_order(sub) if p != CLICK]:
            _, o_all, *_ = _change(trials, ds, partner, col)
            flies = o_all.index.intersection(c_all.index)
            if not len(flies):
                continue
            o_pooled, o_fly, n_op, n_on = _change(trials, ds, partner, col, flies)
            c_pooled, c_fly, n_cp, n_cn = _change(trials, ds, CLICK, col, flies)
            scored = sub.loc[(sub["role"] == IN_PAIR) & (sub["partner"] == partner),
                             "scored_odor"].iloc[0]
            rows.append({"dataset": ds, "group": b.GROUP_SHORT[ds], "partner": partner,
                         "kind": KINDS[0], "scored_odor": scored,
                         "n_trials_partner_pair": n_op, "n_trials_partner_naive": n_on,
                         "n_trials_click_pair": n_cp, "n_trials_click_naive": n_cn,
                         "partner_change": o_pooled, "click_change": c_pooled,
                         "value": o_pooled - c_pooled, **_test_row(o_fly - c_fly)})
        c_pooled, c_fly, n_cp, n_cn = _change(trials, ds, CLICK, col)
        scored = sub.loc[(sub["role"] == IN_PAIR) & (sub["partner"] == CLICK),
                         "scored_odor"].iloc[0]
        rows.append({"dataset": ds, "group": b.GROUP_SHORT[ds], "partner": CLICK,
                     "kind": KINDS[1], "scored_odor": scored,
                     "n_trials_partner_pair": 0, "n_trials_partner_naive": 0,
                     "n_trials_click_pair": n_cp, "n_trials_click_naive": n_cn,
                     "partner_change": np.nan, "click_change": c_pooled, "value": c_pooled,
                     **_test_row(c_fly)})
    return rows


def contrast_table(trials: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Every bar of every panel of ``measure``; Holm over each panel's bars (one
    exploratory family: both datasets' partners and both click bars)."""
    out = []
    for col, _ylabel in MEASURES[measure]["panels"]:
        if col not in trials.columns:
            continue
        rows = panel_rows(trials, col)
        if not rows:
            continue
        holm = fs.holm_adjust([r["p_batch"] for r in rows])
        m = int(np.isfinite(np.asarray([r["p_batch"] for r in rows], dtype=float)).sum())
        for r, h in zip(rows, holm):
            r.update(measure=measure, panel=col, p_batch_holm=float(h), n_tests_in_family=m,
                     stars=fs.stars(h).replace("n.s.", ""))
        out += rows
    if not out:
        return pd.DataFrame()
    first = ["measure", "panel", "dataset", "group", "partner", "kind", "scored_odor"]
    t = pd.DataFrame(out)
    return t[first + [c for c in t.columns if c not in first]]


# ── figure ───────────────────────────────────────────────────────────────────
def _p_short(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def _wrap(text: str, width_in: float, pt: float = 8.3) -> str:
    """Wrap to the figure's width (about 15 characters per inch at 8.3 pt)."""
    return "\n".join(textwrap.fill(par, width=max(60, int(width_in * 15 * 8.3 / pt)))
                     for par in text.split("\n"))


def no_star_note(table: pd.DataFrame) -> str:
    """When the batch counts make every Holm-adjusted p >= 0.05, say so: an exact
    batch sign-flip over k batches cannot go below 2 / 2^k."""
    t = table[table["panel"] == table["panel"].iloc[0]]
    m = int(t["n_tests_in_family"].max())
    k = t.groupby("group")["n_batches"].max()
    floor = {g: 2.0 / 2 ** int(n) for g, n in k.items() if n > 0}
    if not floor or m * min(floor.values()) < 0.05:
        return ""
    parts = " / ".join(f"{int(k[g])} {g} batches: {_p_short(p).replace('p=', '')}"
                       for g, p in floor.items())
    return (f"No bar can reach Holm-adjusted p < 0.05 here: the smallest batch p possible "
            f"({parts}) times {m} contrasts is above 0.05.")


def plot_contrasts(table: pd.DataFrame, measure: str, title: str, path: Path, *,
                   definition: str = "") -> list[Path]:
    """A panel per measure column: per partner, one bar per dataset (its change minus the
    click's change); the click's own change hatched on the right. PNG (200 dpi) + PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    b = _b2b()
    colours: Mapping[str, str] = b.GROUP_COLORS
    datasets = [d for d in b.DATASETS if d in set(table["dataset"])]
    panels = [(c, y) for c, y in MEASURES[measure]["panels"] if c in set(table["panel"])]
    odors = [p for p in b.ORDER if p in set(table["partner"]) and p != CLICK]
    xc = {p: float(i) for i, p in enumerate(odors)}
    xc[CLICK] = len(odors) + 0.35                       # set apart from the odors
    k = len(datasets)
    step = 0.80 / max(k, 1)
    width_in = max(8.0, 1.45 * (len(odors) + 1) + 2.0)
    top_in, bottom_in = 1.95, 1.45
    height_in = top_in + bottom_in + 3.9 * len(panels) + 0.75 * (len(panels) - 1)
    fig, axes = plt.subplots(len(panels), 1, figsize=(width_in, height_in), squeeze=False,
                             gridspec_kw={"hspace": 0.75 / 3.9})
    hatch = "////"
    for ax, (col, ylab) in zip(axes[:, 0], panels):
        t = table[table["panel"] == col].set_index(["dataset", "partner"])
        for i in range(len(odors)):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
        lo, hi = 0.0, 0.0
        for j, ds in enumerate(datasets):
            c = colours[ds]
            for partner in odors + [CLICK]:
                if (ds, partner) not in t.index:
                    continue
                r = t.loc[(ds, partner)]
                x = xc[partner] + (j - (k - 1) / 2) * step
                y = float(r["value"])
                if not np.isfinite(y):
                    continue
                ctl = partner == CLICK
                with plt.rc_context({"hatch.color": c, "hatch.linewidth": 0.9}):
                    ax.bar(x, y, width=step * 0.86, color="white" if ctl else c,
                           edgecolor=c if ctl else "none", hatch=hatch if ctl else None,
                           linewidth=1.3 if ctl else 0, zorder=2)
                y_lo, y_hi = float(r["ci_lo"]), float(r["ci_hi"])
                has_ci = np.isfinite(y_lo) and np.isfinite(y_hi)
                if has_ci:
                    ax.errorbar(x, y, yerr=[[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]],
                                fmt="none", ecolor=INK_2, elinewidth=1.0, zorder=3)
                tip = (max(y, y_hi) if has_ci else y) if y >= 0 else (
                    min(y, y_lo) if has_ci else y)
                lo, hi = min(lo, tip), max(hi, tip)
                stars = "" if pd.isna(r["stars"]) else str(r["stars"])
                label = f"{_p_short(float(r['p_batch']))}{stars}"
                if label:
                    ax.annotate(label, (x, tip), xytext=(0, 3 if y >= 0 else -3),
                                textcoords="offset points", ha="center",
                                va="bottom" if y >= 0 else "top", fontsize=7, color=INK,
                                zorder=4)
        span = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.13 * span, hi + 0.13 * span)
        ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
        ax.axvline((len(odors) - 1 + xc[CLICK]) / 2, color=INK_2, lw=0.7,
                   linestyle=(0, (3, 3)), zorder=1)
        ax.set_xlim(-0.55, xc[CLICK] + 0.55)
        ax.set_xticks([xc[p] for p in odors] + [xc[CLICK]])
        ax.set_xticklabels(odors + [f"{CLICK} click\n(its own change)"], fontsize=10,
                           color=INK)
        ax.set_ylabel(ylab, fontsize=10, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2)
        n_tests = int(table.loc[table["panel"] == col, "n_tests_in_family"].max())
        ax.set_title(f"{n_tests} contrasts; * = Holm-adjusted p < 0.05 · exploratory",
                     loc="right", fontsize=8.5, color=INK_2)
    axes[-1, 0].set_xlabel("Partner odor (IAA-first: scored right after IAA; IAA-second: "
                           "presented right before IAA, which is scored)", fontsize=9.5,
                           color=INK)

    first = table[table["panel"] == panels[0][0]]
    n = first.groupby("dataset")[["n_flies", "n_batches"]].max()
    what = {b.FIRST: "partner scored right after IAA",
            b.SECOND: "IAA scored right after the partner"}
    handles = [Patch(color=colours[ds], label=(
        f"{ds} ({what[ds]}; n={int(n.loc[ds, 'n_flies'])} flies / "
        f"{int(n.loc[ds, 'n_batches'])} batches)")) for ds in datasets]
    handles.append(Patch(facecolor="white", edgecolor=INK_2, hatch=hatch,
                         label="hatched: the click's own change, in-pair − naive"))
    fig.suptitle(title, fontsize=13, color=INK, y=1 - 0.12 / height_in)
    note = ("Bar = the partner's change (in-pair odor-2 trials, rounds 1 + 2, minus the naive "
            "pre-test trial of the scored odor) minus the paraffin click's change (master-valve "
            "click, no odor, same odor-2 slot) in the same flies; trials pooled (sum / count), "
            "flies with both. IAA-first: naive = the partner's pre-test trial; IAA-second: "
            "naive = the fly's IAA pre-test trial. p: whole batches sign-flipped together on "
            "one value per fly (its change minus its click change); whiskers: 95% CI of that "
            "per-fly mean from inverting the test (clears 0 exactly when the raw p < 0.05). "
            "Every bar: its raw p; * p<0.05, ** p<0.01, *** p<0.001 only after Holm over "
            "every bar in the panel (exploratory).")
    fig.text(0.5, 1 - 0.40 / height_in, _wrap(note, width_in), ha="center", va="top",
             fontsize=8.3, color=INK_2)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9.3,
               labelcolor=INK, bbox_to_anchor=(0.5, 1 - 1.05 / height_in))
    foot = (f"{definition} Carry-over: odor 2 starts 11 s after odor 1's onset (10 s odor, "
            "1 s gap) but is scored against the block before odor 1. In IAA-first the "
            "partner and the click both follow IAA, so IAA's tail largely cancels. In "
            "IAA-second, IAA after the partner vs IAA after the click still carries the "
            "partner's own late response, so a bar there can be that tail rather than a "
            "change in the response to IAA. " + no_star_note(table)).strip()
    fig.text(0.5, 0.012, _wrap(foot, width_in), ha="center", va="bottom", fontsize=8.3,
             color=INK_2)
    fig.subplots_adjust(top=1 - top_in / height_in, bottom=bottom_in / height_in,
                        left=0.085, right=0.985)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    fig.savefig(path, dpi=200, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [path, pdf]


def _definition(measure: str, eps: float | None, delay: float | None) -> str:
    if measure == "log_auc":
        e = f", ε = {eps:.2f}" if eps is not None else ""
        d = f", +{delay:g} s odor-arrival delay" if delay else ""
        return ("Measure: ln((AUC during + ε) / (AUC before + ε)), the scoring model's "
                f"AUC-During-Before-LogRatio as production computes it (the model's AUC "
                f"threshold{d}{e}); odor 2's before = the block before odor 1.")
    return ("Measure: the PER trace's peak z in SDs of the window just before the odor, the "
            "odor window's length (odor 2: before odor 1); signed = peak during, sign kept, / "
            "|peak before|; unsigned = |during| / |before| (the Sensitivity folder's "
            "trial_peak_z).")


# ── naive vs in pair, no click subtraction (user 2026-11-20) ─────────────────
#: (trial column, y label): the two panels of ``<group>/<group>_peakz_pooled``, the
#: ``<group>_scores_pooled`` figure drawn on the peak-z ratio.
PEAKZ_POOLED_PANELS = (
    ("zpeak_ratio_signed", "Signed peak-z ratio\n(during / |before|)"),
    ("zpeak_ratio_unsigned", "Unsigned peak-z ratio\n(|during| / |before|)"))
PEAKZ_POOLED_STEM = "peakz_pooled"
PEAKZ_POOLED_TEST = ("exact paired sign-flip on per-fly differences (in-pair mean - alone), "
                     "two-sided; Holm over the odors of the panel; batch_level_p: whole batches "
                     "sign-flipped together, secondary")


def naive_vs_pair_table(trials: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bars, tests) of one dataset, per peak-z panel and partner (b2b order, click last).

    Bars ("pre": the naive trial of the scored odor; "pair": its in-pair odor-2 trials,
    rounds 1 + 2) pool the trials of the flies with both, sum / count, never a mean of fly
    means; 95 % CI with flies as the unit (``fs.fly_cluster_mean_ci``). Test, as the
    ``<group>_scores_pooled`` figure: exact paired sign-flip on one value per fly (its
    in-pair mean minus its naive value), Holm over the panel's partners; the batch-level
    sign-flip is kept as the secondary ``batch_level_p``."""
    b = _b2b()
    key = ["fly", "fly_number"]
    sub = trials[trials["dataset"] == dataset]
    bars, tests = [], []
    for col, _ylabel in PEAKZ_POOLED_PANELS:
        if col not in sub.columns:
            continue
        d = sub.assign(_v=pd.to_numeric(sub[col], errors="coerce").astype(float))
        d = d[np.isfinite(d["_v"].to_numpy(dtype=float))]
        for partner in partner_order(sub):
            pair = d[(d["role"] == IN_PAIR) & (d["partner"] == partner)]
            if pair.empty:
                continue
            naive = d[(d["role"] == NAIVE) & d["scored_odor"].isin(set(pair["scored_odor"]))]
            pm, nm = pair.groupby(key)["_v"].mean(), naive.groupby(key)["_v"].mean()
            both = pm.index.intersection(nm.index)
            if not len(both):
                continue
            head = {"panel": col, "dataset": dataset, "group": b.GROUP_SHORT[dataset],
                    "partner": partner, "scored_odor": pair["scored_odor"].iloc[0]}
            for k, rows in (("pre", naive), ("pair", pair)):
                rows = rows[pd.MultiIndex.from_frame(rows[key]).isin(both)]
                v = rows["_v"].to_numpy(dtype=float)
                ids = (rows["fly"].astype(str) + "|" + rows["fly_number"].astype(str)).to_numpy()
                mean, lo, hi = fs.fly_cluster_mean_ci(v, ids)
                bars.append({**head, "key": k, "condition": b.BAR_LABELS[k], "n": len(v),
                             "n_flies": int(len(both)), "mean": mean, "ci_lo": lo, "ci_hi": hi})
            diff = (pm.loc[both] - nm.loc[both]).sort_index()
            dv = diff.to_numpy(dtype=float)
            batch = bs.batch_sign_flip(dv, diff.index.get_level_values("fly").astype(str))
            tests.append({**head, "a": b.BAR_LABELS["pre"], "b": b.BAR_LABELS["pair"],
                          "n_flies": len(dv), "n_batches": int(batch["n_batches"]),
                          "mean_diff": float(dv.mean()), "p": bs.sign_flip_p(dv),
                          "batch_level_p": batch["p"]})
    bars_df, st = pd.DataFrame(bars), pd.DataFrame(tests)
    if not st.empty:
        st["p_holm"] = np.nan
        for panel in pd.unique(st["panel"]):
            sel = st["panel"] == panel
            st.loc[sel, "p_holm"] = fs.holm_adjust(st.loc[sel, "p"].to_numpy(dtype=float))
        st["stars"] = st["p_holm"].map(b.stars)
    return bars_df, st


def _save_figure(fig, path: Path) -> list[Path]:
    """PNG (300 dpi) and a vector PDF beside it; the figure is closed."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [path, pdf]


def _tick(partner: str) -> str:
    return f"{CLICK}\n(click)" if partner == CLICK else partner.replace(" ", "\n")


def naive_vs_pair_figure(bars: pd.DataFrame, st: pd.DataFrame, dataset: str, path: Path, *,
                         started: str = "") -> list[Path]:
    """``<group>_scores_pooled``'s layout on the peak-z ratio: per odor the naive bar
    (gray) and the in-pair bar (blue), (A) signed and (B) unsigned ratio, 95 % CI
    (flies); over every odor a bracket with its raw p, stars only after Holm. Writes the
    PNG, a PDF, the bars (``.csv``) and the tests (``_stats.csv``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    b = _b2b()
    keys = ("pre", "pair")
    panels = [(c, y) for c, y in PEAKZ_POOLED_PANELS if c in set(bars["panel"])]
    partners = [p for p in b.ORDER if p in set(bars["partner"])]
    head = {b.FIRST: "IAA first: each odor alone vs right after IAA",
            b.SECOND: "IAA second: IAA alone vs right after each odor"}[dataset]
    x0 = np.arange(len(partners), dtype=float)
    w, step = 0.36, 0.37
    xpos = {(i, k): x0[i] + (j - 0.5) * step for i in range(len(partners))
            for j, k in enumerate(keys)}
    n_odors = int(st.groupby("panel").size().max()) if len(st) else 0
    with plt.rc_context(b._PUB_RC):
        fig = plt.figure(figsize=(7.2, 1.55 + 2.25 * len(panels) + 0.95))
        gs = fig.add_gridspec(len(panels), 1, left=0.12, right=0.985, top=0.855, bottom=0.185,
                              hspace=0.16)
        axes = [fig.add_subplot(gs[i]) for i in range(len(panels))]
        for ax, (col, ylab) in zip(axes, panels):
            b._style_axis(ax)
            t = bars[bars["panel"] == col]
            for j, k in enumerate(keys):
                d = t[t["key"] == k].set_index("partner").reindex(partners)
                x = np.array([xpos[(i, k)] for i in range(len(partners))])
                m_ = d["mean"].to_numpy(dtype=float)
                ax.bar(x, m_, w, color=b.BAR_COLORS[k], edgecolor="white", linewidth=0.6,
                       zorder=2)
                elo = m_ - d["ci_lo"].to_numpy(dtype=float)
                ehi = d["ci_hi"].to_numpy(dtype=float) - m_
                ok = np.isfinite(elo) & np.isfinite(ehi)
                ax.errorbar(x[ok], m_[ok], yerr=[elo[ok], ehi[ok]], fmt="none", ecolor=b.INK,
                            elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
            ends = pd.concat([t["ci_hi"].fillna(t["mean"]), t["mean"]])
            lows = pd.concat([t["ci_lo"].fillna(t["mean"]), t["mean"]])
            hi, lo = float(np.nanmax(ends)), min(0.0, float(np.nanmin(lows)))
            unit = 0.06 * max(hi - lo, 0.5)
            top_y = hi
            s = st[st["panel"] == col].set_index("partner")
            for i, p in enumerate(partners):
                if p not in s.index:
                    continue
                r = s.loc[p]
                tip = float(np.nanmax(t.loc[t["partner"] == p, ["mean", "ci_hi"]]
                                      .to_numpy(dtype=float)))
                y = max(tip, 0.0) + 0.7 * unit
                xa, xb = xpos[(i, "pre")], xpos[(i, "pair")]
                ax.plot([xa, xa, xb, xb], [y - 0.35 * unit, y, y, y - 0.35 * unit],
                        color=b.INK, lw=0.6, solid_capstyle="butt", zorder=4)
                ax.text((xa + xb) / 2, y + 0.15 * unit, f"{_p_short(float(r['p']))}{r['stars']}",
                        ha="center", va="bottom", fontsize=6.3, color=b.INK, zorder=4,
                        fontweight="bold" if r["stars"] else "normal")
                top_y = max(top_y, y + 1.6 * unit)
            ax.set_ylim(lo - (0.06 * (hi - lo) if lo < 0 else 0.0), top_y)
            ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
            ax.axhline(0, color=b.RULE, linewidth=0.6, zorder=2.5)
            ax.set_xlim(-0.6, len(partners) - 0.4)
            ax.set_xticks(x0)
            ax.set_ylabel(ylab, color=b.INK, fontsize=8.5)
        for ax in axes[:-1]:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0, labelbottom=False)
        axes[-1].set_xticklabels([_tick(p) for p in partners], color=b.INK, fontsize=8)
        axes[-1].set_xlabel(b.BAR_XLABEL[dataset], color=b.INK, fontsize=8.5)
        fig.align_ylabels(axes)
        for ax, letter in zip(axes, "AB"):
            ax.text(-0.11, 1.02, letter, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color=b.INK, va="bottom")
        fig.text(0.12, 0.985, f"{head}: PER-trace peak z", fontsize=10, fontweight="bold",
                 color=b.INK, va="top")
        first = bars[bars["panel"] == panels[0][0]] if panels else bars
        note = b.corner_n(first[["key", "n"]])
        if note is None:
            nf = sorted(set(first["n_flies"].astype(int)))
            note = f"n = {nf[0]}–{nf[-1]} flies per odor" if len(nf) > 1 else f"n = {nf[0]} flies"
        corner = f"{note}  ·  mean, 95 % CI (flies)" + (f"  ·  {started}" if started
                                                             else "")
        fig.text(0.12, 0.945, corner, ha="left", va="top", fontsize=8, color=b.INK2)
        fig.legend(handles=[Patch(facecolor=b.BAR_COLORS[k], label=b.BAR_LABELS[k]) for k in keys],
                   loc="lower left", bbox_to_anchor=(0.12, 0.872), ncol=2, frameon=False,
                   fontsize=8, handlelength=1.1, handleheight=0.9, columnspacing=1.6,
                   borderaxespad=0, labelcolor=b.INK)
        foot = ("Secondary: alone vs in the test block is NOT controlled for time in the rig or "
                "re-testing (controlled: Change-Minus-Click/b2b_peakz_ratio_change_minus_click). "
                "p over every odor: exact paired sign-flip on per-fly differences (in-pair mean "
                "- alone), raw; * p < 0.05, ** p < 0.01, *** p < 0.001 only after Holm over the "
                f"{n_odors} odors of each panel (exploratory). Bars pool trials; whiskers: 95 % CI "
                "with flies as the unit. " + _definition("peakz_ratio", None, None)
                + " Odor 2 is scored against the block before odor 1, so odor 1's late "
                "response can land in its window.")
        fig.text(0.12, 0.008, _wrap(foot, 8.3, pt=6.5), ha="left", va="bottom", fontsize=6.5,
                 color=b.INK2)
        written = _save_figure(fig, path)
    path = Path(path)
    csv = path.with_suffix(".csv")
    bars.drop(columns=["key"]).to_csv(csv, index=False)
    stats_csv = path.with_name(f"{path.stem}_stats.csv")
    st.assign(stars=st["stars"].replace("", "ns"), test=PEAKZ_POOLED_TEST).to_csv(
        stats_csv, index=False)
    return [*written, csv, stats_csv]


# ── driver ───────────────────────────────────────────────────────────────────
def write(out: Path, pre_df: pd.DataFrame, pair_df: pd.DataFrame, odor2_window, *,
          config: Path, started: str = "") -> list[Path]:
    """Write ``<out>/Change-Minus-Click/``: both figures (PNG, PDF, CSV of every bar) and
    the per-trial measures. ``pre_df`` / ``pair_df``: the window's wide-table rows;
    ``odor2_window(row) -> (on, off) | None``. A missing model skips log AUC; a missing
    dataset only loses its bars. Returns every path written."""
    started = str(started or "").strip()
    if started and not started.lower().startswith("flies"):
        started = f"flies started {started}"
    frames = naive_and_pair_rows(pre_df, pair_df, odor2_window)
    if all(f.empty for f in frames):
        print("[b2b] Change-Minus-Click: no naive or odor-2 trials, nothing written")
        return []
    trials = trial_measures(frames, config=config)
    dest = Path(out) / OUT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    eps, delay = model_eps(config) if "log_auc" in trials.columns else (None, None)
    for measure, spec in MEASURES.items():
        t = contrast_table(trials, measure)
        if t.empty:
            continue
        stem = f"b2b_{measure}_change_minus_click"
        title = (f"IAA odor pairs: naive → in-pair change in {spec['phrase']}, minus the "
                 f"paraffin click's change" + (f": {started}" if started else ""))
        written += plot_contrasts(t, measure, title, dest / f"{stem}.png",
                                  definition=_definition(measure, eps, delay))
        t.assign(test=TEST_TEXT).to_csv(dest / f"{stem}.csv", index=False)
        written.append(dest / f"{stem}.csv")
    b = _b2b()
    for ds in [d for d in b.DATASETS if d in set(trials["dataset"])]:
        bars, st = naive_vs_pair_table(trials, ds)
        if bars.empty or st.empty:
            continue
        short = b.GROUP_SHORT[ds]
        written += naive_vs_pair_figure(bars, st, ds,
                                        Path(out) / short / f"{short}_{PEAKZ_POOLED_STEM}.png",
                                        started=started)
    trials.to_csv(dest / PER_TRIAL_CSV, index=False)
    written.append(dest / PER_TRIAL_CSV)
    print(f"[b2b] Change-Minus-Click: {len(trials)} trials measured, {len(written)} files in "
          f"{dest}")
    return written
