#!/usr/bin/env python3
"""Does the odor delivered on the previous trial shift PER on the next one?

Hand-run, not wired into the pipeline. Reads each Sensitivity cohort's
``<folder>/<cohort>/_inputs/predictions.csv`` (the window flies the folder reports,
flagged flies already dropped) and writes ``<folder>/Previous-Odor-Carryover/``.

Design facts the test leans on
------------------------------
* Within each pre-exposure and post-exposure panel the 7 odors run in a fresh random
  order per *batch* (all flies of a batch share one sequence), 240 s apart. The batch,
  not the fly, is the randomization unit, so every permutation moves whole batches.
* Paraffin (newer batches) is fixed first in the pre panel and last-but-one in the
  post panel, and light-only is always last. Neither is a randomized *previous* odor,
  so transitions out of them are dropped. Paraffin as the *current* odor (post panel
  only) keeps a randomized predecessor and stays in.
* The first post-exposure trial follows training, not a panel odor, and the first
  pre-exposure trial follows nothing, so neither has a previous odor.

Method
------
Each trial's response (score >= 2, and the score itself as a secondary outcome) is
residualised on fly x panel fixed effects (each fly compared with itself), current
odor x panel x cohort and trial position x panel. Under "no carry-over" the previous
odor, being randomized, carries no information about that residual. Test statistics:

* ``main``: does the previous odor shift responding to *every* next odor alike
  (between-previous-odor sum of squares of the residual);
* ``pair``: any (current, previous) pair effect at all, e.g. IAA after EB
  (between-cell sum of squares);
* ``consistency``: mean pairwise correlation of the four cohorts' pair-effect
  matrices -- is the same pattern seen in every dataset?
* directed 1-df contrasts: previous odor's attractiveness, same chemical class, and
  (post panel) the trial after the sensitized odor.

p-values: permutation (default 10 000) of the previous-odor labels among the
transitions of each batch panel (see ``Permuter``; calibrated on simulated
no-carry-over data with the real sequences). Cell p-values are BH-adjusted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FOLDER = Path("/home/ramanlab/Documents/cole/Results/Sensitiviy-Results(11-15)")
OUT_NAME = "Previous-Odor-Carryover"
COHORTS = ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1", "3Oct-Sensitivity-1-0.1",
           "Paraffin-Sensitivity-1-100"]
SENSITIZED = {"IAA-Sensitivity-1-1": "Isoamyl Acetate", "Hex-Sensitivity-1-0.1": "Hexanol",
              "3Oct-Sensitivity-1-0.1": "3-Octanol", "Paraffin-Sensitivity-1-100": "Paraffin Oil"}
RESPONSE_THRESHOLD = 2
MIN_CELL_N = 5          # pair cells with fewer trials are shown hatched and left out of BH

ODOR_NAMES = {
    "isoamylacetate": "Isoamyl Acetate", "ethylbutyrate": "Ethyl Butyrate",
    "hexanol": "Hexanol", "3-octonol": "3-Octanol", "3-octanol": "3-Octanol",
    "linalool": "Linalool", "citral": "Citral", "benzaldehyde": "Benzaldehyde",
    "paraffinoil": "Paraffin Oil", "lightonly": "Light only",
    "3octanol": "3-Octanol", "acv": "ACV",
}
RANDOMIZED_ODORS = ["Isoamyl Acetate", "Ethyl Butyrate", "Hexanol", "3-Octanol",
                    "Linalool", "Citral", "Benzaldehyde"]
CURRENT_ODORS = RANDOMIZED_ODORS + ["Paraffin Oil"]
# predecessor categories: the 7 panel odors, plus paraffin as a *training* predecessor
PREV_ODORS = RANDOMIZED_ODORS + ["Paraffin Oil"]
NOT_A_TARGET = {"Light only"}
# chemical class used by the same-class contrast
ODOR_CLASS = {"Isoamyl Acetate": "ester", "Ethyl Butyrate": "ester",
              "Hexanol": "aliphatic alcohol", "3-Octanol": "aliphatic alcohol",
              "Linalool": "terpenoid", "Citral": "terpenoid",
              "Benzaldehyde": "aromatic aldehyde", "Paraffin Oil": "solvent"}

_LABEL = re.compile(r"^(pretest|testing)_(\d+)_(.+?)(?:-\d+(?:\.\d+)?)?$")


# ── data ─────────────────────────────────────────────────────────────────────
def parse_trials(pred: pd.DataFrame, skip_unknown: bool = False) -> pd.DataFrame:
    """One row per scored trial: cohort, batch, fly, block (pre/post), position, odor.

    ``skip_unknown`` drops training rows, odor-pair labels (``X_then_Y``) and unknown
    odor names instead of raising (for the all-datasets table)."""
    m = pred["trial_label"].astype(str).str.extract(_LABEL)
    raw = m[2].str.lower()
    bad = m.isna().any(axis=1) | ~raw.isin(list(ODOR_NAMES))
    if bad.any() and not skip_unknown:
        raise ValueError(f"unparsed trial labels / unknown odors: "
                         f"{list(pred.loc[bad, 'trial_label'].unique()[:5])}")
    pred, m, raw = pred[~bad], m[~bad], raw[~bad]
    out = pd.DataFrame({
        "cohort": pred["dataset"].astype(str).to_numpy(),
        "batch": pred["fly"].astype(str).to_numpy(),
        "fly_number": pred["fly_number"].astype(int).to_numpy(),
        "block": np.where(m[0] == "pretest", "pre", "post"),
        "pos": m[1].astype(int).to_numpy(),
        "odor": raw.map(ODOR_NAMES).to_numpy(),
        "score": pred["score"].astype(float).to_numpy(),
    })
    out["responded"] = out["score"] >= RESPONSE_THRESHOLD
    out["fly_id"] = out["cohort"] + "|" + out["batch"] + "|" + out["fly_number"].astype(str)
    return out


def drop_inconsistent_panels(trials: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop batch panels whose flies disagree on the odor at a position, or where a fly
    has the same trial number twice (aborted-session orphans). Returns (kept, n_dropped)."""
    panel = ["cohort", "batch", "block"]
    n_odor = trials.groupby(panel + ["pos"])["odor"].nunique()
    bad = set(n_odor[n_odor > 1].index.droplevel("pos"))
    dup = trials.duplicated(["fly_id", "block", "pos"], keep=False)
    bad |= set(map(tuple, trials.loc[dup, panel].drop_duplicates().to_numpy()))
    keys = list(map(tuple, trials[panel].to_numpy()))
    keep = np.array([k not in bad for k in keys])
    return trials[keep].reset_index(drop=True), len(bad)


def build_transitions(trials: pd.DataFrame, training_odor: dict | None = None,
                      training_last: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each trial paired with the odor one position earlier in the same batch panel.

    ``training_odor`` (cohort -> odor): post trial 1 gets that cohort's training odor as
    its predecessor (``prev_source`` 'training'); the trained odor may equal the current
    one. ``training_last`` (fly_id, last_training_responded) fills ``prev_responded``."""
    key = ["cohort", "batch", "block", "pos"]
    seq = trials.groupby(key)["odor"].agg(lambda s: s.unique().tolist())
    clash = seq[seq.str.len() > 1]
    if len(clash):
        raise ValueError(f"flies of one batch disagree on the odor sequence: {clash.index[:3].tolist()}")
    seq = seq.str[0].rename("prev_odor").reset_index()
    seq["pos"] += 1                                   # odor at k-1 becomes "previous" of k
    tr = trials.merge(seq, on=key, how="inner")
    own = trials[["fly_id", "block", "pos", "responded"]].copy()
    own["pos"] += 1
    tr = tr.merge(own.rename(columns={"responded": "prev_responded"}),
                  on=["fly_id", "block", "pos"], how="left")
    tr["prev_source"] = "panel"
    tr = tr[tr["prev_odor"].isin(RANDOMIZED_ODORS)]
    if training_odor:
        first = trials[(trials["block"] == "post") & (trials["pos"] == 1)
                       & trials["cohort"].isin(list(training_odor))].copy()
        first["prev_odor"] = first["cohort"].map(training_odor)
        first["prev_source"] = "training"
        if training_last is not None:
            first = first.merge(training_last.rename(columns={"last_training_responded": "prev_responded"}),
                                on="fly_id", how="left")
        else:
            first["prev_responded"] = np.nan
        tr = pd.concat([tr, first[tr.columns]], ignore_index=True)
    tr["prev_responded"] = tr["prev_responded"].astype("boolean")
    tr = tr[tr["odor"].isin(CURRENT_ODORS) & tr["prev_odor"].isin(PREV_ODORS)]
    tr = tr.sort_values(["cohort", "batch", "block", "fly_number", "pos"])
    return tr.reset_index(drop=True)


def load_training_last(folder: Path, cohorts=COHORTS) -> pd.DataFrame:
    """Each fly's response on its last training trial (the one before post trial 1)."""
    frames = []
    for c in cohorts:
        p = folder / c / "_inputs" / "training_predictions.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame(columns=["fly_id", "last_training_responded"])
    t = pd.concat(frames, ignore_index=True)
    t["k"] = t["trial_label"].str.extract(r"^training_(\d+)_")[0].astype(float)
    t = t.dropna(subset=["k"]).sort_values("k").groupby(["dataset", "fly", "fly_number"]).tail(1)
    return pd.DataFrame({
        "fly_id": t["dataset"].astype(str) + "|" + t["fly"].astype(str) + "|" + t["fly_number"].astype(int).astype(str),
        "last_training_responded": (t["score"] >= RESPONSE_THRESHOLD).to_numpy()})


def load(folder: Path, cohorts=COHORTS) -> pd.DataFrame:
    frames = []
    for c in cohorts:
        p = folder / c / "_inputs" / "predictions.csv"
        if not p.exists():
            print(f"[carryover] {c}: no {p}, skipped", file=sys.stderr)
            continue
        frames.append(pd.read_csv(p))
    return parse_trials(pd.concat(frames, ignore_index=True))


# ── residuals ────────────────────────────────────────────────────────────────
def _dummies(codes: np.ndarray, drop_first: bool) -> np.ndarray:
    u, inv = np.unique(codes, return_inverse=True)
    d = np.zeros((len(codes), len(u)))
    d[np.arange(len(codes)), inv] = 1.0
    return d[:, 1:] if drop_first else d


def base_design(tr: pd.DataFrame, pos_fe: str = "block") -> np.ndarray:
    """fly x panel FE + current odor x panel x cohort + position x panel
    (``pos_fe='cohort_block'``: position x panel x cohort, for mixed protocols)."""
    fe = _dummies((tr["fly_id"] + "|" + tr["block"]).to_numpy(), drop_first=False)
    odor = _dummies((tr["cohort"] + "|" + tr["block"] + "|" + tr["odor"]).to_numpy(), True)
    pkey = tr["block"] + "|" + tr["pos"].astype(str)
    if pos_fe == "cohort_block":
        pkey = tr["cohort"] + "|" + pkey
    pos = _dummies(pkey.to_numpy(), True)
    return np.hstack([fe, odor, pos])


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


# ── permutation ──────────────────────────────────────────────────────────────
class Permuter:
    """Shuffle previous-odor labels among the transitions of one batch panel.

    The unit is a batch x panel x position (all flies of a batch move together). Each
    panel keeps its own set of predecessors -- every odor once, as in the real random
    order -- and no panel trial is given its own odor as predecessor (a training slot,
    post trial 1, may be: the trained odor can also be the first test odor). On
    simulated no-carry-over data built on the real sequences this rejected 4.7 % (main)
    and 5.3 % (pair) at alpha 0.05; shuffling *between* batches rejected 10 % (main).
    """

    def __init__(self, tr: pd.DataFrame, rng: np.random.Generator, max_tries: int = 1000):
        self.rng, self.max_tries = rng, max_tries
        unit_key = tr["cohort"] + "|" + tr["batch"] + "|" + tr["block"] + "|" + tr["pos"].astype(str)
        _, self.row_unit = np.unique(unit_key.to_numpy(), return_inverse=True)
        first = pd.Series(np.arange(len(tr))).groupby(self.row_unit).first().to_numpy()
        codes = {o: i for i, o in enumerate(PREV_ODORS)}
        self.unit_prev = np.array([codes[o] for o in tr["prev_odor"].to_numpy()[first]])
        self.unit_cur = np.array([codes.get(o, -1) for o in tr["odor"].to_numpy()[first]])
        src = tr["prev_source"].to_numpy() if "prev_source" in tr else np.full(len(tr), "panel")
        self.unit_panel = src[first] == "panel"
        panel = (tr["cohort"] + "|" + tr["batch"] + "|" + tr["block"]).to_numpy()[first]
        self.groups = [g for g in (np.flatnonzero(panel == s) for s in np.unique(panel)) if len(g) > 1]

    def shuffled_unit_prev(self) -> np.ndarray:
        out = self.unit_prev.copy()
        for g in self.groups:
            pv, cur, chk = self.unit_prev[g], self.unit_cur[g], self.unit_panel[g]
            for _ in range(self.max_tries):
                p = pv[self.rng.permutation(len(g))]
                if not np.any((p == cur) & chk):
                    out[g] = p
                    break
        return out

    def shuffled_prev_codes(self) -> np.ndarray:
        """Codes into PREV_ODORS, aligned with the transition rows."""
        return self.shuffled_unit_prev()[self.row_unit]

    def shuffled_prev(self) -> np.ndarray:
        return np.array(PREV_ODORS)[self.shuffled_prev_codes()]


# ── statistics ───────────────────────────────────────────────────────────────
def _between_ss(r: np.ndarray, codes: np.ndarray, k: int) -> float:
    n = np.bincount(codes, minlength=k)
    s = np.bincount(codes, weights=r, minlength=k)
    ok = n > 0
    return float(np.sum(s[ok] ** 2 / n[ok]))


def _cell_means(r, cur, prev, n_cur, n_prev):
    idx = cur * n_prev + prev
    n = np.bincount(idx, minlength=n_cur * n_prev)
    s = np.bincount(idx, weights=r, minlength=n_cur * n_prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / n).reshape(n_cur, n_prev), n.reshape(n_cur, n_prev)


def _consistency(r, cur, prev, cohort_codes, n_coh, n_cur, n_prev, min_n=5):
    mats = []
    for c in range(n_coh):
        m = cohort_codes == c
        mean, n = _cell_means(r[m], cur[m], prev[m], n_cur, n_prev)
        mean[n < min_n] = np.nan
        mats.append(mean.ravel())
    cors = []
    for i in range(n_coh):
        for j in range(i + 1, n_coh):
            ok = np.isfinite(mats[i]) & np.isfinite(mats[j])
            if ok.sum() >= 10:
                cors.append(np.corrcoef(mats[i][ok], mats[j][ok])[0, 1])
    return float(np.mean(cors)) if cors else np.nan


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out


def leave_batch_out_rate(tr: pd.DataFrame, trials: pd.DataFrame, y: str) -> np.ndarray:
    """(rows x prev-odor) matrix: each odor's mean outcome in the row's cohort x panel,
    from every trial *except* the row's own batch."""
    t = trials[~trials["odor"].isin(NOT_A_TARGET)]
    g = t.groupby(["cohort", "block", "odor", "batch"])[y].agg(["sum", "count"]).reset_index()
    tot = g.groupby(["cohort", "block", "odor"])[["sum", "count"]].sum()
    A = np.full((len(tr), len(PREV_ODORS)), np.nan)
    own = g.set_index(["cohort", "block", "odor", "batch"])
    for j, o in enumerate(PREV_ODORS):
        keys_tot = list(zip(tr["cohort"], tr["block"], [o] * len(tr)))
        T = tot.reindex(keys_tot)
        O = own.reindex(list(zip(tr["cohort"], tr["block"], [o] * len(tr), tr["batch"]))).fillna(0)
        A[:, j] = (T["sum"].to_numpy() - O["sum"].to_numpy()) / (T["count"].to_numpy() - O["count"].to_numpy())
    return A


def run_tests(tr: pd.DataFrame, trials: pd.DataFrame, outcome: str, n_perm: int, seed: int,
              cohorts: list | None = None, sensitized: dict | None = None,
              pos_fe: str = "block", per_cohort_blocks: bool = True):
    """All permutation tests for one outcome ('responded' or 'score')."""
    rng = np.random.default_rng(seed)
    if cohorts is None:
        present = set(tr["cohort"])
        cohorts = [c for c in COHORTS if c in present] + sorted(present - set(COHORTS))
    sensitized = SENSITIZED if sensitized is None else sensitized
    y = tr[outcome].astype(float).to_numpy()
    r = residualize(y, base_design(tr, pos_fe))
    if outcome == "responded":
        r = r * 100.0                                   # percentage points
    cur = pd.Categorical(tr["odor"], categories=CURRENT_ODORS).codes
    prev = pd.Categorical(tr["prev_odor"], categories=PREV_ODORS).codes
    coh = pd.Categorical(tr["cohort"], categories=cohorts).codes
    blk = (tr["block"] == "post").to_numpy().astype(int)
    nC, nP, nK = len(CURRENT_ODORS), len(PREV_ODORS), len(cohorts)
    sens_odor = tr["cohort"].map(sensitized)
    sens_cur = tr["odor"].to_numpy() == sens_odor.to_numpy()
    sens_code = np.array([PREV_ODORS.index(o) if o in RANDOMIZED_ODORS else -1 for o in sens_odor])
    A = leave_batch_out_rate(tr, trials, outcome)
    A = (A - np.nanmean(A, axis=1, keepdims=True))     # centred within row: "more attractive than average"
    if outcome == "responded":
        A = A * 100
    cls = np.array([ODOR_CLASS[o] for o in CURRENT_ODORS])
    pcls = np.array([ODOR_CLASS[o] for o in PREV_ODORS])
    same_cls = cls[:, None] == pcls[None, :]           # (cur, prev)

    src = tr["prev_source"].to_numpy() if "prev_source" in tr else np.full(len(tr), "panel")
    subsets = {"all": np.ones(len(tr), bool), "pre": blk == 0, "post": blk == 1}
    if (src == "training").any():
        subsets["post_trial1_after_training"] = src == "training"
        subsets["panel_transitions_only"] = src == "panel"
    for k, c in enumerate(cohorts):
        subsets[c] = coh == k
        if per_cohort_blocks:
            subsets[f"{c}|pre"] = (coh == k) & (blk == 0)
            subsets[f"{c}|post"] = (coh == k) & (blk == 1)
    exposed = np.isin(coh, [k for k, c in enumerate(cohorts) if sensitized.get(c) in RANDOMIZED_ODORS])
    # cell tables besides 'pooled': pooled pre / post, each cohort (both panels), and
    # each cohort's pre / post
    cell_scopes = [("pooled|pre", blk == 0), ("pooled|post", blk == 1)]
    cell_scopes += [(c, coh == k) for k, c in enumerate(cohorts)]
    if per_cohort_blocks:
        cell_scopes += [(f"{c}|{b}", (coh == k) & (blk == v)) for k, c in enumerate(cohorts)
                        for b, v in (("pre", 0), ("post", 1))]
    rows = np.arange(len(tr))

    def stats(p):
        d = {}
        for name, m in subsets.items():
            d[f"main|{name}"] = _between_ss(r[m], p[m], nP)
            d[f"pair|{name}"] = _between_ss(r[m], cur[m] * nP + p[m], nC * nP)
        for b in ("all", "pre", "post"):
            m = subsets[b]
            d[f"consistency|{b}"] = _consistency(r[m], cur[m], p[m], coh[m], nK, nC, nP, MIN_CELL_N)
        for b, m in subsets.items():
            a = A[rows[m], p[m]]
            d[f"attractiveness|{b}"] = float(np.nansum(r[m] * a) / np.nansum(a * a))
            sc = same_cls[cur[m], p[m]]
            if sc.any() and (~sc).any():
                d[f"same_class|{b}"] = float(r[m][sc].mean() - r[m][~sc].mean())
            ps = (p[m] == sens_code[m]) & exposed[m] & ~sens_cur[m]
            rest = exposed[m] & ~ps & ~sens_cur[m]
            if ps.any() and rest.any():
                d[f"after_sensitized|{b}"] = float(r[m][ps].mean() - r[m][rest].mean())
        means, _ = _cell_means(r, cur, p, nC, nP)
        cohort_cells = [_cell_means(r[m], cur[m], p[m], nC, nP)[0] for _, m in cell_scopes]
        with np.errstate(invalid="ignore", divide="ignore"):
            main_means = np.bincount(p, weights=r, minlength=nP) / np.bincount(p, minlength=nP)
        return d, means, np.stack(cohort_cells), main_means

    obs, obs_cells, obs_coh_cells, obs_main = stats(prev)
    perm_vals = {k: np.full(n_perm, np.nan) for k in obs}
    perm_cells = np.empty((n_perm,) + obs_cells.shape)
    perm_coh_cells = np.empty((n_perm,) + obs_coh_cells.shape)
    perm_main = np.empty((n_perm, nP))
    permuter = Permuter(tr, rng)
    for i in range(n_perm):
        p_codes = permuter.shuffled_prev_codes()
        d, cells, coh_cells, mm = stats(p_codes)
        for k, v in d.items():
            if k in perm_vals:
                perm_vals[k][i] = v
        perm_cells[i], perm_coh_cells[i], perm_main[i] = cells, coh_cells, mm

    def p_upper(o, null):
        null = null[np.isfinite(null)]
        return (1 + np.sum(null >= o - 1e-12)) / (1 + len(null))

    def p_two(o, null):
        null = null[np.isfinite(null)]
        return (1 + np.sum(np.abs(null) >= abs(o) - 1e-12)) / (1 + len(null))

    tests = []
    for k, o in obs.items():
        kind, scope = k.split("|", 1)
        two_sided = kind in ("attractiveness", "same_class", "after_sensitized", "consistency")
        p = p_two(o - np.nanmean(perm_vals[k]), perm_vals[k] - np.nanmean(perm_vals[k])) \
            if two_sided and kind != "consistency" else p_upper(o, perm_vals[k])
        nv = perm_vals[k][np.isfinite(perm_vals[k])]
        mu = float(nv.mean())
        tests.append({"test": kind, "scope": scope, "observed": o, "null_mean": mu,
                      # interval for the contrasts: observed minus the null's spread
                      "ci_low": o - (np.percentile(nv, 97.5) - mu) if two_sided else np.nan,
                      "ci_high": o - (np.percentile(nv, 2.5) - mu) if two_sided else np.nan,
                      "p": p})
    tests = pd.DataFrame(tests)

    # pair cells (pooled + per cohort): two-sided vs their own permutation null
    _, n_cells = _cell_means(r, cur, prev, nC, nP)
    raw_rate = tr.groupby(["odor", "prev_odor"])[outcome].mean()
    cell_rows = []
    for i, co in enumerate(CURRENT_ODORS):
        for j, po in enumerate(PREV_ODORS):
            if n_cells[i, j] == 0:
                continue
            null = perm_cells[:, i, j]
            cell_rows.append({"cohort": "pooled", "current_odor": co, "previous_odor": po,
                              "n_trials": int(n_cells[i, j]), "raw_mean": raw_rate.get((co, po), np.nan),
                              "adjusted_effect": obs_cells[i, j],
                              "ci_low": np.nanpercentile(null, 2.5) + obs_cells[i, j] - np.nanmean(null),
                              "ci_high": np.nanpercentile(null, 97.5) + obs_cells[i, j] - np.nanmean(null),
                              "p": p_two(obs_cells[i, j], null[np.isfinite(null)])})
    for k, (c, m) in enumerate(cell_scopes):
        _, nck = _cell_means(r[m], cur[m], prev[m], nC, nP)
        rr = tr[m].groupby(["odor", "prev_odor"])[outcome].mean()
        for i, co in enumerate(CURRENT_ODORS):
            for j, po in enumerate(PREV_ODORS):
                if nck[i, j] == 0:
                    continue
                null = perm_coh_cells[:, k, i, j]
                cell_rows.append({"cohort": c, "current_odor": co, "previous_odor": po,
                                  "n_trials": int(nck[i, j]), "raw_mean": rr.get((co, po), np.nan),
                                  "adjusted_effect": obs_coh_cells[k, i, j],
                                  "ci_low": np.nanpercentile(null, 2.5) + obs_coh_cells[k, i, j] - np.nanmean(null),
                                  "ci_high": np.nanpercentile(null, 97.5) + obs_coh_cells[k, i, j] - np.nanmean(null),
                                  "p": p_two(obs_coh_cells[k, i, j], null[np.isfinite(null)])})
    cells = pd.DataFrame(cell_rows)
    # BH within each cohort, over cells with >= MIN_CELL_N trials only (a 2-trial cell's
    # permutation p is meaningless and would otherwise be the 'hit')
    tested = cells["n_trials"] >= MIN_CELL_N
    cells["q_bh_within_cohort"] = np.nan
    cells.loc[tested, "q_bh_within_cohort"] = (
        cells[tested].groupby("cohort")["p"].transform(lambda s: bh(s.to_numpy())))

    main = pd.DataFrame({"previous_odor": PREV_ODORS, "adjusted_effect": obs_main,
                         "n_trials": np.bincount(prev, minlength=nP),
                         "p": [p_two(obs_main[j], perm_main[:, j]) for j in range(nP)]})
    main = main[main["n_trials"] > 0].reset_index(drop=True)
    main["q_bh"] = bh(main["p"].to_numpy())
    return {"tests": tests, "cells": cells, "main": main, "residual": r}


def autocorrelation(tr: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """Did the same fly's PER on the previous trial raise PER on this one?

    Not randomized, so read it as fly state, not odor. The within-fly slope is biased
    downwards by the fly x panel fixed effects (Nickell bias: ~7 trials per panel), so
    the null is the same slope with the 'previous' response taken from a random
    non-adjacent trial of the same fly panel; effect = observed - null mean."""
    t = tr.dropna(subset=["prev_responded"]).reset_index(drop=True)
    raw = t.groupby("prev_responded")["responded"].mean() * 100
    X = base_design(t)
    Xp = np.linalg.pinv(X)                              # once; residualize = x - X @ (Xp @ x)
    y = t["responded"].astype(float).to_numpy()
    y = y - X @ (Xp @ y)

    def slope(x):
        xr = x - X @ (Xp @ x)
        return float(xr @ y / (xr @ xr)) * 100

    obs = slope(t["prev_responded"].astype(float).to_numpy())
    rng = np.random.default_rng(seed)
    fb = (t["fly_id"] + "|" + t["block"]).to_numpy()
    resp = t["responded"].astype(float).to_numpy()
    pos = t["pos"].to_numpy()
    # padded candidate table: for each row, the same fly panel's trials that are
    # neither this trial nor the one just before it
    cands = []
    for g in (np.flatnonzero(fb == v) for v in np.unique(fb)):
        for k in g:
            cands.append((k, g[(pos[g] != pos[k]) & (pos[g] != pos[k] - 1)]))
    width = max(len(c) for _, c in cands)
    table = np.zeros((len(t), max(width, 1)), dtype=int)
    count = np.zeros(len(t), dtype=int)
    for k, c in cands:
        table[k, :len(c)], count[k] = c, len(c)
    has = count > 0
    null = np.empty(n_perm)
    for i in range(n_perm):
        pick = table[np.arange(len(t)), (rng.random(len(t)) * np.maximum(count, 1)).astype(int)]
        x = np.where(has, resp[pick], np.nan)
        x[~has] = np.nanmean(x)
        null[i] = slope(x)
    eff = obs - null.mean()
    p = (1 + np.sum(np.abs(null - null.mean()) >= abs(eff))) / (1 + n_perm)
    return {"rate_after_no_PER_pct": float(raw.get(False, np.nan)),
            "rate_after_PER_pct": float(raw.get(True, np.nan)),
            "within_fly_slope_pp": obs, "null_mean_slope_pp (fixed-effect bias)": float(null.mean()),
            "bias_corrected_effect_pp": float(eff),
            "null_95pct_pp": [float(np.percentile(null, 2.5) - null.mean() + eff),
                              float(np.percentile(null, 97.5) - null.mean() + eff)],
            "p": float(p), "n_trials": int(len(t)), "n_perm": n_perm}


def _batch_boot_ci(values: np.ndarray, batches: np.ndarray, rng, n_boot=2000):
    ub, inv = np.unique(batches, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(ub))
    n = np.bincount(inv, minlength=len(ub))
    if len(ub) < 2:
        return np.nan, np.nan
    draws = rng.integers(0, len(ub), size=(n_boot, len(ub)))
    means = s[draws].sum(1) / n[draws].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def after_training_table(tr: pd.DataFrame, r: np.ndarray, outcome: str, seed: int) -> pd.DataFrame:
    """Post trial 1 (right after training) by cohort x current odor: raw rate, the same
    odor's rate at post trials 2+, and the adjusted effect with a batch-bootstrap CI."""
    rng = np.random.default_rng(seed)
    scale = 100 if outcome == "responded" else 1
    t = tr.assign(_r=r)
    first = t[t["prev_source"] == "training"]
    later = t[(t["block"] == "post") & (t["pos"] >= 2)]
    rows = []
    for (c, o), g in list(first.groupby(["cohort", "odor"])) + \
            [((c, "(all odors)"), g) for c, g in first.groupby("cohort")]:
        lat = later[(later["cohort"] == c) & ((later["odor"] == o) if o != "(all odors)" else True)]
        lo, hi = _batch_boot_ci(g["_r"].to_numpy(), g["batch"].to_numpy(), rng)
        rows.append({"cohort": c, "trained_odor": g["prev_odor"].iloc[0], "first_test_odor": o,
                     "same_as_trained": o == g["prev_odor"].iloc[0],
                     "n_flies": len(g), "n_batches": g["batch"].nunique(),
                     "raw_trial1": g[outcome].mean() * scale,
                     "raw_same_odor_post_trials_2plus": lat[outcome].mean() * scale,
                     "adjusted_effect": g["_r"].mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def load_external(pred_csv: Path, wide_dir: Path, flagged_csv: Path,
                  exclude: list[str]) -> tuple[pd.DataFrame, dict]:
    """Every other dataset's pre/post panels: flagged flies and frozen folders dropped,
    inconsistent panels dropped, and only dataset x panel types whose odor order was
    randomized per batch (>= 80 % distinct sequences, >= 3 batches)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from fbpipe.config import load_flagged_fly_exclusions
    info = {}
    pred = pd.read_csv(pred_csv)
    pred = pred[~pred["dataset"].isin(exclude)]
    info["rows_in"] = len(pred)
    flagged = load_flagged_fly_exclusions(flagged_csv)
    key = list(zip(pred["dataset"].astype(str), pred["fly"].astype(str), pred["fly_number"].astype(str)))
    pred = pred[[k not in flagged for k in key]]
    info["rows_after_flagged"] = len(pred)
    frozen = set()
    for name in ("all_envelope_rows_wide_combined_base.parquet",
                 "all_envelope_rows_wide_combined_base_pretest.parquet"):
        p = wide_dir / name
        if p.exists():
            w = pd.read_parquet(p, columns=["dataset", "fly", "fly_number", "trial_label", "frozen"])
            w = w[w["frozen"].fillna(False).astype(bool)]
            frozen |= set(zip(w["dataset"].astype(str), w["fly"].astype(str),
                              w["fly_number"].astype(int), w["trial_label"].astype(str)))
    key = list(zip(pred["dataset"].astype(str), pred["fly"].astype(str),
                   pred["fly_number"].astype(int), pred["trial_label"].astype(str)))
    pred = pred[[k not in frozen for k in key]]
    info["rows_after_frozen"] = len(pred)
    trials = parse_trials(pred, skip_unknown=True)
    trials, info["panels_dropped_inconsistent"] = drop_inconsistent_panels(trials)
    seqs = (trials.sort_values("pos").groupby(["cohort", "batch", "block"])["odor"]
            .agg(tuple).reset_index())
    kind = seqs.groupby(["cohort", "block"])["odor"].agg(["size", "nunique"])
    ok = kind[(kind["size"] >= 3) & (kind["nunique"] / kind["size"] >= 0.8)].index
    info["dataset_panels_randomized"] = [f"{c}|{b}" for c, b in ok]
    info["dataset_panels_fixed_or_small"] = [f"{c}|{b}" for c, b in kind.index.difference(ok)]
    keep = pd.MultiIndex.from_frame(trials[["cohort", "block"]]).isin(ok)
    trials = trials[keep].reset_index(drop=True)
    info["flies"] = int(trials["fly_id"].nunique())
    info["batches"] = int(trials[["cohort", "batch"]].drop_duplicates().shape[0])
    info["datasets"] = int(trials["cohort"].nunique())
    return trials, info


FOCUS_PAIRS = [("Ethyl Butyrate", "Hexanol"), ("Hexanol", "Ethyl Butyrate"),
               ("Ethyl Butyrate", "3-Octanol"), ("Isoamyl Acetate", "Ethyl Butyrate")]


def cluster_check(tr: pd.DataFrame, outcome: str, current: str, previous: str,
                  pos_fe: str = "block") -> dict:
    """Model-based check of one pair, independent of the permutation: OLS of the outcome
    on the base design (fly x panel, current odor x panel x cohort, position) plus an
    indicator for (current, previous), batch-clustered SE (CR1), t with G-1 df. The
    estimate is the pair vs every other trial of the same odor in that cohort x panel."""
    from scipy import stats as sst
    x = ((tr["odor"] == current) & (tr["prev_odor"] == previous)).to_numpy(float)
    if x.sum() == 0:
        return {"n_trials": 0}
    y = tr[outcome].astype(float).to_numpy() * (100 if outcome == "responded" else 1)
    X = base_design(tr, pos_fe)
    xr, yr = residualize(x, X), residualize(y, X)
    b = float(xr @ yr / (xr @ xr))
    e = yr - b * xr
    _, gi = np.unique((tr["cohort"] + "|" + tr["batch"]).to_numpy(), return_inverse=True)
    G = gi.max() + 1
    sc = np.bincount(gi, weights=xr * e, minlength=G)
    se = float(np.sqrt(G / (G - 1) * np.sum(sc ** 2)) / (xr @ xr))
    tq = sst.t.ppf(0.975, G - 1)
    return {"n_trials": int(x.sum()), "n_batches_with_pair": int(tr.loc[x > 0, ["cohort", "batch"]]
                                                                 .drop_duplicates().shape[0]),
            "raw_rate_after_pair": float(y[x > 0].mean()),
            "raw_rate_same_odor_all": float(y[(tr["odor"] == current).to_numpy()].mean()),
            "cluster_estimate": b, "cluster_ci_low": b - tq * se, "cluster_ci_high": b + tq * se,
            "cluster_p": float(2 * sst.t.sf(abs(b / se), G - 1)), "clusters": int(G)}


PANELS = (("pre", "pre"), ("post", "post"), ("pre+post", None))


def focus_pair_table(sources: list, outcome: str) -> pd.DataFrame:
    """The focus pairs for each panel (pre, post, pre+post) and each data source:
    permutation cell + batch-clustered regression side by side.

    ``sources``: (label, transitions, results, pos_fe, split_by_cohort)."""
    rows = []
    for cur_o, prev_o in FOCUS_PAIRS:
        for panel, blk in PANELS:
            for label, tr, res, pfe, per_cohort in sources:
                t = tr if blk is None else tr[tr["block"] == blk]
                parts = [(label, t, "pooled" if blk is None else f"pooled|{blk}")]
                if per_cohort:
                    parts += [(c, t[t["cohort"] == c], c if blk is None else f"{c}|{blk}") for c in COHORTS]
                for lab, tt, cell_scope in parts:
                    cells = res[outcome]["cells"]
                    c = cells[(cells["cohort"] == cell_scope) & (cells["current_odor"] == cur_o)
                              & (cells["previous_odor"] == prev_o)]
                    row = {"pair": f"{prev_o} -> {cur_o}", "previous_odor": prev_o, "current_odor": cur_o,
                           "panel": panel, "data": lab, "outcome": outcome,
                           "perm_effect": float(c["adjusted_effect"].iloc[0]) if len(c) else np.nan,
                           "perm_ci_low": float(c["ci_low"].iloc[0]) if len(c) else np.nan,
                           "perm_ci_high": float(c["ci_high"].iloc[0]) if len(c) else np.nan,
                           "perm_p": float(c["p"].iloc[0]) if len(c) else np.nan}
                    if len(tt):
                        row.update(cluster_check(tt.reset_index(drop=True), outcome, cur_o, prev_o, pfe))
                    rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    ap.add_argument("--csv-dir", type=Path, default=Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys"))
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-perm-external", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-external", action="store_true")
    args = ap.parse_args(argv)

    out = args.folder / OUT_NAME
    out.mkdir(exist_ok=True)
    trials = load(args.folder)
    training_last = load_training_last(args.folder)
    tr = build_transitions(trials, training_odor=SENSITIZED, training_last=training_last)
    tr_panel = build_transitions(trials)
    tr.to_csv(out / "transitions.csv", index=False)
    results, results_panel = {}, {}
    for outcome in ("responded", "score"):
        tag = "pct_responding" if outcome == "responded" else "score"
        res = run_tests(tr, trials, outcome, args.n_perm, args.seed)
        res["tests"].to_csv(out / f"tests_{tag}.csv", index=False)
        res["cells"].to_csv(out / f"pair_cells_{tag}.csv", index=False)
        res["main"].to_csv(out / f"previous_odor_main_{tag}.csv", index=False)
        res["after_training"] = after_training_table(tr, res["residual"], outcome, args.seed)
        res["after_training"].to_csv(out / f"after_training_first_test_{tag}.csv", index=False)
        results[outcome] = res
        rp = run_tests(tr_panel, trials, outcome, args.n_perm, args.seed)
        rp["tests"].to_csv(out / f"tests_{tag}_panel_only.csv", index=False)
        rp["cells"].to_csv(out / f"pair_cells_{tag}_panel_only.csv", index=False)
        results_panel[outcome] = rp
    # the fly's own previous PER: panel transitions only (training trials carry the
    # opto light, which drives PER, so their 'response' is not comparable)
    ac = autocorrelation(tr_panel, min(args.n_perm, 2000), args.seed)
    (out / "previous_response_autocorrelation.json").write_text(json.dumps(ac, indent=2))

    ext = None
    if not args.no_external:
        ext_out = out / "Other-Datasets"
        ext_out.mkdir(exist_ok=True)
        ext_trials, info = load_external(args.csv_dir / "model_predictions.csv", args.csv_dir,
                                         args.csv_dir / "flagged-flys-truth.csv", COHORTS)
        ext_tr = build_transitions(ext_trials)
        info["transitions"] = len(ext_tr)
        ext_tr.to_csv(ext_out / "transitions.csv", index=False)
        ext = {"tr": ext_tr, "info": info, "trials": ext_trials}
        for outcome in ("responded", "score"):
            tag = "pct_responding" if outcome == "responded" else "score"
            res = run_tests(ext_tr, ext_trials, outcome, args.n_perm_external, args.seed,
                            sensitized={}, pos_fe="cohort_block", per_cohort_blocks=False)
            res["tests"].to_csv(ext_out / f"tests_{tag}.csv", index=False)
            res["cells"].to_csv(ext_out / f"pair_cells_{tag}.csv", index=False)
            ext[outcome] = res
        (ext_out / "data_used.json").write_text(json.dumps(info, indent=2))
    sources = [("Sensitivity cohorts (pooled)", tr, results, "block", True)]
    if ext is not None:
        sources.append(("Other datasets (pooled)", ext["tr"], ext, "cohort_block", False))
        # every dataset together: Sensitivity panel transitions + the other datasets
        # (elsewhere training ends 30 min before testing, so no training predecessor)
        comb_out = out / "All-Datasets-Combined"
        comb_out.mkdir(exist_ok=True)
        comb_trials = pd.concat([trials, ext["trials"]], ignore_index=True)
        comb_tr = pd.concat([tr_panel, ext["tr"]], ignore_index=True)
        comb = {"tr": comb_tr, "info": {"datasets": int(comb_tr["cohort"].nunique()),
                                        "flies": int(comb_tr["fly_id"].nunique()),
                                        "batches": int(comb_tr[["cohort", "batch"]].drop_duplicates().shape[0]),
                                        "transitions": len(comb_tr)}}
        for outcome in ("responded", "score"):
            tag = "pct_responding" if outcome == "responded" else "score"
            r_ = run_tests(comb_tr, comb_trials, outcome, args.n_perm_external, args.seed,
                           sensitized=SENSITIZED, pos_fe="cohort_block", per_cohort_blocks=False)
            r_["tests"].to_csv(comb_out / f"tests_{tag}.csv", index=False)
            r_["cells"].to_csv(comb_out / f"pair_cells_{tag}.csv", index=False)
            comb[outcome] = r_
        (comb_out / "data_used.json").write_text(json.dumps(comb["info"], indent=2))
        sources.append(("All datasets combined", comb_tr, comb, "cohort_block", False))
        results["combined"] = comb
    focus = pd.concat([focus_pair_table(sources, o) for o in ("responded", "score")], ignore_index=True)
    focus.to_csv(out / "focus_pairs.csv", index=False)
    results["focus"] = focus
    if not args.no_figures:
        import previous_odor_carryover_figures as figs
        figs.draw_all(out, tr, results, ext=ext, trials=trials)
    return tr, results, ac, ext


if __name__ == "__main__":
    main()
