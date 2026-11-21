#!/usr/bin/env python3
"""Does the trial number predict PER better than the odor delivered?

Hand-run, not wired into the pipeline. Sensitivity cohorts (the window flies of
``Results/Sensitiviy-Results(11-15)``) plus, as a replication, every other dataset whose
panels ran in a per-batch random order (same loader as previous_odor_carryover.py).
Writes ``<folder>/Trial-Number-vs-Odor/``.

Only the 7 randomized odors enter (paraffin and light-only sit at fixed positions, so
for them odor and trial number cannot be told apart). Every comparison is *within fly*:
responses are centred on each fly's own panel mean, so the question is "which of a
fly's trials did it respond on", not "which flies respond more".

* variance explained: within-fly R^2 of odor alone, trial number alone (categorical,
  and as a straight line), both; "unique" = what one adds on top of the other.
* held-out: the same R^2 for batches left out one at a time (effects fitted on the
  other batches), so the extra levels of either factor cannot buy a better score.
* exact randomization tests: the odor order is a uniform random permutation per batch
  panel, so shuffling trial numbers among a panel's trials (odor and response fixed)
  is the null for "trial number adds nothing beyond odor", and shuffling odors the
  null for "odor adds nothing beyond trial number". All flies of a batch move together.
* odor minus trial number (held-out R^2) with a batch-bootstrap 95 % CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import previous_odor_carryover as poc  # noqa: E402

OUT_NAME = "Trial-Number-vs-Odor"


def prepare(trials: pd.DataFrame) -> pd.DataFrame:
    t = trials[trials["odor"].isin(poc.RANDOMIZED_ODORS)].copy()
    t["panel"] = t["cohort"] + "|" + t["batch"] + "|" + t["block"]
    t["fly_panel"] = t["fly_id"] + "|" + t["block"]
    t["slot"] = t["panel"] + "|" + t["pos"].astype(str)
    return t.reset_index(drop=True)


def _demean(M: np.ndarray, g: np.ndarray, n_g: int) -> np.ndarray:
    M = np.atleast_2d(M.T).T if M.ndim == 1 else M
    s = np.zeros((n_g, M.shape[1]))
    np.add.at(s, g, M)
    cnt = np.bincount(g, minlength=n_g)[:, None]
    return M - (s / np.maximum(cnt, 1))[g]


def _dummies(keys: np.ndarray) -> np.ndarray:
    u, inv = np.unique(keys, return_inverse=True)
    d = np.zeros((len(keys), len(u)))
    d[np.arange(len(keys)), inv] = 1
    return d


class Scope:
    """One subset of trials with its fly-panel grouping and slot structure."""

    def __init__(self, t: pd.DataFrame, y: str, pooled: bool, by_block: bool = False):
        self.t = t.reset_index(drop=True)
        self.pooled = pooled
        # factor levels are split by cohort (pooled scopes) and by panel (pre+post scopes)
        pre = np.full(len(self.t), "", dtype=object)
        if pooled:
            pre = pre + self.t["cohort"].to_numpy(dtype=object) + "|"
        if by_block:
            pre = pre + self.t["block"].to_numpy(dtype=object) + "|"
        self.prefix = pre.astype(str)
        _, self.g = np.unique(self.t["fly_panel"].to_numpy(), return_inverse=True)
        self.ng = self.g.max() + 1
        yy = self.t[y].astype(float).to_numpy() * (100 if y == "responded" else 1)
        self.y = _demean(yy[:, None], self.g, self.ng)[:, 0]
        self.sst = float(self.y @ self.y)
        # slots: a batch panel x position; permutations act on slots
        slot_u, self.row_slot = np.unique(self.t["slot"].to_numpy(), return_inverse=True)
        first = pd.Series(np.arange(len(self.t))).groupby(self.row_slot).first().to_numpy()
        self.slot_odor = self.t["odor"].to_numpy()[first]
        self.slot_pos = self.t["pos"].to_numpy()[first]
        panel = self.t["panel"].to_numpy()[first]
        self.panels = [np.flatnonzero(panel == p) for p in np.unique(panel)]
        self.cohort = self.t["cohort"].to_numpy()
        self.batch = (self.t["cohort"] + "|" + self.t["batch"]).to_numpy()

    def design(self, odor: np.ndarray, pos: np.ndarray, which: str) -> np.ndarray:
        pre = self.prefix
        parts = []
        if "odor" in which:
            parts.append(_dummies(np.char.add(pre, odor.astype(str))))
        if "pos" in which:
            parts.append(_dummies(np.char.add(pre, pos.astype(str))))
        if "line" in which:      # trial number as a straight line (slope per cohort x panel)
            parts.append(_dummies(pre) * pos[:, None])
        return _demean(np.hstack(parts), self.g, self.ng)

    def r2(self, X: np.ndarray) -> float:
        b, *_ = np.linalg.lstsq(X, self.y, rcond=None)
        e = self.y - X @ b
        return 1 - float(e @ e) / self.sst

    def r2_heldout(self, X: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Leave-one-batch-out R^2; also returns per-batch SS (resid, total) for bootstrap."""
        ub, bi = np.unique(self.batch, return_inverse=True)
        ss_res = np.zeros(len(ub))
        ss_tot = np.zeros(len(ub))
        for k in range(len(ub)):
            te = bi == k
            b, *_ = np.linalg.lstsq(X[~te], self.y[~te], rcond=None)
            e = self.y[te] - X[te] @ b
            ss_res[k], ss_tot[k] = e @ e, self.y[te] @ self.y[te]
        return 1 - ss_res.sum() / ss_tot.sum(), ss_res, ss_tot

    def rows(self, slot_values: np.ndarray) -> np.ndarray:
        return slot_values[self.row_slot]

    def shuffled(self, rng, values: np.ndarray) -> np.ndarray:
        out = values.copy()
        for p in self.panels:
            out[p] = values[p[rng.permutation(len(p))]]
        return out


def analyse_scope(name: str, sc: Scope, n_perm: int, rng) -> dict:
    odor, pos = sc.rows(sc.slot_odor), sc.rows(sc.slot_pos)
    X = {w: sc.design(odor, pos, w) for w in ("odor", "pos", "line", "odor+pos", "odor+line")}
    r2 = {w: sc.r2(X[w]) for w in X}
    ho = {w: sc.r2_heldout(X[w]) for w in ("odor", "pos", "line")}
    uniq_pos = r2["odor+pos"] - r2["odor"]
    uniq_line = r2["odor+line"] - r2["odor"]
    uniq_odor = r2["odor+pos"] - r2["pos"]
    null_pos, null_line, null_odor = (np.empty(n_perm) for _ in range(3))
    for i in range(n_perm):
        sp = sc.rows(sc.shuffled(rng, sc.slot_pos))          # trial numbers shuffled, odor fixed
        xo = X["odor"]
        null_pos[i] = sc.r2(np.hstack([xo, sc.design(odor, sp, "pos")])) - r2["odor"]
        null_line[i] = sc.r2(np.hstack([xo, sc.design(odor, sp, "line")])) - r2["odor"]
        so = sc.rows(sc.shuffled(rng, sc.slot_odor))         # odors shuffled, trial number fixed
        null_odor[i] = sc.r2(np.hstack([X["pos"], sc.design(so, pos, "odor")])) - r2["pos"]
    # batch bootstrap of the held-out difference odor - trial number
    _, res_o, tot = ho["odor"]
    _, res_p, _ = ho["pos"]
    draws = rng.integers(0, len(tot), size=(2000, len(tot)))
    diff = (1 - res_o[draws].sum(1) / tot[draws].sum(1)) - (1 - res_p[draws].sum(1) / tot[draws].sum(1))
    p = lambda o, n: float((1 + np.sum(n >= o - 1e-15)) / (1 + len(n)))  # noqa: E731
    return {"scope": name, "n_trials": len(sc.y), "n_flies": int(sc.ng),
            "n_batches": int(len(np.unique(sc.batch))),
            "r2_odor_pct": 100 * r2["odor"], "r2_trial_number_pct": 100 * r2["pos"],
            "r2_trial_number_line_pct": 100 * r2["line"],
            "heldout_r2_odor_pct": 100 * ho["odor"][0], "heldout_r2_trial_number_pct": 100 * ho["pos"][0],
            "heldout_r2_trial_number_line_pct": 100 * ho["line"][0],
            "odor_minus_trial_number_heldout_pct": 100 * (ho["odor"][0] - ho["pos"][0]),
            "diff_ci_low": 100 * float(np.percentile(diff, 2.5)),
            "diff_ci_high": 100 * float(np.percentile(diff, 97.5)),
            "unique_odor_pct": 100 * uniq_odor, "p_odor_beyond_trial_number": p(uniq_odor, null_odor),
            "unique_trial_number_pct": 100 * uniq_pos,
            "p_trial_number_beyond_odor": p(uniq_pos, null_pos),
            "unique_trial_number_line_pct": 100 * uniq_line,
            "p_trial_number_line_beyond_odor": p(uniq_line, null_line),
            "n_perm": n_perm}


def scopes_for(t: pd.DataFrame, y: str, cohorts: list[str], label: str = "") -> list[tuple[str, Scope]]:
    out = []
    for block in ("pre", "post", "pre+post"):
        tb = t if block == "pre+post" else t[t["block"] == block]
        both = block == "pre+post"
        if tb.empty:
            continue
        out.append((f"{label}all cohorts|{block}", Scope(tb, y, pooled=True, by_block=both)))
        for c in cohorts:
            tc = tb[tb["cohort"] == c]
            if tc["batch"].nunique() >= 3:
                out.append((f"{c}|{block}", Scope(tc, y, pooled=False, by_block=both)))
    return out


def rate_tables(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c, b), g in list(t.groupby(["cohort", "block"])) + [(("all", b), g) for b, g in t.groupby("block")]:
        for kind, col in (("odor", "odor"), ("trial_number", "pos")):
            for lev, h in g.groupby(col):
                rows.append({"cohort": c, "block": b, "factor": kind, "level": lev,
                             "n_trials": len(h), "pct_responding": 100 * h["responded"].mean(),
                             "mean_score": h["score"].mean()})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folder", type=Path, default=poc.DEFAULT_FOLDER)
    ap.add_argument("--csv-dir", type=Path, default=Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys"))
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = args.folder / OUT_NAME
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t = prepare(poc.load(args.folder))
    data = {"sensitivity": t}
    if not args.no_external:
        ext, info = poc.load_external(args.csv_dir / "model_predictions.csv", args.csv_dir,
                                      args.csv_dir / "flagged-flys-truth.csv", poc.COHORTS)
        data["other_datasets"] = prepare(ext)
        (out / "other_datasets_used.json").write_text(json.dumps(info, indent=2))
    results = {}
    for outcome in ("responded", "score"):
        rows = []
        for sc_name, sc in scopes_for(t, outcome, poc.COHORTS):
            rows.append(analyse_scope(sc_name, sc, args.n_perm, rng))
        if "other_datasets" in data:
            te = data["other_datasets"]
            for block in ("pre", "post", "pre+post"):
                tb = te if block == "pre+post" else te[te["block"] == block]
                rows.append(analyse_scope(f"other datasets (pooled)|{block}",
                                          Scope(tb, outcome, pooled=True, by_block=block == "pre+post"),
                                          args.n_perm, rng))
        res = pd.DataFrame(rows)
        tag = "pct_responding" if outcome == "responded" else "score"
        res.to_csv(out / f"trial_number_vs_odor_{tag}.csv", index=False)
        results[outcome] = res
    rates = rate_tables(t)
    rates.to_csv(out / "rates_by_odor_and_trial_number.csv", index=False)
    if not args.no_figures:
        import previous_odor_carryover_figures as figs
        figs.trial_number_figure(out, results["responded"], rates)
    return results, rates


if __name__ == "__main__":
    main()
