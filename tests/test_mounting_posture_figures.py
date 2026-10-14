"""The Mounting-Posture figure set and ``report.md`` (``scripts/analysis/mounting_posture_figures.py``).

Rules pinned here, on synthetic fly-trial / fly tables (the fly table is always built from the trial table):

* the six PNGs of the Design are written at 300 dpi, and every figure carries an ``n=`` label (``n=… trials / … flies``
  for pooled-trial bars, ``n=… flies`` for per-fly plots);
* the fly is the unit of inference: per-fly scatters use the FLY table's rig-centred pitch, their permutation p is
  stratified by dataset x BATCH (flies of one batch share the mounting day, the glue session and the starvation, so only
  they are interchangeable; a batch nests a rig) and decides every verdict, the dataset x rig p is printed beside it as a
  secondary number, the tercile p is the per-fly test of response rate vs pitch, and the baseline-state p is the
  within-fly permutation (whose limits -- odor and trial position are ignored -- the report says); a stratified test in
  which no stratum holds two flies has nothing to shuffle and is "Not enough data to tell"; every p is labelled
  "uncorrected";
* the report records what produced it: the time (UTC), every mounting setting, and for each rig the retracted reference
  it used, how many flies it came from, whether it was configured, and a WARNING when most of the rig's flies never rest
  below it;
* displayed rates pool trials (a tercile of one 1-of-1 fly and one 0-of-9 fly is 10 %, never 50 %), and trials of a fly
  whose identity drifted never enter a pooled bar;
* the response analyses use only scored trials (pretest + testing rows with a score), and the within-fly test shuffles
  the state inside each fly's trials of one trial type, so a pretest-vs-testing difference cannot make its p small; the
  difference shown beside that p is the Mantel-Haenszel one over the same fly x trial-type groups (the bars stay pooled);
* a comparison with fewer than 3 flies to compare is "Not enough data to tell" (never "No clear difference" or "Yes"),
  and no figure or report says "1 flies" or "+nan points";
* a figure that fails to build leaves no stale PNG behind, and so does a run with nothing in scope; the report names
  every figure that was not drawn;
* titles are the question, and the plain-language answer is computed from the data (never hard-coded);
* style: blue per-fly points, orange ``//`` for held / spontaneous states, filled marker = held back and hollow = held
  down, thin bars, no twin axes, no text over text, and the global matplotlib style is left alone;
* ``report.md`` lists every exclusions key, the rho / p of every per-fly scatter and every permutation p, and says
  "uncorrected"; with nothing in scope it says so and no figure is drawn; further sections can be appended.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.analysis import mounting_posture as MP  # noqa: E402
from scripts.analysis import mounting_posture_figures as F  # noqa: E402
from fbpipe.analysis import mounting as M  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
MCFG = M.parse_mounting_block({})
NAMES = ("mount_pitch_per_fly", "mount_pitch_vs_spontaneous_per", "mount_pitch_vs_held_out",
         "response_by_mount_pitch_tercile", "baseline_state_vs_response", "pullins_held_vs_retracted")
N_PERM = 300

# ---------------------------------------------------------------------------
# Synthetic tables
# ---------------------------------------------------------------------------

CELLS = (("DS-A", "BehaviorLocust"), ("DS-A", "Flybehavior2"), ("DS-B", "Flybehavior2"), ("DS-B", "Flybehavior3"))
RIG_SHIFT = {"BehaviorLocust": 2.0, "Flybehavior2": -4.0, "Flybehavior3": 12.0}      # camera roll differs per rig


def trial_table(rows) -> pd.DataFrame:
    """A fly-trial table with every documented column; what a row does not set is NaN (identity consistent by default)."""
    df = pd.DataFrame(rows)
    for column in MP.FLY_TRIAL_COLUMNS:
        if column not in df:
            df[column] = np.nan
    df["identity_consistent"] = df["identity_consistent"].fillna(True).astype(bool)
    return df.reindex(columns=MP.FLY_TRIAL_COLUMNS)


def row(fly_id, rig="R", dataset="D", batch="b1", **cols):
    number = int(fly_id.split("fly")[-1])
    return {"dataset": dataset, "batch": batch, "fly_number": number, "fly_id": f"{dataset}/{batch}/{fly_id}", "rig": rig,
            "trial_type": "testing", "identity_consistent": True, **cols}


def cohort(seed=1, per_cell=8, *, tilt=0.0, hold=0.0, respond=0.0, batches=1) -> pd.DataFrame:
    """Flies with unbalanced trial counts; ``tilt`` / ``hold`` / ``respond`` make the spontaneous-PER, held-out and
    response probabilities rise with how head-up the fly is mounted relative to its rig (0 = no relation). Each
    (dataset, rig) cell is one batch, or ``batches`` of them (flies alternate), so the batch strata can differ from the
    rig strata."""
    rng = np.random.default_rng(seed)
    rows = []
    for cell, (dataset, rig) in enumerate(CELLS):
        for k in range(per_cell):
            dev = rng.normal(0.0, 4.0)
            z = dev / 4.0
            lean_back = rng.random()
            batch = f"batch_{cell}" if batches == 1 else f"batch_{cell}_{k % batches}"
            for t in range(int(rng.integers(5, 14))):
                held = rng.random() < 1 / (1 + np.exp(1.2 - hold * z))
                spont = rng.random() < np.clip(0.2 + tilt * z, 0.02, 0.95)
                responded = rng.random() < np.clip(0.25 + 0.2 * held + 0.25 * spont + respond * z, 0.02, 0.95)
                scored = rng.random() > 0.1
                known = rng.random() > 0.05
                kind = "pretest" if t < 2 else "testing"
                rows.append({
                    "dataset": dataset, "batch": batch, "fly_number": k + 1,
                    "fly_id": f"{dataset}/{batch}/fly{k + 1}", "rig": rig, "trial": f"t{t}", "trial_type": kind,
                    "trial_index": t, "identity_consistent": True,
                    "trial_body_pitch_deg": RIG_SHIFT[rig] + dev + rng.normal(0.0, 0.5),
                    "rest_eu": rng.uniform(0.4, 0.8), "pre_min": 0.5, "stim_min": float(rng.choice([1 / 15, 0.5])),
                    "held_out": float(held), "held_subtype": ("held_back" if rng.random() < lean_back else "held_down") if held
                    else "retracted",
                    "spont_per_events": float(rng.poisson(1.5)) if spont and known else 0.0 if known else np.nan,
                    "spont_per_any": float(spont) if known else np.nan,
                    "pullin_events_pre": float(rng.poisson(0.5 if held else 0.1)) if known else np.nan,
                    "pullin_events_stim": float(rng.poisson(2.0 if held else 0.2)) if known else np.nan,
                    "score": (3.0 if responded else 0.0) if scored else np.nan,
                    "responded": float(responded) if scored else np.nan})
    return trial_table(rows)


def fly_table(ft) -> pd.DataFrame:
    return MP.build_fly_table(ft, MCFG)


@pytest.fixture(scope="module")
def data():
    """Spontaneous PER and the response rise with pitch (strongly, so the answer is "Yes"); held-out does not."""
    ft = cohort(seed=7, tilt=0.30, respond=0.15)
    fly = fly_table(ft)
    return SimpleNamespace(ft=ft, fly=fly, stats=F.compute_stats(ft, fly, n_perm=N_PERM, seed=0))


@pytest.fixture(scope="module")
def figures(data):
    return F.build_figures(data.ft, data.fly, data.stats)


@pytest.fixture(scope="module")
def null_data():
    """The same flies with no relation between pitch and anything."""
    ft = cohort(seed=11)
    fly = fly_table(ft)
    return SimpleNamespace(ft=ft, fly=fly, stats=F.compute_stats(ft, fly, n_perm=N_PERM, seed=0))


def squash(text: str) -> str:
    return " ".join(text.split())


def all_text(fig) -> str:
    """Every piece of text a figure carries (titles, labels, annotations, tick labels, legends), whitespace squashed."""
    out = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        out += [ax.get_title(loc) for loc in ("center", "left", "right")] + [ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.texts]
        out += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        if ax.get_legend():
            out += [t.get_text() for t in ax.get_legend().get_texts()]
    for legend in fig.legends:
        out += [t.get_text() for t in legend.get_texts()]
    return squash(" ".join(s for s in out if s))


def drawn_texts(fig):
    """``(text, bbox)`` of every text artist that is actually drawn (a tick label outside the view is not)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden = set()
    for ax in fig.axes:
        for axis, limits, index in ((ax.xaxis, ax.get_xlim(), 0), (ax.yaxis, ax.get_ylim(), 1)):
            lo, hi = sorted(limits)
            for label in axis.get_ticklabels(minor=False) + axis.get_ticklabels(minor=True):
                if not lo - 1e-9 <= label.get_position()[index] <= hi + 1e-9:
                    hidden.add(id(label))
    return [(t.get_text(), t.get_window_extent(renderer)) for t in fig.findobj(Text)
            if id(t) not in hidden and t.get_visible() and t.get_text().strip()]


# ---------------------------------------------------------------------------
# The statistics behind the figures
# ---------------------------------------------------------------------------

def test_the_per_fly_scatters_use_the_rig_centred_pitch_of_the_fly_table(data):
    fly, s = data.fly, data.stats["spontaneous"]
    rho, p = M.spearman(fly["mount_pitch_centred_deg"], fly["spont_per_frac"])
    raw_rho, _ = M.spearman(fly["mount_pitch_deg"], fly["spont_per_frac"])
    assert s["rho"] == pytest.approx(rho) and s["p"] == pytest.approx(p)
    assert s["n_flies"] == int(fly[["mount_pitch_centred_deg", "spont_per_frac"]].notna().all(axis=1).sum()) == 32
    assert abs(rho - raw_rho) > 0.05                              # the cohort's rig offsets make the two differ


def batch_strata(fly) -> np.ndarray:
    return (fly["dataset"] + " | " + fly["batch"]).to_numpy()


def rig_strata(fly) -> np.ndarray:
    return (fly["dataset"] + " | " + fly["rig"]).to_numpy()


def test_the_scatter_permutation_p_is_stratified_by_dataset_and_batch_with_dataset_and_rig_beside_it(data):
    fly = data.fly
    for key, column in (("spontaneous", "spont_per_frac"), ("held_out", "held_out_frac")):
        by_batch = M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly[column], batch_strata(fly), n=N_PERM, seed=0)
        by_rig = M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly[column], rig_strata(fly), n=N_PERM, seed=0)
        assert data.stats[key]["p_perm"] == by_batch and data.stats[key]["p_perm_rig"] == by_rig
        assert data.stats[key]["n_strata"] == 4 and data.stats[key]["n_strata_shuffled"] == 4


def test_batches_inside_a_rig_are_separate_strata_and_the_rig_strata_are_counted_apart():
    ft = cohort(seed=7, tilt=0.30, respond=0.15, batches=2)                 # 8 batches, two in each of the 4 rig cells
    fly = fly_table(ft)
    stats = F.compute_stats(ft, fly, n_perm=N_PERM, seed=0)
    for key, column in (("spontaneous", "spont_per_frac"), ("held_out", "held_out_frac"), ("tercile", "response_rate")):
        s = stats[key]
        assert s["p_perm"] == M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly[column], batch_strata(fly),
                                                        n=N_PERM, seed=0), key
        assert s["p_perm_rig"] == M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly[column], rig_strata(fly),
                                                            n=N_PERM, seed=0), key
        assert (s["n_strata"], s["n_strata_shuffled"]) == (8, 8) and (s["n_strata_rig"], s["n_strata_rig_shuffled"]) == (4, 4), key


def batch_confound():
    """Twelve flies of one dataset and rig in two batches: pitch and spontaneous PER both rise from B1 to B2 (the second
    batch was mounted more head-up on another day), but inside each batch the two are unrelated, so only a test that
    shuffles within the batch sees there is nothing."""
    rows = []
    for batch, offset, pitches, ranks in (("B1", 0, [-3.0, -2.6, -2.2, -1.8, -1.4, -1.0], [3, 6, 1, 5, 2, 4]),
                                          ("B2", 6, [1.0, 1.4, 1.8, 2.2, 2.6, 3.0], [2, 5, 3, 6, 1, 4])):
        for k, (pitch, rank) in enumerate(zip(pitches, ranks)):
            n_spont = rank - 1 if batch == "B1" else rank + 4         # B1: 0-5 of 10 trials, B2: 5-10
            for t in range(10):
                rows.append(row(f"fly{offset + k + 1}", batch=batch, trial_body_pitch_deg=pitch, trial_index=t,
                                spont_per_any=float(t < n_spont), spont_per_events=float(t < n_spont)))
    return trial_table(rows)


def test_a_batch_effect_on_both_pitch_and_outcome_does_not_make_the_answer_yes():
    ft = batch_confound()
    fly = fly_table(ft)
    s = F.compute_stats(ft, fly, n_perm=2000, seed=0)["spontaneous"]
    assert s["p_perm_rig"] < 0.05, "shuffled over the whole rig, the batch shift alone reads as a link"
    assert s["p_perm"] > 0.3 and s["verdict"] == "No clear link", "shuffled inside each batch there is nothing"
    assert (s["n_strata"], s["n_strata_rig"]) == (2, 1)


def singleton_batches():
    """Four flies, one per batch (one dataset, one rig): every stratum holds a single fly, so nothing can be shuffled."""
    rows = []
    for k in range(4):
        for t in range(6):
            held = t <= k
            rows.append(row(f"fly{k + 1}", batch=f"B{k}", trial_body_pitch_deg=float(k), trial_index=t,
                            spont_per_any=float(held), spont_per_events=float(held),
                            held_subtype="held_down" if held else "retracted", held_out=float(held),
                            responded=float(t <= k + 1), score=3.0 * float(t <= k + 1)))
    return trial_table(rows)


def test_with_no_batch_holding_two_flies_every_per_fly_test_is_not_enough_data_never_no_clear_link():
    ft = singleton_batches()
    fly = fly_table(ft)
    stats = F.compute_stats(ft, fly, n_perm=N_PERM)
    for key in ("spontaneous", "held_out", "tercile"):
        s = stats[key]
        assert s["n_flies"] == 4 and s["n_strata"] == 4 and s["n_strata_shuffled"] == 0, key
        assert s["p_perm"] == 1.0 and s["rho"] == pytest.approx(1.0), key         # the permutation had nothing to move ...
        assert s["verdict"].startswith("Not enough data to tell"), key              # ... so it must not say "No clear link"
        assert "dataset × batch" in s["verdict"] and "No clear" not in s["verdict"], key
    figs = F.build_figures(ft, fly, stats)
    assert "Not enough data to tell" in all_text(figs["mount_pitch_vs_spontaneous_per"])
    assert "Not enough data to tell" in all_text(figs["response_by_mount_pitch_tercile"])
    assert "No clear" not in all_text(figs["mount_pitch_vs_spontaneous_per"])


def test_one_batch_with_two_flies_is_enough_to_shuffle_and_the_answer_is_a_real_one():
    ft = singleton_batches()
    ft["batch"] = ft["batch"].replace({"B1": "B0"})                              # flies 1 and 2 now share a batch
    fly = fly_table(ft)
    s = F.compute_stats(ft, fly, n_perm=N_PERM)["spontaneous"]
    assert s["n_strata"] == 3 and s["n_strata_shuffled"] == 1
    assert not s["verdict"].startswith("Not enough data to tell")


def strata_confound():
    """Twelve flies of one rig in two datasets: pitch and spontaneous PER both rise from D1 to D2, but inside each
    dataset the two are unrelated, so only a test that shuffles within the dataset sees there is nothing."""
    rows = []
    for dataset, pitches, ranks in (("D1", [-3.0, -2.6, -2.2, -1.8, -1.4, -1.0], [3, 6, 1, 5, 2, 4]),
                                    ("D2", [1.0, 1.4, 1.8, 2.2, 2.6, 3.0], [2, 5, 3, 6, 1, 4])):
        for k, (pitch, rank) in enumerate(zip(pitches, ranks)):
            n_spont = rank - 1 if dataset == "D1" else rank + 4       # D1: 0-5 of 10 trials, D2: 5-10
            for t in range(10):
                rows.append(row(f"fly{k + 1}", dataset=dataset, trial_body_pitch_deg=pitch, trial_index=t,
                                spont_per_any=float(t < n_spont), spont_per_events=float(t < n_spont)))
    return trial_table(rows)


def test_the_answer_follows_the_stratified_p_not_the_pooled_spearman_p():
    ft = strata_confound()
    fly = fly_table(ft)
    stats = F.compute_stats(ft, fly, n_perm=2000, seed=0)["spontaneous"]
    strata = batch_strata(fly)
    pooled = M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly["spont_per_frac"], ["all"] * len(fly),
                                        n=2000, seed=0)
    assert stats["p"] < 0.05 and pooled < 0.05                    # pooled over the datasets the two look related
    assert stats["p_perm"] == M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly["spont_per_frac"], strata,
                                                         n=2000, seed=0) > 0.3
    assert stats["verdict"] == "No clear link" and stats["n_strata"] == 2


def test_the_held_out_scatter_uses_the_held_out_fraction_of_the_fly_table(data):
    fly, s = data.fly, data.stats["held_out"]
    rho, p = M.spearman(fly["mount_pitch_centred_deg"], fly["held_out_frac"])
    assert s["rho"] == pytest.approx(rho) and s["p"] == pytest.approx(p) and s["n_flies"] == 32


def hand_built_terciles():
    """Six flies of one rig and dataset: two per tercile of the centred pitch, with deliberately unequal trial counts."""
    rows = []
    for fid, pitch, responded in (("fly1", -3, [1]), ("fly2", -2, [0] * 9), ("fly3", 0, [1, 0]), ("fly4", 1, [0, 1]),
                                  ("fly5", 4, [1] * 4), ("fly6", 5, [0])):
        rows += [row(fid, trial_body_pitch_deg=float(pitch), responded=float(r), score=3.0 * r, trial_index=i)
                 for i, r in enumerate(responded)]
    return trial_table(rows)


def test_the_tercile_bars_pool_trials_and_never_average_the_fly_rates():
    ft = hand_built_terciles()
    fly = fly_table(ft)
    bins = F.compute_stats(ft, fly, n_perm=N_PERM)["tercile"]["bins"]
    assert [b["n_trials"] for b in bins] == [10, 4, 5]
    assert [b["n_flies"] for b in bins] == [2, 2, 2]
    assert [b["rate"] for b in bins] == pytest.approx([0.1, 0.5, 0.8])             # pooled: 1/10, 2/4, 4/5
    per_fly_mean = fly.assign(third=[0, 0, 1, 1, 2, 2]).groupby("third")["response_rate"].mean()
    assert list(per_fly_mean) == pytest.approx([0.5, 0.5, 0.5])                    # what averaging averages would show


def test_the_terciles_split_the_flies_by_the_rig_centred_pitch(data):
    bins = data.stats["tercile"]["bins"]
    fly = data.fly[data.fly["response_rate"].notna()]
    assert [b["n_flies"] for b in bins] == [len(a) for a in np.array_split(np.arange(len(fly)), 3)]
    assert bins[0]["hi_deg"] <= bins[1]["lo_deg"] and bins[1]["hi_deg"] <= bins[2]["lo_deg"]
    centred = fly["mount_pitch_centred_deg"]
    assert bins[0]["lo_deg"] == pytest.approx(centred.min()) and bins[2]["hi_deg"] == pytest.approx(centred.max())


def test_the_tercile_trial_counts_are_the_scored_trials_of_the_flies_in_each_third(data):
    ft, bins = data.ft, data.stats["tercile"]["bins"]
    assert sum(b["n_trials"] for b in bins) == int(ft["responded"].notna().sum())
    assert sum(b["n_responded"] for b in bins) == int(ft["responded"].sum())


def test_the_tercile_p_is_the_per_fly_stratified_permutation_of_response_rate_against_pitch(data):
    fly, t = data.fly, data.stats["tercile"]
    assert t["p_perm"] == M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly["response_rate"],
                                                     batch_strata(fly), n=N_PERM, seed=0)
    assert t["p_perm_rig"] == M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly["response_rate"],
                                                         rig_strata(fly), n=N_PERM, seed=0)
    assert t["rho"] == pytest.approx(M.spearman(fly["mount_pitch_centred_deg"], fly["response_rate"])[0])
    assert t["n_flies"] == 32


def phased(df: pd.DataFrame) -> pd.DataFrame:
    """The key the within-fly test shuffles inside: one group per fly and trial type."""
    return df.assign(fly_phase=df["fly_id"] + "|" + df["trial_type"])


def test_the_baseline_state_p_is_the_within_fly_permutation_inside_each_fly_and_trial_type(data):
    ft, base = data.ft, data.stats["baseline"]
    scored = phased(ft[ft["score"].notna() & ft["held_subtype"].notna()])
    scored = scored.assign(held_any=(scored["held_subtype"] != "retracted").astype(float))
    pooled, p = M.within_fly_permutation(scored, state="held_any", outcome="responded", fly="fly_phase", n=N_PERM, seed=0)
    assert base["proboscis"]["diff_pooled"] == pytest.approx(pooled) and base["proboscis"]["p_perm"] == p
    spont = phased(ft[ft["score"].notna() & ft["spont_per_any"].notna()])
    pooled, p = M.within_fly_permutation(spont, state="spont_per_any", outcome="responded", fly="fly_phase", n=N_PERM, seed=0)
    assert base["spontaneous"]["diff_pooled"] == pytest.approx(pooled) and base["spontaneous"]["p_perm"] == p


def test_the_difference_shown_is_the_mantel_haenszel_one_over_the_strata_the_test_shuffles_in(data):
    ft, base = data.ft, data.stats["baseline"]
    scored = phased(ft[ft["score"].notna() & ft["held_subtype"].notna()])
    scored = scored.assign(held_any=(scored["held_subtype"] != "retracted").astype(float))
    mh = M.stratified_risk_difference(scored, state="held_any", outcome="responded", stratum="fly_phase")
    assert base["proboscis"]["diff"] == pytest.approx(mh)
    assert base["proboscis"]["diff"] != pytest.approx(base["proboscis"]["diff_pooled"])      # this cohort tells them apart
    spont = phased(ft[ft["score"].notna() & ft["spont_per_any"].notna()])
    mh = M.stratified_risk_difference(spont, state="spont_per_any", outcome="responded", stratum="fly_phase")
    assert base["spontaneous"]["diff"] == pytest.approx(mh)


def test_the_baseline_test_counts_the_flies_and_groups_seen_in_both_states(data):
    ft, test = data.ft, data.stats["baseline"]["spontaneous"]
    use = phased(ft[ft["score"].notna() & ft["spont_per_any"].notna()])
    both = use.groupby("fly_phase")["spont_per_any"].nunique()
    in_both = use[use["fly_phase"].isin(both[both == 2].index)]
    assert test["n_groups_both"] == int((both == 2).sum())
    assert test["n_flies_both"] == in_both["fly_id"].nunique() < use["fly_id"].nunique()    # some flies never show both
    assert test["n_trials_both"] == len(in_both)


def phase_confound(n_flies=12) -> pd.DataFrame:
    """Every fly has 16 pretest and 16 testing trials. The state (held out / spontaneous PER) and the response are both far
    more common in testing than in pretest, but inside each trial type they are exactly unrelated (25 % / 25 % and
    75 % / 75 %), so trial type alone drives the pooled difference of +25 points."""
    rows = []
    for k in range(n_flies):
        for phase, n_on, n_on_responding, n_off_responding in (("pretest", 4, 1, 3), ("testing", 12, 9, 3)):
            for t in range(16):
                on = t < n_on
                responded = t < n_on_responding if on else t - n_on < n_off_responding
                rows.append(row(f"fly{k + 1}", trial_type=phase, trial_index=t, held_subtype="held_down" if on else "retracted",
                                spont_per_any=float(on), responded=float(responded), score=3.0 * responded,
                                trial_body_pitch_deg=float(k)))
    return trial_table(rows)


def test_a_trial_type_that_drives_both_state_and_response_does_not_give_a_small_p():
    ft = phase_confound()
    fly = fly_table(ft)
    naive = ft.assign(held_any=(ft["held_subtype"] != "retracted").astype(float))
    assert M.within_fly_permutation(naive, state="held_any", outcome="responded", fly="fly_id", n=2000, seed=0)[1] < 0.01
    base = F.compute_stats(ft, fly, n_perm=2000, seed=0)["baseline"]    # shuffling inside a fly alone would call this a link
    for key in ("proboscis", "spontaneous"):
        assert base[key]["diff_pooled"] == pytest.approx(0.25), key               # the pooled difference is large ...
        assert base[key]["diff"] == pytest.approx(0.0, abs=1e-9), key             # ... and the one that goes with the p is nil
        assert base[key]["p_perm"] > 0.3 and base[key]["verdict"] == "No clear difference", key
        assert base[key]["n_flies_both"] == 12 and base[key]["n_groups_both"] == 24


def hand_built_states():
    rows = []
    for fid, spec in {"fly1": ["retracted", "retracted", "held_down", "held_down"],
                      "fly2": ["retracted", "held_back", "held_back"],
                      "fly3": ["retracted", "retracted"]}.items():
        for i, state in enumerate(spec):
            responded = 1.0 if (state != "retracted" and fid != "fly3") or (fid == "fly1" and i == 0) else 0.0
            rows.append(row(fid, held_subtype=state, responded=responded, score=3.0 * responded, trial_index=i,
                            spont_per_any=float(i % 2), pullin_events_pre=1.0, pullin_events_stim=2.0,
                            pre_min=0.5, stim_min=0.1))
    return trial_table(rows)


def test_the_baseline_bars_pool_the_trials_of_each_proboscis_state():
    ft = hand_built_states()
    bars = {b["state"]: b for b in F.compute_stats(ft, fly_table(ft), n_perm=N_PERM)["baseline"]["proboscis"]["bars"]}
    assert (bars["retracted"]["n_trials"], bars["retracted"]["n_flies"]) == (5, 3)
    assert bars["retracted"]["rate"] == pytest.approx(1 / 5)
    assert (bars["held_down"]["n_trials"], bars["held_down"]["n_flies"], bars["held_down"]["rate"]) == (2, 1, 1.0)
    assert (bars["held_back"]["n_trials"], bars["held_back"]["rate"]) == (2, 1.0)
    assert "held_forward" not in bars                                              # drawn only when a trial has it


def test_a_held_forward_trial_gets_its_own_bar_and_counts_as_held_in_the_test():
    ft = hand_built_states()
    ft.loc[ft.index[0], "held_subtype"] = "held_forward"
    stats = F.compute_stats(ft, fly_table(ft), n_perm=N_PERM)["baseline"]["proboscis"]
    assert [b["state"] for b in stats["bars"]] == ["retracted", "held_down", "held_back", "held_forward"]


def sparse_states(n_both: int) -> pd.DataFrame:
    """Six flies; ``n_both`` of them are seen both held out and retracted (the response follows the state exactly), the others
    only retracted, so only ``n_both`` flies can be compared."""
    rows = []
    for k in range(6):
        for t in range(20):
            on = k < n_both and t % 2 == 0
            rows.append(row(f"fly{k + 1}", trial_index=t, held_subtype="held_down" if on else "retracted",
                            spont_per_any=float(on), responded=float(on), score=3.0 * on, trial_body_pitch_deg=float(k)))
    return trial_table(rows)


@pytest.mark.parametrize("n_both, answer", [(0, "Not enough data to tell"), (1, "Not enough data to tell"),
                                            (2, "Not enough data to tell"),
                                            (3, "Yes — trials with the proboscis held out respond more often")])
def test_a_baseline_comparison_needs_three_flies_that_have_both_states(n_both, answer):
    ft = sparse_states(n_both)
    pro = F.compute_stats(ft, fly_table(ft), n_perm=2000, seed=0)["baseline"]["proboscis"]
    assert pro["n_flies_both"] == n_both and pro["verdict"] == answer
    if n_both == 2:
        assert pro["p_perm"] < 0.05                                  # the p alone would have said "Yes": the gate is the flies


def test_the_figure_and_the_report_show_the_mantel_haenszel_difference_beside_the_pooled_bars(data, figures, report):
    text = all_text(figures["baseline_state_vs_response"])
    for key in ("proboscis", "spontaneous"):
        panel = data.stats["baseline"][key]
        assert f"within fly × trial type (MH): {F.points_text(panel['diff'])}" in text, key
        assert f"{F.points_text(panel['diff'], 1)} (Mantel–Haenszel" in report, key
        # diff_pooled pools only the fly × trial-type groups seen in both states -- not the trials the bars count
        assert ("the pooled difference over those same fly × trial-type groups (not adjusted for trial type) is "
                f"{F.points_text(panel['diff_pooled'], 1)}") in report, key
    assert "pooled difference of the bars" not in report
    assert "phase" not in text.lower() and "phase" not in report.lower(), "one name for the grouping: fly × trial type"
    assert "Mantel–Haenszel" in report.split("## Exclusions")[0]
    assert "- Held out (any sub-type) minus retracted, within fly × trial type: " in report      # a sentence, not a fragment
    assert "- Spontaneous PER minus none, within fly × trial type: " in report


def test_a_difference_of_nothing_is_written_as_zero_points_never_minus_zero():
    assert F.points_text(-1e-17) == "+0 points" and F.points_text(1e-17, 1) == "+0.0 points"
    assert F.points_text(-0.0004, 1) == "+0.0 points"


@pytest.mark.parametrize("n_both, wording", [(0, "no fly was seen in both states"),
                                             (1, "1 fly has both states (needs at least 3)"),
                                             (2, "2 flies have both states (needs at least 3)")])
def test_the_baseline_note_says_one_fly_not_one_flies_and_never_nan(n_both, wording, tmp_path):
    ft = sparse_states(n_both)
    fly = fly_table(ft)
    stats = F.compute_stats(ft, fly, n_perm=N_PERM)
    figure = all_text(F.build_figures(ft, fly, stats)["baseline_state_vs_response"])
    report = F.write_report(ft, fly, EXCLUSIONS, stats, tmp_path).read_text(encoding="utf-8")
    assert wording in figure and "Not enough data to tell" in figure
    assert ("fewer than 3 flies" in report) == (n_both > 0) and "Not enough data to tell" in report
    for text in (figure, report):
        assert not re.search(r"(?i)\bnan\b|\b1 flies\b", text), text


def test_the_report_says_one_fly_for_a_single_fly_and_one_stratum_for_a_single_stratum(tmp_path):
    ft = sparse_states(1)
    fly = fly_table(ft)
    report = F.write_report(ft, fly, EXCLUSIONS, F.compute_stats(ft, fly, n_perm=N_PERM), tmp_path).read_text(encoding="utf-8")
    assert re.search(r"\b1 fly\b", report) and re.search(r"\b1 stratum\b", report)
    assert not re.search(r"\b1 flies\b|\b1 strata\b", report)


def test_the_pull_in_rates_pool_events_over_minutes_for_held_and_retracted_trials():
    ft = hand_built_states()
    ft["stim_min"] = [0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]                # row 2 is a held trial with a long stimulus
    groups = F.compute_stats(ft, fly_table(ft), n_perm=N_PERM)["pullins"]["groups"]
    held, retracted = groups["held"], groups["retracted"]
    assert (held["n_trials"], held["n_flies"]) == (4, 2) and (retracted["n_trials"], retracted["n_flies"]) == (5, 3)
    assert held["rate_pre"] == pytest.approx(4 * 1.0 / (4 * 0.5)) and retracted["rate_pre"] == pytest.approx(2.0)
    assert held["rate_stim"] == pytest.approx(4 * 2.0 / 0.7)                       # pooled events / pooled minutes ...
    assert held["rate_stim"] != pytest.approx(np.mean([2.0 / 0.4, 20.0, 20.0, 20.0]))   # ... not the mean of trial rates
    assert (held["events_stim"], held["minutes_stim"]) == (8.0, pytest.approx(0.7))


def test_a_trial_with_an_empty_window_does_not_count_in_the_pull_in_rates():
    ft = hand_built_states()
    ft["stim_min"] = 0.1
    base = F.compute_stats(ft, fly_table(ft), n_perm=N_PERM)["pullins"]["groups"]
    ft.loc[ft.index[2], ["stim_min", "pullin_events_stim"]] = [0.0, 50.0]          # a held trial whose stimulus window is empty
    groups = F.compute_stats(ft, fly_table(ft), n_perm=N_PERM)["pullins"]["groups"]
    assert groups["held"]["n_trials"] == base["held"]["n_trials"] - 1 and groups["held"]["events_stim"] == 6.0
    assert groups["held"]["rate_stim"] == pytest.approx(6.0 / 0.3) and groups["retracted"] == base["retracted"]


def test_an_identity_flag_that_came_back_from_a_csv_as_text_is_still_read_as_a_flag(data):
    ft = data.ft
    ghost = ft[ft["fly_id"] == ft["fly_id"].iloc[0]].copy()
    ghost["fly_id"], ghost["identity_consistent"], ghost["responded"] = "DS-A/batch_0/ghost", False, 1.0
    both = pd.concat([ft, ghost], ignore_index=True)
    as_text = both.assign(identity_consistent=both["identity_consistent"].map({True: "True", False: "False"}))
    assert F.compute_stats(as_text, data.fly, n_perm=N_PERM)["tercile"]["bins"] == data.stats["tercile"]["bins"]
    assert F.compute_stats(as_text, data.fly, n_perm=N_PERM)["n_trials"] == len(ft)


def test_trials_of_a_fly_whose_identity_drifted_never_enter_a_pooled_bar(data):
    ft = data.ft
    ghost = ft[ft["fly_id"] == ft["fly_id"].iloc[0]].copy()
    ghost["fly_id"], ghost["identity_consistent"] = "DS-A/batch_0/ghost", False
    ghost["responded"], ghost["held_subtype"], ghost["spont_per_any"] = 1.0, "held_down", 1.0
    both = pd.concat([ft, ghost], ignore_index=True)
    stats = F.compute_stats(both, fly_table(both), n_perm=N_PERM, seed=0)
    assert stats["tercile"]["bins"] == data.stats["tercile"]["bins"]
    assert stats["baseline"] == data.stats["baseline"] and stats["pullins"] == data.stats["pullins"]
    assert stats["n_trials"] == data.stats["n_trials"] and stats["n_fly_trials"] == len(both)


def counts(stats: dict) -> dict:
    """Trials, flies and responders of every bar the response analyses draw (no NaN, so two of them compare with ==)."""
    bars = [(b["n_trials"], b["n_flies"], b["n_responded"]) for b in stats["tercile"]["bins"]]
    for panel in stats["baseline"].values():
        bars += [(b["state"], b["n_trials"], b["n_flies"], b["n_responded"]) for b in panel["bars"]]
    return bars


def test_only_scored_pretest_and_testing_trials_enter_the_response_bars():
    extra = trial_table([
        row("fly2", trial_type="training", trial_index=0, score=3.0, responded=1.0, held_subtype="held_down",
            spont_per_any=1.0, trial_body_pitch_deg=-2.0),                               # a training row that carries a score
        row("fly2", trial_type="pretest", trial_index=0, score=np.nan, responded=1.0, held_subtype="held_down",
            spont_per_any=1.0, trial_body_pitch_deg=-2.0),                               # a response with no score behind it
        row("fly2", trial_type="testing", trial_index=0, score=3.0, responded=np.nan, held_subtype="held_down",
            spont_per_any=1.0, trial_body_pitch_deg=-2.0)])                              # a score that was never binarised
    for build in (hand_built_terciles, hand_built_states):
        ft = build()
        both = pd.concat([ft, extra], ignore_index=True)
        before = F.compute_stats(ft, fly_table(ft), n_perm=N_PERM)
        after = F.compute_stats(both, fly_table(both), n_perm=N_PERM)
        assert counts(after) == counts(before), build.__name__
    assert any(c[1] for c in counts(after) if len(c) == 4), "the states set has bars"      # so the comparison above has teeth


def test_the_response_figures_and_the_report_say_pretest_plus_testing_trials(figures, report):
    tercile, baseline = figures["response_by_mount_pitch_tercile"], figures["baseline_state_vs_response"]
    assert "pretest + testing" in squash(" ".join(t.get_text() for t in tercile.texts))          # in the subtitle ...
    assert "pretest + testing" in tercile.axes[0].get_xlabel()                                      # ... and on the axis
    assert "pretest + testing" in squash(" ".join(t.get_text() for t in baseline.texts))
    header = report.split("## Exclusions")[0]
    assert "pretest + testing" in header and "trial type" in header and "scored trials" in header


def test_trials_without_a_score_or_a_known_state_are_not_in_the_baseline_bars(data):
    ft, base = data.ft, data.stats["baseline"]
    n_state = int((ft["responded"].notna() & ft["held_subtype"].notna()).sum())
    assert sum(b["n_trials"] for b in base["proboscis"]["bars"]) == n_state < len(ft)
    n_spont = int((ft["responded"].notna() & ft["spont_per_any"].notna()).sum())
    assert sum(b["n_trials"] for b in base["spontaneous"]["bars"]) == n_spont


def test_the_rig_summary_is_the_median_and_quartiles_of_the_raw_fly_pitch(data):
    rigs = {r["rig"]: r for r in data.stats["rigs"]}
    assert sorted(rigs) == ["BehaviorLocust", "Flybehavior2", "Flybehavior3"]
    for rig, g in data.fly.groupby("rig"):
        assert rigs[rig]["n_flies"] == len(g) and rigs[rig]["median_deg"] == pytest.approx(g["mount_pitch_deg"].median())
    assert rigs["Flybehavior2"]["n_flies"] == 16                                  # two datasets share this rig


def test_the_stats_are_repeatable_for_a_seed_and_the_permutation_count_is_recorded(data):
    again = F.compute_stats(data.ft, data.fly, n_perm=N_PERM, seed=0)
    assert again == data.stats and again["n_perm"] == N_PERM and again["seed"] == 0


# ---------------------------------------------------------------------------
# The wording: the question is the title and the answer comes from the data
# ---------------------------------------------------------------------------

def test_the_verdict_says_yes_in_the_direction_of_the_effect_only_below_alpha():
    up, down = "tilted more head-up", "tilted more head-down"
    assert F.verdict(0.03, +0.4, up=up, down=down) == f"Yes — {up}"
    assert F.verdict(0.03, -0.4, up=up, down=down) == f"Yes — {down}"
    assert F.verdict(0.2, +0.4, up=up, down=down) == "No clear link"
    assert F.verdict(0.05, +0.4, up=up, down=down) == "No clear link"                 # strictly below alpha
    assert F.verdict(float("nan"), +0.4, up=up, down=down) == "Not enough data to tell"
    assert F.verdict(0.2, +0.4, up=up, down=down, none="No clear difference") == "No clear difference"


def test_p_values_are_printed_short_on_figures_and_longer_in_the_report():
    assert F.p_text(0.6123) == "p=0.61" and F.p_text(0.0123) == "p=0.012" and F.p_text(0.0004) == "p<0.001"
    assert F.p_text(float("nan")) == "p=n/a"
    assert F.p_text(0.0123, digits=4) == "p=0.0123" and F.p_text(0.00004, digits=4) == "p<0.0001"


def test_a_missing_difference_is_n_a_never_nan_points():
    assert F.points_text(0.0941, 1) == "+9.4 points" and F.points_text(-0.2) == "-20 points"
    assert F.points_text(float("nan")) == "n/a"


def test_the_n_labels_name_trials_and_flies():
    assert F.n_label(120, 14) == "n=120 trials / 14 flies"
    assert F.n_label(1, 1) == "n=1 trial / 1 fly"
    assert F.n_flies_label(48) == "n=48 flies" and F.n_flies_label(1) == "n=1 fly"


def test_wrapping_never_splits_an_n_label_a_p_value_or_a_number_of_points():
    fig = F.new_figure(6.0, 3.0)
    lines = F.wrap_text(fig, "word " * 30 + "n=12 trials / 3 flies and p=0.04 (uncorrected) and +9 points, -12.5 points end",
                        8.5, "normal", 3.0)
    assert len(lines) > 3
    text = "\n".join(lines)
    assert "n=12\u00a0trials\u00a0/\u00a03\u00a0flies" in text and "p=0.04\u00a0(uncorrected)" in text
    assert "+9\u00a0points" in text and "-12.5\u00a0points" in text
    assert squash(text).endswith("n=12 trials / 3 flies and p=0.04 (uncorrected) and +9 points, -12.5 points end")


def test_each_figure_title_is_a_question(figures):
    titles = {name: squash(fig.texts[0].get_text()) for name, fig in figures.items()}
    assert titles["mount_pitch_vs_spontaneous_per"] == "Does mounting angle change spontaneous PER?"
    assert all(t.endswith("?") for t in titles.values()), titles


def test_a_clear_relation_is_answered_yes_and_none_is_answered_no_clear_link(figures, null_data, data):
    assert data.stats["spontaneous"]["p_perm"] < 0.05 and data.stats["spontaneous"]["rho"] > 0
    assert data.stats["held_out"]["p_perm"] > 0.05
    text = all_text(figures["mount_pitch_vs_spontaneous_per"])
    assert "Yes — flies tilted more head-up show more spontaneous PER" in text
    assert "No clear link" in all_text(figures["mount_pitch_vs_held_out"])
    assert "No clear" not in text and "Yes" not in all_text(figures["mount_pitch_vs_held_out"])
    nulls = F.build_figures(null_data.ft, null_data.fly, null_data.stats)
    assert null_data.stats["spontaneous"]["p_perm"] > 0.05
    assert "No clear link" in all_text(nulls["mount_pitch_vs_spontaneous_per"])
    assert "Yes" not in all_text(nulls["mount_pitch_vs_spontaneous_per"])


def test_a_negative_relation_is_answered_in_the_other_direction():
    ft = cohort(seed=7, tilt=-0.30)
    fly = fly_table(ft)
    stats = F.compute_stats(ft, fly, n_perm=N_PERM)
    assert stats["spontaneous"]["rho"] < 0 and stats["spontaneous"]["p_perm"] < 0.05
    figs = F.build_figures(ft, fly, stats)
    assert "Yes — flies tilted more head-down show more spontaneous PER" in all_text(figs["mount_pitch_vs_spontaneous_per"])


def test_the_subtitle_carries_rho_p_the_permutation_p_and_n_flies(figures, data):
    s = data.stats["spontaneous"]
    text = all_text(figures["mount_pitch_vs_spontaneous_per"])
    assert f"ρ={s['rho']:.2f}" in text and f"{F.p_text(s['p'])} (uncorrected)" in text
    assert f"permutation {F.p_text(s['p_perm'])} (uncorrected) within dataset × batch" in text and "n=32 flies" in text
    assert f"secondary {F.p_text(s['p_perm_rig'])} (uncorrected) within dataset × rig" in text


def test_the_tercile_answer_comes_from_the_per_fly_test(figures, data):
    t = data.stats["tercile"]
    assert t["p_perm"] < 0.05 and t["rho"] > 0
    text = all_text(figures["response_by_mount_pitch_tercile"])
    assert "Yes — flies tilted more head-up respond more often" in text
    assert f"{F.p_text(t['p_perm'])} (uncorrected) within dataset × batch" in text
    assert f"secondary {F.p_text(t['p_perm_rig'])} (uncorrected) within dataset × rig" in text


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------

def test_render_figures_writes_the_six_pngs_named_in_the_design(data, tmp_path):
    written = F.render_figures(data.ft, data.fly, tmp_path / "out", n_perm=N_PERM)
    assert tuple(written) == NAMES
    assert {p.name for p in (tmp_path / "out").glob("*.png")} == {f"{n}.png" for n in NAMES}
    for name, path in written.items():
        assert path == tmp_path / "out" / f"{name}.png" and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_pngs_are_300_dpi(data, tmp_path):
    from PIL import Image
    written = F.render_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM)
    for path in written.values():
        with Image.open(path) as image:
            assert image.info["dpi"][0] == pytest.approx(300, abs=0.5)
            assert image.size[0] >= 1500                                              # a 5+ inch figure at 300 dpi


def test_render_figures_takes_precomputed_stats_and_gives_the_same_pictures(data, tmp_path):
    a = F.render_figures(data.ft, data.fly, tmp_path / "a", n_perm=N_PERM, stats=data.stats)
    b = F.render_figures(data.ft, data.fly, tmp_path / "b", n_perm=N_PERM)
    assert all(a[n].read_bytes() == b[n].read_bytes() for n in NAMES)


def test_every_figure_carries_an_n_label(figures):
    assert tuple(figures) == NAMES
    for name, fig in figures.items():
        assert re.search(r"n=\d", all_text(fig)), name


def test_per_fly_figures_say_n_flies_and_pooled_figures_say_n_trials_and_flies(figures, data):
    for name in NAMES[:3]:
        assert re.search(r"n=\d+ flies", all_text(figures[name])), name
    for name in NAMES[3:]:
        assert re.search(r"n=[\d,]+ trials / \d+ flies", all_text(figures[name])), name


def test_the_strip_plot_labels_every_rig_with_its_number_of_flies(figures, data):
    text = all_text(figures["mount_pitch_per_fly"])
    for r in data.stats["rigs"]:
        assert f"{r['rig']} n={r['n_flies']} flies" in text
    assert "n=32 flies" in text


def test_every_tercile_bar_is_labelled_with_its_trials_and_flies(figures, data):
    text = all_text(figures["response_by_mount_pitch_tercile"])
    for b in data.stats["tercile"]["bins"]:
        assert F.n_label(b["n_trials"], b["n_flies"]) in text


def test_the_baseline_panels_label_each_bar_with_its_trials_and_flies(figures, data):
    text = all_text(figures["baseline_state_vs_response"])
    for panel in data.stats["baseline"].values():
        for b in panel["bars"]:
            assert F.n_label(b["n_trials"], b["n_flies"]) in text


def test_the_pull_in_legend_carries_the_trial_and_fly_counts_of_both_states(figures, data):
    text = all_text(figures["pullins_held_vs_retracted"])
    for g in data.stats["pullins"]["groups"].values():
        assert F.n_label(g["n_trials"], g["n_flies"]) in text


def test_every_p_value_on_a_figure_is_labelled_uncorrected(figures):
    for name in NAMES:
        text = all_text(figures[name])
        for match in re.finditer(r"\bp[=<]\s?[0-9.]+", text):
            assert "uncorrected" in text[match.end():match.end() + 40], (name, text[match.start():match.end() + 40])


def test_no_figure_uses_a_twin_or_secondary_axis(figures):
    for name, fig in figures.items():
        assert len(fig.axes) == (2 if name == "baseline_state_vs_response" else 1), name
        for ax in fig.axes:
            assert ax.child_axes == [] and len(ax.get_shared_x_axes().get_siblings(ax)) == 1, name


def test_per_fly_points_are_blue_with_a_surface_ring(figures):
    ax = figures["mount_pitch_vs_spontaneous_per"].axes[0]
    points = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(points) == 1 and len(points[0].get_offsets()) == 32
    assert np.allclose(points[0].get_facecolors(), to_rgba(BLUE), atol=0.01)
    assert np.allclose(points[0].get_edgecolors(), to_rgba("#ffffff"), atol=0.01)
    strip = [c for c in figures["mount_pitch_per_fly"].axes[0].collections if isinstance(c, PathCollection)]
    assert sum(len(c.get_offsets()) for c in strip) == 32
    assert all(np.allclose(c.get_facecolors(), to_rgba(BLUE), atol=0.01) for c in strip)


def test_marker_fill_shows_the_dominant_held_subtype(figures, data):
    ax = figures["mount_pitch_vs_held_out"].axes[0]
    by_label = {c.get_label(): c for c in ax.collections if isinstance(c, PathCollection)}
    fly = data.fly
    counts = fly["held_subtype_dominant"].value_counts()
    assert len(by_label["mostly held back"].get_offsets()) == counts["held_back"] > 0
    assert len(by_label["mostly held down"].get_offsets()) == counts["held_down"] > 0
    back, down = by_label["mostly held back"], by_label["mostly held down"]
    assert np.allclose(back.get_facecolors(), to_rgba(BLUE), atol=0.01)              # filled = held back
    assert np.allclose(down.get_facecolors(), to_rgba("#ffffff"), atol=0.01)         # hollow = held down
    assert np.allclose(down.get_edgecolors(), to_rgba(BLUE), atol=0.01)
    assert sum(len(c.get_offsets()) for c in by_label.values()) == 32
    texts = all_text(figures["mount_pitch_vs_held_out"])
    assert "mostly held back" in texts and "mostly held down" in texts


def test_held_and_spontaneous_bars_are_orange_with_the_slash_hatch_and_the_rest_blue(figures):
    ax_state, ax_spont = figures["baseline_state_vs_response"].axes
    for ax, n_hatched in ((ax_state, 2), (ax_spont, 1)):
        bars = [p for p in ax.patches if p.get_height() > 0 or p.get_hatch()]
        hatched = [p for p in bars if p.get_hatch()]
        assert len(hatched) == n_hatched and all(p.get_hatch() == "//" for p in hatched)
        assert all(to_hex(p.get_facecolor()) == ORANGE for p in hatched)
        assert all(to_hex(p.get_facecolor()) == BLUE for p in bars if not p.get_hatch())
    ax = figures["pullins_held_vs_retracted"].axes[0]
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == 2 and all(to_hex(p.get_facecolor()) == ORANGE and p.get_hatch() == "//" for p in hatched)


def test_the_tercile_bars_are_blue_and_plain(figures):
    bars = figures["response_by_mount_pitch_tercile"].axes[0].patches
    assert len(bars) == 3 and all(to_hex(p.get_facecolor()) == BLUE and not p.get_hatch() for p in bars)


def test_bars_are_thin(figures):
    for name in NAMES[3:]:
        fig = figures[name]
        fig.canvas.draw()
        for ax in fig.axes:
            for bar in ax.patches:
                assert bar.get_window_extent().width / fig.dpi * 72 <= 18.5, name        # 24 CSS px = 18 pt


def points(ax) -> list:
    """The (x, y) of every scatter marker of an axes, rounded and sorted (the order of the collections is not the table's)."""
    xy = np.vstack([c.get_offsets() for c in ax.collections if isinstance(c, PathCollection)])
    return sorted(map(tuple, np.round(xy, 6).tolist()))


def test_the_per_fly_scatters_plot_the_fly_table_values(figures, data):
    fly = data.fly
    for name, column in (("mount_pitch_vs_spontaneous_per", "spont_per_frac"), ("mount_pitch_vs_held_out", "held_out_frac")):
        expected = np.column_stack([fly["mount_pitch_centred_deg"], 100 * fly[column]])
        assert points(figures[name].axes[0]) == sorted(map(tuple, np.round(expected, 6).tolist())), name


def test_the_strip_plot_shows_the_raw_pitch_of_every_fly(figures, data):
    ys = [y for _, y in points(figures["mount_pitch_per_fly"].axes[0])]
    assert sorted(ys) == sorted(np.round(data.fly["mount_pitch_deg"], 6).tolist())


def bar_heights(ax) -> list:
    return [p.get_height() for p in sorted(ax.patches, key=lambda p: p.get_x()) if p.get_height() > 0]


def test_the_bars_are_drawn_at_the_pooled_rates_of_the_stats(figures, data):
    s = data.stats
    assert bar_heights(figures["response_by_mount_pitch_tercile"].axes[0]) == pytest.approx(
        [100 * b["rate"] for b in s["tercile"]["bins"]])
    for ax, key in zip(figures["baseline_state_vs_response"].axes, ("proboscis", "spontaneous")):
        assert bar_heights(ax) == pytest.approx([100 * b["rate"] for b in s["baseline"][key]["bars"] if b["n_trials"]])
    g = s["pullins"]["groups"]
    assert bar_heights(figures["pullins_held_vs_retracted"].axes[0]) == pytest.approx(
        [g["retracted"]["rate_pre"], g["held"]["rate_pre"], g["retracted"]["rate_stim"], g["held"]["rate_stim"]])


def test_a_bar_value_is_written_at_its_tip(figures, data):
    ax = figures["response_by_mount_pitch_tercile"].axes[0]
    labels = sorted(t.get_text() for t in ax.texts if re.fullmatch(r"\d+%", t.get_text()))
    assert labels == sorted(f"{b['rate'] * 100:.0f}%" for b in data.stats["tercile"]["bins"])


def test_no_text_overflows_the_figure_or_overlaps_other_text(figures):
    for name, fig in figures.items():
        width, height = fig.get_size_inches() * fig.dpi
        boxes = drawn_texts(fig)
        for text, box in boxes:
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width + 1 and box.y1 <= height + 1, (name, text)
        for i, (a, box_a) in enumerate(boxes):
            for b, box_b in boxes[i + 1:]:
                inside = 3                                                        # px of slack at 300 dpi
                clash = (min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0) > inside
                         and min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0) > inside)
                assert not clash, (name, a, b)


def test_rendering_leaves_the_global_matplotlib_style_alone(data, tmp_path):
    sentinel = {"font.family": ["monospace"], "axes.spines.top": True, "savefig.dpi": 77.0, "figure.dpi": 66.0,
                "axes.grid": True, "axes.unicode_minus": True, "hatch.linewidth": 3.0}      # nothing like the lab style
    with matplotlib.rc_context(sentinel):
        F.render_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
        F.build_figures(data.ft, data.fly, data.stats)
        assert {k: matplotlib.rcParams[k] for k in sentinel} == sentinel


def test_no_figure_defines_a_score_colour_ramp(figures):
    palette = {BLUE, ORANGE, F.SURFACE, F.INK, F.INK_SECONDARY, F.MUTED, F.GRID, F.AXIS}
    for name, fig in figures.items():
        for ax in fig.axes:
            assert not ax.images, name
            for artist in [*ax.collections, *ax.patches]:
                assert getattr(artist, "get_array", lambda: None)() is None, name        # nothing is colour-mapped
                colours = np.atleast_2d(artist.get_facecolor()).tolist() + np.atleast_2d(artist.get_edgecolor()).tolist()
                assert {to_hex(c) for c in colours if len(c) and c[3] > 0} <= palette, (name, artist)       # ignoring 'none'
            assert {to_hex(line.get_color()) for line in ax.lines} <= palette, name


def test_a_figure_without_enough_flies_is_a_placeholder_that_says_why_and_keeps_its_n_label():
    ft = cohort(seed=3, per_cell=1)
    ft = ft[ft["fly_id"].isin(ft["fly_id"].unique()[:2])]
    fly = fly_table(ft)
    stats = F.compute_stats(ft, fly, n_perm=N_PERM)
    assert set(stats["placeholders"]) >= {"mount_pitch_vs_spontaneous_per", "mount_pitch_vs_held_out",
                                          "response_by_mount_pitch_tercile"}
    figs = F.build_figures(ft, fly, stats)
    assert tuple(figs) == NAMES
    text = all_text(figs["mount_pitch_vs_spontaneous_per"])
    assert "Not enough data" in text and "n=2 flies" in text and "at least 3" in text
    assert figs["mount_pitch_vs_spontaneous_per"].axes == []                         # no axes, so nothing plotted


def test_a_trait_every_fly_shares_is_answered_as_no_variation_not_as_no_link():
    ft = cohort(seed=3, per_cell=4)
    ft[["spont_per_any", "spont_per_events"]] = 0.0
    fly = fly_table(ft)
    s = F.compute_stats(ft, fly, n_perm=N_PERM)["spontaneous"]
    assert np.isnan(s["rho"]) and s["n_flies"] == 16
    assert s["verdict"] == "Not enough data to tell (every fly has the same value of one of the two)"
    text = all_text(F.build_figures(ft, fly, F.compute_stats(ft, fly, n_perm=N_PERM))["mount_pitch_vs_spontaneous_per"])
    assert "Not enough data to tell" in text and "ρ=n/a" in text and "p=n/a (uncorrected)" in text


def test_a_cohort_whose_flies_all_drifted_gets_placeholders_and_a_report(tmp_path):
    ft = cohort(seed=3, per_cell=3)
    ft["identity_consistent"] = False
    fly = fly_table(ft)
    assert fly.empty
    stats = F.compute_stats(ft, fly, n_perm=N_PERM)
    assert set(stats["placeholders"]) == set(NAMES) and stats["n_flies"] == 0 and stats["n_trials"] == 0
    written = F.render_figures(ft, fly, tmp_path, n_perm=N_PERM, stats=stats)
    assert tuple(written) == NAMES
    assert "Placeholder, not drawn" in F.write_report(ft, fly, EXCLUSIONS, stats, tmp_path).read_text(encoding="utf-8")


def test_a_cohort_with_only_unknown_states_still_renders_every_figure(tmp_path):
    ft = cohort(seed=5, per_cell=3)
    ft[["held_out", "held_subtype", "spont_per_any", "spont_per_events", "responded", "score", "pullin_events_pre",
        "pullin_events_stim"]] = np.nan
    fly = fly_table(ft)
    written = F.render_figures(ft, fly, tmp_path, n_perm=N_PERM)
    assert tuple(written) == NAMES and all(p.is_file() for p in written.values())


def test_a_single_rig_single_fly_cohort_does_not_crash_the_figures(tmp_path):
    ft = trial_table([row("fly1", trial_body_pitch_deg=4.0, responded=1.0, held_subtype="retracted", spont_per_any=0.0,
                          pullin_events_pre=0.0, pullin_events_stim=1.0, pre_min=0.5, stim_min=0.1, held_out=0.0)])
    written = F.render_figures(ft, fly_table(ft), tmp_path, n_perm=N_PERM)
    assert tuple(written) == NAMES


def test_one_failing_figure_is_logged_with_its_traceback_and_does_not_stop_the_others(data, tmp_path, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setitem(F.BUILDERS, "mount_pitch_vs_held_out", boom)
    with caplog.at_level(logging.WARNING):
        written = F.render_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM)
    assert tuple(written) == tuple(n for n in NAMES if n != "mount_pitch_vs_held_out")
    assert "mount_pitch_vs_held_out" in caplog.text and "RuntimeError: synthetic failure" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text                       # where it came from, not just what it said
    assert not (tmp_path / "mount_pitch_vs_held_out.png").exists()


def test_empty_input_draws_no_figure_and_creates_nothing(tmp_path, caplog):
    ft, fly = trial_table([]), pd.DataFrame(columns=MP.FLY_COLUMNS)
    with caplog.at_level(logging.WARNING):
        assert F.render_figures(ft, fly, tmp_path / "out", n_perm=N_PERM) == {}
    assert not (tmp_path / "out").exists() and caplog.text == ""


def test_with_nothing_in_scope_the_pictures_of_an_earlier_run_are_removed(data, tmp_path):
    F.render_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    (tmp_path / "notes.png").write_bytes(b"mine")
    (tmp_path / "report.md").write_text("old", encoding="utf-8")
    ft, fly = trial_table([]), pd.DataFrame(columns=MP.FLY_COLUMNS)
    assert F.render_figures(ft, fly, tmp_path, n_perm=N_PERM) == {}
    assert [p.name for p in tmp_path.glob("*.png")] == ["notes.png"]                 # the six are gone, nothing else is touched
    assert (tmp_path / "notes.png").read_bytes() == b"mine" and (tmp_path / "report.md").read_text(encoding="utf-8") == "old"


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

EXCLUSIONS = {
    "n_batches": 4, "n_trials": 40, "n_fly_trials": 96, "eye_mismatch": 3, "non_square_frame": 5, "warmup_slots": 4,
    "no_production_table": 2, "unreadable_geometry": 1, "flagged_flies": 2, "flagged_fly_trials": 11,
    "identity_drift": [{"fly_id": "DS-A/batch_1/fly3", "dataset": "DS-A", "batch": "batch_1", "fly_number": 3,
                        "spread_px": 61.2, "n_trials": 8},
                       {"fly_id": "DS-B/batch_2/fly1", "dataset": "DS-B", "batch": "batch_2", "fly_number": 1,
                        "spread_px": 140.5, "n_trials": 5}],
    "prediction_exact_duplicates": 9, "prediction_conflicts": 2, "prediction_same_score_duplicates": 1,
    "prediction_key_conflicts": 1, "predictions_csv": "/x/model_predictions.csv", "predictions_missing": False,
    "pending_trials": 7}


@pytest.fixture(scope="module")
def report(data, tmp_path_factory):
    path = F.write_report(data.ft, data.fly, EXCLUSIONS, data.stats, tmp_path_factory.mktemp("report"))
    return path.read_text(encoding="utf-8")


# ---- run information: what produced the report ---------------------------------------------------------------

WARNING_TEXT = ("Flybehavior3: 3 of 4 identity-consistent flies have their lowest trial rest at or above the retracted "
                "reference plus held_margin_eu (0.30 + 0.25 = 0.55 eu); the reference may sit at the held-out level, and "
                "held-out flies would be labelled retracted")
RUN_INFO = {
    "generated_utc": "2026-10-14T09:30:05Z",
    "settings": {**{k: v for k, v in dataclasses.asdict(MCFG).items()}, "batch_date_from": "2026-10-06",
                 "csv_out_dir": "/x/CSVs/Mounting-Posture", "retracted_ref_eu": None},
    "retracted_ref": {
        "Flybehavior2": {"retracted_ref_eu": 0.52, "n_flies": 9, "from_config": False, "n_flies_at_held_level": 1,
                         "warning": False},
        "Flybehavior3": {"retracted_ref_eu": 0.3, "n_flies": 4, "from_config": True, "n_flies_at_held_level": 3,
                         "warning": True},
        "unknown": {"retracted_ref_eu": None, "n_flies": 0, "from_config": False, "n_flies_at_held_level": 0,
                    "warning": False}},
    "warnings": [WARNING_TEXT]}


@pytest.fixture(scope="module")
def run_report(data, tmp_path_factory):
    exclusions = {**EXCLUSIONS, "run_info": RUN_INFO}
    return F.write_report(data.ft, data.fly, exclusions, data.stats, tmp_path_factory.mktemp("run_report")).read_text(
        encoding="utf-8")


def run_section(text: str) -> str:
    return text.split("## Run information")[1].split("\n## ")[0]


def test_the_report_has_a_run_information_section_between_the_header_and_the_exclusions(run_report):
    assert run_report.index("# Mounting posture report") < run_report.index("## Run information") \
        < run_report.index("## Exclusions") < run_report.index("## Mounting")


def test_the_run_information_gives_the_generation_time_in_utc(run_report):
    assert "2026-10-14T09:30:05Z" in run_section(run_report) and "UTC" in run_section(run_report)


def test_the_run_information_lists_every_mounting_setting_with_its_value(run_report):
    section = run_section(run_report)
    for f in dataclasses.fields(M.MountingSettings):
        assert f"`{f.name}`" in section, f.name
    for needle in ("| `rest_quantile` | 0.1 |", "| `held_margin_eu` | 0.25 |", "| `n_permutations` | 10000 |",
                   "| `enabled` | true |", "| `retracted_ref_eu` | null |", "| `batch_date_from` | 2026-10-06 |",
                   "| `csv_out_dir` | /x/CSVs/Mounting-Posture |"):
        assert needle in section, needle


def test_the_run_information_gives_each_rigs_reference_its_source_and_the_flies_behind_it(run_report):
    lines = {rig: next(ln for ln in run_section(run_report).splitlines() if ln.startswith(f"| {rig} "))
             for rig in ("Flybehavior2", "Flybehavior3", "unknown")}
    assert "0.52" in lines["Flybehavior2"] and "derived" in lines["Flybehavior2"] and "| 9 |" in lines["Flybehavior2"]
    assert "0.30" in lines["Flybehavior3"] and "configured" in lines["Flybehavior3"] and "| 4 |" in lines["Flybehavior3"]
    assert "| 3 |" in lines["Flybehavior3"], "the flies at the held-out level"
    assert "n/a" in lines["unknown"] and "| 0 |" in lines["unknown"]


def test_a_reference_warning_is_a_warning_line_in_the_report_before_the_numbers(run_report):
    assert f"**WARNING:** {WARNING_TEXT}" in run_report
    assert run_report.index("**WARNING:**") < run_report.index("## Exclusions")
    assert run_report.count("**WARNING:**") == 1


def test_a_report_without_a_reference_warning_has_no_warning_line(data, tmp_path):
    quiet = {**RUN_INFO, "warnings": [], "retracted_ref": {k: {**v, "warning": False} for k, v in RUN_INFO["retracted_ref"].items()}}
    text = F.write_report(data.ft, data.fly, {**EXCLUSIONS, "run_info": quiet}, data.stats, tmp_path).read_text(encoding="utf-8")
    assert "WARNING" not in text and "## Run information" in text


SCORING = {"pretest": {"scoreable": 120, "scored": 118, "scored_identity_consistent": 104},
           "testing": {"scoreable": 240, "scored": 231, "scored_identity_consistent": 205}}


def test_the_report_gives_the_scoreable_and_the_scored_fly_trials_per_trial_type(data, tmp_path):
    text = F.write_report(data.ft, data.fly, {**EXCLUSIONS, "scoring": SCORING}, data.stats, tmp_path).read_text(
        encoding="utf-8")
    exclusions = text.split("## Exclusions")[1].split("\n## ")[0]
    assert "| Trial type | Scoreable fly-trials | Scored fly-trials | Scored, identity-consistent flies |" in exclusions
    assert "| pretest | 120 | 118 | 104 |" in exclusions and "| testing | 240 | 231 | 205 |" in exclusions
    assert "training trials get none" in exclusions
    assert "(`scoring`) |" not in text, "a table of its own, not a counter row"


def test_a_report_without_scoring_counts_has_no_scoring_table(report):
    assert "Scoreable fly-trials" not in report


def test_the_run_info_is_not_one_of_the_exclusion_counters(run_report):
    exclusions = run_report.split("## Exclusions")[1].split("\n## ")[0]
    assert "run_info" not in exclusions and "(`run_info`) |" not in run_report
    for key in EXCLUSIONS:
        assert f"`{key}`" in exclusions, key


def test_a_report_of_exclusions_without_run_info_has_no_run_information_section(report):
    assert "## Run information" not in report


def test_the_run_information_is_there_when_nothing_is_in_scope(tmp_path):
    ft, fly = trial_table([]), pd.DataFrame(columns=MP.FLY_COLUMNS)
    text = F.write_report(ft, fly, {**EXCLUSIONS, "run_info": RUN_INFO}, F.compute_stats(ft, fly, n_perm=N_PERM),
                          tmp_path).read_text(encoding="utf-8")
    assert "no new-glue fly-trials in scope" in text and "## Run information" in text and "2026-10-14T09:30:05Z" in text


def test_the_sections_of_a_report_with_run_info_are_header_run_information_exclusions_mounting(data):
    sections = F.report_sections(data.ft, data.fly, {**EXCLUSIONS, "run_info": RUN_INFO}, data.stats)
    assert [s.splitlines()[0] for s in sections] == ["# Mounting posture report", "## Run information", "## Exclusions",
                                                     "## Mounting"]


def test_the_report_is_written_as_report_md_in_the_figure_directory(data, tmp_path):
    path = F.write_report(data.ft, data.fly, EXCLUSIONS, data.stats, tmp_path / "new")
    assert path == tmp_path / "new" / "report.md" and path.is_file()


def test_the_report_lists_every_exclusions_key_with_its_count(report):
    for key in EXCLUSIONS:
        assert f"`{key}`" in report, key
    for needle in ("Eye mismatch", "Identity drift", "Conflicting scores", "Warm-up slots", "Non-square frames",
                   "Flagged flies", "Pending trials"):
        assert needle in report, needle
    for line in (r"`eye_mismatch`\W+3\b", r"`warmup_slots`\W+4\b", r"`non_square_frame`\W+5\b", r"`flagged_flies`\W+2\b",
                 r"`prediction_conflicts`\W+2\b", r"`prediction_key_conflicts`\W+1\b", r"`pending_trials`\W+7\b"):
        assert re.search(line, report), line


def test_the_report_names_the_flies_with_identity_drift(report):
    assert "DS-A/batch_1/fly3" in report and "DS-B/batch_2/fly1" in report
    assert "61.2" in report and "140.5" in report


def test_an_exclusions_key_the_report_does_not_know_is_still_listed(data, tmp_path):
    path = F.write_report(data.ft, data.fly, {**EXCLUSIONS, "made_up_counter": 12}, data.stats, tmp_path)
    assert re.search(r"`made_up_counter`\W+12\b", path.read_text(encoding="utf-8"))


def test_a_pending_trial_walk_that_failed_is_reported_as_such(data, tmp_path):
    exc = {**EXCLUSIONS, "pending_trials": None, "pending_trials_error": "OSError: no such directory"}
    text = F.write_report(data.ft, data.fly, exc, data.stats, tmp_path).read_text(encoding="utf-8")
    assert "could not be counted" in text and "OSError: no such directory" in text


def test_the_report_gives_rho_and_p_for_every_per_fly_scatter(report, data):
    for key in ("spontaneous", "held_out", "tercile"):
        s = data.stats[key]
        assert f"ρ={s['rho']:.3f}" in report, key
        assert F.p_text(s["p"], digits=4).replace("p=", "") in report, key


def test_the_report_gives_every_permutation_p_and_what_was_shuffled(report, data):
    s = data.stats
    for p in (s["spontaneous"]["p_perm"], s["held_out"]["p_perm"], s["tercile"]["p_perm"],
              s["baseline"]["proboscis"]["p_perm"], s["baseline"]["spontaneous"]["p_perm"]):
        assert F.p_text(p, digits=4) in report
    assert "within-fly" in report.lower()
    assert re.search(rf"Stratified permutation p: .*dataset × batch.*{N_PERM} permutations, seed 0", report)
    assert re.search(rf"Within-fly permutation p: .*{N_PERM} permutations, seed 0", report)
    for key in ("spontaneous", "held_out", "tercile"):                     # the secondary dataset x rig p of each scatter
        assert F.p_text(s[key]["p_perm_rig"], digits=4) in report, key
    assert report.count("within dataset × batch strata") >= 3 and report.count("within dataset × rig strata") >= 3


def test_the_header_says_why_batch_is_the_stratum_and_the_dataset_rig_p_only_secondary(report):
    header = report.split("## Exclusions")[0]
    assert "shuffled among flies of the same dataset × batch" in header
    assert "Flies of one batch share the mounting day, the glue session and the starvation" in header
    assert "only flies of the same batch are interchangeable" in header
    assert "decides every \"Yes\"" in header and "dataset × rig is printed beside it as a secondary number" in header


def test_the_header_states_what_the_within_fly_permutation_ignores(report):
    header = report.split("## Exclusions")[0]
    assert "shuffles the state among a fly's trials of one trial type" in header
    assert "ignores which odor a trial had and where it sat in the session" in header
    assert "Odor order is fixed per protocol" in header


def test_the_report_says_uncorrected_next_to_the_p_values_and_in_the_header(report):
    assert "uncorrected" in report.split("## Exclusions")[0]
    assert report.count("uncorrected") >= 8


def test_the_report_puts_the_figure_answers_beside_the_numbers(report, data):
    assert data.stats["spontaneous"]["verdict"] in report and data.stats["tercile"]["verdict"] in report
    for name in NAMES:
        assert f"{name}.png" in report


def test_the_report_gives_the_pooled_counts_behind_every_bar(report, data):
    for b in data.stats["tercile"]["bins"]:
        assert F.n_label(b["n_trials"], b["n_flies"]) in report
    for g in data.stats["pullins"]["groups"].values():
        assert F.n_label(g["n_trials"], g["n_flies"]) in report


def test_a_figure_that_fails_to_build_leaves_no_stale_picture_and_the_report_names_it(data, tmp_path, monkeypatch):
    first = F.render_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    stale = tmp_path / "mount_pitch_vs_held_out.png"
    assert tuple(first) == NAMES and stale.is_file()                                # last run's picture is on disk
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setitem(F.BUILDERS, "mount_pitch_vs_held_out", boom)
    again = F.render_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    assert tuple(again) == tuple(n for n in NAMES if n != "mount_pitch_vs_held_out")
    assert not stale.exists()                                                       # ... and is gone, not vouched for
    assert sorted(p.stem for p in tmp_path.glob("*.png")) == sorted(again)
    text = F.write_report(data.ft, data.fly, EXCLUSIONS, data.stats, tmp_path, figures=again).read_text(encoding="utf-8")
    assert "Figures missing from this folder:** `mount_pitch_vs_held_out.png` (" in text
    monkeypatch.undo()
    fixed = F.render_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    assert tuple(fixed) == NAMES and stale.is_file()
    text = F.write_report(data.ft, data.fly, EXCLUSIONS, data.stats, tmp_path, figures=fixed).read_text(encoding="utf-8")
    assert "Figures missing" not in text


def test_a_report_written_without_the_list_of_figures_makes_no_claim_about_them(data, tmp_path):
    text = F.write_report(data.ft, data.fly, EXCLUSIONS, data.stats, tmp_path).read_text(encoding="utf-8")
    assert "Figures missing" not in text


def test_rendering_removes_only_the_six_figures_of_a_previous_run(data, tmp_path):
    (tmp_path / "notes.png").write_bytes(b"mine")
    (tmp_path / "report.md").write_text("old", encoding="utf-8")
    F.render_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    assert (tmp_path / "notes.png").read_bytes() == b"mine" and (tmp_path / "report.md").read_text(encoding="utf-8") == "old"


def test_the_report_has_a_mounting_section_that_comes_after_the_exclusions(report):
    assert report.index("## Exclusions") < report.index("## Mounting")


def test_further_sections_are_appended_after_the_mounting_section(data, tmp_path):
    extra = "## Abdomen movement\n\nsomething for task 8B (p=0.5, uncorrected)\n"
    path = F.write_report(data.ft, data.fly, EXCLUSIONS, data.stats, tmp_path, extra_sections=[extra])
    text = path.read_text(encoding="utf-8")
    assert text.index("## Mounting") < text.index("## Abdomen movement") and "something for task 8B" in text
    sections = F.report_sections(data.ft, data.fly, EXCLUSIONS, data.stats)
    assert [s.splitlines()[0] for s in sections] == ["# Mounting posture report", "## Exclusions", "## Mounting"]


def test_the_report_marks_a_placeholder_figure_and_why(tmp_path):
    ft = cohort(seed=3, per_cell=1)
    ft = ft[ft["fly_id"].isin(ft["fly_id"].unique()[:2])]
    fly = fly_table(ft)
    stats = F.compute_stats(ft, fly, n_perm=N_PERM)
    text = F.write_report(ft, fly, EXCLUSIONS, stats, tmp_path).read_text(encoding="utf-8")
    assert "not enough data" in text.lower() and "at least 3" in text


def test_the_report_of_an_empty_input_says_nothing_is_in_scope_and_still_lists_the_exclusions(tmp_path):
    ft, fly = trial_table([]), pd.DataFrame(columns=MP.FLY_COLUMNS)
    path = F.write_report(ft, fly, EXCLUSIONS, F.compute_stats(ft, fly, n_perm=N_PERM), tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "no new-glue fly-trials in scope" in text
    assert "`eye_mismatch`" in text and "uncorrected" in text
    assert "## Mounting" not in text


def test_computing_the_stats_of_an_empty_input_does_not_raise():
    stats = F.compute_stats(trial_table([]), pd.DataFrame(columns=MP.FLY_COLUMNS), n_perm=N_PERM)
    assert stats["n_fly_trials"] == 0 and stats["n_flies"] == 0
