"""The Abdomen-Movement figure set and the "Abdomen movement" section of ``report.md``
(``scripts/analysis/abdomen_movement_figures.py``).

Rules pinned here, on synthetic fly-trial / fly tables (the fly table is always built from the trial table):

* the six PNGs of the Design are written at 300 dpi, and every figure carries an ``n=`` label (``n=… flies`` for per-fly
  plots, ``n=… trials / … flies`` for the pooled-trial bars);
* the fly is the unit of inference: the per-fly scatters use the FLY table, the rig-centred mounting angle (the tip pitch
  is rig-centred too: camera roll shifts both) and a permutation p stratified by dataset x rig; the coupling strip is
  descriptive;
* displayed event rates pool trials (events over minutes of the SAME trials, odor and light windows apart, never a mean of
  per-trial rates); the stimulus comparison is a placebo design: the last L s before the stimulus against the first L s
  of it, both scored against ONE resting bend, the median of the early baseline windows of the fly's OTHER trials (a
  baseline fitted to the pre-stimulus window itself, or taken from the same trial next to the placebo window, makes the
  placebo quieter than the start of the stimulus and reads as "significant" when a few hundred trials are pooled), and its
  p is the Monte-Carlo window-label swap permutation (the two windows of a trial are as long as each other; it assumes the
  bend is stationary within a trial; the randomisation is by trial, not by fly);
* only quality-ok windows of identity-consistent flies enter a number, and the windows the ``abd_min_tracked_frac`` gate
  dropped are counted in the report;
* a comparison with fewer than 3 flies to compare is "Not enough data to tell", every p is "uncorrected", and no text says
  "nan" or "1 flies";
* titles are the question and the answer is computed from the data; blue per-fly points and pre-stimulus bars, orange
  stimulus-window bars, ``//`` on the light ones, no twin axes, the two-panel figures have two axes;
* a figure that fails to build leaves no stale picture and is named in the report; an input with no usable abdomen window
  (or none at all) draws nothing, removes the pictures of an earlier run and says so; a failure of the whole abdomen
  analysis costs neither the other figures nor the mounting report.
"""
from __future__ import annotations

import logging
import math
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
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.analysis import abdomen_movement_figures as AM  # noqa: E402
from scripts.analysis import mounting_posture as MP  # noqa: E402
from scripts.analysis import mounting_posture_figures as FIG  # noqa: E402
from fbpipe.analysis import mounting as M  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
MCFG = M.parse_mounting_block({})
NAMES = ("abdomen_rest_bend_per_fly", "mount_pitch_vs_abdomen_rest_bend", "abdomen_curl_events_by_window",
         "abdomen_vs_spontaneous_per", "per_abdomen_coupling", "mount_pitch_vs_abdomen_activity")
TWO_PANEL = {"mount_pitch_vs_abdomen_rest_bend", "abdomen_curl_events_by_window", "mount_pitch_vs_abdomen_activity"}
N_PERM = 300
PLACEBO_STREAM = 11                       # the second random stream of ``abd_cohort`` (the placebo-design columns)

# ---------------------------------------------------------------------------
# Synthetic tables
# ---------------------------------------------------------------------------

CELLS = (("DS-A", "BehaviorLocust"), ("DS-A", "Flybehavior2"), ("DS-B", "Flybehavior2"), ("DS-B", "Flybehavior3"))
RIG_SHIFT = {"BehaviorLocust": 2.0, "Flybehavior2": -4.0, "Flybehavior3": 12.0}       # camera roll differs per rig
TIP_SHIFT = {"BehaviorLocust": 5.0, "Flybehavior2": -9.0, "Flybehavior3": 20.0}      # ... and shifts the tip pitch too
COUPLING = {"retracted": 0.05, "held_down": 0.30, "held_back": -0.20}
ABDOMEN_COLUMNS = ([c for c in MP.ABDOMEN if c not in ("abd_quality_ok", "abd_quality_ok_stim")]
                   + list(MP.PLACEBO))                                  # every abdomen number a trial row carries


def trial_table(rows) -> pd.DataFrame:
    """A fly-trial table with every documented column; what a row does not set is NaN (identity consistent by default)."""
    df = pd.DataFrame(rows)
    for column in MP.FLY_TRIAL_COLUMNS:
        if column not in df:
            df[column] = np.nan
    for flag in ("identity_consistent", "abd_quality_ok", "abd_quality_ok_stim"):
        df[flag] = df[flag].fillna(flag == "identity_consistent").astype(bool)
    return df.reindex(columns=MP.FLY_TRIAL_COLUMNS)


def fly_table(ft) -> pd.DataFrame:
    return MP.build_fly_table(ft, MCFG)


def row(fly_id, rig="R", dataset="D", **cols):
    batch, number = "b1", int(fly_id.split("fly")[-1])
    return {"dataset": dataset, "batch": batch, "fly_number": number, "fly_id": f"{dataset}/{batch}/{fly_id}", "rig": rig,
            "trial_type": "testing", "stim_kind": "odor", "identity_consistent": True, "trial_body_pitch_deg": 0.0, **cols}


def windows(fly, placebo=0, during=0, *, pb=0.25, straighten=(0, 0), kind="odor", base=True, placebo_ok=True,
            during_ok=True, **cols):
    """One trial of a hand-built table: curl events (and straightenings) in the placebo window and in the during window of
    ``pb`` minutes each, against a resting bend from the fly's other trials that is there (``base``) or not; a window that is
    not usable has NaN counts, and so is every count without a resting bend. Any column can be overridden by keyword (a
    keyword wins over what the flags above set)."""
    in_placebo, in_during = base and placebo_ok, base and during_ok
    nan = np.nan
    return row(fly, **{
        "stim_kind": kind, "pb_window_min": pb, "abd_quality_ok": True, "abd_tracked_frac_base": 0.9 if base else 0.2,
        "abd_tracked_frac_placebo": 0.9 if placebo_ok else 0.2, "abd_tracked_frac_during": 0.9 if during_ok else 0.2,
        "abd_base_ok": base, "abd_rest_bend_loo_deg": 3.0 if base else nan,
        "abd_curl_events_placebo": placebo if in_placebo else nan, "abd_curl_events_during": during if in_during else nan,
        "abd_straighten_events_placebo": straighten[0] if in_placebo else nan,
        "abd_straighten_events_during": straighten[1] if in_during else nan, **cols})


def abd_cohort(seed=1, per_cell=8, *, bend=0.0, tip=0.0, activity=0.0, restless=0.0, per_link=None, boost=0.0,
               flaws=True) -> pd.DataFrame:
    """Flies with unbalanced trial counts and the abdomen columns filled in.

    ``bend`` / ``tip`` / ``activity`` make the resting bend, the (rig-shifted) tip pitch and the bend SD / tip moving
    fraction rise with how head-up the fly is mounted relative to its rig, ``restless`` makes the spontaneous curl rate
    rise with it (and the PER rate too, unless ``per_link`` gives the PER rate its own slope), ``boost`` multiplies the curl
    rate during the stimulus (0 = no effect); ``flaws`` switches on the poorly tracked windows (10 % of pre-windows, 8 % of
    the remaining stimulus windows; 8 % of the trials have no resting bend, 6 % of the placebo and of the during windows
    are poorly tracked).

    The placebo-design columns (the last L s before the stimulus, the first L s of it, L = the shorter of the stimulus and
    half the pre-stimulus window) are drawn from a stream of their own, so every other column is the same for a seed
    whatever they hold.
    """
    rng = np.random.default_rng(seed)
    pb_rng = np.random.default_rng([seed, PLACEBO_STREAM])
    rows = []
    for cell, (dataset, rig) in enumerate(CELLS):
        for k in range(per_cell):
            dev = rng.normal(0.0, 4.0)
            z = dev / 4.0
            state = str(rng.choice(["retracted", "held_down", "held_back"], p=[0.5, 0.3, 0.2]))
            curl_rate = 1.0 * np.exp(restless * z)
            per_rate = 0.8 * np.exp((restless if per_link is None else per_link) * z)
            rest_bend = 6.0 + bend * dev + rng.normal(0.0, 1.0)
            tip_pitch = TIP_SHIFT[rig] + tip * dev + rng.normal(0.0, 1.0)
            bend_sd = max(0.5, 3.0 + activity * z + rng.normal(0.0, 0.3))
            moving = float(np.clip(0.15 + 0.05 * activity * z + rng.normal(0.0, 0.03), 0.01, 0.9))
            for t in range(int(rng.integers(6, 14))):
                kind = "odor+light" if 2 <= t < 4 else "light" if t >= 4 and t % 4 == 3 else "odor"
                pre_min, stim_min = 0.5, (1 / 15 if kind == "odor" else 0.5)
                pre_ok = (not flaws) or rng.random() > 0.10
                stim_ok = pre_ok and ((not flaws) or rng.random() > 0.08)
                known = rng.random() > 0.05
                spont = rng.poisson(per_rate * pre_min)
                nan = np.nan
                length = min(stim_min, pre_min / 2)                                  # L, in minutes
                base_ok = (not flaws) or pb_rng.random() > 0.08
                placebo_ok = base_ok and ((not flaws) or pb_rng.random() > 0.06)
                during_ok = base_ok and ((not flaws) or pb_rng.random() > 0.06)
                rows.append({
                    "dataset": dataset, "batch": f"batch_{cell}", "fly_number": k + 1,
                    "fly_id": f"{dataset}/batch_{cell}/fly{k + 1}", "rig": rig, "trial": f"t{t}",
                    "trial_type": "pretest" if t < 2 else "training" if t < 4 else "testing", "trial_index": t,
                    "stim_kind": kind, "identity_consistent": True, "pre_window_known": True, "pre_min": pre_min,
                    "stim_min": stim_min, "trial_body_pitch_deg": RIG_SHIFT[rig] + dev + rng.normal(0.0, 0.5),
                    "held_subtype": state if rng.random() < 0.8 else "retracted",
                    "spont_per_events": float(spont) if known else nan, "spont_per_any": float(spont > 0) if known else nan,
                    "abd_tracked_frac_pre": 0.95 if pre_ok else 0.2, "abd_tracked_frac_stim": 0.9 if stim_ok else 0.1,
                    "abd_quality_ok": pre_ok, "abd_quality_ok_stim": stim_ok,
                    "abd_rest_bend_deg": rest_bend + rng.normal(0.0, 0.5) if pre_ok else nan,
                    "abd_bend_sd_pre_deg": bend_sd + rng.normal(0.0, 0.2) if pre_ok else nan,
                    "abd_bend_sd_stim_deg": bend_sd + rng.normal(0.0, 0.2) if stim_ok else nan,
                    "abd_curl_events_pre": float(rng.poisson(curl_rate * pre_min)) if pre_ok else nan,
                    "abd_curl_events_stim": float(rng.poisson(curl_rate * (1 + boost) * stim_min)) if stim_ok else nan,
                    "abd_straighten_events_pre": float(rng.poisson(0.5 * curl_rate * pre_min)) if pre_ok else nan,
                    "abd_straighten_events_stim": float(rng.poisson(0.5 * curl_rate * stim_min)) if stim_ok else nan,
                    "abd_curl_delta_deg": float(rng.normal(boost, 1.0)) if stim_ok else nan,
                    "pb_window_min": length,
                    "abd_tracked_frac_base": 0.95 if base_ok else 0.2,
                    "abd_tracked_frac_placebo": 0.9 if placebo_ok else 0.1,
                    "abd_tracked_frac_during": 0.9 if during_ok else 0.1,
                    "abd_base_ok": base_ok,
                    "abd_rest_bend_loo_deg": rest_bend + pb_rng.normal(0.0, 0.5) if base_ok else nan,
                    "abd_curl_events_placebo": float(pb_rng.poisson(curl_rate * length)) if placebo_ok else nan,
                    "abd_curl_events_during": float(pb_rng.poisson(curl_rate * (1 + boost) * length)) if during_ok else nan,
                    "abd_straighten_events_placebo": float(pb_rng.poisson(0.5 * curl_rate * length)) if placebo_ok else nan,
                    "abd_straighten_events_during": float(pb_rng.poisson(0.5 * curl_rate * length)) if during_ok else nan,
                    "tip_pitch_rest_deg": tip_pitch + rng.normal(0.0, 0.5) if pre_ok else nan,
                    "tip_moving_frac_pre": moving if pre_ok else nan,
                    "per_abdomen_coupling_rho": COUPLING[state] + rng.normal(0.0, 0.05) if pre_ok else nan,
                    "fly_abd_rest_bend_deg": rest_bend, "fly_tip_pitch_rest_deg": tip_pitch,
                    "fly_tip_moving_frac_pre": moving})
    return trial_table(rows)


@pytest.fixture(scope="module")
def data():
    """Everything rises with head-up mounting (strongly, so the answers are "Yes"); the stimulus raises the curl rate."""
    ft = abd_cohort(seed=7, bend=0.9, tip=0.8, activity=1.0, restless=0.7, boost=3.0)
    fly = fly_table(ft)
    return SimpleNamespace(ft=ft, fly=fly, stats=AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM, seed=0))


@pytest.fixture(scope="module")
def null_data():
    """The same flies with no relation between pitch and anything, and no effect of the stimulus."""
    ft = abd_cohort(seed=24)
    fly = fly_table(ft)
    return SimpleNamespace(ft=ft, fly=fly, stats=AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM, seed=0))


@pytest.fixture(scope="module")
def figures(data):
    return AM.build_abdomen_figures(data.ft, data.fly, data.stats)


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


def points(ax) -> list:
    """The (x, y) of every scatter marker of an axes, rounded and sorted (the collections' order is not the table's)."""
    xy = np.vstack([c.get_offsets() for c in ax.collections if isinstance(c, PathCollection)])
    return sorted(map(tuple, np.round(xy, 6).tolist()))


def expected_points(x, y) -> list:
    return sorted(map(tuple, np.round(np.column_stack([x, y]), 6).tolist()))


def bar_heights(ax) -> list:
    return [p.get_height() for p in sorted(ax.patches, key=lambda p: p.get_x()) if p.get_height() > 0]


def strata_of(fly) -> np.ndarray:
    return (fly["dataset"] + " | " + fly["rig"]).to_numpy()


# ---------------------------------------------------------------------------
# The statistics behind the figures
# ---------------------------------------------------------------------------

def test_the_rest_bend_strip_summarises_the_fly_table_by_rig(data):
    rigs = {r["rig"]: r for r in data.stats["rest_bend"]["rigs"]}
    assert sorted(rigs) == ["BehaviorLocust", "Flybehavior2", "Flybehavior3"]
    for rig, g in data.fly.groupby("rig"):
        values = g["abd_rest_bend_deg"].dropna()
        assert rigs[rig]["n_flies"] == len(values) and rigs[rig]["median_deg"] == pytest.approx(values.median())
        assert rigs[rig]["q1_deg"] == pytest.approx(values.quantile(0.25))
        assert rigs[rig]["q3_deg"] == pytest.approx(values.quantile(0.75))
    assert rigs["Flybehavior2"]["n_flies"] == 16                                  # two datasets share this rig
    assert data.stats["rest_bend"]["n_flies"] == int(data.fly["abd_rest_bend_deg"].notna().sum())
    lo, hi = min(rigs.values(), key=lambda r: r["median_deg"]), max(rigs.values(), key=lambda r: r["median_deg"])
    assert data.stats["rest_bend"]["answer"] == (
        f"Rig medians of the resting bend span {hi['median_deg'] - lo['median_deg']:.1f}° "
        f"({lo['rig']} {lo['median_deg']:+.1f}° to {hi['rig']} {hi['median_deg']:+.1f}°)")


def test_the_rest_bend_answer_says_so_when_no_fly_has_a_resting_bend():
    ft = trial_table([windows(f"fly{k + 1}", 1, 1, trial_index=t) for k in range(3) for t in range(2)])
    stats = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)
    assert stats["rest_bend"]["answer"] == "No fly has a resting bend" and stats["rest_bend"]["n_flies"] == 0
    assert stats["placeholders"]["abdomen_rest_bend_per_fly"]["reason"] == "no fly has a resting bend"


def test_the_rest_bend_answer_for_one_rig_names_it():
    ft = trial_table([windows(f"fly{k + 1}", 1, 1, trial_index=t, rig="Flybehavior2", abd_rest_bend_deg=3.0 + k,
                              fly_abd_rest_bend_deg=3.0 + k) for k in range(3) for t in range(2)])
    stats = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)
    assert stats["rest_bend"]["answer"] == "One rig: the median resting bend is +4.0° (Flybehavior2)"


def test_the_pitch_scatters_use_the_rig_centred_pitch_of_the_fly_table_and_a_dataset_by_rig_permutation(data):
    fly, s = data.fly, data.stats["bend_vs_pitch"]
    rho, p = M.spearman(fly["mount_pitch_centred_deg"], fly["abd_rest_bend_deg"])
    raw_rho, _ = M.spearman(fly["mount_pitch_deg"], fly["abd_rest_bend_deg"])
    assert s["rho"] == pytest.approx(rho) and s["p"] == pytest.approx(p)
    assert s["p_perm"] == M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly["abd_rest_bend_deg"],
                                                     strata_of(fly), n=N_PERM, seed=0)
    assert s["n_flies"] == int(fly[["mount_pitch_centred_deg", "abd_rest_bend_deg"]].notna().all(axis=1).sum())
    assert s["n_strata"] == 4 and s["n_strata_shuffled"] == 4
    assert abs(rho - raw_rho) > 0.01                                              # this cohort's rig offsets tell them apart


def test_the_tip_pitch_is_centred_on_its_rig_median_like_the_mounting_angle(data):
    fly, s = data.fly, data.stats["tip_vs_pitch"]
    tip = fly["tip_pitch_rest_deg"]
    centred = tip - tip.groupby(fly["rig"]).transform("median")
    rho, p = M.spearman(fly["mount_pitch_centred_deg"], centred)
    assert s["rho"] == pytest.approx(rho) and s["p"] == pytest.approx(p)
    assert s["p_perm"] == M.stratified_permutation_p(fly["mount_pitch_centred_deg"], centred, strata_of(fly),
                                                     n=N_PERM, seed=0)
    assert abs(rho - M.spearman(fly["mount_pitch_centred_deg"], tip)[0]) > 0.01        # the rig offsets would have mattered


def confound_flies():
    """Twelve flies of one rig in two datasets: pitch and resting bend both rise from D1 to D2, but inside each dataset the
    two are unrelated, so only a test that shuffles within the dataset sees there is nothing."""
    rows = []
    for dataset, pitches, bends in (("D1", [-3.0, -2.6, -2.2, -1.8, -1.4, -1.0], [3.0, 6.0, 1.0, 5.0, 2.0, 4.0]),
                                    ("D2", [1.0, 1.4, 1.8, 2.2, 2.6, 3.0], [12.0, 15.0, 13.0, 16.0, 11.0, 14.0])):
        for k, (pitch, bend) in enumerate(zip(pitches, bends)):
            for t in range(3):
                rows.append(row(f"fly{k + 1}", dataset=dataset, trial_body_pitch_deg=pitch, trial_index=t,
                                fly_abd_rest_bend_deg=bend, abd_quality_ok=True, abd_rest_bend_deg=bend))
    return trial_table(rows)


def test_the_answer_follows_the_stratified_p_not_the_pooled_spearman_p():
    ft = confound_flies()
    fly = fly_table(ft)
    s = AM.compute_abdomen_stats(ft, fly, n_perm=2000, seed=0)["bend_vs_pitch"]
    assert s["p"] < 0.05                                                  # pooled over the datasets the two look related
    assert s["p_perm"] > 0.3 and s["verdict"] == "No clear link" and s["n_strata"] == 2


def test_with_no_dataset_by_rig_holding_two_flies_there_is_nothing_to_shuffle_and_the_answer_is_not_enough_data():
    rows = []
    for k in range(4):                                  # one rig (so the pitch varies after centring), one fly per dataset
        for t in range(2):
            rows.append(row(f"fly{k + 1}", rig="R", dataset=f"D{k}", trial_body_pitch_deg=float(k), trial_index=t,
                            fly_abd_rest_bend_deg=float(k), abd_quality_ok=True, abd_rest_bend_deg=float(k)))
    ft = trial_table(rows)
    fly = fly_table(ft)
    s = AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM)["bend_vs_pitch"]
    assert s["n_flies"] == 4 and s["n_strata"] == 4 and s["n_strata_shuffled"] == 0
    assert s["p_perm"] == 1.0 and s["rho"] == pytest.approx(1.0)                  # the permutation had nothing to move ...
    assert s["verdict"].startswith("Not enough data to tell")                    # ... so it must not say "No clear link"
    two = AM.compute_abdomen_stats(ft[ft["fly_id"].str.contains("D0|D1")], fly[fly["dataset"].isin(["D0", "D1"])],
                                   n_perm=N_PERM)["bend_vs_pitch"]
    assert two["n_flies"] == 2 and two["verdict"] == "Not enough data to tell"      # and two flies are never enough, plainly


def hand_windows():
    """Three flies, odor trials only: curl events in the placebo and during windows, which are as long as each other in a
    trial (``pb`` minutes) and different between trials."""
    spec = {"fly1": [(2, 1, 0.25), (0, 3, 0.25)], "fly2": [(1, 0, 0.5)], "fly3": [(4, 2, 0.25), (0, 0, 0.125)]}
    rows = [windows(fid, placebo, during, pb=pb, straighten=(1, 0), trial_index=i)
            for fid, trials in spec.items() for i, (placebo, during, pb) in enumerate(trials)]
    return trial_table(rows)


def test_event_rates_pool_events_over_the_minutes_of_the_same_trials_and_never_average_trial_rates():
    ft = hand_windows()
    cell = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["events"]["curl"]["odor"]
    assert (cell["n_trials"], cell["n_flies"]) == (5, 3)
    assert (cell["placebo"]["events"], cell["placebo"]["minutes"]) == (7.0, pytest.approx(1.375))
    assert (cell["during"]["events"], cell["during"]["minutes"]) == (6.0, pytest.approx(1.375))
    assert cell["placebo"]["rate"] == pytest.approx(7 / 1.375) and cell["during"]["rate"] == pytest.approx(6 / 1.375)
    assert cell["during"]["rate"] != pytest.approx(np.mean([4, 12, 0, 8, 0]))         # what averaging trial rates would show
    assert cell["diff"] == pytest.approx((6 - 7) / 1.375)


def test_curls_and_straightenings_are_counted_apart():
    ft = hand_windows()
    events = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["events"]
    assert events["curl"]["odor"]["placebo"]["events"] == 7.0 and events["straighten"]["odor"]["placebo"]["events"] == 5.0
    assert events["straighten"]["odor"]["during"]["events"] == 0.0
    assert events["straighten"]["odor"]["placebo"]["minutes"] == pytest.approx(1.375)


def test_the_p_is_the_window_swap_permutation_of_the_placebo_and_during_counts_of_the_pooled_trials(data):
    ft = data.ft
    for event in ("curl", "straighten"):
        for kind in ("odor", "light"):
            use = ft[(ft["stim_kind"] == kind) & ft[f"abd_{event}_events_placebo"].notna()
                     & ft[f"abd_{event}_events_during"].notna()]
            diff, p = M.paired_window_permutation(
                use, events_pre=f"abd_{event}_events_placebo", min_pre="pb_window_min",
                events_stim=f"abd_{event}_events_during", min_stim="pb_window_min", n=N_PERM, seed=0, scheme="swap")
            cell = data.stats["events"][event][kind]
            assert cell["diff"] == pytest.approx(diff) and cell["p_perm"] == p, (event, kind)
            assert cell["n_trials"] == len(use) and cell["n_flies"] == use["fly_id"].nunique()


def test_a_stimulus_that_changes_nothing_is_not_called_significant(null_data):
    """The null cohort has the same event rate in the last L s before the stimulus and in the first L s of it: none of the
    four comparisons says "Yes"."""
    events = null_data.stats["events"]
    for event in ("curl", "straighten"):
        for kind in ("odor", "light"):
            assert events[event][kind]["verdict"] == "No clear difference", (event, kind, events[event][kind]["p_perm"])


def _simulated_trials(rng, *, flies=6, trials=8, evoked=0.0, tau_s=3.0):
    """The placebo columns of simulated odor trials, as the table builder writes them (10 fps): every fly has a resting level
    of its own and an AR(1) bend (SD 4 degrees) in every trial, the stimulus starts 30 s in and runs to the end of the 45 s
    trace, so L = 15 s; ``evoked`` = mean number of 0.5 s, 15 degree curls added to the first 15 s of the stimulus."""
    a = math.exp(-1.0 / (tau_s * 10.0))
    rows = []
    for fly in range(flies):
        level, traces = rng.normal(0.0, 3.0), []
        for _ in range(trials):
            bend, _ = signal.lfilter([1.0], [1.0, -a], rng.normal(0.0, 4.0 * math.sqrt(1.0 - a * a), 450),
                                     zi=[a * rng.normal(0.0, 4.0)])
            bend = bend + 6.0 + level
            for start in rng.integers(300, 430, rng.poisson(evoked)):
                bend[start:start + 5] += 15.0
            traces.append(M.PlaceboTrial(bend, 300, None, 10.0))
        for columns in M.placebo_fly(traces, rise_deg=10.0, min_s=0.3, min_tracked_frac=0.5):
            rows.append({**columns, "fly_id": f"D/b/fly{fly}", "stim_kind": "odor"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("tau_s", [3.0, 10.0])
def test_simulated_traces_on_which_the_stimulus_does_nothing_are_not_answered_yes(tau_s):
    """From the bend trace to the answer in the figure: the real event counter, the leave-one-out placebo columns and the
    window-swap p of ``_event_stats``. 60 replicates of 6 flies x 8 trials; the whole-window design said "Yes" here in a
    quarter of them and the adjacent baseline, with more trials, in a third (``tests/test_mounting_metrics.py``)."""
    yes = {"curl": 0, "straighten": 0}
    for rep in range(60):
        events = AM._event_stats(_simulated_trials(np.random.default_rng(9000 + rep), tau_s=tau_s), n_perm=199, seed=rep)
        for event in yes:
            cell = events[event]["odor"]
            assert cell["n_flies"] == 6 and cell["window_s"]["median"] == pytest.approx(15.0)
            yes[event] += cell["verdict"].startswith("Yes")
    assert max(yes.values()) <= 6, yes


def test_simulated_traces_with_curls_added_to_the_stimulus_are_answered_yes_for_curls_only():
    for rep in range(10):
        events = AM._event_stats(_simulated_trials(np.random.default_rng(9500 + rep), evoked=1.0), n_perm=199, seed=rep)
        assert events["curl"]["odor"]["verdict"] == "Yes — more often than just before the stimulus", rep
        assert events["straighten"]["odor"]["verdict"] == "No clear difference", rep


def test_the_event_stats_use_the_placebo_and_during_columns_and_not_the_pre_and_stim_ones(data):
    wild = data.ft.copy()
    wild[["abd_curl_events_pre", "abd_curl_events_stim", "abd_straighten_events_pre", "abd_straighten_events_stim"]] = 999.0
    wild["pre_min"], wild["stim_min"] = 0.5, 0.01
    again = AM.compute_abdomen_stats(wild, data.fly, n_perm=N_PERM, seed=0)
    assert again["events"] == data.stats["events"]


def test_the_permutation_p_values_use_the_seed_that_was_given(null_data):
    ft, fly = null_data.ft, null_data.fly
    other = AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM, seed=3)
    assert other["seed"] == 3
    p_events, p_scatters = [], []
    for event in ("curl", "straighten"):
        for kind in ("odor", "light"):
            use = ft[(ft["stim_kind"] == kind) & ft[f"abd_{event}_events_placebo"].notna()
                     & ft[f"abd_{event}_events_during"].notna()]
            expected = M.paired_window_permutation(
                use, events_pre=f"abd_{event}_events_placebo", min_pre="pb_window_min",
                events_stim=f"abd_{event}_events_during", min_stim="pb_window_min", n=N_PERM, seed=3, scheme="swap")[1]
            assert other["events"][event][kind]["p_perm"] == expected, (event, kind)
            p_events.append((other["events"][event][kind]["p_perm"], null_data.stats["events"][event][kind]["p_perm"]))
    for key, column in (("bend_vs_pitch", "abd_rest_bend_deg"), ("restless", None), ("tip_vs_pitch", None)):
        if column:
            expected = M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly[column], strata_of(fly),
                                                  n=N_PERM, seed=3)
            assert other[key]["p_perm"] == expected, key
        p_scatters.append((other[key]["p_perm"], null_data.stats[key]["p_perm"]))
    assert any(a != b for a, b in p_events) and any(a != b for a, b in p_scatters)       # the seed does change the draws
    for key in ("bend_sd", "tip_moving"):
        assert other["activity"][key]["p_perm"] == M.stratified_permutation_p(
            fly["mount_pitch_centred_deg"], fly["abd_bend_sd_pre_deg" if key == "bend_sd" else "tip_moving_frac_pre"],
            strata_of(fly), n=N_PERM, seed=3), key


def test_only_trials_with_a_usable_baseline_and_both_windows_enter_the_event_rates():
    ft = trial_table([windows("fly1", 2, 1, trial_index=0), windows("fly2", 1, 1, trial_index=0),
                      windows("fly3", 3, 2, trial_index=0),
                      windows("fly4", 50, 50, trial_index=0, placebo_ok=False),            # the placebo window is not usable
                      windows("fly5", 60, 60, trial_index=0, during_ok=False),            # the stimulus window is not usable
                      windows("fly6", 70, 70, trial_index=0, base=False)])                # no baseline: no resting bend
    cell = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["events"]["curl"]["odor"]
    assert (cell["n_trials"], cell["n_flies"]) == (3, 3)
    assert cell["placebo"]["events"] == 6.0 and cell["during"]["events"] == 4.0
    assert cell["placebo"]["minutes"] == pytest.approx(0.75) and cell["during"]["minutes"] == pytest.approx(0.75)


def test_a_trial_whose_count_is_missing_never_enters_a_pooled_rate_or_its_minutes():
    ft = trial_table([windows("fly1", 2, 1, trial_index=0), windows("fly2", 1, 1, trial_index=0),
                      windows("fly3", 3, 2, trial_index=0, pb=0.5),
                      windows("fly4", 9, 9, trial_index=0, pb=0.5, placebo_ok=False),        # its minutes must not dilute
                      windows("fly5", 9, 9, trial_index=0, pb=0.5, during_ok=False)])
    cell = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["events"]["curl"]["odor"]
    assert (cell["n_trials"], cell["n_flies"]) == (3, 3)
    assert cell["placebo"]["minutes"] == pytest.approx(1.0) and cell["during"]["minutes"] == pytest.approx(1.0)
    assert cell["placebo"]["events"] == 6.0 and cell["during"]["events"] == 4.0
    assert cell["placebo"]["rate"] == pytest.approx(6.0) and cell["during"]["rate"] == pytest.approx(4.0)


def test_an_event_cell_leaves_out_a_trial_that_has_only_one_count_of_that_event():
    ft = trial_table([windows("fly1", 2, 1, trial_index=0), windows("fly2", 1, 1, trial_index=0),
                      windows("fly3", 3, 2, trial_index=0),
                      windows("fly3", 9, 9, trial_index=1, straighten=(5, np.nan))])    # no straightening count at the start
    events = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["events"]
    curl, straighten = events["curl"]["odor"], events["straighten"]["odor"]
    assert (curl["n_trials"], curl["placebo"]["events"], curl["during"]["events"]) == (4, 15.0, 13.0)
    assert curl["placebo"]["minutes"] == pytest.approx(1.0) and curl["during"]["minutes"] == pytest.approx(1.0)
    assert (straighten["n_trials"], straighten["placebo"]["events"]) == (3, 0.0)         # not the 5 of the half pair
    assert straighten["placebo"]["minutes"] == pytest.approx(0.75) and straighten["during"]["minutes"] == pytest.approx(0.75)


def test_a_trial_without_a_usable_window_length_is_left_out():
    ft = trial_table([windows("fly1", 2, 1, trial_index=0), windows("fly2", 1, 1, trial_index=0),
                      windows("fly3", 3, 2, trial_index=0),
                      windows("fly4", 9, 9, trial_index=0, pb=np.nan), windows("fly5", 9, 9, trial_index=0, pb=0.0),
                      windows("fly6", 9, 9, trial_index=0, pb=-0.25)])
    cell = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["events"]["curl"]["odor"]
    assert (cell["n_trials"], cell["n_flies"]) == (3, 3) and cell["placebo"]["events"] == 6.0


def test_the_placebo_bar_comes_from_the_trials_whose_stimulus_window_is_usable_too():
    ft = trial_table([windows("fly1", 2, 1, trial_index=0), windows("fly2", 1, 1, trial_index=0),
                      windows("fly3", 3, 2, trial_index=0),
                      windows("fly3", 9, 5, trial_index=1, during_ok=False)])     # a good placebo window, no stimulus window
    cell = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["events"]["curl"]["odor"]
    assert cell["placebo"]["events"] == 6.0 and cell["placebo"]["minutes"] == pytest.approx(0.75) and cell["n_trials"] == 3


def test_odor_and_light_trials_are_kept_apart_and_odor_plus_light_trials_are_in_neither():
    ft = trial_table([windows("fly1", 2, 1, trial_index=0), windows("fly2", 1, 1, trial_index=0),
                      windows("fly3", 3, 2, trial_index=0),
                      windows("fly1", 4, 9, trial_index=1, kind="light", pb=0.5),
                      windows("fly2", 2, 7, trial_index=1, kind="light", pb=0.5),
                      windows("fly3", 1, 8, trial_index=1, kind="light", pb=0.5),
                      windows("fly1", 70, 70, trial_index=2, kind="odor+light"),
                      windows("fly2", 70, 70, trial_index=2, kind="none"),
                      windows("fly3", 70, 70, trial_index=2, kind="none", during_ok=False)])    # not scored: not counted
    events = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["events"]
    odor, light = events["curl"]["odor"], events["curl"]["light"]
    assert (odor["placebo"]["events"], odor["during"]["events"], odor["n_trials"]) == (6.0, 4.0, 3)
    assert (light["placebo"]["events"], light["during"]["events"], light["n_trials"]) == (7.0, 24.0, 3)
    assert light["during"]["minutes"] == pytest.approx(1.5) and events["n_other"] == 2


def test_trials_of_a_drifted_fly_never_enter_a_pooled_rate(data):
    ft = data.ft
    ghost = ft[ft["fly_id"] == ft["fly_id"].iloc[0]].copy()
    ghost["fly_id"], ghost["identity_consistent"] = "DS-A/batch_0/ghost", False
    ghost[["abd_curl_events_placebo", "abd_curl_events_during", "abd_curl_events_pre", "abd_curl_events_stim"]] = 99.0
    ghost[["abd_quality_ok", "abd_quality_ok_stim"]] = True
    both = pd.concat([ft, ghost], ignore_index=True)
    stats = AM.compute_abdomen_stats(both, fly_table(both), n_perm=N_PERM, seed=0)
    assert stats["events"] == data.stats["events"] and stats["coupling"] == data.stats["coupling"]
    assert stats["windows"] == data.stats["windows"] and stats["n_trials"] == data.stats["n_trials"]
    assert stats["n_fly_trials"] == len(both)


def test_a_flag_that_came_back_from_a_csv_as_text_is_still_read_as_a_flag(data):
    as_text = data.ft.copy()
    for flag in ("identity_consistent", "abd_quality_ok", "abd_quality_ok_stim"):
        as_text[flag] = as_text[flag].map({True: "True", False: "False"})
    again = AM.compute_abdomen_stats(as_text, data.fly, n_perm=N_PERM, seed=0)
    assert again["events"] == data.stats["events"] and again["windows"] == data.stats["windows"]


@pytest.mark.parametrize("n_flies, answer", [(0, "Not enough data to tell"), (1, "Not enough data to tell"),
                                             (2, "Not enough data to tell"),
                                             (3, "Yes — more often than just before the stimulus")])
def test_a_stimulus_comparison_needs_three_flies(n_flies, answer):
    rows = [windows(f"fly{k + 1}", 0, 3, trial_index=t) for k in range(n_flies) for t in range(12)]
    rows += [windows(f"fly{k + 10}", 0, 0, trial_index=t, kind="light") for k in range(3) for t in range(2)]
    ft = trial_table(rows)
    cell = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=2000, seed=0)["events"]["curl"]["odor"]
    assert cell["n_flies"] == n_flies and cell["verdict"] == answer
    if n_flies == 2:
        assert cell["p_perm"] < 0.05                                             # the p alone would have said "Yes"


def test_a_stimulus_that_lowers_the_curl_rate_is_answered_in_the_other_direction():
    rows = [windows(f"fly{k + 1}", 2, 0, trial_index=t) for k in range(4) for t in range(12)]
    ft = trial_table(rows)
    cell = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=2000, seed=0)["events"]["curl"]["odor"]
    assert cell["diff"] < 0 and cell["verdict"] == "Yes — less often than just before the stimulus"


def test_a_stimulus_that_changes_nothing_is_no_clear_difference():
    rows = [windows(f"fly{k + 1}", t % 3, (t + 1) % 3, trial_index=t) for k in range(4) for t in range(9)]
    ft = trial_table(rows)                                              # the same counts before and during, in another order
    cell = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=2000, seed=0)["events"]["curl"]["odor"]
    assert cell["diff"] == pytest.approx(0.0, abs=1e-12) and cell["p_perm"] == 1.0
    assert cell["verdict"] == "No clear difference"


def test_the_restless_scatter_compares_the_spontaneous_curl_rate_with_the_spontaneous_per_rate(data):
    fly, s = data.fly, data.stats["restless"]
    pair = fly[["abd_curl_rate_pre_per_min", "spont_per_rate_per_min"]].notna().all(axis=1)
    assert s["n_flies"] == int(pair.sum())
    rho, p = M.spearman(fly["abd_curl_rate_pre_per_min"], fly["spont_per_rate_per_min"])
    assert s["rho"] == pytest.approx(rho) and s["p"] == pytest.approx(p)
    assert s["p_perm"] == M.stratified_permutation_p(fly["abd_curl_rate_pre_per_min"], fly["spont_per_rate_per_min"],
                                                     strata_of(fly), n=N_PERM, seed=0)


def test_the_coupling_strip_groups_flies_by_their_dominant_proboscis_state(data):
    fly, c = data.fly, data.stats["coupling"]
    got = {g["state"]: g for g in c["groups"]}
    assert list(got) == ["retracted", "held_down", "held_back"]                    # held forward only when a fly has it
    have = fly[fly["per_abdomen_coupling_rho"].notna() & fly["proboscis_state_dominant"].notna()]
    for state, g in have.groupby("proboscis_state_dominant"):
        assert got[state]["n_flies"] == len(g)
        assert got[state]["median"] == pytest.approx(g["per_abdomen_coupling_rho"].median())
        assert got[state]["n_trials"] == int(g["n_coupling_trials"].sum())
    assert c["n_flies"] == len(have) and c["median"] == pytest.approx(have["per_abdomen_coupling_rho"].median())
    assert got["held_down"]["median"] > got["retracted"]["median"] > got["held_back"]["median"]      # the states differ


def test_flies_without_a_coupling_or_a_dominant_state_are_left_out_of_the_coupling_strip():
    ft = trial_table([row("fly1", held_subtype="held_down", per_abdomen_coupling_rho=0.5, abd_quality_ok=True),
                      row("fly2", held_subtype="held_down", per_abdomen_coupling_rho=np.nan, abd_quality_ok=True),
                      row("fly3", held_subtype=None, per_abdomen_coupling_rho=0.9, abd_quality_ok=True),
                      row("fly4", held_subtype="retracted", per_abdomen_coupling_rho=-0.1, abd_quality_ok=True)])
    c = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["coupling"]
    assert c["n_flies"] == 2
    assert {g["state"]: g["n_flies"] for g in c["groups"]} == {"retracted": 1, "held_down": 1, "held_back": 0}
    assert c["answer"] == ("Overall median ρ=+0.20 over 2 flies; by proboscis state: retracted -0.10, held down +0.50, "
                           "held back no flies (descriptive, no test)")


def test_a_state_without_flies_keeps_its_slot_and_label_but_has_no_dots():
    ft = trial_table([row("fly1", held_subtype="held_down", per_abdomen_coupling_rho=0.5, abd_quality_ok=True),
                      row("fly2", held_subtype="retracted", per_abdomen_coupling_rho=-0.1, abd_quality_ok=True),
                      row("fly3", held_subtype="retracted", per_abdomen_coupling_rho=0.1, abd_quality_ok=True)])
    fly = fly_table(ft)
    fig = AM.build_abdomen_figures(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM))["per_abdomen_coupling"]
    ax = fig.axes[0]
    assert [t.get_text().split("\n")[0] for t in ax.get_xticklabels()] == ["Retracted", "Held down", "Held back"]
    assert "Held back n=0 flies" in all_text(fig) and len(points(ax)) == 3
    assert sorted(t.get_text() for t in ax.texts) == ["+0.00", "+0.50"]            # no label for the empty slot
    assert sorted({round(x) for x, _ in points(ax)}) == [0, 1]                       # nothing was drawn in the third slot


def test_a_held_forward_fly_gets_its_own_group():
    ft = trial_table([row("fly1", held_subtype="held_forward", per_abdomen_coupling_rho=0.5, abd_quality_ok=True),
                      row("fly2", held_subtype="retracted", per_abdomen_coupling_rho=0.1, abd_quality_ok=True)])
    groups = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["coupling"]["groups"]
    assert [g["state"] for g in groups] == ["retracted", "held_down", "held_back", "held_forward"]


def test_the_activity_scatters_use_the_bend_sd_and_the_tip_moving_fraction(data):
    fly, a = data.fly, data.stats["activity"]
    for key, column in (("bend_sd", "abd_bend_sd_pre_deg"), ("tip_moving", "tip_moving_frac_pre")):
        rho, p = M.spearman(fly["mount_pitch_centred_deg"], fly[column])
        assert a[key]["rho"] == pytest.approx(rho) and a[key]["p"] == pytest.approx(p), key
        assert a[key]["p_perm"] == M.stratified_permutation_p(fly["mount_pitch_centred_deg"], fly[column], strata_of(fly),
                                                              n=N_PERM, seed=0), key
        assert a[key]["n_flies"] == int(fly[["mount_pitch_centred_deg", column]].notna().all(axis=1).sum()), key


def test_the_dropped_windows_are_counted_for_the_pre_stimulus_window_and_the_three_windows_of_the_placebo_design():
    nan = np.nan
    no_stimulus = dict.fromkeys(("abd_tracked_frac_placebo", "abd_tracked_frac_during", "abd_curl_events_placebo",
                                 "abd_curl_events_during", "abd_straighten_events_placebo",
                                 "abd_straighten_events_during"), nan)      # a trial whose placebo / during window is absent
    ft = trial_table([
        windows("fly1", 1, 1, trial_index=0, abd_tracked_frac_pre=0.9),
        windows("fly1", 1, 1, trial_index=1, abd_tracked_frac_pre=0.9),
        windows("fly1", 1, 1, trial_index=2, during_ok=False, abd_tracked_frac_pre=0.9),        # the stimulus window is poor
        windows("fly1", 1, 1, trial_index=3, placebo_ok=False, abd_tracked_frac_pre=0.9),       # so is a placebo window
        windows("fly2", 1, 1, trial_index=0, base=False, abd_tracked_frac_pre=0.2, abd_quality_ok=False),
        windows("fly2", 1, 1, trial_index=1, base=False, abd_tracked_frac_pre=0.1, abd_quality_ok=False),
        windows("fly2", 1, 1, trial_index=2, abd_tracked_frac_pre=0.8, **no_stimulus),
        windows("fly3", 1, 1, trial_index=0, base=False, abd_quality_ok=False,                 # no window of any kind
                **{**no_stimulus, "abd_tracked_frac_pre": nan, "abd_tracked_frac_base": nan}),
        windows("fly2", 1, 1, trial_index=3, abd_rest_bend_loo_deg=nan),          # counts, but no resting bend to score by
        windows("fly2", 1, 1, trial_index=4, abd_tracked_frac_pre=0.3, abd_quality_ok=False),   # a poor pre window only
        windows("fly9", 1, 1, trial_index=0, base=False, identity_consistent=False, abd_tracked_frac_pre=0.0,
                abd_quality_ok=False)])
    w = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM, min_tracked_frac=0.5)["windows"]
    assert w["pre"] == {"with_window": 8, "usable": 5, "dropped": 3}
    assert w["base"] == {"with_window": 9, "usable": 7, "dropped": 2}       # whether a trial can LEND its baseline
    for name in ("placebo", "during"):                  # usable + dropped + blank = the windows that exist
        assert w[name] == {"with_window": 8, "usable": 4, "dropped": 1, "blank_by_base": 3}, name


def test_the_gate_value_is_recorded_when_it_is_given(data):
    assert AM.compute_abdomen_stats(data.ft, data.fly, n_perm=N_PERM, min_tracked_frac=0.25)["min_tracked_frac"] == 0.25
    assert data.stats["min_tracked_frac"] is None


def test_the_stats_are_repeatable_for_a_seed_and_record_the_permutation_count(data):
    again = AM.compute_abdomen_stats(data.ft, data.fly, n_perm=N_PERM, seed=0)
    assert again == data.stats and again["n_perm"] == N_PERM and again["seed"] == 0


def test_computing_the_stats_of_an_empty_input_or_one_without_abdomen_data_does_not_raise():
    empty = AM.compute_abdomen_stats(trial_table([]), pd.DataFrame(columns=MP.FLY_COLUMNS), n_perm=N_PERM)
    assert empty["n_fly_trials"] == 0 and empty["no_abdomen_data"] is True
    ft = abd_cohort(seed=3, per_cell=2)
    ft[ABDOMEN_COLUMNS] = np.nan
    ft[["abd_quality_ok", "abd_quality_ok_stim"]] = False
    none = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)
    assert none["no_abdomen_data"] is True and none["n_fly_trials"] == len(ft)
    assert none["windows"]["pre"]["usable"] == 0


def test_usable_windows_of_a_drifted_fly_alone_are_no_abdomen_data(data, caplog):
    ft = data.ft.copy()
    ft["identity_consistent"] = False
    ft.loc[ft["fly_id"] == ft["fly_id"].iloc[0], "identity_consistent"] = True
    ft.loc[ft["fly_id"] == ft["fly_id"].iloc[0], ["abd_quality_ok", "abd_quality_ok_stim"]] = False
    stats = AM.compute_abdomen_stats(ft, data.fly, n_perm=N_PERM)
    assert stats["no_abdomen_data"] is True and stats["n_trials"] == int((ft["fly_id"] == ft["fly_id"].iloc[0]).sum())
    with caplog.at_level(logging.WARNING):
        assert AM.build_abdomen_figures(ft, data.fly, stats) == {}
    assert caplog.text == ""                                                        # no six failed attempts to draw nothing


def test_a_table_without_the_abdomen_columns_has_no_abdomen_data():
    gone = ABDOMEN_COLUMNS + ["abd_quality_ok", "abd_quality_ok_stim", "stim_kind"]
    ft = abd_cohort(seed=3, per_cell=2).drop(columns=gone)
    stats = AM.compute_abdomen_stats(ft, fly_table(abd_cohort(seed=3, per_cell=2)), n_perm=N_PERM)
    assert stats["no_abdomen_data"] is True


def test_a_difference_of_rates_is_written_with_its_sign_and_never_as_minus_zero():
    assert AM._signed_rate(-1.234) == "-1.2" and AM._signed_rate(0.5) == "+0.50" and AM._signed_rate(12.4) == "+12"
    assert AM._signed_rate(-0.0004) == "+0.00" and AM._signed_rate(float("nan")) == "n/a"
    assert AM._signed(-0.2) == "-0.20" and AM._signed(0.07) == "+0.07" and AM._signed(float("nan")) == "n/a"


# ---------------------------------------------------------------------------
# The wording: the question is the title and the answer comes from the data
# ---------------------------------------------------------------------------

def test_each_figure_title_is_a_question_and_the_names_are_the_designs(figures):
    assert tuple(AM.NAMES) == NAMES and tuple(figures) == NAMES
    titles = {name: squash(fig.texts[0].get_text()) for name, fig in figures.items()}
    assert all(t.endswith("?") for t in titles.values()), titles
    assert titles["per_abdomen_coupling"] == "Do the proboscis and the abdomen move together?"
    assert len(set(titles.values())) == 6


def test_clear_relations_are_answered_yes_in_the_direction_of_the_effect(figures, data):
    s = data.stats
    assert s["bend_vs_pitch"]["p_perm"] < 0.05 and s["bend_vs_pitch"]["rho"] > 0
    text = all_text(figures["mount_pitch_vs_abdomen_rest_bend"])
    assert "Resting curl: Yes — flies tilted more head-up rest more curled" in text
    assert "Resting tip pitch: Yes — flies tilted more head-up have the tip pointing further down" in text
    text = all_text(figures["mount_pitch_vs_abdomen_activity"])
    assert "Bend variability: Yes — flies tilted more head-up have a more variable bend" in text
    assert "Tip movement: Yes — flies tilted more head-up move the tip more often" in text
    text = all_text(figures["abdomen_vs_spontaneous_per"])
    assert "Yes — flies that curl more often also show more spontaneous PER" in text


def test_a_negative_relation_is_answered_in_the_other_direction():
    ft = abd_cohort(seed=7, bend=-0.9, restless=0.7, per_link=-0.7)
    fly = fly_table(ft)
    stats = AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM)
    assert stats["bend_vs_pitch"]["rho"] < 0 and stats["bend_vs_pitch"]["p_perm"] < 0.05
    figs = AM.build_abdomen_figures(ft, fly, stats)
    text = all_text(figs["mount_pitch_vs_abdomen_rest_bend"])
    assert "Resting curl: Yes — flies tilted more head-down rest more curled" in text
    assert stats["restless"]["rho"] < 0
    assert "Yes — flies that curl more often show less spontaneous PER" in all_text(figs["abdomen_vs_spontaneous_per"])


def test_no_relation_is_answered_no_clear_link(null_data):
    s = null_data.stats
    for key in ("bend_vs_pitch", "tip_vs_pitch", "restless"):
        assert s[key]["p_perm"] > 0.05, key
    figs = AM.build_abdomen_figures(null_data.ft, null_data.fly, s)
    text = all_text(figs["mount_pitch_vs_abdomen_rest_bend"])
    assert "Resting curl: No clear link" in text and "Resting tip pitch: No clear link" in text and "Yes" not in text
    assert "No clear link" in all_text(figs["abdomen_vs_spontaneous_per"])


def test_the_stimulus_effect_is_answered_per_event_and_window(figures, data, null_data):
    curl = data.stats["events"]["curl"]
    assert curl["odor"]["p_perm"] < 0.05 and curl["odor"]["diff"] > 0
    text = all_text(figures["abdomen_curl_events_by_window"])
    assert "Curls at the start of the odor: Yes — more often than just before the stimulus" in text
    assert "Straightenings at the start of the odor: No clear difference" in text
    nulls = AM.build_abdomen_figures(null_data.ft, null_data.fly, null_data.stats)
    assert "Curls at the start of the odor: No clear difference" in all_text(nulls["abdomen_curl_events_by_window"])


def test_the_event_figure_says_in_plain_words_what_is_compared(figures):
    text = all_text(figures["abdomen_curl_events_by_window"])
    assert ("Each pair compares the last L s before the stimulus with the first L s of it "
            "(L = 4 s for the odor, 15 s for the light), both against the fly's resting bend from its other trials"
            in text)
    assert "during minus just before" in text and "uncorrected" in text
    assert "swaps the two window labels at random within each trial" in text
    assert "trial-level: trials, not flies, are shuffled" in text
    assert "it assumes the bend is stationary within a trial" in text


def test_no_text_claims_the_windows_are_compared_fairly(figures, section):
    sources = [Path(AM.__file__), Path(M.__file__)]
    for text in [section, *(all_text(f) for f in figures.values()), *(s.read_text(encoding="utf-8") for s in sources)]:
        assert "compared fairly" not in text


def test_no_text_says_the_windows_differ_only_by_chance_or_that_the_p_is_exact(figures, section):
    """The two windows are interchangeable only under an assumption (within-trial stationarity), and the p is a
    Monte-Carlo estimate: neither is a claim the report or a docstring may make without it."""
    sources = [Path(AM.__file__), Path(M.__file__), Path(MP.__file__)]
    for text in [section, *(all_text(f) for f in figures.values()), *(s.read_text(encoding="utf-8") for s in sources)]:
        flat = squash(text)
        assert "differ only by chance" not in flat and "exact window-label" not in flat
        assert "exchangeable when the stimulus does nothing" not in flat and "exact permutation" not in flat


def test_the_section_says_the_p_is_a_monte_carlo_estimate(section):
    assert re.search(rf"Window-swap permutation p: .*{N_PERM} permutations, seed 0: a Monte-Carlo permutation estimate",
                     section)


def test_the_window_length_is_reported_for_each_stimulus(data, section):
    odor, light = data.stats["events"]["curl"]["odor"], data.stats["events"]["curl"]["light"]
    assert odor["window_s"] == {"min": pytest.approx(4.0), "median": pytest.approx(4.0), "max": pytest.approx(4.0)}
    assert light["window_s"]["median"] == pytest.approx(15.0)
    assert "Window length L: odor 4 s, light 15 s" in section


def test_the_window_length_says_its_range_when_the_trials_differ():
    ft = trial_table([windows("fly1", 2, 1, trial_index=0, pb=0.25), windows("fly2", 1, 1, trial_index=0, pb=0.125),
                      windows("fly3", 3, 2, trial_index=0, pb=0.125), windows("fly3", 1, 1, trial_index=1, pb=0.0625)])
    fly = fly_table(ft)
    stats = AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM)
    cell = stats["events"]["curl"]["odor"]
    assert cell["window_s"] == {"min": pytest.approx(3.75), "median": pytest.approx(7.5), "max": pytest.approx(15.0)}
    assert "Window length L: odor 3.8–15 s (median 7.5 s), light n/a" in AM.abdomen_report_section(ft, fly, stats)
    assert "L = 3.8–15 s for the odor" in all_text(AM.build_abdomen_figures(ft, fly, stats)["abdomen_curl_events_by_window"])


def test_the_coupling_answer_is_descriptive_and_computed(figures, data):
    text = all_text(figures["per_abdomen_coupling"])
    c = data.stats["coupling"]
    assert "descriptive, no test" in text and f"{c['median']:+.2f}" in text
    for g in c["groups"]:
        assert f"{g['median']:+.2f}" in text
    labels = sorted(t.get_text() for t in figures["per_abdomen_coupling"].axes[0].texts)    # the median beside each bar
    assert labels == sorted(f"{g['median']:+.2f}" for g in c["groups"])


def test_every_p_value_on_a_figure_is_labelled_uncorrected(figures):
    seen = 0
    for name, fig in figures.items():
        text = all_text(fig)
        for match in re.finditer(r"\bp[=<]\s?[0-9.]+", text):
            seen += 1
            assert "uncorrected" in text[match.end():match.end() + 40], (name, text[match.start():match.end() + 40])
    assert seen >= 12


def test_no_text_says_nan_or_one_flies(figures, data):
    section = AM.abdomen_report_section(data.ft, data.fly, data.stats)
    for text in [section, *(all_text(f) for f in figures.values())]:
        assert not re.search(r"(?i)\bnan\b|\b1 flies\b|\b1 trials\b", text), re.findall(r"(?i).{30}\bnan\b.{30}", text)


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------

def test_render_writes_the_six_pngs_named_in_the_design(data, tmp_path):
    written = AM.render_abdomen_figures(data.ft, data.fly, tmp_path / "out", n_perm=N_PERM)
    assert tuple(written) == NAMES
    assert {p.name for p in (tmp_path / "out").glob("*.png")} == {f"{n}.png" for n in NAMES}
    for name, path in written.items():
        assert path == tmp_path / "out" / f"{name}.png" and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_pngs_are_300_dpi(data, tmp_path):
    from PIL import Image
    for path in AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM).values():
        with Image.open(path) as image:
            assert image.info["dpi"][0] == pytest.approx(300, abs=0.5)
            assert image.size[0] >= 1500                                              # a 5+ inch figure at 300 dpi


def test_render_takes_precomputed_stats_and_gives_the_same_pictures(data, tmp_path):
    a = AM.render_abdomen_figures(data.ft, data.fly, tmp_path / "a", n_perm=N_PERM, stats=data.stats)
    b = AM.render_abdomen_figures(data.ft, data.fly, tmp_path / "b", n_perm=N_PERM)
    assert all(a[n].read_bytes() == b[n].read_bytes() for n in NAMES)


def test_render_draws_from_the_stats_it_is_given(data, tmp_path):
    custom = {**data.stats, "placeholders": {"abdomen_rest_bend_per_fly": {"reason": "no reason", "n": "n=0 flies"}}}
    plain = AM.render_abdomen_figures(data.ft, data.fly, tmp_path / "a", n_perm=N_PERM, stats=data.stats)
    given = AM.render_abdomen_figures(data.ft, data.fly, tmp_path / "b", n_perm=N_PERM, stats=custom)
    assert plain["abdomen_rest_bend_per_fly"].read_bytes() != given["abdomen_rest_bend_per_fly"].read_bytes()
    assert plain["per_abdomen_coupling"].read_bytes() == given["per_abdomen_coupling"].read_bytes()


def test_every_figure_carries_an_n_label(figures):
    for name, fig in figures.items():
        assert re.search(r"n=\d", all_text(fig)), name


def test_per_fly_figures_say_n_flies_and_the_pooled_one_says_n_trials_and_flies(figures):
    for name in set(NAMES) - {"abdomen_curl_events_by_window"}:
        assert re.search(r"n=\d+ flies", all_text(figures[name])), name
    assert re.search(r"n=[\d,]+ trials / \d+ flies", all_text(figures["abdomen_curl_events_by_window"]))


def test_the_strip_plot_labels_every_rig_with_its_number_of_flies(figures, data):
    text = all_text(figures["abdomen_rest_bend_per_fly"])
    for r in data.stats["rest_bend"]["rigs"]:
        assert f"{r['rig']} n={r['n_flies']} flies" in text
    assert f"n={data.stats['rest_bend']['n_flies']} flies" in text


def test_every_bar_of_the_event_figure_is_labelled_with_its_trials_and_flies(figures, data):
    text = all_text(figures["abdomen_curl_events_by_window"])
    for event in ("curl", "straighten"):
        for cell in data.stats["events"][event].values():
            assert FIG.n_label(cell["n_trials"], cell["n_flies"]) in text, (event, cell["kind"])


def test_the_coupling_strip_labels_every_state_with_its_number_of_flies(figures, data):
    text = all_text(figures["per_abdomen_coupling"])
    for g in data.stats["coupling"]["groups"]:
        assert f"{FIG.STATE_LABELS[g['state']]} n={g['n_flies']} flies" in text


def test_the_two_panel_figures_have_two_axes_and_no_figure_has_a_twin_axis(figures):
    for name, fig in figures.items():
        assert len(fig.axes) == (2 if name in TWO_PANEL else 1), name
        for ax in fig.axes:
            assert ax.child_axes == [] and len(ax.get_shared_x_axes().get_siblings(ax)) == 1, name
            assert len(ax.get_shared_y_axes().get_siblings(ax)) == 1, name
        if name in TWO_PANEL:
            a, b = fig.axes
            assert a.get_position().x1 < b.get_position().x0                       # side by side, not stacked on one frame
            assert a.get_ylabel() and b.get_ylabel() and a.get_ylabel() != b.get_ylabel()
    for name in ("mount_pitch_vs_abdomen_rest_bend", "mount_pitch_vs_abdomen_activity"):
        text = all_text(figures[name])                                             # the mounting-angle axis, once per panel
        assert text.count("Mounting angle, rig-centred (°)") == 2 and text.count("← head-down") == 2, name
        assert text.count("head-up →") == 2, name


def test_per_fly_points_are_blue_with_a_surface_ring(figures, data):
    ax = figures["abdomen_vs_spontaneous_per"].axes[0]
    marks = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(marks) == 1 and len(marks[0].get_offsets()) == data.stats["restless"]["n_flies"]
    assert np.allclose(marks[0].get_facecolors(), to_rgba(BLUE), atol=0.01)
    assert np.allclose(marks[0].get_edgecolors(), to_rgba("#ffffff"), atol=0.01)
    for name in ("abdomen_rest_bend_per_fly", "per_abdomen_coupling", "mount_pitch_vs_abdomen_rest_bend"):
        for ax in figures[name].axes:
            strip = [c for c in ax.collections if isinstance(c, PathCollection)]
            assert strip and all(np.allclose(c.get_facecolors(), to_rgba(BLUE), atol=0.01) for c in strip), name


def test_the_just_before_bars_are_blue_odor_bars_orange_and_light_bars_orange_with_the_slash_hatch(figures):
    for ax in figures["abdomen_curl_events_by_window"].axes:
        bars = sorted((p for p in ax.patches if p.get_height() > 0), key=lambda p: p.get_x())
        assert len(bars) == 4
        before_odor, during_odor, before_light, during_light = bars          # just before, start of, just before, start of
        for bar in (before_odor, before_light):
            assert to_hex(bar.get_facecolor()) == BLUE and not bar.get_hatch()
        assert to_hex(during_odor.get_facecolor()) == ORANGE and not during_odor.get_hatch()
        assert to_hex(during_light.get_facecolor()) == ORANGE and during_light.get_hatch() == "//"


def test_the_window_labels_name_what_each_bar_is(figures):
    ax = figures["abdomen_curl_events_by_window"].axes[0]
    labels = [" ".join(t.get_text().split("\n")[:2]) for t in ax.get_xticklabels()]
    assert labels == ["Just before the odor", "Start of the odor", "Just before the light", "Start of the light"]


def test_a_cohort_with_odor_trials_only_draws_the_odor_pair_only():
    ft = trial_table([windows(f"fly{k + 1}", 1, 2, trial_index=t, straighten=(1, 1)) for k in range(4) for t in range(3)])
    fly = fly_table(ft)
    figs = AM.build_abdomen_figures(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM))
    for ax in figs["abdomen_curl_events_by_window"].axes:
        bars = [p for p in ax.patches if p.get_height() > 0]
        assert len(bars) == 2 and not any(p.get_hatch() for p in bars)
    assert "light" not in all_text(figs["abdomen_curl_events_by_window"]).lower()


def test_the_scatters_plot_the_fly_table_values(figures, data):
    fly = data.fly
    centred_tip = fly["tip_pitch_rest_deg"] - fly["tip_pitch_rest_deg"].groupby(fly["rig"]).transform("median")
    x = fly["mount_pitch_centred_deg"]
    cases = [("mount_pitch_vs_abdomen_rest_bend", 0, x, fly["abd_rest_bend_deg"]),
             ("mount_pitch_vs_abdomen_rest_bend", 1, x, centred_tip),
             ("mount_pitch_vs_abdomen_activity", 0, x, fly["abd_bend_sd_pre_deg"]),
             ("mount_pitch_vs_abdomen_activity", 1, x, 100 * fly["tip_moving_frac_pre"]),
             ("abdomen_vs_spontaneous_per", 0, fly["abd_curl_rate_pre_per_min"], fly["spont_per_rate_per_min"])]
    for name, panel, xs, ys in cases:
        done = xs.notna() & ys.notna()
        assert points(figures[name].axes[panel]) == expected_points(xs[done], ys[done]), (name, panel)


def test_the_strip_plots_show_every_fly_with_its_value(figures, data):
    fly = data.fly
    ys = [y for _, y in points(figures["abdomen_rest_bend_per_fly"].axes[0])]
    assert sorted(ys) == sorted(np.round(fly["abd_rest_bend_deg"].dropna(), 6).tolist())
    have = fly[fly["per_abdomen_coupling_rho"].notna() & fly["proboscis_state_dominant"].notna()]
    ys = [y for _, y in points(figures["per_abdomen_coupling"].axes[0])]
    assert sorted(ys) == sorted(np.round(have["per_abdomen_coupling_rho"], 6).tolist())
    low, high = figures["per_abdomen_coupling"].axes[0].get_ylim()
    assert low == pytest.approx(-high) and high >= max(abs(v) for v in ys)     # zero in the middle: the sign is the story
    for name, slots in (("abdomen_rest_bend_per_fly", len(data.stats["rest_bend"]["rigs"])),
                        ("per_abdomen_coupling", len(data.stats["coupling"]["groups"]))):
        ax = figures[name].axes[0]
        assert ax.get_xlim() == pytest.approx((-0.6, slots - 0.4)), name              # every slot, with a margin
        zero = [line for line in ax.lines if len(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), 0.0)]
        assert zero, name                                                             # 0 = straight / no coupling


def test_the_jitter_of_a_strip_follows_the_seed_and_nothing_else(data):
    a = AM.build_abdomen_figures(data.ft, data.fly, AM.compute_abdomen_stats(data.ft, data.fly, n_perm=N_PERM, seed=0))
    b = AM.build_abdomen_figures(data.ft, data.fly, AM.compute_abdomen_stats(data.ft, data.fly, n_perm=N_PERM, seed=3))
    pa, pb = points(a["abdomen_rest_bend_per_fly"].axes[0]), points(b["abdomen_rest_bend_per_fly"].axes[0])
    assert [x for x, _ in pa] != [x for x, _ in pb]                                  # another jitter ...
    assert sorted(round(y, 6) for _, y in pa) == sorted(round(y, 6) for _, y in pb)  # ... of the same values


def test_the_bars_are_drawn_at_the_pooled_rates_of_the_stats(figures, data):
    for ax, event in zip(figures["abdomen_curl_events_by_window"].axes, ("curl", "straighten")):
        cells = data.stats["events"][event]
        expected = [cells["odor"]["placebo"]["rate"], cells["odor"]["during"]["rate"],
                    cells["light"]["placebo"]["rate"], cells["light"]["during"]["rate"]]
        assert bar_heights(ax) == pytest.approx(expected)


def test_a_bar_value_is_written_at_its_tip_and_the_p_of_each_pair_above_it(figures, data):
    ax = figures["abdomen_curl_events_by_window"].axes[0]
    cells = data.stats["events"]["curl"]
    tips = sorted(t.get_text() for t in ax.texts if re.fullmatch(r"\d+(\.\d+)?", t.get_text()))
    assert tips == sorted(FIG.num_text(v) for c in cells.values() for v in (c["placebo"]["rate"], c["during"]["rate"]))
    text = squash(" ".join(t.get_text() for t in ax.texts))
    for cell in cells.values():
        assert f"{FIG.p_text(cell['p_perm'])} (uncorrected)" in text


def test_bars_are_thin(figures):
    fig = figures["abdomen_curl_events_by_window"]
    fig.canvas.draw()
    for ax in fig.axes:
        for bar in ax.patches:
            assert bar.get_window_extent().width / fig.dpi * 72 <= 18.5              # 24 CSS px = 18 pt


def test_both_event_panels_share_one_scale_that_leaves_room_for_the_notes(figures, data):
    ev = data.stats["events"]
    tallest = max(c[w]["rate"] for e in ("curl", "straighten") for c in ev[e].values() for w in ("placebo", "during"))
    for ax in figures["abdomen_curl_events_by_window"].axes:
        assert ax.get_ylim() == (0.0, pytest.approx(1.4 * tallest))


def test_every_point_lies_inside_its_axes(figures):
    for name in ("abdomen_rest_bend_per_fly", "mount_pitch_vs_abdomen_rest_bend", "abdomen_vs_spontaneous_per",
                 "per_abdomen_coupling", "mount_pitch_vs_abdomen_activity"):
        for ax in figures[name].axes:
            (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
            marks = [c.get_offsets() for c in ax.collections if isinstance(c, PathCollection)]
            assert marks, name
            for xy in marks:
                assert (xy[:, 0] >= x0).all() and (xy[:, 0] <= x1).all(), name
                assert (xy[:, 1] >= y0).all() and (xy[:, 1] <= y1).all(), name


def test_a_pair_with_fewer_than_three_flies_says_so_instead_of_giving_a_p():
    ft = trial_table([windows(f"fly{k + 1}", 0, 3, trial_index=t) for k in range(2) for t in range(12)])
    fly = fly_table(ft)
    stats = AM.compute_abdomen_stats(ft, fly, n_perm=2000)
    assert stats["events"]["curl"]["odor"]["p_perm"] < 0.05                         # the p alone would have said "Yes"
    text = all_text(AM.build_abdomen_figures(ft, fly, stats)["abdomen_curl_events_by_window"])
    assert "fewer than 3 flies" in text and "Curls at the start of the odor: Not enough data to tell" in text
    assert not re.search(r"\bp[=<]", text)


def test_no_text_overflows_the_figure_or_overlaps_other_text(data):
    for name, fig in AM.iter_abdomen_figures(data.ft, data.fly, data.stats):      # measured as it is saved: the style is on
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
        AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
        AM.build_abdomen_figures(data.ft, data.fly, data.stats)
        assert {k: matplotlib.rcParams[k] for k in sentinel} == sentinel


def test_the_figures_are_drawn_in_the_lab_style_with_the_ascii_minus_of_the_text(data):
    negatives = 0
    for name, fig in AM.iter_abdomen_figures(data.ft, data.fly, data.stats):      # the style is on while a figure is held
        text = all_text(fig)
        assert "\u2212" not in text, name                                      # axes.unicode_minus is off, as in the text
        negatives += len(re.findall(r"(?<![\w.])-\d", text))
    assert negatives > 5                                                             # so the check had something to check


def test_no_figure_defines_a_second_colour_ramp(figures):
    palette = {BLUE, ORANGE, FIG.SURFACE, FIG.INK, FIG.INK_SECONDARY, FIG.MUTED, FIG.GRID, FIG.AXIS}
    for name, fig in figures.items():
        for ax in fig.axes:
            assert not ax.images, name
            for artist in [*ax.collections, *ax.patches]:
                assert getattr(artist, "get_array", lambda: None)() is None, name        # nothing is colour-mapped
                colours = np.atleast_2d(artist.get_facecolor()).tolist() + np.atleast_2d(artist.get_edgecolor()).tolist()
                assert {to_hex(c) for c in colours if len(c) and c[3] > 0} <= palette, (name, artist)
            assert {to_hex(line.get_color()) for line in ax.lines} <= palette, name


def test_flies_that_share_a_point_are_labelled_with_how_many_sit_there():
    rows = []
    for k, (curl, per) in enumerate([(0, 0), (0, 0), (0, 0), (0, 0), (2, 1), (3, 2), (1, 3), (0, 1)]):
        for t in range(2):
            rows.append(row(f"fly{k + 1}", trial_index=t, pre_min=0.5, abd_quality_ok=True, abd_curl_events_pre=curl * 0.5,
                            spont_per_events=per * 0.5, spont_per_any=float(per > 0)))
    ft = trial_table(rows)
    fly = fly_table(ft)
    figs = AM.build_abdomen_figures(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM))
    ax = figs["abdomen_vs_spontaneous_per"].axes[0]
    counts = [t.get_text() for t in ax.texts if re.fullmatch(r"×\d+", t.get_text())]
    assert counts == ["×4"]                                                        # only a shared point is labelled


def test_a_figure_without_enough_flies_is_a_placeholder_that_says_why_and_keeps_its_n_label():
    ft = abd_cohort(seed=3, per_cell=1, flaws=False)
    ft = ft[ft["fly_id"].isin(ft["fly_id"].unique()[:2])]
    fly = fly_table(ft)
    stats = AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM)
    assert set(stats["placeholders"]) >= {"mount_pitch_vs_abdomen_rest_bend", "abdomen_vs_spontaneous_per",
                                          "mount_pitch_vs_abdomen_activity"}
    figs = AM.build_abdomen_figures(ft, fly, stats)
    assert tuple(figs) == NAMES
    text = all_text(figs["abdomen_vs_spontaneous_per"])
    assert "Not enough data" in text and "n=2 flies" in text and "at least 3" in text
    assert figs["abdomen_vs_spontaneous_per"].axes == []                             # no axes, so nothing plotted


def test_the_descriptive_figures_are_drawn_for_a_single_fly_and_the_scatters_need_three():
    full = abd_cohort(seed=3, per_cell=1, flaws=False)                              # four flies, one per dataset x rig
    for n, scatters_drawn in ((1, False), (2, False), (3, True)):
        ft = full[full["fly_id"].isin(full["fly_id"].unique()[:n])]
        holes = AM.compute_abdomen_stats(ft, fly_table(ft), n_perm=N_PERM)["placeholders"]
        assert "abdomen_rest_bend_per_fly" not in holes and "per_abdomen_coupling" not in holes, n
        for name in ("mount_pitch_vs_abdomen_rest_bend", "abdomen_vs_spontaneous_per", "mount_pitch_vs_abdomen_activity"):
            assert (name not in holes) == scatters_drawn, (n, name)


def test_the_event_figure_is_a_placeholder_when_no_odor_or_light_trial_has_both_windows():
    ft = abd_cohort(seed=3, per_cell=3, flaws=False)
    ft["stim_kind"] = "none"
    fly = fly_table(ft)
    stats = AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM)
    assert "abdomen_curl_events_by_window" in stats["placeholders"] and stats["events"]["n_other"] == len(ft)
    fig = AM.build_abdomen_figures(ft, fly, stats)["abdomen_curl_events_by_window"]
    text = all_text(fig)
    assert fig.axes == [] and "Not enough data" in text and "n=0 trials / 0 flies" in text
    assert "resting bend from its fly's other trials, a just-before window and a start-of-stimulus window" in text


def test_a_two_panel_figure_with_one_empty_panel_keeps_both_panels():
    ft = abd_cohort(seed=5, per_cell=3, bend=0.5)
    ft["fly_tip_pitch_rest_deg"] = np.nan
    ft["tip_pitch_rest_deg"] = np.nan
    fly = fly_table(ft)
    stats = AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM)
    assert stats["tip_vs_pitch"]["n_flies"] == 0 and stats["bend_vs_pitch"]["n_flies"] >= 3
    assert "mount_pitch_vs_abdomen_rest_bend" not in stats["placeholders"]
    fig = AM.build_abdomen_figures(ft, fly, stats)["mount_pitch_vs_abdomen_rest_bend"]
    text = all_text(fig)
    assert len(fig.axes) == 2 and "Resting tip pitch: Not enough data to tell" in text and "n=0 flies" in text


def test_a_cohort_with_only_unknown_states_still_renders_every_figure(tmp_path):
    ft = abd_cohort(seed=5, per_cell=3)
    ft[["held_subtype", "spont_per_any", "spont_per_events"]] = np.nan
    fly = fly_table(ft)
    written = AM.render_abdomen_figures(ft, fly, tmp_path, n_perm=N_PERM)
    assert tuple(written) == NAMES and all(p.is_file() for p in written.values())


def test_a_single_rig_single_fly_cohort_does_not_crash_the_figures(tmp_path):
    ft = trial_table([windows("fly1", 1, 2, trial_body_pitch_deg=4.0, abd_rest_bend_deg=3.0,
                              fly_abd_rest_bend_deg=3.0, per_abdomen_coupling_rho=0.2, held_subtype="retracted",
                              spont_per_events=0.0, spont_per_any=0.0)])
    written = AM.render_abdomen_figures(ft, fly_table(ft), tmp_path, n_perm=N_PERM)
    assert tuple(written) == NAMES


def test_one_failing_figure_is_logged_with_its_traceback_and_does_not_stop_the_others(data, tmp_path, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setitem(AM.BUILDERS, "abdomen_vs_spontaneous_per", boom)
    with caplog.at_level(logging.WARNING):
        written = AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM)
    assert tuple(written) == tuple(n for n in NAMES if n != "abdomen_vs_spontaneous_per")
    assert "abdomen_vs_spontaneous_per" in caplog.text and "RuntimeError: synthetic failure" in caplog.text
    assert "[ABDOMEN]" in caplog.text and "Traceback (most recent call last)" in caplog.text
    assert {r.name for r in caplog.records if "failed" in r.getMessage()} == {"abdomen_movement_figures"}
    assert not (tmp_path / "abdomen_vs_spontaneous_per.png").exists()


def test_a_figure_that_fails_to_build_leaves_no_stale_picture_and_the_section_names_it(data, tmp_path, monkeypatch):
    first = AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    stale = tmp_path / "abdomen_vs_spontaneous_per.png"
    assert tuple(first) == NAMES and stale.is_file()                                # last run's picture is on disk

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setitem(AM.BUILDERS, "abdomen_vs_spontaneous_per", boom)
    again = AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    assert tuple(again) == tuple(n for n in NAMES if n != "abdomen_vs_spontaneous_per")
    assert not stale.exists() and sorted(p.stem for p in tmp_path.glob("*.png")) == sorted(again)
    text = AM.abdomen_report_section(data.ft, data.fly, data.stats, figures=again)
    assert "Figures missing from this folder:** `abdomen_vs_spontaneous_per.png`" in text
    monkeypatch.undo()
    fixed = AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    assert tuple(fixed) == NAMES and stale.is_file()
    assert "Figures missing" not in AM.abdomen_report_section(data.ft, data.fly, data.stats, figures=fixed)


def test_a_section_written_without_the_list_of_figures_makes_no_claim_about_them(data):
    assert "Figures missing" not in AM.abdomen_report_section(data.ft, data.fly, data.stats)


def test_rendering_removes_only_the_six_abdomen_figures_of_a_previous_run(data, tmp_path):
    (tmp_path / "notes.png").write_bytes(b"mine")
    (tmp_path / "mount_pitch_per_fly.png").write_bytes(b"a mounting figure")
    (tmp_path / "report.md").write_text("old", encoding="utf-8")
    AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    assert (tmp_path / "notes.png").read_bytes() == b"mine" and (tmp_path / "report.md").read_text(encoding="utf-8") == "old"
    assert (tmp_path / "mount_pitch_per_fly.png").read_bytes() == b"a mounting figure"


def test_empty_input_draws_no_figure_creates_nothing_and_removes_the_pictures_of_an_earlier_run(data, tmp_path, caplog):
    ft, fly = trial_table([]), pd.DataFrame(columns=MP.FLY_COLUMNS)
    with caplog.at_level(logging.WARNING):
        assert AM.render_abdomen_figures(ft, fly, tmp_path / "out", n_perm=N_PERM) == {}
    assert not (tmp_path / "out").exists() and caplog.text == ""
    AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    (tmp_path / "notes.png").write_bytes(b"mine")
    assert AM.render_abdomen_figures(ft, fly, tmp_path, n_perm=N_PERM) == {}
    assert [p.name for p in tmp_path.glob("*.png")] == ["notes.png"]              # the six are gone, nothing else is touched


def test_an_input_without_a_usable_abdomen_window_draws_nothing_and_removes_the_pictures_of_an_earlier_run(data, tmp_path):
    AM.render_abdomen_figures(data.ft, data.fly, tmp_path, n_perm=N_PERM, stats=data.stats)
    assert len(list(tmp_path.glob("*.png"))) == 6
    ft = data.ft.copy()
    ft[ABDOMEN_COLUMNS] = np.nan
    ft[["abd_quality_ok", "abd_quality_ok_stim"]] = False
    fly = fly_table(ft)
    assert AM.render_abdomen_figures(ft, fly, tmp_path, n_perm=N_PERM) == {}
    assert list(tmp_path.glob("*.png")) == []
    assert AM.render_abdomen_figures(ft, fly, tmp_path / "fresh", n_perm=N_PERM) == {} and not (tmp_path / "fresh").exists()


# ---------------------------------------------------------------------------
# The "Abdomen movement" section of report.md
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def section(data):
    return AM.abdomen_report_section(data.ft, data.fly, data.stats)


def test_the_section_is_a_second_level_heading_with_the_questions_below_it(section):
    assert section.startswith("## Abdomen movement")
    for name in NAMES:
        assert f"{name}.png" in section
    for title in AM.TITLES.values():
        assert f"### {title}" in section


def test_the_section_lists_the_pooled_rates_per_window_with_trials_and_flies(section, data):
    for event in ("curl", "straighten"):
        for kind, cell in data.stats["events"][event].items():
            assert FIG.n_label(cell["n_trials"], cell["n_flies"]) in section, (event, kind)
            for window in ("placebo", "during"):
                w = cell[window]
                assert f"{w['events']:g} / {w['minutes']:.2f} min = {w['rate']:.2f}/min" in section, (event, kind, window)


def test_the_section_gives_rho_and_p_for_every_per_fly_scatter(section, data):
    for key in ("bend_vs_pitch", "tip_vs_pitch", "restless"):
        s = data.stats[key]
        assert f"ρ={s['rho']:.3f}" in section, key
        assert FIG.p_text(s["p"], digits=4) in section, key
    for key in ("bend_sd", "tip_moving"):
        s = data.stats["activity"][key]
        assert f"ρ={s['rho']:.3f}" in section and FIG.p_text(s["p"], digits=4) in section, key


def test_the_section_gives_every_permutation_p_and_what_was_shuffled(section, data):
    s = data.stats
    for p in (s["bend_vs_pitch"]["p_perm"], s["tip_vs_pitch"]["p_perm"], s["restless"]["p_perm"],
              s["activity"]["bend_sd"]["p_perm"], s["activity"]["tip_moving"]["p_perm"]):
        assert FIG.p_text(p, digits=4) in section
    for event in ("curl", "straighten"):
        for cell in s["events"][event].values():
            assert FIG.p_text(cell["p_perm"], digits=4) in section
    assert re.search(rf"Stratified permutation p: .*dataset × rig.*{N_PERM} permutations, seed 0", section)
    assert "curl rate shuffled within dataset × rig strata" in section                 # what each scatter shuffled
    assert section.count("mounting angle shuffled within dataset × rig strata") == 4
    assert re.search(rf"Window-swap permutation p: .*labels of a trial's two windows.*{N_PERM} permutations, seed 0",
                     section)
    assert section.count("uncorrected") >= 12


def test_the_section_counts_the_windows_the_tracked_fraction_gate_dropped(section, data):
    w = data.stats["windows"]
    assert all(w[k]["dropped"] > 0 for k in ("pre", "base", "placebo", "during"))    # the cohort has poorly tracked windows
    assert f"{w['pre']['dropped']} of {w['pre']['with_window']} pre-stimulus windows" in section
    assert f"{w['base']['dropped']} of {w['base']['with_window']} baseline windows" in section
    assert f"{w['placebo']['dropped']} of {w['placebo']['with_window']} just-before windows" in section
    assert f"{w['during']['dropped']} of {w['during']['with_window']} start-of-stimulus windows" in section
    assert "abd_min_tracked_frac" in section
    stats = AM.compute_abdomen_stats(data.ft, data.fly, n_perm=N_PERM, min_tracked_frac=0.5)
    assert "tracked in fewer than 50% of their frames" in AM.abdomen_report_section(data.ft, data.fly, stats)


def test_the_section_describes_the_placebo_design_in_plain_words(section):
    assert "L is the shorter of the stimulus and half the pre-stimulus window" in section
    assert "median bend over the early baseline windows" in section
    assert "of the fly's OTHER trials, pooled over frames" in section
    assert "the trial itself and every trial's just-before and start-of-stimulus frames are left out" in section
    assert "in the last L s before the stimulus" in section and "in the first L s of the stimulus" in section
    assert "differ only by chance" not in section and "Event-split" not in section


def test_the_section_states_the_stationarity_assumption_and_the_trials_a_fly_needs(section):
    assert ("Assumption (within-trial stationarity): with no stimulus effect the two windows of a trial are "
            "interchangeable") in section
    assert "A fly needs at least 3 trials with a known stimulus onset" in section
    assert "at least 2 of the fly's other trials have a usable baseline window" in section
    assert "flies whose identity is not consistent are left out" in section


def test_the_section_says_the_window_swap_is_trial_level_and_names_its_limitation(section):
    assert "Window-swap permutation p" in section and "trial-level" in section
    assert "trials, not flies, are shuffled" in section and "treats trials as independent" in section
    assert "Limitation: a bend that drifts steadily inside the trial breaks the assumption above" in section
    assert "the more trials are pooled the more a small drift shows as a difference" in section


def test_the_events_table_names_its_two_windows_and_the_test(section):
    header = next(line for line in section.splitlines() if line.startswith("| Stimulus | Event |"))
    assert header == ("| Stimulus | Event | Just before the stimulus | Start of the stimulus | Difference | "
                      "Trials / flies | Window-swap permutation p |")


def test_the_section_says_the_alpha_the_other_stimulus_trials_and_has_a_heading_for_the_windows(section, data):
    assert "below 0.05" in section and "### Windows dropped for poor tracking" in section
    assert data.stats["events"]["n_other"] > 0 and "with another stimulus" in section


def test_the_section_has_a_row_for_every_stimulus_and_state_even_when_there_is_nothing_in_it():
    ft = trial_table([windows(f"fly{k + 1}", 1, 2, trial_index=t, straighten=(1, 1),
                              held_subtype="retracted" if k < 3 else "held_down", per_abdomen_coupling_rho=0.1 * k)
                      for k in range(4) for t in range(3)])
    fly = fly_table(ft)
    text = AM.abdomen_report_section(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM))
    assert "| light | Curls | no trial | no trial | n/a | n=0 trials / 0 flies | n/a |" in text
    assert "| light | Straightenings | no trial |" in text
    assert "| Held back | n=0 flies | 0 | n/a | n/a |" in text


def test_the_section_defines_the_signs_and_the_events(section):
    assert "0 = straight" in section and "curled toward the proboscis" in section
    assert "sustained" in section and "curl_rise_deg" in section


def test_the_section_says_one_fly_for_a_single_fly_and_one_stratum_for_a_single_stratum():
    ft = trial_table([windows("fly1", 1, 2, trial_index=t, trial_body_pitch_deg=1.0, abd_rest_bend_deg=3.0,
                              fly_abd_rest_bend_deg=3.0) for t in range(4)])
    fly = fly_table(ft)
    text = AM.abdomen_report_section(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM))
    assert re.search(r"\b1 fly\b", text) and re.search(r"\b1 stratum\b", text)
    assert not re.search(r"\b1 flies\b|\b1 strata\b|\b1 trials\b", text)


def test_a_placeholder_bullet_is_followed_by_a_blank_line_so_markdown_keeps_the_next_block_apart():
    two = abd_cohort(seed=3, per_cell=1, flaws=False)
    two = two[two["fly_id"].isin(two["fly_id"].unique()[:2])]
    none = abd_cohort(seed=3, per_cell=3, flaws=False).assign(stim_kind="none")
    seen = 0
    for ft in (two, none):
        fly = fly_table(ft)
        text = AM.abdomen_report_section(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM))
        for match in re.finditer(r"- \*\*Placeholder, not drawn:\*\*[^\n]*\n", text):
            seen += 1
            assert text[match.end()] == "\n", text[match.start():match.end() + 60]
    assert seen >= 4                                                    # curl / tip pitch, activity, restless; the events


def test_the_section_counts_a_single_blank_window_in_the_singular():
    ft = trial_table([windows("fly1", 1, 1, trial_index=0), windows("fly1", 1, 1, trial_index=1),
                      windows("fly1", 1, 1, trial_index=2, base=False, abd_tracked_frac_placebo=0.9,
                              abd_tracked_frac_during=np.nan)])
    fly = fly_table(ft)
    text = AM.abdomen_report_section(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM))
    assert "1 more window was blank because the fly had fewer than 2 other trials with a usable baseline window" in text
    assert "windows were blank" not in text
    more = trial_table([windows("fly1", 1, 1, trial_index=t, base=False) for t in range(2)]
                       + [windows("fly1", 1, 1, trial_index=9)])
    text = AM.abdomen_report_section(more, fly_table(more), AM.compute_abdomen_stats(more, fly_table(more), n_perm=N_PERM))
    assert "2 more windows were blank because" in text


def test_the_section_marks_a_placeholder_figure_and_why():
    ft = abd_cohort(seed=3, per_cell=1, flaws=False)
    ft = ft[ft["fly_id"].isin(ft["fly_id"].unique()[:2])]
    fly = fly_table(ft)
    text = AM.abdomen_report_section(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM))
    assert "Placeholder, not drawn" in text and "at least 3" in text


def test_the_section_of_an_input_without_a_usable_window_says_there_is_no_abdomen_data(data):
    ft = data.ft.copy()
    ft[ABDOMEN_COLUMNS] = np.nan
    ft[["abd_quality_ok", "abd_quality_ok_stim"]] = False
    fly = fly_table(ft)
    text = AM.abdomen_report_section(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM, min_tracked_frac=0.5),
                                     figures={})
    assert text.startswith("## Abdomen movement") and "No abdomen data in scope" in text
    assert f"{len(ft)} fly-trials" in text and "abd_min_tracked_frac" in text and "dropped" in text
    assert "Figures missing" not in text and "###" not in text                       # no empty analyses are described


def test_a_section_without_the_list_of_figures_does_not_claim_that_pictures_were_removed(data):
    ft = data.ft.copy()
    ft[ABDOMEN_COLUMNS] = np.nan
    ft[["abd_quality_ok", "abd_quality_ok_stim"]] = False
    fly = fly_table(ft)
    stats = AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM)
    assert "removed" not in AM.abdomen_report_section(ft, fly, stats)
    assert "pictures of an earlier run were removed" in AM.abdomen_report_section(ft, fly, stats, figures={})


def test_the_section_of_an_empty_input_says_nothing_is_in_scope():
    ft, fly = trial_table([]), pd.DataFrame(columns=MP.FLY_COLUMNS)
    text = AM.abdomen_report_section(ft, fly, AM.compute_abdomen_stats(ft, fly, n_perm=N_PERM), figures={})
    assert "No abdomen data in scope" in text and "no new-glue fly-trials" in text


def test_the_section_goes_into_the_report_after_the_mounting_section(data, section, tmp_path):
    stats = FIG.compute_stats(data.ft, data.fly, n_perm=N_PERM)
    path = FIG.write_report(data.ft, data.fly, {"n_fly_trials": len(data.ft)}, stats, tmp_path, extra_sections=[section])
    text = path.read_text(encoding="utf-8")
    assert text.index("## Mounting") < text.index("## Abdomen movement") and "uncorrected" in text


def test_an_abdomen_analysis_that_fails_costs_nothing_else(data, tmp_path, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    (tmp_path / "abdomen_rest_bend_per_fly.png").write_bytes(b"stale")
    monkeypatch.setattr(AM, "compute_abdomen_stats", boom)
    with caplog.at_level(logging.WARNING):
        text, figures = AM.abdomen_outputs(data.ft, data.fly, tmp_path, n_perm=N_PERM)
    assert figures == {} and list(tmp_path.glob("*.png")) == []                      # the stale picture is gone as well
    assert text.startswith("## Abdomen movement") and "could not be computed" in text
    assert "RuntimeError: synthetic failure" in text and "RuntimeError: synthetic failure" in caplog.text


def test_the_outputs_are_the_section_and_the_figures_that_were_written(data, tmp_path):
    text, figures = AM.abdomen_outputs(data.ft, data.fly, tmp_path / "out", n_perm=N_PERM)
    assert tuple(figures) == NAMES and text.startswith("## Abdomen movement") and "Figures missing" not in text
    text, figures = AM.abdomen_outputs(data.ft, data.fly, None, n_perm=N_PERM)
    assert figures is None and "## Abdomen movement" in text and "Figures missing" not in text
    assert "could not be computed" not in text                                       # nothing was asked of a folder


def test_the_outputs_name_a_figure_that_failed_to_build(data, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setitem(AM.BUILDERS, "per_abdomen_coupling", boom)
    text, figures = AM.abdomen_outputs(data.ft, data.fly, tmp_path, n_perm=N_PERM)
    assert "per_abdomen_coupling" not in figures and len(figures) == 5
    assert "Figures missing from this folder:** `per_abdomen_coupling.png`" in text


def test_the_gate_value_reaches_the_section_through_the_outputs(data, tmp_path):
    text, _ = AM.abdomen_outputs(data.ft, data.fly, tmp_path, n_perm=N_PERM, min_tracked_frac=0.4)
    assert "tracked in fewer than 40% of their frames" in text
