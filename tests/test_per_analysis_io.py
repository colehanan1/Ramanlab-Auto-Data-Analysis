"""per_analysis data layer: label parsing, clock windows, the tidy-table builder, its
validation and cache, and one integration test on the real data (skipped off-site)."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from per_analysis import config as C
from per_analysis import io

THR = 2
WINDOWS = C.DEFAULT_WINDOWS
COHORTS = {"IAA-Sensitivity-1-1": "exp1", "IAA-PairFirst-Control-1-1": "exp2",
           "IAA-PairSecond-Control-1-1": "exp3"}
E1, E2, E3 = "IAA-Sensitivity-1-1", "IAA-PairFirst-Control-1-1", "IAA-PairSecond-Control-1-1"
F1, F2, F3 = "november_01_batch_1_rig_2", "november_17_batch_2", "november_13_batch_1_rig_3"


# ── label / name parsing ─────────────────────────────────────────────────────
@pytest.mark.parametrize("label, phase, idx, odors, light, odor2, base", [
    ("pretest_3_isoamylacetate-1", "pretest", 3, ("IAA",), False, False,
     "pretest_3_isoamylacetate-1"),
    ("training_2_3-octonol-0.1", "training", 2, ("3-Octanol",), False, False,
     "training_2_3-octonol-0.1"),
    ("testing_8_paraffinoil-100", "testing", 8, ("Paraffin Oil",), False, False,
     "testing_8_paraffinoil-100"),
    ("testing_5_ethylbutyrate-1", "testing", 5, ("Ethyl Butyrate",), False, False,
     "testing_5_ethylbutyrate-1"),
    ("testing_4_isoamylacetate-1_then_hexanol-0.1", "testing", 4, ("IAA", "Hexanol"), False,
     False, "testing_4_isoamylacetate-1_then_hexanol-0.1"),
    ("testing_4_isoamylacetate-1_then_hexanol-0.1__odor2", "testing", 4, ("IAA", "Hexanol"),
     False, True, "testing_4_isoamylacetate-1_then_hexanol-0.1"),
    ("testing_12_benzaldehyde-0.1_then_isoamylacetate-1__odor2", "testing", 12,
     ("Benzaldehyde", "IAA"), False, True, "testing_12_benzaldehyde-0.1_then_isoamylacetate-1"),
    ("testing_17_lightonly", "testing", 17, (), True, False, "testing_17_lightonly"),
    ("testing_9_LightOnly", "testing", 9, (), True, False, "testing_9_LightOnly"),
    ("pretest_6_citral-1", "pretest", 6, ("Citral",), False, False, "pretest_6_citral-1"),
    ("pretest_7_linalool-1", "pretest", 7, ("Linalool",), False, False, "pretest_7_linalool-1"),
])
def test_parse_trial_label(label, phase, idx, odors, light, odor2, base):
    p = io.parse_trial_label(label)
    assert (p.phase, p.trial_index, p.odors, p.is_light_probe, p.is_odor2, p.base_label) == \
        (phase, idx, odors, light, odor2, base)


@pytest.mark.parametrize("label", ["", "foo", "testing_x_iaa", "trial_1_hexanol-0.1"])
def test_parse_trial_label_rejects(label):
    assert io.parse_trial_label(label) is None


def test_raw_file_tokens_map_like_table_labels():
    # the rig writes CamelCase; lower-cased it is the table's label
    assert io.canonical_odor("IsoamylAcetate-1") == "IAA"
    assert io.canonical_odor("3-Octonol-0.1") == "3-Octanol"
    assert io.canonical_odor("ParaffinOil-100") == "Paraffin Oil"
    assert io.canonical_odor("EthylButyrate-1") == "Ethyl Butyrate"


def test_names_mirror_b2b_pair_figures():
    b2b = pytest.importorskip("scripts.analysis.b2b_pair_figures")
    assert io.NAMES == b2b.NAMES
    assert set(b2b.ORDER) == set(C.ODOR_ORDER)


def test_parse_raw_filename():
    got = io.parse_raw_filename(
        "output_november_17_batch_2_testing_10_IsoamylAcetate-1_then_3-Octonol-0.1_"
        "20261117_201438.csv")
    assert got == ("testing_10_isoamylacetate-1_then_3-octonol-0.1",
                   dt.datetime(2026, 11, 17, 20, 14, 38))
    assert io.parse_raw_filename("output_x_testing_17_LightOnly_20261117_204600.csv") == \
        ("testing_17_lightonly", dt.datetime(2026, 11, 17, 20, 46))
    assert io.parse_raw_filename("output_x_testing_1_Citral-1_20261117_204600.txt") is None
    assert io.parse_raw_filename("odor_vials.csv") is None


@pytest.mark.parametrize("name, odor, protocol, starv, conc", [
    ("3Oct-Sensitivity-1-0.1", "3-Octanol", "Sensitivity", 1, 0.1),
    ("Hex-Sensitivity-1-0.1", "Hexanol", "Sensitivity", 1, 0.1),
    ("Paraffin-Sensitivity-1-100", "Paraffin Oil", "Sensitivity", 1, 100.0),
    ("IAA-PairFirst-Control-1-1", "IAA", "PairFirst-Control", 1, 1.0),
])
def test_parse_dataset(name, odor, protocol, starv, conc):
    p = io.parse_dataset(name)
    assert (p["name_odor"], p["protocol"], p["starvation_h"], p["conc"]) == \
        (odor, protocol, starv, conc)


@pytest.mark.parametrize("folder, rig, session, host", [
    ("november_17_batch_2", 1, "november_17_batch_2", "Flybehavior2"),
    ("november_17_batch_2_rig_2", 2, "november_17_batch_2", "BehaviorLocust"),
    ("november_13_batch_1_rig_3", 3, "november_13_batch_1", "Flybehavior3"),
])
def test_folder_identity(folder, rig, session, host):
    assert (io.rig_of(folder), io.session_of(folder), io.host_of(io.rig_of(folder))) == \
        (rig, session, host)


# ── clock windows ─────────────────────────────────────────────────────────────
def test_parse_clock_forms():
    assert C.parse_clock("16:40") == dt.time(16, 40)
    assert C.parse_clock(1000) == dt.time(16, 40)       # YAML 1.1 sexagesimal 16:40
    assert C.parse_clock(580) == dt.time(9, 40)
    assert C.parse_clock(dt.time(9, 40)) == dt.time(9, 40)
    with pytest.raises(ValueError):
        C.parse_clock("late")


def test_windows_from_unquoted_yaml():
    text = ("windows:\n- {out_dir: /r/b2b, start: 09:40, end: 16:20}\n"
            "- {out_dir: /r/b2b-Evening, start: 16:40, end: 21:20}\n")
    entries = yaml.safe_load(text)["windows"]
    assert entries[1]["start"] == 1000                  # what an unquoted 16:40 becomes
    w = C.windows_from_entries(entries)
    assert [(x.name, x.start, x.end) for x in w] == [
        ("day", dt.time(9, 40), dt.time(16, 20)), ("evening", dt.time(16, 40), dt.time(21, 20))]


def test_live_config_windows_and_threshold():
    if not C.CONFIG_YAML.exists():
        pytest.skip("config_new.yaml not on disk")
    assert [w.name for w in C.load_windows()] == ["day", "evening"]
    assert C.load_threshold() == C.RESPONSE_THRESHOLD


@pytest.mark.parametrize("clock, name", [
    ("09:39:59", "other"), ("09:40:00", "day"), ("16:19:59", "day"),
    ("16:20:00", "other"),                                  # end exclusive
    ("16:40:00", "evening"), ("21:19:59", "evening"), ("21:20:00", "other"), ("03:00:00", "other"),
])
def test_assign_window(clock, name):
    ts = pd.Timestamp(f"2026-11-17 {clock}")
    assert io.assign_window(ts, WINDOWS) == name


def test_assign_window_unknown_start():
    assert io.assign_window(pd.NaT, WINDOWS) == "other"


# ── synthetic inputs ──────────────────────────────────────────────────────────
def _stamp(day, hms):
    return dt.datetime.strptime(f"2026-11-{day} {hms}", "%Y-%m-%d %H:%M:%S")


RAW = [  # dataset, folder, label, time
    (E1, F1, "pretest_1_isoamylacetate-1", _stamp("01", "10:00:00")),
    (E1, F1, "pretest_2_hexanol-0.1", _stamp("01", "10:05:00")),
    (E1, F1, "training_1_isoamylacetate-1", _stamp("01", "11:00:00")),
    (E1, F1, "training_2_isoamylacetate-1", _stamp("01", "11:01:30")),
    (E1, F1, "testing_1_hexanol-0.1", _stamp("01", "11:31:30")),
    (E1, F1, "testing_2_isoamylacetate-1", _stamp("01", "11:36:30")),
    (E1, F1, "testing_3_lightonly", _stamp("01", "11:38:00")),
    (E2, F2, "pretest_1_paraffinoil-100", _stamp("17", "18:24:34")),
    (E2, F2, "pretest_2_isoamylacetate-1", _stamp("17", "18:29:34")),
    (E2, F2, "testing_1_isoamylacetate-1_then_paraffinoil-100", _stamp("17", "19:00:00")),
    (E2, F2, "testing_2_isoamylacetate-1_then_isoamylacetate-1", _stamp("17", "19:05:00")),
    (E2, F2, "testing_3_isoamylacetate-1_then_paraffinoil-100", _stamp("17", "19:10:00")),
    (E2, F2, "testing_4_isoamylacetate-1_then_isoamylacetate-1", _stamp("17", "19:15:00")),
    (E2, F2, "testing_5_lightonly", _stamp("17", "19:20:00")),
    (E3, F3, "pretest_1_isoamylacetate-1", _stamp("13", "16:20:00")),
    (E3, F3, "testing_1_paraffinoil-100_then_isoamylacetate-1", _stamp("13", "17:00:00")),
    (E3, F3, "testing_2_paraffinoil-100_then_isoamylacetate-1", _stamp("13", "17:05:00")),
]
def _raw_frame():
    return pd.DataFrame([dict(dataset=d, folder=f, trial_label=l, trial_time=t, n_files=1,
                              root="/raw") for d, f, l, t in RAW])


def _meta_frame():
    rows = []
    for d, f, label, _t in RAW:
        flies = {F1: (1, 2), F2: (1,), F3: (3,)}[f]
        for n in flies:
            on = 30.0
            off = 40.0 if d != E1 else 60.0
            if d == E3 and label.startswith("pretest_1"):
                off = 30.025
            rows.append(dict(dataset=d, fly=f, fly_number=n, trial_type=label.split("_")[0],
                             trial_label=label, trial_odor_on_s=on, trial_odor_off_s=off,
                             trial_duration_s=90.0, fps=40.0, frozen=(f == F1 and n == 2)))
    return pd.DataFrame(rows)


SCORES = {  # (folder, fly, label) -> score; pair labels carry odor 1, '__odor2' odor 2
    (F1, 1, "pretest_1_isoamylacetate-1"): 0,
    (F1, 1, "pretest_2_hexanol-0.1"): 1,
    (F1, 1, "training_1_isoamylacetate-1"): 3,
    (F1, 1, "training_2_isoamylacetate-1"): 2,
    (F1, 1, "testing_1_hexanol-0.1"): 4,
    (F1, 1, "testing_2_isoamylacetate-1"): -1,
    (F1, 1, "testing_3_lightonly"): 5,
    (F2, 1, "pretest_1_paraffinoil-100"): 0,
    (F2, 1, "pretest_2_isoamylacetate-1"): 2,
    (F2, 1, "testing_1_isoamylacetate-1_then_paraffinoil-100"): 1,
    (F2, 1, "testing_1_isoamylacetate-1_then_paraffinoil-100__odor2"): 0,
    (F2, 1, "testing_2_isoamylacetate-1_then_isoamylacetate-1"): 5,
    (F2, 1, "testing_2_isoamylacetate-1_then_isoamylacetate-1__odor2"): 3,
    (F2, 1, "testing_3_isoamylacetate-1_then_paraffinoil-100"): 0,
    (F2, 1, "testing_3_isoamylacetate-1_then_paraffinoil-100__odor2"): 2,
    (F2, 1, "testing_4_isoamylacetate-1_then_isoamylacetate-1"): 1,
    (F2, 1, "testing_4_isoamylacetate-1_then_isoamylacetate-1__odor2"): 1,
    (F2, 1, "testing_5_lightonly"): 4,
    (F3, 3, "pretest_1_isoamylacetate-1"): 1,
    (F3, 3, "testing_1_paraffinoil-100_then_isoamylacetate-1"): 0,
    (F3, 3, "testing_1_paraffinoil-100_then_isoamylacetate-1__odor2"): 4,
    (F3, 3, "testing_2_paraffinoil-100_then_isoamylacetate-1"): 2,
    (F3, 3, "testing_2_paraffinoil-100_then_isoamylacetate-1__odor2"): 0,
}
DS_OF = {F1: E1, F2: E2, F3: E3}


def _pred_frame(extra=()):
    rows = [dict(dataset=DS_OF[f], fly=f, fly_number=n, trial_label=l, score=s)
            for (f, n, l), s in SCORES.items()]
    # the frozen fly 2 of F1 is scored too: it must be dropped by `frozen`
    rows.append(dict(dataset=E1, fly=F1, fly_number=2, trial_label="pretest_1_isoamylacetate-1",
                     score=0))
    rows += list(extra)
    return pd.DataFrame(rows)


PARAFFIN_OFF = frozenset({(E3, F3, "testing_2")})


def _build(meta=None, preds=None, raw=None, paraffin_off=PARAFFIN_OFF):
    return io.build_tidy(_meta_frame() if meta is None else meta,
                         _pred_frame() if preds is None else preds,
                         _raw_frame() if raw is None else raw,
                         cohorts=COHORTS, windows=WINDOWS, threshold=THR,
                         paraffin_off=paraffin_off)


@pytest.fixture(scope="module")
def tidy():
    return _build()


def _row(t, folder, label, n=None):
    sel = t[(t.batch == folder) & (t.trial_label == label)]
    if n is not None:
        sel = sel[sel.fly_id.str.endswith(f"|{n}")]
    assert len(sel) == 1, (folder, label, len(sel))
    return sel.iloc[0]


def test_contract_columns(tidy):
    assert list(tidy.columns[:len(io.CONTRACT_COLUMNS)]) == io.CONTRACT_COLUMNS


def test_frozen_rows_dropped(tidy):
    assert not tidy.fly_id.str.endswith(f"{F1}|2").any()
    assert tidy.fly_id.nunique() == 3


def test_one_row_per_scored_trial(tidy):
    # 7 exp1 + (2 alone + 4 pair + 1 light) exp2 + (1 alone + 2 pair) exp3
    assert len(tidy) == 7 + 7 + 3
    assert not tidy.duplicated(["fly_id", "trial_label", "is_odor2"]).any()


def test_identity(tidy):
    r = _row(tidy, F1, "pretest_1_isoamylacetate-1")
    assert (r.experiment, r.cohort, r.exposure_odor, r.exposure_conc, r.starvation_h) == \
        ("exp1", E1, "IAA", 1.0, 1)
    assert (r.fly_id, r.batch, r.session, r.rig, r.host) == \
        (f"{E1}|{F1}|1", F1, "november_01_batch_1", 2, "BehaviorLocust")
    r = _row(tidy, F2, "pretest_2_isoamylacetate-1")
    assert r.experiment == "exp2" and pd.isna(r.exposure_odor) and np.isnan(r.exposure_conc)
    assert (r.rig, r.host, r.starvation_h) == (1, "Flybehavior2", 1)


def test_blocks(tidy):
    got = tidy.groupby("experiment").block.agg(lambda s: sorted(set(s))).to_dict()
    assert got == {"exp1": ["exposure", "post", "pre"], "exp2": ["alone", "pair"],
                   "exp3": ["alone", "pair"]}


def test_window_and_start(tidy):
    w = tidy.groupby("batch").window.agg(set).to_dict()
    assert w == {F1: {"day"}, F2: {"evening"}, F3: {"other"}}     # F3 starts 16:20:00 sharp
    r = _row(tidy, F2, "testing_5_lightonly")
    assert r.start_time == pd.Timestamp("2026-11-17 18:24:34")
    assert r.start_time_h == pytest.approx(18 + 24 / 60 + 34 / 3600)
    assert r.date == pd.Timestamp("2026-11-17")
    assert r.trial_time == pd.Timestamp("2026-11-17 19:20:00")


def test_pair_odor_columns(tidy):
    r = _row(tidy, F2, "testing_1_isoamylacetate-1_then_paraffinoil-100")
    assert (r.odor, r.prime, r.partner, r.presentation, r.score, r.odor1_score) == \
        ("Paraffin Oil", "IAA", "Paraffin Oil", 1, 0, 1)
    assert r.is_odor2 and r.is_click and r.block == "pair"
    r = _row(tidy, F2, "testing_3_isoamylacetate-1_then_paraffinoil-100")
    assert (r.presentation, r.score, r.odor1_score, r.responded) == (2, 2, 0, True)
    r = _row(tidy, F2, "testing_4_isoamylacetate-1_then_isoamylacetate-1")
    assert (r.odor, r.partner, r.presentation, r.is_click) == ("IAA", "IAA", 2, False)
    # exp3: the partner is odor 1, the scored odor is IAA (odor 2)
    r = _row(tidy, F3, "testing_1_paraffinoil-100_then_isoamylacetate-1")
    assert (r.odor, r.prime, r.partner, r.presentation, r.score, r.odor1_score) == \
        ("IAA", "Paraffin Oil", "Paraffin Oil", 1, 4, 0)
    assert r.is_click


def test_alone_and_exp1_odor_columns(tidy):
    r = _row(tidy, F2, "pretest_1_paraffinoil-100")
    assert (r.block, r.odor, r.partner, r.is_click) == ("alone", "Paraffin Oil", "Paraffin Oil",
                                                          True)
    assert pd.isna(r.prime) and np.isnan(r.presentation) and np.isnan(r.odor1_score)
    r = _row(tidy, F1, "testing_1_hexanol-0.1")
    assert (r.block, r.odor, r.partner) == ("post", "Hexanol", "Hexanol")
    r = _row(tidy, F1, "training_2_isoamylacetate-1")
    assert (r.block, r.odor) == ("exposure", "IAA")


def test_light_probe(tidy):
    lp = tidy[tidy.is_light_probe]
    assert sorted(lp.trial_label) == ["testing_3_lightonly", "testing_5_lightonly"]
    assert lp.odor.isna().all() and lp.partner.isna().all()
    assert set(lp.block) == {"post", "pair"}


def test_position_in_block(tidy):
    f1 = tidy[tidy.batch == F1].set_index("trial_label").position_in_block.to_dict()
    assert f1 == {"pretest_1_isoamylacetate-1": 1, "pretest_2_hexanol-0.1": 2,
                  "training_1_isoamylacetate-1": 1, "training_2_isoamylacetate-1": 2,
                  "testing_1_hexanol-0.1": 1, "testing_2_isoamylacetate-1": 2,
                  "testing_3_lightonly": 3}
    pair = tidy[(tidy.batch == F2) & (tidy.block == "pair")].sort_values("trial_index")
    assert pair.position_in_block.tolist() == [1, 2, 3, 4, 5]


def test_position_in_block_uses_trial_order_not_row_order():
    t = _build(preds=_pred_frame().iloc[::-1].reset_index(drop=True))
    r = _row(t, F1, "testing_2_isoamylacetate-1")
    assert r.position_in_block == 2


def test_minutes_since_exposure_end(tidy):
    f1 = tidy[tidy.batch == F1].set_index("trial_label").minutes_since_exposure_end
    assert f1["testing_1_hexanol-0.1"] == pytest.approx(30.0)
    assert f1["testing_2_isoamylacetate-1"] == pytest.approx(35.0)
    assert f1["testing_3_lightonly"] == pytest.approx(36.5)
    assert f1[["pretest_1_isoamylacetate-1", "training_2_isoamylacetate-1"]].isna().all()
    assert tidy.loc[tidy.experiment != "exp1", "minutes_since_exposure_end"].isna().all()


def test_minutes_since_exposure_end_ignores_unscored_last_pulse():
    """The last pulse's stamp comes from the folder's raw file even when this fly's
    score for it is missing."""
    preds = _pred_frame()
    preds = preds[preds.trial_label != "training_2_isoamylacetate-1"]
    t = _build(preds=preds)
    r = _row(t, F1, "testing_1_hexanol-0.1")
    assert r.minutes_since_exposure_end == pytest.approx(30.0)


def test_minutes_since_block_start(tidy):
    f1 = tidy[tidy.batch == F1].set_index("trial_label").minutes_since_block_start
    assert f1["training_2_isoamylacetate-1"] == pytest.approx(1.5)
    assert f1["testing_2_isoamylacetate-1"] == pytest.approx(5.0)
    assert f1["pretest_1_isoamylacetate-1"] == 0


def test_flags(tidy):
    r = _row(tidy, F3, "testing_2_paraffinoil-100_then_isoamylacetate-1")
    assert r.paraffin_off_trial and r.paraffin_off_protocol and not r.is_click
    r = _row(tidy, F3, "testing_1_paraffinoil-100_then_isoamylacetate-1")
    assert not r.paraffin_off_trial and r.is_click
    r = _row(tidy, F3, "pretest_1_isoamylacetate-1")
    assert r.odor_window_s == pytest.approx(0.025)
    assert (tidy.responded == (tidy.score >= THR)).all()
    assert tidy.score.dtype.kind == "i"


def test_paraffin_off_exposure_marks_the_whole_fly():
    t = _build(paraffin_off=frozenset({(E1, F1, "training_1")}))
    f1 = t[t.batch == F1]
    assert f1.paraffin_off_protocol.all()
    assert f1.paraffin_off_trial.sum() == 1


def test_click_needs_the_fixed_protocol_date():
    raw = _raw_frame()
    raw.loc[raw.folder == F2, "trial_time"] -= pd.Timedelta(days=10)    # Nov 7: pre-fix
    t = _build(raw=raw)
    assert not t[t.batch == F2].is_click.any()


def test_orphan_odor1_row_is_dropped_and_logged(caplog):
    preds = _pred_frame()
    preds = preds[preds.trial_label !=
                  "testing_4_isoamylacetate-1_then_isoamylacetate-1__odor2"]
    with caplog.at_level("WARNING", logger="per_analysis.io"):
        t = _build(preds=preds)
    assert "testing_4_isoamylacetate-1_then_isoamylacetate-1" not in set(t.trial_label)
    assert "odor-1 row without its odor-2 row" in caplog.text


# ── validation ────────────────────────────────────────────────────────────────
def test_validate_accepts(tidy):
    io.validate_trials(tidy, threshold=THR)


def test_validate_rejects_duplicates(tidy):
    with pytest.raises(AssertionError, match="duplicate"):
        io.validate_trials(pd.concat([tidy, tidy.iloc[[0]]]), threshold=THR)


def test_validate_rejects_out_of_range_score(tidy):
    bad = tidy.copy()
    bad.loc[bad.index[0], "score"] = 6
    with pytest.raises(AssertionError, match="score"):
        io.validate_trials(bad, threshold=THR)


def test_build_rejects_out_of_range_score():
    preds = _pred_frame()
    preds.loc[0, "score"] = 9
    with pytest.raises(AssertionError, match="score"):
        _build(preds=preds)


def _b2b_csv(t, window):
    """What the b2b step writes for one window: 'pre' per partner, 'post' per pair trial."""
    w = t[(t.window == window) & ~t.is_light_probe & t.experiment.isin(["exp2", "exp3"])]
    rows = [dict(dataset=r.dataset, fly=r.fly, fly_number=r.fly_number,
                 role="post" if r.block == "pair" else "pre", label=r.trial_label,
                 score=r.score) for r in w.itertuples()]
    return pd.DataFrame(rows, columns=["dataset", "fly", "fly_number", "role", "label", "score"])


def test_b2b_counts_check(tidy, tmp_path):
    t = tidy.assign(window=np.where(tidy.batch == F3, "day", tidy.window))
    paths = {}
    for w in ("day", "evening"):
        paths[w] = tmp_path / f"{w}.csv"
        _b2b_csv(t, w).to_csv(paths[w], index=False)
    table, ok = io.b2b_counts_check(t, paths)
    assert ok, table.to_string()
    total = table[table.window == "day+evening"].set_index("dataset")
    assert total.loc[E2, "pair_here"] == 4 and total.loc[E3, "alone_here"] == 1
    b = pd.read_csv(paths["evening"])
    b.iloc[1:].to_csv(paths["evening"], index=False)             # one presentation short
    table, ok = io.b2b_counts_check(t, paths)
    assert not ok
    assert table.loc[table.ok == False, "window"].tolist() == ["evening", "day+evening"]  # noqa: E712


def test_column_mapping_covers_contract():
    m = io.column_mapping()
    assert set(io.CONTRACT_COLUMNS) <= set(m)
    assert all({"source", "description"} <= set(v) for v in m.values())


# ── load_trials end to end on a synthetic tree, with the cache ─────────────────
def _write_tree(root: Path) -> C.AnalysisConfig:
    data, raw_root = root / "data", root / "raw"
    data.mkdir()
    meta = _meta_frame()
    meta["dir_val_0"] = 0.0
    tables = {"testing": "t.parquet", "pretest": "p.parquet", "training": "tr.parquet"}
    for kind, name in tables.items():
        meta[meta.trial_type == kind].to_parquet(data / name, index=False)
    preds = _pred_frame()
    is_tr = preds.trial_label.str.startswith("training")
    preds[~is_tr].to_csv(data / "pred.csv", index=False)
    preds[is_tr].to_csv(data / "pred_tr.csv", index=False)
    for d, f, label, t in RAW:
        folder = raw_root / d / f
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"output_{f}_{label}_{t:%Y%m%d_%H%M%S}.csv").write_text("x\n")
    cfg_yaml = root / "config.yaml"
    cfg_yaml.write_text("reaction_prediction:\n  binary_threshold: 2\n")
    return C.AnalysisConfig(data_dir=data, raw_roots=(raw_root,), config_yaml=cfg_yaml,
                            paraffin_off_csv=root / "none.csv", results_dir=root / "results",
                            cohorts=COHORTS, threshold=THR, windows=WINDOWS,
                            trial_tables=tables, prediction_files=("pred.csv", "pred_tr.csv"))


def test_load_trials_synthetic_tree_and_cache(tmp_path, monkeypatch):
    cfg = _write_tree(tmp_path)
    t = io.load_trials(cfg)
    pd.testing.assert_frame_equal(t.reset_index(drop=True),
                                  _build(paraffin_off=frozenset()).reset_index(drop=True),
                                  check_dtype=False)
    cache = cfg.qc_dir / "tidy_trials.parquet"
    assert cache.exists()

    def boom(*a, **k):
        raise RuntimeError("rebuilt")
    monkeypatch.setattr(io, "_build_from_inputs", boom)
    assert len(io.load_trials(cfg)) == len(t)            # served from the cache
    p = cfg.data_dir / "pred.csv"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(RuntimeError, match="rebuilt"):  # an input changed -> rebuild
        io.load_trials(cfg)


def test_write_qc_inputs(tmp_path):
    cfg = _write_tree(tmp_path)
    out = tmp_path / "qc_out"
    paths = io.write_qc_inputs(out, config=cfg)
    assert (out / "column_mapping.yaml").exists() and (out / "tidy_trials.parquet").exists()
    m = yaml.safe_load((out / "column_mapping.yaml").read_text())
    assert "minutes_since_exposure_end" in m
    assert set(paths) >= {out / "column_mapping.yaml", out / "tidy_trials.parquet"}


# ── the real data ─────────────────────────────────────────────────────────────
REAL = pytest.mark.skipif(not (C.DATA_DIR / C.TRIAL_TABLES["testing"]).exists()
                          or not C.RAW_ROOTS[0].exists(), reason="real data not on this machine")


def _expected_fly_counts() -> dict:
    """Independently: flies with any scored row, less frozen trial-table rows."""
    import pyarrow.parquet as pq
    preds = pd.concat([pd.read_csv(C.DATA_DIR / f, usecols=["dataset", "fly", "fly_number",
                                                             "trial_label"])
                       for f in C.PREDICTION_FILES])
    preds = preds[preds.dataset.isin(C.COHORTS)]
    preds["trial_label"] = preds.trial_label.str.replace("__odor2", "", regex=False)
    meta = pd.concat([pq.read_table(C.DATA_DIR / t, columns=["dataset", "fly", "fly_number",
                                                             "trial_label", "frozen"],
                                    filters=[("dataset", "in", list(C.COHORTS))]).to_pandas()
                      for t in C.TRIAL_TABLES.values()])
    m = preds.merge(meta, on=["dataset", "fly", "fly_number", "trial_label"], how="inner")
    m = m[~m.frozen.astype(bool)]
    return m.drop_duplicates(["dataset", "fly", "fly_number"]).groupby("dataset").size().to_dict()


@REAL
def test_real_load_trials(tmp_path):
    cfg = C.AnalysisConfig(results_dir=tmp_path)
    t = io.load_trials(cfg)
    assert set(t.cohort) == set(C.COHORTS)
    got = t.groupby("cohort").fly_id.nunique().to_dict()
    assert got == _expected_fly_counts()
    # protocol facts from the brief hold
    facts = pd.read_csv(cfg.qc_dir / "protocol_facts.csv")
    assert facts.ok.all(), facts[~facts.ok].to_string()
    # every pair trial in a window matches the b2b step's presentations
    table, ok = io.b2b_counts_check(t)
    assert ok, table.to_string()
