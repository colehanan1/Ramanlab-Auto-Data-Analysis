"""Data layer: the tidy trial table, one row per scored trial.

Sources (read only, never modified):

- trial metadata: the three wide trial tables in ``DATA_DIR`` (metadata columns only,
  via pyarrow column projection; the ~12k ``dir_val_*`` trace columns are never read);
- scores: ``model_predictions.csv`` + ``model_predictions_training.csv``. A pair trial
  has two scored rows, ``<label>`` (odor 1) and ``<label>__odor2`` (odor 2);
- trial times: the stamp in each raw trial file name,
  ``<root>/<dataset>/<folder>/output_*_<label>_<YYYYMMDD_HHMMSS>.csv`` (Pi clock);
- ``config/paraffin_off_protocol_trials.csv`` for the paraffin-off flag.

A pair trial becomes ONE row: its scored odor is odor 2, and odor 1's score rides along
as ``odor1_score``. The contract columns are ``CONTRACT_COLUMNS``; their meaning is in
``column_mapping()`` (also written to ``results/01_data_qc/column_mapping.yaml``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from . import config as C

log = logging.getLogger(__name__)

KEY = ["dataset", "fly", "fly_number"]
META_COLS = ["dataset", "fly", "fly_number", "trial_type", "trial_label", "trial_odor_on_s",
             "trial_odor_off_s", "trial_duration_s", "fps", "frozen"]
PRED_COLS = ["dataset", "fly", "fly_number", "trial_label", "score"]
ODOR2_SUFFIX = "__odor2"
LIGHT_TOKEN = "lightonly"

#: Rig odor token (concentration stripped, lower case) -> canonical name. Mirrors
#: ``NAMES`` in scripts/analysis/b2b_pair_figures.py (pinned by a test). Note the rig's
#: spelling "octonol".
NAMES = {"isoamylacetate": "IAA", "hexanol": "Hexanol", "linalool": "Linalool",
         "benzaldehyde": "Benzaldehyde", "ethylbutyrate": "Ethyl Butyrate", "citral": "Citral",
         "3-octonol": "3-Octanol", "paraffinoil": "Paraffin Oil"}

#: phase (label prefix) -> block, per experiment.
BLOCKS = {"exp1": {"pretest": "pre", "training": "exposure", "testing": "post"},
          "exp2": {"pretest": "alone", "testing": "pair"},
          "exp3": {"pretest": "alone", "testing": "pair"}}
#: Folder suffix -> rig number -> host (no suffix = rig 1).
HOSTS = {1: "Flybehavior2", 2: "BehaviorLocust", 3: "Flybehavior3"}

CONTRACT_COLUMNS = [
    # who
    "experiment", "cohort", "exposure_odor", "exposure_conc", "starvation_h", "fly_id",
    "batch", "session", "rig", "host",
    # when
    "window", "start_time", "start_time_h", "date",
    # which block
    "block", "trial_label", "trial_index", "position_in_block",
    # which odor
    "odor", "prime", "partner", "presentation",
    # flags
    "is_light_probe", "is_click", "paraffin_off_protocol", "odor_window_s",
    # score
    "score", "responded", "odor1_score",
    # time
    "trial_time", "minutes_since_exposure_end", "minutes_since_block_start",
]
EXTRA_COLUMNS = ["dataset", "fly", "fly_number", "phase", "is_odor2", "scored_label",
                 "paraffin_off_trial", "trial_odor_on_s", "trial_odor_off_s", "fps"]

CACHE_NAME = "tidy_trials.parquet"
RAW_CACHE_NAME = "raw_trial_files.parquet"
FACTS_NAME = "protocol_facts.csv"
FINGERPRINT_NAME = "tidy_trials.inputs.json"

_LABEL = re.compile(r"^(pretest|training|testing)_(\d+)_(.+)$", re.IGNORECASE)
_RAW_FILE = re.compile(r"^output_.*?_((?:pretest|training|testing)_\d+_.+)_(\d{8}_\d{6})\.csv$",
                       re.IGNORECASE)
_RIG = re.compile(r"_rig_(\d+)$")


# ── parsing ───────────────────────────────────────────────────────────────────
class TrialLabel(NamedTuple):
    phase: str                 # pretest | training | testing
    trial_index: int           # the N
    odors: tuple               # canonical odors in delivery order; () for the light probe
    is_light_probe: bool
    is_odor2: bool             # the '__odor2' scored row of a pair trial
    base_label: str            # the label without '__odor2'


def canonical_odor(token: str) -> str:
    """``'isoamylacetate-1'`` / ``'IsoamylAcetate-1'`` -> ``'IAA'`` (concentration
    dropped). An unknown token is returned lower-cased (validation rejects it)."""
    base = re.sub(r"-[0-9.]+$", "", str(token).strip().lower())
    return NAMES.get(base, base)


def parse_trial_label(label) -> TrialLabel | None:
    """``pretest_<N>_<odor>-<c>``, ``training_<N>_...``, ``testing_<N>_<o1>-<c>_then_<o2>-<c>``
    (optionally ``__odor2``), ``testing_<N>_lightonly``. None when unparseable."""
    text = str(label).strip()
    is_odor2 = text.endswith(ODOR2_SUFFIX)
    base = text[: -len(ODOR2_SUFFIX)] if is_odor2 else text
    m = _LABEL.match(base)
    if not m:
        return None
    rest = m.group(3)
    light = rest.strip().lower() == LIGHT_TOKEN
    odors = () if light else tuple(canonical_odor(t)
                                   for t in re.split(r"_then_", rest, flags=re.IGNORECASE))
    return TrialLabel(m.group(1).lower(), int(m.group(2)), odors, light, is_odor2, base)


def parse_raw_filename(name: str) -> tuple[str, dt.datetime] | None:
    """Raw trial file name -> (lower-cased trial label, Pi-clock stamp), else None."""
    m = _RAW_FILE.match(name)
    if not m:
        return None
    return m.group(1).lower(), dt.datetime.strptime(m.group(2), "%Y%m%d_%H%M%S")


def parse_dataset(name: str) -> dict:
    """``<Odor>-<Protocol>-<starvation h>-<conc>`` -> name_odor (canonical), protocol,
    starvation_h (int), conc (float)."""
    parts = str(name).split("-")
    if len(parts) < 4:
        raise ValueError(f"dataset name not <Odor>-<Protocol>-<h>-<conc>: {name!r}")
    return dict(name_odor=C.NAME_ODORS.get(parts[0], parts[0]),
                protocol="-".join(parts[1:-2]),
                starvation_h=int(float(parts[-2])), conc=float(parts[-1]))


def rig_of(folder: str) -> int:
    m = _RIG.search(str(folder))
    return int(m.group(1)) if m else 1


def session_of(folder: str) -> str:
    return _RIG.sub("", str(folder))


def host_of(rig: int) -> str:
    return HOSTS.get(int(rig), f"rig_{rig}")


def assign_window(ts, windows: Iterable[C.Window]) -> str:
    """Name of the window holding the clock time of ``ts`` (end exclusive), else
    ``"other"`` (also for an unknown start)."""
    if ts is None or pd.isna(ts):
        return C.OTHER_WINDOW
    t = pd.Timestamp(ts).time()
    for w in windows:
        if w.contains(t):
            return w.name
    return C.OTHER_WINDOW


# ── readers ───────────────────────────────────────────────────────────────────
def read_trial_tables(cfg: C.AnalysisConfig) -> pd.DataFrame:
    """Metadata columns of the three wide trial tables, in-scope datasets only."""
    frames = []
    for kind, name in cfg.trial_tables.items():
        path = Path(cfg.data_dir) / name
        names = pq.read_schema(path).names
        cols = [c for c in META_COLS if c in names]
        missing = {"dataset", "fly", "fly_number", "trial_label"} - set(cols)
        assert not missing, f"{path} lacks {sorted(missing)}"
        t = pq.read_table(path, columns=cols,
                          filters=[("dataset", "in", list(cfg.cohorts))]).to_pandas()
        t["source_table"] = kind
        log.debug("%s: %d rows x %d metadata columns read (of %d columns)", name, len(t),
                  len(cols), len(names))
        frames.append(t)
    meta = pd.concat(frames, ignore_index=True)
    if "frozen" not in meta:
        meta["frozen"] = False
    return meta


def read_predictions(cfg: C.AnalysisConfig) -> pd.DataFrame:
    frames = []
    for name in cfg.prediction_files:
        path = Path(cfg.data_dir) / name
        p = pd.read_csv(path, usecols=lambda c: c in PRED_COLS)
        log.debug("%s: %d scored rows (all datasets)", name, len(p))
        frames.append(p[p["dataset"].isin(list(cfg.cohorts))])
    preds = pd.concat(frames, ignore_index=True)
    log.debug("scored rows in scope: %s", preds.shape)
    return preds


def read_paraffin_off(path: Path) -> frozenset:
    """{(dataset, batch folder, '<phase>_<N>')} of the paraffin-off-protocol trials."""
    path = Path(path)
    if not path.exists():
        log.warning("no paraffin-off manifest at %s: paraffin_off_* flags all False", path)
        return frozenset()
    d = pd.read_csv(path, comment="#")
    keys = frozenset(zip(d["dataset"].astype(str), d["batch"].astype(str),
                         d["trial"].astype(str).str.lower()))
    log.debug("paraffin-off manifest: %d trials", len(keys))
    return keys


def scan_raw_trials(roots: Iterable[Path], folders: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """One row per (dataset, folder, trial label) with a stamped raw trial file: the
    Pi-clock stamp, the root it came from and how many files carried that label. Roots
    are searched in order; a label already found keeps the first root's stamp."""
    rows = []
    for ds, folder in sorted(set(map(tuple, folders))):
        found: dict[str, tuple] = {}
        for root in roots:
            d = Path(root) / ds / folder
            if not d.is_dir():
                continue
            stamps = defaultdict(list)
            for name in os.listdir(d):
                got = parse_raw_filename(name)
                if got:
                    stamps[got[0]].append(got[1])
            for label, ts in stamps.items():
                if len(ts) > 1:
                    log.warning("%s/%s: %d raw files for %s (%s); keeping the earliest",
                                ds, folder, len(ts), label, sorted(ts))
                t0 = min(ts)
                if label in found:
                    if found[label][0] != t0:
                        log.warning("%s/%s %s: stamp %s in %s differs from %s in %s; keeping "
                                    "the first", ds, folder, label, t0, root, found[label][0],
                                    found[label][1])
                    continue
                found[label] = (t0, str(root), len(ts))
        if not found:
            log.warning("no stamped raw trial files for %s/%s in any root", ds, folder)
        rows += [dict(dataset=ds, folder=folder, trial_label=lab, trial_time=t, root=r,
                      n_files=n) for lab, (t, r, n) in sorted(found.items())]
    cols = ["dataset", "folder", "trial_label", "trial_time", "root", "n_files"]
    raw = pd.DataFrame(rows, columns=cols)
    raw["trial_time"] = pd.to_datetime(raw["trial_time"])
    log.debug("raw trial files: %d labels in %d folders", len(raw),
              raw[["dataset", "folder"]].drop_duplicates().shape[0])
    return raw


# ── the builder (pure) ────────────────────────────────────────────────────────
def _mi(*arrays) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([np.asarray(a) for a in arrays])


def _str(values, index) -> pd.Series:
    return pd.Series(list(values), index=index, dtype="str")


def build_tidy(meta: pd.DataFrame, preds: pd.DataFrame, raw: pd.DataFrame, *,
               cohorts: Mapping[str, str], windows: Iterable[C.Window], threshold: int,
               paraffin_off: frozenset = frozenset(),
               click_from: dt.datetime = C.PARAFFIN_CLICK_FROM) -> pd.DataFrame:
    """Trial metadata + scores + raw-file stamps -> the validated tidy table."""
    cohorts, windows = dict(cohorts), tuple(windows)
    meta = meta[meta["dataset"].isin(list(cohorts))].copy()
    preds = preds[preds["dataset"].isin(list(cohorts))].copy()
    for df in (meta, preds):
        df["fly_number"] = pd.to_numeric(df["fly_number"]).astype("int64")
    log.debug("in scope: trial-table rows %s, scored rows %s", meta.shape, preds.shape)

    sc = pd.to_numeric(preds["score"], errors="coerce")
    bad = sc.isna() | (sc != sc.round()) | ~sc.between(C.SCORE_MIN, C.SCORE_MAX)
    assert not bad.any(), (f"score outside {C.SCORE_MIN}..{C.SCORE_MAX} or not an integer: "
                           f"{preds[bad].head().to_dict('records')}")
    preds["score"] = sc.astype("int64")
    for name, df in (("scored", preds), ("trial-table", meta)):
        dup = df.duplicated(KEY + ["trial_label"], keep=False)
        assert not dup.any(), f"duplicate {name} rows: {df[dup].head().to_dict('records')}"

    parsed = preds["trial_label"].map(parse_trial_label)
    unparsed = parsed.isna()
    if unparsed.any():
        log.warning("%d scored rows with an unparseable label dropped: %s", int(unparsed.sum()),
                    sorted(set(preds.loc[unparsed, "trial_label"]))[:10])
        preds, parsed = preds[~unparsed].copy(), parsed[~unparsed]
    preds["base_label"] = [p.base_label for p in parsed]
    preds["phase"] = [p.phase for p in parsed]
    preds["trial_index"] = np.array([p.trial_index for p in parsed], dtype="int64")
    preds["odors"] = [p.odors for p in parsed]
    preds["is_light_probe"] = np.array([p.is_light_probe for p in parsed], dtype=bool)
    preds["is_odor2"] = np.array([p.is_odor2 for p in parsed], dtype=bool)
    preds["is_pair"] = np.array([len(p.odors) == 2 for p in parsed], dtype=bool)
    stray = preds["is_odor2"] & ~preds["is_pair"]
    assert not stray.any(), f"'__odor2' on a single-odor trial: {preds[stray].head()}"

    # trial metadata; rows of frozen folders go
    m = preds.merge(meta.rename(columns={"trial_label": "base_label"}),
                    on=KEY + ["base_label"], how="left", indicator=True)
    no_meta = (m["_merge"] == "left_only").to_numpy()
    if no_meta.any():
        log.warning("%d scored rows without a trial-table row dropped (by dataset: %s)",
                    int(no_meta.sum()), m[no_meta].groupby("dataset").size().to_dict())
    m = m[~no_meta].drop(columns="_merge")
    frozen = m["frozen"].fillna(False).astype(bool).to_numpy()
    log.debug("%d scored rows in frozen folders dropped", int(frozen.sum()))
    m = m[~frozen].copy()
    scored = set(map(tuple, preds[KEY + ["base_label"]].itertuples(index=False)))
    unscored = [k not in scored for k in meta[KEY + ["trial_label"]].itertuples(index=False)]
    log.debug("trial-table rows with no score (flies flagged dead/bad or unscored): %s",
              meta[unscored].groupby("dataset").size().to_dict())

    # a pair trial -> its odor-2 row, carrying odor 1's score
    k4 = KEY + ["base_label"]
    pair1 = (m["is_pair"] & ~m["is_odor2"]).to_numpy()
    pair2 = (m["is_pair"] & m["is_odor2"]).to_numpy()
    o1 = m.loc[pair1].set_index(k4)["score"]
    k1, k2 = set(o1.index), set(map(tuple, m.loc[pair2, k4].itertuples(index=False)))
    if k1 - k2:
        log.warning("%d pair odor-1 row without its odor-2 row(s) dropped: %s", len(k1 - k2),
                    sorted(k1 - k2)[:5])
    if k2 - k1:
        log.warning("%d pair odor-2 rows without an odor-1 row: odor1_score NaN", len(k2 - k1))
    m["odor1_score"] = np.nan
    m.loc[pair2, "odor1_score"] = o1.reindex(_mi(*(m.loc[pair2, c] for c in k4))).to_numpy(
        dtype=float)
    m = m.loc[~pair1].copy()
    log.debug("after folding pair odor-1 rows: %s", m.shape)

    # experiment, block
    m["experiment"] = m["dataset"].map(cohorts)
    m["block"] = [BLOCKS[e].get(ph) for e, ph in zip(m["experiment"], m["phase"])]
    nob = m["block"].isna().to_numpy()
    assert not nob.any(), f"phase not in its experiment: {m.loc[nob, KEY + ['base_label']].head()}"
    assert not (m["is_pair"] & (m["experiment"] == "exp1")).any(), "pair label in exp1"

    # odors
    odor, prime, partner = [], [], []
    for e, ods, light in zip(m["experiment"], m["odors"], m["is_light_probe"]):
        if light:
            o = p = pa = None
        elif len(ods) == 2:
            p, o = ods
            pa = o if e == "exp2" else p
        else:
            o, p = ods[0], None
            pa = o
        odor.append(o)
        prime.append(p)
        partner.append(pa)
    m["odor"], m["prime"], m["partner"] = (_str(odor, m.index), _str(prime, m.index),
                                           _str(partner, m.index))

    # who
    info = {d: parse_dataset(d) for d in cohorts}
    exp1 = (m["experiment"] == "exp1").to_numpy()
    m["cohort"] = m["dataset"]
    m["exposure_odor"] = _str([info[d]["name_odor"] if x else None
                               for d, x in zip(m["dataset"], exp1)], m.index)
    m["exposure_conc"] = [info[d]["conc"] if x else np.nan for d, x in zip(m["dataset"], exp1)]
    m["starvation_h"] = np.array([info[d]["starvation_h"] for d in m["dataset"]], dtype="int64")
    m["fly_id"] = m["dataset"] + "|" + m["fly"] + "|" + m["fly_number"].astype(str)
    m["batch"] = m["fly"]
    m["session"] = m["fly"].map(session_of)
    m["rig"] = m["fly"].map(rig_of).astype("int64")
    m["host"] = m["rig"].map(host_of)

    # when: raw-file stamps (Pi clock)
    raw = raw.copy()
    raw["trial_label"] = raw["trial_label"].str.lower()
    raw["phase"] = raw["trial_label"].str.extract(r"^(pretest|training|testing)_",
                                                  expand=False)
    raw["trial_time"] = pd.to_datetime(raw["trial_time"])
    rt = raw.set_index(["dataset", "folder", "trial_label"])["trial_time"]
    assert rt.index.is_unique, "raw trial files: one label twice in a folder"
    m["trial_time"] = pd.to_datetime(
        rt.reindex(_mi(m["dataset"], m["fly"], m["base_label"].str.lower())).to_numpy())
    no_t = m["trial_time"].isna().to_numpy()
    if no_t.any():
        log.warning("%d scored rows without a stamped raw file: %s", int(no_t.sum()),
                    m.loc[no_t, ["dataset", "fly", "base_label"]].drop_duplicates().head(10)
                    .to_dict("records"))
    fk = _mi(m["dataset"], m["fly"])
    starts = raw[raw["phase"] == "pretest"].groupby(["dataset", "folder"])["trial_time"].min()
    m["start_time"] = pd.to_datetime(starts.reindex(fk).to_numpy())
    phase_start = raw.groupby(["dataset", "folder", "phase"])["trial_time"].min()
    block_start = pd.to_datetime(phase_start.reindex(
        _mi(m["dataset"], m["fly"], m["phase"])).to_numpy())
    minute = pd.Timedelta(minutes=1)
    m["minutes_since_block_start"] = (m["trial_time"] - block_start) / minute
    last_pulse = raw[raw["phase"] == "training"].groupby(["dataset", "folder"])["trial_time"].max()
    last = pd.to_datetime(last_pulse.reindex(fk).to_numpy())
    post = (exp1 & (m["block"] == "post").to_numpy())
    since = ((m["trial_time"] - last) / minute).to_numpy(dtype=float)
    m["minutes_since_exposure_end"] = np.where(post, since, np.nan)
    m["window"] = [assign_window(t, windows) for t in m["start_time"]]
    st = m["start_time"]
    m["start_time_h"] = st.dt.hour + st.dt.minute / 60 + st.dt.second / 3600
    m["date"] = st.dt.normalize()

    # flags
    tkey = m["phase"] + "_" + m["trial_index"].astype(str)
    m["paraffin_off_trial"] = np.array(
        [(d, f, k) in paraffin_off for d, f, k in zip(m["dataset"], m["fly"], tkey)], dtype=bool)
    off_exposure = {(d, f) for d, f, k in paraffin_off if k.startswith("training_")}
    m["paraffin_off_protocol"] = m["paraffin_off_trial"].to_numpy() | np.array(
        [(d, f) in off_exposure for d, f in zip(m["dataset"], m["fly"])], dtype=bool)
    paraffin = ((m["odor"] == C.CLICK_ODOR) | (m["prime"] == C.CLICK_ODOR)).fillna(False)
    m["is_click"] = (paraffin.to_numpy(dtype=bool) & (m["trial_time"] >= click_from).to_numpy()
                     & ~m["paraffin_off_trial"].to_numpy())
    m["odor_window_s"] = (m["trial_odor_off_s"] - m["trial_odor_on_s"]).round(6)

    # score
    m["responded"] = m["score"] >= threshold
    m["scored_label"] = m["trial_label"]
    m["trial_label"] = m["base_label"]

    # order within block, presentation of each pair partner
    m["position_in_block"] = m.groupby(["fly_id", "block"])["trial_index"].rank(
        method="dense").astype("int64")
    m["presentation"] = np.nan
    pr = ((m["block"] == "pair") & ~m["is_light_probe"]).to_numpy()
    m.loc[pr, "presentation"] = m.loc[pr].groupby(["fly_id", "partner"])["trial_index"].rank(
        method="first")

    out = m.sort_values(["experiment", "cohort", "fly_id", "trial_time", "trial_index"])
    out = out[CONTRACT_COLUMNS + EXTRA_COLUMNS].reset_index(drop=True)
    for c in out.columns:                      # one string dtype throughout
        if out[c].dtype == object:
            out[c] = out[c].astype("str")
    validate_trials(out, threshold=threshold)
    log.debug("tidy table: %s", out.shape)
    return out


# ── validation ────────────────────────────────────────────────────────────────
def validate_trials(df: pd.DataFrame, *, threshold: int) -> None:
    """Assert the contract: columns, one row per (fly_id, trial_label, odor-2 flag),
    scores -1..5, blocks, odors, pairs, times."""
    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    assert not missing, f"missing contract columns: {missing}"
    key = ["fly_id", "trial_label"] + (["is_odor2"] if "is_odor2" in df else [])
    dup = df.duplicated(key, keep=False)
    assert not dup.any(), f"duplicate trials on {key}: {df.loc[dup, key].head().to_dict('records')}"
    dup = df.duplicated(["fly_id", "block", "trial_index"], keep=False)
    assert not dup.any(), ("duplicate trial index within fly x block: "
                           f"{df.loc[dup, ['fly_id', 'trial_label']].head().to_dict('records')}")
    s = df["score"]
    assert s.dtype.kind == "i", f"score dtype {s.dtype}, expected int"
    assert s.between(C.SCORE_MIN, C.SCORE_MAX).all(), (
        f"score outside {C.SCORE_MIN}..{C.SCORE_MAX}: {sorted(set(s[~s.between(-1, 5)]))}")
    assert df["experiment"].isin(list(BLOCKS)).all(), "unknown experiment"
    for e, g in df.groupby("experiment"):
        extra = set(g["block"]) - set(BLOCKS[e].values())
        assert not extra, f"{e}: unknown blocks {extra}"
    light = df["is_light_probe"].astype(bool)
    assert df.loc[light, "odor"].isna().all(), "light probe with an odor"
    bad = ~df.loc[~light, "odor"].isin(C.ODOR_ORDER)
    assert not bad.any(), f"non-canonical odors: {sorted(set(df.loc[~light, 'odor'][bad]))}"
    assert (df["responded"] == (df["score"] >= threshold)).all(), "responded != score >= threshold"
    pair = (df["block"] == "pair") & ~light
    assert df.loc[pair, "presentation"].isin([1, 2]).all(), "pair presentation not 1/2"
    assert df.loc[pair, "prime"].notna().all(), "pair row without a prime"
    assert df.loc[~pair, "presentation"].isna().all(), "presentation on a non-pair row"
    assert df["trial_time"].notna().all(), "trial without a raw-file stamp"
    assert df["start_time"].notna().all(), "folder without a pre-test stamp"
    assert (df["trial_time"] >= df["start_time"]).all(), "trial before its folder's start"
    post = (df["experiment"] == "exp1") & (df["block"] == "post")
    since = df.loc[post, "minutes_since_exposure_end"]
    assert since.notna().all() and (since > 0).all(), "post trial not after the last pulse"
    expo = df["block"] == "exposure"
    assert (df.loc[expo, "odor"] == df.loc[expo, "exposure_odor"]).all(), \
        "exposure pulse of another odor than the cohort's"


# ── protocol facts (log and assert; never "fixed") ────────────────────────────
#: Exp 1 cohorts: flies of the trial table never tested on Paraffin Oil (paraffin
#: joined the panel partway). None = every fly.
NO_PARAFFIN_TEST = {"IAA-Sensitivity-1-1": None, "Hex-Sensitivity-1-0.1": 23,
                    "3Oct-Sensitivity-1-0.1": 36}
SHORT_CLICK = ("IAA-PairSecond-Control-1-1", "november_13_batch_1_rig_3", "pretest_1_")


def protocol_facts(meta: pd.DataFrame, tidy: pd.DataFrame) -> pd.DataFrame:
    """The brief's protocol facts, observed vs expected (strings, for one CSV)."""
    rows = []

    def add(fact, expected, observed, ok, detail=""):
        rows.append(dict(fact=fact, expected=str(expected), observed=str(observed),
                         ok=bool(ok), detail=detail))

    live = meta[~meta["frozen"].fillna(False).astype(bool)]
    tests = live[live["trial_label"].str.startswith("testing_")]
    has_p = tests.groupby(KEY)["trial_label"].agg(lambda s: s.str.contains("paraffinoil").any())
    scored_post = tidy[(tidy["block"] == "post") & ~tidy["is_light_probe"]]
    s_has_p = scored_post.groupby("fly_id")["odor"].agg(lambda s: (s == C.CLICK_ODOR).any())
    s_ds = scored_post.groupby("fly_id")["cohort"].first()
    for ds, expected in NO_PARAFFIN_TEST.items():
        h = has_p[has_p.index.get_level_values("dataset") == ds]
        n, lacking = len(h), int((~h).sum())
        sh = s_has_p[s_ds == ds]
        exp = n if expected is None else expected
        add(f"{ds}: trial-table flies never tested on Paraffin Oil", exp, lacking,
            lacking == exp, f"{lacking} of {n} trial-table flies; scored: "
                            f"{int((~sh).sum())} of {len(sh)}")
    pairs = tidy[tidy["experiment"].isin(["exp2", "exp3"]) & ~tidy["is_light_probe"]]
    pp = pairs[(pairs["odor"] == C.CLICK_ODOR) | (pairs["prime"] == C.CLICK_ODOR)]
    add("pair experiments: Paraffin Oil trials that are not a click (pre-fix)", 0,
        int((~pp["is_click"]).sum()), not (~pp["is_click"]).any(),
        f"{len(pp)} paraffin rows; first pair trial {pairs['trial_time'].min()}")
    odor_rows = tidy[~tidy["is_light_probe"]]
    short = odor_rows[odor_rows["odor_window_s"] < 1.0]
    in_known = (short["cohort"] == SHORT_CLICK[0]) & (short["batch"] == SHORT_CLICK[1]) & \
        short["trial_label"].str.startswith(SHORT_CLICK[2])
    add("odor trials with an odor window < 1 s (PairSecond november_13_batch_1_rig_3 "
        "pretest_1 click; metadata bug fixed in code 2026-11-18, tables not rebuilt)",
        "4 (0 after a rebuild)", len(short), len(short) in (0, 4) and bool(in_known.all()),
        "; ".join(f"{r.fly_id} {r.trial_label} {r.odor_window_s:.3f}s"
                  for r in short.itertuples()))
    return pd.DataFrame(rows, columns=["fact", "expected", "observed", "ok", "detail"])


def check_protocol_facts(facts: pd.DataFrame, *, strict: bool = False) -> bool:
    """Log every fact; raise on a failed one when ``strict``."""
    for r in facts.itertuples():
        (log.info if r.ok else log.warning)("protocol fact %s: %s (expected %s, observed %s) %s",
                                            "OK" if r.ok else "CHANGED", r.fact, r.expected,
                                            r.observed, r.detail)
    if strict:
        bad = facts[~facts["ok"]]
        assert bad.empty, f"protocol facts changed:\n{bad.to_string()}"
    return bool(facts["ok"].all())


# ── load with cache ───────────────────────────────────────────────────────────
def _stat(path: Path):
    try:
        s = Path(path).stat()
        return [s.st_mtime_ns, s.st_size]
    except OSError:
        return None


def _fingerprint(cfg: C.AnalysisConfig) -> dict:
    files = [Path(cfg.data_dir) / n for n in cfg.trial_tables.values()]
    files += [Path(cfg.data_dir) / n for n in cfg.prediction_files]
    files += [Path(cfg.config_yaml), Path(cfg.paraffin_off_csv)]
    raw_dirs = [Path(r) / d for r in cfg.raw_roots for d in sorted(cfg.cohorts)]
    code = hashlib.sha256()
    for mod in (Path(__file__), Path(C.__file__)):
        code.update(mod.read_bytes())
    return dict(files={str(p): _stat(p) for p in files},
                raw_dirs={str(p): _stat(p) for p in raw_dirs},
                code=code.hexdigest(), cohorts=sorted(cfg.cohorts.items()),
                threshold=int(cfg.threshold),
                windows=[[w.name, str(w.start), str(w.end)] for w in cfg.windows])


def _build_from_inputs(cfg: C.AnalysisConfig):
    meta = read_trial_tables(cfg)
    preds = read_predictions(cfg)
    folders = meta[["dataset", "fly"]].drop_duplicates().itertuples(index=False)
    raw = scan_raw_trials(cfg.raw_roots, folders)
    tidy = build_tidy(meta, preds, raw, cohorts=cfg.cohorts, windows=cfg.windows,
                      threshold=cfg.threshold, paraffin_off=read_paraffin_off(cfg.paraffin_off_csv))
    facts = protocol_facts(meta, tidy)
    check_protocol_facts(facts, strict=False)
    log.info("tidy table built: %d rows, %d flies, %d cohorts", len(tidy),
             tidy["fly_id"].nunique(), tidy["cohort"].nunique())
    log.debug("rows by experiment x block x window:\n%s", shape_table(tidy).to_string())
    return tidy, raw, facts


def load_trials(config: C.AnalysisConfig | None = None, *, use_cache: bool = True,
                cache_dir: Path | None = None) -> pd.DataFrame:
    """The tidy trial table (``CONTRACT_COLUMNS`` + ``EXTRA_COLUMNS``), validated.

    Cached in ``<results>/01_data_qc/tidy_trials.parquet``; rebuilt when an input's
    mtime/size, an in-scope raw dataset directory, the cohorts/threshold/windows or this
    module's code change. The raw-file listing and protocol facts are cached beside it."""
    cfg = config or C.DEFAULT_CONFIG
    cache_dir = Path(cache_dir) if cache_dir else cfg.qc_dir
    fp = _fingerprint(cfg)
    fp_path, tidy_path = cache_dir / FINGERPRINT_NAME, cache_dir / CACHE_NAME
    if use_cache and fp_path.exists() and tidy_path.exists() \
            and (cache_dir / RAW_CACHE_NAME).exists():
        try:
            same = json.loads(fp_path.read_text()) == json.loads(json.dumps(fp))
        except (OSError, ValueError):
            same = False
        if same:
            tidy = pd.read_parquet(tidy_path)
            validate_trials(tidy, threshold=cfg.threshold)
            log.info("tidy table from cache %s: %s", tidy_path, tidy.shape)
            return tidy
        log.info("inputs changed since the cached tidy table: rebuilding")
    tidy, raw, facts = _build_from_inputs(cfg)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tidy.to_parquet(tidy_path, index=False)
        raw.to_parquet(cache_dir / RAW_CACHE_NAME, index=False)
        facts.to_csv(cache_dir / FACTS_NAME, index=False)
        fp_path.write_text(json.dumps(fp, indent=1))
        log.debug("cached %s, %s, %s", tidy_path.name, RAW_CACHE_NAME, FACTS_NAME)
    return tidy


def load_raw_trials(config: C.AnalysisConfig | None = None, *,
                    cache_dir: Path | None = None) -> pd.DataFrame:
    """The raw-file listing behind the tidy table (refreshing the cache if stale)."""
    cfg = config or C.DEFAULT_CONFIG
    cache_dir = Path(cache_dir) if cache_dir else cfg.qc_dir
    load_trials(cfg, cache_dir=cache_dir)
    return pd.read_parquet(cache_dir / RAW_CACHE_NAME)


def load_protocol_facts(config: C.AnalysisConfig | None = None, *,
                        cache_dir: Path | None = None) -> pd.DataFrame:
    cfg = config or C.DEFAULT_CONFIG
    cache_dir = Path(cache_dir) if cache_dir else cfg.qc_dir
    load_trials(cfg, cache_dir=cache_dir)
    return pd.read_csv(cache_dir / FACTS_NAME)


def shape_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Rows by experiment x block x window (light probes counted separately)."""
    t = trials.assign(kind=np.where(trials["is_light_probe"], "light probe", "odor"))
    return (t.groupby(["experiment", "block", "kind", "window"]).size()
            .unstack("window", fill_value=0))


# ── b2b counts cross-check ────────────────────────────────────────────────────
def b2b_counts_check(trials: pd.DataFrame,
                     presentations: Mapping[str, Path] | None = None) -> tuple[pd.DataFrame, bool]:
    """Per pair dataset and window: alone and pair odor-2 trial counts here vs the b2b
    step's ``b2b_presentations.csv`` ('pre' / 'post' rows) for the same window, and the
    two windows summed. Also flies, the pair label sets and score agreement (reported,
    not required: the b2b CSV is as fresh as its last build)."""
    presentations = dict(presentations or C.B2B_PRESENTATIONS)
    pair_ds = sorted(set(trials.loc[trials["experiment"].isin(["exp2", "exp3"]), "cohort"]))
    odor = trials[~trials["is_light_probe"]]
    rows, totals = [], defaultdict(lambda: defaultdict(int))
    for window, path in presentations.items():
        path = Path(path)
        if not path.exists():
            log.error("b2b counts check: %s missing", path)
            rows.append(dict(window=window, dataset="*", ok=False, note=f"missing {path}"))
            continue
        b = pd.read_csv(path)
        for ds in sorted(set(pair_ds) | set(b["dataset"])):
            mine = odor[(odor["window"] == window) & (odor["cohort"] == ds)]
            bb = b[b["dataset"] == ds]
            mp = mine[mine["block"] == "pair"]
            bp = bb[bb["role"] == "post"]
            mkeys = set(zip(mp["fly"], mp["fly_number"], mp["trial_label"]))
            bkeys = set(zip(bp["fly"], bp["fly_number"], bp["label"]))
            j = mp.merge(bp, left_on=["fly", "fly_number", "trial_label"],
                         right_on=["fly", "fly_number", "label"], suffixes=("", "_b2b"))
            r = dict(window=window, dataset=ds,
                     alone_here=int((mine["block"] == "alone").sum()),
                     alone_b2b=int((bb["role"] == "pre").sum()),
                     pair_here=len(mp), pair_b2b=len(bp),
                     flies_here=mine["fly_id"].nunique(),
                     flies_b2b=bb[["fly", "fly_number"]].drop_duplicates().shape[0],
                     pair_labels_match=mkeys == bkeys,
                     pair_scores_agree=f"{int((j['score'] == j['score_b2b']).sum())}/{len(j)}")
            r["ok"] = r["alone_here"] == r["alone_b2b"] and r["pair_here"] == r["pair_b2b"]
            rows.append(r)
            tot = totals[ds]
            for k in ("alone_here", "alone_b2b", "pair_here", "pair_b2b", "flies_here",
                      "flies_b2b"):
                tot[k] += r[k]
            tot["labels_bad"] += int(not r["pair_labels_match"])
            tot["agree"] += int((j["score"] == j["score_b2b"]).sum())
            tot["joined"] += len(j)
    for ds, tot in totals.items():
        rows.append(dict(window="+".join(presentations), dataset=ds,
                         **{k: tot[k] for k in ("alone_here", "alone_b2b", "pair_here",
                                                "pair_b2b", "flies_here", "flies_b2b")},
                         pair_labels_match=tot["labels_bad"] == 0,
                         pair_scores_agree=f"{tot['agree']}/{tot['joined']}",
                         ok=tot["alone_here"] == tot["alone_b2b"]
                         and tot["pair_here"] == tot["pair_b2b"]))
    table = pd.DataFrame(rows)
    ok = bool(len(table)) and bool(table["ok"].all())
    for r in table.to_dict("records"):
        (log.info if r["ok"] else log.error)("b2b counts check %s: %s", "OK" if r["ok"]
                                             else "MISMATCH", r)
    return table, ok


# ── column mapping ────────────────────────────────────────────────────────────
_MAP = {
    "experiment": ("config.COHORTS[dataset]", "exp1 pre->exposure->post; exp2 IAA first in "
                   "each pair; exp3 IAA second"),
    "cohort": ("dataset", "dataset name <Odor>-<Protocol>-<starvation h>-<conc>"),
    "exposure_odor": ("dataset name", "exp1: canonical exposure odor; exp2/3: NaN (no exposure)"),
    "exposure_conc": ("dataset name", "exp1: exposure concentration; exp2/3: NaN"),
    "starvation_h": ("dataset name", "hours of starvation"),
    "fly_id": ("dataset|fly|fly_number", "unique fly"),
    "batch": ("fly (trial tables)", "batch folder"),
    "session": ("batch folder", "folder without _rig_N: rigs run side by side"),
    "rig": ("batch folder", "_rig_N suffix, 1 when unsuffixed"),
    "host": ("rig", "1 Flybehavior2, 2 BehaviorLocust, 3 Flybehavior3"),
    "window": ("start_time + config_new analysis.b2b_pair_figures.windows",
               "clock window of the folder start, end exclusive: day / evening / other"),
    "start_time": ("raw file names", "folder's earliest pre-test stamp (Pi clock)"),
    "start_time_h": ("start_time", "clock hours as a float"),
    "date": ("start_time", "start date (midnight timestamp)"),
    "block": ("label phase + experiment", "exp1 pre/exposure/post; exp2/3 alone/pair; light "
              "probes sit in the block where they ran (post / pair)"),
    "trial_label": ("trial_label (without __odor2)", "table label, lower case"),
    "trial_index": ("trial_label", "the N of <phase>_<N>_..."),
    "position_in_block": ("trial_index", "1.. by trial_index within fly x block (rows present)"),
    "odor": ("trial_label", "scored odor, canonical; pairs: odor 2; light probe: NaN"),
    "prime": ("trial_label", "pairs: odor 1; else NaN"),
    "partner": ("trial_label", "the varying odor: exp2 odor 2, exp3 odor 1; alone/exp1 = odor"),
    "presentation": ("trial_index", "pairs: 1/2 = 1st/2nd trial of that partner; else NaN"),
    "is_light_probe": ("trial_label", "testing_<N>_lightonly: opto light, no odor (Gr5a>"
                       "CsChrimson positive control); keep out of odor analyses"),
    "is_click": ("odor/prime + trial_time + paraffin-off manifest",
                 "a Paraffin Oil stimulus (scored odor or prime) delivered as the 50 ms "
                 "master-valve click (rig fix 2026-11-10 13:33); every pair-experiment "
                 "paraffin trial, and post-fix exp1 paraffin trials"),
    "paraffin_off_protocol": ("config/paraffin_off_protocol_trials.csv",
                              "this trial ran paraffin shut off 30 s, or this fly's exposure "
                              "pulses did (whole Paraffin-Sensitivity batches)"),
    "odor_window_s": ("trial_odor_off_s - trial_odor_on_s", "odor window in the trial table; "
                      "pairs: the table's window is odor 1's; light probe: the table's value"),
    "score": ("model_predictions(_training).csv", "ordinal -1..5; -1 = retraction (valid)"),
    "responded": ("score", "score >= reaction_prediction.binary_threshold"),
    "odor1_score": ("model_predictions.csv <label> row", "pairs: the same trial's odor-1 "
                    "score (carry-over); else NaN"),
    "trial_time": ("raw file name", "this trial's raw-file stamp (trial start, Pi clock)"),
    "minutes_since_exposure_end": ("raw file names", "exp1 post rows: this trial's stamp minus "
                                   "the folder's last exposure-pulse stamp (start to start), "
                                   "minutes; else NaN"),
    "minutes_since_block_start": ("raw file names", "this trial's stamp minus the folder's first "
                                  "stamp of the same phase, minutes"),
    "dataset": ("trial tables", "= cohort"),
    "fly": ("trial tables", "= batch (the trial tables' name for it)"),
    "fly_number": ("trial tables", "fly slot in the folder"),
    "phase": ("trial_label", "pretest / training / testing"),
    "is_odor2": ("trial_label", "row scored from the '__odor2' prediction (pairs)"),
    "scored_label": ("model_predictions*.csv", "the prediction row's label (with __odor2)"),
    "paraffin_off_trial": ("config/paraffin_off_protocol_trials.csv", "this trial itself is "
                           "in the paraffin-off manifest"),
    "trial_odor_on_s": ("trial tables", "odor onset in the recording, s"),
    "trial_odor_off_s": ("trial tables", "odor offset in the recording, s"),
    "fps": ("trial tables", "frame rate"),
}


def column_mapping() -> dict:
    """Column -> {source, description}, contract columns first."""
    return {c: {"source": _MAP[c][0], "description": _MAP[c][1],
                "contract": c in CONTRACT_COLUMNS}
            for c in CONTRACT_COLUMNS + EXTRA_COLUMNS}


def write_qc_inputs(out_dir: Path | None = None, *,
                    config: C.AnalysisConfig | None = None) -> list[Path]:
    """``column_mapping.yaml`` and ``tidy_trials.parquet`` in ``out_dir``
    (default ``results/01_data_qc``)."""
    cfg = config or C.DEFAULT_CONFIG
    out = Path(out_dir) if out_dir else cfg.qc_dir
    out.mkdir(parents=True, exist_ok=True)
    trials = load_trials(cfg)
    mapping = out / "column_mapping.yaml"
    mapping.write_text(yaml.safe_dump(column_mapping(), sort_keys=False, allow_unicode=True,
                                      width=100))
    tidy = out / CACHE_NAME
    trials.to_parquet(tidy, index=False)
    log.info("wrote %s, %s", mapping, tidy)
    return [mapping, tidy]
