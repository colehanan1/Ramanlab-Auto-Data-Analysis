"""Settings for the PER analysis package: paths, the response threshold, the clock
windows, cohorts, palettes and exclusion switches.

Module constants are what ``stats.py`` / ``plots.py`` import. ``AnalysisConfig`` bundles
the ones the data layer needs so a test can point ``load_trials`` at synthetic inputs.

Gate-like values are read from ``config/config_new.yaml`` at import (the operative
pipeline config; ``config/*`` is gitignored, so it lives only on disk):
``reaction_prediction.binary_threshold`` and ``analysis.b2b_pair_figures.windows``.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"            # gitignored
QC_DIR = RESULTS_DIR / "01_data_qc"
DATA_DIR = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
#: Raw trial folders, searched in this order (local first, then the secured mirror).
RAW_ROOTS = (Path("/home/ramanlab/Documents/cole/Data/flys_New"),
             Path("/securedstorage/DATAsec/cole/Data-secured-New"))
CONFIG_YAML = REPO_ROOT / "config" / "config_new.yaml"
#: Per-trial manifest of paraffin trials run with the paraffin line shut off (local only).
PARAFFIN_OFF_CSV = REPO_ROOT / "config" / "paraffin_off_protocol_trials.csv"
#: The b2b pipeline step's per-window presentation tables (counts cross-check).
B2B_PRESENTATIONS = {"day": Path("/home/ramanlab/Documents/cole/Results/b2b/b2b_presentations.csv"),
                     "evening": Path("/home/ramanlab/Documents/cole/Results/b2b-Evening/"
                                     "b2b_presentations.csv")}

TRIAL_TABLES = {"testing": "all_envelope_rows_wide_combined_base.parquet",
                "pretest": "all_envelope_rows_wide_combined_base_pretest.parquet",
                "training": "all_envelope_rows_wide_combined_base_training.parquet"}
PREDICTION_FILES = ("model_predictions.csv", "model_predictions_training.csv")

# ── statistics ────────────────────────────────────────────────────────────────
SEED = 0
N_BOOT = 10_000
N_PERM = 100_000
SCORE_MIN, SCORE_MAX = -1, 5           # -1 = proboscis retraction, a valid level
DEFAULT_THRESHOLD = 2

# ── cohorts ───────────────────────────────────────────────────────────────────
#: dataset name -> experiment. exp1: pre -> exposure -> post; exp2: IAA first in each
#: pair; exp3: IAA second.
COHORTS: dict[str, str] = {
    "IAA-Sensitivity-1-1": "exp1",
    "Hex-Sensitivity-1-0.1": "exp1",
    "3Oct-Sensitivity-1-0.1": "exp1",
    "Paraffin-Sensitivity-1-100": "exp1",
    "IAA-PairFirst-Control-1-1": "exp2",
    "IAA-PairSecond-Control-1-1": "exp3",
}
CONTROL_COHORT = "Paraffin-Sensitivity-1-100"
#: Dataset-name odor token -> canonical odor.
NAME_ODORS = {"IAA": "IAA", "Hex": "Hexanol", "3Oct": "3-Octanol", "Paraffin": "Paraffin Oil"}

# ── odors and colours ─────────────────────────────────────────────────────────
ODOR_ORDER = ["IAA", "Hexanol", "3-Octanol", "Benzaldehyde", "Citral", "Ethyl Butyrate",
              "Linalool", "Paraffin Oil"]
#: CVD-validated 8-slot categorical palette (same slots, same order, as COHORT_PALETTE
#: in scripts/analysis/sensitivity_window_results.py; copied, not imported).
PALETTE_8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
             "#e87ba4", "#008300", "#6250d6", "#e34948")
ODOR_COLORS = dict(zip(ODOR_ORDER, PALETTE_8))
CONDITION_COLORS = {"pre": "#52514e", "alone": "#52514e", "post": "#2a78d6",
                    "pair": "#2a78d6", 1: "#1baf7a", 2: "#eb6834"}
# Day gold / evening plum. dataviz validate_palette.js --mode light, 2026-11-18:
#   "#d4a600,#7a2e8e": ALL CHECKS PASS (CVD dE 40.0 deutan, normal dE 41.0; contrast WARN
#   #d4a600 2.21:1). With the condition hues, --pairs all
#   "#2a78d6,#1baf7a,#eb6834,#d4a600,#7a2e8e": ALL CHECKS PASS (worst CVD #d4a600/#eb6834
#   dE 7.7 deutan = WARN band, normal dE 15.4). The old #2a78d6/#6250d6 FAILED (dE 4.5
#   deutan) and #2a78d6 means post/pair. Evening bars keep the WINDOW_HATCHES hatch.
WINDOW_COLORS = {"day": "#d4a600", "evening": "#7a2e8e"}
CLICK_ODOR = "Paraffin Oil"   # in the pair experiments this is a master-valve click

# ── exclusions (all off: QC flags flies, nothing is dropped unless switched on) ─
EXCLUDE_NEVER_RESPONDERS = False
EXCLUDE_ALWAYS_RESPONDERS = False
EXCLUDE_INCOMPLETE = False

#: The rig fix that turned a Paraffin Oil trial into a 50 ms master-valve click with the
#: paraffin line left flowing (Pi clock). Earlier paraffin trials are in PARAFFIN_OFF_CSV.
PARAFFIN_CLICK_FROM = dt.datetime(2026, 11, 10, 13, 32, 59)


# ── clock windows ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Window:
    """A clock window of the folder start, ``start <= t < end``."""
    name: str
    start: dt.time
    end: dt.time

    def contains(self, t: dt.time) -> bool:
        return self.start <= t < self.end


def parse_clock(value: Any) -> dt.time:
    """``"09:40"`` / ``dt.time`` / the int YAML 1.1 makes of unquoted ``16:40`` (base-60:
    16*60 + 40 = 1000) -> time of day."""
    if isinstance(value, dt.time):
        return value
    if isinstance(value, bool):
        raise ValueError(f"not a clock time: {value!r}")
    if isinstance(value, int):
        hours, minutes = divmod(value, 60)
        if not (0 <= hours < 24):
            raise ValueError(f"sexagesimal clock out of range: {value!r}")
        return dt.time(hours, minutes)
    text = str(value).strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt).time()
        except ValueError:
            pass
    raise ValueError(f"not a clock time: {value!r}")


def _window_name(entry: Mapping, taken: set) -> str:
    if entry.get("name"):
        name = str(entry["name"])
    else:
        out = Path(str(entry.get("out_dir", ""))).name.lower()
        name = "evening" if "evening" in out else "day"
    base, i = name, 2
    while name in taken:
        name, i = f"{base}{i}", i + 1
    return name


def windows_from_entries(entries: Sequence[Mapping]) -> tuple[Window, ...]:
    """``analysis.b2b_pair_figures.windows`` entries -> Windows. The name is ``name:``
    when set, else "evening" for an out_dir naming it and "day" otherwise."""
    taken: set = set()
    out = []
    for e in entries or ():
        name = _window_name(e, taken)
        taken.add(name)
        out.append(Window(name, parse_clock(e["start"]), parse_clock(e["end"])))
    return tuple(out)


DEFAULT_WINDOWS = (Window("day", dt.time(9, 40), dt.time(16, 20)),
                   Window("evening", dt.time(16, 40), dt.time(21, 20)))
OTHER_WINDOW = "other"


def _read_yaml(path: Path) -> dict:
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except OSError as exc:
        log.warning("cannot read %s (%s); using defaults", path, exc)
        return {}


def load_threshold(path: Path = CONFIG_YAML) -> int:
    """``reaction_prediction.binary_threshold`` (default 2)."""
    value = (_read_yaml(path).get("reaction_prediction") or {}).get("binary_threshold")
    return DEFAULT_THRESHOLD if value is None else int(value)


def load_windows(path: Path = CONFIG_YAML) -> tuple[Window, ...]:
    """``analysis.b2b_pair_figures.windows`` (defaults: day 09:40-16:20, evening
    16:40-21:20)."""
    cfg = ((_read_yaml(path).get("analysis") or {}).get("b2b_pair_figures") or {})
    entries = cfg.get("windows")
    return windows_from_entries(entries) if entries else DEFAULT_WINDOWS


RESPONSE_THRESHOLD = load_threshold()
WINDOWS = load_windows()


@dataclass(frozen=True)
class AnalysisConfig:
    """What ``io.load_trials`` reads. Defaults are the module constants."""
    data_dir: Path = DATA_DIR
    raw_roots: tuple[Path, ...] = RAW_ROOTS
    config_yaml: Path = CONFIG_YAML
    paraffin_off_csv: Path = PARAFFIN_OFF_CSV
    results_dir: Path = RESULTS_DIR
    cohorts: Mapping[str, str] = field(default_factory=lambda: dict(COHORTS))
    threshold: int = RESPONSE_THRESHOLD
    windows: tuple[Window, ...] = WINDOWS
    trial_tables: Mapping[str, str] = field(default_factory=lambda: dict(TRIAL_TABLES))
    prediction_files: tuple[str, ...] = PREDICTION_FILES

    @property
    def qc_dir(self) -> Path:
        return Path(self.results_dir) / "01_data_qc"


DEFAULT_CONFIG = AnalysisConfig()
