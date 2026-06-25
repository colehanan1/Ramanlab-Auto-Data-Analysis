"""Per-trial metadata loader for fly-behaviour pipeline.

Resolves the authoritative per-trial values written by the Pi recording rig
(sidecar ``.txt`` + sibling ``.csv`` with an ``ActiveOFM`` column) and exposes
them through a single ``TrialMetadata`` object. Replaces the pipeline-wide
``3600 / 1260 / 2460 / 40 fps`` literals so that future datasets with different
trial lengths or randomised odor schedules need no code changes.

The Pi rig writes three siblings per trial in the *batch folder* (one level
above each per-trial folder):

    output_<batch>_<cycle>_<Odor>_<timestamp>.txt   <- sidecar
    output_<batch>_<cycle>_<Odor>_<timestamp>.csv   <- per-frame log w/ ActiveOFM
    output_<batch>_<cycle>_<Odor>_<timestamp>.mp4

The sidecar carries ``Cycle Name``, ``Total Frames Captured`` and
``Achieved Frame Rate``. The CSV carries one row per video frame with an
``ActiveOFM`` column that flips from ``off`` to ``OFM_<channel>`` on the exact
odor-on frame and back to ``off`` on the odor-off frame.

Loader falls back gracefully when sidecars are missing (older datasets,
synthetic fixtures, etc.) using configured global defaults.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Cycle-name format: "<trial_type>_<index>[_<Odor>]"  e.g. "testing_3_Hexanol"
_CYCLE_RE = re.compile(
    r"^(?P<type>training|testing)_(?P<index>\d+)(?:_(?P<odor>[A-Za-z0-9._\-]+))?$",
    re.IGNORECASE,
)

# Sidecar filename: "output_<batch>_<cycle>_<Odor>_<timestamp>.txt"
_SIDECAR_RE = re.compile(r"^output_.+_(training|testing)_\d+_.+_\d{8}_\d{6}\.txt$", re.IGNORECASE)

CACHE_FILENAME = "_trial_meta.json"


@dataclass(frozen=True)
class TrialMetadata:
    """Authoritative per-trial metadata.

    Every consumer in the pipeline should read window / fps / length values
    from this object instead of from hardcoded literals.
    """

    dataset: str
    batch: Path
    trial_dir: Path
    cycle_name: str
    trial_index: int
    odor: Optional[str]
    trial_type: str            # final, after dataset_overrides applied
    raw_trial_type: str        # parsed from sidecar/folder before override
    n_frames: int
    fps: float
    odor_on_frame: Optional[int]
    odor_off_frame: Optional[int]
    sidecar_txt: Optional[Path]
    sidecar_csv: Optional[Path]
    source: str                # "sidecar" | "fallback" | "cache"
    # First optogenetic-LED on time (seconds from recording start), read from
    # the sibling ``sensors_output_*.csv`` ``OFM_Event`` column. ``None`` when
    # no sensors log / no light-on event exists for the trial.
    light_on_s: Optional[float] = None

    @property
    def odor_on_s(self) -> Optional[float]:
        if self.odor_on_frame is None:
            return None
        return self.odor_on_frame / max(self.fps, 1e-6)

    @property
    def odor_off_s(self) -> Optional[float]:
        if self.odor_off_frame is None:
            return None
        return self.odor_off_frame / max(self.fps, 1e-6)

    @property
    def duration_s(self) -> float:
        return self.n_frames / max(self.fps, 1e-6)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("batch", "trial_dir", "sidecar_txt", "sidecar_csv"):
            v = d.get(k)
            d[k] = None if v is None else str(v)
        return d


# ---------------------------------------------------------------------------
# Sidecar / CSV parsing
# ---------------------------------------------------------------------------

def parse_sidecar(txt_path: Path) -> dict:
    """Parse the Pi rig sidecar ``.txt`` into a plain dict.

    Keys are normalised to lowercase snake_case for the fields we care about::

        cycle_name, total_frames_captured, achieved_frame_rate,
        duration_seconds, frame_resolution, video_file, frames_csv_file,
        sensors_csv_file
    """
    result: dict = {}
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, _, value = raw_line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        norm = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if norm not in result:
            result[norm] = value
    return result


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip().split()[0])
    except (ValueError, IndexError):
        return None


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip().split()[0])
    except (ValueError, IndexError):
        return None


def parse_active_ofm(csv_path: Path) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Read ``FrameNumber`` + ``ActiveOFM`` and return ``(on_frame, off_frame, channel)``.

    Robust to capitalisation and to trials where the odor never turns on.
    """
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c.lower() in {"framenumber", "activeofm"})
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as exc:
        LOGGER.debug("ActiveOFM read failed for %s: %s", csv_path, exc)
        return None, None, None

    frame_col = next((c for c in df.columns if c.lower() == "framenumber"), None)
    ofm_col = next((c for c in df.columns if c.lower() == "activeofm"), None)
    if frame_col is None or ofm_col is None:
        return None, None, None

    states = df[ofm_col].astype(str).str.strip()
    is_on = (states.str.lower() != "off") & (states != "") & (states.str.lower() != "nan")
    if not is_on.any():
        return None, None, None

    on_idx = int(is_on.idxmax())  # first True
    on_frame = int(df[frame_col].iloc[on_idx])
    channel = states.iloc[on_idx] or None

    # First off after on_frame
    tail = ~is_on.iloc[on_idx + 1:]
    off_frame: Optional[int]
    if tail.any():
        off_local = int(tail.idxmax())
        off_frame = int(df[frame_col].loc[off_local])
    else:
        off_frame = None
    return on_frame, off_frame, channel


# Optogenetic-LED "on" markers in the sensors-log ``OFM_Event`` column. The rig
# emits ``LIGHT_SOLID_ON`` / ``LIGHT_PULSE_ON`` (which toggles many thousands of
# times during a pulse train) plus manual / network triggers. Any of these is an
# "on"; ``LIGHT_OFF`` / ``LIGHT_PULSE_OFF`` are explicitly not.
_LIGHT_ON_RE = re.compile(r"^LIGHT_.*(?:_ON|_START|_TRIGGER)$", re.IGNORECASE)


def parse_light_on_seconds(sensors_csv: Path) -> Optional[float]:
    """First light-on time (seconds from recording start) from a sensors CSV.

    The Pi rig logs LED transitions in the ``OFM_Event`` column of
    ``sensors_output_*.csv`` with nanosecond ``MonoNs`` timestamps. The light
    pulses on/off many times within a trial, so we return the time of the
    *first* on-event (``LIGHT_SOLID_ON`` / ``LIGHT_PULSE_ON`` /
    ``LIGHT_PULSE_MANUAL_START`` / ``LIGHT_PULSE_NET_TRIGGER``) relative to the
    first ``RECORDING_START`` row (or the earliest row if that marker is
    absent). Returns ``None`` when the file, columns, or any on-event is
    missing.
    """
    try:
        df = pd.read_csv(
            sensors_csv,
            usecols=lambda c: c.strip().lower() in {"monons", "ofm_event"},
        )
    except (ValueError, FileNotFoundError, OSError, pd.errors.EmptyDataError) as exc:
        LOGGER.debug("Sensors-log read failed for %s: %s", sensors_csv, exc)
        return None

    mono_col = next((c for c in df.columns if c.strip().lower() == "monons"), None)
    event_col = next((c for c in df.columns if c.strip().lower() == "ofm_event"), None)
    if mono_col is None or event_col is None or df.empty:
        return None

    mono = pd.to_numeric(df[mono_col], errors="coerce")
    events = df[event_col].astype(str).str.strip()

    # Recording-start reference: the explicit marker if present, else the
    # earliest valid MonoNs in the file.
    start_mask = events.str.upper() == "RECORDING_START"
    start_vals = mono[start_mask & mono.notna()]
    if not start_vals.empty:
        start_ns = float(start_vals.iloc[0])
    else:
        valid = mono[mono.notna()]
        if valid.empty:
            return None
        start_ns = float(valid.min())

    on_mask = events.str.match(_LIGHT_ON_RE) & mono.notna()
    on_vals = mono[on_mask]
    if on_vals.empty:
        return None

    return (float(on_vals.iloc[0]) - start_ns) / 1e9


def find_sensors_csv(
    *,
    sidecar_csv: Optional[Path] = None,
    batch_dir: Optional[Path] = None,
    trial_type: Optional[str] = None,
    trial_index: Optional[int] = None,
) -> Optional[Path]:
    """Locate the ``sensors_output_*.csv`` sibling for a trial.

    The sensors log is named ``sensors_`` + the frame-sidecar CSV name, so when
    the frame sidecar is known we derive it directly. Otherwise we glob the
    batch folder by ``(trial_type, trial_index)``.
    """
    if sidecar_csv is not None:
        candidate = sidecar_csv.with_name(f"sensors_{sidecar_csv.name}")
        if candidate.exists():
            return candidate
    if batch_dir is not None and batch_dir.is_dir() and trial_type and trial_index is not None:
        pattern = f"sensors_output_*_{trial_type.lower()}_{int(trial_index)}_*.csv"
        candidates = list(batch_dir.glob(pattern))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


# ---------------------------------------------------------------------------
# Sidecar discovery
# ---------------------------------------------------------------------------

def _trial_stem(trial_dir: Path) -> str:
    """Return the cycle-stem used by the sidecar (``<batch>_<type>_<N>``).

    Example: trial folder ``may_15_batch_2_training_5`` → returns
    ``may_15_batch_2_training_5``. This is the substring that appears inside
    the sidecar filename right after ``output_``.
    """
    return trial_dir.name


def find_sidecar(trial_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Locate the sidecar ``.txt`` and its sibling ``.csv`` for ``trial_dir``.

    Searches the trial folder's parent (the batch folder) for files matching
    ``output_<trial_stem>_*.txt``. Returns ``(txt, csv)`` paths, either of
    which may be ``None`` if the rig did not produce it.
    """
    batch = trial_dir.parent
    stem = _trial_stem(trial_dir)
    pattern = f"output_{stem}_*"
    txts: list[Path] = []
    csvs: list[Path] = []
    for path in batch.glob(pattern):
        if not _SIDECAR_RE.match(path.name) and path.suffix.lower() != ".csv":
            continue
        if path.suffix.lower() == ".txt":
            txts.append(path)
        elif path.suffix.lower() == ".csv":
            csvs.append(path)
    # Prefer the most recently modified if duplicates exist.
    txt = max(txts, key=lambda p: p.stat().st_mtime) if txts else None
    csv = max(csvs, key=lambda p: p.stat().st_mtime) if csvs else None
    return txt, csv


def find_sidecar_by_label(
    batch_dir: Path,
    trial_type: str,
    trial_index: int,
) -> tuple[Optional[Path], Optional[Path]]:
    """Locate the sidecar pair when only ``(trial_type, trial_index)`` is known.

    Used by callers that hold an envelope-CSV path (e.g.
    ``<batch>/angle_distance_rms_envelope/training_10_citral_fly1_…csv``)
    instead of the per-trial folder. Globs the batch directory for files
    matching ``output_<…>_<trial_type>_<N>_*.{txt,csv}`` and returns the
    most-recent pair found.
    """
    if not batch_dir.is_dir() or not trial_type or trial_index is None:
        return None, None
    pattern = f"output_*_{trial_type.lower()}_{int(trial_index)}_*"
    txts: list[Path] = []
    csvs: list[Path] = []
    for path in batch_dir.glob(pattern):
        if path.name.startswith("sensors_"):
            continue
        if path.suffix.lower() == ".txt":
            txts.append(path)
        elif path.suffix.lower() == ".csv":
            csvs.append(path)
    txt = max(txts, key=lambda p: p.stat().st_mtime) if txts else None
    csv = max(csvs, key=lambda p: p.stat().st_mtime) if csvs else None
    return txt, csv


# ---------------------------------------------------------------------------
# Dataset / cycle inference
# ---------------------------------------------------------------------------

def infer_dataset_name(trial_dir: Path, known_datasets: Sequence[str]) -> str:
    """Return the dataset name from ``trial_dir`` by matching path components."""
    parts = set(trial_dir.parts)
    # exact match wins
    for ds in known_datasets:
        if ds in parts:
            return ds
    # case-insensitive fallback
    lower = {p.lower(): p for p in parts}
    for ds in known_datasets:
        if ds.lower() in lower:
            return ds
    # last resort: name of the directory just below the dataset roots
    # convention: <data_base>/<dataset>/<batch>/<trial>
    return trial_dir.parts[-3] if len(trial_dir.parts) >= 3 else ""


def _parse_cycle_name(cycle: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    if not cycle:
        return None, None, None
    m = _CYCLE_RE.match(cycle.strip())
    if not m:
        return None, None, None
    return m.group("type").lower(), int(m.group("index")), m.group("odor")


def _odor_from_filename(path: Path) -> Optional[str]:
    """Pull the odor token out of ``output_<batch>_<type>_<N>_<Odor>_<ts>.<ext>``."""
    m = re.match(
        r"^output_.+?_(?:training|testing)_\d+_(?P<odor>[A-Za-z0-9._\-]+?)_\d{8}_\d{6}\.[a-z0-9]+$",
        path.name,
        re.IGNORECASE,
    )
    return m.group("odor") if m else None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(trial_dir: Path) -> Path:
    return trial_dir / CACHE_FILENAME


def _load_cache(trial_dir: Path, sidecar_mtime: Optional[float]) -> Optional[dict]:
    path = _cache_path(trial_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    cached_mtime = data.get("_sidecar_mtime")
    if sidecar_mtime is not None and cached_mtime != sidecar_mtime:
        return None
    return data


def _save_cache(trial_dir: Path, meta: TrialMetadata, sidecar_mtime: Optional[float]) -> None:
    payload = meta.to_dict()
    payload["_sidecar_mtime"] = sidecar_mtime
    try:
        _cache_path(trial_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        LOGGER.debug("Failed to write trial-meta cache for %s: %s", trial_dir, exc)


def _meta_from_cache(data: dict) -> TrialMetadata:
    return TrialMetadata(
        dataset=data["dataset"],
        batch=Path(data["batch"]),
        trial_dir=Path(data["trial_dir"]),
        cycle_name=data["cycle_name"],
        trial_index=int(data["trial_index"]),
        odor=data.get("odor"),
        trial_type=data["trial_type"],
        raw_trial_type=data.get("raw_trial_type", data["trial_type"]),
        n_frames=int(data["n_frames"]),
        fps=float(data["fps"]),
        odor_on_frame=(int(data["odor_on_frame"]) if data.get("odor_on_frame") is not None else None),
        odor_off_frame=(int(data["odor_off_frame"]) if data.get("odor_off_frame") is not None else None),
        sidecar_txt=(Path(data["sidecar_txt"]) if data.get("sidecar_txt") else None),
        sidecar_csv=(Path(data["sidecar_csv"]) if data.get("sidecar_csv") else None),
        source="cache",
        light_on_s=(float(data["light_on_s"]) if data.get("light_on_s") is not None else None),
    )


# ---------------------------------------------------------------------------
# Override resolution
# ---------------------------------------------------------------------------

def _apply_trial_type_override(raw: Optional[str], override: object) -> str:
    """Resolve final trial_type from raw label + dataset override.

    ``override`` may be a string ("testing"/"training") or any object whose
    ``trial_type_override`` attribute / key is set. Anything else means "keep
    the raw label". Raw label is folder-based when no sidecar exists.
    """
    if override is None:
        return (raw or "").lower()
    # Plain string
    if isinstance(override, str):
        if override.strip().lower() in ("training", "testing"):
            return override.strip().lower()
        return (raw or "").lower()
    # Dataclass / dict-like
    forced: Optional[str] = None
    if hasattr(override, "trial_type_override"):
        forced = getattr(override, "trial_type_override")
    elif isinstance(override, dict):
        forced = override.get("trial_type_override")
    if isinstance(forced, str) and forced.strip().lower() in ("training", "testing"):
        return forced.strip().lower()
    return (raw or "").lower()


def _override_odor_window(override: object) -> tuple[Optional[float], Optional[float]]:
    if override is None:
        return None, None
    if hasattr(override, "odor_on_s"):
        return getattr(override, "odor_on_s", None), getattr(override, "odor_off_s", None)
    if isinstance(override, dict):
        return override.get("odor_on_s"), override.get("odor_off_s")
    return None, None


def _fallback_trial_type_from_folder(trial_dir: Path) -> Optional[str]:
    m = re.search(r"(training|testing)", trial_dir.name, re.IGNORECASE)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_trial_metadata(
    trial_dir: Path,
    *,
    fps_default: float = 40.0,
    odor_on_s_default: float = 30.0,
    odor_off_s_default: float = 60.0,
    n_frames_fallback: Optional[int] = None,
    dataset_override: object = None,
    known_datasets: Sequence[str] = (),
    use_cache: bool = True,
    sidecar_pair: Optional[tuple[Optional[Path], Optional[Path]]] = None,
) -> TrialMetadata:
    """Return per-trial metadata, preferring sidecar over fallbacks.

    Parameters
    ----------
    trial_dir
        Path to the *per-trial* folder (e.g. ``.../may_15_batch_2_training_5``).
    fps_default, odor_on_s_default, odor_off_s_default
        Used only when no sidecar / no ActiveOFM data is available.
    n_frames_fallback
        Used when the sidecar is missing and the caller already knows the trial
        length (typically ``len(distances_csv)``).
    dataset_override
        A ``DatasetOverride`` dataclass / dict / plain string holding
        ``trial_type_override`` and optional ``odor_on_s`` / ``odor_off_s``.
    known_datasets
        Dataset names from ``cfg.datasets``; used to infer the dataset for
        ``trial_dir``.
    use_cache
        If True, read/write ``_trial_meta.json`` inside the trial folder so
        re-runs skip the sidecar parse.
    """
    trial_dir = Path(trial_dir).resolve()
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"trial_dir does not exist: {trial_dir}")

    if sidecar_pair is not None:
        sidecar_txt, sidecar_csv = sidecar_pair
    else:
        sidecar_txt, sidecar_csv = find_sidecar(trial_dir)
    sidecar_mtime = sidecar_txt.stat().st_mtime if sidecar_txt else None

    if use_cache:
        cached = _load_cache(trial_dir, sidecar_mtime)
        if cached:
            try:
                return _meta_from_cache(cached)
            except (KeyError, TypeError) as exc:
                LOGGER.debug("Stale trial-meta cache at %s: %s", trial_dir, exc)

    dataset = infer_dataset_name(trial_dir, known_datasets)
    batch = trial_dir.parent

    # Default values
    cycle_name = ""
    trial_index: Optional[int] = None
    cycle_odor: Optional[str] = None
    raw_trial_type: Optional[str] = None
    n_frames: Optional[int] = None
    fps: Optional[float] = None
    odor_on_frame: Optional[int] = None
    odor_off_frame: Optional[int] = None
    source = "fallback"

    # --- Sidecar txt ---
    if sidecar_txt is not None:
        try:
            fields = parse_sidecar(sidecar_txt)
        except OSError as exc:
            LOGGER.warning("Cannot read sidecar %s: %s", sidecar_txt, exc)
            fields = {}
        cycle_name = fields.get("cycle_name", "")
        raw_trial_type, trial_index, cycle_odor = _parse_cycle_name(cycle_name)
        n_frames = _coerce_int(fields.get("total_frames_captured"))
        fps = _coerce_float(fields.get("achieved_frame_rate"))
        source = "sidecar"

    # --- Sidecar csv (ActiveOFM) ---
    if sidecar_csv is not None:
        on_f, off_f, _ = parse_active_ofm(sidecar_csv)
        odor_on_frame = on_f
        odor_off_frame = off_f

    # --- Filename fallback for odor ---
    if not cycle_odor and sidecar_txt is not None:
        cycle_odor = _odor_from_filename(sidecar_txt)
    if not cycle_odor and sidecar_csv is not None:
        cycle_odor = _odor_from_filename(sidecar_csv)

    # --- Folder fallback for trial type ---
    if not raw_trial_type:
        raw_trial_type = _fallback_trial_type_from_folder(trial_dir)
    if trial_index is None:
        m = re.search(r"(?:training|testing)_(\d+)", trial_dir.name, re.IGNORECASE)
        if m:
            trial_index = int(m.group(1))

    # --- Fallbacks for length / fps ---
    if fps is None or fps <= 0:
        fps = float(fps_default)
    if n_frames is None or n_frames <= 0:
        if n_frames_fallback is not None and n_frames_fallback > 0:
            n_frames = int(n_frames_fallback)
        else:
            # Assume the historical 90 s trial when the rig produced no sidecar
            # and the caller did not pass an actual frame count.
            n_frames = int(round((odor_off_s_default + odor_on_s_default) * fps))

    # --- Odor window fallbacks: dataset override -> config defaults ---
    if odor_on_frame is None or odor_off_frame is None:
        override_on_s, override_off_s = _override_odor_window(dataset_override)
        on_s = override_on_s if override_on_s is not None else odor_on_s_default
        off_s = override_off_s if override_off_s is not None else odor_off_s_default
        if odor_on_frame is None:
            odor_on_frame = int(round(on_s * fps))
        if odor_off_frame is None:
            odor_off_frame = int(round(off_s * fps))

    # Clamp to trial length to avoid nonsensical windows
    odor_on_frame = max(0, min(int(odor_on_frame), n_frames - 1))
    odor_off_frame = max(odor_on_frame + 1, min(int(odor_off_frame), n_frames))

    trial_type = _apply_trial_type_override(raw_trial_type, dataset_override)
    if not trial_type:
        trial_type = "testing"  # safest default; matches historical pipeline behaviour

    # --- Optogenetic light-on time from the sensors log ---
    # Uses the *raw* (filename) trial type, since the sensors filename mirrors
    # the rig cycle name, not any dataset trial_type override.
    sensors_csv = find_sensors_csv(
        sidecar_csv=sidecar_csv,
        batch_dir=batch,
        trial_type=raw_trial_type,
        trial_index=trial_index,
    )
    light_on_s = parse_light_on_seconds(sensors_csv) if sensors_csv is not None else None

    meta = TrialMetadata(
        dataset=dataset,
        batch=batch,
        trial_dir=trial_dir,
        cycle_name=cycle_name or (f"{raw_trial_type}_{trial_index}" if raw_trial_type and trial_index else trial_dir.name),
        trial_index=int(trial_index) if trial_index is not None else 0,
        odor=cycle_odor,
        trial_type=trial_type,
        raw_trial_type=(raw_trial_type or "").lower(),
        n_frames=int(n_frames),
        fps=float(fps),
        odor_on_frame=int(odor_on_frame),
        odor_off_frame=int(odor_off_frame),
        sidecar_txt=sidecar_txt,
        sidecar_csv=sidecar_csv,
        source=source,
        light_on_s=light_on_s,
    )

    if use_cache:
        _save_cache(trial_dir, meta, sidecar_mtime)

    return meta
