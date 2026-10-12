"""Automatic dataset freezing: rebuild a dataset only when something new arrived.

The policy (the user, 2026-10-06): a dataset is unfrozen and rebuilt when
something new was added to it, and frozen otherwise -- unless the user
explicitly unfreezes it (``freeze: {data: false}`` in config, or ``--thaw`` /
``--thaw-all`` for one run). A frozen dataset runs no YOLO, no per-trial steps
and no combine; its wide rows splice in whole from the freeze cache
(``fbpipe.freeze``) and its figures are left alone.

"Something new" is decided per dataset root at the start of a run:

* a batch folder appeared or disappeared;
* a raw rig file directly in a batch folder (``output_*``, ``sensors_output_*``,
  ``session_metadata.txt``, ``odor_vials.csv``, hand notes) was added, changed
  or removed -- by content: a file only touched (same bytes) changes nothing;
* a video exists that YOLO would still process (exactly yolo_infer's discovery,
  ``fbpipe.utils.yolo_discovery``).

Files the pipeline writes never count: trial folders and every other directory
below a batch, the per-fly ``*global_distance_stats_class_*.json`` it writes
beside the rig files, and videos (the secure cleanup deletes processed ones).

A dataset with no record yet, a record left ``thawing`` by a run that died, or a
freeze cache that fails the current fingerprint (a config edit that changes how
rows are built) is rebuilt once -- with folder retirements and cutoffs live. A
dataset frozen today (legacy ``data: true``) whose cache is valid is ADOPTED on
its first run instead: its record is taken from the files as they are, and its
cached rows lose their byte-identical duplicate rows (lossless) once.

Records live under ``<cache_dir>/frozen/_auto/<config scope>/``, one JSON file
per dataset, written atomically. A rebuilt dataset is marked ``thawing`` before
any work and recorded ``frozen`` -- with the inputs snapshotted at the START of
the run -- only after the whole run (data and figures) finished, and only from
cached rows this run's walk wrote (``save_slice(built_by=run_id)``); anything
that arrives later is new to the next run. Cascades are reported, never acted
on: a frozen pair arm whose partner got new data, and frozen datasets built
under other per-trial code or settings, are listed with the command that
rebuilds them. So are trials recorded more than once, trials with rig files
but no trial folder and no local video, and adopted rows whose folder marks
today's rules would change on a rebuild.

The plan is shared with figure subprocesses through ``FBPIPE_FREEZE_PLAN`` (a
JSON file); ``fbpipe.freeze.freeze_flags`` and ``figure_frozen_from_raw`` consult
it first, so no figure script needs to know about auto-freeze.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

RECORD_SCHEMA = 1
PLAN_SCHEMA = 1
PLAN_ENV = "FBPIPE_FREEZE_PLAN"

#: Video containers are never inputs (``yolo_discovery`` decides pending videos).
_VIDEO_SUFFIXES = (".mp4", ".avi", ".h264", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v")

#: Names the pipeline itself writes directly inside a batch folder. Besides
#: distance_stats' per-fly JSON: yolo_infer's batch eye template (main, 0636c69)
#: and its atomic-write temp, written the first time YOLO processes a batch --
#: counted, every new batch would be rebuilt a second, pointless time.
_PIPELINE_WRITTEN = re.compile(
    r"^(?:fly\d+_)?global_distance_stats_class_\d+\.json$"
    r"|^batch_manifest\.json$"
    r"|^_trial_meta\.json$"
    r"|^eye_anchor_template\.json(?:\.\d+\.tmp)?$",
    re.IGNORECASE,
)

_TRIAL_DIR = re.compile(r"_(pretest|testing|training)_(\d+)$", re.IGNORECASE)
_FLY_TABLE = re.compile(r"_fly\d+_distances\.(parquet|csv)$", re.IGNORECASE)

#: Where the commands in the cascade report point. The live nightly config.
_REBUILD_PREFIX = "python3 scripts/pipeline/run_workflows.py --config {config}"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    """tmp + fsync + rename: a crash leaves the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").strip() or "_"


def scope_dir(cache_dir: str | Path, config_path: str | Path) -> Path:
    """Records of one config: ``<cache>/frozen/_auto/<12 hex of the config path>``.

    ``config.yaml`` and ``config_new.yaml`` share one cache_dir but freeze
    different things, so their records never mix.
    """
    resolved = str(Path(config_path).expanduser().resolve())
    digest = hashlib.blake2b(resolved.encode("utf-8"), digest_size=6).hexdigest()
    return Path(cache_dir).expanduser() / "frozen" / "_auto" / digest


def record_path(scope: Path, dataset: str) -> Path:
    return scope / f"{_safe(dataset)}.json"


def load_record(scope: Path, dataset: str) -> Optional[dict]:
    path = record_path(scope, dataset)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 -- unreadable == no record: rebuild once
        return {"schema": None, "state": "unreadable"}
    return data if isinstance(data, dict) else {"schema": None, "state": "unreadable"}


def save_record(scope: Path, dataset: str, record: Mapping[str, Any]) -> None:
    _atomic_write_text(record_path(scope, dataset), json.dumps(record, indent=1, sort_keys=True))


# --------------------------------------------------------------------------
# input signature
# --------------------------------------------------------------------------


def _counts_as_input(name: str) -> bool:
    if name.startswith((".", "~")):
        return False
    if name.lower().endswith(_VIDEO_SUFFIXES):
        return False
    return _PIPELINE_WRITTEN.match(name) is None


def scan_inputs(root: str | Path, previous: Optional[Mapping[str, Any]] = None) -> dict:
    """The dataset's input signature: its batch folders and their raw files.

    ``{"dirs": [names], "files": {"<batch>/<file>": [size, mtime_ns, ctime_ns,
    ino, blake2b]}}``. A file is hashed only when its stat differs from
    *previous* (or is new), so a quiet night reads no file contents; the
    ctime/inode in the stat catch ``cp -p`` / ``rsync -t`` / ``touch -r``,
    which all keep the mtime. Hashing happens now, so the stored stat and hash
    always describe the same bytes.
    """
    base = Path(root)
    old_files = (previous or {}).get("files") or {}
    dirs: list[str] = []
    files: dict[str, list] = {}
    if not base.is_dir():
        return {"dirs": dirs, "files": files}
    with os.scandir(base) as entries:
        batch_dirs = sorted(
            (e for e in entries if e.is_dir() and not e.name.startswith((".", "~"))),
            key=lambda e: e.name,
        )
    for entry in batch_dirs:
        dirs.append(entry.name)
        try:
            with os.scandir(entry.path) as inner:
                candidates = sorted(
                    (f for f in inner if f.is_file() and _counts_as_input(f.name)),
                    key=lambda f: f.name,
                )
        except OSError:
            continue
        for f in candidates:
            rel = f"{entry.name}/{f.name}"
            try:
                st = f.stat()
            except OSError:
                continue
            stat_key = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]
            old = old_files.get(rel)
            if old and list(old[:4]) == stat_key and old[4]:
                digest = old[4]
            else:
                try:
                    digest = _file_hash(Path(f.path))
                except OSError:
                    continue
            files[rel] = [*stat_key, digest]
    return {"dirs": dirs, "files": files}


@dataclasses.dataclass
class InputDiff:
    added_dirs: list[str]
    removed_dirs: list[str]
    added: list[str]
    removed: list[str]
    changed: list[str]
    touched: list[str]

    @property
    def is_new(self) -> bool:
        """Anything but a touch: the user's "something new was added"."""
        return bool(self.added_dirs or self.removed_dirs or self.added or self.removed or self.changed)

    def describe(self) -> str:
        parts = []
        if self.added_dirs:
            parts.append(f"+{len(self.added_dirs)} batch folder(s) ({', '.join(self.added_dirs[:3])}"
                         + (", ..." if len(self.added_dirs) > 3 else "") + ")")
        if self.removed_dirs:
            parts.append(f"-{len(self.removed_dirs)} batch folder(s) ({', '.join(self.removed_dirs[:3])}"
                         + (", ..." if len(self.removed_dirs) > 3 else "") + ")")
        for label, items in (("+", self.added), ("~", self.changed), ("-", self.removed)):
            if items:
                parts.append(f"{label}{len(items)} raw file(s) ({', '.join(items[:2])}"
                             + (", ..." if len(items) > 2 else "") + ")")
        return "; ".join(parts) or "no change"


def diff_inputs(old: Optional[Mapping[str, Any]], new: Mapping[str, Any]) -> InputDiff:
    old = old or {"dirs": [], "files": {}}
    old_dirs, new_dirs = set(old.get("dirs") or ()), set(new.get("dirs") or ())
    old_files, new_files = old.get("files") or {}, new.get("files") or {}
    added = sorted(set(new_files) - set(old_files))
    removed = sorted(set(old_files) - set(new_files))
    changed, touched = [], []
    for rel in sorted(set(new_files) & set(old_files)):
        o, n = old_files[rel], new_files[rel]
        if not o[4] or not n[4] or o[4] != n[4] or o[0] != n[0]:
            if o[:5] != n[:5]:
                changed.append(rel)
        elif o[:4] != n[:4]:
            touched.append(rel)
    return InputDiff(
        added_dirs=sorted(new_dirs - old_dirs),
        removed_dirs=sorted(old_dirs - new_dirs),
        added=added,
        removed=removed,
        changed=changed,
        touched=touched,
    )


def processed_batches(root: str | Path) -> set[str]:
    """Batch folders holding at least one trial folder with a per-fly table.

    What a frozen dataset's cached rows must cover: a batch YOLO processed
    after the dataset was frozen is missing from them.
    """
    out: set[str] = set()
    base = Path(root)
    if not base.is_dir():
        return out
    for batch in base.iterdir():
        if not batch.is_dir():
            continue
        try:
            trials = [t for t in batch.iterdir() if t.is_dir() and _TRIAL_DIR.search(t.name)]
        except OSError:
            continue
        for trial in trials:
            try:
                if any(_FLY_TABLE.search(name) for name in os.listdir(trial)):
                    out.add(batch.name)
                    break
            except OSError:
                continue
    return out


# --------------------------------------------------------------------------
# provenance (report only)
# --------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parents[1]

#: Code that derives a dataset's rows: the per-trial steps, the helpers they
#: share, and combine + the wide build. Files that cannot change a row
#: (notifications, backups, scoring, curation) are left out to keep the report
#: quiet. Report only: never thaws anything (the user's rule).
_CODE_GLOBS = ("src/fbpipe/pipeline.py", "src/fbpipe/steps/*.py", "src/fbpipe/utils/*.py",
               "src/fbpipe/analysis/threshold.py", "scripts/analysis/envelope_combined.py")
_CODE_SKIP = {
    "__init__.py", "predict_reactions.py", "reaction_matrix.py", "pseudolabel_export.py",
    "curate_yolo_dataset.py", "compose_videos_rms.py", "check_light_stimulus.py", "move_videos.py",
    "notify.py", "smb_copy.py", "data_secured_sync.py", "pseudolabel.py", "augmentation.py",
    "parallel.py", "video_writer.py", "frozen_folders.py", "yolo_discovery.py",
    "gpu_batch_optimizer.py", "run_lock.py", "light_stimulus.py",
}

#: Settings fields the per-trial steps read.
_SETTING_FIELDS = (
    "model_path", "conf_thres", "iou_match_thres", "max_age", "ema_alpha", "use_optical_flow",
    "flow_skip_edge", "max_flies", "max_proboscis_tracks", "proboscis_match_max_dist_px",
    "pair_rebind_ratio", "zero_iou_epsilon", "class2_min", "class2_max",
    "three_fly_max_eye_prob_distance_px", "rig_gate_overrides", "proboscis_filter",
    "non_reactive_span_px", "fps_default", "window_sec", "odor_on_s", "odor_off_s",
    "odor_latency_s", "protocol",
)


def code_provenance(repo: Path = _REPO) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in _CODE_GLOBS:
        for path in sorted(repo.glob(pattern)):
            if path.name in _CODE_SKIP or not path.is_file():
                continue
            out[path.relative_to(repo).as_posix()] = _file_hash(path)
    return out


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (_dt.date, _dt.datetime, Path)):
        return str(value)
    return value


def settings_provenance(settings: Any) -> dict[str, Any]:
    return {name: _jsonable(getattr(settings, name, None)) for name in _SETTING_FIELDS}


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Decision:
    dataset: str
    root: str
    mode: str                    # "auto" | "live"
    decision: str                # "frozen" | "thawed" | "live" | "untouched"
    reasons: list[str]
    figures_frozen: bool = False
    commit: bool = False         # a successful run records it frozen
    inputs: Optional[dict] = None
    adopted: bool = False
    deduped: dict = dataclasses.field(default_factory=dict)
    pending: int = 0
    figures_mode: str = "auto"
    record: Optional[dict] = None

    @property
    def data_frozen(self) -> bool:
        return self.decision == "frozen"


class Plan:
    """This run's freeze decisions, plus the records they rest on."""

    def __init__(self, *, settings: Any, config_path: Path, scope: Path, run_id: str,
                 blocks: Sequence[tuple], fingerprint: Callable[[str, Mapping], dict],
                 read_only: bool = False):
        self.settings = settings
        self.config_path = Path(config_path)
        self.scope = scope
        self.run_id = run_id
        self.blocks = list(blocks)
        self.fingerprint = fingerprint
        self.read_only = read_only
        self.decisions: dict[str, Decision] = {}
        self.notices: list[str] = []
        self.path = scope / "_plans" / f"{run_id}.json"
        self.code = code_provenance()
        self.settings_prov = settings_provenance(settings)

    # -- lookups ---------------------------------------------------------

    def decision_for_root(self, root: str | Path) -> Optional[Decision]:
        return self.decisions.get(Path(str(root)).name)

    def consulted(self, dataset: str) -> list[tuple[str, dict]]:
        return [(block, dict(kw)) for block, names, kw in self.blocks if dataset in names]

    # -- persistence -------------------------------------------------------

    def as_json(self) -> dict:
        return {
            "schema": PLAN_SCHEMA,
            "run_id": self.run_id,
            "config": str(self.config_path),
            "datasets": {
                name: {
                    "data_frozen": d.data_frozen,
                    "figures_frozen": d.figures_frozen,
                    "decision": d.decision,
                    "reasons": list(d.reasons),
                }
                for name, d in self.decisions.items()
            },
            "notices": list(self.notices),
        }

    def write(self) -> Path:
        """Write the plan file the figure subprocesses read (atomic).

        The last 30 runs' plans are kept, for looking back at a decision.
        """
        _atomic_write_text(self.path, json.dumps(self.as_json(), indent=1, sort_keys=True))
        try:
            plans = sorted(self.path.parent.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
            for old in plans[:-30]:
                old.unlink(missing_ok=True)
        except OSError:
            pass
        return self.path

    def _thawing_record(self, d: Decision, reasons: Sequence[str]) -> dict:
        record = dict(d.record) if isinstance(d.record, dict) and d.record.get("schema") else {
            "schema": RECORD_SCHEMA, "dataset": d.dataset, "inputs": None,
        }
        record.update(
            schema=RECORD_SCHEMA,
            dataset=d.dataset,
            root=d.root,
            config=str(self.config_path),
            state="thawing",
            thawing={"run_id": self.run_id, "at": _now(), "reasons": list(reasons)},
        )
        return record

    def mark_thawing(self) -> None:
        """Before any work: every dataset this run rebuilds is ``thawing``.

        A run that dies anywhere after this leaves it so, and the next run
        rebuilds it -- a half-built dataset is never recorded frozen.
        """
        if self.read_only:
            return
        for d in self.decisions.values():
            if d.commit:
                save_record(self.scope, d.dataset, self._thawing_record(d, d.reasons))

    def late_thaw(self, root: str | Path, pending: Sequence[Path]) -> Optional[Decision]:
        """A frozen dataset whose YOLO slot found new videos: rebuild it now."""
        d = self.decision_for_root(root)
        if d is None or d.decision != "frozen":
            return d
        d.decision = "thawed"
        d.reasons = [f"late: {len(pending)} video(s) YOLO has not processed arrived during the run"]
        d.pending = len(pending)
        d.commit = True
        d.figures_frozen = _figures_frozen(d)
        d.inputs = scan_inputs(d.root, (d.record or {}).get("inputs"))
        if not self.read_only:
            save_record(self.scope, d.dataset, self._thawing_record(d, d.reasons))
            self.write()
        return d

    # -- the end of a successful run ---------------------------------------------

    def _slices_current(self, dataset: str, root: str) -> tuple[bool, str]:
        """Whether every wide table *dataset* feeds now caches THIS run's rows.

        Valid under today's fingerprint is not enough: yesterday's slice of a
        dataset that got new data still validates, and recording it frozen
        when this run's rows never reached the cache (a failed cache write)
        would drop the new batch for good. So the slice must carry this run's
        stamp (``save_slice(built_by=...)``, written by the cache writer).
        """
        from . import freeze as _freeze

        for block, kw in self.consulted(dataset):
            fingerprint = self.fingerprint(dataset, kw)
            ok, why = _freeze.slice_status(self.settings.cache_dir, block, dataset, fingerprint)
            if not ok and why == "missing" and not processed_batches(root):
                # Nothing processed, so the wide build emitted no rows and the
                # cache writer had nothing to save: an empty slice is the truth.
                # Without it an empty dataset would rebuild every night.
                import pandas as pd

                _freeze.save_slice(self.settings.cache_dir, block, dataset, pd.DataFrame(), fingerprint,
                                   built_by=self.run_id)
                continue
            if not ok:
                return False, f"{block}: {why}"
            if _freeze.slice_built_by(self.settings.cache_dir, block, dataset) != self.run_id:
                return False, f"{block}: not rewritten by this run (its rebuilt rows never reached the cache)"
        return True, ""

    def _rows_by_type(self, dataset: str) -> dict:
        from . import freeze as _freeze

        out: dict[str, Optional[dict]] = {}
        for block, _ in self.consulted(dataset):
            try:
                import pyarrow.parquet as pq

                rows = _freeze.slice_dir(self.settings.cache_dir, block, dataset) / "rows.parquet"
                col = pq.read_table(rows, columns=["trial_type"]).column("trial_type").to_pylist()
                counts: dict[str, int] = {}
                for value in col:
                    counts[str(value)] = counts.get(str(value), 0) + 1
                out[block] = dict(sorted(counts.items()))
            except Exception:  # noqa: BLE001 -- a count is for the log only
                out[block] = None
        return out

    def commit(self, *, combined_ran: bool) -> list[str]:
        """Record every dataset this run rebuilt as frozen. Returns log lines.

        Called once the whole run -- data, figures -- finished. A dataset whose
        cached rows are not current for every wide table it is in stays
        ``thawing`` and is rebuilt next run.
        """
        lines: list[str] = []
        if self.read_only:
            return lines
        for d in self.decisions.values():
            if d.decision == "frozen":
                if isinstance(d.record, dict) and d.record.get("figures_pending"):
                    record = dict(d.record)
                    record.pop("figures_pending", None)
                    save_record(self.scope, d.dataset, record)
                continue
            if not d.commit:
                continue
            if not combined_ran:
                lines.append(f"[FREEZE] {d.dataset}: NOT refrozen - the combined stage did not run; rebuilt next run")
                continue
            ok, why = self._slices_current(d.dataset, d.root)
            if not ok:
                lines.append(f"[FREEZE] {d.dataset}: NOT refrozen - cached rows not current ({why}); rebuilt next run")
                continue
            rows = self._rows_by_type(d.dataset)
            previous = (d.record or {}).get("rows") if isinstance(d.record, dict) else None
            record = {
                "schema": RECORD_SCHEMA,
                "dataset": d.dataset,
                "root": d.root,
                "config": str(self.config_path),
                "state": "frozen",
                "mode": d.mode,
                "inputs": d.inputs,
                "built": {"run_id": self.run_id, "at": _now(), "reasons": list(d.reasons)},
                "provenance": {"code": self.code, "settings": self.settings_prov},
                "rows": rows,
            }
            save_record(self.scope, d.dataset, record)
            change = ""
            if previous and previous != rows:
                change = f" (rows {previous} -> {rows})"
            if d.decision == "live":
                # Its record is kept current, so dropping the explicit unfreeze
                # freezes it on the next run unless something new arrived.
                lines.append(f"[FREEZE] {d.dataset}: stays live (freeze.data: false); record refreshed, so "
                             f"removing data: false freezes it with no rebuild{change}")
            else:
                lines.append(f"[FREEZE] {d.dataset}: recorded frozen after its rebuild{change}")
        return lines

    # -- reporting -----------------------------------------------------------

    def summary_lines(self) -> list[str]:
        counts: dict[str, int] = {}
        for d in self.decisions.values():
            counts[d.decision] = counts.get(d.decision, 0) + 1
        head = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "no datasets"
        lines = [f"[FREEZE] plan {self.run_id}: {head}"]
        for name in sorted(self.decisions):
            d = self.decisions[name]
            tag = d.decision.upper()
            extra = "; figures held" if d.figures_frozen else ""
            lines.append(f"[FREEZE] {name}: {tag} - {'; '.join(d.reasons)}{extra}")
        lines.extend(self.notices)
        return lines

    def short_summary(self) -> str:
        counts: dict[str, int] = {}
        for d in self.decisions.values():
            counts[d.decision] = counts.get(d.decision, 0) + 1
        text = "auto-freeze: " + (", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "nothing")
        if self.notices:
            text += f"; {len(self.notices)} notice(s) - see {self.scope / 'status.txt'}"
        return text

    def write_status(self, extra: Sequence[str] = ()) -> Path:
        path = self.scope / "status.txt"
        if not self.read_only:
            _atomic_write_text(path, "\n".join([*self.summary_lines(), *extra]) + "\n")
        return path


# --------------------------------------------------------------------------
# slices: validation, dedupe and the policy-only re-stamp
# --------------------------------------------------------------------------

_POLICY_TOP = {"freeze_folders_before", "freeze_folders_born_on_or_after", "keep_folders_born_on_or_after"}
_POLICY_OVERRIDE = {"freeze_folders", "cutoff_exempt_before"}


def _policy_only_drift(stored: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """True when the two fingerprints differ ONLY in folder-policy fields."""
    if not isinstance(stored, Mapping) or not isinstance(current, Mapping):
        return False
    for key in set(stored) | set(current):
        if key in _POLICY_TOP:
            continue
        if key == "override":
            a, b = stored.get(key) or {}, current.get(key) or {}
            for sub in set(a) | set(b):
                if sub not in _POLICY_OVERRIDE and a.get(sub) != b.get(sub):
                    return False
            continue
        if stored.get(key) != current.get(key):
            return False
    return True


def _mark_flips(settings: Any, dataset: str, root: Path, rows_path: Path) -> tuple[list[str], list[str], str]:
    """``(retire, revive, problem)``: how today's folder rules would re-mark a slice.

    *retire*: folders the slice shows live that the rules retire; *revive*:
    folders it marks frozen that the rules no longer retire; *problem*: why the
    two cannot be compared (a folder marked both ways, or gone from disk).
    """
    import pyarrow.parquet as pq

    from .utils.frozen_folders import frozen_folders_for_root

    marked: dict[str, set[bool]] = {}
    if {"fly", "frozen"} <= set(pq.read_schema(rows_path).names):  # an empty slice marks nothing
        table = pq.read_table(rows_path, columns=["fly", "frozen"]).to_pandas()
        for fly, frozen in zip(table["fly"].astype(str), table["frozen"]):
            marked.setdefault(fly, set()).add(str(frozen).strip().lower() in {"true", "1", "1.0"})
    if any(len(v) > 1 for v in marked.values()):
        return [], [], "a folder is marked both ways"
    on_disk = {p.name for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()
    gone = sorted(set(marked) - on_disk)
    if gone:
        return [], [], f"folder(s) no longer on disk: {', '.join(gone[:3])}"
    retired = frozen_folders_for_root(settings, root, dataset=dataset) if marked else set()
    retire = sorted(f for f, v in marked.items() if f in retired and not next(iter(v)))
    revive = sorted(f for f, v in marked.items() if f not in retired and next(iter(v)))
    return retire, revive, ""


def _marks_unchanged(settings: Any, dataset: str, root: Path, rows_path: Path) -> tuple[bool, str]:
    """Whether today's folder rules mark exactly the folders the slice marks."""
    retire, revive, problem = _mark_flips(settings, dataset, root, rows_path)
    if problem:
        return False, problem
    flipped = sorted(retire + revive)
    if flipped:
        return False, f"the rules now mark {len(flipped)} folder(s) differently ({', '.join(flipped[:3])})"
    return True, ""


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def _override(settings: Any, dataset: str) -> Any:
    return (getattr(settings, "dataset_overrides", None) or {}).get(dataset)


def plan_run(
    settings: Any,
    *,
    config_path: str | Path,
    roots: Iterable[str | Path],
    blocks: Sequence[tuple],
    fingerprint: Callable[[str, Mapping], dict],
    thawed: Iterable[str] = (),
    thaw_all: bool = False,
    folder: Optional[str] = None,
    pairs: Sequence[tuple[str, str]] = (),
    run_id: Optional[str] = None,
    read_only: bool = False,
) -> Plan:
    """Decide, for every dataset root of this run, what it does tonight.

    Writes (unless *read_only*): adoption records, deduplicated slices of
    adopted datasets, policy-only re-stamps, touch-only stat refreshes and the
    ``thawing`` marks of every dataset that is rebuilt.
    """
    from . import freeze as _freeze
    from .utils.yolo_discovery import trial_report

    scope = scope_dir(settings.cache_dir, config_path)
    run_id = run_id or (_dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6])
    plan = Plan(settings=settings, config_path=Path(config_path).expanduser().resolve(), scope=scope,
                run_id=run_id, blocks=blocks, fingerprint=fingerprint, read_only=read_only)
    thawed_keys = {_freeze._freeze_key(t) for t in (thawed or ())}
    folder_key = folder.strip().lower() if folder else None

    for root in roots:
        root_path = Path(str(root)).expanduser()
        dataset = root_path.name
        if not dataset or dataset in plan.decisions:
            continue
        override = _override(settings, dataset)
        mode = getattr(override, "data_mode", "auto") or "auto"
        figures_mode = getattr(override, "figures_mode", "auto") or "auto"
        record = load_record(scope, dataset)
        d = Decision(dataset=dataset, root=str(root_path), mode=mode, decision="thawed",
                     reasons=[], figures_mode=figures_mode, record=record)
        plan.decisions[dataset] = d
        old_inputs = record.get("inputs") if isinstance(record, dict) else None
        inputs = scan_inputs(root_path, old_inputs)
        d.inputs = inputs
        trials = trial_report(settings, root_path)
        pending = trials.pending
        d.pending = len(pending)
        _trial_notices(plan, dataset, trials)
        explicit = thaw_all or _freeze._freeze_key(dataset) in thawed_keys

        if folder_key is not None and dataset.lower() == folder_key:
            d.decision, d.commit = "thawed", True
            d.reasons.append("--folder: rebuilt this run")
        elif explicit:
            d.decision, d.commit = "thawed", True
            d.reasons.append("--thaw-all" if thaw_all else "--thaw: rebuilt this run")
        elif mode == "live":
            d.decision, d.commit = "live", True
            d.reasons.append("freeze.data: false (explicitly live)")
        else:
            frozen, why = _auto_decision(plan, d, record, inputs, pending, override)
            if frozen:
                d.decision = "frozen"
                d.reasons.append(why)
            elif folder_key is not None:
                # --folder touches one dataset; the rest keep their records.
                d.decision = "untouched"
                d.reasons.append(f"{why}; not rebuilt (--folder {folder})")
            else:
                d.decision, d.commit = "thawed", True
                d.reasons.append(why)

        d.figures_frozen = _figures_frozen(d)
        if d.decision != "frozen":
            d.inputs = inputs
    _cascade_notices(plan, pairs, config_path)
    _missing_root_notices(plan, settings)
    if not read_only:
        plan.mark_thawing()
        plan.write()
        plan.write_status()
    return plan


def _figures_frozen(d: Decision) -> bool:
    if d.figures_mode == "live":
        return False
    if d.decision == "frozen":
        if d.deduped:
            return False  # its rows changed this run: redraw once
        return not (isinstance(d.record, dict) and d.record.get("figures_pending"))
    if d.decision in ("live", "untouched"):
        return d.figures_mode == "hold"
    return False


def _auto_decision(plan: Plan, d: Decision, record: Optional[dict], inputs: dict,
                   pending: Sequence[Path], override: Any) -> tuple[bool, str]:
    """``(frozen?, reason)`` for an AUTO dataset nobody thawed explicitly."""
    from . import freeze as _freeze

    settings = plan.settings
    # Every wide table the dataset feeds must splice it from a current slice.
    # (A dataset no wide table reads has nothing to splice: freezing it only
    # skips per-trial work whose inputs did not change.)
    slice_problem = ""
    for block, kw in plan.consulted(d.dataset):
        fp = plan.fingerprint(d.dataset, kw)
        ok, why = _freeze.slice_status(settings.cache_dir, block, d.dataset, fp)
        if not ok and why.startswith("fingerprint drift"):
            ok, why = _restamp_policy_drift(plan, d, block, fp)
        if not ok:
            slice_problem = f"{block} cache {why}"
            break

    pending_txt = f"{len(pending)} video(s) YOLO has not processed" if pending else ""

    if record is None:
        legacy_frozen = bool(getattr(override, "freeze_data", False))
        if not legacy_frozen:
            return False, "no freeze record yet: built once, then frozen"
        if slice_problem:
            return False, f"no freeze record yet and {slice_problem}: built once"
        if pending:
            return False, f"no freeze record yet; {pending_txt}"
        missing = sorted(processed_batches(d.root) - _slice_flies(plan, d.dataset))
        if missing:
            return False, (f"no freeze record yet; {len(missing)} processed batch folder(s) missing "
                           f"from its cached rows ({', '.join(missing[:3])})")
        _adopt(plan, d, inputs)
        return True, "adopted: frozen today with a valid cache" + (
            "; duplicate rows removed (" + ", ".join(
                f"{b} {n0}->{n1}" for b, (n0, n1) in d.deduped.items()) + ")"
            if d.deduped else ""
        )

    if record.get("schema") != RECORD_SCHEMA or record.get("state") == "unreadable":
        return False, "freeze record unreadable or from another version: rebuilt once"
    if record.get("state") == "thawing":
        run = (record.get("thawing") or {}).get("run_id", "?")
        return False, f"the previous rebuild did not finish (run {run})"
    if Path(str(record.get("root", ""))) != Path(d.root):
        return False, "freeze record belongs to another root: rebuilt once"
    if slice_problem:
        return False, f"{slice_problem}: rebuilt once (folder rules live)"
    diff = diff_inputs(record.get("inputs"), inputs)
    reasons = []
    if diff.is_new:
        reasons.append(f"new data: {diff.describe()}")
    if pending:
        reasons.append(pending_txt)
    if reasons:
        return False, "; ".join(reasons)
    if diff.touched and not plan.read_only:
        refreshed = dict(record)
        refreshed["inputs"] = inputs
        save_record(plan.scope, d.dataset, refreshed)
        d.record = refreshed
        return True, f"nothing new ({len(diff.touched)} file(s) touched, same content)"
    return True, "nothing new"


def _slice_flies(plan: Plan, dataset: str) -> set[str]:
    import pyarrow.parquet as pq

    from . import freeze as _freeze

    flies: set[str] = set()
    for block, _ in plan.consulted(dataset):
        rows = _freeze.slice_dir(plan.settings.cache_dir, block, dataset) / "rows.parquet"
        try:
            flies |= {str(f) for f in pq.read_table(rows, columns=["fly"]).column("fly").to_pylist()}
        except Exception:  # noqa: BLE001
            continue
    return flies


def _restamp_policy_drift(plan: Plan, d: Decision, block: str, fingerprint: dict) -> tuple[bool, str]:
    """A folder-policy edit that changes none of THIS dataset's marks.

    The rows a rebuild would produce are the cached ones, so the slice is
    re-stamped with today's fingerprint instead (e.g. adding
    ``cutoff_exempt_before`` to a finished dataset whose rows are all live).
    Any other drift, or any mark that would flip, rebuilds as before.
    """
    from . import freeze as _freeze

    target = _freeze.slice_dir(plan.settings.cache_dir, block, d.dataset)
    try:
        meta = json.loads((target / "meta.json").read_text())
    except Exception:  # noqa: BLE001
        return False, "fingerprint drift (meta unreadable)"
    stored = meta.get("fingerprint")
    current = json.loads(json.dumps(dict(fingerprint)))
    if not _policy_only_drift(stored, current):
        return False, "fingerprint drift: " + (_freeze.drift_reason(stored or {}, current) or "?")
    try:
        same, why = _marks_unchanged(plan.settings, d.dataset, Path(d.root), target / "rows.parquet")
    except Exception as exc:  # noqa: BLE001
        return False, f"fingerprint drift (folder policy) and marks unverifiable ({exc})"
    if not same:
        return False, f"fingerprint drift (folder policy): {why}"
    if not plan.read_only:
        meta["fingerprint"] = current
        _atomic_write_text(target / "meta.json", json.dumps(meta, indent=2, sort_keys=True))
    plan.notices.append(
        f"[FREEZE] {d.dataset}: {block} cache {'would be ' if plan.read_only else ''}re-stamped for a "
        f"folder-policy edit that marks no folder differently"
    )
    return True, ""


def _adoption_mark_notice(plan: Plan, d: Decision) -> None:
    """Say so when today's folder rules would re-mark an adopted dataset's rows.

    A data-frozen dataset's rows were built with the folder rules switched off,
    so its slice can disagree with the always-live rules under an unchanged
    fingerprint. It is adopted with its rows exactly as they are -- freezing
    keeps them -- but its first rebuild would apply the rules.
    """
    from . import freeze as _freeze

    for block, _ in plan.consulted(d.dataset):
        rows = _freeze.slice_dir(plan.settings.cache_dir, block, d.dataset) / "rows.parquet"
        try:
            retire, revive, problem = _mark_flips(plan.settings, d.dataset, Path(d.root), rows)
        except Exception as exc:  # noqa: BLE001 -- a report must never stop the run
            retire, revive, problem = [], [], f"its folder marks could not be checked ({exc})"
        parts = []
        if retire:
            parts.append(f"its first rebuild would retire {len(retire)} folder(s) its rows show today "
                         f"({_examples(retire)}) under today's folder rules")
        if revive:
            parts.append(f"its first rebuild would bring back {len(revive)} retired folder(s) "
                         f"({_examples(revive)}) under today's folder rules")
        if problem:
            parts.append(f"its rows cannot be checked against today's folder rules: {problem}")
        if parts:
            plan.notices.append(
                f"[FREEZE] {d.dataset}: adopted with its rows exactly as they are; {'; '.join(parts)} -- "
                f"review dataset_overrides.{d.dataset}.freeze (folders / cutoff_exempt_before) before "
                f"new data arrives"
            )
            return


def _adopt(plan: Plan, d: Decision, inputs: dict) -> None:
    """Take a dataset frozen today into auto-freeze without rebuilding it."""
    from . import freeze as _freeze

    _adoption_mark_notice(plan, d)
    for block, _ in plan.consulted(d.dataset):
        if plan.read_only:
            before, after = _freeze.count_duplicate_rows(plan.settings.cache_dir, block, d.dataset)
        else:
            before, after = _freeze.dedupe_slice(plan.settings.cache_dir, block, d.dataset)
        if before != after:
            d.deduped[block] = (before, after)
    d.adopted = True
    if plan.read_only:
        return
    record = {
        "schema": RECORD_SCHEMA,
        "dataset": d.dataset,
        "root": d.root,
        "config": str(plan.config_path),
        "state": "frozen",
        "mode": "adopted",
        "inputs": inputs,
        "built": {"run_id": plan.run_id, "at": _now(), "reasons": ["adopted (frozen before auto-freeze)"]},
        # A baseline for the code/settings report, not a claim about how the
        # cached rows were built (unknown before adoption).
        "provenance": {"code": plan.code, "settings": plan.settings_prov, "baseline": "adoption"},
        "deduped": {b: list(v) for b, v in d.deduped.items()},
    }
    if d.deduped:
        record["figures_pending"] = True
    save_record(plan.scope, d.dataset, record)
    d.record = record


def _cascade_notices(plan: Plan, pairs: Sequence[tuple[str, str]], config_path: str | Path) -> None:
    prefix = _REBUILD_PREFIX.format(config=_display_config(config_path))
    frozen = {n for n, d in plan.decisions.items() if d.decision == "frozen"}
    new_data = {n for n, d in plan.decisions.items()
                if d.decision == "thawed" and any(r.startswith(("new data", "late")) or "YOLO has not" in r
                                                   for r in d.reasons)}
    for a, b in pairs:
        for got, other in ((a, b), (b, a)):
            if got in new_data and other in frozen:
                plan.notices.append(
                    f"[FREEZE] cascade (report only): {got} got new data and is rebuilt; its pair arm "
                    f"{other} stays frozen. To rebuild it too: {prefix} --thaw {other}"
                )
    groups: dict[tuple, list[str]] = {}
    for name in sorted(frozen):
        record = plan.decisions[name].record or {}
        prov = record.get("provenance") or {}
        old_code, old_settings = prov.get("code") or {}, prov.get("settings") or {}
        if not old_code and not old_settings:
            continue
        files = tuple(sorted(
            p for p in set(old_code) | set(plan.code) if old_code.get(p) != plan.code.get(p)
        ))
        keys = tuple(sorted(
            k for k in set(old_settings) | set(plan.settings_prov)
            if old_settings.get(k) != plan.settings_prov.get(k)
        ))
        if files or keys:
            groups.setdefault((files, keys), []).append(name)
    for (files, keys), names in groups.items():
        what = []
        if files:
            what.append("per-trial code changed: " + ", ".join(files[:6]) + (" ..." if len(files) > 6 else ""))
        if keys:
            what.append("per-trial settings changed: " + ", ".join(keys))
        plan.notices.append(
            f"[FREEZE] cascade (report only): {len(names)} frozen dataset(s) were built before this change "
            f"({'; '.join(what)}): {', '.join(names)}. To rebuild them: {prefix} "
            + " ".join(f"--thaw {n}" for n in names)
            + "  (--thaw re-runs the per-trial steps, combine and the wide build from the existing YOLO "
              "outputs; YOLO itself only re-runs for a trial whose folder is removed)"
        )


def _examples(items: Sequence[str], n: int = 3) -> str:
    return ", ".join(items[:n]) + (", ..." if len(items) > n else "")


def _trial_notices(plan: Plan, dataset: str, trials: Any) -> None:
    """Trials worth a line every run, whatever the dataset's decision.

    Neither is a reason to rebuild: a collision is not resolved by a rebuild
    (every attempt names one trial folder), and an untracked trial has nothing
    left here that YOLO could process. A frozen dataset is never walked, so
    without these lines nobody would notice either.
    """
    if trials.collisions:
        items = [f"{name} ({n} sessions)" for name, n in sorted(trials.collisions.items())]
        plan.notices.append(
            f"[FREEZE] {dataset}: {len(items)} trial(s) recorded more than once (a restarted session): "
            f"{_examples(items)}; YOLO processes only the latest session of a new one, and the secure "
            f"cleanup keeps every session's video -- move the aborted sessions' files aside to resolve"
        )
    if trials.untracked:
        plan.notices.append(
            f"[FREEZE] {dataset}: {len(trials.untracked)} trial(s) have rig files but were never tracked "
            f"and have no local video: {_examples(trials.untracked)} (a video copied back from the "
            f"secured mirror into its batch folder is new data, and is processed)"
        )


def _missing_root_notices(plan: Plan, settings: Any) -> None:
    for name in getattr(settings, "datasets", ()) or ():
        if name in plan.decisions:
            continue
        record = load_record(plan.scope, name)
        if isinstance(record, dict) and record.get("state") == "frozen":
            plan.notices.append(
                f"[FREEZE] {name}: has a freeze record but its folder is gone from disk; its rows are no "
                f"longer in the wide tables"
            )


def _display_config(config_path: str | Path) -> str:
    path = Path(config_path).expanduser()
    try:
        return path.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# the active plan (in this process and its figure subprocesses)
# --------------------------------------------------------------------------

_ACTIVE: Optional[dict] = None
_LOADED: dict[tuple[str, int], dict] = {}


def set_active_plan(plan: Optional[Plan]) -> None:
    """Make *plan* the one ``freeze_flags`` consults; None clears it.

    Also exported as ``FBPIPE_FREEZE_PLAN`` so figure subprocesses see it.
    """
    global _ACTIVE
    if plan is None:
        _ACTIVE = None
        os.environ.pop(PLAN_ENV, None)
        return
    _ACTIVE = plan.as_json()
    path = plan.write() if not plan.read_only else None
    if path is not None:
        os.environ[PLAN_ENV] = str(path)


def refresh_active_plan(plan: Plan) -> None:
    """Re-publish *plan* after a decision changed mid-run (a late thaw)."""
    if _ACTIVE is not None:
        set_active_plan(plan)


def active_plan() -> Optional[dict]:
    if _ACTIVE is not None:
        return _ACTIVE
    path = os.environ.get(PLAN_ENV)
    if not path:
        return None
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns)
        if key not in _LOADED:
            _LOADED.clear()
            _LOADED[key] = json.loads(Path(path).read_text(encoding="utf-8"))
        data = _LOADED[key]
    except Exception:  # noqa: BLE001 -- no readable plan: legacy flags
        return None
    return data if isinstance(data, dict) and data.get("schema") == PLAN_SCHEMA else None


def active_flags(dataset: str) -> Optional[tuple[bool, bool]]:
    """``(data_frozen, figures_frozen)`` for *dataset* from the active plan.

    None when no plan is active or the plan does not know the dataset.
    """
    data = active_plan()
    if not data:
        return None
    from .freeze import _freeze_key

    datasets = data.get("datasets") or {}
    entry = datasets.get(dataset)
    if entry is None:
        key = _freeze_key(dataset)
        for name, candidate in datasets.items():
            if _freeze_key(name) == key:
                entry = candidate
                break
    if not isinstance(entry, Mapping):
        return None
    return bool(entry.get("data_frozen")), bool(entry.get("figures_frozen"))

