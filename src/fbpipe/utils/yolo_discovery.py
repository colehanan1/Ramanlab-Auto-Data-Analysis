"""Which raw videos YOLO processes, and whether it has finished one.

``yolo_infer.main``, the auto-freeze plan (``fbpipe.autofreeze``: a video YOLO
would still process is "something new") and the secure cleanup (never delete a
local video YOLO has not fully processed) must agree on this exactly, so the
rules live here, free of torch/cv2 imports:

* a raw video is a ``.mp4``/``.avi`` file directly inside a live batch folder
  (``iter_live_batch_dirs``: retired folders are never inferred);
* its trial folder is ``<batch>/<name>`` with ``name`` = fields 1-6 of the stem
  (``output_october_06_batch_1_training_1_IsoamylAcetate-1_20261006_120331`` ->
  ``october_06_batch_1_training_1``), which drops the odor and the stamp;
* YOLO skips a video whose trial folder exists;
* a restarted session records a trial again, and every attempt names the same
  trial folder. YOLO processes the session the trial's metadata is read from
  (``trial_metadata.find_sidecar``: the most recent sidecar ``.txt``, else rig
  CSV), so its tables and its metadata are one recording -- never "the last
  name in sorted order": the odor label precedes the stamp, and a restart may
  carry another label (Hex-Control-24-0.01 april_17_batch_1 testing_3).
  While that session has no video, none is processed (an aborted attempt's
  tables would be paired with the real session's metadata); with no rig file
  at all, the latest ``_YYYYMMDD_HHMMSS`` stamp decides. Several files of one
  session (an ``.avi`` beside the ``.mp4``) keep yolo_infer's first pick. A
  folder YOLO built before this rule may hold an earlier attempt, so the
  cleanup keeps every session's video and auto-freeze reports the trial
  (``trial_sessions``, ``trial_report``);
* a trial is fully processed once its folder holds the merged table and a
  per-fly table for every fly slot the merged table has eye data for --
  yolo_infer writes the merged table first and skips eyeless slots.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

#: What yolo_infer.main infers (matched case-insensitively). A ``.h264`` needs a
#: remux into ``.mp4`` first, so it is never a YOLO input.
RAW_VIDEO_SUFFIXES = (".mp4", ".avi")

#: Every file a recording session leaves directly in its batch folder that names
#: it: the rig's ``output_*.csv`` and the videos, the ``.h264`` capture included.
_SESSION_VIDEO_SUFFIXES = (".mp4", ".avi", ".h264")

MERGED_SUFFIX = "_distances_merged"

#: The rig's session start stamp, the last field of every session file's stem.
_STAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


def trial_folder_name(video: str | Path) -> str:
    """The trial folder ``yolo_infer`` writes *video*'s tables into."""
    base = Path(video).stem
    parts = base.split("_")
    return "_".join(parts[1:7]) if len(parts) >= 7 else base


def raw_videos(batch_dir: str | Path) -> list[Path]:
    """The videos directly in *batch_dir* that yolo_infer reads, in its order."""
    return sorted(
        (f for f in Path(batch_dir).iterdir() if f.suffix.lower() in RAW_VIDEO_SUFFIXES),
        key=str,
    )


def _session_of(video: Path) -> str:
    """The recording session *video* belongs to: its stem, ``_preprocessed`` dropped."""
    return video.stem.replace("_preprocessed", "")


def _stamp_order(session: str) -> tuple[str, str]:
    m = _STAMP_RE.search(session)
    return (m.group(1) if m else "", session)


def _metadata_session(trial_dir: Path) -> Optional[str]:
    """The session the trial's metadata is read from; None without rig files."""
    from .trial_metadata import find_sidecar

    txt, csv = find_sidecar(trial_dir)
    source = txt if txt is not None else csv
    return None if source is None else source.stem.replace("_preprocessed", "")


def _processed_videos(videos: Iterable[str | Path]) -> dict[Path, Optional[Path]]:
    """``{video: the video YOLO processes for its trial}`` (None: it processes none).

    The rules are the module docstring's; a trial recorded once keeps
    yolo_infer's first pick, so only a restarted trial reads its rig files.
    """
    trials: dict[tuple[Path, str], list[Path]] = {}
    for video in sorted({Path(v) for v in videos}, key=str):
        trials.setdefault((video.parent, trial_folder_name(video)), []).append(video)
    listed: dict[Path, dict[str, dict[str, Any]]] = {}
    choice: dict[Path, Optional[Path]] = {}
    for (batch, name), members in trials.items():
        if batch not in listed:
            listed[batch] = _session_groups(batch)
        sessions = set(listed[batch].get(name, {}).get("stems", ())) | {_session_of(m) for m in members}
        kept: Optional[Path] = members[0]
        if len(sessions) > 1:
            real = _metadata_session(batch / name)
            if real is None:
                real = max((_session_of(m) for m in members), key=_stamp_order)
            kept = next((m for m in members if _session_of(m) == real), None)
        for member in members:
            choice[member] = kept
    return choice


def superseded_videos(videos: Iterable[str | Path]) -> set[Path]:
    """Every video of *videos* (one batch's) that YOLO skips for its trial.

    Another video of the trial is the one processed, or -- for a restarted
    trial whose metadata session has no video here -- none is.
    """
    return {video for video, kept in _processed_videos(videos).items() if kept != video}


def latest_session(videos: Iterable[str | Path], video: str | Path) -> Optional[Path]:
    """The video YOLO processes for *video*'s trial among *videos*: *video*
    itself unless superseded, None when it processes none of them."""
    video = Path(video)
    name = trial_folder_name(video)
    same = {Path(v) for v in videos if trial_folder_name(v) == name} | {video}
    return _processed_videos(same)[video]


def pending_videos(cfg: Any, root: str | Path) -> list[Path]:
    """The videos under dataset *root* that a YOLO run would process now.

    Exactly yolo_infer.main's discovery: live batch folders (folder rules
    applied), raw videos in sorted order, a restarted trial counted once as
    the video YOLO processes for it (none while its metadata session has no
    video), and a video whose trial folder exists left alone.
    """
    return trial_report(cfg, root).pending


def _session_stem(name: str) -> Optional[str]:
    """The session a file directly in a batch folder belongs to, or None.

    ``_preprocessed`` is dropped, as yolo_infer does when it looks up a video's
    rig CSV; dot files (an AppleDouble ``._output_*.mp4``) are no session.
    """
    if name.startswith("."):
        return None
    lower = name.lower()
    if lower.startswith("output_") and lower.endswith(".csv"):
        stem = name[: -len(".csv")]
    elif lower.endswith(_SESSION_VIDEO_SUFFIXES):
        stem = name.rsplit(".", 1)[0]
    else:
        return None
    return stem.replace("_preprocessed", "")


def _trial_of_stem(stem: str) -> str:
    # trial_folder_name without Path(): a stem may hold a dot ("Hexanol-0.1").
    parts = stem.split("_")
    return "_".join(parts[1:7]) if len(parts) >= 7 else stem


def _session_groups(batch_dir: str | Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    try:
        names = os.listdir(batch_dir)
    except OSError:
        return groups
    for name in names:
        stem = _session_stem(name)
        if stem is None:
            continue
        group = groups.setdefault(_trial_of_stem(stem), {"stems": set(), "csv": False, "video": False})
        group["stems"].add(stem)
        group["csv" if name.lower().endswith(".csv") else "video"] = True
    return groups


def trial_sessions(batch_dir: str | Path) -> dict[str, list[str]]:
    """``{trial folder name: [session stem, ...]}`` for one batch folder.

    More than one stem means the trial was recorded more than once (a restarted
    session): every attempt names the same trial folder, and nothing on disk
    says which attempt that folder was built from. Read from the rig CSVs as
    well as the videos, so it still holds once the videos are deleted.
    """
    return {name: sorted(g["stems"]) for name, g in _session_groups(batch_dir).items()}


@dataclasses.dataclass
class TrialReport:
    """One dataset root's trials, as one pass over its live batch folders sees them."""

    pending: list[Path]
    #: ``"<batch>/<trial>"`` -> how many sessions recorded it (only when > 1).
    collisions: dict[str, int]
    #: ``"<batch>/<trial>"`` with a rig CSV but no trial folder and no local
    #: video: never tracked, and nothing here YOLO could still process.
    untracked: list[str]
    #: ``"<batch>/<trial>"`` whose trial folder holds no merged table beside
    #: its video: yolo_infer creates the folder before it opens the video, so a
    #: video OpenCV cannot open is never tried again, nor pending (the folder is
    #: empty, or holds only what the per-trial steps write, _trial_meta.json).
    unopened: list[str] = dataclasses.field(default_factory=list)


def trial_report(cfg: Any, root: str | Path) -> TrialReport:
    """:func:`pending_videos`, plus the trials auto-freeze reports, in one pass."""
    from .frozen_folders import iter_live_batch_dirs

    report = TrialReport(pending=[], collisions={}, untracked=[])
    root_path = Path(root)
    if not root_path.is_dir():
        return report
    for batch in iter_live_batch_dirs(cfg, root_path):
        videos = raw_videos(batch)
        skipped = superseded_videos(videos)
        for video in videos:
            if video in skipped:
                continue
            trial = batch / trial_folder_name(video)
            if not trial.exists():
                report.pending.append(video)
            elif trial.is_dir() and _existing_table(trial / f"{trial.name}{MERGED_SUFFIX}") is None:
                report.unopened.append(f"{batch.name}/{trial.name}")
        for name, group in sorted(_session_groups(batch).items()):
            if len(group["stems"]) > 1:
                report.collisions[f"{batch.name}/{name}"] = len(group["stems"])
            if group["csv"] and not group["video"] and not (batch / name).exists():
                report.untracked.append(f"{batch.name}/{name}")
    return report


def _existing_table(stem: Path) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = stem.with_name(stem.name + suffix)
        if candidate.is_file():
            return candidate
    return None


def _eye_slots(merged: Path) -> list[int]:
    """Fly slots (0-based) whose ``eye_<i>_x`` column holds any value."""
    if merged.suffix == ".parquet":
        import pyarrow.parquet as pq

        names = [n for n in pq.read_schema(merged).names if n.startswith("eye_") and n.endswith("_x")]
        if not names:
            return []
        table = pq.read_table(merged, columns=names)
        counts = {n: table.num_rows - table.column(n).null_count for n in names}
    else:
        import pandas as pd

        header = pd.read_csv(merged, nrows=0).columns
        names = [n for n in header if str(n).startswith("eye_") and str(n).endswith("_x")]
        if not names:
            return []
        frame = pd.read_csv(merged, usecols=names)
        counts = {n: int(frame[n].notna().sum()) for n in names}
    slots = []
    for name, count in counts.items():
        try:
            index = int(name[len("eye_"):-len("_x")])
        except ValueError:
            continue
        if count > 0:
            slots.append(index)
    return sorted(slots)


def trial_tables_complete(out_dir: str | Path, name: str) -> tuple[bool, str]:
    """``(True, "")`` when YOLO fully processed the trial in *out_dir*.

    Otherwise ``(False, why)``. Anything unreadable counts as NOT processed:
    the failure mode of a doubtful check must be a video kept, never one lost.
    """
    trial = Path(out_dir)
    if not trial.is_dir():
        return False, "no trial folder"
    merged = _existing_table(trial / f"{name}{MERGED_SUFFIX}")
    if merged is None:
        return False, "no merged table in its trial folder"
    try:
        slots = _eye_slots(merged)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        return False, f"merged table unreadable ({type(exc).__name__})"
    missing = [
        f"fly{i + 1}"
        for i in slots
        if _existing_table(trial / f"{name}_fly{i + 1}_distances") is None
    ]
    if missing:
        return False, "no per-fly table for " + ", ".join(missing)
    return True, ""
