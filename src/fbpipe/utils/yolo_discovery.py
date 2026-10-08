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
* several videos of one batch can name the same trial folder -- a restarted
  session records the trial again -- and the LATEST one (last in sorted order:
  the stamp is in the name) is the real session, as the trial metadata
  (``trial_metadata.find_sidecar``: most recent sidecar) already assumes;
* a trial is fully processed once its folder holds the merged table and a
  per-fly table for every fly slot the merged table has eye data for --
  yolo_infer writes the merged table first and skips eyeless slots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

#: What yolo_infer.main infers (matched case-insensitively). A ``.h264`` needs a
#: remux into ``.mp4`` first, so it is never a YOLO input.
RAW_VIDEO_SUFFIXES = (".mp4", ".avi")

MERGED_SUFFIX = "_distances_merged"


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


def superseded_videos(videos: Iterable[str | Path]) -> set[Path]:
    """Every video of *videos* that a LATER one naming the same trial supersedes.

    *videos* is one batch's list in yolo_infer order (sorted); the last member
    of each trial-folder group is the session that counts.
    """
    latest: dict[str, Path] = {}
    group: dict[str, list[Path]] = {}
    for video in sorted((Path(v) for v in videos), key=str):
        name = trial_folder_name(video)
        group.setdefault(name, []).append(video)
        latest[name] = video
    return {v for name, members in group.items() for v in members if v != latest[name]}


def latest_session(videos: Iterable[str | Path], video: str | Path) -> Path:
    """The video that supersedes *video* among *videos* (itself if none does)."""
    name = trial_folder_name(video)
    same = sorted((Path(v) for v in videos if trial_folder_name(v) == name), key=str)
    return same[-1] if same else Path(video)


def pending_videos(cfg: Any, root: str | Path) -> list[Path]:
    """The videos under dataset *root* that a YOLO run would process now.

    Exactly yolo_infer.main's discovery: live batch folders (folder rules
    applied), raw videos in sorted order, a collision group counted once as
    its latest session, and a video whose trial folder exists left alone.
    """
    from .frozen_folders import iter_live_batch_dirs

    root_path = Path(root)
    if not root_path.is_dir():
        return []
    pending: list[Path] = []
    for batch in iter_live_batch_dirs(cfg, root_path):
        videos = raw_videos(batch)
        skipped = superseded_videos(videos)
        for video in videos:
            if video in skipped:
                continue
            if not (batch / trial_folder_name(video)).exists():
                pending.append(video)
    return pending


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
