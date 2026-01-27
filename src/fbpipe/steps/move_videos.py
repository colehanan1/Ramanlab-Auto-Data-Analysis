
from __future__ import annotations
from pathlib import Path
import shutil, re
from ..config import Settings, get_main_directories

DEST_FOLDER = "videos_with_rms"
VIDEO_EXTS = {".mp4",".mov",".avi",".mkv",".mpg",".mpeg",".m4v"}
TRIAL_DIR_RE = re.compile(r"^(?P<fly>.+)_(?P<phase>testing|training)_(?P<idx>\d+)$", re.IGNORECASE)

def is_video(p: Path): return p.is_file() and p.suffix.lower() in VIDEO_EXTS

def ensure_unique_path(dst: Path) -> Path:
    if not dst.exists(): return dst
    stem, suf = dst.stem, dst.suffix; n=1
    while True:
        cand = dst.with_name(f"{stem}_dup{n}{suf}")
        if not cand.exists(): return cand
        n+=1

def move_videos_from_trial(trial_dir: Path, fly_dir: Path, phase: str, dry_run=False):
    dest_dir = fly_dir / DEST_FOLDER / phase.lower()
    if not dry_run: dest_dir.mkdir(parents=True, exist_ok=True)
    moved=0
    for vid in sorted(trial_dir.iterdir()):
        if not is_video(vid): continue
        dst = ensure_unique_path(dest_dir / vid.name)
        print(f"{'DRY-RUN: ' if dry_run else ''}Moving {vid.name} -> {dst}")
        if not dry_run: shutil.move(str(vid), str(dst))
        moved += 1
    return moved

def main(cfg: Settings):
    roots = get_main_directories(cfg)
    for root in roots:
        for fly in [p for p in root.iterdir() if p.is_dir()]:
            name = fly.name
            if "_fly_" not in name:
                continue
            for sub in [p for p in fly.iterdir() if p.is_dir()]:
                if sub.name.lower().startswith(DEST_FOLDER.lower()):
                    continue
                m = TRIAL_DIR_RE.match(sub.name)
                if not m:
                    continue
                phase = m.group("phase").lower()
                if m.group("fly") != name:
                    continue
                moved = move_videos_from_trial(sub, fly, phase, dry_run=False)
                if moved==0:
                    print(f"[VIDS] No videos in {sub}")
