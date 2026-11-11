#!/usr/bin/env python3
import csv, os, subprocess, argparse, fcntl, hashlib, math
from pathlib import Path

# ── OFM → Odor name mapping ──────────────────────────────────────────────────
# Fallback only: what sits in each valve changes whenever vials are swapped, so the
# label normally comes from the rig's per-trial .txt (see choose_trial_label).
OFM_NAME_MAP = {
    "OFM_A": "ACV",
    "OFM_B": "Benzaldehyde",
    "OFM_C": "Citral",
    "OFM_E": "Ethyl Butyrate",
    "OFM_H": "Hexanol",
    "OFM_L": "Linalool",
    "OFM_O": "3-Octonol",
}

def humanize_ofm_label(label: str) -> str:
    if not label: return ""
    s = label.strip().upper()
    if s == "OFF": return "off"
    return OFM_NAME_MAP.get(s, label.strip())

def read_trial_odor(txt_path: Path):
    """The "Trial odor:" line of the rig's per-trial .txt (e.g. "Hexanol-0.1"), or None."""
    try: text = txt_path.read_text(encoding="utf-8", errors="ignore")
    except OSError: return None
    for line in text.splitlines():
        if line.startswith("Trial odor:"):
            odor = line.split(":", 1)[1].strip()
            return odor if odor and odor.upper() != "N/A" else None
    return None

def choose_trial_label(frames_csv_path: Path, segments):
    """(odor name to burn in or None for the valve names, note for the log).

    The CSV only names the valve (ActiveOFM = OFM_A); the .txt the rig writes next to
    it says what was sent ("Trial odor: IsoamylAcetate-1").
    """
    txt = frames_csv_path.with_suffix(".txt")
    odor = read_trial_odor(txt)
    if odor is None:
        return None, f"valve names (no Trial odor in {txt.name})"
    valves = {pin for _, _, pin in segments}
    if len(valves) > 1:  # one name for the trial can't say which valve held it
        return None, f"valve names ({len(valves)} valves opened, {txt.name} names one odor)"
    return odor, f"{odor} (from {txt.name})"

def parse_ofm_segments_from_frame_csv(frames_csv_path: Path):
    segments, first_ts_ns, last_t_rel = [], None, 0.0
    on_time, current_pin = None, None
    with frames_csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = (row.get("SensorNs") or "").strip()
            if not s: continue
            try: sensor_ns = int(s)
            except ValueError: continue
            if first_ts_ns is None: first_ts_ns = sensor_ns
            t_rel = (sensor_ns - first_ts_ns) / 1e9
            last_t_rel = t_rel
            label = (row.get("ActiveOFM", "off") or "off").strip()
            if label != "off" and on_time is None:
                on_time, current_pin = t_rel, label
            elif on_time is not None and label != current_pin:
                segments.append((on_time, t_rel, current_pin))
                on_time, current_pin = (t_rel, label) if label != "off" else (None, None)
    if on_time is not None:
        segments.append((on_time, last_t_rel, current_pin))
    return segments

def compute_fps_from_frame_csv(frames_csv_path: Path) -> float:
    sensor_ns = []
    with frames_csv_path.open(newline='') as f:
        for row in csv.DictReader(f):
            s = (row.get("SensorNs") or "").strip()
            if s:
                try: sensor_ns.append(int(s))
                except ValueError: pass
    if len(sensor_ns) >= 2:
        duration_s = (sensor_ns[-1] - sensor_ns[0]) / 1e9
        n_frames = len(sensor_ns)
        if duration_s > 0 and n_frames > 1:
            fps = (n_frames - 1) / duration_s
            if fps > 0: return fps

    intervals = []
    with frames_csv_path.open(newline='') as f:
        for row in csv.DictReader(f):
            v = (row.get("FrameInterval_s") or "").strip()
            if v:
                try:
                    x = float(v)
                    if x > 0: intervals.append(x)
                except ValueError:
                    pass
    if intervals:
        intervals.sort()
        core = intervals
        if len(intervals) >= 40:
            k = max(1, int(0.025 * len(intervals)))
            if k < len(intervals)//2: core = intervals[k:-k]
        mean_dt = sum(core) / len(core)
        if mean_dt > 0: return 1.0 / mean_dt
    raise ValueError("Unable to infer FPS from CSV timing.")

FONT_SIZE = 20

def text_width_px(text, font_ttf, font_size):
    try:
        from PIL import ImageFont
        return ImageFont.truetype(font_ttf, font_size).getlength(text)
    except Exception:  # no Pillow: guess wide (odor labels run ~0.62 em per character)
        return 0.8 * font_size * len(text)

def build_vf_filter(segments, scale_factor, out_fps, font_ttf, trial_odor=None):
    vf = []
    if scale_factor and scale_factor != 1.0:
        vf.append(f"scale=iw*{scale_factor}:ih*{scale_factor}:flags=bicubic")
    vf += ["unsharp=5:5:2:5:5:2", "eq=1.5:brightness=0.25"]
    for start, end, pin in segments:
        odor = trial_odor or humanize_ofm_label(pin)
        safe = (odor or "").replace(":", "\\:").replace("'", "\\'")
        # Text starts 5 px into the box; the box runs 10 px past its end. Never below
        # 100 px wide: odor_box_alignment.py samples x 20..100 inside it.
        box_w = max(100, math.ceil(text_width_px(odor or "", font_ttf, FONT_SIZE)) + 15)
        vf.append(
            f"drawbox=x=10:y=10:width={box_w}:height=50:color=white@1:t=fill:"
            f"enable='between(t\\,{start:.6f}\\,{end:.6f})'"
        )
        vf.append(
            f"drawtext=fontfile={font_ttf}:text='{safe}':x=15:y=35:fontsize={FONT_SIZE}:"
            f"fontcolor=black:box=0:enable='between(t\\,{start:.6f}\\,{end:.6f})'"
        )
    if out_fps is not None:
        vf.append(f"fps={out_fps:.6f}")
    return ",".join(vf)

def remux_video_using_csv_timing(h264_path: Path, frames_csv_path: Path, mp4_out_path: Path,
                                 crf, preset, profile, level, pix_fmt, faststart,
                                 scale_factor, fps_cap, pass_fds=()):
    in_fps = compute_fps_from_frame_csv(frames_csv_path)
    segments = parse_ofm_segments_from_frame_csv(frames_csv_path)
    trial_odor, label_note = choose_trial_label(frames_csv_path, segments)
    if segments:
        print(f"   odor label: {label_note}", flush=True)
    out_fps = min(in_fps, float(fps_cap)) if fps_cap else None
    vf_str = build_vf_filter(segments, scale_factor, out_fps,
                             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", trial_odor)
    gop = int(round((out_fps or in_fps) * 2))
    common = [
        "-vf", vf_str,
        "-c:v", "libx264",
        "-preset", preset,
        "-profile:v", profile,
        "-level:v", level,
        "-pix_fmt", pix_fmt,
        "-g", str(max(12, gop)),
        "-bf", "3",
        "-b_strategy", "1",
    ]
    if faststart:
        common += ["-movflags", "+faststart"]
    cmd = [
        "ffmpeg", "-y",
        "-f", "h264", "-r", f"{in_fps:.6f}", "-i", str(h264_path),
        *common, "-crf", str(crf),
        # The output is a hidden ".<stem>.mp4.part" temp, so ffmpeg can't guess the
        # container from the extension.
        "-f", "mp4",
        str(mp4_out_path),
    ]
    try:
        # pass_fds hands ffmpeg the folder-lock fd (see acquire_folder_lock): if this
        # runner is killed mid-encode the orphaned ffmpeg keeps the folder locked until
        # it exits, so a relaunch cannot start a second ffmpeg on the same temp file.
        r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           pass_fds=pass_fds)
        return True, in_fps, r.stderr.decode(errors="ignore")
    except subprocess.CalledProcessError as e:
        return False, in_fps, e.stderr.decode(errors="ignore")

# ── One run per folder ───────────────────────────────────────────────────────
# The rigs launch this script over ssh after every push, so a second run can start on
# a folder while the first is still encoding. Two encodes of the same .h264 write the
# same .mp4 at once and the +faststart rewrites interleave (two moov boxes, 0 frames
# decodable). The lock lives outside the data folder, and the kernel drops it when
# the process dies, however it dies.
def folder_lock_path(input_path: Path) -> Path:
    lock_dir = os.environ.get("REMUX_LOCK_DIR")
    lock_dir = Path(lock_dir) if lock_dir else Path.home() / ".cache" / "remux_from_csv"
    return lock_dir / (hashlib.sha1(os.fsencode(str(input_path))).hexdigest() + ".lock")

def acquire_folder_lock(input_path: Path) -> int:
    """Block until no other run owns input_path; print one line saying if we had to wait.

    Returns the lock fd. It stays open for the life of the process (and is passed on to
    each ffmpeg, see remux_video_using_csv_timing); closing it would release the lock.
    """
    try:
        lock_path = folder_lock_path(input_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except (OSError, RuntimeError) as ex:
        # Never fall back to running unlocked: that is the bug.
        raise SystemExit(f"Cannot take the run lock for {input_path}: {ex}")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"[LOCK] no other run is active on {input_path}; lock acquired ({lock_path})", flush=True)
    except BlockingIOError:
        print(f"[LOCK] another run is active on {input_path}; waiting for it to finish ({lock_path})", flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
    return fd

def main():
    ap = argparse.ArgumentParser(description="Batch H.264 → MP4 using CSV timing")
    ap.add_argument("--input-dir", required=True, help="Directory to scan for .h264/.csv pairs")
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--profile", default="high")
    ap.add_argument("--level", default="4.1")
    ap.add_argument("--pix-fmt", default="yuv420p")
    ap.add_argument("--faststart", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fps-cap", type=float, default=50.0)
    ap.add_argument("--delete-min-bytes", type=int, default=1024)
    ap.add_argument("--log-tail", type=int, default=20)
    args = ap.parse_args()

    input_path = Path(args.input_dir).resolve()
    if not input_path.is_dir():
        raise SystemExit(f"Not a directory: {input_path}")

    # Lock first, scan second: a run that had to wait must list what the previous run
    # left behind, not what was there when it started.
    lock_fd = acquire_folder_lock(input_path)

    h264_files = sorted(input_path.rglob("*.h264"))
    if not h264_files:
        print(f"[INFO] No .h264 files in {input_path}")
        return

    for h264 in h264_files:
        if not h264.exists():
            print(f"[SKIP] {h264.name} → source no longer exists")
            continue

        frames_csv = h264.with_suffix(".csv")
        if not frames_csv.exists():
            frames_csv = h264.with_name(h264.stem + "_frames.csv")
        mp4_file   = h264.with_suffix(".mp4")
        tmp_file   = mp4_file.with_name(f".{mp4_file.name}.part")
        if not frames_csv.exists():
            print(f"[SKIP] {h264.name} → no {frames_csv.name}")
            continue

        print(f"🔄 Transcoding {h264.name}")
        # ffmpeg writes the hidden temp; only a finished, big-enough encode is renamed
        # onto the final .mp4, so that name never holds a partial file.
        published = False
        try:
            ok, fps_used, log = remux_video_using_csv_timing(
                h264, frames_csv, tmp_file,
                args.crf, args.preset, args.profile, args.level, args.pix_fmt,
                args.faststart, args.scale, args.fps_cap, pass_fds=(lock_fd,)
            )
            if ok and tmp_file.exists() and tmp_file.stat().st_size >= args.delete_min_bytes:
                os.replace(tmp_file, mp4_file)
                published = True
        finally:
            tmp_file.unlink(missing_ok=True)  # no-op once published; else failed/too small/interrupted
        if published:
            try:
                h264.unlink()
                print(f"✅ {mp4_file.name} (input FPS {fps_used:.6f}) — deleted source {h264.name}")
            except Exception as ex:
                print(f"⚠️  {mp4_file.name} created (input FPS {fps_used:.6f}) but could not delete {h264.name}: {ex}")
        else:
            print(f"❌ Failed: {h264.name} (input FPS {fps_used:.6f})")
            if log:
                print("\n".join(log.splitlines()[-args.log_tail:]))

if __name__ == "__main__":
    main()

