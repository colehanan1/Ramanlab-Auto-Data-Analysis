"""Rebuild an MP4 corrupted by two concurrent `ffmpeg -movflags +faststart` writers.

Damage signature: `ftyp | moov | moov(junk) | free | mdat`; samples before a mid-file junction
sit one moov-length past the offsets the real (first) moov records, samples after it sit at the
recorded offsets, and a few samples at the junction were overwritten.

Rebuild:
  1. resolve every sample (decode order) at shift 0 or +moov_size by an exact NAL-length walk;
  2. write a repaired container: ftyp + first moov (chunk offsets re-pointed) + mdat holding the
     resolved samples contiguously, overwritten samples zero-filled (the decoder drops them);
  3. re-encode at the exact container frame rate with constant-frame-rate output, so each
     dropped sample's display slot is filled with the previous frame and frame i still lines
     up with row i of the rig's frame CSV.

Writes <out_dir>/<name>.mp4 and <out_dir>/<name>.rebuild.json (what was lost, where).
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from mp4box import CONTAINERS, iter_boxes, nal_valid, parse_moov, top_boxes

FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"


def find_box(buf: bytes, start: int, end: int, want: bytes):
    """Absolute (payload_start, box_end) of the first `want` box below buf[start:end]."""
    for typ, _s, p, e in iter_boxes(buf, start, end):
        if e is None:
            return None
        if typ == want:
            return p, e
        if typ in CONTAINERS:
            hit = find_box(buf, p, e, want)
            if hit:
                return hit
    return None


def resolve(buf, track, moov_size):
    """Place every sample; return (file offsets or None, lost indices, ambiguous count).

    The overwritten window is one moov long and ends where the unshifted run begins, so the
    last shifted sample before that run keeps a valid head but loses its tail: it is lost too.
    """
    offs = track.sample_offsets()
    shifts, ambiguous = [], 0
    prev = moov_size
    for o, sz in zip(offs, track.sizes):
        ok = [sh for sh in (moov_size, 0) if nal_valid(buf, o + sh, sz, track.length_size)]
        if not ok:
            shifts.append(None)
            continue
        if len(ok) == 2:
            ambiguous += 1
            sh = prev  # keep the surrounding run's placement
        else:
            sh = ok[0]
        shifts.append(sh)
        prev = sh
    first_unshifted = next((i for i, sh in enumerate(shifts) if sh == 0), None)
    if first_unshifted is not None:
        last_shifted = max((i for i in range(first_unshifted) if shifts[i] == moov_size), default=None)
        if last_shifted is not None:
            shifts[last_shifted] = None
    placed = [None if sh is None else o + sh for o, sh in zip(offs, shifts)]
    lost = [i for i, sh in enumerate(shifts) if sh is None]
    return placed, lost, ambiguous


def display_order(track):
    """Decode index -> display index (from stts/ctts)."""
    dts, t = [], 0
    for count, delta in track.stts:
        for _ in range(count):
            dts.append(t)
            t += delta
    cto = []
    for count, off in track.ctts or [(len(dts), 0)]:
        cto += [off] * count
    pts = [d + c for d, c in zip(dts, cto)]
    order = sorted(range(len(pts)), key=lambda i: pts[i])
    disp = [0] * len(pts)
    for rank, i in enumerate(order):
        disp[i] = rank
    return disp


def repaired_container(buf, moov_s, moov_e, track, placed) -> bytes:
    ftyp_s, ftyp_e = next((s, e) for t, s, e in top_boxes(buf) if t == "ftyp")
    ftyp = buf[ftyp_s:ftyp_e]
    moov = bytearray(buf[moov_s:moov_e])
    payload = bytearray()
    starts = []
    for p, sz in zip(placed, track.sizes):
        starts.append(len(payload))
        payload += buf[p:p + sz] if p is not None else bytes(sz)
    mdat_payload_at = len(ftyp) + len(moov) + 8
    # first sample index of every chunk
    first_sample, s = [], 0
    runs = track.stsc + [(len(track.chunk_offsets) + 1, 0, 0)]
    for k in range(len(runs) - 1):
        f, per, _ = runs[k]
        for _chunk in range(f, runs[k + 1][0]):
            first_sample.append(s)
            s += per
    new_offsets = [mdat_payload_at + starts[fs] for fs in first_sample[:len(track.chunk_offsets)]]
    hit = find_box(buf, moov_s + 8, moov_e, b"stco")
    if hit:
        p, _e = hit
        rel = p - moov_s + 8  # skip v/f(4) + count(4)
        struct.pack_into(f">{len(new_offsets)}I", moov, rel, *new_offsets)
    else:
        p, _e = find_box(buf, moov_s + 8, moov_e, b"co64")
        rel = p - moov_s + 8
        struct.pack_into(f">{len(new_offsets)}Q", moov, rel, *new_offsets)
    if len(payload) + 8 >= 2**32:
        raise SystemExit("mdat too large for a 32-bit box header")
    return ftyp + bytes(moov) + struct.pack(">I4s", len(payload) + 8, b"mdat") + bytes(payload)


def count_frames(path: Path) -> int:
    out = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.split(",")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--crf", default="18")
    ap.add_argument("--keep-repaired", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for src in map(Path, a.inputs):
        buf = src.read_bytes()
        moovs = [(s, e) for t, s, e in top_boxes(buf) if t == "moov"]
        if len(moovs) != 2:
            print(f"SKIP {src.name}: {len(moovs)} moov box(es), not the double-moov signature")
            continue
        (m_s, m_e), (m2_s, m2_e) = moovs
        track = parse_moov(buf, m_s, m_e)
        moov_size = m_e - m_s
        if (m2_e - m2_s) != moov_size or not track.sizes:
            print(f"SKIP {src.name}: unexpected moov layout")
            continue
        deltas = {d for _c, d in track.stts}
        if len(deltas) != 1:
            print(f"SKIP {src.name}: variable frame durations {sorted(deltas)[:5]}")
            continue
        rate = Fraction(track.timescale, deltas.pop())
        placed, lost, ambiguous = resolve(buf, track, moov_size)
        disp = display_order(track)
        sync = set(track.stss)
        n = len(track.sizes)
        # frames decoded from a reference chain broken by a lost sample: up to the next sync sample
        conceal = set()
        for i in lost:
            j = i + 1
            while j < n and (j + 1) not in sync:
                conceal.add(j)
                j += 1
        conceal -= set(lost)
        repaired = out_dir / (src.stem + ".repaired.mp4")
        repaired.write_bytes(repaired_container(buf, m_s, m_e, track, placed))
        final = out_dir / src.name
        tmp = out_dir / ("." + src.name + ".part")
        # Place every frame the decoder outputs by its timestamp; fill empty display slots
        # (lost samples, and frames the decoder drops for a missing reference) with the
        # previous frame, so frame i stays aligned with rig-CSV row i.
        delta = track.stts[0][1]
        probe = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "frame=pts", "-of", "csv=p=0", str(repaired)],
                               capture_output=True, text=True, check=True).stdout.split()
        pts = [int(x.strip(",")) for x in probe if x.strip(",").lstrip("-").isdigit()]
        base = min(pts)
        if any((p - base) % delta for p in pts):
            print(f"FAIL {src.name}: decoded timestamps are not on the {delta}-tick grid")
            failures += 1
            continue
        slots = [(p - base) // delta for p in pts]
        fsize = track.width * track.height * 3 // 2
        dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(repaired), "-map", "0:v:0",
                                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        by_slot, extra = {}, 0
        for k in slots:
            frame = dec.stdout.read(fsize)
            if len(frame) < fsize:
                break
            if k in by_slot or not 0 <= k < n:
                extra += 1
                continue
            by_slot[k] = frame
        tail = dec.stdout.read()
        dec_err = dec.stderr.read().decode(errors="ignore")
        dec.wait()
        if tail or len(by_slot) + extra != len(slots):
            print(f"FAIL {src.name}: ffprobe listed {len(slots)} frames but the decode gave a different count")
            failures += 1
            continue
        filled = [k for k in range(n) if k not in by_slot]
        enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                                "-s", f"{track.width}x{track.height}",
                                "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
                                "-c:v", "libx264", "-preset", "slow", "-crf", a.crf, "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", "-f", "mp4", str(tmp)],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        prev_frame = by_slot.get(min(by_slot))
        for k in range(n):
            frame = by_slot.pop(k, None)
            if frame is None:
                frame = prev_frame
            enc.stdin.write(frame)
            prev_frame = frame
        enc.stdin.close()
        enc_err = enc.stderr.read().decode(errors="ignore")
        enc.wait()
        decode_errors = len([ln for ln in dec_err.splitlines() if ln.strip()])
        if enc.returncode != 0 or not tmp.exists():
            print(f"FAIL {src.name}: encoder rc={enc.returncode}\n{enc_err[-800:]}")
            failures += 1
            continue
        tmp.replace(final)
        frames = count_frames(final)
        report = {
            "source": str(src), "rebuilt": str(final), "samples": n, "frames_out": frames,
            "fps": float(rate), "fps_exact": f"{rate.numerator}/{rate.denominator}",
            "moov_size": moov_size, "ambiguous_samples_resolved_by_run": ambiguous,
            "lost_samples_decode_order": lost,
            "lost_display_frames": sorted(disp[i] for i in lost),
            "lost_time_s": [round(min(disp[i] for i in lost) / float(rate), 3),
                            round((max(disp[i] for i in lost) + 1) / float(rate), 3)] if lost else None,
            "concealed_frames_until_next_keyframe": len(conceal),
            "concealed_display_range": [min(disp[i] for i in conceal), max(disp[i] for i in conceal)] if conceal else None,
            "decoder_messages": decode_errors,
            "filled_display_frames": filled,
            "filled_time_s": [round(filled[0] / float(rate), 3), round((filled[-1] + 1) / float(rate), 3)] if filled else None,
            "duplicate_or_out_of_range_frames_dropped": extra,
            "method": "resolve shift {0,+moov} by NAL walk; zero-fill lost; cfr re-encode libx264 crf "
                      + a.crf + " preset slow",
        }
        (out_dir / (src.stem + ".rebuild.json")).write_text(json.dumps(report, indent=1))
        if not a.keep_repaired:
            repaired.unlink()
        status = "OK " if frames == n else "COUNT-MISMATCH"
        failures += frames != n
        print(f"{status} {src.name[:60]}: {n} samples -> {frames} frames; lost {len(lost)} "
              f"(display {report['lost_display_frames'][:1]}..{report['lost_display_frames'][-1:]}, "
              f"t={report['lost_time_s']} s); filled slots {len(filled)} {filled[:1]}..{filled[-1:]}; ambiguous {ambiguous}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
