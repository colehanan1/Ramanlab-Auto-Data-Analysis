#!/usr/bin/env python3
"""Pre-label images for X-AnyLabeling with the geometry rules already applied.

X-AnyLabeling's ``yolov8_obb`` handler only does NMS and confidence
thresholding -- there is no hook in its model YAML for geometric
post-processing, so the rules in ``utils/fly_geometry`` cannot run inside the
app. This script applies them here instead and writes the app's own JSON label
format, so opening the folder shows shapes that already passed every gate.

Two things you get that the plain ONNX model cannot give you:

  * detections that fail a rule are dropped (or kept and marked, with
    ``--keep-rejected``) instead of silently becoming a bad label
  * the parts of one fly share a ``group_id``, so eye/proboscis/abdomen/tip are
    already linked per fly rather than being four loose boxes

Usage
-----
    python3 prelabel_with_rules.py --model best.pt --images DIR
    python3 prelabel_with_rules.py --model best.pt --video V.mp4 --every 40 \
        --out-dir frames/
"""
import argparse
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

REPO = '/home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis'
sys.path.insert(0, os.path.join(REPO, 'src'))
from fbpipe.utils import fly_geometry as FG                     # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
XANY_VERSION = '2.4.4'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', help='folder of images to label in place')
    src.add_argument('--video', help='video to sample frames from')
    p.add_argument('--out-dir', default=None, help='where frames+json go (video mode)')
    p.add_argument('--every', type=int, default=40, help='video: keep every Nth frame')
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--imgsz', type=int, default=1088)
    p.add_argument('--device', default=None)
    p.add_argument('--no-rules', action='store_true',
                   help='write raw detections, exactly what X-AnyLabeling would show')
    p.add_argument('--keep-rejected', action='store_true',
                   help='keep rule-failing shapes, marked in their description')
    p.add_argument('--overwrite', action='store_true',
                   help='replace existing .json files (default: skip them)')
    return p.parse_args()


def shape(label, pts, score, direction=0.0, group_id=None, description=None):
    """One X-AnyLabeling rotation shape."""
    return {
        'kie_linking': [],
        'direction': float(direction),
        'label': label,
        'score': None if score is None or not np.isfinite(score) else float(score),
        'points': [[float(x), float(y)] for x, y in pts],
        'group_id': group_id,
        'description': description,
        'difficult': False,
        'shape_type': 'rotation',
        'flags': {},
        'attributes': {},
    }


def detections(model, frame, a, names):
    """-> list of (label, corners(4,2), score, angle)."""
    res = model.predict(frame, conf=a.conf, iou=a.iou, imgsz=a.imgsz,
                        device=a.device, verbose=False)[0]
    o = res.obb
    if o is None or not len(o):
        return []
    pts = o.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    ang = o.xywhr.cpu().numpy()[:, 4]
    return [(names[int(c)], p, float(s), float(r)) for p, s, c, r in
            zip(pts, o.conf.cpu().numpy(), o.cls.cpu().numpy().astype(int), ang)]


def build_shapes(dets, W, H, use_rules, keep_rejected):
    """Apply the rules and emit shapes, grouped one group_id per fly."""
    if not use_rules:
        return [shape(lb, p, s, r) for lb, p, s, r in dets], Counter()

    parts = [FG.Part(lb, p, s) for lb, p, s, _ in dets]
    angle_of = {id(pt): r for pt, (_, _, _, r) in zip(parts, dets)}
    flies = FG.assemble(parts, W, H)

    shapes, stats = [], Counter()
    claimed = set()
    for gid, fly in enumerate(flies, start=1):
        note = '; '.join(fly.rejected) if fly.rejected else None
        for part, role in ((fly.eye, 'eye'), (fly.proboscis, 'proboscis'),
                           (fly.abdomen, 'abdomen'), (fly.tip, 'abdomenTip')):
            if part is None:
                continue
            claimed.add(id(part))
            shapes.append(shape(part.label, part.pts, part.score,
                                angle_of.get(id(part), 0.0), group_id=gid,
                                description=note if role == 'eye' else None))
            stats[f'kept:{part.label}'] += 1
        if not fly.trackable:
            stats['fly_without_trackable_abdomen'] += 1
    for part in parts:                       # detections no fly claimed
        if id(part) in claimed:
            continue
        stats[f'rejected:{part.label}'] += 1
        if keep_rejected:
            shapes.append(shape(part.label, part.pts, part.score,
                                angle_of.get(id(part), 0.0),
                                description='REJECTED: failed fly_geometry rules'))
    return shapes, stats


def write_json(img_path, shapes, W, H):
    out = os.path.splitext(img_path)[0] + '.json'
    with open(out, 'w') as fh:
        json.dump({'version': XANY_VERSION, 'flags': {}, 'shapes': shapes,
                   'imagePath': os.path.basename(img_path), 'imageData': None,
                   'imageHeight': H, 'imageWidth': W, 'description': ''},
                  fh, indent=2)
    return out


def main():
    a = parse_args()
    if not os.path.isfile(a.model):
        sys.exit(f'model not found: {a.model}')
    from ultralytics import YOLO
    model = YOLO(a.model)
    names = model.names
    print(f'model {a.model}  classes={list(names.values())}')
    print(f'rules {"OFF (raw detections)" if a.no_rules else "ON"}'
          f'{"  (rejected shapes kept and marked)" if a.keep_rejected else ""}')

    jobs = []                                # (image_path, frame or None)
    if a.images:
        if not os.path.isdir(a.images):
            sys.exit(f'not a folder: {a.images}')
        jobs = [(os.path.join(a.images, f), None)
                for f in sorted(os.listdir(a.images)) if f.lower().endswith(IMG_EXT)]
        print(f'{len(jobs)} images in {a.images}')
    else:
        out = a.out_dir or './prelabelled'
        os.makedirs(out, exist_ok=True)
        cap = cv2.VideoCapture(a.video)
        if not cap.isOpened():
            sys.exit(f'could not open {a.video}')
        stem = os.path.splitext(os.path.basename(a.video))[0]
        idx = kept = 0
        while True:
            if not cap.grab():
                break
            if idx % a.every == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                p = os.path.join(out, f'{stem}_f{idx:06d}.jpg')
                cv2.imwrite(p, fr)
                jobs.append((p, fr))
                kept += 1
                if a.max_frames and kept >= a.max_frames:
                    break
            idx += 1
        cap.release()
        print(f'extracted {len(jobs)} frames to {out}')

    total = Counter()
    written = skipped = 0
    for path, frame in jobs:
        if not a.overwrite and os.path.exists(os.path.splitext(path)[0] + '.json'):
            skipped += 1
            continue
        if frame is None:
            frame = cv2.imread(path)
            if frame is None:
                print(f'  ! unreadable: {path}')
                continue
        H, W = frame.shape[:2]
        dets = detections(model, frame, a, names)
        shapes, stats = build_shapes(dets, W, H, not a.no_rules, a.keep_rejected)
        write_json(path, shapes, W, H)
        total.update(stats)
        written += 1
        if written % 50 == 0:
            print(f'  {written} labelled', flush=True)

    print(f'\nwrote {written} .json files' + (f', skipped {skipped} existing' if skipped else ''))
    if total:
        print('per-class outcome:')
        for k in sorted(total):
            print(f'  {total[k]:6d}  {k}')
    print('\nOpen the folder in X-AnyLabeling; each fly\'s parts share a group_id.')


if __name__ == '__main__':
    main()
