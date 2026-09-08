#!/usr/bin/env python3
"""Find trials where a fly's abdomen tip cannot be in frame.

Freezes eyes + posterior axis from the first N frames exactly as deployment
does, then predicts where each fly's tip must lie. A negative margin means the
tip is off the image, so that fly is a half-fly: its abdomen must not be tracked.
"""
import argparse, glob, math, os, sys
import cv2, numpy as np
from collections import Counter

REPO = '/home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis'
sys.path.insert(0, os.path.join(REPO, 'src'))
from fbpipe.utils.multi_fly import EyeAnchorManager, ZERO_IOU_EPS
from fbpipe.utils import fly_geometry as FG

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--glob', required=True, help='video glob')
ap.add_argument('--frames', type=int, default=5)
ap.add_argument('--limit', type=int, default=40)
ap.add_argument('--imgsz', type=int, default=1088)
ap.add_argument('--conf', type=float, default=0.25)
a = ap.parse_args()

from ultralytics import YOLO
model = YOLO(a.model); names = model.names
L = FG.DEFAULT_LIMITS


def parts_of(res):
    o = res.obb
    if o is None or not len(o):
        return []
    pts = o.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    return [(names[int(c)], p, float(s)) for p, s, c in
            zip(pts, o.conf.cpu().numpy(), o.cls.cpu().numpy().astype(int))]


vids = sorted(glob.glob(a.glob))[:a.limit]
print(f'{len(vids)} videos\n')
risky = []
for v in vids:
    cap = cv2.VideoCapture(v)
    if not cap.isOpened():
        continue
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scan = []
    for _ in range(a.frames):
        ok, fr = cap.read()
        if not ok:
            break
        scan.append(model.predict(fr, conf=a.conf, imgsz=a.imgsz, verbose=False)[0])
    cap.release()
    if not scan:
        continue
    counts = [sum(1 for lb, _, _ in parts_of(r) if lb == 'eye') for r in scan]
    n = Counter(counts).most_common(1)[0][0]
    mgr = EyeAnchorManager(max_eyes=n, zero_iou_eps=ZERO_IOU_EPS)
    pool, pairs = [], []
    for r in scan:
        ps = parts_of(r)
        eyes = [(p, s) for lb, p, s in ps if lb == 'eye']
        if eyes:
            xy = np.stack([np.array([p[:, 0].min(), p[:, 1].min(),
                                     p[:, 0].max(), p[:, 1].max()], np.float32) for p, _ in eyes])
            mgr.try_update_from_dets(xy, np.array([s for _, s in eyes], np.float32))
        pool += [p for p, _ in eyes]
        flies = FG.assemble([FG.Part(lb, p, s) for lb, p, s in ps], W, H, L)
        pairs += [(f.eye.centroid, f.abdomen.centroid) for f in flies if f.abdomen]
    axis = FG.estimate_posterior_axis(pairs, limits=L)
    if axis is None or not pool:
        continue
    out = []
    for c in mgr.get_centers():
        c = np.asarray(c, float)
        best = min(pool, key=lambda p: np.linalg.norm(FG.polygon_centroid(p) - c))
        s = float(np.sqrt(FG.polygon_area(best)))
        if s <= 0:
            continue
        p = c + axis * L.posterior_reach * s
        m = min(p[0], W - p[0], p[1], H - p[1]) / s
        out.append((c, s, m))
    bad = [o for o in out if o[2] < 0]
    tag = os.path.basename(os.path.dirname(v)) + '/' + os.path.basename(v)[:44]
    ang = math.degrees(math.atan2(axis[1], axis[0])) % 360
    if bad:
        risky.append((tag, bad, out, ang))
        print(f'OUT-OF-FRAME  {tag}')
        print(f'   axis {ang:5.1f} deg, {len(out)} flies; margins: '
              + ', '.join(f'{o[2]:+.1f}' for o in out))
        for c, s, m in bad:
            print(f'   eye@({c[0]:.0f},{c[1]:.0f}) scale {s:.0f}px  margin {m:+.2f} eye-widths')
print(f'\n{len(risky)}/{len(vids)} videos have >=1 fly whose tip cannot be in frame')
