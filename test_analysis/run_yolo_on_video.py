#!/usr/bin/env python3
"""Run a YOLO-OBB model over a trial video and show how it performs.

Mirrors how the model is actually deployed in fbpipe rather than running a
generic prediction loop:

  * eyes are frozen from the first ``--freeze-eyes`` frames (5, as in
    ``yolo_infer._scan_initial_fly_count``) using the real
    ``EyeAnchorManager`` -- highest-scoring, pairwise zero-IoU boxes, then
    locked and numbered top-to-bottom. Every gate downstream measures from that
    frozen eye, so the QC has to as well.
  * the posterior (head->abdomen) axis is frozen from those same frames, since
    it is a property of the rig, not of the fly.
  * the proboscis gate is the production anisotropic blob, drawn with
    ``distance_sanity.anisotropic_boundary_offsets`` and sized from
    config_new.yaml, so what you see is the gate the pipeline applies.
  * abdomen / abdomenTip are gated by ``utils/fly_geometry``.

With ``--draw-boundaries`` every one of those regions is drawn on the video, so
a detection landing outside its allowed area is visible rather than inferred.

There is no ground truth for a raw trial, so performance is reported with the
checks that need no labels: per-class detection rate and confidence, fly-count
stability, completeness, positional jitter (tethered flies should barely move),
and what fraction of detections fall inside their rule boundary.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

REPO = '/home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis'
CONFIG = os.path.join(REPO, 'config', 'config_new.yaml')
sys.path.insert(0, os.path.join(REPO, 'src'))

from fbpipe.utils.multi_fly import EyeAnchorManager, ZERO_IOU_EPS          # noqa: E402
from fbpipe.utils.distance_sanity import anisotropic_boundary_offsets      # noqa: E402
from fbpipe.utils import fly_geometry as FG                                # noqa: E402

CLS_COLOR = {                       # BGR
    'eye': (255, 208, 0),
    'proboscis': (0, 212, 255),
    'abdomen': (210, 77, 255),
    'abdomenTip': (0, 252, 124),
}
GATE_COLOR = {
    'proboscis': (0, 170, 255),
    'abdomen': (190, 60, 230),
    'abdomenTip': (0, 210, 110),
}
OK, BAD, FROZEN = (120, 255, 120), (60, 60, 255), (0, 255, 255)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--imgsz', type=int, default=1088)
    p.add_argument('--device', default=None)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--freeze-eyes', type=int, default=5,
                   help='frames used to lock the eye anchors (deployment uses 5)')
    p.add_argument('--max-flies', type=int, default=None,
                   help='fly slots; default = modal eye count in the scan frames')
    p.add_argument('--draw-boundaries', action='store_true',
                   help='draw the rule regions each part must fall inside')
    p.add_argument('--no-annotated', action='store_true')
    p.add_argument('--config', default=CONFIG, help='config to read gate values from')
    return p.parse_args()


def load_gates(path):
    """Read the gates the production code reads -- both keys, explicitly.

    ``proboscis_filter.max_eye_prob_distance_px`` is the spatial gate applied at
    every fly count; ``distance_limits.three_fly_max_eye_prob_distance_px`` is
    the separate >=3-fly pairing cap. They happen to be equal in config_new.yaml,
    so reading one in place of the other is correct only by coincidence.
    """
    gates = dict(max_eye_prob_distance_px=160.0, up_divisor=4.0,
                 three_fly_max_eye_prob_distance_px=160.0, source='defaults')
    try:
        import yaml
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        pf = raw.get('proboscis_filter') or {}
        dl = raw.get('distance_limits') or {}
        if 'max_eye_prob_distance_px' in pf:
            gates['max_eye_prob_distance_px'] = float(pf['max_eye_prob_distance_px'])
        if 'up_divisor' in pf:
            gates['up_divisor'] = float(pf['up_divisor'])
        if 'three_fly_max_eye_prob_distance_px' in dl:
            gates['three_fly_max_eye_prob_distance_px'] = float(
                dl['three_fly_max_eye_prob_distance_px'])
        gates['source'] = path
    except Exception as e:                                   # noqa: BLE001
        print(f'  (could not read {path}: {e}; using dataclass defaults)')
    return gates


def obb_parts(res, names):
    """-> list of (label, corners(4,2), score) for one result."""
    o = getattr(res, 'obb', None)
    if o is not None and len(o):
        pts = o.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        cf, cl = o.conf.cpu().numpy(), o.cls.cpu().numpy().astype(int)
    else:
        b = getattr(res, 'boxes', None)
        if b is None or not len(b):
            return []
        xy = b.xyxy.cpu().numpy()
        pts = np.stack([xy[:, [0, 1]], xy[:, [2, 1]], xy[:, [2, 3]], xy[:, [0, 3]]], 1)
        cf, cl = b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
    return [(names[int(c)], p, float(s)) for p, s, c in zip(pts, cf, cl)]


def to_xyxy(pts):
    return np.array([pts[:, 0].min(), pts[:, 1].min(),
                     pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def sector(center, r0, r1, ang_c, half, n=28):
    """Closed annular-sector polygon: radii r0..r1 over ang_c +/- half (radians)."""
    a = np.linspace(ang_c - half, ang_c + half, n)
    outer = np.stack([center[0] + r1 * np.cos(a), center[1] + r1 * np.sin(a)], 1)
    inner = np.stack([center[0] + r0 * np.cos(a[::-1]), center[1] + r0 * np.sin(a[::-1])], 1)
    return np.concatenate([outer, inner], 0).astype(np.int32)


def draw_boundaries(frame, anchors, axis, gates, limits):
    """Draw the region each part must land inside, around every frozen eye."""
    ang = math.atan2(axis[1], axis[0]) if axis is not None else 0.0
    half = math.radians(limits.dir_tol_deg)
    pro_off = np.array(anisotropic_boundary_offsets(
        gates['max_eye_prob_distance_px'], gates['up_divisor']), dtype=np.float32)
    ov = frame.copy()
    for c, s in anchors:
        c = np.asarray(c, dtype=np.float32)
        # production proboscis gate (absolute px, anisotropic)
        cv2.polylines(ov, [(pro_off + c).astype(np.int32)], True, GATE_COLOR['proboscis'], 2)
        if axis is None:
            continue
        # eye-relative abdomen / tip sectors
        cv2.polylines(ov, [sector(c, limits.d_eye_abd_min * s, limits.d_eye_abd_max * s,
                                  ang, half)], True, GATE_COLOR['abdomen'], 2)
        cv2.polylines(ov, [sector(c, limits.d_eye_tip_min * s, limits.d_eye_tip_max * s,
                                  ang, half)], True, GATE_COLOR['abdomenTip'], 2)
        # posterior axis and the predicted tip used by the visibility rule
        tipp = c + np.asarray(axis) * limits.posterior_reach * s
        cv2.arrowedLine(ov, tuple(c.astype(int)), tuple(tipp.astype(int)),
                        (255, 255, 255), 1, tipLength=0.04)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    return frame


def draw_frame(frame, parts, flies, anchors, axis, gates, limits, draw_bounds):
    if draw_bounds and anchors:
        draw_boundaries(frame, anchors, axis, gates, limits)
    for label, pts, score in parts:
        col = CLS_COLOR.get(label, (200, 200, 200))
        cv2.polylines(frame, [pts.astype(np.int32)], True, col, 2)
        cv2.putText(frame, f'{label} {score:.2f}',
                    (int(pts[:, 0].min()), int(pts[:, 1].min()) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for i, (c, s) in enumerate(anchors):                  # frozen eyes, as deployed
        c = np.asarray(c)
        cv2.circle(frame, tuple(c.astype(int)), int(0.7 * s), FROZEN, 2)
        cv2.putText(frame, f'eye{i} (frozen)', (int(c[0]) - 30, int(c[1]) - int(0.8 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, FROZEN, 1, cv2.LINE_AA)
    for f in flies:
        e = tuple(np.int32(f.eye.centroid))
        if f.abdomen is not None:
            cv2.line(frame, e, tuple(np.int32(f.abdomen.centroid)), OK, 2)
            if f.tip is not None:
                cv2.line(frame, tuple(np.int32(f.abdomen.centroid)),
                         tuple(np.int32(f.tip.centroid)), OK, 1)
        else:
            cv2.circle(frame, e, 24, BAD, 2)
            if f.rejected:
                cv2.putText(frame, f.rejected[0][:44], (e[0] - 40, e[1] + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, BAD, 1, cv2.LINE_AA)
    return frame


def main():
    a = parse_args()
    for p, what in ((a.model, 'model'), (a.video, 'video')):
        if not os.path.isfile(p):
            sys.exit(f'{what} not found: {p}')
    out = a.out_dir or f'./yolo_qc_{os.path.splitext(os.path.basename(a.video))[0]}'
    os.makedirs(out, exist_ok=True)
    limits = FG.DEFAULT_LIMITS
    gates = load_gates(a.config)

    from ultralytics import YOLO
    model = YOLO(a.model)
    names = model.names
    predict = lambda fr: model.predict(fr, conf=a.conf, iou=a.iou, imgsz=a.imgsz,
                                       device=a.device, verbose=False)[0]
    print(f'model  {a.model}\n       task={model.task} classes={list(names.values())}')
    print(f'gates  {gates["source"]}\n'
          f'       proboscis_filter.max_eye_prob_distance_px={gates["max_eye_prob_distance_px"]}'
          f'  up_divisor={gates["up_divisor"]}\n'
          f'       distance_limits.three_fly_max_eye_prob_distance_px='
          f'{gates["three_fly_max_eye_prob_distance_px"]}')

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f'could not open {a.video}')
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'video  {a.video}\n       {W}x{H}  {total} frames  {fps:.1f} fps  stride {a.stride}')

    # ---- phase 1: freeze eyes (and the posterior axis) from the first frames ----
    scan = []
    for _ in range(a.freeze_eyes):
        ok, fr = cap.read()
        if not ok:
            break
        scan.append(predict(fr))
    counts = [sum(1 for lb, _, _ in obb_parts(r, names) if lb == 'eye') for r in scan]
    n_flies = a.max_flies or (Counter(counts).most_common(1)[0][0] if counts else 0)
    mgr = EyeAnchorManager(max_eyes=n_flies, zero_iou_eps=ZERO_IOU_EPS)
    corner_pool = []
    for r in scan:
        ps = obb_parts(r, names)
        eyes = [(p, s) for lb, p, s in ps if lb == 'eye']
        if eyes:
            mgr.try_update_from_dets(np.stack([to_xyxy(p) for p, _ in eyes]),
                                     np.array([s for _, s in eyes], np.float32))
        corner_pool.extend(p for p, _ in eyes)
    if not mgr.confirmed:
        print(f'  ! eyes not confirmed in {a.freeze_eyes} frames '
              f'({len(mgr.anchors_xyxy)}/{n_flies} anchors) - '
              'deployment would fall back too')
    # exact eye_scale: match each frozen anchor to the OBB corners it came from
    anchors = []
    for c in mgr.get_centers():
        c = np.asarray(c, float)
        if corner_pool:
            best = min(corner_pool, key=lambda p: np.linalg.norm(FG.polygon_centroid(p) - c))
            anchors.append((c, float(np.sqrt(FG.polygon_area(best)))))
        else:
            anchors.append((c, 0.0))
    print(f'frozen {len(anchors)} eyes from {len(scan)} frames '
          f'(slots numbered top-to-bottom): '
          + ', '.join(f'({c[0]:.0f},{c[1]:.0f}) scale {s:.1f}px' for c, s in anchors))

    # posterior axis, frozen from the same frames
    axis_pairs = []
    for r in scan:
        ps = obb_parts(r, names)
        flies = FG.assemble([FG.Part(lb, p, s) for lb, p, s in ps], W, H, limits)
        axis_pairs += [(f.eye.centroid, f.abdomen.centroid) for f in flies if f.abdomen]
    axis = FG.estimate_posterior_axis(axis_pairs, limits=limits)
    if axis is not None:
        print(f'frozen posterior axis {math.degrees(math.atan2(axis[1], axis[0])) % 360:.1f} deg '
              f'(0=right, 90=down) from {len(axis_pairs)} pairs')

    # ---- phase 2: run the video ----
    writer = None
    if not a.no_annotated:
        writer = cv2.VideoWriter(os.path.join(out, 'annotated.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps / a.stride, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows, per_frame, conf_by = [], [], defaultdict(list)
    eyes_seq, reasons = [], Counter()
    kept = withheld = 0
    inside = Counter(); outside = Counter()
    prod_in = Counter(); prod_out = Counter()
    idx = done = 0
    t0 = time.time()
    while True:
        if not cap.grab():
            break
        if idx % a.stride:
            idx += 1
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        parts = obb_parts(predict(fr), names)
        cnt = Counter(lb for lb, _, _ in parts)
        per_frame.append(cnt)
        flies = FG.assemble([FG.Part(lb, p, s) for lb, p, s in parts], W, H, limits)
        # Attribute each part to the fly the rules assigned it to. Measuring an
        # abdomen or tip against the NEAREST frozen eye is wrong: they sit 3.4-10
        # eye-widths out, so the nearest eye is often the neighbouring fly's.
        owner = {}
        for f in flies:
            for part in (f.proboscis, f.abdomen, f.tip):
                if part is not None:
                    owner[id(part)] = f
        assigned = {lb: set() for lb in ('proboscis', 'abdomen', 'abdomenTip')}
        for f in flies:
            for part, lb in ((f.proboscis, 'proboscis'), (f.abdomen, 'abdomen'),
                             (f.tip, 'abdomenTip')):
                if part is not None:
                    assigned[lb].add(id(part))
        for lb, p, s in parts:
            conf_by[lb].append(s)
            c = FG.polygon_centroid(p)
            rows.append(dict(frame=idx, cls=lb, conf=round(s, 4),
                             cx=round(float(c[0]), 2), cy=round(float(c[1]), 2),
                             pts=';'.join(f'{x:.1f},{y:.1f}' for x, y in p)))
        # per-class: did the part get claimed by a fly under the combined rules?
        for lb in ('proboscis', 'abdomen', 'abdomenTip'):
            n_lb = cnt[lb]
            n_ok = len(assigned[lb])
            inside[lb] += n_ok
            outside[lb] += max(0, n_lb - n_ok)
        # the production proboscis gate is anisotropic and anchored on the FROZEN
        # eye, so it is checked separately against the nearest anchor (valid here:
        # a proboscis sits ~1.3 eye-widths out, the next eye >=1.6x farther)
        if anchors:
            for lb, p, _ in parts:
                if lb != 'proboscis':
                    continue
                c = FG.polygon_centroid(p)
                ec, es = min(anchors, key=lambda t: np.linalg.norm(t[0] - c))
                dx, dy = c - ec
                R = gates['max_eye_prob_distance_px']
                b_ = R / gates['up_divisor'] if dy < 0 else R
                if (dx / R) ** 2 + (dy / b_) ** 2 <= 1.0:
                    prod_in['proboscis'] += 1
                else:
                    prod_out['proboscis'] += 1
        flies = FG.assemble([FG.Part(lb, p, s) for lb, p, s in parts], W, H, limits)
        for f in flies:
            if f.abdomen is not None:
                kept += 1
            else:
                withheld += 1
                reasons.update(f.rejected or ['unspecified'])
        eyes_seq.append((idx, sorted([FG.polygon_centroid(p) for lb, p, _ in parts
                                      if lb == 'eye'], key=lambda c: c[1])))
        if writer is not None:
            writer.write(draw_frame(fr, parts, flies, anchors, axis, gates, limits,
                                    a.draw_boundaries))
        done += 1
        idx += 1
        if done % 500 == 0:
            print(f'  {done} frames  {done/(time.time()-t0):.1f} fps', flush=True)
        if a.max_frames and done >= a.max_frames:
            break
    cap.release()
    if writer is not None:
        writer.release()
    el = time.time() - t0
    if not done:
        sys.exit('no frames processed')

    with open(os.path.join(out, 'detections.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['frame', 'cls', 'conf', 'cx', 'cy', 'pts'])
        w.writeheader(); w.writerows(rows)

    eye_counts = Counter(c['eye'] for c in per_frame)
    expect = n_flies or (eye_counts.most_common(1)[0][0] if eye_counts else 0)
    S = dict(model=a.model, video=a.video, frames=done, stride=a.stride, conf=a.conf,
             imgsz=a.imgsz, seconds=round(el, 1), fps=round(done / el, 2),
             freeze_eyes=a.freeze_eyes, frozen_anchors=len(anchors),
             expected_flies=expect, gates=gates,
             posterior_axis_deg=(None if axis is None else
                                 round(math.degrees(math.atan2(axis[1], axis[0])) % 360, 1)),
             classes={})
    print(f'\n{"="*70}\n{done} frames in {el:.1f}s ({done/el:.1f} fps)\n{"="*70}')
    print(f'{"class":12s} {"dets":>7s} {"per frame":>10s} {"frames>=1":>10s} '
          f'{"conf p5":>8s} {"conf p50":>9s} {"claimed":>9s} {"prod gate":>10s}')
    for lb in ('eye', 'proboscis', 'abdomen', 'abdomenTip'):
        n = sum(c[lb] for c in per_frame)
        got = sum(1 for c in per_frame if c[lb])
        cf = np.array(conf_by.get(lb) or [0.0])
        tot_b = inside[lb] + outside[lb]
        ib = f'{100*inside[lb]/tot_b:.1f}%' if tot_b else '-'
        tot_p = prod_in[lb] + prod_out[lb]
        pg = f'{100*prod_in[lb]/tot_p:.1f}%' if tot_p else '-'
        print(f'{lb:12s} {n:7d} {n/done:10.2f} {100*got/done:9.1f}% '
              f'{np.percentile(cf,5):8.2f} {np.median(cf):9.2f} {ib:>9s} {pg:>10s}')
        S['classes'][lb] = dict(total=n, per_frame=round(n / done, 3),
                                frames_with_any=round(100 * got / done, 2),
                                conf_p5=round(float(np.percentile(cf, 5)), 3),
                                conf_p50=round(float(np.median(cf)), 3),
                                claimed_pct=(round(100 * inside[lb] / tot_b, 2)
                                             if tot_b else None),
                                production_gate_pct=(round(100 * prod_in[lb] / tot_p, 2)
                                                     if tot_p else None))
    print(f'\nfly-count stability (expected {expect}):')
    for k in sorted(eye_counts):
        print(f'  {k} eyes  {"#"*max(1,int(60*eye_counts[k]/done))} {eye_counts[k]:5d} '
              f'({100*eye_counts[k]/done:5.1f}%)' + ('  <-- expected' if k == expect else ''))
    S['eye_count_hist'] = {int(k): v for k, v in eye_counts.items()}

    full = sum(1 for c in per_frame
               if min(c['eye'], c['proboscis'], c['abdomen'], c['abdomenTip']) >= expect)
    print(f'\nframes with a COMPLETE set for all {expect} flies: {full}/{done} '
          f'({100*full/done:.1f}%)')
    S['pct_frames_complete'] = round(100 * full / done, 2)

    steps = defaultdict(list)
    for (i0, e0), (i1, e1) in zip(eyes_seq, eyes_seq[1:]):
        if i1 - i0 != a.stride or len(e0) != expect or len(e1) != expect:
            continue
        for s, (c0, c1) in enumerate(zip(e0, e1)):
            steps[s].append(float(np.linalg.norm(c1 - c0)))
    print('\npositional jitter of eye centroids (tethered -> should be ~0):')
    for s, tr in sorted(steps.items()):
        d = np.array(tr)
        if len(d) < 2:
            continue
        print(f'  slot {s}: n={len(d):5d}  median {np.median(d):5.2f} px  '
              f'p95 {np.percentile(d,95):6.2f}  max {d.max():7.1f}')
        S.setdefault('eye_jitter', {})[f'slot{s}'] = dict(
            n=len(d), median=round(float(np.median(d)), 2),
            p95=round(float(np.percentile(d, 95)), 2))
    if kept + withheld:
        tot = kept + withheld
        print(f'\ncombined geometry rules: {kept}/{tot} flies kept a trackable abdomen '
              f'({100*kept/tot:.1f}%)')
        for r, v in reasons.most_common(5):
            print(f'   withheld {v:6d}  {r[:62]}')
        S['rules'] = dict(kept=kept, withheld=withheld,
                          pct_kept=round(100 * kept / tot, 2),
                          reasons=dict(reasons.most_common(8)))
    with open(os.path.join(out, 'summary.json'), 'w') as fh:
        json.dump(S, fh, indent=2, default=str)
    print(f'\nwrote {out}/detections.csv, summary.json'
          + ('' if a.no_annotated else ', annotated.mp4'))


if __name__ == '__main__':
    main()
