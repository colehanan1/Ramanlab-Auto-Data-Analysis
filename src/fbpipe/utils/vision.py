
from __future__ import annotations
import numpy as np

def order_corners(corners):
    pts = np.array(corners, dtype=np.float32)
    cx, cy = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    sorted_idx = np.argsort(angles)
    pts_sorted = pts[sorted_idx]
    return pts_sorted.tolist()

def xyxy_to_cxcywh(b):
    x1,y1,x2,y2 = b
    w = max(0.0, x2-x1); h = max(0.0, y2-y1)
    cx = x1 + w/2; cy = y1 + h/2
    return np.array([cx, cy, w, h], dtype=np.float32)

def cxcywh_to_xyxy(s):
    cx,cy,w,h = s
    x1 = cx - w/2; y1 = cy - h/2
    x2 = cx + w/2; y2 = cy + h/2
    return np.array([x1,y1,x2,y2], dtype=np.float32)

def iou(a, b):
    N = a.shape[0]; M = b.shape[0]
    if N==0 or M==0: return np.zeros((N,M), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:,0][:,None], a[:,1][:,None], a[:,2][:,None], a[:,3][:,None]
    bx1, by1, bx2, by2 = b[:,0][None,:], b[:,1][None,:], b[:,2][None,:], b[:,3][None,:]
    inter_w = np.maximum(0, np.minimum(ax2,bx2)-np.maximum(ax1,bx1))
    inter_h = np.maximum(0, np.minimum(ay2,by2)-np.maximum(ay1,by1))
    inter   = inter_w * inter_h
    area_a  = (ax2-ax1)*(ay1*-1+ay2)
    area_b  = (bx2-bx1)*(by1*-1+by2)
    union   = area_a + area_b - inter + 1e-6
    return (inter/union).astype(np.float32)
