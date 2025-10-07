
from __future__ import annotations
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

from .vision import cxcywh_to_xyxy, iou, xyxy_to_cxcywh

class KalmanBBox:
    def __init__(self):
        self.x = np.zeros((8,1), dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32)*10.0
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4): self.F[i, i+4] = 1.0
        self.H = np.zeros((4,8), dtype=np.float32)
        self.H[0,0]=self.H[1,1]=self.H[2,2]=self.H[3,3]=1.0
        self.Q = np.eye(8, dtype=np.float32)*0.02
        self.R = np.eye(4, dtype=np.float32)*1.0

    def init(self, cxcywh):
        self.x[:4,0] = cxcywh; self.x[4:,0] = 0.0
        self.P = np.eye(8, dtype=np.float32)*10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4,0].copy()

    def update(self, z):
        z = z.reshape(4,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

class Track:
    _next_id = 1
    def __init__(self, cxcywh, score, ema_alpha=0.2):
        self.id = Track._next_id; Track._next_id += 1
        self.kf = KalmanBBox(); self.kf.init(cxcywh)
        self.score = float(score)
        self.box_xyxy = cxcywh_to_xyxy(cxcywh)
        self.time_since_update = 0
        self.hits = 1
        self.history = deque(maxlen=30)
        self.ema_alpha = ema_alpha

    def predict(self):
        pred = self.kf.predict()
        box = cxcywh_to_xyxy(pred)
        self.box_xyxy = box
        self.history.append(box.copy())
        self.time_since_update += 1
        return box

    def correct(self, cxcywh, score):
        self.kf.update(cxcywh)
        box = cxcywh_to_xyxy(self.kf.x[:4,0])
        self.box_xyxy = (1-self.ema_alpha)*box + self.ema_alpha*self.box_xyxy
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.box_xyxy.copy())

class _BaseTracker:
    def __init__(self, iou_thres=0.25, max_age=15, ema_alpha=0.2):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.ema_alpha = ema_alpha
        self.tracks: List[Track] = []

    def _associate(self, det_xyxy: np.ndarray, det_scores: np.ndarray) -> Tuple[set, set]:
        preds = [t.predict() for t in self.tracks]
        assigned_tr, assigned_det = set(), set()
        if len(self.tracks) and len(det_xyxy):
            M = iou(np.stack(preds), det_xyxy)
            while True:
                i, j = np.unravel_index(np.argmax(M), M.shape)
                if M[i, j] < self.iou_thres:
                    break
                if i in assigned_tr or j in assigned_det:
                    M[i, j] = -1
                    continue
                self.tracks[i].correct(xyxy_to_cxcywh(det_xyxy[j]), det_scores[j])
                assigned_tr.add(i)
                assigned_det.add(j)
                M[i, :] = -1
                M[:, j] = -1
        return assigned_tr, assigned_det

    def _spawn_tracks(self, det_xyxy: np.ndarray, det_scores: np.ndarray, assigned_det: set, max_tracks: Optional[int] = None) -> None:
        for j in range(len(det_xyxy)):
            if j in assigned_det:
                continue
            if max_tracks is not None and len(self.tracks) >= max_tracks:
                break
            self.tracks.append(Track(xyxy_to_cxcywh(det_xyxy[j]), det_scores[j], self.ema_alpha))

    def _prune(self) -> None:
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]


class SingleClassTracker(_BaseTracker):
    def step(self, det_xyxy: np.ndarray, det_scores: np.ndarray) -> Optional[Track]:
        _, assigned_det = self._associate(det_xyxy, det_scores)
        self._spawn_tracks(det_xyxy, det_scores, assigned_det)
        self._prune()
        if not self.tracks:
            return None
        self.tracks.sort(key=lambda t: (t.time_since_update, -t.hits, -t.score))
        return self.tracks[0]


class MultiObjectTracker(_BaseTracker):
    """Track multiple instances of a single class, returning the live tracks."""

    def __init__(self, iou_thres=0.25, max_age=15, ema_alpha=0.2, max_tracks=4):
        super().__init__(iou_thres=iou_thres, max_age=max_age, ema_alpha=ema_alpha)
        self.max_tracks = max_tracks

    def step(self, det_xyxy: np.ndarray, det_scores: np.ndarray) -> List[Track]:
        _, assigned_det = self._associate(det_xyxy, det_scores)
        self._spawn_tracks(det_xyxy, det_scores, assigned_det, max_tracks=self.max_tracks)
        self._prune()
        self.tracks.sort(key=lambda t: (t.time_since_update, -t.hits, -t.score))
        return self.tracks
