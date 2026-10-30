"""Multi-face tracker (Activity 2.1).

Associates detections across frames using a blend of IoU and normalised centroid
distance (Hungarian assignment). Faces that disappear are kept "lost" for a short
time and can be *re-identified* when they come back, using a lightweight HSV colour
histogram of the face region. No face image or embedding ever leaves memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection, iou


def face_histogram(frame_bgr: np.ndarray, det: Detection) -> Optional[np.ndarray]:
    H, W = frame_bgr.shape[:2]
    # centre of the face box: avoids hair/background
    mx, my = int(det.w * 0.15), int(det.h * 0.15)
    x0, y0 = max(0, det.x + mx), max(0, det.y + my)
    x1, y1 = min(W, det.x + det.w - mx), min(H, det.y + det.h - my)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


@dataclass
class Track:
    track_id: int
    box: tuple
    last_t: float
    hist: Optional[np.ndarray] = None
    hits: int = 1
    prev_center: Optional[tuple] = None
    prev_t: Optional[float] = None


class FaceTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.active: Dict[int, Track] = {}
        self.lost: Dict[int, Track] = {}
        self._next_id = 1

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _affinity(track_box, det_box) -> float:
        tx, ty, tw, th = track_box
        dx, dy, dw, dh = det_box
        i = iou(track_box, det_box)
        tc = (tx + tw / 2, ty + th / 2)
        dc = (dx + dw / 2, dy + dh / 2)
        dist = np.hypot(tc[0] - dc[0], tc[1] - dc[1]) / max(tw, th, 1)
        cent = max(0.0, 1.0 - dist / 1.5)           # 1 when same centre, 0 when >1.5 face-sizes away
        size_ratio = min(tw, dw) / max(tw, dw, 1)
        if size_ratio < 0.4:
            return 0.0
        return max(i, 0.9 * cent * size_ratio)

    def update(self, t: float, detections: List[Detection], frame_bgr=None) -> List[dict]:
        cfg = self.cfg
        # move stale active tracks to "lost"
        for tid in list(self.active):
            if t - self.active[tid].last_t > cfg.max_missing_seconds:
                self.lost[tid] = self.active.pop(tid)
        for tid in list(self.lost):
            if t - self.lost[tid].last_t > cfg.reid_window_seconds:
                del self.lost[tid]

        ids = list(self.active)
        assigned: Dict[int, int] = {}      # detection idx -> track id
        if ids and detections:
            cost = np.ones((len(ids), len(detections)))
            for i, tid in enumerate(ids):
                for j, d in enumerate(detections):
                    cost[i, j] = 1.0 - self._affinity(self.active[tid].box, d.box)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if 1.0 - cost[r, c] >= cfg.iou_threshold:
                    assigned[c] = ids[r]

        out: List[dict] = []
        for j, d in enumerate(detections):
            hist = face_histogram(frame_bgr, d) if frame_bgr is not None else None
            tid = assigned.get(j)
            if tid is None and not d.spawn:
                continue                       # unmatched profile hit: ignore
            if tid is None:
                tid = self._try_reid(hist, d, assigned.values())
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                trk = Track(tid, d.box, t, hist)
            else:
                trk = self.active.pop(tid, None) or self.lost.pop(tid)
                trk.hits += 1
                if hist is not None:
                    trk.hist = hist if trk.hist is None else 0.8 * trk.hist + 0.2 * hist
            cx, cy = d.center
            motion = 0.0
            if trk.prev_center is not None and trk.prev_t is not None and t > trk.prev_t:
                dist = np.hypot(cx - trk.prev_center[0], cy - trk.prev_center[1])
                motion = float(dist / max(d.w, 1) / (t - trk.prev_t))   # face-widths per second
            trk.prev_center, trk.prev_t = (cx, cy), t
            trk.box, trk.last_t = d.box, t
            self.active[tid] = trk
            out.append(dict(t=round(t, 3), track_id=tid, x=d.x, y=d.y, w=d.w, h=d.h,
                            cx=cx, cy=cy, motion=motion, yaw=d.yaw, pitch=d.pitch,
                            roll=d.roll, facing_forward=d.facing_forward, score=d.score))
        return out

    def _try_reid(self, hist, det, taken) -> Optional[int]:
        if hist is None or not self.lost:
            return None
        best, best_s = None, self.cfg.reid_hist_threshold
        for tid, trk in self.lost.items():
            if trk.hist is None or tid in taken:
                continue
            s = float(cv2.compareHist(trk.hist, hist, cv2.HISTCMP_CORREL))
            # prefer faces re-appearing roughly where they left
            if s > best_s:
                best, best_s = tid, s
        return best
