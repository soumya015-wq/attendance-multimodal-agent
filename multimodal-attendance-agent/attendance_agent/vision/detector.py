"""Face detection backends (Activity 2.1).

Two interchangeable backends share one interface:

* ``MediaPipeDetector`` - MediaPipe face detection + face-mesh landmarks, which
  gives true head pose (yaw / pitch / roll) via PnP. Preferred.
* ``HaarDetector``      - OpenCV Haar cascades. Ships with OpenCV, so the
  system always works offline even if MediaPipe cannot be installed. Frontal
  faces => "facing forward"; profile faces => "looking away".

``create_detector("auto")`` picks MediaPipe when its legacy ``solutions`` API is
available and falls back to Haar otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    yaw: Optional[float] = None       # degrees, None when the backend cannot estimate it
    pitch: Optional[float] = None
    roll: Optional[float] = None
    facing_forward: Optional[bool] = None
    spawn: bool = True                # may this detection start a NEW track? (profile hits may not)

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def _nms(dets: List[Detection], iou_thr: float = 0.4) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    keep: List[Detection] = []
    for d in dets:
        if all(iou(d.box, k.box) < iou_thr for k in keep):
            keep.append(d)
    return keep


def overlap_min(a, b) -> float:
    """Intersection over the *smaller* box - catches a small box lying beside/inside a bigger one."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / max(1, min(aw * ah, bw * bh))


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class BaseDetector:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:  # pragma: no cover
        raise NotImplementedError

    def close(self):
        pass


# --------------------------------------------------------------------------- Haar
class HaarDetector(BaseDetector):
    name = "haar"

    def __init__(self, cfg):
        super().__init__(cfg)
        base = cv2.data.haarcascades
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # local contrast; global equalizeHist hurts on plain backgrounds
        self.frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self.frontal.empty() or self.profile.empty():
            raise RuntimeError("OpenCV Haar cascades could not be loaded")

    def detect(self, frame_bgr):
        s = self.cfg.detect_scale
        small = cv2.resize(frame_bgr, None, fx=s, fy=s) if s != 1.0 else frame_bgr
        gray = self._clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        min_sz = max(20, int(self.cfg.min_face_px * s))
        out: List[Detection] = []

        for (x, y, w, h) in self.frontal.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_sz, min_sz)):
            out.append(Detection(int(x / s), int(y / s), int(w / s), int(h / s), 1.0,
                                 facing_forward=True))

        prof = []
        for (x, y, w, h) in self.profile.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_sz, min_sz)):
            prof.append(Detection(int(x / s), int(y / s), int(w / s), int(h / s), 0.6,
                                  facing_forward=False))
        flipped = cv2.flip(gray, 1)
        W = gray.shape[1]
        for (x, y, w, h) in self.profile.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=6, minSize=(min_sz, min_sz)):
            prof.append(Detection(int((W - x - w) / s), int(y / s), int(w / s), int(h / s), 0.6,
                                  facing_forward=False))
        # Profile hits are noisy: drop any that touch a frontal face, and never let them
        # start a new participant (they may only flag an existing track as "looking away").
        frontal = [d for d in out]
        prof = [p for p in _nms(prof) if all(overlap_min(p.box, f.box) < 0.15 for f in frontal)]
        for p in prof:
            p.spawn = False
        out = [d for d in frontal + prof if min(d.w, d.h) >= self.cfg.min_face_px]
        return out[: self.cfg.max_faces]


# ----------------------------------------------------------------------- MediaPipe
# Canonical 3-D face model (mm) for PnP head pose.
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),            # nose tip            (mesh 1)
    (0.0, -330.0, -65.0),       # chin                (mesh 152)
    (-225.0, 170.0, -135.0),    # left eye outer      (mesh 33)
    (225.0, 170.0, -135.0),     # right eye outer     (mesh 263)
    (-150.0, -150.0, -125.0),   # left mouth corner   (mesh 61)
    (150.0, -150.0, -125.0),    # right mouth corner  (mesh 291)
], dtype=np.float64)
_MESH_IDX = [1, 152, 33, 263, 61, 291]


def head_pose_from_landmarks(landmarks, width: int, height: int):
    """Return (yaw, pitch, roll) in degrees from 468 face-mesh landmarks."""
    pts = np.array([(landmarks[i].x * width, landmarks[i].y * height) for i in _MESH_IDX],
                   dtype=np.float64)
    focal = float(width)
    cam = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = (float(a) for a in angles)

    def norm(a):  # fold angles into (-90, 90]
        while a > 90:
            a -= 180
        while a <= -90:
            a += 180
        return a

    return norm(yaw), norm(pitch), norm(roll)


class MediaPipeDetector(BaseDetector):
    name = "mediapipe"

    def __init__(self, cfg):
        super().__init__(cfg)
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError("mediapipe legacy `solutions` API not available in this version")
        self._fd = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                             min_detection_confidence=0.5)
        self._fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                   refine_landmarks=False,
                                                   min_detection_confidence=0.4)

    def detect(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._fd.process(rgb)
        out: List[Detection] = []
        for det in (res.detections or [])[: self.cfg.max_faces]:
            bb = det.location_data.relative_bounding_box
            x, y = int(bb.xmin * W), int(bb.ymin * H)
            w, h = int(bb.width * W), int(bb.height * H)
            x, y = max(0, x), max(0, y)
            w, h = min(w, W - x), min(h, H - y)
            if min(w, h) < self.cfg.min_face_px:
                continue
            d = Detection(x, y, w, h, float(det.score[0]) if det.score else 1.0)
            # pose from landmarks on a padded crop
            pad = int(0.35 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = rgb[y0:y1, x0:x1]
            if crop.size:
                mesh = self._fm.process(crop)
                if mesh.multi_face_landmarks:
                    pose = head_pose_from_landmarks(mesh.multi_face_landmarks[0].landmark,
                                                    crop.shape[1], crop.shape[0])
                    if pose:
                        d.yaw, d.pitch, d.roll = pose
                        d.facing_forward = (abs(d.yaw) <= self.cfg.yaw_limit_deg and
                                            abs(d.pitch) <= self.cfg.pitch_limit_deg)
            out.append(d)
        return out

    def close(self):
        self._fd.close()
        self._fm.close()


def create_detector(cfg) -> BaseDetector:
    backend = cfg.detector_backend
    if backend in ("auto", "mediapipe"):
        try:
            return MediaPipeDetector(cfg)
        except Exception as exc:  # noqa: BLE001
            if backend == "mediapipe":
                raise
            print(f"[vision] MediaPipe unavailable ({exc}); falling back to OpenCV Haar cascades.")
    return HaarDetector(cfg)
