"""Video -> per-detection table (Activities 2.1, 6.1).

Reads a webcam index or a video file, samples frames at ``cfg.sample_fps``, detects
and tracks faces, and returns a tidy ``DataFrame`` with one row per tracked face per
sampled frame. Optionally writes an annotated video (faces blurred by default).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np
import pandas as pd

from .detector import create_detector
from .tracker import FaceTracker

COLUMNS = ["t", "track_id", "x", "y", "w", "h", "cx", "cy", "motion",
           "yaw", "pitch", "roll", "facing_forward", "score"]


@dataclass
class VisionResult:
    detections: pd.DataFrame
    duration: float          # seconds of video analysed
    video_fps: float
    frames_sampled: int
    backend: str
    frame_size: tuple


def process_video(source: Union[str, int, Path], cfg, annotate_path: Optional[str] = None,
                  max_seconds: Optional[float] = None,
                  progress: Optional[Callable[[float], None]] = None,
                  show: bool = False) -> VisionResult:
    is_cam = isinstance(source, int)
    cap = cv2.VideoCapture(source if is_cam else str(source))
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 1 or fps > 240:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / cfg.sample_fps)))
    sample_dt = step / fps

    detector = create_detector(cfg)
    tracker = FaceTracker(cfg)
    writer = None
    if annotate_path:
        Path(annotate_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(annotate_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 cfg.sample_fps, (W, H))

    rows, idx, sampled, t_wall0 = [], 0, 0, time.time()
    last_t = 0.0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = (time.time() - t_wall0) if is_cam else idx / fps
                if max_seconds is not None and t > max_seconds:
                    break
                dets = detector.detect(frame)
                rows.extend(tracker.update(t, dets, frame))
                sampled += 1
                last_t = t
                if writer is not None or show:
                    vis = _annotate(frame, rows, t, cfg)
                    if writer is not None:
                        writer.write(vis)
                    if show:
                        cv2.imshow("Attendance Agent (q to stop)", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                if progress and total:
                    progress(min(1.0, idx / total))
            idx += 1
    finally:
        cap.release()
        detector.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = _drop_spurious_tracks(df, cfg)
    duration = last_t + sample_dt if sampled else 0.0
    return VisionResult(df, float(duration), float(fps), sampled, detector.name, (W, H))


def _drop_spurious_tracks(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Remove tracks seen only a handful of times (flicker / false positives)."""
    if df.empty:
        return df
    counts = df.groupby("track_id")["t"].transform("count")
    return df[counts >= cfg.min_track_detections].reset_index(drop=True)


def _annotate(frame, rows, t, cfg):
    vis = frame.copy()
    for r in reversed(rows):
        if r["t"] != round(t, 3):
            break
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        if cfg.blur_faces_in_annotation:
            roi = vis[max(0, y):y + h, max(0, x):x + w]
            if roi.size:
                k = max(3, (min(w, h) // 3) | 1)
                vis[max(0, y):y + h, max(0, x):x + w] = cv2.GaussianBlur(roi, (k, k), 0)
        color = (60, 200, 60) if r["facing_forward"] is not False else (40, 140, 240)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, f"ID {int(r['track_id'])}", (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(vis, f"t={t:6.1f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis
