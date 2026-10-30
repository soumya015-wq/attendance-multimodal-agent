"""Presence-based attendance logic and timestamped logs (Activity 2.2).

Attendance is *never* decided from a single frame. For each tracked participant we
build a presence timeline on the sampling grid, bridge short gaps (blinks, brief
occlusions), accumulate presence and mark the participant present only when the
accumulated time passes a threshold.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd


def sample_grid(duration: float, sample_fps: float) -> np.ndarray:
    n = max(1, int(np.ceil(duration * sample_fps)))
    return np.arange(n) / sample_fps


def presence_mask(times: np.ndarray, grid: np.ndarray, sample_fps: float,
                  merge_gap: float) -> np.ndarray:
    """Boolean mask over ``grid``: True while a participant is considered present.

    A grid point is "detected" if a detection lies within half a sample step.
    Gaps between detections no longer than ``merge_gap`` are bridged.
    """
    if len(times) == 0 or len(grid) == 0:
        return np.zeros(len(grid), dtype=bool)
    times = np.sort(np.asarray(times, dtype=float))
    dt = 1.0 / sample_fps
    pos = np.searchsorted(times, grid)
    left = np.where(pos > 0, times[np.clip(pos - 1, 0, len(times) - 1)], -np.inf)
    right = np.where(pos < len(times), times[np.clip(pos, 0, len(times) - 1)], np.inf)
    detected = np.minimum(np.abs(grid - left), np.abs(right - grid)) <= dt * 0.55
    # bridge gaps: present if a detection exists both before and after within merge_gap total
    gap_ok = (grid - left >= 0) & (right - grid >= 0) & ((right - left) <= merge_gap + dt)
    return detected | gap_ok


def compute_attendance(det: pd.DataFrame, duration: float, cfg,
                       session_start: datetime | None = None):
    """Return (attendance_df, events_df, masks) for all tracks."""
    session_start = session_start or datetime.now().replace(microsecond=0)
    grid = sample_grid(duration, cfg.sample_fps)
    threshold = min(cfg.min_presence_seconds, cfg.min_presence_ratio * max(duration, 1e-9))
    threshold = max(threshold, 3.0 / cfg.sample_fps)
    dt = 1.0 / cfg.sample_fps

    masks: Dict[int, np.ndarray] = {}
    records: List[dict] = []
    events: List[dict] = []

    ids = sorted(det["track_id"].unique()) if not det.empty else []
    # label participants P1, P2.. in order of first appearance
    first = {tid: det.loc[det.track_id == tid, "t"].min() for tid in ids}
    ids = sorted(ids, key=lambda i: first[i])
    for n, tid in enumerate(ids, start=1):
        pid = f"P{n}"
        times = det.loc[det.track_id == tid, "t"].to_numpy()
        mask = presence_mask(times, grid, cfg.sample_fps, cfg.merge_gap_seconds)
        masks[pid] = mask
        presence_s = float(mask.sum() * dt)
        present = presence_s >= threshold
        segs = _segments(mask, dt)
        records.append(dict(
            participant=pid, track_id=int(tid),
            status="Present" if present else "Absent (below threshold)",
            marked_present=bool(present),
            presence_seconds=round(presence_s, 1),
            presence_pct=round(100 * presence_s / max(duration, 1e-9), 1),
            first_seen=_ts(session_start, segs[0][0]) if segs else "",
            last_seen=_ts(session_start, segs[-1][1]) if segs else "",
            first_seen_s=round(segs[0][0], 1) if segs else None,
            last_seen_s=round(segs[-1][1], 1) if segs else None,
            segments=len(segs),
        ))
        for k, (a, b) in enumerate(segs):
            events.append(dict(participant=pid, event="appeared" if k == 0 else "returned",
                               timestamp=_ts(session_start, a), t_seconds=round(a, 1)))
            events.append(dict(participant=pid, event="left" if k == len(segs) - 1 else "stepped_out",
                               timestamp=_ts(session_start, b), t_seconds=round(b, 1)))
        if present:
            # moment attendance was confirmed = when cumulative presence crossed the threshold
            cum = np.cumsum(mask) * dt
            t_conf = float(grid[int(np.argmax(cum >= threshold))])
            events.append(dict(participant=pid, event="attendance_confirmed",
                               timestamp=_ts(session_start, t_conf), t_seconds=round(t_conf, 1)))
    att = pd.DataFrame(records)
    ev = pd.DataFrame(events)
    if not ev.empty:
        ev = ev.sort_values(["t_seconds", "participant"]).reset_index(drop=True)
    return att, ev, masks, {int(r["track_id"]): r["participant"] for r in records}


def _segments(mask: np.ndarray, dt: float):
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * dt, i * dt))
            start = None
    if start is not None:
        segs.append((start * dt, len(mask) * dt))
    return segs


def _ts(start: datetime, seconds: float) -> str:
    return (start + timedelta(seconds=float(seconds))).strftime("%Y-%m-%d %H:%M:%S")
