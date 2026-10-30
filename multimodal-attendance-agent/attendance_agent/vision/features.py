"""Behavioural features from tracks: attention (head pose), movement, inactivity
(Activity 4.2 - the visual half of the fusion)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..attendance import sample_grid


def _movement_score(m: np.ndarray, cfg) -> np.ndarray:
    """1.0 for calm or natural movement; decays towards 0.2 for excessive fidgeting.
    (Being still is *not* penalised here - prolonged stillness is handled by the inactivity feature.)"""
    s = np.full(m.shape, np.nan)
    ok = ~np.isnan(m)
    mm = m[ok]
    s[ok] = np.where(mm <= cfg.fidget_motion_thr, 1.0,
                     np.maximum(0.2, 1.0 - (mm - cfg.fidget_motion_thr) / cfg.fidget_motion_thr))
    return s


def participant_series(det: pd.DataFrame, track_id: int, grid: np.ndarray, cfg) -> dict:
    """Project one track's detections onto the sampling grid."""
    n = len(grid)
    d = det[det.track_id == track_id]
    detected = np.zeros(n, dtype=bool)
    attentive = np.zeros(n, dtype=bool)
    motion = np.full(n, np.nan)
    idx = np.clip(np.rint(d["t"].to_numpy() * cfg.sample_fps).astype(int), 0, n - 1)
    ff = d["facing_forward"].to_numpy(dtype=object)
    yaw = d["yaw"].to_numpy(dtype=float)
    pitch = d["pitch"].to_numpy(dtype=float)
    for k, i in enumerate(idx):
        detected[i] = True
        if not np.isnan(yaw[k]) and not np.isnan(pitch[k]):
            a = abs(yaw[k]) <= cfg.yaw_limit_deg and abs(pitch[k]) <= cfg.pitch_limit_deg
        elif ff[k] is None or (isinstance(ff[k], float) and np.isnan(ff[k])):
            a = True            # backend cannot tell: assume attentive while visible
        else:
            a = bool(ff[k])
        attentive[i] = a
        motion[i] = d["motion"].iloc[k]
    # smooth motion over ~1 s to suppress detector jitter
    k = max(1, int(round(cfg.sample_fps)))
    motion = np.array(pd.Series(motion).rolling(k, min_periods=1, center=True).mean().to_numpy(), copy=True)
    motion[~detected] = np.nan
    return dict(detected=detected, attentive=attentive, motion=motion)


def window_vision_features(det, masks, id_map, duration, cfg) -> pd.DataFrame:
    grid = sample_grid(duration, cfg.sample_fps)
    n_win = max(1, int(np.ceil(duration / cfg.window_seconds)))
    win_of = np.minimum((grid // cfg.window_seconds).astype(int), n_win - 1)
    dt = 1.0 / cfg.sample_fps
    rows = []
    for tid, pid in id_map.items():
        s = participant_series(det, tid, grid, cfg)
        present = masks[pid]
        # stillness run-length (seconds) while present
        still = present & (np.nan_to_num(s["motion"], nan=0.0) < cfg.still_motion_thr)
        run = np.zeros(len(grid))
        for i in range(len(grid)):
            run[i] = (run[i - 1] + dt) if (i > 0 and still[i]) else (dt if still[i] else 0.0)
        inactive = run >= cfg.inactivity_seconds
        mv = _movement_score(s["motion"], cfg)
        for w in range(n_win):
            sel = win_of == w
            n_sel = int(sel.sum())
            n_pres = int((present & sel).sum())
            if n_sel == 0:
                continue
            pr = n_pres / n_sel
            if n_pres == 0:
                att = mvs = ina = np.nan
            else:
                att = float((s["attentive"] & present & sel).sum() / n_pres)
                m = mv[sel & s["detected"]]
                mvs = float(np.nanmean(m)) if m.size and not np.all(np.isnan(m)) else 0.5
                ina = float((inactive & present & sel).sum() / n_pres)
            rows.append(dict(participant=pid, window=w, t_start=w * cfg.window_seconds,
                             t_end=min(duration, (w + 1) * cfg.window_seconds),
                             present_ratio=round(pr, 3), attention=att, movement=mvs,
                             inactivity=ina))
    return pd.DataFrame(rows)
