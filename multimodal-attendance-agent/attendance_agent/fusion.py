"""Multimodal signal fusion (Activity 4.2) and engagement scoring (Activity 4.3).

Vision indicators (presence, attention, movement, inactivity) and audio indicators
(speech activity, noise) are synchronised on a shared window index, then combined by
transparent, adjustable rules into an engagement score in [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ("Low", "Medium", "High")


def label_for(score: float, cfg) -> str:
    if score is None or np.isnan(score):
        return "n/a"
    if score >= cfg.high_thr:
        return "High"
    if score < cfg.low_thr:
        return "Low"
    return "Medium"


def audio_context_score(speech_ratio: float) -> float:
    """Silence is neutral (0.5: listening / reading); speech raises interaction up to 1.0."""
    if speech_ratio is None or np.isnan(speech_ratio):
        return np.nan
    return 0.5 + 0.5 * min(1.0, speech_ratio / 0.4)


def fuse(vision_w: pd.DataFrame, audio_w: pd.DataFrame, attendance: pd.DataFrame, cfg):
    """Return (student_windows, class_timeline)."""
    present_ids = set(attendance.loc[attendance.marked_present, "participant"])
    v = vision_w[vision_w.participant.isin(present_ids)].copy()
    if v.empty:
        return v, pd.DataFrame()
    a = audio_w.rename(columns={"t_start": "a_start", "t_end": "a_end"})
    m = v.merge(a[["window", "speech_ratio", "noise_ratio", "audio_state", "mean_db"]],
                on="window", how="left")

    wsum = cfg.w_movement + cfg.w_activity
    behaviour = (cfg.w_movement * m.movement + cfg.w_activity * (1 - m.inactivity)) / wsum
    visual = m.attention * behaviour            # attention gates engagement
    audio = m.speech_ratio.apply(audio_context_score)
    both = visual.notna() & audio.notna()
    score = np.where(both, (cfg.w_visual * visual + cfg.w_audio * audio) / (cfg.w_visual + cfg.w_audio),
                     visual)
    noisy = (m.audio_state == "noisy").to_numpy()
    score = np.where(noisy, score - cfg.noise_penalty, score)
    # windows where the participant was barely present are not scored
    score = np.where(m.present_ratio < 0.2, np.nan, score)
    m["behaviour"] = behaviour.round(3)
    m["visual_score"] = visual.round(3)
    m["audio_score"] = audio.round(3)
    m["engagement"] = np.clip(score, 0, 1).round(3)
    m["label"] = m.engagement.apply(lambda s: label_for(s, cfg))

    # ---- classroom level timeline ----
    g = m.groupby("window")
    tl = g.agg(t_start=("t_start", "first"), t_end=("t_end", "first"),
               participants=("participant", "nunique"),
               engagement=("engagement", "mean"), attention=("attention", "mean"),
               movement=("movement", "mean"), inactivity=("inactivity", "mean"),
               speech_ratio=("speech_ratio", "first"), noise_ratio=("noise_ratio", "first"),
               audio_state=("audio_state", "first")).reset_index()
    tl["engagement"] = tl.engagement.round(3)
    tl["engagement_smooth"] = (tl.engagement.rolling(cfg.smooth_windows, min_periods=1, center=True)
                               .mean().round(3))
    tl["label"] = tl.engagement_smooth.apply(lambda s: label_for(s, cfg))
    return m, tl
