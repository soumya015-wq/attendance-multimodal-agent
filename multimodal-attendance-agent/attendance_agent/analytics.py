"""Student-facing session analytics (Activity 5.2).

Reflective, private summaries: attendance duration, engagement consistency and
notable trends. By design there are **no grades, rankings or peer comparisons** -
each report describes only the individual's own session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .agent import mmss, _runs


def student_report(pid: str, att_row: pd.Series, windows: pd.DataFrame, cfg) -> dict:
    w = windows[windows.participant == pid].sort_values("window").reset_index(drop=True)
    base = dict(
        participant=pid,
        attendance=dict(status=att_row["status"], presence_seconds=float(att_row["presence_seconds"]),
                        presence_pct=float(att_row["presence_pct"]), first_seen=att_row["first_seen"],
                        last_seen=att_row["last_seen"], times_stepped_out=int(max(0, att_row["segments"] - 1))),
    )
    if w.empty or w.engagement.isna().all():
        base["engagement"] = None
        base["notes"] = ["Not enough presence to build an engagement profile for this session."]
        return base

    e = w.engagement.to_numpy(dtype=float)
    mean = float(np.nanmean(e))
    std = float(np.nanstd(e))
    consistency = float(1 - min(1.0, std / 0.35))
    dist = w.label.value_counts(normalize=True).to_dict()
    focus = [(w.t_start[a], w.t_end[b]) for a, b in _runs((w.engagement >= cfg.high_thr).to_numpy(), 1)]
    distract = [(w.t_start[a], w.t_end[b]) for a, b in _runs((w.engagement < cfg.low_thr).to_numpy(), 1)]

    notes = []
    notes.append(f"You were present for {mmss(att_row['presence_seconds'])} "
                 f"({att_row['presence_pct']:.0f}% of the session).")
    if focus:
        a, b = max(focus, key=lambda p: p[1] - p[0])
        notes.append(f"Your longest high-focus stretch was {mmss(a)}-{mmss(b)}. "
                     "Notice what helped you concentrate then.")
    if distract:
        a, b = max(distract, key=lambda p: p[1] - p[0])
        notes.append(f"Focus dipped around {mmss(a)}-{mmss(b)}. A short stretch or note-taking "
                     "break at such moments may help.")
    notes.append("Your engagement was steady through the session." if consistency >= 0.7 else
                 "Your engagement varied noticeably across the session." if consistency < 0.4 else
                 "Your engagement showed some natural ups and downs.")
    if len(e) >= 4:
        h = len(e) // 2
        d = np.nanmean(e[h:]) - np.nanmean(e[:h])
        if d > 0.1:
            notes.append("You grew more engaged as the session went on.")
        elif d < -0.1:
            notes.append("Your engagement tapered towards the end - this is common in longer sessions.")
    base["engagement"] = dict(
        mean=round(mean, 3), label=("High" if mean >= cfg.high_thr else "Low" if mean < cfg.low_thr else "Medium"),
        consistency=round(consistency, 3),
        share_high=round(dist.get("High", 0), 3), share_medium=round(dist.get("Medium", 0), 3),
        share_low=round(dist.get("Low", 0), 3),
        focus_periods=[f"{mmss(a)}-{mmss(b)}" for a, b in focus],
        distraction_periods=[f"{mmss(a)}-{mmss(b)}" for a, b in distract],
        timeline=[dict(t_start=float(r.t_start), t_end=float(r.t_end),
                       engagement=None if np.isnan(r.engagement) else float(r.engagement), label=r.label)
                  for r in w.itertuples()],
    )
    base["notes"] = notes
    return base
