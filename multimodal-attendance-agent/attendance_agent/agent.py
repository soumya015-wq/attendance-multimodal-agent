"""Teaching-suggestion agent (Activities 4.1 & 5.1).

A transparent *perceive -> reason -> recommend* loop. The agent reads the fused
engagement timeline, evaluates a library of interpretable rules, and returns
context-aware, **non-prescriptive** suggestions. Every rule that was evaluated - fired
or not - is recorded in a reasoning trace so instructors can see *why* a suggestion
appeared. The agent never enforces decisions; it supports them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


@dataclass
class Suggestion:
    rule_id: str
    priority: str            # "high" | "medium" | "info"
    category: str            # pacing | interaction | classroom-management | attendance | positive | data
    title: str
    message: str
    time_range: str = ""
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentReport:
    summary: dict
    suggestions: List[Suggestion]
    trace: List[dict]

    def to_dict(self):
        return dict(summary=self.summary, suggestions=[s.to_dict() for s in self.suggestions],
                    trace=self.trace)


def _runs(flags: np.ndarray, min_len: int):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


class EngagementAgent:
    def __init__(self, cfg):
        self.cfg = cfg

    # ---- perceive ------------------------------------------------------
    def perceive(self, timeline: pd.DataFrame, attendance: pd.DataFrame, duration: float) -> dict:
        return dict(tl=timeline.reset_index(drop=True), att=attendance, duration=duration)

    # ---- reason + recommend ---------------------------------------------
    def run(self, timeline, attendance, duration) -> AgentReport:
        obs = self.perceive(timeline, attendance, duration)
        sugg: List[Suggestion] = []
        trace: List[dict] = []
        tl = obs["tl"]

        if tl.empty:
            s = Suggestion("R0", "info", "data", "No participants confirmed present",
                           "No one met the presence threshold, so engagement could not be analysed. "
                           "Check the camera angle, lighting and the presence threshold settings.")
            return AgentReport(dict(note="no data"), [s], [dict(rule="R0", fired=True,
                               reason="empty timeline")])

        rules = [self.r_audio_missing, self.r_sustained_low, self.r_sharp_drop, self.r_fatigue,
                 self.r_noise, self.r_silence, self.r_peak, self.r_late_early, self.r_all_good]
        for rule in rules:
            out, why = rule(obs)
            trace.append(dict(rule=rule.__name__[2:], fired=bool(out), reason=why))
            sugg.extend(out)

        order = {"high": 0, "medium": 1, "info": 2}
        sugg.sort(key=lambda s: (order[s.priority], s.time_range))
        return AgentReport(self.summarise(tl, duration), sugg, trace)

    # ---- summary ----------------------------------------------------------
    def summarise(self, tl: pd.DataFrame, duration: float) -> dict:
        e = tl.engagement_smooth.to_numpy(dtype=float)
        mean = float(np.nanmean(e))
        slope = float(np.polyfit(np.arange(len(e)), e, 1)[0]) if len(e) >= 2 else 0.0
        counts = tl.label.value_counts(normalize=True).to_dict()
        best = tl.loc[tl.engagement_smooth.idxmax()]
        worst = tl.loc[tl.engagement_smooth.idxmin()]
        return dict(
            mean_engagement=round(mean, 3),
            overall_label=("High" if mean >= self.cfg.high_thr else "Low" if mean < self.cfg.low_thr else "Medium"),
            trend_per_window=round(slope, 4),
            trend=("rising" if slope > 0.01 else "falling" if slope < -0.01 else "steady"),
            share_high=round(counts.get("High", 0), 3), share_medium=round(counts.get("Medium", 0), 3),
            share_low=round(counts.get("Low", 0), 3),
            peak_window=f"{mmss(best.t_start)}-{mmss(best.t_end)}", peak_value=round(float(best.engagement_smooth), 3),
            lowest_window=f"{mmss(worst.t_start)}-{mmss(worst.t_end)}", lowest_value=round(float(worst.engagement_smooth), 3),
            session_length=mmss(duration),
        )

    # ---- rules ------------------------------------------------------------
    def r_audio_missing(self, o):
        tl = o["tl"]
        if (tl.audio_state == "no_audio").all():
            return [Suggestion("audio_missing", "info", "data", "Audio unavailable - vision-only analysis",
                               "No usable audio track was found, so engagement uses visual signals only. "
                               "Interaction and noise insights are limited; add a microphone or audio file "
                               "for fuller session dynamics.")], "no audio in any window"
        return [], "audio present"

    def r_sustained_low(self, o):
        tl, cfg = o["tl"], self.cfg
        runs = _runs((tl.engagement_smooth < cfg.low_thr).to_numpy(), cfg.sustained_low_windows)
        out = []
        for a, b in runs:
            seg = tl.iloc[a:b + 1]
            out.append(Suggestion(
                "sustained_low", "high", "interaction", "Extended low-engagement period",
                f"Engagement stayed low from {mmss(seg.t_start.iloc[0])} to {mmss(seg.t_end.iloc[-1])} "
                f"(average {seg.engagement_smooth.mean():.2f}). Consider introducing an interactive activity - "
                "a quick poll, think-pair-share or a short worked problem - or revisiting the material "
                "from a different angle.",
                f"{mmss(seg.t_start.iloc[0])}-{mmss(seg.t_end.iloc[-1])}",
                dict(avg_engagement=round(float(seg.engagement_smooth.mean()), 3),
                     avg_attention=round(float(seg.attention.mean()), 3), windows=len(seg))))
        return out, (f"{len(runs)} run(s) of >= {cfg.sustained_low_windows} low windows" if runs
                     else "no sustained low runs")

    def r_sharp_drop(self, o):
        tl, cfg = o["tl"], self.cfg
        d = tl.engagement_smooth.diff().to_numpy()
        out = []
        for i in np.where(d <= -cfg.drop_delta)[0]:
            row = tl.iloc[i]
            out.append(Suggestion(
                "sharp_drop", "medium", "pacing", "Sudden attention drop",
                f"Engagement fell by {abs(d[i]):.2f} around {mmss(row.t_start)}. This often follows a long "
                "stretch of uninterrupted explanation or a dense slide. A short recap, a check-for-"
                "understanding question or a brief pause may help refocus the room.",
                f"{mmss(row.t_start)}-{mmss(row.t_end)}", dict(drop=round(float(abs(d[i])), 3))))
        return out, f"{len(out)} drop(s) >= {cfg.drop_delta}" if out else "no sharp drops"

    def r_fatigue(self, o):
        tl, cfg = o["tl"], self.cfg
        if len(tl) < 4:
            return [], "session too short for trend"
        h = len(tl) // 2
        first, second = tl.engagement.iloc[:h].mean(), tl.engagement.iloc[h:].mean()
        if first - second >= cfg.fatigue_delta:
            mid = tl.t_start.iloc[h]
            return [Suggestion(
                "fatigue", "medium", "pacing", "Engagement declined over the session",
                f"Average engagement was {first:.2f} in the first half and {second:.2f} in the second. "
                f"Consider a short break or energiser near {mmss(mid)}, or moving demanding content earlier "
                "and lighter, practice-based work later.", "second half",
                dict(first_half=round(float(first), 3), second_half=round(float(second), 3)))], "declining"
        return [], f"first={first:.2f} second={second:.2f} (change below {cfg.fatigue_delta})"

    def r_noise(self, o):
        tl, cfg = o["tl"], self.cfg
        noisy = (tl.audio_state == "noisy").to_numpy()
        share = float(noisy.mean())
        out = []
        if share >= cfg.noisy_session_ratio:
            out.append(Suggestion(
                "noise_session", "medium", "classroom-management", "Noisy classroom conditions",
                f"{share:.0%} of the session had high background noise. Consider a quiet-signal routine, "
                "clearer norms for group work, or checking for external noise sources.", "whole session",
                dict(noisy_share=round(share, 3))))
        for a, b in _runs(noisy, 2):
            seg = tl.iloc[a:b + 1]
            out.append(Suggestion(
                "noise_interval", "info", "classroom-management", "Noisy interval",
                f"Noise was elevated between {mmss(seg.t_start.iloc[0])} and {mmss(seg.t_end.iloc[-1])} "
                f"(engagement averaged {seg.engagement_smooth.mean():.2f}). A classroom-management cue "
                "such as a reset signal might help here.",
                f"{mmss(seg.t_start.iloc[0])}-{mmss(seg.t_end.iloc[-1])}"))
        return out, f"noisy share={share:.0%}"

    def r_silence(self, o):
        tl, cfg = o["tl"], self.cfg
        if (tl.audio_state == "no_audio").all():
            return [], "skipped (no audio)"
        quiet = (tl.speech_ratio.fillna(0) < 0.05).to_numpy()
        out = []
        for a, b in _runs(quiet, cfg.long_silence_windows):
            seg = tl.iloc[a:b + 1]
            out.append(Suggestion(
                "long_silence", "info", "interaction", "Long stretch with little spoken interaction",
                f"There was almost no speech from {mmss(seg.t_start.iloc[0])} to {mmss(seg.t_end.iloc[-1])}. "
                "If that was not intentional (e.g. silent work), questions, discussion prompts or pair "
                "talk could re-open participation.",
                f"{mmss(seg.t_start.iloc[0])}-{mmss(seg.t_end.iloc[-1])}"))
        return out, f"{len(out)} long silent run(s)"

    def r_peak(self, o):
        tl, cfg = o["tl"], self.cfg
        runs = _runs((tl.engagement_smooth >= cfg.high_thr).to_numpy(), 2)
        out = []
        for a, b in runs:
            seg = tl.iloc[a:b + 1]
            out.append(Suggestion(
                "peak", "info", "positive", "Effective teaching segment",
                f"Engagement peaked between {mmss(seg.t_start.iloc[0])} and {mmss(seg.t_end.iloc[-1])} "
                f"(average {seg.engagement_smooth.mean():.2f}). Reflect on what you were doing then - the "
                "format, pace or interaction - and consider reusing it.",
                f"{mmss(seg.t_start.iloc[0])}-{mmss(seg.t_end.iloc[-1])}",
                dict(avg_engagement=round(float(seg.engagement_smooth.mean()), 3))))
        return out, f"{len(out)} peak segment(s)"

    def r_late_early(self, o):
        att, dur = o["att"], o["duration"]
        p = att[att.marked_present]
        if p.empty:
            return [], "no present participants"
        late_s = max(5.0, self.cfg.late_arrival_fraction * dur)
        late = p[p.first_seen_s > late_s]
        early = p[p.last_seen_s < 0.85 * dur]
        out = []
        if len(late):
            out.append(Suggestion(
                "late_arrivals", "info", "attendance", "Late arrivals",
                f"{len(late)} participant(s) joined after the first {mmss(late_s)}. A brief recap of "
                "key points once they settle can help them catch up.", "start of session",
                dict(count=int(len(late)))))
        if len(early):
            out.append(Suggestion(
                "early_exits", "info", "attendance", "Early departures",
                f"{len(early)} participant(s) were no longer visible for the final part of the session. "
                "Consider sharing a closing summary or materials afterwards.", "end of session",
                dict(count=int(len(early)))))
        return out, f"late={len(late)} early={len(early)}"

    def r_all_good(self, o):
        tl, cfg = o["tl"], self.cfg
        if (tl.engagement_smooth >= cfg.high_thr).mean() >= 0.7:
            return [Suggestion("steady_high", "info", "positive", "Strong engagement throughout",
                               "Most of the session showed high engagement. The current pacing and mix of "
                               "activities appear to be working well.", "whole session")], "mostly high"
        return [], "not mostly high"
