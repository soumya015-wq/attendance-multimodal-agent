"""Self-contained offline HTML report (Activity 6.2/6.3 deliverable).

Charts are rendered with matplotlib and embedded as base64 PNG, so the file opens
anywhere without network access.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agent import mmss

PALETTE = dict(high="#2e9e5b", mid="#e0a526", low="#d6453d", blue="#2b6cb0", grey="#8a94a6")


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def engagement_chart(tl: pd.DataFrame, cfg_low=0.45, cfg_high=0.70):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = (tl.t_start + tl.t_end) / 2 / 60
    ax.axhspan(cfg_high, 1, color=PALETTE["high"], alpha=.08)
    ax.axhspan(0, cfg_low, color=PALETTE["low"], alpha=.08)
    ax.plot(x, tl.engagement, color=PALETTE["grey"], lw=1, marker="o", ms=3, label="per window")
    ax.plot(x, tl.engagement_smooth, color=PALETTE["blue"], lw=2.4, label="smoothed")
    if "speech_ratio" in tl and tl.speech_ratio.notna().any():
        ax.fill_between(x, 0, tl.speech_ratio.fillna(0), color="#7c5cbf", alpha=.18, label="speech activity")
    ax.set_ylim(0, 1)
    ax.set_xlabel("minutes")
    ax.set_ylabel("engagement")
    ax.set_title("Classroom engagement over time")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    ax.grid(alpha=.25)
    return fig


def attendance_chart(att: pd.DataFrame, duration: float):
    fig, ax = plt.subplots(figsize=(9, max(1.8, 0.45 * len(att) + 1)))
    for i, r in enumerate(att.itertuples()):
        a = r.first_seen_s or 0
        b = r.last_seen_s or 0
        col = PALETTE["high"] if r.marked_present else PALETTE["grey"]
        ax.barh(i, max(b - a, 0.5) / 60, left=a / 60, color=col, height=.55)
    ax.set_yticks(range(len(att)))
    ax.set_yticklabels(att.participant)
    ax.invert_yaxis()
    ax.set_xlim(0, duration / 60)
    ax.set_xlabel("minutes")
    ax.set_title("Presence span (green = marked present)")
    ax.grid(axis="x", alpha=.25)
    return fig


_CSS = """body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6fa;color:#1d2433}
.wrap{max-width:980px;margin:auto;padding:28px}h1{margin:0 0 4px}h2{margin-top:34px;border-bottom:2px solid #dbe2ef;padding-bottom:6px}
.card{background:#fff;border-radius:12px;padding:16px 20px;margin:12px 0;box-shadow:0 1px 4px #0001}
.kpis{display:flex;gap:12px;flex-wrap:wrap}.kpi{flex:1;min-width:150px;background:#fff;border-radius:12px;padding:14px;box-shadow:0 1px 4px #0001}
.kpi b{display:block;font-size:26px}.kpi span{color:#5b6477;font-size:13px}
table{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden}th,td{padding:8px 10px;border-bottom:1px solid #edf0f6;text-align:left;font-size:14px}
th{background:#eaf0fb}.pill{padding:2px 9px;border-radius:99px;font-size:12px;color:#fff}.high{background:#d6453d}.medium{background:#e0a526}.info{background:#2b6cb0}
img{max-width:100%}.small{color:#5b6477;font-size:13px}"""


def write_html_report(out_dir: Path, summary: dict, att, timeline, audio_w, report, students) -> Path:
    e = summary["engagement"]
    h = [f"<html><head><meta charset='utf-8'><title>{html.escape(summary['session_id'])}</title>"
         f"<style>{_CSS}</style></head><body><div class='wrap'>",
         "<h1>Multimodal Attendance &amp; Engagement Report</h1>",
         f"<div class='small'>{html.escape(summary['session_id'])} &middot; processed fully offline &middot; "
         f"backend: {summary['backend']}</div>"]
    present = summary["participants_present"]
    h.append("<div class='kpis'>"
             f"<div class='kpi'><b>{present}/{summary['participants_detected']}</b><span>marked present / detected</span></div>"
             f"<div class='kpi'><b>{e.get('session_length', '-')}</b><span>session length</span></div>"
             f"<div class='kpi'><b>{e.get('mean_engagement', '-')}</b><span>mean engagement ({e.get('overall_label', '-')})</span></div>"
             f"<div class='kpi'><b>{e.get('trend', '-')}</b><span>engagement trend</span></div></div>")

    h.append("<h2>Attendance</h2>")
    if len(att):
        h.append(f"<div class='card'><img src='data:image/png;base64,{_png(attendance_chart(att, summary['duration_seconds']))}'></div>")
        cols = ["participant", "status", "presence_seconds", "presence_pct", "first_seen", "last_seen", "segments"]
        h.append(att[cols].to_html(index=False, border=0, escape=True))
    else:
        h.append("<div class='card'>No faces were tracked.</div>")

    h.append("<h2>Engagement</h2>")
    if len(timeline):
        h.append(f"<div class='card'><img src='data:image/png;base64,{_png(engagement_chart(timeline))}'></div>")
        h.append(f"<p class='small'>Peak: {e['peak_window']} ({e['peak_value']}) &middot; Lowest: "
                 f"{e['lowest_window']} ({e['lowest_value']}) &middot; High {e['share_high']:.0%} / "
                 f"Medium {e['share_medium']:.0%} / Low {e['share_low']:.0%} of windows</p>")
    h.append("<h2>Teaching suggestions <span class='small'>(decision support, not instructions)</span></h2>")
    for s in report.suggestions:
        h.append(f"<div class='card'><span class='pill {s.priority}'>{s.priority}</span> "
                 f"<b>{html.escape(s.title)}</b> <span class='small'>{html.escape(s.time_range)}</span>"
                 f"<p>{html.escape(s.message)}</p></div>")
    h.append("<h2>Why these suggestions? (agent reasoning trace)</h2><table><tr><th>rule</th><th>fired</th><th>observation</th></tr>")
    for t in report.trace:
        h.append(f"<tr><td>{t['rule']}</td><td>{'yes' if t['fired'] else 'no'}</td><td>{html.escape(str(t['reason']))}</td></tr>")
    h.append("</table>")
    h.append("<h2>Privacy</h2><div class='card small'>All processing ran locally. No face images, "
             "audio recordings or transcripts were stored - only bounding boxes, scores and aggregated audio levels. "
             "Student session analytics are individual and reflective; there is no ranking or peer comparison.</div>")
    h.append("</div></body></html>")
    p = Path(out_dir) / "report.html"
    p.write_text("\n".join(h), encoding="utf-8")
    return p
