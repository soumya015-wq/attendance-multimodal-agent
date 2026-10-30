"""Streamlit dashboard (Activity 6.2).   Run:  streamlit run dashboard.py

Everything is local: it reads session folders under ./sessions and can run the
pipeline on a video file path or an uploaded file. No network calls.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from attendance_agent.config import Config
from attendance_agent.session import run_session

ROOT = Path("sessions")
st.set_page_config(page_title="Attendance & Engagement Agent", layout="wide")
st.title("Multimodal Attendance & Engagement Agent")
st.caption("Offline, privacy-preserving classroom analytics - decision support, not enforcement.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Run a new session")
    up = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv"])
    path = st.text_input("...or path to a video file", "")
    backend = st.selectbox("Face detector", ["auto", "mediapipe", "haar"])
    min_presence = st.slider("Presence needed for attendance (s)", 5, 300, 20)
    window = st.slider("Engagement window (s)", 5, 60, 10)
    annotate = st.checkbox("Write annotated video (faces blurred)", False)
    if st.button("Analyse", type="primary"):
        src = path.strip()
        if up is not None:
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getbuffer())
            src = str(tmp)
        if not src or not Path(src).exists():
            st.error("Provide an uploaded video or a valid file path.")
        else:
            cfg = Config(detector_backend=backend, min_presence_seconds=float(min_presence),
                         window_seconds=float(window))
            bar = st.progress(0.0, text="Analysing ...")
            res = run_session(src, cfg, out_root=ROOT, annotate=annotate,
                              progress=lambda f: bar.progress(min(f, 1.0), text="Analysing ..."))
            bar.empty()
            st.session_state["selected"] = res.session_id
            st.success(f"Done: {res.session_id}")

    sessions = sorted([p.name for p in ROOT.glob("*") if (p / "summary.json").exists()], reverse=True)
    st.header("Open session")
    if not sessions:
        st.info("No sessions yet. Run `python scripts/make_demo.py` then analyse data/demo_class.mp4.")
        st.stop()
    default = sessions.index(st.session_state["selected"]) if st.session_state.get("selected") in sessions else 0
    sid = st.selectbox("Session", sessions, index=default)

d = ROOT / sid
summary = json.loads((d / "summary.json").read_text())
agent = json.loads((d / "suggestions.json").read_text())
att = pd.read_csv(d / "attendance.csv")
events = pd.read_csv(d / "attendance_events.csv") if (d / "attendance_events.csv").stat().st_size > 5 else pd.DataFrame()
tl = pd.read_csv(d / "engagement_timeline.csv") if (d / "engagement_timeline.csv").stat().st_size > 5 else pd.DataFrame()
audio = pd.read_csv(d / "audio_metrics.csv")
e = summary["engagement"]

c = st.columns(4)
c[0].metric("Present / detected", f"{summary['participants_present']} / {summary['participants_detected']}")
c[1].metric("Session length", e.get("session_length", "-"))
c[2].metric("Mean engagement", e.get("mean_engagement", "-"), e.get("overall_label"))
c[3].metric("Trend", e.get("trend", "-"))

tabs = st.tabs(["Attendance", "Engagement", "Audio", "Teaching suggestions", "Student analytics", "Privacy & files"])

with tabs[0]:
    st.dataframe(att.drop(columns=["track_id", "first_seen_s", "last_seen_s"]), use_container_width=True, hide_index=True)
    if len(events):
        st.subheader("Timestamped log")
        st.dataframe(events, use_container_width=True, hide_index=True)
    st.download_button("Download attendance.csv", att.to_csv(index=False), "attendance.csv")

with tabs[1]:
    if tl.empty:
        st.warning("No engagement data (nobody met the presence threshold).")
    else:
        tl["minute"] = (tl.t_start + tl.t_end) / 120
        st.line_chart(tl.set_index("minute")[["engagement", "engagement_smooth"]])
        st.bar_chart(tl.set_index("minute")[["attention", "inactivity"]])
        st.dataframe(tl.drop(columns=["minute"]).round(3), use_container_width=True, hide_index=True)

with tabs[2]:
    audio["minute"] = (audio.t_start + audio.t_end) / 120
    st.line_chart(audio.set_index("minute")[["speech_ratio", "noise_ratio"]])
    st.dataframe(audio.drop(columns=["minute"]), use_container_width=True, hide_index=True)

with tabs[3]:
    st.caption("Suggestions are assistive. Each is grounded in the data trend shown in its evidence.")
    for s in agent["suggestions"]:
        icon = {"high": "🔴", "medium": "🟠", "info": "🔵"}[s["priority"]]
        with st.expander(f"{icon} {s['title']}  ({s['time_range']})", expanded=s["priority"] == "high"):
            st.write(s["message"])
            if s["evidence"]:
                st.json(s["evidence"])
    with st.expander("Why these suggestions? (reasoning trace)"):
        st.dataframe(pd.DataFrame(agent["trace"]), use_container_width=True, hide_index=True)

with tabs[4]:
    files = sorted((d / "student_reports").glob("*.json"))
    if not files:
        st.info("No participant met the attendance threshold.")
    else:
        pid = st.selectbox("Participant", [f.stem for f in files])
        rep = json.loads((d / "student_reports" / f"{pid}.json").read_text())
        st.subheader(f"{pid} - personal session summary")
        for n in rep["notes"]:
            st.write("• " + n)
        if rep.get("engagement"):
            eg = rep["engagement"]
            a, b, c2 = st.columns(3)
            a.metric("Your engagement", eg["mean"], eg["label"])
            b.metric("Consistency", eg["consistency"])
            c2.metric("Time present", f"{rep['attendance']['presence_seconds']:.0f}s")
            t = pd.DataFrame(eg["timeline"]).dropna()
            t["minute"] = (t.t_start + t.t_end) / 120
            st.line_chart(t.set_index("minute")["engagement"])
        st.caption("Reflective only - no grades, rankings or comparisons with classmates.")

with tabs[5]:
    st.json(summary["privacy"])
    st.write("Local files:", sorted(p.name for p in d.iterdir()))
    rep_html = d / "report.html"
    if rep_html.exists():
        st.download_button("Download full HTML report", rep_html.read_text(encoding="utf-8"), "report.html", "text/html")
