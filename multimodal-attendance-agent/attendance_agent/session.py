"""End-to-end session runner (Activities 6.1): vision -> audio -> fusion -> agent -> files.

Everything is written under ``sessions/<session_id>/`` on the local disk. No network
access is used anywhere in the pipeline.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from . import __version__
from .agent import EngagementAgent
from .analytics import student_report
from .attendance import compute_attendance
from .audio.analysis import analyse_audio, waveform_envelope
from .audio.extract import load_audio
from .config import Config
from .fusion import fuse
from .vision.features import window_vision_features
from .vision.pipeline import process_video


@dataclass
class SessionResult:
    session_id: str
    out_dir: Path
    summary: dict
    attendance: pd.DataFrame
    events: pd.DataFrame
    timeline: pd.DataFrame
    student_windows: pd.DataFrame
    audio: pd.DataFrame
    agent_report: dict
    students: dict


def analyse(det, duration, audio_y, audio_sr, cfg, session_start=None):
    """Pure analysis stage (no I/O) - also what the unit tests exercise."""
    att, events, masks, id_map = compute_attendance(det, duration, cfg, session_start)
    audio_w = analyse_audio(audio_y, audio_sr, duration, cfg)
    vision_w = window_vision_features(det, masks, id_map, duration, cfg) if id_map else pd.DataFrame()
    if vision_w.empty:
        student_w, timeline = pd.DataFrame(), pd.DataFrame()
    else:
        student_w, timeline = fuse(vision_w, audio_w, att, cfg)
    report = EngagementAgent(cfg).run(timeline, att, duration)
    students = ({r["participant"]: student_report(r["participant"], r, student_w, cfg)
                 for _, r in att.iterrows() if r["marked_present"]} if not att.empty else {})
    return att, events, timeline, student_w, audio_w, report, students


def run_session(video: Union[str, int, Path], cfg: Optional[Config] = None,
                audio: Optional[Union[str, Path]] = None, out_root: Union[str, Path] = "sessions",
                session_id: Optional[str] = None, annotate: bool = False,
                max_seconds: Optional[float] = None, live_audio: bool = False,
                show: bool = False, progress: Optional[Callable[[float], None]] = None) -> SessionResult:
    cfg = cfg or Config()
    started = datetime.now().replace(microsecond=0)
    session_id = session_id or started.strftime("session_%Y%m%d_%H%M%S")
    out = Path(out_root) / session_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- audio source ------------------------------------------------------
    recorder = None
    if live_audio and isinstance(video, int):
        from .audio.extract import LiveRecorder
        recorder = LiveRecorder(cfg.audio_sr)
        recorder.start()

    # --- Activity 2.1: vision -------------------------------------------------
    vr = process_video(video, cfg, annotate_path=str(out / "annotated.mp4") if annotate else None,
                       max_seconds=max_seconds, progress=progress, show=show)

    # --- Activity 3.1: audio -------------------------------------------------
    if recorder is not None:
        y, sr = recorder.stop(), cfg.audio_sr
    elif audio:
        y, sr = load_audio(audio, cfg.audio_sr)
    elif isinstance(video, (str, Path)):
        y, sr = load_audio(video, cfg.audio_sr)       # audio track of the same video
    else:
        y, sr = np.zeros(0, dtype=np.float32), cfg.audio_sr
    if max_seconds is not None and len(y):
        y = y[: int(max_seconds * sr)]

    att, events, timeline, student_w, audio_w, report, students = analyse(
        vr.detections, vr.duration, y, sr, cfg, started)

    # --- persist (local only) ---------------------------------------------------
    att.to_csv(out / "attendance.csv", index=False)
    events.to_csv(out / "attendance_events.csv", index=False)
    vr.detections.drop(columns=["score"]).round(3).to_csv(out / "tracks.csv", index=False)  # boxes only, no images
    audio_w.to_csv(out / "audio_metrics.csv", index=False)
    waveform_envelope(y, sr).to_csv(out / "audio_envelope.csv", index=False)
    student_w.to_csv(out / "student_engagement.csv", index=False)
    timeline.to_csv(out / "engagement_timeline.csv", index=False)
    (out / "suggestions.json").write_text(json.dumps(report.to_dict(), indent=2))
    sdir = out / "student_reports"
    sdir.mkdir(exist_ok=True)
    for pid, rep in students.items():
        (sdir / f"{pid}.json").write_text(json.dumps(rep, indent=2, default=float))
    cfg.save(out / "config_used.json")

    summary = dict(
        session_id=session_id, started=started.isoformat(), version=__version__,
        source=str(video), duration_seconds=round(vr.duration, 1), backend=vr.backend,
        audio_available=bool(len(y)), frames_sampled=vr.frames_sampled,
        participants_detected=int(len(att)), participants_present=int(att.marked_present.sum()) if len(att) else 0,
        processing_seconds=round(time.time() - t0, 1), engagement=report.summary,
        privacy=dict(offline=True, face_images_stored=False, audio_content_stored=False,
                     note="Only bounding boxes, scores and aggregated audio levels are stored."),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    from .reports import write_html_report
    write_html_report(out, summary, att, timeline, audio_w, report, students)
    return SessionResult(session_id, out, summary, att, events, timeline, student_w, audio_w,
                         report.to_dict(), students)
