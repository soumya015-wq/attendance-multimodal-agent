"""Command-line interface.

    python -m attendance_agent.main --video class.mp4
    python -m attendance_agent.main --video class.mp4 --audio mic.wav --annotate
    python -m attendance_agent.main --webcam 0 --duration 120 --live-audio
"""
from __future__ import annotations

import argparse
import sys

from .config import Config
from .session import run_session


def build_parser():
    p = argparse.ArgumentParser(description="Multimodal Attendance & Engagement Agent (offline)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to a recorded classroom video")
    src.add_argument("--webcam", type=int, help="webcam index (e.g. 0) for live capture")
    p.add_argument("--audio", help="optional separate audio file (default: audio track of the video)")
    p.add_argument("--live-audio", action="store_true", help="record the microphone during webcam capture")
    p.add_argument("--duration", type=float, help="stop after N seconds (useful for webcam / quick tests)")
    p.add_argument("--out", default="sessions", help="output root folder (default: sessions)")
    p.add_argument("--session-id", help="custom session folder name")
    p.add_argument("--config", help="JSON file overriding default thresholds")
    p.add_argument("--backend", choices=["auto", "mediapipe", "haar"], help="face detector backend")
    p.add_argument("--annotate", action="store_true", help="also write annotated.mp4 (faces blurred)")
    p.add_argument("--show", action="store_true", help="live preview window (press q to stop)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    if args.backend:
        cfg.detector_backend = args.backend
    source = args.webcam if args.webcam is not None else args.video
    last = {"p": -1}

    def prog(f):
        pct = int(f * 100)
        if pct != last["p"] and pct % 5 == 0:
            last["p"] = pct
            print(f"\rprocessing {pct:3d}%", end="", file=sys.stderr, flush=True)

    res = run_session(source, cfg, audio=args.audio, out_root=args.out, session_id=args.session_id,
                      annotate=args.annotate, max_seconds=args.duration,
                      live_audio=args.live_audio, show=args.show,
                      progress=prog)
    print(file=sys.stderr)
    s = res.summary
    e = s["engagement"]
    print(f"Session      : {s['session_id']}  ({s['duration_seconds']} s, backend={s['backend']})")
    print(f"Attendance   : {s['participants_present']} present / {s['participants_detected']} detected")
    if "mean_engagement" in e:
        print(f"Engagement   : {e['mean_engagement']} ({e['overall_label']}), trend {e['trend']}")
    print(f"Suggestions  : {len(res.agent_report['suggestions'])}")
    print(f"Outputs in   : {res.out_dir}  (open report.html, or run: streamlit run dashboard.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
