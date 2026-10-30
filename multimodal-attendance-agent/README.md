# Multimodal Attendance & Engagement Agent

Offline, privacy-preserving classroom intelligence: **computer vision + audio signal processing + a rule-based reasoning agent**.
It tracks attendance from sustained presence, fuses visual and audio cues into an engagement timeline, produces
non-prescriptive teaching suggestions, and gives each student a private reflective summary.

## Quick start
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt                         # Python 3.9-3.12 recommended

python scripts/make_demo.py                             # builds data/demo_class.mp4 (150 s synthetic class)
python -m attendance_agent.main --video data/demo_class.mp4 --annotate
streamlit run dashboard.py                              # interactive dashboard
pytest -q                                               # 16 tests
```
Other inputs: `--video class.mp4 --audio mic.wav`, `--webcam 0 --duration 120 --live-audio`, `--backend haar|mediapipe`,
`--config my_thresholds.json` (any field of `attendance_agent/config.py`).
Results land in `sessions/<id>/` (open `report.html`).

## Pipeline -> project activities
| Activity | Where |
|---|---|
| 1.1 / 1.2 requirements, virtualenv | `requirements.txt`, Quick start |
| 2.1 face detection & tracking | `vision/detector.py` (MediaPipe + OpenCV Haar), `vision/tracker.py` (IoU/Hungarian + HSV re-ID), `vision/pipeline.py` |
| 2.2 presence-based attendance | `attendance.py` - gap bridging, cumulative presence threshold, timestamped events |
| 3.1 audio extraction | `audio/extract.py` - ffmpeg decode / live mic, mono 16 kHz |
| 3.2 noise & speech analysis | `audio/analysis.py` - RMS dB, spectral flatness, ZCR, adaptive + absolute noise tests |
| 4.1 multimodal agent | `agent.py` (perceive -> reason -> recommend) |
| 4.2 signal fusion | `vision/features.py`, `fusion.py` - windows aligned by timestamp |
| 4.3 engagement scoring | `fusion.py` - `visual = attention x behaviour`; `score = 0.7 visual + 0.3 audio - noise penalty` |
| 5.1 teaching suggestions | `agent.py` - 9 rules + reasoning trace |
| 5.2 student analytics | `analytics.py` - individual only, no ranking/comparison |
| 6.1 run core app | `main.py`, `session.py` |
| 6.2 dashboard | `dashboard.py` (Streamlit) |
| 6.3 validation & edge cases | `tests/test_core.py` |

## How the scoring works (transparent & adjustable)
* **Attendance**: present once cumulative presence >= `min(min_presence_seconds, min_presence_ratio x session)`; gaps <= 3 s are bridged.
* **Attention**: share of present time with head pose inside yaw/pitch limits (MediaPipe) or a frontal face (Haar).
* **Behaviour**: not fidgeting (movement) and not inactive (still for >= `inactivity_seconds`).
* **Audio context**: silence is neutral (0.5), speech raises it to 1.0, loud non-speech noise subtracts a penalty. Audio is one room-level mic, so it is context, not per-student evidence.
* Labels: High >= 0.70, Low < 0.45. Everything is in `Config`.

## Privacy
Runs fully offline. Face images are never saved; only boxes, scores and aggregated audio levels are stored. Audio is never transcribed.
`annotated.mp4` (optional) blurs faces. Student reports have no grades, ranks or peer comparison.

## Sample output
`docs/sample_outputs/` and `sessions/demo_run/` hold a completed run on the demo class.

## Limitations (please read)
* Thresholds are sensible defaults, **not validated on real classrooms** - tune them and check results against ground truth.
* The demo is synthetic (one stock photo + generated audio); it proves the logic, not real-world accuracy.
* Haar fallback only distinguishes frontal vs. profile; precise head pose needs MediaPipe (`mediapipe==0.10.14`; newer versions removed the API used here and the code falls back to Haar automatically). Large head-turn angles were not tested on real footage.
* Identity is a track ID (P1, P2, ...) - there is no face recognition; a person who leaves for a long time may get a new ID.
* Engagement from video/audio is a proxy. Use it as decision support, never for grading or discipline, and get consent per your institution's policy.
