"""Central, adjustable configuration. Every threshold used by the system lives here
so scoring stays transparent and can be tuned per classroom (Activity 4.3)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Config:
    # ---------------- Vision sampling / detection ----------------
    sample_fps: float = 5.0            # frames analysed per second of video
    detector_backend: str = "auto"     # "auto" | "mediapipe" | "haar"
    min_face_px: int = 40              # ignore faces smaller than this (px)
    detect_scale: float = 1.0          # downscale frames before detection (speed)
    max_faces: int = 60

    # ---------------- Tracking ----------------
    iou_threshold: float = 0.20        # minimum affinity to continue a track
    max_missing_seconds: float = 1.5   # keep a lost track alive this long
    reid_window_seconds: float = 45.0  # try to re-identify a returning face within this window
    reid_hist_threshold: float = 0.75  # HSV-histogram correlation needed for re-identification
    min_track_detections: int = 3      # discard tracks seen fewer times (false positives)

    # ---------------- Attendance (Activity 2.2) ----------------
    merge_gap_seconds: float = 3.0     # gaps shorter than this still count as "present"
    min_presence_seconds: float = 20.0 # cumulative presence needed to be marked present
    min_presence_ratio: float = 0.25   # ...or this share of the session (the smaller bound wins)

    # ---------------- Head pose / attention ----------------
    yaw_limit_deg: float = 30.0        # |yaw| above this => looking away
    pitch_limit_deg: float = 35.0      # |pitch| above this => looking far up/down

    # ---------------- Movement / inactivity ----------------
    still_motion_thr: float = 0.12     # smoothed motion (face-widths / s) below this = still
    fidget_motion_thr: float = 1.00    # above this = excessive movement
    inactivity_seconds: float = 60.0   # being still this long counts as inactivity

    # ---------------- Audio (Activity 3.x) ----------------
    audio_sr: int = 16000
    audio_frame: int = 512
    audio_hop: int = 256
    speech_margin_db: float = 8.0      # dB above the noise floor to count as active sound
    speech_flatness_max: float = 0.30  # spectral flatness below this => tonal (speech-like)
    speech_zcr_max: float = 0.35       # very high ZCR => hiss / noise, not voiced speech
    noisy_abs_db: float = -22.0        # loud, noise-like frames above this are "noise"
    noisy_ratio_thr: float = 0.25      # share of noise frames that makes a window "noisy"

    # ---------------- Fusion & scoring (Activity 4.x) ----------------
    window_seconds: float = 10.0       # engagement is scored per window
    # visual = attention x behaviour;  behaviour = weighted mix of "not fidgeting" and "not inactive"
    w_movement: float = 0.40
    w_activity: float = 0.60           # (1 - inactivity)
    w_visual: float = 0.70             # visual vs audio contribution to the final score
    w_audio: float = 0.30
    noise_penalty: float = 0.15        # subtracted when the room is noisy
    high_thr: float = 0.70
    low_thr: float = 0.45
    smooth_windows: int = 3

    # ---------------- Agent rules ----------------
    sustained_low_windows: int = 3     # consecutive low windows => sustained disengagement
    drop_delta: float = 0.20           # engagement fall between smoothed windows
    noisy_session_ratio: float = 0.30  # share of noisy windows that triggers a noise suggestion
    long_silence_windows: int = 4      # consecutive windows with almost no speech
    fatigue_delta: float = 0.10        # second-half vs first-half decline
    late_arrival_fraction: float = 0.08  # joined after this share of the session => 'late'

    # ---------------- Privacy ----------------
    store_face_images: bool = False    # never persisted by default
    blur_faces_in_annotation: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path=None) -> "Config":
        if not path:
            return cls()
        data = json.loads(Path(path).read_text())
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
