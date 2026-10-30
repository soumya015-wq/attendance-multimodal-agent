"""Noise and speech activity analysis (Activity 3.2).

Per-frame features (RMS energy in dB, spectral flatness, zero-crossing rate) are
used with an adaptive noise floor to separate silence, background noise and active
speech - no trained model and no transcription, so no spoken content is ever
stored. Results are aggregated into fixed windows that line up with the vision
engagement windows.
"""
from __future__ import annotations

import librosa
import numpy as np
import pandas as pd

AUDIO_COLUMNS = ["window", "t_start", "t_end", "mean_db", "noise_floor_db", "speech_ratio",
                 "noise_ratio", "audio_state"]


def frame_features(y: np.ndarray, sr: int, cfg) -> dict:
    n, hop = cfg.audio_frame, cfg.audio_hop
    rms = librosa.feature.rms(y=y, frame_length=n, hop_length=hop)[0]
    flat = librosa.feature.spectral_flatness(y=y, n_fft=n, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n, hop_length=hop)[0]
    m = min(len(rms), len(flat), len(zcr))
    rms, flat, zcr = rms[:m], flat[:m], zcr[:m]
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    t = librosa.frames_to_time(np.arange(m), sr=sr, hop_length=hop, n_fft=n)
    return dict(db=db, flatness=flat, zcr=zcr, t=t)


def classify_frames(feat: dict, cfg) -> dict:
    db = feat["db"]
    if len(db) == 0:
        return dict(noise_floor=-80.0, speech=np.zeros(0, bool), noise=np.zeros(0, bool))
    noise_floor = float(np.percentile(db, 10))
    active = db > noise_floor + cfg.speech_margin_db
    tonal = (feat["flatness"] < cfg.speech_flatness_max) & (feat["zcr"] < cfg.speech_zcr_max)
    speech = active & tonal
    # speech is bursty: close <120 ms holes and drop <60 ms specks
    speech = _smooth_binary(speech, cfg, close_s=0.12, open_s=0.06)
    # Noise uses an ABSOLUTE loudness test as well: a room that is loud the whole time would
    # otherwise raise its own adaptive floor and look like silence.
    noise = ~speech & ~tonal & (db > cfg.noisy_abs_db)
    return dict(noise_floor=noise_floor, speech=speech, noise=noise)


def _smooth_binary(x: np.ndarray, cfg, close_s: float, open_s: float) -> np.ndarray:
    hop_s = cfg.audio_hop / cfg.audio_sr
    from scipy.ndimage import binary_closing, binary_opening
    c = max(1, int(close_s / hop_s))
    o = max(1, int(open_s / hop_s))
    return binary_opening(binary_closing(x, np.ones(c)), np.ones(o))


def analyse_audio(y: np.ndarray, sr: int, duration: float, cfg) -> pd.DataFrame:
    """Return a DataFrame with one row per ``cfg.window_seconds`` window."""
    n_win = max(1, int(np.ceil(duration / cfg.window_seconds)))
    rows = []
    if len(y) < cfg.audio_frame * 2:
        for w in range(n_win):
            rows.append(dict(window=w, t_start=w * cfg.window_seconds,
                             t_end=min(duration, (w + 1) * cfg.window_seconds),
                             mean_db=np.nan, noise_floor_db=np.nan, speech_ratio=np.nan,
                             noise_ratio=np.nan, audio_state="no_audio"))
        return pd.DataFrame(rows, columns=AUDIO_COLUMNS)

    feat = frame_features(y, sr, cfg)
    cls = classify_frames(feat, cfg)
    t = feat["t"]
    for w in range(n_win):
        a, b = w * cfg.window_seconds, min(duration, (w + 1) * cfg.window_seconds)
        sel = (t >= a) & (t < b)
        if sel.sum() == 0:
            rows.append(dict(window=w, t_start=a, t_end=b, mean_db=np.nan,
                             noise_floor_db=cls["noise_floor"], speech_ratio=np.nan,
                             noise_ratio=np.nan, audio_state="no_audio"))
            continue
        sp = float(cls["speech"][sel].mean())
        nz = float(cls["noise"][sel].mean())
        mdb = float(feat["db"][sel].mean())
        if nz >= cfg.noisy_ratio_thr:
            state = "noisy"
        elif sp >= 0.10:
            state = "speech"
        elif mdb > cls["noise_floor"] + 3:
            state = "ambient"
        else:
            state = "silence"
        rows.append(dict(window=w, t_start=a, t_end=b, mean_db=round(mdb, 2),
                         noise_floor_db=round(cls["noise_floor"], 2),
                         speech_ratio=round(sp, 3), noise_ratio=round(nz, 3), audio_state=state))
    return pd.DataFrame(rows, columns=AUDIO_COLUMNS)


def waveform_envelope(y: np.ndarray, sr: int, points: int = 800) -> pd.DataFrame:
    """Down-sampled amplitude envelope (timestamp + amplitude) for storage/plots."""
    if len(y) == 0:
        return pd.DataFrame(columns=["t", "amplitude"])
    pts = min(points, len(y))
    edges = np.linspace(0, len(y), pts + 1, dtype=int)
    amp = [float(np.abs(y[a:b]).max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    t = (edges[:-1] / sr)
    return pd.DataFrame({"t": np.round(t, 3), "amplitude": np.round(amp, 4)})
