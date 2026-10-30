"""Audio signal extraction (Activity 3.1).

* from a video / audio file (ffmpeg -> mono, 16 kHz, float32),
* or from a live microphone (``sounddevice``).

All audio is normalised to the same sample-rate / channel layout so downstream
analysis is identical for every source.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.") from exc


def load_audio(path: str | Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode the audio track of any media file to a mono float32 array at ``sr``.

    Returns ``(empty_array, sr)`` when the file has no audio stream, so the rest of
    the system can degrade gracefully to vision-only mode.
    """
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        cmd = [_ffmpeg_exe(), "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(sr), "-f", "wav", str(wav)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not wav.exists() or wav.stat().st_size < 100:
            return np.zeros(0, dtype=np.float32), sr
        import soundfile as sf
        y, file_sr = sf.read(str(wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), file_sr


class LiveRecorder:
    """Record the microphone in a background thread while the webcam is processed."""

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self.available = True

    def start(self) -> bool:
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                          callback=lambda d, *_: self._chunks.append(d.copy()))
            self._stream.start()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[audio] microphone unavailable ({exc}); continuing in vision-only mode.")
            self.available = False
            return False

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks, axis=0).reshape(-1)
