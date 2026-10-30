"""Generate a fully synthetic demo classroom (video + audio) so the whole pipeline can be
tried without any real footage:

    python scripts/make_demo.py              # -> data/demo_class.mp4  (150 s)

Faces come from scikit-image's public-domain astronaut photo (hue-shifted per
"student"). The script scripts the session so that every feature of the system has
something to find:

  0-40 s    lecture, calm and attentive, teacher speaking          (high engagement)
  40-110 s  passive stretch: students look down / sit still, silence (sustained LOW)
  110-130 s Q&A: movement + several voices                         (PEAK)
  130-150 s group work: loud overlapping chatter and noise         (NOISY)

  Student A  whole session                      -> present
  Student B  arrives late (15 s)                -> present (late arrival)
  Student C  leaves early (110 s)                -> present (early exit)
  Visitor D  appears for 5 s only               -> NOT marked present
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attendance_agent.audio.extract import _ffmpeg_exe  # noqa: E402

W, H, FPS, SR = 1040, 700, 10, 16000
DUR = 150


def base_face():
    from skimage import data
    from attendance_agent.config import Config
    from attendance_agent.vision.detector import HaarDetector
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    x, y, w, h = HaarDetector(Config()).detect(img)[0].box
    crop = img[max(0, y - int(.45 * h)): y + int(1.45 * h), max(0, x - int(.5 * w)): x + int(1.5 * w)]
    return cv2.resize(crop, None, fx=1.35, fy=1.35)


def tint(face, hue_shift, bright=1.0):
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 2] = np.clip(hsv[..., 2] * bright, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def phase(t):
    return 0 if t < 40 else 1 if t < 110 else 2 if t < 130 else 3


def visible(student, t):
    """(visible, looking_down) for each scripted student at time t."""
    ph = phase(t)
    if student == "A":
        on = True
    elif student == "B":
        on = t >= 15
    elif student == "C":
        on = t < 110
    else:  # D
        return (55 <= t < 60), False
    down = False
    if ph == 1 and student in ("A", "B", "C"):      # passive stretch: heads down most of the time
        shift = {"A": 0.0, "B": 1.3, "C": 2.6}[student]
        down = ((t + shift) % 4) >= 1.5               # visible 1.5 s, down 2.5 s
    return on and not down, down


def motion(student, t, seed):
    ph = phase(t)
    amp = {0: 8, 1: 0.0, 2: 14, 3: 30}[ph]
    f = {0: 0.5, 1: 0.1, 2: 0.9, 3: 1.6}[ph]
    return (amp * np.sin(2 * np.pi * f * t + seed), amp * 0.6 * np.cos(2 * np.pi * f * t * 0.8 + seed))


def render_video(path):
    face0 = base_face()
    fh, fw = face0.shape[:2]
    seats = {"A": (60, 60, 0, 1.0), "B": (390, 60, -5, 0.88), "C": (720, 60, 6, 1.12), "D": (390, 420, 12, 0.95)}
    faces = {k: tint(face0, v[2], v[3]) for k, v in seats.items()}
    bg = np.zeros((H, W, 3), np.uint8)
    for r in range(H):
        bg[r, :] = (60 + r // 14, 52 + r // 18, 46 + r // 20)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(int(DUR * FPS)):
        t = i / FPS
        frame = bg.copy()
        for k, (sx, sy, _, _) in seats.items():
            vis, down = visible(k, t)
            dx, dy = motion(k, t, ord(k))
            x, y = int(sx + dx), int(sy + dy)
            if vis:
                frame[y:y + fh, x:x + fw] = faces[k]
            elif down and k in ("A", "B", "C"):      # head down: only the top of the hair is visible
                frame[y + int(fh * .75):y + fh, x:x + fw] = faces[k][int(fh * .75):, :]
        vw.write(frame)
    vw.release()


def voiced(n, f0, rng, amp=0.2):
    t = np.arange(n) / SR
    f = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    sig = sum((1 / (k ** 1.1)) * np.sin(k * ph) for k in range(1, 18))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.2 * t)           # syllable rhythm
    return amp * sig * env / 2.2


def bursts(t0, t1, f0, rng, duty=0.75, amp=0.2):
    out = np.zeros(int(DUR * SR), dtype=np.float32)
    t = t0
    while t < t1:
        dur = rng.uniform(1.2, 3.0)
        end = min(t1, t + dur)
        a, b = int(t * SR), int(end * SR)
        out[a:b] += voiced(b - a, f0 * rng.uniform(.95, 1.1), rng, amp)
        t = end + rng.uniform(0.2, 0.8) / max(duty, 0.1) * (1 - duty + 0.1)
    return out


def render_audio(path):
    rng = np.random.default_rng(7)
    a = rng.normal(0, 0.0015, int(DUR * SR)).astype(np.float32)               # room tone
    a += bursts(2, 40, 120, rng)                                               # teacher lecture
    a += bursts(110, 130, 120, rng, duty=0.5)                                  # teacher asks questions
    a += bursts(112, 129, 210, rng, duty=0.35, amp=0.15)                       # students answer
    a += bursts(130, 150, 130, rng, amp=0.15) + bursts(130, 150, 190, rng, amp=0.15) \
        + bursts(131, 150, 240, rng, amp=0.15)                                 # chatter
    for s in (133, 137, 141, 145):                                             # clatter / noise
        n = int(1.6 * SR)
        a[int(s * SR):int(s * SR) + n] += rng.normal(0, 0.12, n).astype(np.float32)
    a = np.clip(a, -0.95, 0.95)
    sf.write(path, a, SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_class.mp4")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_v, tmp_a = out.with_suffix(".tmp.mp4"), out.with_suffix(".wav")
    print("rendering video ..."); render_video(tmp_v)
    print("rendering audio ..."); render_audio(tmp_a)
    subprocess.run([_ffmpeg_exe(), "-y", "-v", "error", "-i", str(tmp_v), "-i", str(tmp_a), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    tmp_v.unlink()
    print(f"wrote {out} (+ {tmp_a})")


if __name__ == "__main__":
    main()
