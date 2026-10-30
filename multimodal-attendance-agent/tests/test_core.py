import numpy as np
import pandas as pd
import pytest

from attendance_agent.config import Config
from attendance_agent.attendance import compute_attendance, presence_mask, sample_grid
from attendance_agent.audio.analysis import analyse_audio
from attendance_agent.agent import EngagementAgent
from attendance_agent.session import analyse
from attendance_agent.vision.detector import Detection, iou
from attendance_agent.vision.tracker import FaceTracker

SR = 16000


def make_det(tracks, fps=5.0, **kw):
    """tracks: {track_id: [(t0, t1), ...]} -> detection DataFrame."""
    rows = []
    for tid, spans in tracks.items():
        for a, b in spans:
            for t in np.arange(a, b, 1 / fps):
                rows.append(dict(t=round(float(t), 3), track_id=tid, x=0, y=0, w=100, h=100, cx=50, cy=50,
                                 motion=kw.get("motion", 0.3), yaw=None, pitch=None, roll=None,
                                 facing_forward=kw.get("ff", True), score=1.0))
    return pd.DataFrame(rows)


def speech(n_s, amp=0.2, f0=150):
    t = np.arange(int(n_s * SR)) / SR
    ph = 2 * np.pi * f0 * t
    sig = sum(np.sin(k * ph) / k for k in range(1, 15))
    return (amp * sig * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) / 2).astype(np.float32)


# ---------------------------------------------------------------- attendance
def test_brief_appearance_not_marked_present():
    cfg = Config()
    det = make_det({1: [(0, 60)], 2: [(10, 13)]})
    att, *_ = compute_attendance(det, 60, cfg)
    assert att.set_index("track_id").loc[1, "marked_present"]
    assert not att.set_index("track_id").loc[2, "marked_present"]


def test_short_gap_is_bridged_long_gap_is_not():
    cfg = Config()
    grid = sample_grid(30, cfg.sample_fps)
    short = np.r_[np.arange(0, 10, .2), np.arange(12, 30, .2)]       # 2 s gap
    long_ = np.r_[np.arange(0, 10, .2), np.arange(20, 30, .2)]       # 10 s gap
    assert presence_mask(short, grid, cfg.sample_fps, cfg.merge_gap_seconds).all()
    assert not presence_mask(long_, grid, cfg.sample_fps, cfg.merge_gap_seconds)[60:90].any()


def test_cumulative_presence_across_visits():
    cfg = Config(min_presence_seconds=20, min_presence_ratio=1.0)
    det = make_det({1: [(0, 12), (40, 52)]})                         # 12 + 12 s, far apart
    att, ev, *_ = compute_attendance(det, 60, cfg)
    assert att.loc[0, "marked_present"] and att.loc[0, "segments"] == 2
    assert {"appeared", "stepped_out", "returned", "left", "attendance_confirmed"} <= set(ev.event)


def test_participants_labelled_by_first_appearance():
    det = make_det({7: [(30, 60)], 3: [(0, 60)]})
    att, *_ = compute_attendance(det, 60, Config())
    assert att.set_index("track_id").loc[3, "participant"] == "P1"


def test_empty_detections():
    att, ev, masks, ids = compute_attendance(pd.DataFrame(columns=make_det({1: [(0, 1)]}).columns), 30, Config())
    assert att.empty and ids == {}


# --------------------------------------------------------------------- audio
def test_speech_detected_silence_not_noise_flagged():
    cfg = Config()
    y = np.concatenate([speech(10), np.zeros(10 * SR, np.float32) + np.random.default_rng(0).normal(0, .001, 10 * SR).astype(np.float32),
                        np.random.default_rng(1).normal(0, .15, 10 * SR).astype(np.float32)])
    a = analyse_audio(y, SR, 30, cfg)
    assert a.audio_state.tolist() == ["speech", "silence", "noisy"]
    assert a.speech_ratio[0] > .6 and a.speech_ratio[1] < .05


def test_missing_audio_degrades_gracefully():
    a = analyse_audio(np.zeros(0, np.float32), SR, 30, Config())
    assert (a.audio_state == "no_audio").all()


# ----------------------------------------------------------------- tracking
def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_tracker_keeps_identity_and_separates_people():
    tr = FaceTracker(Config())
    ids = []
    for k in range(10):
        out = tr.update(k * .2, [Detection(100 + k, 100, 80, 80), Detection(400, 100, 80, 80)])
        ids.append(sorted(r["track_id"] for r in out))
    assert all(i == ids[0] for i in ids) and len(set(ids[0])) == 2


def test_tracker_profile_hit_cannot_spawn_track():
    tr = FaceTracker(Config())
    out = tr.update(0, [Detection(10, 10, 80, 80, spawn=False)])
    assert out == []


# ------------------------------------------------------ end-to-end (analysis)
def _scenario():
    """0-30 s attentive + speech, 30-90 s looking away + silence, 90-120 s attentive + speech."""
    cfg = Config()
    d1 = make_det({1: [(0, 30), (90, 120)], 2: [(0, 30), (90, 120)]})
    away = make_det({1: [(30, 90)], 2: [(30, 90)]}, ff=False)
    det = pd.concat([d1, away]).sort_values("t").reset_index(drop=True)
    y = np.concatenate([speech(30), np.random.default_rng(0).normal(0, .001, 60 * SR).astype(np.float32), speech(30)])
    return cfg, det, y


def test_engagement_dips_when_looking_away_and_agent_notices():
    cfg, det, y = _scenario()
    att, ev, tl, sw, aw, rep, students = analyse(det, 120, y, SR, cfg)
    assert att.marked_present.sum() == 2
    assert tl.engagement.iloc[0] > .85 and tl.engagement.iloc[5] < .45 and tl.engagement.iloc[-1] > .85
    assert "sustained_low" in {s.rule_id for s in rep.suggestions}
    assert all(0 <= v <= 1 for v in sw.engagement.dropna())
    assert set(students) == {"P1", "P2"}


def test_noise_reduces_score_and_triggers_suggestion():
    cfg = Config()
    det = make_det({1: [(0, 60)]})
    y = np.random.default_rng(2).normal(0, .15, 60 * SR).astype(np.float32)
    _, _, tl, _, _, rep, _ = analyse(det, 60, y, SR, cfg)
    assert (tl.audio_state == "noisy").all()
    assert "noise_session" in {s.rule_id for s in rep.suggestions}


def test_vision_only_when_no_audio():
    cfg, det, _ = _scenario()
    _, _, tl, _, _, rep, _ = analyse(det, 120, np.zeros(0, np.float32), SR, cfg)
    assert tl.engagement.notna().all()
    assert "audio_missing" in {s.rule_id for s in rep.suggestions}


def test_no_participants_gives_informative_suggestion_not_crash():
    det = make_det({1: [(0, 1)]})
    _, _, tl, _, _, rep, students = analyse(det, 60, np.zeros(0, np.float32), SR, Config())
    assert tl.empty and students == {} and rep.suggestions[0].rule_id == "R0"


def test_student_report_has_no_comparison_fields():
    cfg, det, y = _scenario()
    *_, students = analyse(det, 120, y, SR, cfg)
    blob = str(students["P1"]).lower()
    for banned in ("rank", "percentile", "class average", "better than"):
        assert banned not in blob


def test_config_roundtrip(tmp_path):
    c = Config(min_presence_seconds=33)
    c.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json").min_presence_seconds == 33
