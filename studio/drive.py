"""Clips driven by a real person's movement instead of written choreography.

The motion file (driving/*.json) comes from the website's "Record Aiva's motion"
page (accesstocapital-web: pages/admin/aiva-motion.js). Someone sits in front
of their webcam and follows a short on-screen script (sit, listen, talk, think,
nod, smile ...) while MediaPipe's face tracker measures, in their browser, how
their head turns, when they blink, how their mouth opens and how much they
smile. Only those numbers are saved: no video or picture of the person ever
leaves their computer or lands in this repository. The personas keep their own
faces; only the timing and movement are borrowed.

File format (version 1):
  fps                    frames per second of the tracks (25)
  frames                 equal-length tracks: pitch, yaw, roll (degrees, the
                         person's own), blinkL, blinkR, jaw, smile (0-1), found (0/1)
  clips                  {clip name: [start, end] seconds}
  neutral                [[start, end], ...] moments of looking at the camera, relaxed

Each clip comes out as frame parameters for lpdriver.Driver.render, measured
from the person's resting pose, and eases from and back to the portrait at
both ends like the written clips, so the website can chain them.
"""
import json

import numpy as np

from motions import FPS, window

# How far the head may turn from the portrait, in degrees (pitch, yaw, roll):
# LivePortrait stays photoreal within about this range.
LIMIT = np.array([14.0, 22.0, 12.0], dtype=np.float32)
# MediaPipe's angles, turned into the driver's (positive pitch tips the head
# down, positive yaw turns it to the viewer's left, positive roll tilts it).
# Measured by recording Sienna's own written clips through the recording page:
# the tracker reads the driver's pitch, yaw and roll at +0.54, -0.58 and -0.74
# times (correlation 0.78, 0.97, 0.98), because LivePortrait turns the face a
# little less than the angle it's given and MediaPipe's yaw and roll run the
# other way. These undo that, so a persona turns as far as the person did.
GAIN = np.array([1.85, -1.72, -1.35], dtype=np.float32)
# A fully open mouth from the driver reads about 0.1 of MediaPipe's jawOpen on
# a rendered face; everyday speech is about 0.05-0.25 on a real one.
JAW_SPAN = 0.25
SMILE_SPAN = 0.6
EDGE = 0.4          # seconds to ease in from, and back to, the portrait


def load(path):
    with open(path) as f:
        rec = json.load(f)
    if rec.get("version") != 1:
        raise ValueError(f"{path}: unknown motion file version {rec.get('version')!r}")
    if int(rec.get("fps", FPS)) != FPS:
        raise ValueError(f"{path}: tracks must be {FPS} fps")
    tracks = {k: np.asarray(v, dtype=np.float32) for k, v in rec["frames"].items()}
    n = len(tracks["pitch"])
    if any(len(v) != n for v in tracks.values()):
        raise ValueError(f"{path}: tracks differ in length")
    rec["tracks"] = _clean(tracks)
    rec["rest"] = _rest(rec)
    return rec


def _fill(x, ok):
    """Frames where the face wasn't found take the nearest found values."""
    if ok.all() or not ok.any():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok]).astype(np.float32)


def _smooth(x, taps):
    if taps <= 1:
        return x
    k = np.hanning(taps + 2)[1:-1]
    k /= k.sum()
    pad = taps // 2
    return np.convolve(np.pad(x, pad, mode="edge"), k, mode="valid").astype(np.float32)


def _clean(t):
    ok = t.get("found", np.ones_like(t["pitch"])) > 0.5
    out = {}
    for k, v in t.items():
        if k == "found":
            continue
        v = _fill(v, ok)
        # a light touch on the head angles and mouth (tracking jitter); blinks and nods are fast
        out[k] = _smooth(v, 3 if k in ("pitch", "yaw", "roll", "jaw", "smile") else 1)
    return out


def _rest(rec):
    """The person's resting pose and expression: medians over the neutral moments."""
    t = rec["tracks"]
    spans = rec.get("neutral") or [[0, 3]]
    idx = np.concatenate([np.arange(int(a * FPS), int(b * FPS)) for a, b in spans])
    idx = idx[(idx >= 0) & (idx < len(t["pitch"]))]
    if not len(idx):
        idx = np.arange(min(len(t["pitch"]), 3 * FPS))
    rest = {k: float(np.median(v[idx])) for k, v in t.items()}
    blink = np.maximum(t["blinkL"], t["blinkR"])
    # how far the eyelids travel for this person: a real blink, not a squint
    rest["blink_span"] = max(0.3, float(np.percentile(blink, 99)) - max(rest["blinkL"], rest["blinkR"]))
    return rest


def _peak(template):
    """The strongest moment of an expression template, as one (21, 3) offset."""
    exp, _ = template
    return exp[int(np.argmax(np.linalg.norm(exp.reshape(len(exp), -1), axis=1)))]


def clip(rec, name, templates, rest=None):
    """Frame parameters for one recorded clip: a list of dicts for Driver.render.
    rest: the persona's resting expression (motions.rest_expression), worn
    throughout, with the recorded smile and wink on top."""
    a, b = rec["clips"][name]
    t, rest = rec["tracks"], rec["rest"]
    i0 = max(0, int(round(a * FPS)))
    i1 = min(len(t["pitch"]), int(round(b * FPS)))
    if i1 - i0 < FPS:
        raise ValueError(f"{name}: recorded clip is shorter than a second")
    dur = (i1 - i0) / FPS
    smile_exp = _peak(templates["smile"])
    wink_exp = _peak(templates["wink"])
    b0 = max(rest["blinkL"], rest["blinkR"])
    frames = []
    for k, i in enumerate(range(i0, i1)):
        ease = window(k / FPS, dur, EDGE)
        pose = np.array([t["pitch"][i] - rest["pitch"], t["yaw"][i] - rest["yaw"], t["roll"][i] - rest["roll"]])
        pose = np.clip(pose * GAIN, -LIMIT, LIMIT) * ease
        blink = max(t["blinkL"][i], t["blinkR"][i])
        eyes = float(np.clip((blink - b0) / rest["blink_span"], 0, 1))
        mouth = float(np.clip((t["jaw"][i] - rest["jaw"]) / JAW_SPAN, 0, 1))
        smile = float(np.clip((t["smile"][i] - rest["smile"]) / SMILE_SPAN, 0, 1))
        wink = float(np.clip((abs(t["blinkL"][i] - t["blinkR"][i]) - 0.25) / 0.35, 0, 1))
        exp = (smile * 0.9 * smile_exp + wink * wink_exp) * ease + (0.0 if rest is None else rest)
        frames.append(dict(pitch=float(pose[0]), yaw=float(pose[1]), roll=float(pose[2]),
                           eyes_closed=eyes * ease * (1 - wink), mouth_open=mouth * ease, exp=exp))
    return frames


def names(rec):
    """Clip names this recording has."""
    return [n for n in rec.get("clips", {}) if n != "neutral"]

