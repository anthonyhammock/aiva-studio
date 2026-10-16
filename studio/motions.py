"""Choreography for each clip, as plain functions of time.

Every channel (head angles, eyes, mouth, expression) is zero at the first and
last frame with zero velocity there too, so every clip starts and ends on the
untouched portrait. That's what lets the website chain clips with no cut.

Signs: positive pitch tips the head down (a nod), positive yaw turns it to the
viewer's left, positive roll tilts it. LOOK_UP flips pitch when looking up.
"""
import math
import random

import numpy as np

FPS = 25
LOOK_UP = -1.0


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def window(t, dur, ramp=0.45):
    """1 in the middle, easing to 0 (and flat) at both ends."""
    return smooth(t / ramp) * smooth((dur - t) / ramp)


def hold(t, start, end, ramp=0.35):
    """Ease in at start, hold, ease out by end."""
    return smooth((t - start) / ramp) * smooth((end - t) / ramp)


def pulse(t, at, dur):
    """One smooth 0 → 1 → 0 bump centred on at."""
    u = (t - at) / dur + 0.5
    return 0.0 if u <= 0 or u >= 1 else math.sin(math.pi * u) ** 2


def blink(t, at):
    """Eyelids: close in 70 ms, stay shut 40 ms, open over 150 ms."""
    d = t - at
    if d < 0 or d > 0.26:
        return 0.0
    if d < 0.07:
        return smooth(d / 0.07)
    if d < 0.11:
        return 1.0
    return 1 - smooth((d - 0.11) / 0.15)


def drift(t, dur, seed, amp):
    """Small, slow head drift: a sum of sines, windowed to rest at both ends."""
    rnd = random.Random(seed)
    parts = [(rnd.uniform(0.15, 0.45), rnd.uniform(0, 2 * math.pi), rnd.uniform(0.5, 1.0)) for _ in range(3)]
    v = sum(a * math.sin(2 * math.pi * f * t + p) for f, p, a in parts) / 2.0
    return amp * v * window(t, dur, 0.8)


def syllables(t, start, end, seed, rate=4.2):
    """Mouth opening for natural-looking speech: uneven syllables with short
    pauses between phrases. 0 closed, 1 wide."""
    if t <= start or t >= end:
        return 0.0
    rnd = random.Random(seed)
    s, marks = start, []
    while s < end:
        n = rnd.randint(3, 7)                      # syllables in this phrase
        for _ in range(n):
            d = rnd.uniform(0.7, 1.3) / rate
            marks.append((s, d, rnd.uniform(0.35, 1.0)))
            s += d
        s += rnd.uniform(0.18, 0.4)                # breath between phrases
    for s0, d, a in marks:
        if s0 <= t < s0 + d:
            u = (t - s0) / d
            return a * math.sin(math.pi * u) ** 1.5 * hold(t, start, end, 0.2)
    return 0.0


class Track:
    """Collects expression layers: (template offsets, start time, intensity, ramp)."""

    def __init__(self, templates):
        self.templates = templates
        self.layers = []

    def add(self, name, start, intensity=1.0, speed=1.0, hold_frac=None, ramp=0.3):
        exp, fps = self.templates[name]
        self.layers.append((exp, fps * speed, start, intensity, hold_frac, ramp))
        return self

    def at(self, t):
        total = np.zeros((21, 3), dtype=np.float32)
        for exp, fps, start, k, hold_frac, ramp in self.layers:
            n = len(exp)
            length = n / fps
            u = t - start
            if u <= 0 or u >= length:
                continue
            i = u * fps
            a = min(int(i), n - 1)
            b = min(a + 1, n - 1)
            f = exp[a] * (1 - (i - a)) + exp[b] * (i - a)
            total += k * f * hold(u, 0, length, ramp)
        return total


def clip(name, templates, seed=0):
    """Frame parameters for one clip: a list of dicts for Driver.render."""
    T = Track(templates)
    spec = CLIPS[name]
    dur = spec["dur"]
    spec["build"](T)
    frames = []
    for i in range(int(round(dur * FPS))):
        t = i / FPS
        p = spec["pose"](t, dur, seed)
        frames.append(dict(
            pitch=p[0], yaw=p[1], roll=p[2],
            eyes_closed=max([blink(t, b) for b in spec.get("blinks", [])] + [0.0]),
            mouth_open=spec.get("mouth", lambda t, d, s: 0.0)(t, dur, seed),
            exp=T.at(t) * window(t, dur, 0.3),
        ))
    return frames


def _idle(t, d, s):
    return (drift(t, d, s + 1, 1.2), drift(t, d, s + 2, 2.0), drift(t, d, s + 3, 1.0))


def _glance(t, d, s):
    g = hold(t, 1.0, 2.9, 0.45)
    return (drift(t, d, s + 1, 0.8) + 1.2 * g, drift(t, d, s + 2, 1.0) + 9.0 * g, drift(t, d, s + 3, 0.8) + 1.5 * g)


def _listen(t, d, s):
    nods = 3.2 * (pulse(t, 1.1, 0.55) + pulse(t, 2.5, 0.55))
    return (drift(t, d, s + 1, 0.8) + nods, drift(t, d, s + 2, 1.2), 3.0 * hold(t, 0.4, d - 0.4, 0.8))


def _think(t, d, s):
    g = hold(t, 0.6, 3.4, 0.6)
    return (LOOK_UP * 7.0 * g, 10.0 * g + drift(t, d, s, 0.6), -2.5 * g)


def _talk(t, d, s):
    accents = 1.6 * (pulse(t, 1.3, 0.4) + pulse(t, 3.1, 0.4))
    return (drift(t, d, s + 1, 1.6) + accents, drift(t, d, s + 2, 3.0), drift(t, d, s + 3, 1.6))


def _nod(t, d, s):
    return (6.5 * (pulse(t, 1.0, 0.5) + pulse(t, 1.65, 0.5)), drift(t, d, s, 0.8), 1.5 * hold(t, 0.5, d - 0.5, 0.6))


def _tilt(t, d, s):
    g = hold(t, 0.5, d - 0.6, 0.7)
    return (1.0 * g, -3.0 * g, 8.0 * g)


def _still(t, d, s):
    return (drift(t, d, s + 1, 0.8), drift(t, d, s + 2, 1.2), drift(t, d, s + 3, 0.8))


CLIPS = {
    "idle": dict(dur=5.0, pose=_idle, blinks=[1.7, 4.0], build=lambda T: T),
    "idle2": dict(dur=5.0, pose=_glance, blinks=[0.6, 3.4], build=lambda T: T.add("smile", 1.4, 0.4)),
    "listen": dict(dur=5.0, pose=_listen, blinks=[0.5, 3.6], build=lambda T: T.add("smile", 0.8, 0.3)),
    "think": dict(dur=5.0, pose=_think, blinks=[3.5], build=lambda T: T.add("smile", 3.0, 0.5)),
    "talk": dict(dur=5.0, pose=_talk, blinks=[1.9, 4.2], build=lambda T: T.add("talking", 0.3, 1.0)),
    "talk2": dict(dur=5.0, pose=_talk, blinks=[0.9, 3.3],
                  mouth=lambda t, d, s: syllables(t, 0.35, 4.6, s + 9), build=lambda T: T),
    "nod": dict(dur=4.0, pose=_nod, blinks=[2.6], build=lambda T: T.add("smile", 0.5, 0.6)),
    "smile": dict(dur=4.0, pose=_still, blinks=[3.1], build=lambda T: T.add("smile", 0.4, 1.0)),
    "wink": dict(dur=4.0, pose=_still, blinks=[], build=lambda T: T.add("wink", 0.6, 1.0).add("smile", 0.5, 0.4)),
    "head_tilt": dict(dur=4.0, pose=_tilt, blinks=[2.8], build=lambda T: T.add("smile", 0.5, 0.7)),
    "laugh": dict(dur=4.0, pose=_still, blinks=[], build=lambda T: T.add("laugh", 0.4, 0.8)),
}

TEMPLATES = ["talking", "laugh", "wink"]

# Keypoint 19 carries most of the jaw / lower-lip drop in LivePortrait's
# expression space (it dominates the laugh and talking templates); 17 helps.
# A laugh with those held back is a warm, closed-mouth smile.
MOUTH_OPEN_KP = {19: 0.08, 17: 0.5}


def derive(templates):
    """Adds templates built from the bundled ones: 'smile' (a laugh without
    the open mouth)."""
    laugh, fps = templates["laugh"]
    smile = laugh.copy()
    for kp, keep in MOUTH_OPEN_KP.items():
        smile[:, kp, :] *= keep
    templates["smile"] = (smile, fps)
    return templates
