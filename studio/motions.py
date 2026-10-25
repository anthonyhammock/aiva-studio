"""Choreography for each clip, as plain functions of time.

Every channel (head angles, eyes, mouth, expression) is zero at the first and
last frame with zero velocity there too, so every clip starts and ends on the
same resting pose: the portrait wearing the persona's soft smile (WARMTH).
That's what lets the website chain clips with no cut.

Signs: positive pitch tips the head down (a nod), positive yaw turns it to the
viewer's left, positive roll tilts it. LOOK_UP flips pitch when looking up.
"""
import math
import random

import numpy as np

FPS = 25
LOOK_UP = -1.0

# How warmly each persona rests: a soft, closed-mouth smile held through every
# clip, as a share of the full smile. Rosa and Grace were cast smiling already;
# the others were cast with a neutral resting face, which reads as cool or
# stern on a live call. Changing a persona's warmth remakes all its clips (the
# resting pose is what clips are chained on).
WARMTH = {"rosa": 0.0, "grace": 0.15}
WARMTH_DEFAULT = 0.4


def warmth(person):
    return WARMTH.get(person, WARMTH_DEFAULT) if person else 0.0


def peak(template):
    """The strongest moment of an expression template, as one (21, 3) offset."""
    exp, _ = template
    return exp[int(np.argmax(np.linalg.norm(exp.reshape(len(exp), -1), axis=1)))]


# A smile drawn straight on LivePortrait's expression keypoints: lip corners
# up and back, cheeks up, lower eyelids lifted a touch. Per unit of smile;
# (keypoint, axis): offset. The smile taken from the laugh template alone
# (derive) barely shows on most faces once its open mouth is held back.
KP_SMILE = {(20, 1): -0.01, (14, 1): -0.02, (17, 1): 0.0065, (17, 2): 0.003,
            (13, 1): -0.00275, (16, 1): -0.00275, (3, 1): -0.0035, (7, 1): -0.0035}
# The resting smile: this much of KP_SMILE plus this much of the template's.
REST_MIX = {"kp": 1.0, "template": 0.6}


def kp_smile():
    out = np.zeros((21, 3), dtype=np.float32)
    for (k, axis), v in KP_SMILE.items():
        out[k, axis] = v
    return out


def smile_shape(templates, kp=REST_MIX["kp"], template=REST_MIX["template"]):
    return (kp * kp_smile() + template * peak(templates["smile_tpl"])).astype(np.float32)


def rest_expression(person, templates):
    """The resting expression: the persona's soft smile, held."""
    return (warmth(person) * smile_shape(templates)).astype(np.float32)


# The sheet a run makes with --clips warmth_sheet: the persona at a range of
# resting smiles, to choose WARMTH and REST_MIX by eye in a minute instead of
# remaking every clip. Rows: (label, kp, template) per unit; columns: strengths.
WARMTH_SHEET = {
    "rows": [("keypoints", 1.0, 0.0), ("template", 0.0, 1.0), ("both", 1.0, 0.6)],
    "strengths": [0.0, 0.3, 0.5, 0.8, 1.1],
}


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


def wander(t, dur, seed, amp):
    """Head drift for the long takes: more and slower sines than drift, with
    a longer ease at both ends, so ten seconds in it still hasn't repeated."""
    rnd = random.Random(seed)
    parts = [(rnd.uniform(0.05, 0.3), rnd.uniform(0, 2 * math.pi), rnd.uniform(0.4, 1.0)) for _ in range(5)]
    v = sum(a * math.sin(2 * math.pi * f * t + p) for f, p, a in parts) / 2.5
    return amp * v * window(t, dur, 1.2)


def natural_blinks(dur, seed):
    """Blink times for a long take: every 2-4 seconds, now and then a double."""
    rnd = random.Random(seed)
    t, out = rnd.uniform(0.9, 1.8), []
    while t < dur - 0.7:
        out.append(t)
        if rnd.random() < 0.15 and t + 0.45 < dur - 0.7:
            out.append(t + 0.36)
        t += rnd.uniform(2.0, 4.2)
    return out


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
    """Collects expression layers: (template offsets, start time, intensity, ramp).
    Smiles are scaled by what's left above the resting smile, so a full smile
    peaks at the same strength whatever the persona's warmth."""

    def __init__(self, templates, warm=0.0):
        self.templates = templates
        self.layers = []
        self.warm = warm

    def add(self, name, start, intensity=1.0, speed=1.0, hold_frac=None, ramp=0.3):
        exp, fps = self.templates[name]
        if name == "smile":
            intensity *= 1.0 - self.warm
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


def clip(name, templates, seed=0, person=None):
    """Frame parameters for one clip: a list of dicts for Driver.render. With
    a person, every frame wears their resting smile (rest_expression)."""
    T = Track(templates, warmth(person))
    rest = rest_expression(person, templates) if person else 0.0
    spec = CLIPS[name]
    dur = spec["dur"]
    spec["build"](T)
    blinks = spec.get("blinks", [])
    if callable(blinks):
        blinks = blinks(dur, seed)
    frames = []
    for i in range(int(round(dur * FPS))):
        t = i / FPS
        p = spec["pose"](t, dur, seed)
        frames.append(dict(
            pitch=p[0], yaw=p[1], roll=p[2],
            eyes_closed=max([blink(t, b) for b in blinks] + [0.0]),
            mouth_open=spec.get("mouth", lambda t, d, s: 0.0)(t, dur, seed),
            exp=rest + T.at(t) * window(t, dur, 0.3),
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


# --- the long takes: the website plays these most, so a calm moment or a
# reply isn't the same five seconds over and over.

def _idle_long(t, d, s):
    """Sitting with the person: two unhurried glances away and back, a small
    nod, a warm tilt near the end, over a slow wander."""
    rnd = random.Random(s + 41)
    side = 1 if rnd.random() < 0.5 else -1
    g1, g2 = rnd.uniform(2.5, 4.0), rnd.uniform(7.0, 8.5)
    glance = side * (7.0 * hold(t, g1, g1 + rnd.uniform(1.6, 2.2), 0.6) - 5.0 * hold(t, g2, g2 + 1.6, 0.6))
    nod = 2.4 * pulse(t, rnd.uniform(5.4, 6.2), 0.6)
    tilt = side * 3.5 * hold(t, 10.5, 13.0, 0.8)
    return (wander(t, d, s + 1, 1.4) + nod + 0.15 * abs(glance), wander(t, d, s + 2, 2.4) + glance, wander(t, d, s + 3, 1.2) + tilt)


def _listen_long(t, d, s):
    """Listening: small nods of different sizes, a held tilt, now and then a smile."""
    rnd = random.Random(s + 43)
    nods = sum(rnd.uniform(2.0, 3.6) * pulse(t, at, 0.55) for at in (1.4, 2.6, 5.2, 7.3))
    return (wander(t, d, s + 1, 0.8) + nods, wander(t, d, s + 2, 1.4), 3.0 * hold(t, 0.5, d - 0.5, 0.9) + wander(t, d, s + 3, 0.6))


def _talk_long(t, d, s):
    """Talking: emphasis on a few words, a slow wander, a glance down to think mid-way."""
    accents = 1.6 * sum(pulse(t, at, 0.4) for at in (1.2, 2.9, 4.6, 6.4))
    look = hold(t, 3.6, 4.4, 0.35)
    return (wander(t, d, s + 1, 1.6) + accents + 2.0 * look, wander(t, d, s + 2, 3.0) - 3.0 * look, wander(t, d, s + 3, 1.6))


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
    "idle_long": dict(dur=14.0, pose=_idle_long, blinks=natural_blinks,
                      build=lambda T: T.add("smile", 1.6, 0.45).add("smile", 6.0, 0.35).add("smile", 10.2, 0.6)),
    "listen_long": dict(dur=9.0, pose=_listen_long, blinks=natural_blinks,
                        build=lambda T: T.add("smile", 0.8, 0.35).add("smile", 4.6, 0.5)),
    "talk_long": dict(dur=8.0, pose=_talk_long, blinks=natural_blinks,
                      mouth=lambda t, d, s: syllables(t, 0.35, d - 0.4, s + 21), build=lambda T: T.add("smile", 2.0, 0.3).add("smile", 5.4, 0.35)),
}

TEMPLATES = ["talking", "laugh", "wink"]

# Keypoint 19 carries most of the jaw / lower-lip drop in LivePortrait's
# expression space (it dominates the laugh and talking templates); 17 helps.
# A laugh with those held back is a warm, closed-mouth smile.
MOUTH_OPEN_KP = {19: 0.08, 17: 0.5}


def derive(templates):
    """Adds templates built from the bundled ones: 'smile_tpl' (a laugh
    without the open mouth) and 'smile', which is that plus KP_SMILE in the
    same rhythm, mixed per REST_MIX, so smiles in clips match the resting one."""
    laugh, fps = templates["laugh"]
    tpl = laugh.copy()
    for kp, keep in MOUTH_OPEN_KP.items():
        tpl[:, kp, :] *= keep
    templates["smile_tpl"] = (tpl, fps)
    norm = np.linalg.norm(tpl.reshape(len(tpl), -1), axis=1)
    env = (norm / max(float(norm.max()), 1e-6)).astype(np.float32)[:, None, None]
    smile = REST_MIX["template"] * tpl + REST_MIX["kp"] * env * kp_smile()[None]
    templates["smile"] = (smile.astype(np.float32), fps)
    return templates
