"""Animates one persona's chosen portrait into the clip set the website plays.

    python studio/animate.py --person sienna [--clips idle,talk] [--force] [--motion driving/take.json]

With --motion, clips move the way a real person moved in that recording (see
drive.py); clips the recording doesn't have keep the written choreography.

Reads the pick from picks.json and the portrait from out/candidates/<id>/<n>.webp.
Writes out/people/<id>/: portrait.webp, portrait-sm.webp, <clip>.mp4, meta.json,
plus out/review/<id>/<clip>.jpg (six frames per clip, for a quick look).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import drive  # noqa: E402
import motions  # noqa: E402
from lpdriver import Driver, load_template  # noqa: E402

CLIP_W, CLIP_H, CRF = 576, 720, 22


def encode(frames, path, size):
    w, h = size
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(motions.FPS), "-i", "-", "-an",
                          "-vf", f"scale={CLIP_W}:{CLIP_H}:flags=lanczos,format=yuv420p",
                          "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF), "-profile:v", "high",
                          "-g", str(motions.FPS * 2), "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def review_sheet(frames, path):
    picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 6)]
    small = [cv2.resize(f, (288, 360), interpolation=cv2.INTER_AREA) for f in picks]
    sheet = np.vstack([np.hstack(small[:3]), np.hstack(small[3:])])
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])


def save_webp(rgb, path, width):
    h = round(rgb.shape[0] * width / rgb.shape[1])
    img = cv2.resize(rgb, (width, h), interpolation=cv2.INTER_AREA) if width != rgb.shape[1] else rgb
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 88])


def warmth_sheet(drv, templates, path):
    """The persona at a range of resting smiles (motions.WARMTH_SHEET), labelled."""
    rows = []
    for label, kp, tpl in motions.WARMTH_SHEET["rows"]:
        tiles = []
        for k in motions.WARMTH_SHEET["strengths"]:
            img = drv.render(exp=k * motions.smile_shape(templates, kp, tpl))
            f = drv.face_box()
            H, W = img.shape[:2]
            x0, x1 = int((f["cx"] - 0.62 * f["w"]) * W), int((f["cx"] + 0.62 * f["w"]) * W)
            y0, y1 = int((f["cy"] - 0.6 * f["h"]) * H), int((f["cy"] + 0.7 * f["h"]) * H)
            tile = cv2.resize(np.ascontiguousarray(img[max(0, y0):y1, max(0, x0):x1]), (300, 340), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"{label} {k:.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--clips", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preview", action="store_true", help="write to out/preview/<id>/ and leave out/people alone")
    ap.add_argument("--motion", default="", help="a motion recording (driving/*.json) to move like")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    pick = json.loads((ROOT / "picks.json").read_text()).get(args.person)
    if not pick:
        sys.exit(f"{args.person}: no pick in picks.json yet")
    src = ROOT / "out" / "candidates" / args.person / f"{pick}.webp"
    portrait = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
    # the website shows 4:5; SDXL sizes come in steps of 64 (896x1152), so trim
    # the extra height, a little more from the bottom than the top
    h, w = portrait.shape[:2]
    if h > round(w * 1.25):
        extra = h - round(w * 1.25)
        top = extra // 3
        portrait = np.ascontiguousarray(portrait[top:top + round(w * 1.25)])
    h, w = portrait.shape[:2]

    out = ROOT / "out" / ("preview" if args.preview else "people") / args.person
    review = out / "review" if args.preview else ROOT / "out" / "review" / args.person
    out.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("source") != f"{pick}.webp":
        meta = {}

    t0 = time.time()
    drv = Driver(portrait)
    templates = motions.derive({name: load_template(name) for name in motions.TEMPLATES})
    print(f"{args.person}: models ready in {time.time() - t0:.0f}s; face {drv.face_box()}", flush=True)

    # the resting pose every clip starts and ends on: the portrait with the
    # persona's soft smile (motions.WARMTH)
    warm = motions.warmth(args.person)
    rest = motions.rest_expression(args.person, templates)
    neutral = drv.render(exp=rest)
    save_webp(neutral, out / "portrait.webp", w)
    save_webp(neutral, out / "portrait-sm.webp", 480)

    rec = drive.load(ROOT / args.motion) if args.motion else None
    named = [c for c in args.clips.split(",") if c]
    if "warmth_sheet" in named:
        named.remove("warmth_sheet")
        warmth_sheet(drv, templates, review / "warmth.jpg")
        print(f"{args.person}: warmth sheet in {time.time() - t0:.0f}s", flush=True)
        if not named:
            return
    wanted = named or (drive.names(rec) if rec else list(motions.CLIPS))
    clips = meta.get("clips", {})
    for name in wanted:
        target = out / f"{name}.mp4"
        # named and recorded clips are always remade, and so is a clip made on another resting pose
        same_rest = float(clips.get(name, {}).get("warmth", 0.0)) == warm
        if target.exists() and name in clips and same_rest and not (args.force or named or rec):
            print(f"{args.person}/{name}: already made", flush=True)
            continue
        t1 = time.time()
        recorded = bool(rec and name in rec["clips"])
        params = drive.clip(rec, name, templates, rest=rest) if recorded else motions.clip(name, templates, seed=sum(map(ord, args.person)), person=args.person)
        frames = [neutral] + [drv.render(**p) for p in params[1:]]   # frame 0 is the portrait itself
        encode(frames, target, (w, h))
        review_sheet(frames, review / f"{name}.jpg")
        secs = time.time() - t1
        clips[name] = {"d": round(len(frames) / motions.FPS, 2), "frames": len(frames),
                       "motion": f"recorded:{os.path.basename(args.motion)}" if recorded else "written", "warmth": warm}
        print(f"{args.person}/{name}: {len(frames)} frames in {secs:.0f}s ({secs / len(frames):.2f}s/frame)", flush=True)
        meta.update({"source": f"{pick}.webp", "size": [w, h], "face": drv.face_box(), "clips": clips})
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    meta.update({"source": f"{pick}.webp", "size": [w, h], "face": drv.face_box(), "clips": clips})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{args.person}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
