"""Turns request.json into the workflow's job list (GitHub Actions outputs).

Pushing a motion recording (driving/*.json) on its own, without touching
request.json, re-animates every persona with it.
"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
req = json.loads((ROOT / "request.json").read_text())
people = [p["id"] for p in json.loads((ROOT / "people.json").read_text())["people"]]
only = [x for x in req.get("only", []) if x]
step = req.get("step", "portraits")
unknown = [x for x in only if x not in people]
if unknown and step not in ("music", "sounds"):
    raise SystemExit(f"request.json names unknown personas: {unknown}")
chosen = only or people
motion = req.get("motion", "")
clips = req.get("clips", [])
preview = bool(req.get("preview"))
if os.environ.get("GITHUB_EVENT_NAME") == "push":
    changed = subprocess.run(["git", "diff", "--name-only", "HEAD~1", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.split()
    recordings = [f for f in changed if f.startswith("driving/") and f.endswith(".json")]
    if recordings and "request.json" not in changed:
        step, chosen, motion, clips, preview = "animate", people, recordings[-1], [], False
if step == "music":
    # the matrix is the genres to make (music.json), not personas
    genres = list(json.loads((ROOT / "music.json").read_text())["genres"])
    unknown = [x for x in only if x not in genres]
    if unknown:
        raise SystemExit(f"request.json names unknown genres: {unknown}")
    chosen = only or genres
if step == "sounds":
    # one job; `only` picks probes from sounds.json
    chosen = only
if motion and not (ROOT / motion).exists():
    raise SystemExit(f"motion recording {motion} isn't in the repository")
if step == "animate":
    picks = json.loads((ROOT / "picks.json").read_text())
    chosen = [p for p in chosen if picks.get(p)]
if step not in ("portraits", "animate", "music", "sounds"):
    raise SystemExit(f"request.json step must be portraits, animate, music or sounds, not {step!r}")
out = {"step": step, "people": json.dumps(chosen), "count": str(req.get("count", 3)),
       "model": req.get("model", "lightning"), "clips": ",".join(clips),
       "preview": "--preview" if preview else "", "motion": motion, "seconds": str(req.get("seconds", 0))}
with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
    for k, v in out.items():
        f.write(f"{k}={v}\n")
