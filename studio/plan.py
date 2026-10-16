"""Turns request.json into the workflow's job list (GitHub Actions outputs)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
req = json.loads((ROOT / "request.json").read_text())
people = [p["id"] for p in json.loads((ROOT / "people.json").read_text())["people"]]
only = [x for x in req.get("only", []) if x]
unknown = [x for x in only if x not in people]
if unknown:
    raise SystemExit(f"request.json names unknown personas: {unknown}")
chosen = only or people
step = req.get("step", "portraits")
if step == "animate":
    picks = json.loads((ROOT / "picks.json").read_text())
    chosen = [p for p in chosen if picks.get(p)]
if step not in ("portraits", "animate"):
    raise SystemExit(f"request.json step must be portraits or animate, not {step!r}")
out = {"step": step, "people": json.dumps(chosen), "count": str(req.get("count", 3)),
       "model": req.get("model", "lightning"), "clips": ",".join(req.get("clips", [])),
       "preview": "--preview" if req.get("preview") else ""}
with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
    for k, v in out.items():
        f.write(f"{k}={v}\n")
