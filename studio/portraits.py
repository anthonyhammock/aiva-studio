"""Portrait candidates for one persona, made on the CPU with stable-diffusion.cpp
and RealVisXL V5.0 (an open, photoreal SDXL model; CreativeML Open RAIL++-M).

    python studio/portraits.py --person sienna --count 3

Writes out/candidates/<id>/<n>.webp plus <n>.json (seed, prompt, settings).
Finished candidates are skipped, so a re-run only makes what's missing.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H = 896, 1120          # 4:5, about one megapixel (an SDXL-native size)

# Lightning: 6 steps with light guidance. Full: the slower, slightly richer model.
SETTINGS = {
    "lightning": dict(steps=6, cfg=1.5, sampler="dpm++2m_sde", scheduler="karras"),
    "full": dict(steps=30, cfg=4.5, sampler="dpm++2m", scheduler="karras"),
}

NEGATIVE = ("cartoon, anime, 3d render, cgi, illustration, painting, plastic skin, airbrushed, "
            "doll, deformed, asymmetric eyes, cross-eyed, bad teeth, extra fingers, hands, blurry, "
            "lowres, jpeg artifacts, watermark, text, logo, frame")


def prompt_for(p):
    return (f"RAW photo, close-up head and shoulders portrait of {p['look']}, {p['place']}, "
            "looking straight into the camera, relaxed natural expression, lips gently closed, "
            "soft natural window light, 85mm lens at f/2, shallow depth of field, natural skin "
            "texture with fine pores, subtle film grain, photorealistic, sharp focus on the eyes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--model", default=os.environ.get("STUDIO_SD_MODEL_KIND", "lightning"), choices=SETTINGS)
    args = ap.parse_args()

    people = {p["id"]: p for p in json.loads((ROOT / "people.json").read_text())["people"]}
    if args.person not in people:
        sys.exit(f"unknown persona {args.person!r}")
    p = people[args.person]
    index = list(people).index(p["id"])
    sd = os.environ["SD_CLI"]
    model = os.environ["SD_MODEL"]
    vae = os.environ.get("SD_VAE")
    s = SETTINGS[args.model]
    out_dir = ROOT / "out" / "candidates" / p["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in range(1, args.count + 1):
        webp = out_dir / f"{n}.webp"
        if webp.exists():
            print(f"{p['id']} #{n}: already made")
            continue
        seed = 7000 + index * 100 + n
        png = out_dir / f".{n}.png"
        cmd = [sd, "--model", model, "--type", "q8_0", "--threads", str(os.cpu_count() or 4),
               "--prompt", prompt_for(p), "--negative-prompt", NEGATIVE,
               "--width", str(W), "--height", str(H), "--steps", str(s["steps"]), "--cfg-scale", str(s["cfg"]),
               "--sampling-method", s["sampler"], "--scheduler", s["scheduler"], "--seed", str(seed),
               "--rng", "cpu", "--vae-tiling", "--output", str(png)]
        if vae:
            cmd[3:3] = ["--vae", vae]
        t0 = time.time()
        print(f"{p['id']} #{n}: generating (seed {seed}, {args.model})", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not png.exists():
            print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
            sys.exit(f"{p['id']} #{n}: stable-diffusion.cpp failed ({r.returncode})")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(png),
                        "-c:v", "libwebp", "-quality", "94", str(webp)], check=True)
        png.unlink()
        (out_dir / f"{n}.json").write_text(json.dumps({
            "persona": p["id"], "n": n, "seed": seed, "model": os.path.basename(model), "settings": s,
            "prompt": prompt_for(p), "negative": NEGATIVE, "size": [W, H], "seconds": round(time.time() - t0),
        }, indent=2) + "\n")
        print(f"{p['id']} #{n}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
