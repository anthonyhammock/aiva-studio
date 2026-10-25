"""Original instrumentals for the website's Motion videos and Aiva, made with
ACE-Step 1.5 (MIT; its makers allow commercial use of what it generates) on
the CPU of a free runner.

    python studio/music.py --genre amapiano [--count 6] [--seconds 75]

Reads the genre's tracks from music.json (caption, BPM, key) and writes
out/music/<genre>/<id>.mp3 plus <id>.json (what it was made from, how long it
took). Tracks already made are skipped, so a run that times out resumes.
Only the diffusion model is used: the caption and metadata are written here,
so the 5 Hz language model (which plans songs from a short idea) isn't needed.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACE_DIR = os.environ.get("ACE_DIR", "acestep")
sys.path.insert(0, ACE_DIR)

MODEL = "acestep-v15-turbo"


def to_mp3(src, dst):
    """A 192 kbps MP3, levelled to -14 LUFS (what streaming and social apps play at)."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-af", "loudnorm=I=-14:TP=-1.0:LRA=11", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "192k", str(dst)],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", required=True)
    ap.add_argument("--count", type=int, default=0, help="the first N tracks of the genre (0: all)")
    ap.add_argument("--seconds", type=float, default=0, help="length of each track (0: music.json's)")
    args = ap.parse_args()

    catalog = json.loads((ROOT / "music.json").read_text())
    genre = catalog["genres"].get(args.genre)
    if not genre:
        sys.exit(f"music.json has no genre {args.genre!r}")
    tracks = genre["tracks"][: args.count or None]
    seconds = args.seconds or float(catalog.get("seconds", 75))
    out = ROOT / "out" / "music" / args.genre
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tracks if not (out / f"{t['id']}.mp3").exists()]
    if not todo:
        print(f"{args.genre}: nothing to make", flush=True)
        return

    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    torch.set_num_threads(os.cpu_count() or 4)
    t0 = time.time()
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=ACE_DIR, config_path=MODEL, device="cpu")
    if not ok:
        sys.exit(f"ACE-Step didn't start: {msg}")
    llm = LLMHandler()          # left unloaded: generation follows the caption and metadata as written
    print(f"{args.genre}: model ready in {time.time() - t0:.0f}s", flush=True)

    for i, t in enumerate(todo):
        seed = sum(map(ord, t["id"])) * 7 + i
        params = GenerationParams(
            task_type="text2music", caption=t["caption"], lyrics="[Instrumental]", instrumental=True,
            bpm=int(t["bpm"]), keyscale=t.get("key", ""), timesignature=str(t.get("time", 4)), duration=seconds,
            inference_steps=8, shift=3.0, seed=seed, thinking=False,
            use_cot_metas=False, use_cot_caption=False, use_cot_language=False, use_cot_lyrics=False)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
        t1 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            res = generate_music(dit, llm, params, config, save_dir=tmp)
            if not res.success or not res.audios:
                print(f"{args.genre}/{t['id']}: failed: {res.error}", flush=True)
                continue
            to_mp3(res.audios[0]["path"], out / f"{t['id']}.mp3")
        secs = time.time() - t1
        meta = {"id": t["id"], "title": t["title"], "genre": args.genre, "genre_name": genre["name"], "bpm": t["bpm"],
                "key": t.get("key"), "seconds": seconds, "caption": t["caption"], "seed": seed, "model": f"ACE-Step 1.5 {MODEL}",
                "made_in_seconds": round(secs), "license": "Original music made with ACE-Step 1.5 (MIT). Free for commercial use."}
        (out / f"{t['id']}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{args.genre}/{t['id']}: {seconds:.0f}s of music in {secs:.0f}s", flush=True)
    print(f"{args.genre}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
