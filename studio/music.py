"""Original instrumentals for the website's Motion videos and Aiva, made with
ACE-Step 1.5 (MIT; its makers allow commercial use of what it generates) on
the CPU of a free runner.

    python studio/music.py --genre amapiano [--count 6] [--seconds 75]
    python studio/music.py --genre amapiano --tidy    (re-check finished tracks for mid-song gaps)

Reads the genre's tracks from music.json (caption, BPM, key) and writes
out/music/<genre>/<id>.mp3 (128 kbps, about 1 MB a minute: MP3 is the one
compressed format every browser decodes, including the open-source Chromium
the website's server-side renderer uses, which has no AAC) plus <id>.json (what it was made from, how long it took), and
out/music/<genre>/index.json listing the genre's finished tracks. Tracks
already made are skipped, so a run that times out resumes.
Only the diffusion model is used: the caption and metadata are written here,
so the 5 Hz language model (which plans songs from a short idea) isn't needed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACE_DIR = os.environ.get("ACE_DIR", "acestep")
sys.path.insert(0, ACE_DIR)

MODEL = "acestep-v15-turbo"


def encode(src, dst):
    """MP3 at 128 kbps, levelled to -14 LUFS (what streaming and social apps
    play at). The model ends its songs a few seconds before the length asked
    for and pads with silence: that tail is cut, after a short fade."""
    chain = ("areverse,silenceremove=start_periods=1:start_silence=0.05:start_threshold=-48dB,areverse,"
             "afade=t=in:d=0.02,loudnorm=I=-14:TP=-1.0:LRA=11")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-af", chain,
                    "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(dst)], check=True)


def cut_gap(path, after=20.0):
    """The model sometimes stops for a few seconds mid-song and starts again,
    which would land in a video as dead air. If a near-silent stretch of 1.5 s
    or more comes after at least `after` seconds of music, the track ends there
    instead, with a one-second fade. Returns whether it was cut."""
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "silencedetect=n=-40dB:d=1.5",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    cut = next((s for s in map(float, re.findall(r"silence_start: ([0-9.]+)", log)) if s >= after), None)
    if cut is None:
        return False
    tmp = path.with_suffix(".cut.mp3")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-t", f"{cut:.2f}",
                    "-af", f"afade=t=out:st={cut - 1:.2f}:d=1", "-c:a", "libmp3lame", "-b:a", "128k", str(tmp)], check=True)
    tmp.replace(path)
    return True


def length_of(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:   # ffmpeg without ffprobe: read the duration ffmpeg reports
        log = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
        h, m, sec = re.search(r"Duration: (\d+):(\d+):([\d.]+)", log).groups()
        out = int(h) * 3600 + int(m) * 60 + float(sec)
    return round(float(out or 0), 2)


def write_index(out, genre_id, genre):
    """The genre's finished tracks, in music.json's order, for the website."""
    done = []
    for t in genre["tracks"]:
        meta = out / f"{t['id']}.json"
        if (out / f"{t['id']}.mp3").exists() and meta.exists():
            m = json.loads(meta.read_text())
            done.append({k: m.get(k) for k in ("id", "title", "bpm", "key", "length", "caption")})
    (out / "index.json").write_text(json.dumps({"genre": genre_id, "name": genre["name"], "tracks": done}, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", required=True)
    ap.add_argument("--count", type=int, default=0, help="the first N tracks of the genre (0: all)")
    ap.add_argument("--seconds", type=float, default=0, help="length of each track (0: music.json's)")
    ap.add_argument("--tidy", action="store_true", help="only re-check finished tracks for mid-song gaps (cut_gap)")
    args = ap.parse_args()

    catalog = json.loads((ROOT / "music.json").read_text())
    genre = catalog["genres"].get(args.genre)
    if not genre:
        sys.exit(f"music.json has no genre {args.genre!r}")
    tracks = genre["tracks"][: args.count or None]
    seconds = args.seconds or float(catalog.get("seconds", 75))
    out = ROOT / "out" / "music" / args.genre
    out.mkdir(parents=True, exist_ok=True)
    if args.tidy:
        for t in tracks:
            mp3, meta = out / f"{t['id']}.mp3", out / f"{t['id']}.json"
            if mp3.exists() and meta.exists() and cut_gap(mp3):
                m = json.loads(meta.read_text())
                m["length"] = length_of(mp3)
                meta.write_text(json.dumps(m, indent=2) + "\n")
                print(f"{args.genre}/{t['id']}: cut at a gap, now {m['length']}s", flush=True)
        write_index(out, args.genre, genre)
        return
    todo = [t for t in tracks if not (out / f"{t['id']}.mp3").exists()]
    if not todo:
        write_index(out, args.genre, genre)
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
            encode(res.audios[0]["path"], out / f"{t['id']}.mp3")
            cut_gap(out / f"{t['id']}.mp3")
        secs = time.time() - t1
        meta = {"id": t["id"], "title": t["title"], "genre": args.genre, "genre_name": genre["name"], "bpm": t["bpm"],
                "key": t.get("key"), "seconds": seconds, "length": length_of(out / f"{t['id']}.mp3"), "caption": t["caption"],
                "seed": seed, "model": f"ACE-Step 1.5 {MODEL}", "made_in_seconds": round(secs),
                "license": "Original music made with ACE-Step 1.5 (MIT). Free for commercial use."}
        (out / f"{t['id']}.json").write_text(json.dumps(meta, indent=2) + "\n")
        write_index(out, args.genre, genre)
        print(f"{args.genre}/{t['id']}: {seconds:.0f}s of music in {secs:.0f}s", flush=True)
    print(f"{args.genre}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
