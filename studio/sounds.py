"""Real recordings for sound effects: what the platform's free sound engine
would pick for a request, checked by ear-equivalent before we rely on it.

    python studio/sounds.py [--only gong,whoosh] [--per 8]

For each probe in sounds.json, searches Openverse (public-domain and CC0
audio, which needs no credit line: Freesound, Wikimedia Commons and others)
the way the backend does, downloads the top candidates, measures them
(length, loudness, peak, leading silence) and labels them with an AudioSet
classifier (AST, BSD-3), so we can see whether the top pick really is a gong,
a whoosh or a cash register. Writes out/sounds/<probe>/candidates.json and a
20-second MP3 of each candidate; out/sounds/api_sample.json keeps one raw
search response.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.openverse.org/v1/audio/"
UA = "BlissPointAccess-studio/1.0 (https://github.com/anthonyhammock/aiva-studio)"


def search(q, n=20):
    r = requests.get(API, params={"q": q, "license": "cc0,pdm", "page_size": n}, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json()


def ffmpeg(*args, capture=True):
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], capture_output=capture, text=True)


def measure(path):
    """Length, integrated loudness, true peak and leading silence, from ffmpeg."""
    log = ffmpeg("-i", str(path), "-af", "silencedetect=n=-50dB:d=0.3,ebur128=peak=true", "-f", "null", "-").stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", log)
    seconds = int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3]) if dur else None
    lufs = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    peak = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", log)
    first = re.search(r"silence_start: (-?[\d.]+)\s.*?silence_end: ([\d.]+)", log, re.S)
    lead = float(first[2]) if first and float(first[1]) <= 0.05 else 0.0
    return {"seconds": round(seconds, 2) if seconds else None, "lufs": float(lufs[-1]) if lufs else None,
            "peak_db": float(peak[-1]) if peak else None, "lead_silence": round(lead, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--per", type=int, default=8)
    args = ap.parse_args()
    probes = json.loads((ROOT / "sounds.json").read_text())["probes"]
    only = [x for x in args.only.split(",") if x]
    probes = [p for p in probes if not only or p["id"] in only]

    from transformers import pipeline
    clf = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device=-1)

    out_root = ROOT / "out" / "sounds"
    out_root.mkdir(parents=True, exist_ok=True)
    for p in probes:
        out = out_root / p["id"]
        out.mkdir(exist_ok=True)
        try:
            res = search(p["query"])
        except Exception as e:
            print(f"{p['id']}: search failed: {e}", flush=True)
            continue
        if not (out_root / "api_sample.json").exists():
            (out_root / "api_sample.json").write_text(json.dumps(res, indent=2)[:60000] + "\n")
        rows = []
        for rank, r in enumerate(res.get("results") or []):
            if len([x for x in rows if x.get("labels")]) >= args.per:
                break
            row = {"rank": rank, **{k: r.get(k) for k in ("id", "title", "license", "source", "provider", "creator", "duration",
                                                           "filetype", "bit_rate", "sample_rate", "category", "url",
                                                           "foreign_landing_url")},
                   "tags": [t.get("name") for t in (r.get("tags") or [])][:12]}
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    src = Path(tmp) / "src"
                    resp = requests.get(r["url"], headers={"User-Agent": UA}, timeout=60)
                    resp.raise_for_status()
                    if len(resp.content) > 40_000_000:
                        raise ValueError("over 40 MB")
                    src.write_bytes(resp.content)
                    row.update(measure(src))
                    wav = Path(tmp) / "clip.wav"
                    lead = row.get("lead_silence") or 0
                    ffmpeg("-y", "-ss", str(lead), "-t", "10", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav))
                    row["labels"] = [{"label": x["label"], "score": round(x["score"], 3)} for x in clf(str(wav), top_k=6)]
                    ffmpeg("-y", "-ss", str(lead), "-t", "20", "-i", str(src), "-af", "afade=t=out:st=19:d=1,loudnorm=I=-16:TP=-1",
                           "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(out / f"{rank:02d}.mp3"))
            except Exception as e:
                row["error"] = str(e)[:200]
            rows.append(row)
            print(f"{p['id']} #{rank} {row.get('title')!r} {row.get('source')} {row.get('seconds')}s "
                  f"{[(x['label'], x['score']) for x in row.get('labels', [])[:3]]} {row.get('error', '')}", flush=True)
            time.sleep(0.5)
        (out / "candidates.json").write_text(json.dumps({"probe": p, "count": res.get("result_count"), "candidates": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
