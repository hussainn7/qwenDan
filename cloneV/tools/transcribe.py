"""Whisper large-v3-turbo (mlx) over many wavs in one process; writes <wav>.whisper.json next to each (skips existing).
  ~/reels/.venv/bin/python vo/tools/transcribe.py a.wav b.wav ...   [--force] [--clip START END]
--clip transcribes only that span of the first file and prints words (no file written)."""
import json, os, sys
import mlx_whisper
import numpy as np, soundfile as sf, librosa

MODEL = "mlx-community/whisper-large-v3-turbo"

def load16k(path, clip=None):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(1)
    if clip: x = x[int(clip[0] * sr):int(clip[1] * sr)]
    return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x

def run(x):
    r = mlx_whisper.transcribe(x, path_or_hf_repo=MODEL, language="en", word_timestamps=True,
                               condition_on_previous_text=False, temperature=0.0)
    words = [dict(w=w["word"].strip(), s=round(w["start"], 3), e=round(w["end"], 3), p=round(w["probability"], 3))
             for seg in r["segments"] for w in seg.get("words", [])]
    return dict(text=r["text"].strip(), words=words)

if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args; args = [a for a in args if a != "--force"]
    if "--clip" in args:
        i = args.index("--clip"); s, e = float(args[i + 1]), float(args[i + 2])
        r = run(load16k(args[0], (s, e)))
        print(r["text"]); print(" ".join(f"{w['w']}[{w['s'] + s:.2f}-{w['e'] + s:.2f}]" for w in r["words"]))
        sys.exit()
    for p in args:
        out = p[:-4] + ".whisper.json"
        if os.path.exists(out) and not force: continue
        r = run(load16k(p))
        json.dump(r, open(out, "w"), indent=0)
        print(os.path.basename(p), "|", r["text"], flush=True)
