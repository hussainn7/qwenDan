"""Cuts the real take into per-sentence clips (24 kHz mono) and the reference clips used for cloning.
  ~/reels/.venv/bin/python vo/tools/cut_real.py
-> vo/real/<id>.wav (from voice.wav) and <id>.raw.wav (from raw44.wav), vo/real/clips.cut.json (snapped times),
   vo/refs/<name>.wav (+ .raw.wav) and vo/refs/refs.json (audio + exact transcript) for every REFS entry."""
import json, os
import numpy as np, soundfile as sf, librosa

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT = os.path.join(os.path.dirname(VO), "edit")
SR = 24000
PAD = 0.18          # search window past each Whisper word edge for the quietest 10 ms
LEAD, TAIL = 0.08, 0.15   # silence kept before / after the snapped edges
FLUB = (25.3, 28.8)

# reference clips: list of sentence ids; consecutive ids are cut as one continuous stretch (real pauses kept),
# non-consecutive ones are joined with GAP of the take's own room tone
REFS = {
    "A1": ["S01"],                              # 6 s, one energetic question
    "A_s5": ["S05"],                            # 2.7 s, short
    "A3": ["S01", "S02", "S03"],                # ~14 s continuous
    "A_mid": ["S03", "S04", "S05", "S06"],      # ~15 s continuous, statements
    "A_all": ["S01", "S02", "S03", "S04", "S05", "S06"],          # ~24 s continuous
    "A_all7": ["S01", "S02", "S03", "S04", "S05", "S06", "S07"],  # + S07 spliced over the flub
    "B1": ["S10"],
    "B3": ["S08", "S09", "S10"],
    "B_mid": ["S10", "S11", "S12", "S13"],      # ~15 s continuous, fold-B twin of A_mid
    "B_all": ["S08", "S09", "S10", "S11", "S12", "S13", "S14"],
    "ALL": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"],
}
GAP = 0.45

def load(name):
    x, sr = sf.read(os.path.join(EDIT, name), dtype="float32")
    return librosa.resample(x, orig_sr=sr, target_sr=SR)

def energy_db(x):
    hop = SR // 100; n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)

def snap(d, t, lo, hi):
    i0, i1 = max(int(lo * 100), 0), min(int(hi * 100), len(d) - 1)
    return (i0 + int(np.argmin(d[i0:i1 + 1]))) / 100

def fade(y, ms=8):
    n = int(SR * ms / 1000); w = np.linspace(0, 1, n)
    y = y.copy(); y[:n] *= w; y[-n:] *= w[::-1]; return y

def main():
    spec = json.load(open(os.path.join(VO, "real", "clips.json")))
    clips = spec["clips"]; ids = list(clips)
    voice, raw = load("voice.wav"), load("raw44.wav")
    d = energy_db(voice)
    cut = {}
    for k, c in clips.items():
        i = ids.index(k)
        prev_e = clips[ids[i - 1]]["e"] if i else 0
        next_s = clips[ids[i + 1]]["s"] if i + 1 < len(ids) else len(voice) / SR
        if k == "S07": prev_e = FLUB[1] + 0.55       # 'Third is,' ends at 29.36
        if k == "S06": next_s = FLUB[0]
        s = snap(d, c["s"], max(c["s"] - PAD, (prev_e + c["s"]) / 2), c["s"] + 0.01)
        e = snap(d, c["e"], c["e"] - 0.01, min(c["e"] + PAD, (c["e"] + next_s) / 2))
        s, e = max(s - LEAD, (prev_e + s) / 2 if i else 0), min(e + TAIL, (e + next_s) / 2)
        if k == "S07": s = max(s, 29.38)
        if k == "S06": e = min(e, FLUB[0])
        cut[k] = dict(s=round(s, 3), e=round(e, 3), text=c["text"])
        for src, suf in ((voice, ""), (raw, ".raw")):
            sf.write(os.path.join(VO, "real", f"{k}{suf}.wav"), fade(src[int(s * SR):int(e * SR)]), SR, subtype="PCM_24")
    json.dump(cut, open(os.path.join(VO, "real", "clips.cut.json"), "w"), indent=1)

    os.makedirs(os.path.join(VO, "refs"), exist_ok=True)
    room = voice[int(0.1 * SR):int(0.1 * SR) + int(GAP * SR)]   # the take's own room tone, before the first word
    meta = {}
    for name, seq in REFS.items():
        parts, texts, runs = {"": [], ".raw": []}, [], [[seq[0]]]
        for k in seq[1:]:
            if ids.index(k) == ids.index(runs[-1][-1]) + 1 and k != "S07": runs[-1].append(k)
            else: runs.append([k])
        for j, run in enumerate(runs):
            s, e = cut[run[0]]["s"], cut[run[-1]]["e"]
            for src, suf in ((voice, ""), (raw, ".raw")):
                if j: parts[suf].append(room if suf == "" else raw[int(0.1 * SR):int(0.1 * SR) + len(room)])
                parts[suf].append(fade(src[int(s * SR):int(e * SR)]))
            texts += [cut[k]["text"] for k in run]
        for suf, p in parts.items():
            sf.write(os.path.join(VO, "refs", f"{name}{suf}.wav"), np.concatenate(p), SR, subtype="PCM_24")
        meta[name] = dict(audio=f"{name}.wav", sentences=seq, text=" ".join(texts),
                          seconds=round(len(np.concatenate(parts[""])) / SR, 2))
        print(f"{name:7s} {meta[name]['seconds']:5.1f}s  {' '.join(seq)}")
    json.dump(meta, open(os.path.join(VO, "refs", "refs.json"), "w"), indent=1)
    for k, c in cut.items(): print(k, c["s"], c["e"], round(c["e"] - c["s"], 2))

if __name__ == "__main__":
    main()
