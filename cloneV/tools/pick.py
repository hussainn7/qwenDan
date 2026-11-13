"""Ranks candidate takes of one line: speaker closeness to all 14 real sentences (0 = a different voice, 100 = the
user's real take, same scale as score.py), minus penalties for Whisper word errors and artifacts.
  vo/.venv-score/bin/python vo/tools/pick.py "<intended text>" cand1.wav cand2.wav ...   (needs .whisper.json each)
Prints one JSON line per take, best first."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf
import score as S

def main():
    text, takes = sys.argv[1], sys.argv[2:]
    E = S.Embedders()
    real = [E(os.path.join(S.VO, "real", f"{sid}.wav")) for sid in S.CLIPS["clips"]]
    cen = S.centroid(real)
    floor = [E(p) for p in S.DEREK]
    for i, k in enumerate(("sim_ecapa", "sim_wavlm", "sim_resem")):
        lo = np.mean([S.cos(f[i], cen[i]) for f in floor])
        hi = np.mean([S.cos(r[i], S.centroid([q for q in real if q is not r])[i]) for r in real])
        S.SCALE[k] = (lo, hi)
    rows = []
    for p in takes:
        e = E(p); heard, w = S.words(p)
        r = dict(take=p, sim_ecapa=S.cos(e[0], cen[0]), sim_wavlm=S.cos(e[1], cen[1]), sim_resem=S.cos(e[2], cen[2]),
                 wer=S.wer(text, heard), heard=heard)
        r["close"] = S.close(r)
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        flags = []
        if (np.abs(x) >= 0.99).sum() > 3: flags.append("clip")
        gaps = [b["s"] - a["e"] for a, b in zip(w, w[1:])]
        if gaps and max(gaps) > 0.7: flags.append("gap")
        if w and S.tail_energy(p, w[-1]["e"]): flags.append("tail")
        nh = S.norm(heard); nt = S.norm(text)
        if any(a == b for a, b in zip(nh, nh[1:])) and not any(a == b for a, b in zip(nt, nt[1:])): flags.append("rep")
        r["flags"] = flags
        r["rank"] = r["close"] - 150 * r["wer"] - 15 * len(flags)
        rows.append(r)
    E.save()
    for r in sorted(rows, key=lambda r: -r["rank"]):
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))

if __name__ == "__main__":
    main()
