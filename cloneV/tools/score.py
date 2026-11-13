"""Scores cloned takes against the real held-out sentences of the take.
  vo/.venv-score/bin/python vo/tools/score.py [config ...]      (all of vo/sweep/takes/* by default)
Needs <take>.whisper.json from tools/transcribe.py. Writes vo/sweep/scores.json (per take) and prints a table per
config. Real-vs-real rows give the ceiling (same voice, different sentence) and the Derek row a different-voice floor.

Per take:
  close   0 = a different voice (Derek), 100 = the user's real sentences: each model's sim rescaled between those two
          references, then averaged over the three models.
  sim_*   cosine of the take's speaker embedding to the centroid of the held-out real sentences of its fold, leaving
          out the one with the same words. ecapa = speechbrain spkrec-ecapa-voxceleb, wavlm = microsoft/wavlm-base-plus-sv,
          resem = Resemblyzer. sim = mean of ecapa and wavlm.
  wer     Whisper large-v3-turbo vs the intended text (lowercase, no punctuation, numbers as digits).
  f0_med / f0_rng   median F0 (Hz) and 5-95 % range (semitones), Praat autocorrelation 70-400 Hz.
  d_f0    F0 median minus the real sentence's, semitones. rate = real speech span / take speech span (Whisper word edges),
          >1 means faster than the user said it.  mcd = DTW mel-cepstral distance to the real sentence (dB, MFCC 1-24 of
          an 80-band mel; only comparable within this table).
  flags   clip (samples at full scale), gap (a pause >0.3 s longer than the real sentence's longest), tail (sound >0.4 s
          after the last word), rep (a word repeated that the text doesn't repeat), wer>15 %."""
import glob, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, torch
import parselmouth

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = json.load(open(os.path.join(VO, "real", "clips.json")))
CACHE = os.path.join(VO, "sweep", "emb_cache.npz")
DEREK = ["/Users/hussain/animations/kit/voice/ref/hook.wav", "/Users/hussain/animations/kit/voice/ref/body.wav",
         "/Users/hussain/animations/kit/voice/ref/turn.wav", "/Users/hussain/animations/kit/voice/ref/end.wav"]


def load16(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    return librosa.resample(x.mean(1), orig_sr=sr, target_sr=16000)


class Embedders:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import AutoFeatureExtractor, WavLMForXVector
        from resemblyzer import VoiceEncoder
        self.ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir=os.path.expanduser("~/.cache/speechbrain/ecapa"))
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        self.resem = VoiceEncoder(device="cpu", verbose=False)
        self.cache = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
        self.dirty = False

    def __call__(self, p):
        key = f"{p}|{os.path.getmtime(p)}"
        if key not in self.cache:
            x = load16(p)
            with torch.no_grad():
                e = self.ecapa.encode_batch(torch.from_numpy(x)[None]).squeeze().numpy()
                inp = self.fe(x, sampling_rate=16000, return_tensors="pt")
                w = self.wavlm(**inp).embeddings.squeeze().numpy()
            from resemblyzer import preprocess_wav
            r = self.resem.embed_utterance(preprocess_wav(x, source_sr=16000))
            self.cache[key] = np.array([e / np.linalg.norm(e), w / np.linalg.norm(w), r / np.linalg.norm(r)], dtype=object)
            self.dirty = True
        return self.cache[key]

    def save(self):
        if self.dirty: np.savez(CACHE, **self.cache)


def cos(a, b): return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def centroid(embs):  # list of [ecapa, wavlm, resem]
    return [np.mean([e[i] for e in embs], 0) for i in range(3)]


NUM = {"twenty eleven": "2011", "twenty ten": "2010", "four s": "4s", "l l m": "llm"}

def norm(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = " ".join(t.split())
    for k, v in NUM.items(): t = t.replace(k, v)
    return t.split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def pitch(p):
    snd = parselmouth.Sound(p)
    f = snd.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=400).selected_array["frequency"]
    f = f[f > 0]
    if len(f) < 10: return np.nan, np.nan
    st = 12 * np.log2(f / np.median(f))
    f = f[np.abs(st) < 9]                          # drop octave jumps
    st = 12 * np.log2(f / np.median(f))
    return float(np.median(f)), float(np.percentile(st, 95) - np.percentile(st, 5))


def mfcc(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = librosa.resample(x.mean(1), orig_sr=sr, target_sr=22050)
    x, _ = librosa.effects.trim(x, top_db=35)
    return librosa.feature.mfcc(y=x, sr=22050, n_mfcc=25, n_fft=1024, hop_length=256, n_mels=80)[1:]


def mcd(a, b):
    D, wp = librosa.sequence.dtw(a, b, metric="euclidean")
    diff = a[:, wp[:, 0]] - b[:, wp[:, 1]]          # librosa cepstra are in dB (20/ln10 x natural-log cepstra)
    return float(np.mean(0.5 * np.sqrt(2 * (diff ** 2).sum(0))))


def words(p):
    j = json.load(open(p[:-4] + ".whisper.json"))
    return j["text"], j["words"]


def tail_energy(p, after):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1)[int((after + 0.4) * sr):]
    if len(x) < sr * 0.05: return False
    hop = sr // 100; n = len(x) // hop
    return bool((20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9) > -38).sum() > 8)


def real_stats(E):
    out = {}
    for sid in CLIPS["clips"]:
        p = os.path.join(VO, "real", f"{sid}.wav")
        t, w = words(p)
        out[sid] = dict(emb=E(p), f0=pitch(p), span=w[-1]["e"] - w[0]["s"], maxgap=max([b["s"] - a["e"] for a, b in zip(w, w[1:])] or [0]),
                        mfcc=mfcc(p), wer=wer(CLIPS["clips"][sid]["text"], t))
    return out


def fold_of(sid):
    return next(f for f, v in CLIPS["folds"].items() if sid in v["test"])


def score_take(p, meta, R, E):
    sid = meta["sentence"]; fold = CLIPS["folds"][fold_of(sid)]["test"]
    cen = centroid([R[s]["emb"] for s in fold if s != sid])
    e = E(p); text, w = words(p)
    s = dict(sim_ecapa=cos(e[0], cen[0]), sim_wavlm=cos(e[1], cen[1]), sim_resem=cos(e[2], cen[2]))
    s["sim"] = (s["sim_ecapa"] + s["sim_wavlm"]) / 2
    s["sim_same"] = (cos(e[0], R[sid]["emb"][0]) + cos(e[1], R[sid]["emb"][1])) / 2
    s["wer"] = wer(meta["text"], text)
    f0, rng = pitch(p); s["f0_med"], s["f0_rng"] = f0, rng
    s["d_f0"] = 12 * np.log2(f0 / R[sid]["f0"][0]) if f0 == f0 else np.nan
    span = (w[-1]["e"] - w[0]["s"]) if w else np.nan
    s["rate"] = R[sid]["span"] / span if span else np.nan
    s["mcd"] = mcd(mfcc(p), R[sid]["mfcc"])
    x, _ = sf.read(p, dtype="float32", always_2d=True)
    flags = []
    if (np.abs(x) >= 0.99).sum() > 3: flags.append("clip")
    gaps = [b["s"] - a["e"] for a, b in zip(w, w[1:])]
    if gaps and max(gaps) > R[sid]["maxgap"] + 0.3: flags.append("gap")
    if w and tail_energy(p, w[-1]["e"]): flags.append("tail")
    nt = norm(meta["text"]); nh = norm(text)
    if any(a == b for a, b in zip(nh, nh[1:])) and not any(a == b for a, b in zip(nt, nt[1:])): flags.append("rep")
    if s["wer"] > 0.15: flags.append("wer")
    s["flags"] = flags; s["text_heard"] = text; s["dur"] = len(x) / 24000
    return s


SCALE = {}   # per model: (floor, ceiling) mean similarity, set in main


def close(r):
    """0 = a different voice (Derek), 100 = the user's own real sentences, averaged over the three speaker models."""
    return 100 * np.mean([(r[k] - SCALE[k][0]) / (SCALE[k][1] - SCALE[k][0]) for k in ("sim_ecapa", "sim_wavlm", "sim_resem")])


def fmt_row(name, rows, R=None):
    a = lambda k: np.nanmean([r[k] for r in rows])
    nflag = sum(bool(r["flags"]) for r in rows)
    return (f"{name:16s} {len(rows):3d}  {np.mean([close(r) for r in rows]):5.1f}  {a('sim'):.3f}  {a('sim_ecapa'):.3f} {a('sim_wavlm'):.3f} {a('sim_resem'):.3f}  "
            f"{a('sim_same'):.3f}  {100 * a('wer'):5.1f}  {a('f0_med'):5.0f} {a('f0_rng'):5.1f} {np.nanmean([abs(r['d_f0']) for r in rows]):5.2f}  "
            f"{a('rate'):4.2f}  {a('mcd'):5.2f}  {nflag:2d}")


HEAD = (f"{'config':16s} {'n':>3s}  {'close':>5s}  {'SIM':5s}  {'ecapa':5s} {'wavlm':5s} {'resem':5s}  {'same':5s}  {'WER%':>5s}  "
        f"{'F0Hz':>5s} {'rngST':>5s} {'|dF0|':>5s}  {'rate':4s}  {'MCD':>5s}  {'fl':>2s}")


def main():
    E = Embedders(); R = real_stats(E)
    # ceiling: each real held-out sentence vs the other real sentences of its fold
    ceil = []
    for sid, r in R.items():
        cen = centroid([R[s]["emb"] for s in CLIPS["folds"][fold_of(sid)]["test"] if s != sid])
        sc = dict(sim_ecapa=cos(r["emb"][0], cen[0]), sim_wavlm=cos(r["emb"][1], cen[1]), sim_resem=cos(r["emb"][2], cen[2]),
                  sim_same=1.0, wer=r["wer"], f0_med=r["f0"][0], f0_rng=r["f0"][1], d_f0=0.0, rate=1.0, mcd=0.0, flags=[])
        sc["sim"] = (sc["sim_ecapa"] + sc["sim_wavlm"]) / 2; sc["sentence"] = sid; ceil.append(sc)
    floor = []
    for p in DEREK:
        e = E(p)
        for f in CLIPS["folds"].values():
            cen = centroid([R[s]["emb"] for s in f["test"]])
            sc = dict(sim_ecapa=cos(e[0], cen[0]), sim_wavlm=cos(e[1], cen[1]), sim_resem=cos(e[2], cen[2]), sim_same=np.nan,
                      wer=np.nan, f0_med=pitch(p)[0], f0_rng=pitch(p)[1], d_f0=np.nan, rate=np.nan, mcd=np.nan, flags=[])
            sc["sim"] = (sc["sim_ecapa"] + sc["sim_wavlm"]) / 2; floor.append(sc)
    for k in ("sim_ecapa", "sim_wavlm", "sim_resem"):
        SCALE[k] = (np.mean([f[k] for f in floor]), np.mean([c[k] for c in ceil]))
    want = sys.argv[1:] or sorted(os.listdir(os.path.join(VO, "sweep", "takes")))
    out_path = os.path.join(VO, "sweep", "scores.json")
    allsc = json.load(open(out_path)) if os.path.exists(out_path) else {}
    print(HEAD)
    print(fmt_row("REAL (ceiling)", ceil)); print(fmt_row("REAL fold-B tst", [c for c in ceil if c["sentence"] in CLIPS["folds"]["A"]["ref_pool"]]))
    print(fmt_row("REAL fold-A tst", [c for c in ceil if c["sentence"] in CLIPS["folds"]["A"]["test"]]))
    print(fmt_row("Derek (floor)", floor))
    for cfg in want:
        rows = []
        for p in sorted(glob.glob(os.path.join(VO, "sweep", "takes", cfg, "*.wav"))):
            if not os.path.exists(p[:-4] + ".whisper.json"): continue
            meta = json.load(open(p[:-4] + ".json"))
            s = score_take(p, meta, R, E); s.update(config=cfg, sentence=meta["sentence"], seed=meta["seed"]); s["close"] = close(s)
            allsc[os.path.relpath(p, VO)] = s; rows.append(s)
        if rows: print(fmt_row(cfg, rows))
    E.save()
    json.dump(allsc, open(out_path, "w"), indent=0, default=float)


if __name__ == "__main__":
    main()
