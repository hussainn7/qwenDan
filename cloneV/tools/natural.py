# Naturalness scores, the "does it sound robotic" half that score.py (likeness) doesn't measure.
#   .venv-score/bin/python tools/natural.py OUT.json WAV [WAV ...]
# Per take:
#   mos    SQUIM subjective MOS (1-5), judged against 3 real sentences as non-matching references
#   stoi/pesq/sisdr  SQUIM objective, reference-free (noise and distortion)
#   gv     spectral "global variance" vs the real takes, dB (negative = over-smoothed, the classic buzzy-TTS sign)
#   dyn    std of frame loudness over speech, dB (flat = monotone energy)
#   jit    median frame-to-frame F0 change, semitones (micro pitch movement; ~0 = robotic)
# Cached by path + mtime in OUT.json, so re-runs only score new files.
import json, os, sys, glob
import numpy as np, librosa, torch, torchaudio, parselmouth
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = sorted(glob.glob(os.path.join(HERE, 'real', 'S??.wav')))
NMR = [os.path.join(HERE, 'real', f'{s}.wav') for s in ('S03', 'S09', 'S12')]
SR = 16000

subj = SQUIM_SUBJECTIVE.get_model().eval()
obj = SQUIM_OBJECTIVE.get_model().eval()


def load(p, sr=SR):
    return librosa.load(p, sr=sr, mono=True)[0].astype(np.float32)


nmr = [torch.from_numpy(load(p))[None] for p in NMR]


def logmel(y, sr=24000):
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80, fmin=60, fmax=11000)
    lm = np.log(m + 1e-8)
    e = lm.mean(0)
    return lm[:, e > e.max() - 6]          # speech frames only (within ~26 dB of the loudest)


def frame_db(y, sr=24000):
    r = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    d = 20 * np.log10(r + 1e-6)
    return d[d > d.max() - 30]


def jitter(p):
    f0 = parselmouth.Sound(p).to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=400).selected_array['frequency']
    st = 12 * np.log2(np.where(f0 > 0, f0, np.nan) / 100)
    d = np.abs(np.diff(st))
    d = d[np.isfinite(d) & (d < 2)]      # within-voicing steps only
    return float(np.median(d)) if len(d) else float('nan')


_gv_real = None


def gv_real():
    global _gv_real
    if _gv_real is None:
        _gv_real = np.concatenate([logmel(load(p, 24000)) for p in REAL], 1).var(1)
    return _gv_real


@torch.no_grad()
def score(p):
    y = load(p)
    t = torch.from_numpy(y)[None]
    mos = float(np.mean([subj(t, r).item() for r in nmr]))
    stoi, pesq, sisdr = (v.item() for v in obj(t))
    y24 = load(p, 24000)
    gv = float(np.mean(10 * np.log10(logmel(y24).var(1) / gv_real())))
    return dict(mos=round(mos, 3), stoi=round(stoi, 3), pesq=round(pesq, 3), sisdr=round(sisdr, 2),
                gv=round(gv, 2), dyn=round(float(frame_db(y24).std()), 2), jit=round(jitter(p), 3))


if __name__ == '__main__':
    out, wavs = sys.argv[1], sys.argv[2:]
    cache = json.load(open(out)) if os.path.exists(out) else {}
    for i, p in enumerate(wavs):
        key = os.path.relpath(os.path.abspath(p), HERE)
        mt = os.path.getmtime(p)
        if key in cache and cache[key].get('mtime') == mt:
            continue
        cache[key] = dict(score(p), mtime=mt)
        if i % 25 == 0:
            print(f'{i}/{len(wavs)} {key} {cache[key]}', flush=True)
            json.dump(cache, open(out, 'w'), indent=0)
    json.dump(cache, open(out, 'w'), indent=0)
    print('done', len(cache))
