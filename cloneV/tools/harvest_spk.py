# Which source takes are the user talking? ECAPA cosine of 3 s speech windows vs the centroid of the 14 real Siri
# sentences. Writes data/speaker.json {name: {dur, speech_s, med, frac}}.
#   .venv-score/bin/python tools/harvest_spk.py data/src/*.wav
import json, os, sys, glob
import numpy as np, librosa, torch
from speechbrain.inference.speaker import EncoderClassifier

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'data', 'speaker.json')
enc = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                     savedir=os.path.expanduser('~/.cache/speechbrain/ecapa'), run_opts={'device': 'cpu'})


@torch.no_grad()
def emb(y):
    e = enc.encode_batch(torch.from_numpy(y)[None]).squeeze().numpy()
    return e / np.linalg.norm(e)


real = np.stack([emb(librosa.load(p, sr=16000)[0]) for p in sorted(glob.glob(os.path.join(HERE, 'real', 'S??.wav')))])
cen = real.mean(0); cen /= np.linalg.norm(cen)
print('real vs centroid', np.round(real @ cen, 3).tolist())

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for p in sys.argv[1:]:
    name = os.path.splitext(os.path.basename(p))[0]
    if name in res: continue
    y = librosa.load(p, sr=16000)[0]
    win = 48000
    rms = np.array([np.sqrt(np.mean(y[i:i + win] ** 2)) for i in range(0, len(y) - win + 1, win)])
    if not len(rms): continue
    loud = rms > max(rms.max() * 0.1, 1e-3)
    sims = [float(emb(y[i * win:(i + 1) * win]) @ cen) for i in np.nonzero(loud)[0][:120]]
    res[name] = dict(dur=round(len(y) / 16000, 1), speech_s=int(loud.sum()) * 3,
                     med=round(float(np.median(sims)), 3) if sims else None,
                     frac=round(float(np.mean(np.array(sims) > 0.45)), 2) if sims else 0)
    print(name, res[name], flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
