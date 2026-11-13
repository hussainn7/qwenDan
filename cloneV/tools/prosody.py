# Delivery closeness: does a generated sentence move the way the user said that same sentence?
#   .venv-score/bin/python tools/prosody.py CONFIG [CONFIG ...]      (configs = dirs in sweep/takes; "real" = self-check)
# Per take, against real/<sid>.wav of the same sentence:
#   f0r   correlation of the semitone pitch contours after DTW alignment on MFCCs (voiced frames in both)
#   enr   correlation of the loudness contours along the same path (stress pattern)
#   durr  correlation of per-word durations (Whisper word timings; only when the word lists match)
#   gap   mean |pause difference| after each word, ms (where the user paused and for how long)
# Writes sweep/prosody.json and prints the per-config means.
import json, os, sys, glob, re
import numpy as np, librosa, parselmouth

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR, HOP = 16000, 160


def feats(p):
    y = librosa.load(p, sr=SR)[0]
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, hop_length=HOP, n_fft=512)[1:]
    m = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-6)
    f0 = parselmouth.Sound(y, SR).to_pitch(time_step=HOP / SR, pitch_floor=70, pitch_ceiling=400).selected_array['frequency']
    n = m.shape[1]
    f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
    st = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1) / 100), np.nan)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=HOP)[0][:n]
    db = 20 * np.log10(rms + 1e-6)
    keep = db > db.max() - 35                       # trim silence so DTW aligns speech to speech
    idx = np.nonzero(keep)[0]
    sl = slice(idx[0], idx[-1] + 1)
    return m[:, sl], st[sl], db[sl]


def words(p):
    j = p[:-4] + '.whisper.json'
    if not os.path.exists(j): return None
    return [(re.sub(r'[^a-z0-9]', '', w['w'].lower()), w['s'], w['e']) for w in json.load(open(j))['words']]


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10: return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


_real = {}


def real(sid):
    if sid not in _real:
        p = os.path.join(VO, 'real', f'{sid}.wav')
        _real[sid] = (feats(p), words(p))
    return _real[sid]


def score(p, sid):
    (rm, rst, rdb), rw = real(sid)
    m, st, db = feats(p)
    _, wp = librosa.sequence.dtw(X=rm, Y=m, metric='euclidean')
    wp = wp[::-1]
    a, b = wp[:, 0], wp[:, 1]
    out = dict(f0r=corr(rst[a] - np.nanmean(rst), st[b] - np.nanmean(st)), enr=corr(rdb[a], db[b]))
    tw = words(p)
    if rw and tw and [w for w, _, _ in rw] == [w for w, _, _ in tw] and len(rw) > 3:
        rd = np.array([e - s for _, s, e in rw]); td = np.array([e - s for _, s, e in tw])
        out['durr'] = corr(rd / rd.sum(), td / td.sum())
        rg = np.array([rw[i + 1][1] - rw[i][2] for i in range(len(rw) - 1)])
        tg = np.array([tw[i + 1][1] - tw[i][2] for i in range(len(tw) - 1)])
        out['gap'] = float(np.mean(np.abs(rg - tg)) * 1000)
    return out


if __name__ == '__main__':
    path = os.path.join(VO, 'sweep', 'prosody.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    for cfg in sys.argv[1:]:
        rows = []
        for p in sorted(glob.glob(os.path.join(VO, 'sweep', 'takes', cfg, 'S??-s*.wav'))):
            key = os.path.relpath(p, VO)
            if key not in res:
                res[key] = score(p, os.path.basename(p)[:3])
            rows.append(res[key])
        if not rows: continue
        mean = lambda k: np.nanmean([r.get(k, np.nan) for r in rows])
        print(f"{cfg:24s} n={len(rows):3d}  f0r {mean('f0r'):.3f}  enr {mean('enr'):.3f}  durr {mean('durr'):.3f}  gap {mean('gap'):5.0f} ms")
    json.dump(res, open(path, 'w'), indent=0)
