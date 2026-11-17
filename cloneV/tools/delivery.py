# How the user delivers each sentence, measured the same way as cram-calc/vo/energy.py (which matched "excited" by
# ear): pitch level (median F0), pitch range (5-95 % semitones), pitch movement (std, semitones), pace (words/s),
# loudness swing (std of voiced-frame dB), plus a channel signature (1/3-octave long-term spectrum) to tell
# recording setups apart.
#   .venv-score/bin/python tools/delivery.py [WAV ...]        (no args: data/utt.jsonl + real/S??.wav + refs/*.wav)
# Writes data/delivery.json {path: {f0, rng, mov, wps, loud, dur, ltas}}; cached by path + mtime.
import json, os, sys, glob, re
import numpy as np, librosa, parselmouth

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(VO, 'data', 'delivery.json')
BANDS = 125 * 2 ** (np.arange(0, 19) / 3)          # 125 Hz .. 8 kHz, 1/3 octave


def nwords(p):
    for j in (p[:-4] + '.whisper.json', p + '.whisper.json'):
        if os.path.exists(j):
            return len(json.load(open(j))['words']), json.load(open(j))['words']
    return None, None


def measure(p, text=None):
    y, sr = librosa.load(p, sr=16000)
    snd = parselmouth.Sound(y, sr)
    f0 = snd.to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=400).selected_array['frequency']
    f = f0[f0 > 0]
    st = 12 * np.log2(f / np.median(f))
    rms = librosa.feature.rms(y=y, frame_length=640, hop_length=160)[0]
    db = 20 * np.log10(rms + 1e-9)
    sp = db > db.max() - 30
    voiced = np.zeros(len(db), bool)
    voiced[:min(len(f0), len(db))] = (f0[:len(db)] > 0)
    on = np.nonzero(sp)[0]
    span = (on[-1] - on[0]) * 0.01 if len(on) > 1 else len(y) / sr
    n = len(re.findall(r"[A-Za-z0-9']+", text)) if text else (nwords(p)[0] or 0)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    e = librosa.feature.rms(S=np.sqrt(S))[0]
    S = S[:, e > e.max() * 10 ** (-30 / 20)]
    lt = np.array([10 * np.log10(S[(fr >= b / 2 ** (1 / 6)) & (fr < b * 2 ** (1 / 6))].mean() + 1e-12) for b in BANDS])
    return dict(f0=float(np.median(f)), rng=float(np.percentile(st, 95) - np.percentile(st, 5)), mov=float(st.std()),
                wps=n / span, loud=float(db[voiced & sp].std()), dur=len(y) / sr, ltas=(lt - lt.mean()).round(2).tolist())


if __name__ == '__main__':
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs = []
    if sys.argv[1:]:
        jobs = [(p, None) for p in sys.argv[1:]]
    else:
        jobs += [(os.path.join(VO, json.loads(l)['audio']), json.loads(l)['text']) for l in open(os.path.join(VO, 'data', 'utt.jsonl'))]
        clips = json.load(open(os.path.join(VO, 'real', 'clips.json')))['clips']
        jobs += [(os.path.join(VO, 'real', f'{s}.wav'), c['text']) for s, c in clips.items()]
        refs = json.load(open(os.path.join(VO, 'refs', 'refs.json')))
        jobs += [(os.path.join(VO, 'refs', r['audio']), r['text']) for r in refs.values()]
    for i, (p, t) in enumerate(jobs):
        k = os.path.relpath(p, VO)
        mt = os.path.getmtime(p)
        if k in res and res[k].get('mtime') == mt: continue
        res[k] = dict(measure(p, t), mtime=mt)
        if i % 50 == 0: print(i, len(jobs), k, {a: round(b, 2) for a, b in res[k].items() if a not in ('ltas', 'mtime')}, flush=True)
    json.dump(res, open(OUT, 'w'))
    print('done', len(res))
