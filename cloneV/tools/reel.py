# A/B listening reel: for each sentence, the real reading, then each config's take, loudness-matched.
#   .venv-score/bin/python tools/reel.py OUT.wav --sentences S01 S05 ... --configs ICL=2A_3sub05+2B_3sub05 FT=ftx3_xvB3
# Per config and sentence the take is picked from seeds 1-2 by the same rule for everyone: fewest Whisper word
# errors, then highest closeness (scores.json). Lines without a real reading (new text) use --texts IDS instead of
# --sentences and skip the real slot. Writes OUT.wav and OUT.json (what plays when).
import json, os, sys, glob
import numpy as np, soundfile as sf, librosa

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 24000
S = json.load(open(os.path.join(VO, 'sweep', 'scores.json')))
LINES = {}   # --lines JSON {id: text}: new lines are scored here by Whisper word errors only


def wer_new(p, sid):
    import re
    j = p[:-4] + '.whisper.json'
    if sid not in LINES or not os.path.exists(j): return 0
    n = lambda t: re.sub(r"[^a-z' ]", ' ', t.lower()).split()
    r, h = n(LINES[sid]), n(json.load(open(j))['text'])
    d = np.arange(len(h) + 1)
    for i, w in enumerate(r, 1):
        prev, d[0] = d.copy(), i
        for k in range(1, len(h) + 1):
            d[k] = min(prev[k] + 1, d[k - 1] + 1, prev[k - 1] + (w != h[k - 1]))
    return d[-1] / max(1, len(r))


def opt(args, flag):
    i = args.index(flag); vals = []
    for v in args[i + 1:]:
        if v.startswith('--'): break
        vals.append(v)
    return vals


def level(y, db=-22.2):
    r = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    d = 20 * np.log10(r + 1e-9)
    cur = 10 * np.log10(np.mean(10 ** (d[d > d.max() - 30] / 10)))
    y = y * 10 ** ((db - cur) / 20)
    return y * min(1, 0.95 / (np.abs(y).max() + 1e-9))


def trim(y, keep=0.08):
    r = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    on = np.nonzero(20 * np.log10(r + 1e-9) > 20 * np.log10(r.max() + 1e-9) - 40)[0]
    a, b = max(0, on[0] * 128 - int(keep * SR)), min(len(y), on[-1] * 128 + 512 + int(keep * SR))
    return y[a:b]


def pick(cfgs, sid):
    cands = []
    for c in cfgs.split('+'):
        for seed in (1, 2):
            p = os.path.join(VO, 'sweep', 'takes', c, f'{sid}-s{seed}.wav')
            if not os.path.exists(p): continue
            s = S.get(os.path.relpath(p, VO), {})
            cands.append((s.get('wer', wer_new(p, sid)), -s.get('close', 0), p))
    return sorted(cands)[0][2] if cands else None


if __name__ == '__main__':
    args = sys.argv[1:]
    out = args[0]
    sids = opt(args, '--sentences') if '--sentences' in args else opt(args, '--texts')
    real = '--sentences' in args
    cfgs = [c.split('=', 1) for c in opt(args, '--configs')]
    if '--lines' in args: LINES.update(json.load(open(opt(args, '--lines')[0])))
    gap, between = np.zeros(int(0.5 * SR)), np.zeros(int(1.3 * SR))
    parts, log, t = [], [], 0.0
    for sid in sids:
        slots = ([('real', os.path.join(VO, 'real', f'{sid}.wav'))] if real else []) + [(n, pick(c, sid)) for n, c in cfgs]
        for i, (name, p) in enumerate(slots):
            if p is None: continue
            y = level(trim(librosa.load(p, sr=SR)[0]))
            log.append(dict(t=round(t, 2), sentence=sid, version=name, file=os.path.relpath(p, VO)))
            parts += [y, gap if i < len(slots) - 1 else between]
            t += (len(y) + len(parts[-1])) / SR
    sf.write(out, np.concatenate(parts), SR, subtype='PCM_16')
    json.dump(log, open(out[:-4] + '.json', 'w'), indent=1)
    print(f'{out}: {t:.1f} s, {len(log)} clips')
