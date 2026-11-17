# Score the mood_test.py takes: likeness to the user, word errors, and whether the reference's mood carried over.
#   .venv-score/bin/python tools/mood_score.py [--lines FILE]
# likeness: 0 = Derek, 100 = the user's own setup-1 sentences against a centroid built without their take
#   (ECAPA, WavLM-SV and Resemblyzer rescaled between those points, averaged).
# delivery: tools/delivery.py measures (F0 median, range, movement, words/s, loudness swing), per take.
# Writes sweep/mood_scores.json; prints per reference and per mood, and how much each output measure follows its
# reference (correlation across references).
import json, os, sys, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import Embedders, cos, DEREK
from delivery import measure

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
LINES = None
if '--lines' in args:
    LINES = json.load(open(args[args.index('--lines') + 1]))
else:
    from mood_test import LINES
setups = json.load(open(os.path.join(VO, 'data', 'setups.json')))
rows = [json.loads(l) for l in open(os.path.join(VO, 'data', 'utt.jsonl'))]
rows = [r for r in rows if setups.get(r['take']) == 1]
E = Embedders()
emb = {r['audio']: E(os.path.join(VO, r['audio'])) for r in rows}
unit = lambda v: v / np.linalg.norm(v)
cen = [unit(np.mean([unit(e[k]) for e in emb.values()], 0)) for k in range(3)]
# ceiling: each sentence vs the centroid of the other takes
ceil = []
for t in sorted({r['take'] for r in rows}):
    rest = [emb[r['audio']] for r in rows if r['take'] != t]
    c = [unit(np.mean([unit(e[k]) for e in rest], 0)) for k in range(3)]
    ceil += [[cos(emb[r['audio']][k], c[k]) for k in range(3)] for r in rows if r['take'] == t]
ceil = np.mean(ceil, 0)
floor = np.mean([[cos(E(p)[k], cen[k]) for k in range(3)] for p in DEREK], 0)
like = lambda e: float(100 * np.mean([(cos(e[k], cen[k]) - floor[k]) / (ceil[k] - floor[k]) for k in range(3)]))


def wer(ref, hyp):
    n = lambda t: re.sub(r"[^a-z0-9' ]", ' ', t.lower().replace('-', ' ')).split()
    r, h = n(ref), n(hyp)
    d = np.arange(len(h) + 1)
    for i, w in enumerate(r, 1):
        prev, d[0] = d.copy(), i
        for k in range(1, len(h) + 1):
            d[k] = min(prev[k] + 1, d[k - 1] + 1, prev[k - 1] + (w != h[k - 1]))
    return d[-1] / max(1, len(r))


moods = json.load(open(os.path.join(VO, 'refs', 'moods.json')))
dl = json.load(open(os.path.join(VO, 'data', 'delivery.json')))
path = os.path.join(VO, 'sweep', 'mood_scores.json')
res = json.load(open(path)) if os.path.exists(path) else {}
for d in sorted(glob.glob(os.path.join(VO, 'sweep', 'takes', 'mood_*'))):
    rid = os.path.basename(d)[5:]
    for p in sorted(glob.glob(os.path.join(d, 'T*-s*.wav')) + glob.glob(os.path.join(d, 'V*-s*.wav'))):
        k = os.path.relpath(p, VO)
        if k in res or not os.path.exists(p[:-4] + '.whisper.json'): continue
        meta = json.load(open(p[:-4] + '.json'))
        hyp = json.load(open(p[:-4] + '.whisper.json'))['text']
        m = measure(p, meta['text'])
        m.pop('ltas')
        res[k] = dict(m, ref=rid, mood=meta['mood'], line=meta['line'], like=like(E(p)), wer=wer(meta['text'], hyp))
E.save()
json.dump(res, open(path, 'w'), indent=0)

keys = ['f0', 'rng', 'mov', 'wps', 'loud']
by = {}
for k, v in res.items():
    if LINES and v['line'] not in LINES: continue
    by.setdefault(v['ref'], []).append(v)
ref_feat = {rid: dl.get(f"refs/{moods[rid]['audio']}") or dl.get(moods[rid]['source']) for rid in moods}
base = json.load(open(os.path.join(VO, 'refs', 'refs.json')))
for b in ('B3', 'A3'): ref_feat[b] = dl.get(f"refs/{base[b]['audio']}")
print(f"likeness scale: 0 = Derek, 100 = your own sentences (ceiling cos {np.round(ceil, 3).tolist()})")
print(f"{'ref':13s} {'mood':8s} {'n':>2s} {'like':>6s} {'WER%':>5s} | out: {'F0':>5s} {'rng':>5s} {'mov':>5s} {'w/s':>4s} {'loud':>4s} | ref: {'F0':>5s} {'rng':>5s} {'w/s':>4s}")
order = sorted(by, key=lambda r: (['current', 'hype', 'hook', 'story', 'explain', 'serious'].index(by[r][0]['mood']), r))
for rid in order:
    v = by[rid]; a = lambda k: np.mean([x[k] for x in v]); rf = ref_feat.get(rid) or {}
    print(f"{rid:13s} {v[0]['mood']:8s} {len(v):2d} {a('like'):6.1f} {100 * a('wer'):5.1f} | out: {a('f0'):5.0f} {a('rng'):5.1f} {a('mov'):5.2f} "
          f"{a('wps'):4.1f} {a('loud'):4.1f} | ref: {rf.get('f0', 0):5.0f} {rf.get('rng', 0):5.1f} {rf.get('wps', 0):4.1f}")
print('\nhow much the output follows the reference (correlation across references):')
for k in keys:
    xs = [ref_feat[r][k] for r in order if ref_feat.get(r)]
    ys = [np.mean([x[k] for x in by[r]]) for r in order if ref_feat.get(r)]
    print(f'  {k:5s} r = {np.corrcoef(xs, ys)[0, 1]:+.2f}')
