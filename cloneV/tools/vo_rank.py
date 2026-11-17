# Rank voiceover takes: sounds like the user, says the words, carries the mood.
#   .venv-score/bin/python tools/vo_rank.py JOBS.json      JOBS = [{"line": id, "path": wav, "text": str, "target": {f0, rng, mov, wps, loud} | null}]
# Needs <wav>.whisper.json next to each take. Writes JOBS.ranked.json {line: [take, ...] best first}, each take with
#   like  0 = Derek, 100 = the user's own sentences (setup-1 centroid; ECAPA, WavLM-SV, Resemblyzer)
#   wer   Whisper word error rate against the text
#   mood  mean |z| distance of the take's delivery from the target (the mood reference), in the user's own spread
#   score like - 150 * wer - 8 * mood
import json, os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import Embedders, cos, DEREK
from delivery import measure

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ['f0', 'rng', 'mov', 'wps', 'loud']
unit = lambda v: v / np.linalg.norm(v)


def norm_words(t):
    t = t.lower().replace('-', ' ')
    return re.sub(r"[^a-z0-9' ]", ' ', t).split()


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = np.arange(len(h) + 1)
    for i, w in enumerate(r, 1):
        prev, d[0] = d.copy(), i
        for k in range(1, len(h) + 1):
            d[k] = min(prev[k] + 1, d[k - 1] + 1, prev[k - 1] + (w != h[k - 1]))
    return d[-1] / max(1, len(r))


class Likeness:
    def __init__(self):
        setups = json.load(open(os.path.join(VO, 'data', 'setups.json')))
        rows = [json.loads(l) for l in open(os.path.join(VO, 'data', 'utt.jsonl'))]
        rows = [r for r in rows if setups.get(r['take']) == 1]
        self.E = E = Embedders()
        emb = [E(os.path.join(VO, r['audio'])) for r in rows]
        self.cen = [unit(np.mean([unit(e[k]) for e in emb], 0)) for k in range(3)]
        self.ceil = np.mean([[cos(e[k], self.cen[k]) for k in range(3)] for e in emb], 0)
        self.floor = np.mean([[cos(E(p)[k], self.cen[k]) for k in range(3)] for p in DEREK], 0)
        dl = json.load(open(os.path.join(VO, 'data', 'delivery.json')))
        A = np.array([[dl[r['audio']][k] for k in KEYS] for r in rows])
        self.sd = A.std(0)

    def __call__(self, p):
        e = self.E(p)
        return float(100 * np.mean([(cos(e[k], self.cen[k]) - self.floor[k]) / (self.ceil[k] - self.floor[k]) for k in range(3)]))


if __name__ == '__main__':
    jobs = json.load(open(sys.argv[1]))
    L = Likeness()
    out = {}
    for j in jobs:
        hyp = json.load(open(j['path'][:-4] + '.whisper.json'))['text']
        m = measure(j['path'], j['text']); m.pop('ltas')
        t = j.get('target')
        mood = float(np.mean([abs(m[k] - t[k]) / L.sd[i] for i, k in enumerate(KEYS)])) if t else 0.0
        r = dict(path=j['path'], like=round(L(j['path']), 1), wer=round(wer(j['text'], hyp), 3), mood=round(mood, 2),
                 heard=hyp, **{k: round(m[k], 2) for k in KEYS})
        r['score'] = round(r['like'] - 150 * r['wer'] - 8 * r['mood'], 1)
        out.setdefault(j['line'], []).append(r)
    for k in out: out[k].sort(key=lambda r: -r['score'])
    L.E.save()
    json.dump(out, open(sys.argv[1][:-5] + '.ranked.json', 'w'), indent=1)
    for k, v in out.items():
        b = v[0]
        print(f"{k}: best {os.path.basename(b['path'])} score {b['score']} like {b['like']} wer {b['wer']} mood {b['mood']}")
