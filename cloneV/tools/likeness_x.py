# Channel-fair likeness. score.py's centroid is the Siri take itself, which favours a clone that copies that
# recording's room and mic (ICL from a Siri clip). This one scores against the 3 held-out takes of the fine-tune
# (ft/RUN/run.json val_takes): other rooms, never trained on, never used as an ICL reference.
#   .venv-score/bin/python tools/likeness_x.py [--run r16] NAME=CFG[+CFG...] ...
# 0 = Derek, 100 = the user's real Siri sentences (a different recording of the user vs those same takes).
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import Embedders, cos, DEREK

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
run = 'r16'
if '--run' in args:
    i = args.index('--run'); run = args[i + 1]; del args[i:i + 2]
val = set(json.load(open(os.path.join(VO, 'ft', run, 'run.json')))['val_takes'])
rows = [json.loads(l) for l in open(os.path.join(VO, 'data', 'utt.jsonl'))]
E = Embedders()
ref = [E(os.path.join(VO, r['audio'])) for r in rows if r['take'] in val]
cen = [np.mean([e[k] / np.linalg.norm(e[k]) for e in ref], 0) for k in range(3)]


def sims(paths):
    return np.array([[cos(E(p)[k], cen[k]) for k in range(3)] for p in paths])


ceil = sims(sorted(glob.glob(os.path.join(VO, 'real', 'S??.wav')))).mean(0)
floor = sims(DEREK).mean(0)
scale = lambda s: 100 * np.mean((s - floor) / (ceil - floor), -1)
print(f"reference: {len(ref)} utterances from {len(val)} held-out takes")
print(f"{'':26s} {'n':>3s}  {'close':>6s}  {'ecapa':>6s} {'wavlm':>6s} {'resem':>6s}")
print(f"{'real Siri sentences':26s} {14:3d}  {100:6.1f}  " + ' '.join(f'{v:6.3f}' for v in ceil))
print(f"{'Derek':26s} {len(DEREK):3d}  {0:6.1f}  " + ' '.join(f'{v:6.3f}' for v in floor))
for a in args:
    name, cfgs = a.split('=', 1)
    ps = [p for c in cfgs.split('+') for p in sorted(glob.glob(os.path.join(VO, 'sweep', 'takes', c, 'S??-s[12].wav')))]
    if not ps: print(f'{name:26s}   0'); continue
    s = sims(ps)
    print(f"{name:26s} {len(ps):3d}  {scale(s).mean():6.1f}  " + ' '.join(f'{v:6.3f}' for v in s.mean(0)))
E.save()
