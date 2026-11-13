# One table across configs: likeness (score.py), delivery (prosody.py) and naturalness (natural.py), per take.
#   .venv-score/bin/python tools/compare.py [--seeds 1 2] [--sentences S01 ...] NAME=CFG[+CFG...] ...
# e.g. tools/compare.py "ICL now=2A_3sub05+2B_3sub05" "ft pass 2=ft_r16_e2"
# Means are over single takes (no best-of-N), restricted to the given seeds and sentences, so rows are comparable.
import json, os, sys
import numpy as np

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda n: json.load(open(os.path.join(VO, 'sweep', n)))
S, P, N = J('scores.json'), J('prosody.json'), J('natural.json')
args = sys.argv[1:]


def opt(flag):
    if flag not in args: return None
    i = args.index(flag); vals = []
    for v in args[i + 1:]:
        if v.startswith('--') or '=' in v: break
        vals.append(v)
    del args[i:i + 1 + len(vals)]
    return vals


seeds = [int(s) for s in (opt('--seeds') or [])]
sents = opt('--sentences')
cols = [('close', S, 'close', '{:6.1f}'), ('ecapa', S, 'sim_ecapa', '{:6.3f}'), ('WER%', S, 'wer', '{:6.1f}'),
        ('rngST', S, 'f0_rng', '{:6.1f}'), ('|dF0|', S, 'd_f0', '{:6.2f}'), ('rate', S, 'rate', '{:6.2f}'),
        ('f0r', P, 'f0r', '{:6.3f}'), ('enr', P, 'enr', '{:6.3f}'), ('durr', P, 'durr', '{:6.3f}'),
        ('MOS', N, 'mos', '{:6.2f}'), ('SISDR', N, 'sisdr', '{:6.1f}')]
print(f"{'':26s} {'n':>3s} " + ' '.join(f'{c[0]:>6s}' for c in cols))
for a in args:
    name, cfgs = a.split('=', 1)
    keys = []
    for cfg in cfgs.split('+'):
        for k in S:
            parts = k.split('/')
            if len(parts) < 4 or parts[2] != cfg: continue
            sid, seed = parts[3][:3], int(parts[3].split('-s')[1][:-4])
            if seeds and seed not in seeds: continue
            if sents and sid not in sents: continue
            keys.append(k)
    if not keys:
        print(f'{name:26s}   0'); continue
    out = []
    for _, src, f, fmt in cols:
        v = [src.get(k, {}).get(f, np.nan) for k in keys]
        v = [abs(x) if f == 'd_f0' else x for x in v if x is not None]
        m = np.nanmean(v) if len(v) else np.nan
        out.append(fmt.format(100 * m if f == 'wer' else m))
    print(f'{name:26s} {len(keys):3d} ' + ' '.join(out))
