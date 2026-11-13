"""Is --takes N worth it, judged honestly? Picks the best of N seeds per sentence using only audio the model could
know in practice (the fold's reference-pool real sentences as the speaker centroid, plus Whisper WER and flags), then
reports the held-out closeness (score.py's test-fold centroid) of the picked take vs the average take.
  vo/.venv-score/bin/python vo/tools/pick_eval.py 2A_3sub05 2B_3sub05 ..."""
import collections, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import score as S

E = S.Embedders()
scores = json.load(open(os.path.join(S.VO, "sweep", "scores.json")))
for cfg in sys.argv[1:]:
    rows = {k: v for k, v in scores.items() if v["config"] == cfg}
    fold = S.fold_of(next(iter(rows.values()))["sentence"])
    pool = S.centroid([E(os.path.join(S.VO, "real", f"{s}.wav")) for s in S.CLIPS["folds"][fold]["ref_pool"]])
    by = collections.defaultdict(list)
    for k, v in rows.items():
        e = E(os.path.join(S.VO, k))
        known = np.mean([S.cos(e[i], pool[i]) for i in range(2)])
        by[v["sentence"]].append((known * 100 - 150 * v["wer"] - 15 * len(v["flags"]), v["close"]))
    avg = np.mean([c for l in by.values() for _, c in l])
    picked = np.mean([max(l)[1] for l in by.values()])
    oracle = np.mean([max(c for _, c in l) for l in by.values()])
    print(f"{cfg:14s} average take {avg:5.1f}   picked of {len(next(iter(by.values())))} (honest) {picked:5.1f}   oracle {oracle:5.1f}")
