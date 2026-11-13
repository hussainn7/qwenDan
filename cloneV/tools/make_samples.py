"""A/B samples from the sweep: for each chosen sentence, the best takes (by score.py's held-out closeness, word errors
and flags) of the winning config of the fold where that sentence was held out, post-processed exactly like clone.py
(trim, loudness), next to the real sentence at the same loudness.
  ~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python vo/tools/make_samples.py <configA> <configB> S01 S05 S10 ... [--best 2]
-> vo/samples/<S>-real.wav, <S>-clone-<rank>.wav, <S>-AB.wav (real, pause, best clone), ab-reel.wav (all of them)
   and samples/index.json (scores of every kept take)."""
import json, os, sys
import numpy as np, soundfile as sf

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VO); sys.path.insert(0, os.path.join(VO, "tools"))
from clone import loudness_match, trim

SR = 24000


def main():
    args = sys.argv[1:]
    best = 2
    if "--best" in args:
        i = args.index("--best"); best = int(args[i + 1]); del args[i:i + 2]
    cfg = {"A": args[0], "B": args[1]}; sents = args[2:]
    settings = json.load(open(os.path.join(VO, "settings.json")))
    clips = json.load(open(os.path.join(VO, "real", "clips.json")))
    scores = json.load(open(os.path.join(VO, "sweep", "scores.json")))
    out_dir = os.path.join(VO, "samples"); os.makedirs(out_dir, exist_ok=True)
    pause = np.zeros(int(0.7 * SR), np.float32)
    reel, index = [], {}
    for sid in sents:
        fold = next(f for f, v in clips["folds"].items() if sid in v["test"])
        rows = [(k, v) for k, v in scores.items() if v["config"] == cfg[fold] and v["sentence"] == sid]
        rows.sort(key=lambda kv: -(kv[1]["close"] - 150 * kv[1]["wer"] - 15 * len(kv[1]["flags"])))
        real = loudness_match(sf.read(os.path.join(VO, "real", f"{sid}.wav"), dtype="float32")[0], settings["loudness_db"])
        sf.write(os.path.join(out_dir, f"{sid}-real.wav"), real, SR, subtype="PCM_24")
        index[sid] = dict(text=clips["clips"][sid]["text"], config=cfg[fold], takes=[])
        for rank, (k, v) in enumerate(rows[:best], 1):
            y = trim(sf.read(os.path.join(VO, k), dtype="float32")[0], SR)
            y = loudness_match(y, settings["loudness_db"])
            sf.write(os.path.join(out_dir, f"{sid}-clone-{rank}.wav"), y, SR, subtype="PCM_24")
            index[sid]["takes"].append(dict(file=f"{sid}-clone-{rank}.wav", source=k, seed=v["seed"],
                                            **{m: round(v[m], 3) for m in ("close", "sim_ecapa", "sim_wavlm", "sim_resem", "wer", "d_f0", "rate")},
                                            flags=v["flags"], heard=v["text_heard"]))
            if rank == 1:
                ab = np.concatenate([real, pause, y])
                sf.write(os.path.join(out_dir, f"{sid}-AB.wav"), ab, SR, subtype="PCM_24")
                reel += [ab, pause, pause]
        t = index[sid]["takes"][0]
        print(f"{sid} [{cfg[fold]}] best close {t['close']:.0f}  WER {100 * t['wer']:.0f}%  dF0 {t['d_f0']:+.1f} st  rate {t['rate']:.2f}  | {t['heard']}")
    sf.write(os.path.join(out_dir, "ab-reel.wav"), np.concatenate(reel), SR, subtype="PCM_24")
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
