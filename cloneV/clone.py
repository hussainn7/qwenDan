"""A line of speech in the user's own voice: local Qwen3-TTS 1.7B Base clone of the Siri-vs-LLMs take, with the
reference clip, mode and sampling that scored closest in the sweep (vo/NOTES.md; settings in vo/settings.json).

  PY=~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python
  $PY vo/clone.py "Here's the thing nobody tells you about Siri." -o line.wav
  $PY vo/clone.py "..." -o line.wav --seed 7                 # other readings: seeds 7..10
  $PY vo/clone.py "..." -o line.wav --takes 1                # one seed only (fast, but ~4 points less close)

By default it reads the line with 4 seeds and keeps the take measured closest to the real take (speaker models +
Whisper word errors, tools/pick.py); all of them stay in line.takes/ with their scores in line.json.
  options: --ref NAME (any clip in vo/refs/refs.json), --temperature, --top-p, --top-k, --sub-temperature,
           --eq (tilt the codec's sound toward the phone mic: less boom, more air; neutral on the speaker scores,
           for cutting clone lines between real ones), --raw-out (no trim, no loudness match)

Write the words the way you'd say them: numbers as words ("twenty eleven"), acronyms in caps are fine (LLM, AI),
but never put a normal word in caps for stress (OCEAN gets misread and Whisper hides it); use punctuation instead.
About 1.5x real time on this Mac's GPU plus ~15 s to load the model. Leave HF_HUB_OFFLINE unset.
"""
import argparse, json, os, re, subprocess, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, soundfile as sf

VO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(VO, "tools"))
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WHISPER_PY = os.path.expanduser("~/reels/.venv/bin/python")
SCORE_PY = os.path.join(VO, ".venv-score", "bin", "python")


def caps_check(text):
    bad = [w for w in re.findall(r"\b[A-Z][A-Z']{3,}\b", text) if re.search(r"[AEIOU]", w)]
    if bad:
        raise SystemExit(f"words in caps get mispronounced: {', '.join(bad)} - write them in lowercase "
                         "and get the stress from punctuation (or pass --allow-caps for a real acronym)")


def loudness_match(y, target_db):
    """Speech-frame RMS to the real take's (so a line drops into the edit at the same level), peak-safe."""
    hop = 240; n = len(y) // hop
    e = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    cur = 20 * np.log10(np.sqrt(np.mean(e[e > np.percentile(e, 40)] ** 2)))
    y = y * 10 ** ((target_db - cur) / 20)
    peak = np.abs(y).max()
    return y * (0.95 / peak) if peak > 0.95 else y


def trim(y, sr, keep=0.12):
    hop = sr // 100; n = len(y) // hop
    db = 20 * np.log10(np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    on = np.flatnonzero(db > db.max() - 40)
    if not len(on): return y
    a = max(0, on[0] * hop - int(keep * sr)); b = min(len(y), (on[-1] + 1) * hop + int(keep * sr))
    return y[a:b]


def main():
    S = json.load(open(os.path.join(VO, "settings.json")))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=S["seed"])
    ap.add_argument("--takes", type=int, default=S["takes"])
    ap.add_argument("--ref", default=S["ref"])
    ap.add_argument("--temperature", type=float, default=S["temperature"])
    ap.add_argument("--top-p", type=float, default=S["top_p"])
    ap.add_argument("--top-k", type=int, default=S["top_k"])
    ap.add_argument("--repetition-penalty", type=float, default=S["repetition_penalty"])
    ap.add_argument("--sub-temperature", type=float, default=S["subtalker_temperature"])
    ap.add_argument("--eq", action="store_true")
    ap.add_argument("--raw-out", action="store_true")
    ap.add_argument("--allow-caps", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite -o")
    a = ap.parse_args()
    text = " ".join(a.text.split())
    if not a.allow_caps: caps_check(text)
    out = os.path.abspath(a.out)
    if os.path.exists(out) and not a.force: raise SystemExit(f"{out} exists (--force to overwrite)")

    import torch
    from qwen_tts import Qwen3TTSModel
    refs = json.load(open(os.path.join(VO, "refs", "refs.json")))
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="mps:0", dtype=torch.float32)
    item = model.create_voice_clone_prompt(ref_audio=os.path.join(VO, "refs", refs[a.ref]["audio"]),
                                           ref_text=refs[a.ref]["text"], x_vector_only_mode=False)[0]
    if S.get("spk"):
        item.ref_spk_embedding = model.create_voice_clone_prompt(
            ref_audio=os.path.join(VO, "refs", refs[S["spk"]]["audio"]), x_vector_only_mode=True)[0].ref_spk_embedding
    print(f"model + prompt ready in {time.perf_counter() - t0:.1f}s", flush=True)

    seeds = list(range(a.seed, a.seed + a.takes))
    cand_dir = out[:-4] + ".takes" if a.takes > 1 else None
    if cand_dir: os.makedirs(cand_dir, exist_ok=True)
    cands = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        t1 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=[item], non_streaming_mode=S["non_streaming"],
            do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
            repetition_penalty=a.repetition_penalty, subtalker_temperature=a.sub_temperature,
            subtalker_top_p=S["subtalker_top_p"], subtalker_top_k=S["subtalker_top_k"],
            max_new_tokens=int(12.5 * (len(text.split()) * 0.6 + 4)))
        torch.mps.synchronize()
        y = np.asarray(wavs[0], dtype=np.float32).squeeze()
        if not a.raw_out:
            y = trim(y, sr)
            if a.eq:
                import eq
                y = eq.apply(os.path.join(VO, "eq.json"), y)
            y = loudness_match(y, S["loudness_db"])
        p = os.path.join(cand_dir, f"s{seed}.wav") if cand_dir else out
        sf.write(p, y, sr, subtype="PCM_24")
        cands.append(p)
        print(f"seed {seed}: {len(y) / sr:.2f}s in {time.perf_counter() - t1:.1f}s", flush=True)

    meta = dict(text=text, ref=a.ref, seed=a.seed, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                repetition_penalty=a.repetition_penalty, subtalker_temperature=a.sub_temperature,
                eq=a.eq and not a.raw_out, model=MODEL_ID)
    if cand_dir:
        del model; torch.mps.empty_cache()
        subprocess.run([WHISPER_PY, os.path.join(VO, "tools", "transcribe.py"), *cands], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        res = subprocess.run([SCORE_PY, os.path.join(VO, "tools", "pick.py"), text, *cands], check=True,
                             capture_output=True, text=True).stdout
        ranked = [json.loads(l) for l in res.splitlines() if l.startswith("{")]
        for r in ranked:
            print(f"  {os.path.basename(r['take'])}  close {r['close']:5.1f}  WER {100 * r['wer']:4.1f}%  "
                  f"{' '.join(r['flags']) or 'ok':8s} heard: {r['heard']}")
        best = ranked[0]["take"]
        sf.write(out, sf.read(best, dtype="float32")[0], sr, subtype="PCM_24")
        meta.update(seed=int(os.path.basename(best)[1:-4]), takes=ranked)
        print(f"kept {os.path.basename(best)}")
    json.dump(meta, open(out[:-4] + ".json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
