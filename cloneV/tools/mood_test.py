# Read the same lines from every mood reference (refs/moods.json) plus the current B3/A3, with clone.py's settings,
# in one model load.
#   ~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python tools/mood_test.py [--seeds 1 2] [--only ID ...] [--lines FILE]
# Takes land in sweep/takes/mood_<ref>/<line>-s<seed>.wav (+ .json); existing ones are skipped.
import argparse, json, os, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, soundfile as sf, torch

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VO)
from clone import trim, loudness_match, MODEL_ID

LINES = {
    "T1": "This one app replaced three tools I was paying for every single month.",
    "T2": "Last summer I moved across the country with one bag and no plan, and somehow it turned out to be the best decision I ever made.",
    "T3": "It reads your screen, finds the button you need, and clicks it for you, so you never have to leave the app.",
    "T4": "And I think that's what most people get wrong about building something from nothing.",
}


def refs():
    R = {k: dict(v, path=os.path.join(VO, "refs", v["audio"])) for k, v in json.load(open(os.path.join(VO, "refs", "moods.json"))).items()}
    base = json.load(open(os.path.join(VO, "refs", "refs.json")))
    for k in ("B3", "A3"):
        R[k] = dict(mood="current", path=os.path.join(VO, "refs", base[k]["audio"]), text=base[k]["text"])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--lines")
    a = ap.parse_args()
    lines = json.load(open(a.lines)) if a.lines else LINES
    S = json.load(open(os.path.join(VO, "settings.json")))
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="mps:0", dtype=torch.float32)
    kw = dict(temperature=S["temperature"], top_p=S["top_p"], top_k=S["top_k"], repetition_penalty=S["repetition_penalty"],
              subtalker_temperature=S["subtalker_temperature"], subtalker_top_p=S["subtalker_top_p"], subtalker_top_k=S["subtalker_top_k"])
    for rid, r in refs().items():
        if a.only and rid not in a.only: continue
        out_dir = os.path.join(VO, "sweep", "takes", f"mood_{rid}"); os.makedirs(out_dir, exist_ok=True)
        todo = [(lid, s) for lid in lines for s in a.seeds if not os.path.exists(os.path.join(out_dir, f"{lid}-s{s}.wav"))]
        if not todo: continue
        item = model.create_voice_clone_prompt(ref_audio=r["path"], ref_text=r["text"], x_vector_only_mode=False)[0]
        for lid, seed in todo:
            text = lines[lid]
            torch.manual_seed(seed); np.random.seed(seed)
            t1 = time.perf_counter()
            wavs, sr = model.generate_voice_clone(text=text, language="English", voice_clone_prompt=[item],
                                                  non_streaming_mode=S["non_streaming"], do_sample=True,
                                                  max_new_tokens=int(12.5 * (len(text.split()) * 0.6 + 4)), **kw)
            y = loudness_match(trim(np.asarray(wavs[0], dtype=np.float32).squeeze(), sr), S["loudness_db"])
            out = os.path.join(out_dir, f"{lid}-s{seed}.wav")
            sf.write(out, y, sr, subtype="PCM_24")
            json.dump(dict(ref=rid, mood=r["mood"], line=lid, text=text, seed=seed, duration_s=round(len(y) / sr, 2),
                           generation_s=round(time.perf_counter() - t1, 2)), open(out[:-4] + ".json", "w"), indent=1)
            print(f"{rid} {r['mood']} {lid} s{seed} {len(y) / sr:.1f}s", flush=True)


if __name__ == "__main__":
    main()
