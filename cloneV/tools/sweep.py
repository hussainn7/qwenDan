"""Generates the held-out test sentences for every config in a sweep spec, with the model loaded once.
  ~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python vo/tools/sweep.py vo/sweep/stage1.json [config ...]
-> vo/sweep/takes/<config>/<sentence>-s<seed>.wav + .json (skips takes that exist)

A config: {"ref": "A3", "src": "voice"|"raw", "mode": "icl"|"xvec", "spk": "<ref whose speaker embedding replaces
the ICL clip's own>", "ref_text": "<override transcript>", "fold": "A"|"B", "seeds": [1, 2], "emotions": [...],
"temperature", "top_p", "top_k", "repetition_penalty", "subtalker_temperature", "subtalker_top_p", "subtalker_top_k",
"non_streaming": bool}. Unset sampling keys fall back to the model's generation_config.json (0.9 / 1.0 / 50 / 1.05).
Leave HF_HUB_OFFLINE unset (transformers calls the Hub even when offline and crashes)."""
import ast, copy, json, os, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, soundfile as sf, torch

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
NODE_FILE = "/Users/hussain/ComfyUI-Qwen3TTS-Emotional/qwen3_emotional_clone.py"
SAMPLING = ("temperature", "top_p", "top_k", "repetition_penalty", "subtalker_temperature", "subtalker_top_p", "subtalker_top_k")


def emotion_mix(cfg):
    """The ComfyUI node's emotion presets shift temperature / top_p / repetition (same averaging + clamping)."""
    tree = ast.parse(open(NODE_FILE).read())
    P = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "EMOTION_PRESETS" for t in n.targets))
    m = [P[e] for e in cfg["emotions"]]; n = len(m)
    cfg["temperature"] = max(0.1, min(2.0, cfg.get("temperature", 0.9) + sum(x["temp"] for x in m) / n))
    cfg["top_p"] = max(0.1, min(1.0, cfg.get("top_p", 1.0) + sum(x["top_p"] for x in m) / n))
    cfg["repetition_penalty"] = max(1.0, min(2.0, cfg.get("repetition_penalty", 1.05) + sum(x["rep_pen"] for x in m) / n))


def main():
    spec_path = os.path.abspath(sys.argv[1])
    spec = json.load(open(spec_path))
    clips = json.load(open(os.path.join(VO, "real", "clips.json")))
    refs = json.load(open(os.path.join(VO, "refs", "refs.json")))
    want = sys.argv[2:] or list(spec["configs"])
    from qwen_tts import Qwen3TTSModel
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="mps:0", dtype=torch.float32)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    prompts = {}

    def prompt(cfg):
        key = (cfg["ref"], cfg.get("src", "voice"), cfg.get("mode", "icl"), cfg.get("spk"), cfg.get("ref_text"))
        if key not in prompts:
            suf = ".raw" if cfg.get("src") == "raw" else ""
            audio = os.path.join(VO, "refs", f"{cfg['ref']}{suf}.wav")
            text = cfg.get("ref_text") or refs[cfg["ref"]]["text"]
            item = model.create_voice_clone_prompt(ref_audio=audio, ref_text=text,
                                                   x_vector_only_mode=cfg.get("mode") == "xvec")[0]
            if cfg.get("spk"):
                other = model.create_voice_clone_prompt(ref_audio=os.path.join(VO, "refs", f"{cfg['spk']}{suf}.wav"),
                                                        x_vector_only_mode=True)[0]
                item = copy.copy(item); item.ref_spk_embedding = other.ref_spk_embedding
            prompts[key] = item
        return prompts[key]

    for name in want:
        cfg = dict(spec.get("defaults", {}), **spec["configs"][name])
        if cfg.get("emotions"): emotion_mix(cfg)
        fold = clips["folds"][cfg.get("fold", "A")]
        tests = cfg.get("test") or fold["test"]
        assert cfg.get("mode") == "xvec" or not set(refs[cfg["ref"]]["sentences"]) & set(tests), f"{name}: ref overlaps test"
        out_dir = os.path.join(VO, "sweep", "takes", name); os.makedirs(out_dir, exist_ok=True)
        item = prompt(cfg)
        kw = {k: cfg[k] for k in SAMPLING if k in cfg}
        for sid in tests:
            text = cfg.get("texts", {}).get(sid) or clips["clips"][sid]["text"]
            for seed in cfg.get("seeds", [1, 2]):
                out = os.path.join(out_dir, f"{sid}-s{seed}.wav")
                if os.path.exists(out): continue
                torch.manual_seed(seed); np.random.seed(seed)
                t1 = time.perf_counter()
                wavs, sr = model.generate_voice_clone(
                    text=text, language="English", voice_clone_prompt=[item],
                    non_streaming_mode=bool(cfg.get("non_streaming")), do_sample=True,
                    max_new_tokens=int(12.5 * (len(text.split()) * 0.6 + 4)), **kw)
                torch.mps.synchronize()
                wav = np.asarray(wavs[0], dtype=np.float32).squeeze()
                sf.write(out, wav, sr, subtype="PCM_24")
                meta = dict(config=name, sentence=sid, seed=seed, text=text, ref=cfg["ref"], src=cfg.get("src", "voice"),
                            mode=cfg.get("mode", "icl"), spk=cfg.get("spk"), non_streaming=bool(cfg.get("non_streaming")),
                            **{k: round(v, 3) if isinstance(v, float) else v for k, v in kw.items()},
                            duration_s=round(len(wav) / sr, 3), generation_s=round(time.perf_counter() - t1, 2))
                json.dump(meta, open(out[:-4] + ".json", "w"), indent=1)
                print(f"{name} {sid} s{seed} {meta['duration_s']}s in {meta['generation_s']}s", flush=True)


if __name__ == "__main__":
    main()
