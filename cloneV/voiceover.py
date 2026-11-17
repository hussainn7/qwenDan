"""A whole voiceover in the user's cloned voice, with a mood per line, laid in like his real talking.
  PY=~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python
  $PY ~/animations/siri-llm/vo/voiceover.py script.txt -o out/vo [--takes 4] [--finish office-lav] [--tone]

script.txt: one sentence per line. A [mood] tag starts a line or sits on its own line and holds until the next tag:
  [hook] This chip costs as much as a brand new car.
  [story] A week ago, my whole trip got cancelled.
  (blank line = a paragraph break, a longer pause)
Moods: hype, hook, story, explain, serious, and current (the old B3 clip). Each mood clones from one of his own real
sentences said that way (refs/moods.json, "use": true). Write numbers as spoken, no caps words for stress.

For every line: --takes readings (one sentence per generation, never sped up), trimmed and level-matched; then
Whisper + tools/vo_rank.py keep the take that sounds most like him, says the words, and carries the mood. Lines are
laid end to end at 1.0x with his own pauses (0.26 s after a sentence, 0.7 s at a paragraph; never packed tighter),
optionally finished in a real room (vo/finish/<preset>, with --eq tilt first as the preset expects) with its room
tone under everything.
Out: <out>/lines/NN.wav (+ .takes/), <out>/voiceover.wav, <out>/voiceover.json (times, mood, ref, scores),
<out>/voiceover.srt.
"""
import argparse, json, os, re, subprocess, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, soundfile as sf

VO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VO)
from clone import trim, loudness_match, caps_check, MODEL_ID, WHISPER_PY, SCORE_PY

GAP_SENT, GAP_PARA, GAP_ASK = 0.26, 0.7, 0.4    # his median pause after a sentence; paragraph; after ? or !


def parse(path):
    lines, mood, para = [], "story", 0
    for raw in open(path):
        s = raw.strip()
        if not s:
            para += 1; continue
        m = re.match(r"^\[(\w+)\]\s*(.*)$", s)
        if m:
            mood, s = m.group(1).lower(), m.group(2).strip()
            if not s: continue
        lines.append(dict(id=f"{len(lines) + 1:02d}", text=s, mood=mood, para=para))
    return lines


def mood_refs():
    M = json.load(open(os.path.join(VO, "refs", "moods.json")))
    R = {}
    for rid, r in M.items():
        if r.get("use"): R[r["mood"]] = dict(r, id=rid, path=os.path.join(VO, "refs", r["audio"]))
    base = json.load(open(os.path.join(VO, "refs", "refs.json")))
    R["current"] = dict(id="B3", path=os.path.join(VO, "refs", base["B3"]["audio"]), text=base["B3"]["text"], target=None)
    return R


def srt_time(t):
    h, m = divmod(t, 3600); m, s = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--takes", type=int, default=4)
    ap.add_argument("--finish", metavar="PRESET")
    ap.add_argument("--tone", action="store_true")
    ap.add_argument("--allow-caps", action="store_true")
    a = ap.parse_args()
    S = json.load(open(os.path.join(VO, "settings.json")))
    L = parse(a.script)
    R = mood_refs()
    for l in L:
        if l["mood"] not in R: sys.exit(f"line {l['id']}: unknown mood [{l['mood']}] (have: {', '.join(sorted(R))})")
        if not a.allow_caps: caps_check(l["text"])
    os.makedirs(os.path.join(a.out, "lines"), exist_ok=True)

    import torch
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="mps:0", dtype=torch.float32)
    kw = dict(temperature=S["temperature"], top_p=S["top_p"], top_k=S["top_k"], repetition_penalty=S["repetition_penalty"],
              subtalker_temperature=S["subtalker_temperature"], subtalker_top_p=S["subtalker_top_p"], subtalker_top_k=S["subtalker_top_k"])
    prompts, jobs = {}, []
    for l in L:
        r = R[l["mood"]]
        if l["mood"] not in prompts:
            prompts[l["mood"]] = model.create_voice_clone_prompt(ref_audio=r["path"], ref_text=r["text"], x_vector_only_mode=False)[0]
        tdir = os.path.join(a.out, "lines", f"{l['id']}.takes"); os.makedirs(tdir, exist_ok=True)
        for seed in range(1, a.takes + 1):
            p = os.path.join(tdir, f"s{seed}.wav")
            if not os.path.exists(p):
                torch.manual_seed(seed); np.random.seed(seed)
                wavs, sr = model.generate_voice_clone(text=l["text"], language="English", voice_clone_prompt=[prompts[l["mood"]]],
                                                      non_streaming_mode=S["non_streaming"], do_sample=True,
                                                      max_new_tokens=int(12.5 * (len(l["text"].split()) * 0.6 + 4)), **kw)
                y = loudness_match(trim(np.asarray(wavs[0], dtype=np.float32).squeeze(), sr), S["loudness_db"])
                sf.write(p, y, sr, subtype="PCM_24")
                print(f"{l['id']} [{l['mood']}] s{seed} {len(y) / sr:.1f}s", flush=True)
            jobs.append(dict(line=l["id"], path=p, text=l["text"], target=r.get("target")))
    del model; torch.mps.empty_cache()

    subprocess.run([WHISPER_PY, os.path.join(VO, "tools", "transcribe.py"), *[j["path"] for j in jobs]], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    jp = os.path.join(a.out, "jobs.json"); json.dump(jobs, open(jp, "w"), indent=1)
    subprocess.run([SCORE_PY, os.path.join(VO, "tools", "vo_rank.py"), jp], check=True, stdout=subprocess.DEVNULL)
    ranked = json.load(open(jp[:-5] + ".ranked.json"))

    if a.finish:
        import finish
        sys.path.insert(0, os.path.join(VO, "tools"))
        import eq
    out_sr = 48000 if a.finish else 24000
    parts, meta, t = [], [], 0.0
    for i, l in enumerate(L):
        best = ranked[l["id"]][0]
        y, sr = sf.read(best["path"], dtype="float32")
        if a.finish:   # the presets were fitted on --eq takes (clone.py --eq --finish), so tilt first
            y = eq.apply(os.path.join(VO, "eq.json"), y)
            y, sr = finish.finish(y, sr, a.finish, False)
        sf.write(os.path.join(a.out, "lines", f"{l['id']}.wav"), y, sr, subtype="PCM_24")
        if i:
            prev = L[i - 1]
            gap = GAP_PARA if l["para"] != prev["para"] else (GAP_ASK if prev["text"][-1:] in "?!" else GAP_SENT)
            parts.append(np.zeros(int(gap * sr))); t += gap
        meta.append(dict(l, ref=R[l["mood"]]["id"], start=round(t, 3), end=round(t + len(y) / sr, 3),
                         take=os.path.basename(best["path"]), like=best["like"], wer=best["wer"], mood_dist=best["mood"],
                         heard=best["heard"]))
        parts.append(y.astype(np.float64)); t += len(y) / sr
    vo = np.concatenate(parts)
    if a.finish and a.tone:
        tone, _ = sf.read(os.path.join(VO, "finish", a.finish, "room-tone.wav"), dtype="float64")
        xf = int(0.25 * out_sr); ramp = np.linspace(0, 1, xf); bed = np.zeros(len(vo)); pos = 0
        while pos < len(vo):
            piece = tone.copy(); piece[:xf] *= ramp; piece[-xf:] *= ramp[::-1]
            n = min(len(piece), len(vo) - pos); bed[pos:pos + n] += piece[:n]; pos += len(tone) - xf
        vo = vo + bed
    vo *= min(1.0, 0.95 / (np.abs(vo).max() + 1e-9))
    sf.write(os.path.join(a.out, "voiceover.wav"), vo, out_sr, subtype="PCM_24")
    json.dump(dict(script=os.path.abspath(a.script), takes=a.takes, finish=a.finish, tone=a.tone, lines=meta),
              open(os.path.join(a.out, "voiceover.json"), "w"), indent=1)
    with open(os.path.join(a.out, "voiceover.srt"), "w") as f:
        for i, m in enumerate(meta, 1):
            f.write(f"{i}\n{srt_time(m['start'])} --> {srt_time(m['end'])}\n{m['text']}\n\n")
    for m in meta:
        print(f"{m['id']} [{m['mood']:7s}] {m['start']:6.2f}-{m['end']:6.2f}  like {m['like']:5.1f}  wer {m['wer']:.2f}  {m['text'][:60]}")
    print(f"{os.path.join(a.out, 'voiceover.wav')}  {len(vo) / out_sr:.1f} s")


if __name__ == "__main__":
    main()
