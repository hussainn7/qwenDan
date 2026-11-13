# LoRA fine-tune of Qwen3-TTS 1.7B Base on the user's own utterances (data/utt.jsonl from tools/segment.py).
#   PY=~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python
#   $PY tools/finetune.py prep                       # codec codes + x-vectors -> data/codes.pt
#   $PY tools/finetune.py train RUN [--epochs 4 --lr 1e-4 --rank 16]   # -> ft/RUN/epoch-N.safetensors
#   $PY tools/finetune.py gen RUN EPOCH [--seeds 1 2] # the 14 held-out Siri sentences -> sweep/takes/ft_RUN_eN/
#
# Training sequences are built exactly like generate() builds them for x-vector mode, language "English",
# non_streaming_mode=True: role tokens, [think, think_bos, english, think_eos, speaker, pad] under tts_pad/tts_bos,
# the whole text (text_projection'd) over codec_pad, tts_eos, codec_bos, then the audio frames over tts_pad.
# Loss = codebook-0 CE (incl. the eos) + 0.3 x sub-talker CE, where the sub-talker for frame t gets the hidden state
# that predicted frame t's codebook 0, as at inference. (The official sft_12hz.py skips text_projection,
# shifts the sub-talker hidden by one frame, and its sub-talker loss shifts labels again; on the untrained model
# those give code0 CE 10.3 vs 1.3 and sub CE 7.3 vs 5.7 here, so this follows the inference path instead.)
# The speaker vector is the mean x-vector of all training utterances (data/codes.pt "xvec"); inference passes it as
# an x-vector-only prompt, so the base model's Base/ICL paths keep working with the same weights.
import argparse, json, math, os, random, sys, time
import numpy as np, soundfile as sf, librosa, torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import save_file, load_file

VO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEV = "mps"


def load_model():
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="mps:0", dtype=torch.float32)


class LoRA(nn.Module):
    def __init__(self, base, r, alpha, drop=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def add_lora(talker, r, alpha):
    names = []
    for name, mod in list(talker.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in TARGETS and isinstance(child, nn.Linear):
                setattr(mod, child_name, LoRA(child, r, alpha))
                names.append(f"{name}.{child_name}")
    return names


def lora_state(talker):
    return {k: v.detach().cpu().contiguous() for k, v in talker.state_dict().items() if k.endswith((".A", ".B"))}


def merge_lora(talker, path, scale_override=None):
    """fold saved LoRA deltas into the plain Linear weights (no wrapper modules at inference)"""
    sd = load_file(path)
    meta = json.load(open(path[:-12] + ".json")) if os.path.exists(path[:-12] + ".json") else {}
    scale = scale_override if scale_override is not None else meta.get("alpha", 32) / meta.get("rank", 16)
    mods = dict(talker.named_modules())
    n = 0
    for k in sd:
        if not k.endswith(".A"): continue
        base = k[:-2]
        lin = mods[base]
        A, B = sd[k].to(lin.weight), sd[base + ".B"].to(lin.weight)
        with torch.no_grad():
            lin.weight += (B @ A) * scale
        n += 1
    return n


# ---------------------------------------------------------------- data
def prep():
    tts = load_model()
    rows = [json.loads(l) for l in open(os.path.join(VO, "data", "utt.jsonl"))]
    codes, xv = {}, []
    for i, r in enumerate(rows):
        y, sr = sf.read(os.path.join(VO, r["audio"]), dtype="float32")
        with torch.inference_mode():
            c = tts.model.speech_tokenizer.encode(y, sr=sr).audio_codes[0].cpu()
            e = tts.model.extract_speaker_embedding(librosa.resample(y, orig_sr=sr, target_sr=24000) if sr != 24000 else y, 24000)
        codes[r["audio"]] = c.to(torch.int16)
        xv.append(e.float().cpu())
        if i % 50 == 0: print(i, len(rows), tuple(c.shape), flush=True)
    xv = torch.stack(xv)
    torch.save(dict(codes=codes, xvec=xv.mean(0), xvecs=xv), os.path.join(VO, "data", "codes.pt"))
    print("saved", len(codes), "codes; x-vector norm", xv.mean(0).norm().item())


class Builder:
    """the generate() prefix for x-vector + English + non-streaming, as embeddings"""

    def __init__(self, tts):
        self.tts, self.m = tts, tts.model
        cfg = self.m.config
        tc = cfg.talker_config
        self.talker = self.m.talker
        self.ids = dict(bos=cfg.tts_bos_token_id, eos=cfg.tts_eos_token_id, pad=cfg.tts_pad_token_id)
        self.prefill = [tc.codec_think_id, tc.codec_think_bos_id, tc.codec_language_id["english"], tc.codec_think_eos_id]
        self.c_pad, self.c_bos, self.c_eos = tc.codec_pad_id, tc.codec_bos_id, tc.codec_eos_token_id
        self.ngroups = tc.num_code_groups

    def text_ids(self, text):
        return self.tts._tokenize_texts([self.tts._build_assistant_text(text)])[0]

    def tproj(self, ids):
        return self.talker.text_projection(self.talker.get_text_embeddings()(ids))

    def cemb(self, ids):
        return self.talker.get_input_embeddings()(ids)

    def sequence(self, text, codes, xvec):
        """inputs_embeds (1, L, d), index of the codec_bos position; codes (T, 16) long on device"""
        d = lambda x: torch.tensor([x], device=DEV)
        ids = self.text_ids(text)
        tts_bos, tts_eos, tts_pad = self.tproj(d([self.ids["bos"], self.ids["eos"], self.ids["pad"]])).chunk(3, dim=1)
        codec = torch.cat([self.cemb(d(self.prefill)), xvec.view(1, 1, -1), self.cemb(d([self.c_pad, self.c_bos]))], 1)
        role = self.tproj(ids[:, :3])
        head = torch.cat([tts_pad.expand(-1, codec.shape[1] - 2, -1), tts_bos], 1) + codec[:, :-1]
        body = torch.cat([self.tproj(ids[:, 3:-5]), tts_eos], 1) + self.cemb(d([self.c_pad] * (ids.shape[1] - 8 + 1)))
        bos = tts_pad + self.cemb(d([self.c_bos]))
        prefix = torch.cat([role, head, body, bos], 1)
        p_bos = prefix.shape[1] - 1
        if codes is None:
            return prefix, p_bos
        frames = self.cemb(codes[:, 0][None])
        for i in range(1, self.ngroups):
            frames = frames + self.talker.code_predictor.get_input_embeddings()[i - 1](codes[:, i][None])
        return torch.cat([prefix, frames + tts_pad], 1), p_bos


def step_loss(B, text, codes, xvec):
    x, p_bos = B.sequence(text, codes, xvec)
    T = codes.shape[0]
    L = x.shape[1]
    out = B.talker(inputs_embeds=x, attention_mask=torch.ones(1, L, dtype=torch.long, device=DEV),
                   use_cache=False, output_hidden_states=True)
    h = out.hidden_states[0][-1]                    # (1, L, d) final layer
    logits = out.logits[0, p_bos:p_bos + T + 1]      # predicts code0[0..T-1], then eos
    target = torch.cat([codes[:, 0], torch.tensor([B.c_eos], device=DEV)])
    loss0 = F.cross_entropy(logits.float(), target)
    hid = h[0, p_bos:p_bos + T]                      # the state that predicted each frame's codebook 0
    # its built-in loss runs the causal-LM loss, which shifts labels by one more step; score the logits directly
    lg, _ = B.talker.forward_sub_talker_finetune(codes, hid)    # (T, 15, V): codebooks 1..15
    loss_sub = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), codes[:, 1:].reshape(-1))
    return loss0, loss_sub


# ---------------------------------------------------------------- train
def train(a):
    random.seed(0); torch.manual_seed(0)
    tts = load_model()
    tts.model.speech_tokenizer = None             # codec not needed to train; frees GPU memory
    torch.mps.empty_cache()
    data = torch.load(os.path.join(VO, "data", "codes.pt"))
    rows = [json.loads(l) for l in open(os.path.join(VO, "data", "utt.jsonl"))]
    rows = [r for r in rows if r["audio"] in data["codes"]]
    takes = sorted({r["take"] for r in rows})
    val_takes = set(random.Random(1).sample(takes, max(2, len(takes) // 12)))
    tr = [r for r in rows if r["take"] not in val_takes]
    va = [r for r in rows if r["take"] in val_takes]
    print(f"train {len(tr)} utt ({sum(r['dur'] for r in tr) / 60:.1f} min), val {len(va)} utt from {len(val_takes)} takes")
    xvec = data["xvec"].to(DEV)

    for p in tts.model.parameters(): p.requires_grad_(False)
    talker = tts.model.talker
    names = add_lora(talker, a.rank, a.alpha)
    params = [p for n, p in talker.named_parameters() if n.endswith((".A", ".B"))]
    print(f"LoRA r={a.rank} on {len(names)} linears, {sum(p.numel() for p in params) / 1e6:.1f}M params")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    B = Builder(tts)
    total = a.epochs * math.ceil(len(tr) / a.accum)
    warm = max(10, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))))
    out_dir = os.path.join(VO, "ft", a.run); os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a) | dict(train=len(tr), val=len(va), val_takes=sorted(val_takes)), open(os.path.join(out_dir, "run.json"), "w"), indent=1)

    def evaluate():
        talker.eval(); l0 = ls = 0
        with torch.no_grad():
            for r in va:
                a0, s0 = step_loss(B, r["text"], data["codes"][r["audio"]].long().to(DEV), xvec)
                l0 += a0.item(); ls += s0.item()
        talker.train(); torch.mps.empty_cache()
        return l0 / len(va), ls / len(va)

    v = evaluate(); print(f"epoch 0 val code0 {v[0]:.3f} sub {v[1]:.3f}", flush=True)
    step, t0 = 0, time.time()
    for ep in range(1, a.epochs + 1):
        random.shuffle(tr)
        talker.train(); run0 = runs = 0
        for i, r in enumerate(tr):
            l0, ls = step_loss(B, r["text"], data["codes"][r["audio"]].long().to(DEV), xvec)
            ((l0 + 0.3 * ls) / a.accum).backward()
            run0 += l0.item(); runs += ls.item()
            if (i + 1) % a.accum == 0 or i == len(tr) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                torch.mps.empty_cache()                # varying lengths otherwise grow the MPS cache into swap
                if step % 5 == 0:
                    n = (i % (a.accum * 5)) + 1
                    print(f"ep {ep} step {step}/{total} code0 {run0 / n:.3f} sub {runs / n:.3f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{(time.time() - t0) / step:.1f}s/step {torch.mps.driver_allocated_memory() / 1e9:.1f}GB", flush=True)
                    run0 = runs = 0
        v = evaluate(); print(f"epoch {ep} val code0 {v[0]:.3f} sub {v[1]:.3f}", flush=True)
        path = os.path.join(out_dir, f"epoch-{ep}.safetensors")
        save_file(lora_state(talker), path)
        json.dump(dict(rank=a.rank, alpha=a.alpha, epoch=ep, val_code0=v[0], val_sub=v[1]), open(path[:-12] + ".json", "w"))


# ---------------------------------------------------------------- generate the held-out sentences
def gen(a):
    tts = load_model()
    data = torch.load(os.path.join(VO, "data", "codes.pt"))
    if a.epoch > 0:
        n = merge_lora(tts.model.talker, os.path.join(VO, "ft", a.run, f"epoch-{a.epoch}.safetensors"), a.scale)
        print("merged", n, "LoRA deltas")
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    clips = json.load(open(os.path.join(VO, "real", "clips.json")))
    S = json.load(open(os.path.join(VO, "settings.json")))
    if a.icl:
        refs = json.load(open(os.path.join(VO, "refs", "refs.json")))
        item = tts.create_voice_clone_prompt(ref_audio=os.path.join(VO, "refs", refs[a.icl]["audio"]),
                                             ref_text=refs[a.icl]["text"], x_vector_only_mode=False)[0]
        if not a.ref_xvec: item.ref_spk_embedding = data["xvec"].to(DEV)
    else:
        item = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=data["xvec"].to(DEV), x_vector_only_mode=True, icl_mode=False)
    name = a.name or f"ft_{a.run}_e{a.epoch}" + (f"_icl{a.icl}" if a.icl else "")
    out_dir = os.path.join(VO, "sweep", "takes", name); os.makedirs(out_dir, exist_ok=True)
    sids = a.sentences or sorted(clips["clips"])
    for sid in sids:
        text = clips["clips"][sid]["text"]
        for seed in a.seeds:
            out = os.path.join(out_dir, f"{sid}-s{seed}.wav")
            if os.path.exists(out): continue
            torch.manual_seed(seed); np.random.seed(seed)
            t1 = time.perf_counter()
            wavs, sr = tts.generate_voice_clone(
                text=text, language="English", voice_clone_prompt=[item], non_streaming_mode=not a.streaming,
                do_sample=True, subtalker_temperature=a.sub_temp, temperature=a.temperature,
                max_new_tokens=int(12.5 * (len(text.split()) * 0.6 + 4)))
            wav = np.asarray(wavs[0], dtype=np.float32).squeeze()
            sf.write(out, wav, sr, subtype="PCM_24")
            meta = dict(config=name, sentence=sid, seed=seed, text=text, ref=a.icl or "xvec-mean", src="ft",
                        mode="icl" if a.icl else "xvec", run=a.run, epoch=a.epoch, non_streaming=not a.streaming,
                        subtalker_temperature=a.sub_temp, temperature=a.temperature,
                        duration_s=round(len(wav) / sr, 3), generation_s=round(time.perf_counter() - t1, 2))
            json.dump(meta, open(out[:-4] + ".json", "w"), indent=1)
            print(f"{name} {sid} s{seed} {meta['duration_s']}s in {meta['generation_s']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    t = sub.add_parser("train"); t.add_argument("run")
    t.add_argument("--epochs", type=int, default=4); t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=16); t.add_argument("--alpha", type=float, default=32)
    t.add_argument("--accum", type=int, default=8)
    g = sub.add_parser("gen"); g.add_argument("run"); g.add_argument("epoch", type=int)
    g.add_argument("--seeds", type=int, nargs="+", default=[1, 2]); g.add_argument("--sentences", nargs="*")
    g.add_argument("--icl", help="ICL ref name from refs.json (default: x-vector only)")
    g.add_argument("--ref-xvec", action="store_true", help="with --icl: keep the ref's own x-vector")
    g.add_argument("--streaming", action="store_true"); g.add_argument("--sub-temp", type=float, default=0.9)
    g.add_argument("--temperature", type=float, default=0.9); g.add_argument("--scale", type=float)
    g.add_argument("--name")
    a = ap.parse_args()
    {"prep": lambda: prep(), "train": lambda: train(a), "gen": lambda: gen(a)}[a.cmd]()
