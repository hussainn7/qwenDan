# Cut the user's source takes into clean training utterances for the fine-tune.
#   .venv-score/bin/python tools/segment.py
# Reads data/src/<take>.wav + .whisper.json (takes listed in data/mine.txt), splits at pauses into 2.5-14 s utterances
# that end on a pause, snaps each cut to the quietest 10 ms near the Whisper edge, and keeps an utterance only if
#   - every word is confident (min p >= 0.35, mean >= 0.8) and the pace is plausible (1.6-5.2 words/s, no word > 1.2 s)
#   - it is the user (ECAPA cosine to the real-sentence centroid >= 0.45)
#   - it is clean (SQUIM SI-SDR >= 8 dB, MOS >= 3.6)
# Writes data/utt/<take>_<n>.wav (24 kHz, speech RMS -22.2 dB) and data/utt.jsonl {audio, text, take, dur, cos, mos, sisdr}.
# The Siri take (ADCC3379) is held out: its sentences are the test set.
import json, os, re, glob
import numpy as np, librosa, soundfile as sf, torch
from speechbrain.inference.speaker import EncoderClassifier
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, 'data')
HOLDOUT = {'ADCC3379-7417-41AB-8F36-029230C4EFED'}
SR = 24000
TARGET_DB = -22.2
KEEP_CAPS = {'AI', 'UI', 'LLM', 'LLMS', 'API', 'APIS', 'MCP', 'PDF', 'SF', 'US', 'USA', 'CEO', 'CTO', 'GPU', 'CPU', 'OK',
             'TV', 'PC', 'IT', 'SAT', 'GPA', 'MIT', 'NYC', 'LA', 'UK', 'EU', 'OS', 'IOS', 'MACOS', 'URL', 'HTML', 'CSS', 'JS'}

enc = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                     savedir=os.path.expanduser('~/.cache/speechbrain/ecapa'), run_opts={'device': 'cpu'})
subj = SQUIM_SUBJECTIVE.get_model().eval()
obj = SQUIM_OBJECTIVE.get_model().eval()
nmr = [torch.from_numpy(librosa.load(os.path.join(HERE, 'real', f'{s}.wav'), sr=16000)[0])[None] for s in ('S03', 'S09', 'S12')]


@torch.no_grad()
def ecapa(y16):
    e = enc.encode_batch(torch.from_numpy(y16)[None]).squeeze().numpy()
    return e / np.linalg.norm(e)


cen = np.mean([ecapa(librosa.load(p, sr=16000)[0]) for p in sorted(glob.glob(os.path.join(HERE, 'real', 'S??.wav')))], 0)
cen /= np.linalg.norm(cen)


def clean_text(words):
    out = []
    for w in words:
        t = w['w']
        core = re.sub(r'[^A-Za-z]', '', t)
        if len(core) >= 2 and core.isupper() and core not in KEEP_CAPS and re.search('[AEIOU]', core):
            t = t.lower()
        out.append(t)
    s = ' '.join(out)
    s = re.sub(r'\s+([,.?!;:])', r'\1', s).strip()
    return s[0].upper() + s[1:] if s else s


def quietest(y, a, b):
    """time (s) of the quietest 10 ms window between a and b"""
    a, b = max(0, int(a * SR)), min(len(y), int(b * SR))
    if b - a < 480: return (a + b) / 2 / SR
    win = 240
    e = np.convolve(y[a:b] ** 2, np.ones(win) / win, 'valid')
    return (a + int(np.argmin(e)) + win // 2) / SR


def segments(words, total):
    """indices [i, j) of word runs: break after word k if the pause after it is long enough"""
    brk = []
    for k in range(len(words)):
        gap = (words[k + 1]['s'] if k + 1 < len(words) else total) - words[k]['e']
        end_punct = words[k]['w'][-1:] in '.?!'
        if gap >= 0.45 or (gap >= 0.22 and (end_punct or words[k]['w'][-1:] == ',')):
            brk.append((k, end_punct, gap))
    runs, i = [], 0
    while i < len(words):
        best = None
        for k, ep, gap in brk:
            if k < i: continue
            dur = words[k]['e'] - words[i]['s']
            if dur > 14: break
            if dur >= 2.5:
                best = k
                if ep and dur >= 4: break        # prefer a sentence end once there's enough
        if best is None:                          # nothing fits: skip to the next break
            nxt = [k for k, _, _ in brk if k >= i]
            i = (nxt[0] + 1) if nxt else len(words)
            continue
        runs.append((i, best + 1))
        i = best + 1
    return runs


def speech_rms_db(y):
    r = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    d = 20 * np.log10(r + 1e-9)
    return 10 * np.log10(np.mean(10 ** (d[d > d.max() - 30] / 10)))


@torch.no_grad()
def main():
    os.makedirs(os.path.join(D, 'utt'), exist_ok=True)
    takes = [os.path.splitext(os.path.basename(l.strip()))[0] for l in open(os.path.join(D, 'mine.txt')) if l.strip()]
    rows, drop = [], {}
    for take in takes:
        if take in HOLDOUT: continue
        wj = os.path.join(D, 'src', f'{take}.whisper.json')
        if not os.path.exists(wj): continue
        words = [w for w in json.load(open(wj))['words'] if w['w']]
        y, _ = librosa.load(os.path.join(D, 'src', f'{take}.wav'), sr=SR)
        total = len(y) / SR
        for n, (i, j) in enumerate(segments(words, total)):
            ws = words[i:j]
            prev_e = words[i - 1]['e'] if i > 0 else 0
            next_s = words[j]['s'] if j < len(words) else total
            a = quietest(y, max(prev_e, ws[0]['s'] - 0.3), ws[0]['s'] - 0.02)
            b = quietest(y, ws[-1]['e'] + 0.03, min(next_s, ws[-1]['e'] + 0.4))
            a, b = max(0.0, a), min(total, b)
            seg = y[int(a * SR):int(b * SR)].copy()
            dur = len(seg) / SR
            ps = np.array([w['p'] for w in ws])
            wl = max(w['e'] - w['s'] for w in ws)
            rate = len(ws) / max(ws[-1]['e'] - ws[0]['s'], 1e-3)
            why = None
            if not 1.5 <= dur <= 15: why = 'length'
            elif ps.min() < 0.35 or ps.mean() < 0.8: why = 'asr'
            elif not 1.6 <= rate <= 5.2 or wl > 1.2: why = 'pace'
            if why is None:
                y16 = librosa.resample(seg, orig_sr=SR, target_sr=16000)
                cos = float(ecapa(y16) @ cen)
                t = torch.from_numpy(y16)[None]
                mos = float(np.mean([subj(t, r).item() for r in nmr]))
                sisdr = obj(t)[2].item()
                if cos < 0.45: why = 'speaker'
                elif sisdr < 8 or mos < 3.6: why = 'quality'
            if why:
                drop[why] = drop.get(why, 0) + 1
                continue
            fade = int(0.01 * SR)
            seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
            seg *= 10 ** ((TARGET_DB - speech_rms_db(seg)) / 20)
            seg *= min(1, 0.95 / (np.abs(seg).max() + 1e-9))
            name = f'{take[:8]}_{n:02d}'
            p = os.path.join(D, 'utt', f'{name}.wav')
            sf.write(p, seg, SR, subtype='PCM_16')
            rows.append(dict(audio=os.path.relpath(p, HERE), text=clean_text(ws), take=take, dur=round(dur, 2),
                             cos=round(cos, 3), mos=round(mos, 2), sisdr=round(sisdr, 1)))
        print(take, len(rows), drop, flush=True)
    with open(os.path.join(D, 'utt.jsonl'), 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    print(f'{len(rows)} utterances, {sum(r["dur"] for r in rows) / 60:.1f} min; dropped {drop}')


if __name__ == '__main__':
    main()
