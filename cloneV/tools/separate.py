# Split reel audio into the user's voice and the music under it (Demucs htdemucs_ft, the vocals specialist).
#   .venv-sep/bin/python tools/separate.py data/reels/R*.mix.wav
# Writes <name>.vocals.wav (44.1 kHz mono) and <name>.music.wav next to each mix, and prints how loud the music
# sits under the voice: median and 90th-percentile music-to-voice ratio over voiced 0.5 s windows, and the
# fraction of windows where music is within 20 dB of the voice (where separation artefacts are likely).
import os, sys
import numpy as np, soundfile as sf, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = get_model('htdemucs_ft').eval()
srcs = model.sources                      # drums, bass, other, vocals


def db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


for p in sys.argv[1:]:
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    assert sr == model.samplerate, sr
    wav = torch.from_numpy(x.T.copy())
    ref = wav.mean(0)
    w = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, w[None], device=dev, shifts=2, split=True, overlap=0.25, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    voc = out[srcs.index('vocals')].mean(0).numpy()
    mus = sum(out[i] for i, s in enumerate(srcs) if s != 'vocals').mean(0).numpy()
    base = p[:-len('.mix.wav')] if p.endswith('.mix.wav') else p[:-4]
    sf.write(base + '.vocals.wav', voc, sr, subtype='FLOAT')
    sf.write(base + '.music.wav', mus, sr, subtype='FLOAT')
    hop = sr // 2
    rows = [(db(voc[i:i + hop]), db(mus[i:i + hop])) for i in range(0, len(voc) - hop, hop)]
    vmax = max(v for v, _ in rows)
    voiced = [(v, m) for v, m in rows if v > vmax - 25]
    ratio = np.array([m - v for v, m in voiced])
    print(f'{os.path.basename(base)}: {len(voc) / sr:.1f} s, voice {np.mean([v for v, _ in voiced]):.1f} dB, '
          f'music-to-voice median {np.median(ratio):.1f} dB, p90 {np.percentile(ratio, 90):.1f} dB, '
          f'music within 20 dB in {np.mean(ratio > -20) * 100:.0f}% of voiced windows', flush=True)
