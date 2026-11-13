"""Long-term-spectrum match: the clone's codec sound is ~2 dB bassier and 2-3 dB duller above 5 kHz than the
user's mic. fit() measures that difference on speech frames and stores a smooth correction curve; apply() filters
a take with it (linear phase, so timing is untouched).
  python vo/tools/eq.py fit vo/eq.json --real vo/real/S0[1-7].wav --gen vo/sweep/takes/<cfg>/*.wav
  python vo/tools/eq.py apply vo/eq.json in.wav out.wav"""
import json, sys
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve

N = 1024; HOP = 256; SR = 24000; LIMIT = 6.0


def ltas(files):
    acc, n = 0, 0
    for f in files:
        x, sr = sf.read(f, dtype="float32", always_2d=True); x = x.mean(1)
        assert sr == SR, f
        fr = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP] * np.hanning(N)
        e = (fr ** 2).mean(1); keep = e > np.percentile(e, 40)          # speech frames only
        acc = acc + (np.abs(np.fft.rfft(fr[keep], axis=1)) ** 2).sum(0); n += keep.sum()
    return 10 * np.log10(acc / n + 1e-12)


def smooth_octave(f, db, frac=3):
    out = np.empty_like(db)
    for i, fc in enumerate(f):
        lo, hi = fc * 2 ** (-0.5 / frac), fc * 2 ** (0.5 / frac)
        m = (f >= lo) & (f <= hi); out[i] = db[m].mean() if m.any() else db[i]
    return out


def fit(path, real, gen):
    f = np.fft.rfftfreq(N, 1 / SR)
    d = smooth_octave(f, ltas(real) - ltas(gen))
    band = (f > 500) & (f < 2000); d -= d[band].mean()                  # level-neutral around the voice's core
    d[f < 70] = d[np.searchsorted(f, 70)]
    d = np.clip(d, -LIMIT, LIMIT)
    json.dump(dict(freq=f.round(2).tolist(), gain_db=d.round(3).tolist(), real=real, gen_count=len(gen)), open(path, "w"))
    return f, d


def apply(path, x):
    c = json.load(open(path))
    g = 10 ** (np.array(c["gain_db"]) / 20)
    h = np.fft.fftshift(np.fft.irfft(g, N)) * np.hanning(N)              # linear-phase FIR
    y = fftconvolve(x, h, mode="full")[N // 2: N // 2 + len(x)]
    return y.astype(np.float32)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        a = sys.argv[3:]; i, j = a.index("--real"), a.index("--gen")
        real, gen = (a[i + 1:j], a[j + 1:]) if i < j else (a[i + 1:], a[j + 1:i])
        f, d = fit(sys.argv[2], real, gen)
        for fc in (100, 200, 400, 800, 1600, 3200, 5000, 7000, 9000, 11000):
            print(f"{fc:6d} Hz {d[np.argmin(abs(f - fc))]:+5.1f} dB")
    else:
        x, sr = sf.read(sys.argv[3], dtype="float32"); assert sr == SR
        sf.write(sys.argv[4], apply(sys.argv[2], x), SR, subtype="PCM_24")
