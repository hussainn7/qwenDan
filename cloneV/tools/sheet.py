"""A/B picture: for each pair (real, clone) a spectrogram of each plus both pitch contours on one axis, so artifacts
(buzz, smears, gaps, clipped tails) and intonation differences show without listening.
  vo/.venv-score/bin/python vo/tools/sheet.py out.png real1.wav clone1.wav [real2.wav clone2.wav ...]"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, parselmouth
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def f0(p):
    pt = parselmouth.Sound(p).to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=400)
    f = pt.selected_array["frequency"].astype(float); f[f == 0] = np.nan
    return pt.xs(), f


def spec(ax, p, title):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1024, hop_length=240)), ref=np.max)
    ax.imshow(S[:400], origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=0,
              extent=[0, len(x) / sr, 0, 400 * sr / 1024 / 1000])
    ax.set_ylabel("kHz"); ax.set_title(title, fontsize=9, loc="left")


def main():
    out, files = sys.argv[1], sys.argv[2:]
    pairs = list(zip(files[::2], files[1::2]))
    fig, axes = plt.subplots(len(pairs), 3, figsize=(16, 2.6 * len(pairs)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for (real, clone), row in zip(pairs, axes):
        spec(row[0], real, "real: " + os.path.basename(real)); spec(row[1], clone, "clone: " + os.path.basename(clone))
        for p, c, lab in ((real, "#1f77b4", "real"), (clone, "#d62728", "clone")):
            t, f = f0(p); row[2].plot(t, f, ".", ms=2.5, color=c, label=lab)
        row[2].set_ylim(70, 300); row[2].set_ylabel("F0 Hz"); row[2].legend(loc="upper right", fontsize=8)
        tmax = max(row[0].get_xlim()[1], row[1].get_xlim()[1])
        for ax in row: ax.set_xlim(0, tmax)
    fig.tight_layout(); fig.savefig(out, dpi=80)


if __name__ == "__main__":
    main()
