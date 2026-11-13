# cloneV: cloning my own voice with Qwen3-TTS 1.7B

This is a snapshot of the work, taken 2026-11-12 and stopped mid-way.
- **Working copy:** `~/animations/siri-llm/vo/`. The scripts in this folder assume that layout (`VO` = the folder
  above `tools/`).
- **Kept local, never pushed:**
  - every recording, transcript and sentence cut of my voice;
  - the speaker vectors;
  - the trained LoRA weights.

  With those, anyone could make my voice. This repo is public.

## Where it stands

| | status |
|---|---|
| ICL clone (10.6 s reference, best of 4) | done, in use (`clone.py`, `settings.json`, `CLONE_NOTES.md`) |
| "It sounds robotic / generic": measured | done (below) |
| My voice in older takes: found, checked and cut | done: 200 sentences, 24.3 min, from 45 takes |
| LoRA fine-tune on those 24 min | trained (6 passes); pass 2 is the best checkpoint by held-out loss |
| Measuring the fine-tuned voice | done (section 5): no version beats the ICL clone on likeness yet |
| Listening A/B (real / ICL / fine-tune) | reels made, waiting on my ear |
| Wiring the fine-tuned voice into `clone.py` / qwen-voice | only if it wins the listen |

## 1. The current clone (ICL)

`clone.py` clones from the 10.6 s `B3` reference clip (Siri sentences S08–S10). It uses the model's sampling
defaults except **sub-talker temperature 0.5**, reads 4 seeds and keeps the take closest to my voice. All
scores below are on held-out sentences.

| | closeness¹ | ECAPA | WER | F0 median | F0 range |
|---|---|---|---|---|---|
| my real sentences | 100 | 0.749 | 3.4 % | 115 Hz | 10.1 st |
| clone, one take | 96.9 | 0.704 | 0.3 % | 114 Hz | 11.1 st |
| clone, best of 4 | ~100.6 | | | | |
| Derek (another voice) | 0 | 0.118 | | 149 Hz | 13.8 st |

¹ The closeness scale: 0 = another man's voice, 100 = my own sentences compared with each other. ECAPA, WavLM-SV
and Resemblyzer are each rescaled between those two points, then averaged.

`CLONE_NOTES.md` has the full sweep: about 750 takes across reference length, mode, sampling, EQ and acronym
spelling.

## 2. Why it still sounds "a bit robotic / generic"

The speaker-ID models say the clone *is* me, but it still sounds generic. So I measured naturalness separately
(`tools/natural.py`), on 14 real sentences and 168 clone takes:

| | SQUIM MOS | PESQ | SI-SDR | spectral GV | loudness spread | pitch jitter |
|---|---|---|---|---|---|---|
| my real sentences | 4.45 | 3.33 | 21.6 dB | −0.38 | 7.18 dB | 0.26 st |
| clone, sub-talker 0.9 (default) | 4.43–4.49 | 3.9–4.0 | 27–30 dB | −0.64 / −0.20 | 7.2–7.5 | 0.22–0.26 |
| clone, sub-talker 0.5 (chosen) | 4.46–4.51 | 4.0–4.1 | 28–30 dB | −0.63 / −0.14 | 7.3–7.4 | 0.22–0.25 |
| clone, sub-talker 0.3 | 4.42–4.52 | 3.95–4.09 | 27–31 dB | −0.61 / −0.17 | 7.2–7.3 | 0.22–0.25 |

Each clone row is a pair of 7-sentence folds (A / B); a spread gives both folds' means.

**Findings:**
- **Sound quality isn't the problem.** The clone is as natural per frame as my recordings: same MOS, same
  spectral detail, same loudness spread, same pitch micro-movement. It's actually *cleaner*: 6–9 dB more SI-SDR,
  because it has no room noise.
- **The sub-talker temperature isn't the cause either.** 0.3, 0.5 and 0.9 measure the same.
- **What's left is delivery:** pitch shape, stress and timing. The clone is more sing-song than me (11.1 vs 10.1
  semitones of range), sometimes pushes stressed words to 200 Hz where I peak at about 150 Hz, misses my low,
  quiet endings, and is weaker on long run-on lines.
- **Why it's generic:** all it knows of my delivery is 10 s of audio, so for everything else it falls back on
  its own default way of reading.

`tools/prosody.py` measures delivery directly. For each test sentence it compares the clone with how I said that
same sentence: pitch-contour correlation after DTW alignment, loudness-contour correlation, and per-word
duration correlation. The current clone scores:

| config | pitch contour r | loudness r | word timing r |
|---|---|---|---|
| ICL, sub 0.5, fold A | 0.70 | 0.70 | 0.77 |
| ICL, sub 0.5, fold B | 0.52 | 0.80 | 0.76 |
| ICL, sub 0.9, fold A | 0.67 | 0.70 | 0.70 |
| ICL, sub 0.9, fold B | 0.56 | 0.79 | 0.77 |

This is the baseline the fine-tune has to beat.

Two things outside the voice model also make the film reads sound more mechanical:
- the film assembles lines one by one (each line read separately);
- it then speeds the whole voice up 8 % with atempo.

## 3. More of my voice: 50× the data

My raw talking-head takes were already on the Mac. Most are iPhone originals in `~/Downloads`.

1. **Extract:** `tools/extract_src.sh` pulled audio from 75 original takes (about 100 min). It skips " vN" edits,
   which have music.
2. **Check it's me:** `tools/harvest_spk.py` takes 3 s windows and measures ECAPA cosine to my real-sentence
   centroid.
   - 51 takes are clearly me (median 0.52–0.80), about 38 min of speech.
   - The other 24 are not (median under 0.35): other voices, or audio that isn't speech. That includes two
     16–19 min recordings.
3. **Transcribe:** Whisper large-v3-turbo (mlx), with word timestamps.
4. **Cut into sentences:** `tools/segment.py` cuts 2.5–14 s utterances that end on pauses. Each cut snaps to the
   quietest 10 ms near the Whisper edge. An utterance is kept only if all of these hold:
   - every word is confident (min p ≥ 0.35, mean ≥ 0.8);
   - the pace is plausible;
   - it is me (ECAPA ≥ 0.45);
   - it is clean (SQUIM SI-SDR ≥ 8 dB, MOS ≥ 3.6).

   Result: **200 utterances, 24.3 min, from 45 takes.** Each is loudness-matched to −22.2 dB. Dropped: 28 for
   the transcript, 15 for quality, 10 for pace, 5 for speaker.
5. **Hold out the test set:** the Siri take is excluded completely, so its 14 sentences stay a fair test set.

## 4. LoRA fine-tune (`tools/finetune.py`)

The official recipe is `QwenLM/Qwen3-TTS/finetuning/sft_12hz.py`. It's a full fine-tune in bf16 with
flash-attention (CUDA), and its training input doesn't match how the model speaks. I checked by scoring the
untrained model on my utterances under each input layout:

| training input layout | code-0 CE | sub-talker CE |
|---|---|---|
| matches `generate()` (text_projection, English tag, x-vector, non-streaming) | **1.30** | **5.66** |
| official: text embeddings without `text_projection` | 10.31 | |
| official: sub-talker fed the hidden state one frame late | | 7.32 |

The table shows two problems in the official script. In addition, its sub-talker loss goes through the
causal-LM loss, which shifts the labels one more step (11.45). `finetune.py` builds each training sequence the
same way `generate()` does, and scores the sub-talker logits directly.

**Setup:**
- LoRA, rank 16, alpha 32, on all attention and MLP linears of the talker and the code predictor. That's 231
  layers and 19.2M trainable parameters; the fp32 base stays frozen.
- AdamW, lr 1e-4 with warmup and cosine decay; 8-sample gradient accumulation.
- Loss: code-0 CE + 0.3 × sub-talker CE.
- Speaker: the mean x-vector of all training utterances.
- Inference: an x-vector-only prompt in non-streaming mode. The base weights still work for ICL, so ICL can be
  combined with the LoRA too.
- Runs on the M4 Pro (24 GB) at about 1 s per utterance, so 6 passes take about 20 min. Freeing the MPS cache
  every step is required: without it the cache grows past 20 GB into swap.

**Held-out loss** (13 utterances from 3 takes not used for training):

| pass | 0 | 1 | **2** | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| code-0 CE | 1.215 | 1.048 | **1.032** | 1.058 | 1.129 | 1.199 | 1.231 |
| sub-talker CE | 5.472 | 5.289 | 5.233 | **5.227** | 5.242 | 5.257 | 5.267 |

It learns my voice for 2–3 passes, then starts memorising those 24 min. **Passes 2 and 3 are the candidates.**

## 5. Test results (2026-11-12)

All rows are single takes (seeds 1–2, no best-of-N), on the 14 held-out Siri sentences unless noted.
- **likeness:** score.py's closeness against the Siri recording.
- **fair likeness:** `tools/likeness_x.py`, against the 3 held-out takes from other rooms that no version heard.
- **pitch shape / word timing:** `prosody.py` correlations with my real reading of the same sentence.

| version | likeness | fair likeness | ECAPA | WER | pitch range (me 10.1) | pitch shape r | word timing r |
|---|---|---|---|---|---|---|---|
| **ICL now** (B3 clip, sub 0.5) | **98.5** | **97.8** | 0.706 | **0.3 %** | 11.1 | 0.604 | **0.747** |
| fine-tune 1 (mean x-vector), pass 2 | 82.5 | 90.3 | 0.659 | 1.7 % | 9.0 | 0.622 | 0.730 |
| fine-tune 1, pass 3 | 85.2 | 94.7 | 0.663 | 3.3 % | 9.3 | 0.587 | 0.703 |
| fine-tune 1, pass 2, Siri-clip x-vector | 93.4 | 88.9 | 0.695 | 4.9 % | 10.3 | 0.633 | 0.744 |
| fine-tune 2 (per-utterance x-vectors), pass 2, Siri x-vector | 94.7 | 87.6 | 0.707 | 2.4 % | 11.1 | 0.586 | 0.685 |
| **fine-tune 2, pass 3, Siri x-vector** | 95.6 | 89.9 | 0.703 | 4.0 % | 11.0 | **0.664** | 0.659 |

The ICL-plus-LoRA variants were scored on Siri sentences S01–S07 only, against ICL now (98.2 likeness, 0 % WER):

| ICL + LoRA variant (S01–S07) | likeness | WER |
|---|---|---|
| fine-tune 1, pass 2, streaming | 93.4 | 8.5 % |
| fine-tune 1, pass 2, non-streaming | 94.8 | 4.0 % |
| fine-tune 1, pass 2, half strength | 95.2 | 2.0 % |
| fine-tune 2, pass 3 | 90.9 | 4.6 % |

**Findings:**
- **Likeness: no fine-tune beats the ICL clone yet.** That holds for both the Siri-room measure and the fair,
  other-room measure. A 10.6 s ICL reference of one recording still pins the voice better than a LoRA on 24 min of
  mixed rooms.
- **A fixed mean x-vector averages me across rooms.** Fine-tune 1 trained on one constant speaker vector, so it
  learned "average me": lower pitch (−2 st), slower, flatter (9.0 st range), and furthest from any one recording.
- **Per-utterance x-vectors help the fine-tune but don't win.** Fine-tune 2 lowers held-out loss at every pass
  (0.992 vs 1.032) and, given the Siri clip's x-vector, matches the ICL's ECAPA score (0.703 vs 0.706).
- **It follows my pitch shape best of all** (r 0.664 vs 0.604), which is the "says it like me" part.
- **But it costs word timing and accuracy.** Timing r is 0.659 vs 0.747, and WER 2–5 % vs 0.3 %. One new line got
  a misplaced pause; another got "how it worked" for "how I work".
- **Possible cause of the slips:** Whisper transcripts in the training set drop fillers and restarts, so some
  training text doesn't match its audio exactly.
- Naturalness (SQUIM MOS 4.4–4.5) is the same for every version.

Listening reels (local, not pushed: they are my voice):
- `samples/ft/reel-siri-sentences.wav`: real → ICL → fine-tune 2 pass 3, for 6 sentences;
- `samples/ft/reel-new-lines.wav`: ICL → fine-tune, for the 5 lines in `new_lines.json`.

## 6. Next steps

1. **Listen and decide.** The numbers are a trade-off: the fine-tune is closer in pitch shape, the ICL is better
   on likeness, timing and accuracy.
   - If the fine-tune sounds more like me, wire it in: add `--ft` to `clone.py` (merging the LoRA at load,
     x-vector from a chosen ref) and add a qwen-voice voice type for it.
   - Otherwise keep the ICL.
2. **Cleaner training text.** Re-transcribe the training set with a verbatim setting (keep "um", repeats), or drop
   utterances whose phoneme recognition (`elevator/vo/qwen/phones.py`) disagrees with the text. That should
   remove the slips and timing drift.
3. **Train in ICL format.** Put a reference utterance from the same take (codes + text) in the prefix, as
   `generate_icl_prompt` does. The fine-tune then learns to copy the reference's room and mood exactly, the way
   the ICL clone does, while adding my general delivery habits. This is the most promising route to beat ICL on
   both likeness and delivery.
4. **One room.** Fine-tune only on takes that match the Siri recording's room and mic (median ECAPA ≥ 0.7 to it: 9 takes,
   about 7 min of speech), or record 5–10 min in that setup.
5. **More data.** Each new talking-head take is about 40 s more of my voice. Re-run:
   - `extract_src.sh` → `harvest_spk.py` → `transcribe.py` → `segment.py` → `finetune.py prep` → `train`.

   Calm reading and low, quiet line endings are what's missing.
6. **Film side:** read paragraphs, not single lines, and don't speed the voice up with atempo.

**How to run:**
- `tools/finetune.py train RUN --epochs 3 --per-utt-xvec`
- `tools/eval_ft.sh RUN EPOCH --xvec-ref B3` (or `--icl B3 --streaming --sub-temp 0.5 --sentences …`)
- `tools/compare.py` and `tools/likeness_x.py` for the tables
- `tools/reel.py` for A/B reels; new lines via `finetune.py gen … --texts new_lines.json`

## Files

- `clone.py`, `settings.json`: the current ICL clone (see `CLONE_NOTES.md`).
- `tools/finetune.py`: the LoRA trainer.
  - `prep`: codec codes + x-vectors.
  - `train`: trains and saves per-pass checkpoints.
  - `gen`: generates the held-out test sentences from a checkpoint.
- `tools/eval_ft.sh`: generate, then transcribe, then score.
- `tools/compare.py`, `tools/likeness_x.py`: the result tables; `tools/reel.py`: A/B listening reels.
- Data harvest: `tools/extract_src.sh`, `tools/harvest_spk.py`, `tools/segment.py`.
- Measures:
  - `tools/score.py`: closeness, WER, pitch, MCD;
  - `tools/natural.py`: naturalness;
  - `tools/prosody.py`: delivery against the real reading;
  - `tools/pick.py`: best-of-N;
  - `tools/transcribe.py`: Whisper.
- From the ICL sweep: `tools/sweep.py`, `tools/cut_real.py`, `tools/eq.py`, `tools/sheet.py`,
  `tools/make_samples.py`, `tools/pick_eval.py`.

**Environments:**
- Qwen3-TTS (`qwen-tts` 0.1.1, torch on MPS) runs `clone.py`, `finetune.py` and `sweep.py`.
- A scoring venv (speechbrain, transformers, resemblyzer, librosa, parselmouth, torchaudio 2.8 for SQUIM) runs
  the harvest and measure scripts.
- mlx-whisper runs `transcribe.py`.
