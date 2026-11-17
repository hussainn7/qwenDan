# Your voice, cloned locally (Qwen3-TTS 1.7B Base)

```bash
PY=~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python
$PY ~/animations/siri-llm/vo/clone.py "Okay, so here's the part nobody tells you." -o line.wav
```

`clone.py` reads the line with 4 seeds and keeps the take that measures closest to your real voice. All 4 takes
stay in `line.takes/`, with their scores in `line.json`. `--seed 7` gives four other readings. `--takes 1` is the
fast mode: about 4 points less close on average. `--ref A3` gives a punchier read, with more pitch movement.
`--eq` tilts the sound toward your phone mic. Write numbers as you say them ("twenty eleven"). Never put a normal
word in caps for stress; acronyms like LLM are fine. One line takes ~1.5× its length per take on this Mac's GPU,
plus ~15 s to load the model and ~20 s to transcribe and score the takes.

Source: the Siri-vs-LLMs talking-head take (`~/Downloads/ADCC3379-….MP4`), via `edit/voice.wav` (pops softened).
That's 14 sentences, ~50 s of speech. The flubbed restart at 25.3–28.8 s is never used.

## How close it gets

Closeness is measured on held-out sentences only: the model never heard the real version of a line it was scored
on. The scale is a "closeness" score:
- **0** = a different male voice (Derek, the old narrator).
- **100** = your own real sentences compared with each other.
- Each of three speaker-ID models is rescaled between those two points, then the three are averaged.

| | closeness | ECAPA cos | WavLM-SV cos | Resemblyzer cos | WER | F0 median | F0 range | pace |
|---|---|---|---|---|---|---|---|---|
| your real sentences | 100 | 0.749 | 0.947 | 0.891 | 3.4 %¹ | 115 Hz | 10.1 st | 1.00 |
| **clone, one take** (chosen settings) | **96.9** | 0.704 | 0.946 | 0.887 | 0.3 % | 114 Hz | 11.1 st | 1.00 |
| **clone, best of 4** (what clone.py keeps)² | **~100.6** | | | | | | | |
| old tunnel settings, same reference | 94.7 | 0.692 | 0.944 | 0.879 | 1.1 % | 114 Hz | 10.9 st | 1.04 |
| Derek (different voice) | 0 | 0.118 | 0.851 | 0.641 | | 149 Hz | 13.8 st | |

¹ Whisper's own mistakes on your real audio ("So you started" for "Siri started"). ² The best take is chosen using
only reference-side audio, so this number isn't picked by peeking at the held-out sentences (tools/pick_eval.py).

In plain terms:
- **Timbre and identity are very close.** Speaker-ID models put a single take 97 % of the way from "another man"
  to "you". The best of 4 seeds lands where your own sentences sit relative to each other.
- **ECAPA is the strictest judge.** It rates the clone 0.70 vs 0.75 for your own sentences against each other
  (Derek: 0.12).
- **It's intelligible.** Whisper heard the intended words in 54 of the 56 chosen-config takes. The two slips were
  "Notice a problem" and "So you can"; the slip that recurs across the whole sweep is "So it can" → "So we can",
  which is your own fast elision. A phoneme recogniser (no
  language model, so it doesn't paper over mistakes the way Whisper does) confirms "Siri", "iPhone" and "4S".
- **Acronyms are the weak spot:** "LLM" comes out as a clean "el-el-em" in about 3 of 4 seeds. The rest squash it
  to "el-lam", and Whisper still writes "LLM" (stage 4 below). Listen to acronym lines, or check them with
  `~/animations/elevator/vo/qwen/phones.py`.
- **Pace and pitch level match:** median 114 vs 115 Hz, and speech span equal on average (±15 % per line).
- **Pitch movement differs.** It's a bit more sing-song than you: 11.1 vs 10.1 semitones of range. It sometimes
  pushes a stressed word to ~200–210 Hz where you peaked at ~150 Hz (samples/ab-sheet.png).
- **Long lines lose a little.** It lands 10–19 points under the real sentence on the longest lines: the "sorry,
  I can't do that" rant (S10: 104 vs 123), S13, S06 and S11. Short 1–2 s lines are noisy to measure, even for
  real audio.
- **It misses deliberate low landings.** On "And this is the big shift" you dropped low and quiet; the clone said
  it 4.5 semitones higher. Expect to pick seeds, or end on the reference's mood, for big tonal moves.
- **Sound quality:** clean, with no clipping, buzz or runaway tails. The codec is ~2 dB bassier and 2–3 dB duller
  above 5 kHz than your phone mic. `--eq` corrects that, but it did not raise the speaker scores out of fold, so
  it's off by default. The noise floor matches (−57.7 vs −56.6 dB).

## What was tried (all numbers on held-out sentences)

Two folds: references cut from S01–S07 and tested on S08–S14 (A), and the reverse (B). Each take was scored on:
- three speaker-ID models (vs the other held-out real sentences),
- Whisper large-v3-turbo WER,
- Praat F0 median and 5–95 % range,
- pace (real speech span ÷ take span),
- DTW mel-cepstral distance to the real sentence,
- flags: clipping, a pause >0.3 s longer than the real one, sound after the last word, repeated words.

**Stage 1**, fold A, 2 seeds × 7 sentences, one change at a time:

| config | closeness | ECAPA | WER % | F0 range | pace | flags |
|---|---|---|---|---|---|---|
| model defaults, A3 ref (3 sentences, 14 s) | 95.1 | 0.676 | 1.3 | 11.6 | 1.03 | 0 |
| ref: 1 sentence, 3 s | 91.7 | 0.617 | 1.4 | 11.0 | 0.98 | 2 |
| ref: 1 sentence, 6.6 s | 94.0 | 0.645 | 0.6 | 12.1 | 0.99 | 0 |
| ref: 4 statements, 15 s | 99.1 | 0.691 | 1.9 | 9.0 | 1.05 | 2 |
| ref: 6 sentences, 24.5 s | 93.5 | 0.679 | 2.6 | 8.4 | 1.03 | 2 |
| ref: 7 sentences spliced over the flub, 28 s | 96.8 | 0.686 | 2.6 | 11.1 | 0.98 | 1 |
| ref from raw44.wav (pops not softened) | 96.3 | 0.691 | 0.6 | 10.8 | 1.02 | 0 |
| x-vector only (no ICL), 24.5 s | 91.4 | 0.648 | 0.6 | 11.8 | **0.86** | 2 |
| ICL A3 + speaker embedding of the 24.5 s clip | 98.5 | 0.692 | 0.4 | 11.3 | 0.99 | 0 |
| temperature 0.7 | 96.8 | 0.682 | 0.6 | 12.0 | 1.02 | 1 |
| temperature 0.7 + sub-talker 0.7 | 95.3 | 0.670 | 0.6 | 11.6 | 1.00 | 2 |
| temperature 1.0 | 96.1 | 0.678 | 1.3 | 11.5 | 1.03 | 0 |
| top-k 20, top-p 0.8 | 93.8 | 0.688 | 1.0 | 11.0 | 1.04 | 0 |
| **sub-talker temperature 0.5** | **98.7** | **0.704** | 0.6 | 11.6 | 0.98 | 0 |
| non-streaming text mode | 91.7 | 0.655 | 1.3 | 11.4 | 1.01 | 1 |
| old tunnel runner settings (emotion mix, temp 1.15, top-k 20) | 94.2 | 0.691 | 0.6 | 11.4 | 1.04 | 0 |

Per-config standard error is ~3 points (mostly sentence-to-sentence), so only the leaders went on.

**Stage 2**, both folds, 4 seeds × 7 sentences (56 takes per row):

| config | fold A | fold B | mean |
|---|---|---|---|
| old tunnel settings, 3-sentence ref | 92.5 | 96.8 | 94.7 |
| model defaults, 3-sentence ref | 93.0 | 96.6 | 94.8 |
| sub-talker 0.3 | 95.8 | 95.7 | 95.8 |
| **sub-talker 0.5** | 96.8 | 97.0 | **96.9** |
| sub-talker 0.5 + long-clip speaker embedding | 96.2 | 97.0 | 96.6 |
| sub-talker 0.5, 15 s ref (4 sentences) | 97.1 | 96.4 | 96.8 |
| sub-talker 0.5, 22–28 s ref (all of the fold) | 100.0 | 93.1 | 96.6 |

- **Sub-talker temperature 0.5 is the one change that wins on both folds.** The sub-talker samples the residual
  codebooks, which carry fine timbre detail.
- **Reference length past ~10–15 s doesn't help reliably.** It wins on one fold and loses on the other. Under ~7 s
  is clearly worse.
- **Best of 4 (pick_eval.py), judged by reference-side audio only:** the kept take averages 101.0 (A) and 100.3 (B)
  vs 96.8 / 97.0 for an average take. That's why clone.py defaults to 4 takes.
- **EQ match:** in-fold it gained +0.4 to +1.1; out of fold it lost 0.4–0.6. So it's opt-in.

**Stage 3**, choosing the default reference: 4 new lines that aren't in the take × 4 seeds, with sub-talker 0.5.
- B3 (S08–S10: "Here's a simplified version… sorry, I can't do that", 10.6 s): mean closeness 107, F0 range 11.2 st.
- A3 (S01–S03: the hook question + the Apple history, 14.3 s): 104, F0 range 12.8 st.
- Real: 10.1 st. Both read zero word errors.
- Closeness here is against all 14 real sentences, which include the reference audio, so it runs over 100.
  It's only used to compare the two refs.
- B3 is the default; A3 reads more animated and stays available as `--ref A3`.

**Stage 4**, how to write an acronym: the same line, 4 seeds per spelling. The phoneme recogniser judged whether
"LLM" came out as three letters.

| written as | clean "el-el-em" |
|---|---|
| `LLM` | 3/4 (and 2/4 in the stage-2 takes) |
| `L L M` | 3/4 |
| `L.L.M.` | 2/4 (one "el-oh-em") |
| `el-el-em` | 2/4, one garbled, one cut off |

Keep acronyms as written, and pick by ear when one matters.

**New lines through clone.py** (samples/new-*.wav, 4 takes each, closeness vs all 14 real sentences):
- 0 word errors and no flags on all 12 takes.
- Kept takes scored 118, 113 and 109.
- Seed spread within a line was up to 17 points (new-3: 92–109). That spread is why picking matters.

## Chosen settings (settings.json)

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` on mps, float32, English.
- ICL clone (`x_vector_only_mode=False`) from `refs/B3.wav` (voice.wav, 10.6 s), with its exact transcript.
- The model's own sampling (temperature 0.9, top-p 1.0, top-k 50, repetition 1.05) and streaming text mode.
- **Sub-talker temperature 0.5.**
- Best of 4 seeds (1–4), picked by `tools/pick.py`: closeness to all 14 real sentences − 150 × WER − 15 per flag.
- Silence trimmed to 0.12 s, then loudness matched to your take's speech level (−22.2 dB RMS on speech frames).

## Files

- `clone.py`, `settings.json`: the line maker.
- `refs/`: reference clips + exact transcripts (`refs.json`). `real/`: the 14 real sentences, with `clips.json`
  holding the times and what was said (checked by Whisper and a phoneme recogniser: "Apple brought", "a pattern it
  learned", "Third is, it would…").
- `tools/`:
  - `cut_real.py`: cuts real/ and refs/.
  - `sweep.py`: generates a sweep spec.
  - `transcribe.py`: Whisper, in the `~/reels` venv.
  - `score.py`: the table above.
  - `pick.py`, `pick_eval.py`: take picking.
  - `eq.py`: mic-match EQ (`eq.json`).
  - `sheet.py`: spectrogram and pitch A/B picture.
  - `make_samples.py`.
- `sweep/`: stage specs, logs, all ~750 takes, `scores.json`.
- `.venv-score/`: speechbrain ECAPA, WavLM-SV, Resemblyzer, Praat, matplotlib (torch 2.8; the Qwen venv is untouched).
- `samples/`:
  - `<S>-real.wav` + `<S>-clone-1/2.wav` + `<S>-AB.wav` (real, pause, clone) for S01 S05 S07 S10 S12 S13. Each
    clone came from a reference that did not contain that sentence; these are the 2 best of 4 seeds by score.
  - `ab-reel.wav`: all six A/Bs back to back.
  - `ab-sheet.png`.
  - `new-1..3.wav`: lines that aren't in the take, straight from clone.py.

## Getting closer still

The remaining gap is prosody, not timbre: pitch peaks, low endings, and the energy of long run-on lines. With one
50 s take, the in-context clone is near its ceiling. The next real step is more of your voice:
- 3–5 minutes of clean takes in the same mic and setup, with different moods (hook, calm explainer, low landing).
- Use them as per-mood references (like the kit's hook/body/turn/end).
- Or fine-tune the 1.7B Base on them, with the Qwen3-TTS SFT recipe. That needs a CUDA GPU or a long MPS run.

## Finishing presets: your real rooms (`finish.py`, `vo/finish/`)

The clone comes out dry and thin, like a locked-down studio. A finishing preset puts a line in one of your real rooms:
that room's echo, your mic's tone, your speaking level, and optionally your room tone underneath.

```bash
$PY ~/animations/siri-llm/vo/clone.py "text" -o line.wav --eq --finish office-lav          # line.wav finished, line.dry.wav raw
$PY ~/animations/siri-llm/vo/clone.py "text" -o line.wav --eq --finish office-lav --tone   # + room tone under it
$PY ~/animations/siri-llm/vo/finish.py any-clone-line.wav -o out.wav [--preset office-lav] [--tone]
```

**office-lav** (2026-11-13, fixed in the CS projects short, `~/animations/cs-projects`): your office (whiteboard,
desk, monitor) on the clip-on lav. `vo/finish/office-lav/preset.json` records every number and how it was measured:
- **Room:** RT60 0.3 s at 250-500 Hz, 0.5 s at 1-2 kHz, 0.4 s at 4 kHz. The tail sits 9.75 dB under the voice, after
  early reflections at 3-19 ms. The dry clone's sound fell at 610-850 dB/s after each syllable, against 350-450 for
  your real lines; finished, it falls at 380-500.
- **Mic:** about +10 dB at 160-400 Hz and -10 dB at 10 kHz (4095-tap linear-phase filter). The finished clone is within
  ±1.5 dB of your real lines from 100 Hz to 10 kHz.
- **Level:** speech at -25.1 dBFS. The room tone is -71.6 dBFS.
- **Use it with `--eq` on**: the preset was fitted on `--eq` takes.
- **Proof:** running `finish.py` on the film's three raw lines reproduces the lines used in the film (correlation
  1.000000, residual -62 dB).
- **Words:** write them plainly as said, e.g. "resumes", not "résumés" (that read French). Check odd words by phonemes,
  because Whisper hides mispronunciations.

**Rebuild or add a room:** the fit is in `cs-projects/vo`. Run `room.py` (echo, measured from running speech per octave),
then `match.py` (EQ after the echo), then `export_voice.py` (writes the preset here). For a new room or mic, point
their `TAKE`/`REAL` at a take filmed there and export under a new name.

### What keeps it from sounding robotic (learned 2026-11-16)

The CS projects short (`~/animations/cs-projects`) is the reference: the user says that clone sounds "very nice", while
later films with the same `--eq --finish office-lav` lines sounded robotic. Same voice; the difference is how the
lines were laid into the film:
- **Natural speed (1.0x).** gpu-camry ran at 1.08x and charge-left at 1.1x, through atempo.
- **The clone's own pauses left alone.** Those films capped pauses at 0.30-0.34 s and packed lines 0.08-0.4 s apart.
- **Room tone under everything**, so no word or gap ever drops to digital silence. Use `--tone`, or lay the take's
  room tone the way `cs-projects/edit/build.py` does. The later films had none.
- **No noise reduction or presence boost on the clone.** frist-hackathon ran afftdn plus +2 dB at 3.2 kHz over its
  narration, and it was one long generated line; the clone is weakest on long run-on lines.
- **Each sentence generated as its own line**, best of 4, with odd words checked by phonemes. Then a gentle 3:1
  compressor and a linear loudnorm, nothing more.

## Moods: your voice with emotion (`voiceover.py`, `clone.py --mood`, 2026-11-17)

```bash
PY=~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python
$PY ~/animations/siri-llm/vo/voiceover.py script.txt -o out/vo --finish office-lav --tone   # a whole voiceover
$PY ~/animations/siri-llm/vo/clone.py "One line." -o line.wav --mood hook                   # a single line
```

`script.txt` has one sentence per line. A `[mood]` tag holds until the next tag; a blank line is a paragraph break.
`samples/moods/demo-script.txt` is an example.

**What it does.** The old clone read everything from one clip (B3: lively, 11 semitones of pitch range), so every line
got the same sing-song delivery whatever it said. That was the "generic" sound. The model copies the delivery of the
clip it clones from (measured over 17 clips × 4 lines × 2 seeds: the clone's pitch level, range and movement follow
the clip's at r = 0.75, its pace at 0.60, its loudness swing at 0.47). So each mood now clones from one of your own
real sentences said that way:

| mood | your clip it clones from (local only) | the clone's read (4 test lines) | likeness |
|---|---|---|---|
| hype | 0BB3650A_01 | widest: 13.3 st range | 84.8 |
| hook | 3D6ABD96_00 | slowest, most emphatic: 3.7 w/s | 81.3 |
| story | 267E87EC_04 | warm, lively, a bit higher | 92.0 |
| explain | 7EC168F2_07 | clear, even; the most like you | 98.8 |
| serious | A2308E6E_01 | low, flat (90 Hz, 6.3 st): the low landing | 85.4 |
| current | B3 (Siri take) | 10.1 st, 105 Hz | 95.6 |

Likeness: 0 = Derek, 100 = your own sentences (setup-1 centroid). Every clone lands 14–22 points under the real clip it
copies, B3 included (−13.8); excited or low delivery just sounds less like your *average* voice even when it's really
you (your real hype clip scores 104, your serious one 100). So the moods cost no likeness beyond the mood itself.
All 136 test takes had 0 word errors.

**Demo** (`samples/moods/`): the same 9-line script read from B3 only (`1-before-one-mood.wav`) and with moods
(`2-after-with-moods.wav`); same average likeness (84). With B3, the "serious" line came out at
136 Hz with 13 st of swing; with `serious` it lands at 100 Hz, and the next line at 89 Hz. The hook reads at 3.3 w/s
instead of 4.6. `same-line-every-mood.wav` has two lines in all six moods.

**How voiceover.py picks and lays in.** Each line is its own generation (4 seeds), never sped up. Whisper plus
`tools/vo_rank.py` keep the take with the best `likeness − 150 × WER − 8 × mood distance`, where mood distance is how
far the take's pitch level, range, movement, pace and loudness swing sit from the mood clip's, in your own spread.
Lines are laid end to end with your own pauses, measured from 597 sentence ends in your takes: 0.26 s after a
sentence (median), 0.4 s after ? or !, and 0.7 s at a paragraph. With `--finish office-lav` each line gets the `--eq`
tilt, then the office/lav room, and `--tone` lays the room tone under everything. Out: `voiceover.wav`, per-line wavs,
`voiceover.json` (times, mood, take, scores) and `voiceover.srt`.

**Where the clips came from.** There are now 234 clean sentences (28.8 min) from 55 takes, including 7 takes since
11-12 and your 3 reels (`data/reels/`, music removed with Demucs htdemucs_ft in `.venv-sep`; the music sat 12–28 dB
under your voice). Only reel sentences that still pass the quality filter are used, because any music left in a
clip would be cloned too. A recording-setup clustering (`data/setups.json`, 1/3-octave long-term spectrum) puts 33
takes, the Siri take and the reels in one setup; all mood clips come from that setup, so the moods sound like one
room and the office-lav finish fits them. Tools: `tools/separate.py`, `tools/delivery.py`,
`tools/mood_test.py`, `tools/mood_score.py`, `tools/vo_rank.py`.

**Adding or swapping a mood clip:** add an entry to `refs/moods.json` (audio in `refs/moods/`, the exact words, numbers
as spoken), run `tools/mood_test.py --only ID`, transcribe, `tools/mood_score.py`, and set `"use": true` on the winner.
