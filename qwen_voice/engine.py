"""Qwen3-TTS voice-clone engine: one model, cached reference prompts, chunked synthesis."""

import os
import re
import threading

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from . import emotions as emo

MODEL_ID = os.environ.get("QWEN_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# ~100 words (~35 s of speech) per generation pass is proven on a 24 GB M4 Pro; longer
# scripts are split at paragraph/sentence boundaries and stitched with a short gap.
MAX_CHUNK_CHARS = 600
CHUNK_GAP_S = 0.3


def split_text(text):
    chunks = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= MAX_CHUNK_CHARS:
            chunks.append(para)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?…])\s+", para):
            if current and len(current) + 1 + len(sentence) > MAX_CHUNK_CHARS:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)
    return chunks


def _device():
    if torch.backends.mps.is_available():
        return "mps:0", torch.float32
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.prompts = {}

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        if self.model is None:
            from qwen_tts import Qwen3TTSModel

            device, dtype = _device()
            self.model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
        return self.model

    def _prompt(self, voice):
        key = (voice.name, voice.ref_audio, os.path.getmtime(voice.ref_audio), voice.transcript)
        if key not in self.prompts:
            self.prompts[key] = self.model.create_voice_clone_prompt(
                ref_audio=voice.ref_audio,
                ref_text=voice.transcript,
                x_vector_only_mode=voice.transcript is None,
            )
        return self.prompts[key]

    def warm(self, voice=None):
        with self.lock:
            self.load()
            if voice is not None:
                self._prompt(voice)

    def synthesize(self, text, voice, emotions=None, intensity=None, seed=None, language=None):
        """Returns (waveform float32, sample_rate, settings dict)."""
        chunks = split_text(text)
        if not chunks:
            raise ValueError("text is empty")
        emotions = list(emotions) if emotions else list(voice.emotions)
        intensity = voice.intensity if intensity is None else float(intensity)
        seed = voice.seed if seed is None else int(seed)
        language = language or voice.language
        temperature, top_p, repetition_penalty = emo.mix(emotions, intensity)

        with self.lock:
            model = self.load()
            prompt = self._prompt(voice)
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            pieces, sr = [], 24000
            for chunk in chunks:
                wavs, sr = model.generate_voice_clone(
                    text=chunk,
                    language=language,
                    voice_clone_prompt=prompt,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=emo.BASE_TOP_K,
                    repetition_penalty=repetition_penalty,
                )
                pieces.append(np.asarray(wavs[0], dtype=np.float32).squeeze())

        gap = np.zeros(int(CHUNK_GAP_S * sr), dtype=np.float32)
        joined = [pieces[0]]
        for piece in pieces[1:]:
            joined += [gap, piece]
        settings = {
            "voice": voice.name,
            "mode": voice.mode,
            "emotions": emotions,
            "intensity": intensity,
            "seed": seed,
            "language": language,
            "chunks": len(chunks),
            "temperature": round(temperature, 3),
            "top_p": round(top_p, 3),
            "repetition_penalty": round(repetition_penalty, 3),
        }
        return np.concatenate(joined), sr, settings

    def synthesize_batch(self, texts, voice, emotions=None, intensity=None, seed=None, language=None):
        """Several short texts in one generation pass (about 2x the throughput of one-by-one on CPU).
        All texts share the sampling settings and seed, so a take depends on the batch it was made in.
        Returns (list of waveforms float32, sample_rate, settings dict)."""
        texts = [" ".join(t.split()) for t in texts]
        if not texts or any(not t for t in texts):
            raise ValueError("texts must be non-empty")
        if any(len(t) > MAX_CHUNK_CHARS for t in texts):
            raise ValueError(f"batch texts must be <= {MAX_CHUNK_CHARS} characters each")
        emotions = list(emotions) if emotions else list(voice.emotions)
        intensity = voice.intensity if intensity is None else float(intensity)
        seed = voice.seed if seed is None else int(seed)
        language = language or voice.language
        temperature, top_p, repetition_penalty = emo.mix(emotions, intensity)

        with self.lock:
            model = self.load()
            prompt = self._prompt(voice)
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            wavs, sr = model.generate_voice_clone(
                text=texts,
                language=language,
                voice_clone_prompt=prompt,
                max_new_tokens=2048,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=emo.BASE_TOP_K,
                repetition_penalty=repetition_penalty,
            )
        settings = {
            "voice": voice.name,
            "mode": voice.mode,
            "emotions": emotions,
            "intensity": intensity,
            "seed": seed,
            "language": language,
            "batch": len(texts),
            "temperature": round(temperature, 3),
            "top_p": round(top_p, 3),
            "repetition_penalty": round(repetition_penalty, 3),
        }
        return [np.asarray(w, dtype=np.float32).squeeze() for w in wavs], sr, settings

    def unload(self):
        with self.lock:
            self.model = None
            self.prompts.clear()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
