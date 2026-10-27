"""The qwen-voice daemon: holds the model in memory and serves speech over local HTTP.

Clients (CLI, MCP server) start it on demand and it exits after QWEN_VOICE_IDLE_MINUTES
(default 15) without requests, so the ~10 GB the model uses comes back on its own.

    POST /speak             native: writes a file, returns its path + settings
    POST /speak_batch       several short texts in one pass (shared settings + seed), one file each
    POST /v1/audio/speech   OpenAI-compatible: returns audio bytes
    GET  /voices, /emotions, /health      POST /shutdown
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from . import voices as reg
from .emotions import EMOTION_PRESETS
from .engine import Engine

HOST = "127.0.0.1"
PORT = int(os.environ.get("QWEN_VOICE_PORT", "7731"))

engine = Engine()
app = FastAPI(title="qwen-voice")
state = {"last_activity": time.monotonic(), "server": None}


class SpeakRequest(BaseModel):
    text: str
    voice: str | None = None
    emotions: list[str] | None = None
    intensity: float | None = None
    seed: int | None = None
    language: str | None = None
    output_path: str | None = None
    format: str = "wav"
    play: bool = False


class SpeakBatchRequest(BaseModel):
    texts: list[str]
    output_paths: list[str] | None = None
    voice: str | None = None
    emotions: list[str] | None = None
    intensity: float | None = None
    seed: int | None = None
    language: str | None = None
    format: str = "wav"


class OpenAISpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = None
    response_format: str = "mp3"
    speed: float | None = None
    instructions: str | None = None


def find_ffmpeg():
    for candidate in (shutil.which("ffmpeg"), Path.home() / ".local/bin/ffmpeg",
                      "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def write_audio(wav, sr, path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp3":
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise HTTPException(400, "mp3 output needs ffmpeg on PATH; use wav instead")
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, wav, sr, subtype="PCM_16")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp.name,
                            "-codec:a", "libmp3lame", "-q:a", "2", str(path)], check=True)
    else:
        sf.write(path, wav, sr, subtype="PCM_24")
    return path


def default_output(text, fmt):
    slug = "-".join(re.sub(r"[^\w\s-]", "", text.lower()).split()[:6]) or "speech"
    return reg.OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.{fmt}"


def resolve_voice(name, fallback_to_default=False):
    try:
        return reg.get_voice(name)
    except LookupError as e:
        if fallback_to_default and name:
            try:
                return reg.get_voice(None)
            except LookupError:
                pass
        raise HTTPException(404, str(e))


def touch():
    state["last_activity"] = time.monotonic()


def synthesize(text, voice, **kw):
    touch()
    try:
        return engine.synthesize(text, voice, **kw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        touch()


@app.get("/health")
def health():
    return {"ok": True, "pid": os.getpid(), "model_loaded": engine.loaded,
            "default_voice": reg.default_voice_name()}


@app.get("/voices")
def get_voices():
    default = reg.default_voice_name()
    return [v.summary(default) for v in reg.list_voices()]


@app.get("/emotions")
def get_emotions():
    return sorted(EMOTION_PRESETS)


@app.post("/speak")
def speak(req: SpeakRequest):
    fmt = req.format.lower().lstrip(".")
    if fmt not in ("wav", "mp3"):
        raise HTTPException(400, "format must be wav or mp3")
    voice = resolve_voice(req.voice)
    started = time.perf_counter()
    wav, sr, settings = synthesize(req.text, voice, emotions=req.emotions, intensity=req.intensity,
                                   seed=req.seed, language=req.language)
    path = write_audio(wav, sr, req.output_path or default_output(req.text, fmt))
    if req.play and sys.platform == "darwin":
        subprocess.Popen(["afplay", str(path)])
    return {
        "path": str(path),
        "duration_s": round(len(wav) / sr, 2),
        "generation_s": round(time.perf_counter() - started, 2),
        "sample_rate": sr,
        **settings,
    }


@app.post("/speak_batch")
def speak_batch(req: SpeakBatchRequest):
    fmt = req.format.lower().lstrip(".")
    if fmt not in ("wav", "mp3"):
        raise HTTPException(400, "format must be wav or mp3")
    if req.output_paths is not None and len(req.output_paths) != len(req.texts):
        raise HTTPException(400, "output_paths must match texts")
    voice = resolve_voice(req.voice)
    started = time.perf_counter()
    touch()
    try:
        wavs, sr, settings = engine.synthesize_batch(req.texts, voice, emotions=req.emotions,
                                                     intensity=req.intensity, seed=req.seed, language=req.language)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        touch()
    items = []
    for i, (text, wav) in enumerate(zip(req.texts, wavs)):
        if req.output_paths:
            out = req.output_paths[i]
        else:  # numbered, so texts with the same opening words don't overwrite each other
            out = default_output(text, fmt)
            out = out.with_stem(f"{out.stem}-{i + 1}")
        path = write_audio(wav, sr, out)
        items.append({"path": str(path), "duration_s": round(len(wav) / sr, 2)})
    return {"items": items, "generation_s": round(time.perf_counter() - started, 2), "sample_rate": sr, **settings}


@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest):
    fmt = req.response_format.lower()
    if fmt not in ("wav", "mp3"):
        raise HTTPException(400, "response_format must be wav or mp3")
    # Unknown names (e.g. OpenAI's "alloy") fall back to the default voice so stock
    # OpenAI clients work unchanged.
    voice = resolve_voice(req.voice, fallback_to_default=True)
    wav, sr, settings = synthesize(req.input, voice)
    with tempfile.TemporaryDirectory() as tmp:
        data = write_audio(wav, sr, Path(tmp) / f"speech.{fmt}").read_bytes()
    return Response(data, media_type="audio/mpeg" if fmt == "mp3" else "audio/wav",
                    headers={"X-Qwen-Voice": settings["voice"]})


@app.post("/shutdown")
def shutdown():
    if state["server"]:
        state["server"].should_exit = True
    return {"ok": True}


def _warm_up():
    try:
        name = reg.default_voice_name()
        engine.warm(reg.get_voice(name) if name else None)
        print(f"model ready (default voice: {name})", flush=True)
    except Exception as e:  # first /speak will surface the real error
        print(f"warm-up failed: {e}", flush=True)


def _idle_watch(server, idle_s):
    while not server.should_exit:
        time.sleep(15)
        if not engine.lock.locked() and time.monotonic() - state["last_activity"] > idle_s:
            print(f"idle for {idle_s // 60} min, exiting", flush=True)
            server.should_exit = True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qwen-voice serve")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--idle-minutes", type=float,
                    default=float(os.environ.get("QWEN_VOICE_IDLE_MINUTES", "15")),
                    help="exit after this long without requests (0 = never)")
    ap.add_argument("--no-warm", action="store_true", help="load the model on first request")
    a = ap.parse_args(argv)

    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=a.port, log_level="warning"))
    state["server"] = server
    if not a.no_warm:
        threading.Thread(target=_warm_up, daemon=True).start()
    if a.idle_minutes > 0:
        threading.Thread(target=_idle_watch, args=(server, int(a.idle_minutes * 60)), daemon=True).start()
    print(f"qwen-voice listening on http://{HOST}:{a.port} (pid {os.getpid()})", flush=True)
    server.run()


if __name__ == "__main__":
    main()
