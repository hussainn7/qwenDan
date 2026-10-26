"""MCP server (stdio) exposing qwen-voice to agents: Claude Desktop, Claude Code, Cursor, Codex…

It is a thin client: the model lives in the shared daemon, so several agents can use
the same loaded voice without each loading ~10 GB.
"""

import asyncio
import time

from mcp.server.mcpserver import Context, MCPServer

from . import client
from . import voices as reg
from .emotions import EMOTION_PRESETS

mcp = MCPServer(
    "qwen-voice",
    instructions=(
        "Local text-to-speech with cloned voices (Qwen3-TTS, runs on this machine). "
        "Call `speak` to turn text into an audio file; it returns the file path. "
        "Omit `voice` to use the default voice and omit `emotions` to keep that voice's "
        "natural style. Commas and '…' give short natural pauses; blank lines separate "
        "paragraphs. The first call after idle takes ~15 s extra while the model loads."
    ),
    log_level="WARNING",
)


@mcp.tool()
async def speak(
    text: str,
    ctx: Context,
    voice: str | None = None,
    emotions: list[str] | None = None,
    intensity: float | None = None,
    seed: int | None = None,
    language: str | None = None,
    output_path: str | None = None,
    format: str = "wav",
    play: bool = False,
) -> dict:
    """Synthesize speech in a cloned voice and save it as an audio file.

    Args:
        text: What to say. Any length; long scripts are split and stitched automatically.
        voice: Registered voice name (see list_voices). Default voice if omitted.
        emotions: Up to three emotion presets to blend, e.g. ["calm", "confident"]
            (see list_emotions). Omit to keep the voice's own natural style.
        intensity: Emotion strength, 0.0 (neutral) to 2.0 (extreme). Voice default if omitted.
        seed: Take number. The same seed and settings reproduce the same delivery.
        language: English, Chinese, French, German, Italian, Japanese, Korean,
            Portuguese, Russian, Spanish or Auto. Voice default if omitted.
        output_path: Absolute .wav/.mp3 path to write. Defaults to ~/.qwen-voice/out/.
        format: "wav" or "mp3" when output_path is not given.
        play: Also play the audio on this computer's speakers.

    Returns the file path, duration and the exact settings used.
    """
    task = asyncio.create_task(asyncio.to_thread(
        client.speak, text, voice=voice, emotions=emotions, intensity=intensity, seed=seed,
        language=language, output_path=output_path, format=format, play=play,
    ))
    started = time.monotonic()
    # Progress notifications keep clients with request timeouts from giving up on long scripts.
    while not task.done():
        await asyncio.wait({task}, timeout=5)
        if not task.done():
            await ctx.report_progress(time.monotonic() - started, None,
                                      f"generating speech… {time.monotonic() - started:.0f}s")
    return task.result()


@mcp.tool()
def list_voices() -> list[dict]:
    """List registered voices with their default style; `default: true` marks the default voice."""
    default = reg.default_voice_name()
    return [v.summary(default) for v in reg.list_voices()]


@mcp.tool()
def list_emotions() -> list[str]:
    """List the emotion presets that `speak` accepts in `emotions`."""
    return sorted(EMOTION_PRESETS)


@mcp.tool()
def add_voice(
    name: str,
    audio_path: str,
    transcript: str | None = None,
    emotions: list[str] | None = None,
    intensity: float = 1.0,
    seed: int = 0,
    language: str = "English",
    description: str = "",
    make_default: bool = False,
) -> dict:
    """Register a reference recording as a new named voice.

    Args:
        name: lowercase id, e.g. "narrator".
        audio_path: Absolute path to a clean 10-60 s recording of one speaker.
        transcript: Exact words spoken in the recording. Strongly recommended: with it the
            clone matches delivery and pacing, without it only the timbre.
        emotions: Default style for this voice, e.g. ["confident", "playful"].
        intensity, seed, language: Defaults used when `speak` doesn't override them.
        description: Short note on what the voice is for.
        make_default: Use this voice when `speak` gets no voice.
    """
    voice = reg.add_voice(name, audio_path, transcript=transcript, language=language,
                          emotions=emotions, intensity=intensity, seed=seed,
                          description=description, make_default=make_default)
    return voice.summary(reg.default_voice_name())


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
