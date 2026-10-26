# qwenDan

Local voice cloning for AI agents. [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) runs on your
machine, and agents call it through **MCP**, the **CLI**, or an **OpenAI-compatible API**.

## Install

```bash
git clone https://github.com/hussainn7/qwenDan && cd qwenDan
uv venv --python 3.12 && uv pip install -e .
source .venv/bin/activate
```

You need about 16 GB of RAM (Apple Silicon or CUDA). The model (~4 GB) downloads on the first
run. Install `ffmpeg` if you want mp3 output.

## Add a voice

```bash
qwen-voice add-voice narrator clip.wav -t "exact words spoken in the clip" --default
```

Use a clean 10–60 s clip of one speaker. Including the transcript copies the delivery as well
as the timbre.

## Connect your agents

```bash
claude mcp add --scope user qwen-voice -- "$PWD/.venv/bin/qwen-voice-mcp"
```

For Claude Desktop, Cursor or Codex, `qwen-voice mcp-config` prints the config to paste.

## Use it

```bash
qwen-voice speak "Hey, I'm your agent." --play
qwen-voice speak - -o script.mp3 < script.txt
curl localhost:7731/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input":"Hello","voice":"narrator"}' -o hello.mp3
```

The model lives in one background daemon. It starts on the first request and exits after
15 idle minutes.

## Agent instructions quick start

Paste this into `CLAUDE.md`, `AGENTS.md` or a system prompt:

```md
## Voice (qwen-voice)
You have local text-to-speech through the `qwen-voice` MCP server.
- `speak(text, voice?, emotions?, intensity?, seed?, output_path?, format?, play?)`
  returns `{path, duration_s, ...settings}`. Hand the user the `path`.
- Omit `voice` and `emotions` to use the default voice in its natural style.
- `list_voices()` and `list_emotions()` show the options. Blend 1–3 emotions;
  keep intensity between 0.5 and 1.5.
- The same seed and settings give the same take.
- Commas and "…" make short pauses. Blank lines split paragraphs.
- The first call after idle takes ~40 s while the model loads. After that, ~6 s per line.
- Without MCP, `qwen-voice speak "text" --json` prints the same JSON.
```

## Config

| env | default |
|---|---|
| `QWEN_VOICE_HOME` | `~/.qwen-voice` (voices, outputs, logs) |
| `QWEN_VOICE_PORT` | `7731` |
| `QWEN_VOICE_IDLE_MINUTES` | `15` |

GPL-3.0
