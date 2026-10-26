# qwen-voice

Local voice cloning for agents. [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B runs on
your machine (Apple GPU, CUDA or CPU), and any agent can use it through:

- **MCP** (`qwen-voice-mcp`) for Claude Desktop, Claude Code, Cursor, Codex and any other MCP client
- **CLI** (`qwen-voice speak "…"`) for agents with a shell
- **HTTP**, including an OpenAI-compatible `POST /v1/audio/speech`

All three talk to one background daemon that keeps the model loaded. It starts on the first
request and exits after 15 idle minutes, which frees the ~10 GB the model uses. Several agents
can share it without each one loading its own copy.

```
agent ──MCP──▶ qwen-voice-mcp ─┐
agent ──CLI──▶ qwen-voice     ─┼─HTTP─▶ daemon (127.0.0.1:7731) ── Qwen3-TTS 1.7B Base
any OpenAI TTS client ─────────┘           ▲ voices from ~/.qwen-voice/voices
```

## Install

```bash
git clone https://github.com/hussainn7/qwen-voice && cd qwen-voice
uv venv --python 3.12 && uv pip install -e .
```

The model (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, ~4 GB) downloads from Hugging Face on first use.
mp3 output needs `ffmpeg` on PATH.

## Add a voice

A voice is a clean 10–60 s clip of one speaker plus, ideally, the exact words in it. With the
transcript, the clone copies the delivery and pacing as well as the timbre. Without it, only
the timbre is copied (x-vector mode).

```bash
qwen-voice add-voice narrator ~/clips/vo.wav --transcript-file ~/clips/vo.txt \
  -e confident,playful,enthusiastic --default \
  -d "Warm documentary narrator"
qwen-voice voices
```

`-e/-i/-s` set the voice's default style (emotion blend, intensity, seed). Callers can
override them per request. The same seed and settings reproduce the same take.

## Connect agents (MCP)

`qwen-voice mcp-config` prints the snippet with your absolute path.

**Claude Desktop**: add to `~/Library/Application Support/Claude/claude_desktop_config.json`, then restart:

```json
{ "mcpServers": { "qwen-voice": { "command": "/path/to/qwen-voice/.venv/bin/qwen-voice-mcp" } } }
```

**Claude Code**:

```bash
claude mcp add --scope user qwen-voice -- /path/to/qwen-voice/.venv/bin/qwen-voice-mcp
```

**Cursor** (`~/.cursor/mcp.json`) uses the same JSON as Claude Desktop. **Codex** (`~/.codex/config.toml`):

```toml
[mcp_servers.qwen-voice]
command = "/path/to/qwen-voice/.venv/bin/qwen-voice-mcp"
```

Tools:

| tool | what it does |
|---|---|
| `speak(text, voice?, emotions?, intensity?, seed?, language?, output_path?, format?, play?)` | writes a wav/mp3, returns `{path, duration_s, …settings}` |
| `list_voices()` | registered voices and their default styles |
| `list_emotions()` | the 80+ emotion presets |
| `add_voice(name, audio_path, transcript?, …)` | register a new voice |

## CLI

```bash
qwen-voice speak "Hey, I'm your agent."                  # prints the output path
qwen-voice speak - -o out.mp3 < script.txt               # long scripts from stdin
qwen-voice speak "Uh oh." -e nervous,apologetic -i 1.5 --play --json
qwen-voice voices | emotions | status | stop
```

## HTTP

```bash
curl -s localhost:7731/speak -H 'content-type: application/json' \
  -d '{"text":"Hello","voice":"narrator","format":"mp3"}'

# OpenAI-compatible: point any OpenAI TTS client at base_url http://127.0.0.1:7731/v1
curl -s localhost:7731/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"narrator","input":"Hello"}' -o hello.mp3
```

Unknown voice names such as `alloy` fall back to the default voice, so stock OpenAI clients
work without changes. The daemon only listens on 127.0.0.1.

## Writing text that sounds natural

- Commas and `…` add short pauses. Avoid long stage-direction pauses.
- Blank lines split paragraphs. Each paragraph (or each ~600 characters) is generated
  separately and the pieces are joined with a 0.3 s gap.
- Emotions shift the sampling (temperature, top-p, repetition penalty). They don't act as
  instructions, so keep blends to 1–3 related presets and intensity between 0.5 and 1.5.

## Configuration

| env | default | |
|---|---|---|
| `QWEN_VOICE_HOME` | `~/.qwen-voice` | voices, config, `out/`, `daemon.log` |
| `QWEN_VOICE_PORT` | `7731` | daemon port |
| `QWEN_VOICE_IDLE_MINUTES` | `15` | daemon exits after this long idle (0 = never) |
| `QWEN_VOICE_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | any Qwen3-TTS *Base* checkpoint |

Measured on an M4 Pro (24 GB): the first request takes ~40 s (daemon start, model load and
warm-up), a short line after that takes ~6 s, and the peak GPU memory is ~10–14 GB.

Emotion presets come from
[ComfyUI-Qwen3TTS-Emotional](https://github.com/Dawizzer/ComfyUI-Qwen3TTS-Emotional).

## License

GPL-3.0
