"""qwen-voice command line. Every command prints JSON with --json for agents."""

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import voices as reg
from .emotions import EMOTION_PRESETS


def _emotions(value):
    return [e.strip() for e in value.split(",") if e.strip() and e.strip() != "none"] if value else None


def cmd_speak(a):
    from . import client

    text = sys.stdin.read() if a.text == "-" else a.text
    result = client.speak(text, voice=a.voice, emotions=_emotions(a.emotions), intensity=a.intensity,
                          seed=a.seed, language=a.language, format=a.format,
                          output_path=str(Path(a.output).expanduser().resolve()) if a.output else None,
                          play=a.play)
    print(json.dumps(result, indent=2) if a.json else result["path"])


def cmd_voices(a):
    default = reg.default_voice_name()
    voices = [v.summary(default) for v in reg.list_voices()]
    if a.json:
        print(json.dumps(voices, indent=2, ensure_ascii=False))
        return
    if not voices:
        print("No voices yet. Add one: qwen-voice add-voice NAME clip.wav --transcript '...'")
    for v in voices:
        star = "*" if v["default"] else " "
        style = "+".join(v["emotions"]) + f" x{v['intensity']:g} seed {v['seed']}"
        print(f"{star} {v['name']:<16} {v['mode']:<8} {style:<44} {v['description']}")


def cmd_add_voice(a):
    transcript = a.transcript
    if a.transcript_file:
        transcript = Path(a.transcript_file).expanduser().read_text()
    voice = reg.add_voice(a.name, a.audio, transcript=transcript, language=a.language,
                          emotions=_emotions(a.emotions), intensity=a.intensity, seed=a.seed,
                          description=a.description, make_default=a.default, overwrite=a.force)
    print(json.dumps(voice.summary(reg.default_voice_name()), indent=2, ensure_ascii=False))


def cmd_remove_voice(a):
    reg.remove_voice(a.name)
    print(f"removed {a.name}")


def cmd_default(a):
    reg.set_default(a.name)
    print(f"default voice: {a.name}")


def cmd_emotions(a):
    names = sorted(EMOTION_PRESETS)
    print(json.dumps(names) if a.json else "\n".join(names))


def cmd_serve(a):
    from . import server

    server.main(a.server_args)


def cmd_status(a):
    from . import client

    h = client.health()
    print(json.dumps(h or {"ok": False, "running": False}, indent=2))


def cmd_stop(a):
    from . import client

    print("stopped" if client.stop() else "not running")


def cmd_mcp_config(a):
    exe = shutil.which("qwen-voice-mcp") or str(Path(sys.executable).with_name("qwen-voice-mcp"))
    print(json.dumps({"mcpServers": {"qwen-voice": {"command": exe}}}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qwen-voice", description="Local Qwen3-TTS voice cloning")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("speak", help="synthesize text; prints the output file path")
    p.add_argument("text", help="text to speak, or - to read stdin")
    p.add_argument("-v", "--voice", help="registered voice name (default voice if omitted)")
    p.add_argument("-e", "--emotions", help="comma-separated, e.g. calm,confident (voice default if omitted)")
    p.add_argument("-i", "--intensity", type=float)
    p.add_argument("-s", "--seed", type=int, help="same seed = same take")
    p.add_argument("-l", "--language")
    p.add_argument("-o", "--output", help="output .wav or .mp3 path")
    p.add_argument("-f", "--format", default="wav", choices=["wav", "mp3"])
    p.add_argument("--play", action="store_true", help="also play it on this Mac's speakers")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("voices", help="list registered voices (* = default)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("add-voice", help="register a reference clip as a named voice")
    p.add_argument("name")
    p.add_argument("audio", help="clean 10-60 s clip of the voice")
    p.add_argument("-t", "--transcript", help="exact words in the clip (strongly recommended)")
    p.add_argument("--transcript-file")
    p.add_argument("-e", "--emotions", default="neutral", help="default style, e.g. confident,playful")
    p.add_argument("-i", "--intensity", type=float, default=1.0)
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("-l", "--language", default="English")
    p.add_argument("-d", "--description", default="")
    p.add_argument("--default", action="store_true", help="make it the default voice")
    p.add_argument("--force", action="store_true", help="replace an existing voice")
    p.set_defaults(func=cmd_add_voice)

    p = sub.add_parser("remove-voice")
    p.add_argument("name")
    p.set_defaults(func=cmd_remove_voice)

    p = sub.add_parser("default", help="set the default voice")
    p.add_argument("name")
    p.set_defaults(func=cmd_default)

    p = sub.add_parser("emotions", help="list emotion presets")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_emotions)

    p = sub.add_parser("serve", help="run the daemon in the foreground (clients start it automatically)")
    p.add_argument("server_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_serve)

    sub.add_parser("status", help="daemon health").set_defaults(func=cmd_status)
    sub.add_parser("stop", help="stop the daemon and free its memory").set_defaults(func=cmd_stop)
    sub.add_parser("mcp-config", help="print the MCP client config snippet").set_defaults(func=cmd_mcp_config)

    a = ap.parse_args(argv)
    try:
        a.func(a)
    except (LookupError, ValueError, FileNotFoundError, FileExistsError, RuntimeError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
