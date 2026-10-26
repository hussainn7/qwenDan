"""Voice registry on disk.

    ~/.qwen-voice/
      config.json                  {"default_voice": "narrator"}
      voices/<name>/voice.json     transcript, default style, description
      voices/<name>/ref.<ext>      the reference clip
      out/                         generated audio (default output location)

Set QWEN_VOICE_HOME to move it. Nothing here imports torch, so the CLI and MCP
server stay fast; only the daemon loads the model.
"""

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .emotions import EMOTION_PRESETS

HOME = Path(os.environ.get("QWEN_VOICE_HOME", Path.home() / ".qwen-voice")).expanduser()
VOICES_DIR = HOME / "voices"
OUT_DIR = HOME / "out"
CONFIG = HOME / "config.json"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


@dataclass
class Voice:
    name: str
    ref_audio: str
    # Exact words spoken in ref_audio. With it the model clones timbre *and* delivery
    # (ICL mode); without it only the speaker embedding is used (x-vector mode).
    transcript: str | None = None
    language: str = "English"
    emotions: list[str] = field(default_factory=lambda: ["neutral"])
    intensity: float = 1.0
    seed: int = 0
    description: str = ""

    @property
    def mode(self):
        return "icl" if self.transcript else "x_vector"

    def summary(self, default_name=None):
        d = asdict(self)
        d["mode"] = self.mode
        d["default"] = self.name == default_name
        return d


def _read_config():
    try:
        return json.loads(CONFIG.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_config(cfg):
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")


def list_voices():
    voices = []
    for path in sorted(VOICES_DIR.glob("*/voice.json")):
        data = json.loads(path.read_text())
        data["ref_audio"] = str(path.parent / data["ref_audio"])
        voices.append(Voice(**data))
    return voices


def default_voice_name():
    name = _read_config().get("default_voice")
    names = [v.name for v in list_voices()]
    if name in names:
        return name
    return names[0] if names else None


def get_voice(name=None):
    name = name or default_voice_name()
    if not name:
        raise LookupError("No voices registered yet; add one with `qwen-voice add-voice`")
    for voice in list_voices():
        if voice.name == name:
            return voice
    raise LookupError(f"Unknown voice {name!r}; registered: {[v.name for v in list_voices()]}")


def set_default(name):
    get_voice(name)
    cfg = _read_config()
    cfg["default_voice"] = name
    _write_config(cfg)


def add_voice(name, audio_path, transcript=None, language="English", emotions=None,
              intensity=1.0, seed=0, description="", make_default=False, overwrite=False):
    if not NAME_RE.match(name):
        raise ValueError("Voice names use lowercase letters, digits, '-' and '_'")
    src = Path(audio_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"No audio file at {src}")
    emotions = list(emotions or ["neutral"])
    unknown = [e for e in emotions if e not in EMOTION_PRESETS]
    if unknown:
        raise ValueError(f"Unknown emotion(s) {unknown}")

    vdir = VOICES_DIR / name
    if vdir.exists():
        if not overwrite:
            raise FileExistsError(f"Voice {name!r} exists; pass overwrite to replace it")
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True)
    ref_name = "ref" + (src.suffix.lower() or ".wav")
    shutil.copy2(src, vdir / ref_name)

    voice = Voice(name=name, ref_audio=ref_name, transcript=(transcript or "").strip() or None,
                  language=language, emotions=emotions, intensity=float(intensity),
                  seed=int(seed), description=description)
    (vdir / "voice.json").write_text(json.dumps(asdict(voice), indent=2, ensure_ascii=False) + "\n")
    if make_default or default_voice_name() == name:
        set_default(name)
    return get_voice(name)


def remove_voice(name):
    vdir = VOICES_DIR / name
    if not (vdir / "voice.json").exists():
        raise LookupError(f"Unknown voice {name!r}")
    shutil.rmtree(vdir)
