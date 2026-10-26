"""Emotion presets and mixing, matching FB_Qwen3TTSEmotionalVoiceClone in
ComfyUI-Qwen3TTS-Emotional. Each emotion nudges the sampling parameters; a mix averages
the nudges and scales them by intensity."""

BASE_TEMPERATURE = 1.0
BASE_TOP_P = 0.8
BASE_TOP_K = 20
BASE_REPETITION_PENALTY = 1.05

EMOTION_PRESETS = {
    # Core emotions
    "neutral": {"temp": 0.0, "rep_pen": 0.0, "top_p": 0.0},
    "angry": {"temp": 0.3, "rep_pen": 0.1, "top_p": 0.05},
    "sad": {"temp": -0.2, "rep_pen": -0.05, "top_p": -0.1},
    "happy": {"temp": 0.2, "rep_pen": 0.05, "top_p": 0.05},
    "fearful": {"temp": 0.25, "rep_pen": 0.15, "top_p": 0.0},
    "disgusted": {"temp": 0.1, "rep_pen": 0.1, "top_p": -0.05},
    "surprised": {"temp": 0.35, "rep_pen": 0.1, "top_p": 0.1},

    # Intensity variations
    "furious": {"temp": 0.5, "rep_pen": 0.2, "top_p": 0.1},
    "enraged": {"temp": 0.6, "rep_pen": 0.25, "top_p": 0.15},
    "annoyed": {"temp": 0.15, "rep_pen": 0.05, "top_p": 0.0},
    "melancholic": {"temp": -0.3, "rep_pen": -0.1, "top_p": -0.15},
    "disappointed": {"temp": -0.15, "rep_pen": -0.05, "top_p": -0.05},
    "ecstatic": {"temp": 0.5, "rep_pen": 0.2, "top_p": 0.15},
    "content": {"temp": -0.1, "rep_pen": 0.0, "top_p": 0.0},
    "terrified": {"temp": 0.4, "rep_pen": 0.25, "top_p": 0.05},
    "anxious": {"temp": 0.2, "rep_pen": 0.15, "top_p": 0.0},

    # Social/Interpersonal
    "sarcastic": {"temp": 0.15, "rep_pen": 0.05, "top_p": 0.0},
    "mocking": {"temp": 0.2, "rep_pen": 0.1, "top_p": 0.05},
    "condescending": {"temp": 0.1, "rep_pen": 0.05, "top_p": -0.05},
    "dismissive": {"temp": 0.05, "rep_pen": 0.0, "top_p": -0.1},
    "contemptuous": {"temp": 0.15, "rep_pen": 0.1, "top_p": -0.05},
    "sympathetic": {"temp": -0.1, "rep_pen": -0.05, "top_p": 0.0},
    "apologetic": {"temp": -0.15, "rep_pen": -0.05, "top_p": -0.05},
    "pleading": {"temp": 0.1, "rep_pen": 0.1, "top_p": 0.0},
    "begging": {"temp": 0.2, "rep_pen": 0.15, "top_p": 0.05},

    # Confidence spectrum
    "confident": {"temp": -0.1, "rep_pen": -0.05, "top_p": 0.0},
    "arrogant": {"temp": 0.0, "rep_pen": 0.0, "top_p": -0.1},
    "smug": {"temp": 0.05, "rep_pen": 0.0, "top_p": -0.05},
    "insecure": {"temp": 0.1, "rep_pen": 0.1, "top_p": 0.0},
    "hesitant": {"temp": 0.15, "rep_pen": 0.15, "top_p": 0.0},
    "timid": {"temp": -0.2, "rep_pen": 0.1, "top_p": -0.1},
    "defeated": {"temp": -0.25, "rep_pen": -0.1, "top_p": -0.1},
    "hopeless": {"temp": -0.3, "rep_pen": -0.15, "top_p": -0.15},

    # Energy levels
    "excited": {"temp": 0.4, "rep_pen": 0.15, "top_p": 0.1},
    "enthusiastic": {"temp": 0.3, "rep_pen": 0.1, "top_p": 0.05},
    "energetic": {"temp": 0.35, "rep_pen": 0.15, "top_p": 0.1},
    "bored": {"temp": -0.2, "rep_pen": -0.1, "top_p": -0.1},
    "tired": {"temp": -0.25, "rep_pen": -0.1, "top_p": -0.1},
    "exhausted": {"temp": -0.3, "rep_pen": -0.15, "top_p": -0.15},
    "lethargic": {"temp": -0.35, "rep_pen": -0.15, "top_p": -0.15},
    "calm": {"temp": -0.3, "rep_pen": -0.1, "top_p": -0.15},

    # Stress/Tension
    "nervous": {"temp": 0.2, "rep_pen": 0.2, "top_p": 0.05},
    "panicked": {"temp": 0.45, "rep_pen": 0.25, "top_p": 0.1},
    "frantic": {"temp": 0.5, "rep_pen": 0.3, "top_p": 0.15},
    "stressed": {"temp": 0.25, "rep_pen": 0.15, "top_p": 0.05},
    "overwhelmed": {"temp": 0.3, "rep_pen": 0.2, "top_p": 0.05},
    "relaxed": {"temp": -0.25, "rep_pen": -0.1, "top_p": -0.1},
    "serene": {"temp": -0.35, "rep_pen": -0.15, "top_p": -0.15},

    # Attitude/Mood
    "playful": {"temp": 0.25, "rep_pen": 0.1, "top_p": 0.05},
    "teasing": {"temp": 0.2, "rep_pen": 0.1, "top_p": 0.05},
    "flirtatious": {"temp": 0.15, "rep_pen": 0.05, "top_p": 0.05},
    "seductive": {"temp": 0.1, "rep_pen": 0.0, "top_p": 0.0},
    "mysterious": {"temp": -0.1, "rep_pen": 0.0, "top_p": -0.05},
    "ominous": {"temp": -0.15, "rep_pen": 0.05, "top_p": -0.1},
    "menacing": {"temp": 0.1, "rep_pen": 0.1, "top_p": -0.05},
    "threatening": {"temp": 0.2, "rep_pen": 0.15, "top_p": 0.0},
    "suspicious": {"temp": 0.1, "rep_pen": 0.1, "top_p": 0.0},
    "paranoid": {"temp": 0.25, "rep_pen": 0.2, "top_p": 0.05},

    # Physical states
    "drunk": {"temp": 0.4, "rep_pen": 0.25, "top_p": 0.1},
    "breathless": {"temp": 0.3, "rep_pen": 0.2, "top_p": 0.1},
    "whispering": {"temp": -0.2, "rep_pen": -0.05, "top_p": -0.1},
    "shouting": {"temp": 0.4, "rep_pen": 0.2, "top_p": 0.1},
    "in_pain": {"temp": 0.3, "rep_pen": 0.2, "top_p": 0.05},
    "sick": {"temp": -0.2, "rep_pen": -0.05, "top_p": -0.1},

    # Complex emotions (pre-mixed)
    "bitter": {"temp": 0.1, "rep_pen": 0.05, "top_p": -0.05},  # sad + angry
    "manic": {"temp": 0.5, "rep_pen": 0.3, "top_p": 0.15},  # excited + nervous
    "reluctant": {"temp": -0.1, "rep_pen": 0.05, "top_p": -0.05},  # defeated + nervous
    "desperate": {"temp": 0.3, "rep_pen": 0.2, "top_p": 0.05},  # fearful + pleading
    "jealous": {"temp": 0.2, "rep_pen": 0.1, "top_p": 0.0},  # angry + sad
    "frustrated": {"temp": 0.25, "rep_pen": 0.1, "top_p": 0.0},  # angry + defeated
    "relieved": {"temp": -0.15, "rep_pen": -0.05, "top_p": 0.0},  # happy + calm
    "guilty": {"temp": -0.1, "rep_pen": 0.05, "top_p": -0.05},  # sad + nervous
    "ashamed": {"temp": -0.2, "rep_pen": -0.05, "top_p": -0.1},  # defeated + sad
    "proud": {"temp": 0.0, "rep_pen": 0.0, "top_p": 0.0},  # confident + happy
    "vengeful": {"temp": 0.3, "rep_pen": 0.15, "top_p": 0.05},  # angry + confident
    "remorseful": {"temp": -0.15, "rep_pen": -0.05, "top_p": -0.05},  # sad + apologetic

    # Character-specific
    "rick_like": {"temp": 0.3, "rep_pen": 0.15, "top_p": 0.05},  # condescending + drunk + dismissive
    "jerry_like": {"temp": 0.1, "rep_pen": 0.15, "top_p": 0.0},  # nervous + insecure + apologetic
    "morty_like": {"temp": 0.25, "rep_pen": 0.25, "top_p": 0.05},  # anxious + overwhelmed
}


def mix(emotions, intensity=1.0):
    """Return (temperature, top_p, repetition_penalty) for a list of emotion names."""
    unknown = [e for e in emotions if e not in EMOTION_PRESETS]
    if unknown:
        raise ValueError(f"Unknown emotion(s) {unknown}; see `qwen-voice emotions`")
    mods = [EMOTION_PRESETS[e] for e in emotions] or [EMOTION_PRESETS["neutral"]]
    n = len(mods)
    t = sum(m["temp"] for m in mods) / n * intensity
    r = sum(m["rep_pen"] for m in mods) / n * intensity
    p = sum(m["top_p"] for m in mods) / n * intensity
    return (
        max(0.1, min(2.0, BASE_TEMPERATURE + t)),
        max(0.1, min(1.0, BASE_TOP_P + p)),
        max(1.0, min(2.0, BASE_REPETITION_PENALTY + r)),
    )
