from __future__ import annotations

import base64
import io
import math
import struct
import wave
from functools import lru_cache


def build_banner_html(text: str, active: bool) -> str:
    if active:
        background = "#7f1d1d"
        border = "#ef4444"
        label = "ALERT"
    else:
        background = "#0f766e"
        border = "#2dd4bf"
        label = "MONITORING"

    return f"""
    <div style="
        margin: 0.5rem 0 1rem 0;
        padding: 0.9rem 1.1rem;
        border-radius: 0.8rem;
        border: 2px solid {border};
        background: {background};
        color: white;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
    ">
        {label}: {text}
    </div>
    """


@lru_cache(maxsize=1)
def get_beep_audio_html() -> str:
    sample_rate = 16000
    duration_seconds = 0.22
    frequency = 880.0
    amplitude = 0.35
    frames = int(sample_rate * duration_seconds)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for idx in range(frames):
            sample = amplitude * math.sin(2.0 * math.pi * frequency * (idx / sample_rate))
            wav_file.writeframesraw(struct.pack("<h", int(sample * 32767)))

    audio_bytes = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        "<audio autoplay>"
        f"<source src=\"data:audio/wav;base64,{audio_bytes}\" type=\"audio/wav\">"
        "</audio>"
    )
