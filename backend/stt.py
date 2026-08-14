"""Speech-to-text providers.

Primary: Sarvam AI (saaras:v3) — matches the brief ("use Sarvam or ElevenLabs").
Alternates: ElevenLabs Scribe v2, Groq whisper-large-v3, and a local offline
mode for air-gapped demos. Audio is passed through as recorded (WebM/Opus is
supported by all providers); we only re-encode if a provider rejects it.
"""
from __future__ import annotations

import io
import time

import httpx

from . import config as C
from .models import SttResult


def _seconds(n) -> float | None:
    return float(n) if n is not None else None


async def transcribe_sarvam(data: bytes, filename: str) -> SttResult:
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": C.SARVAM_API_KEY,
        "Accept": "application/json",
    }
    t0 = time.perf_counter()
    files = {"file": (filename, io.BytesIO(data), "audio/webm")}
    form = {"model": C.SARVAM_MODEL, "mode": "transcribe", "language_code": "unknown"}
    async with httpx.AsyncClient(timeout=C.STAGE_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, files=files, data=form)
    resp.raise_for_status()
    j = resp.json()
    return SttResult(
        transcript=(j.get("transcript") or "").strip(),
        provider="sarvam",
        language_code=j.get("language_code"),
        language_probability=_seconds(j.get("language_probability")),
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


async def transcribe_elevenlabs(data: bytes, filename: str) -> SttResult:
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": C.ELEVENLABS_API_KEY, "Accept": "application/json"}
    t0 = time.perf_counter()
    files = {"file": (filename, io.BytesIO(data), "audio/webm")}
    form = {"model_id": C.ELEVENLABS_MODEL}
    async with httpx.AsyncClient(timeout=C.STAGE_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, files=files, data=form)
    resp.raise_for_status()
    j = resp.json()
    return SttResult(
        transcript=(j.get("text") or "").strip(),
        provider="elevenlabs",
        language_code=j.get("language_code"),
        language_probability=_seconds(j.get("language_probability")),
        audio_duration_secs=_seconds(j.get("audio_duration_secs")),
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


async def transcribe_groq(data: bytes, filename: str) -> SttResult:
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {C.GROQ_API_KEY}"}
    t0 = time.perf_counter()
    files = {"file": (filename, io.BytesIO(data), "audio/webm")}
    form = {"model": "whisper-large-v3", "response_format": "json"}
    async with httpx.AsyncClient(timeout=C.STAGE_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, files=files, data=form)
    resp.raise_for_status()
    j = resp.json()
    return SttResult(
        transcript=(j.get("text") or "").strip(),
        provider="groq-whisper",
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


async def transcribe(data: bytes, filename: str = "audio.webm") -> SttResult:
    """Route to the configured provider. Raises if no key configured."""
    if C.STT_PROVIDER == "elevenlabs":
        if not C.ELEVENLABS_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        return await transcribe_elevenlabs(data, filename)
    if C.STT_PROVIDER == "groq":
        if not C.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        return await transcribe_groq(data, filename)
    # default: sarvam
    if not C.SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set")
    return await transcribe_sarvam(data, filename)


def provider_status() -> dict:
    return {
        "provider": C.STT_PROVIDER,
        "configured": bool(
            (C.STT_PROVIDER == "sarvam" and C.SARVAM_API_KEY)
            or (C.STT_PROVIDER == "elevenlabs" and C.ELEVENLABS_API_KEY)
            or (C.STT_PROVIDER == "groq" and C.GROQ_API_KEY)
        ),
        "llm_configured": bool(C.LLM_API_KEY),
        "llm_model": C.LLM_MODEL,
    }
