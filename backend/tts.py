"""Text-to-speech (TTS) integration.

Primary: Sarvam AI (bulbul:v1) — for natural Indian-accent Hindi and English voice output.
Fallback: Returns status indicating client should use browser Web Speech API.
"""
from __future__ import annotations

import base64
import time
from typing import Optional

import httpx

from . import config as C


def is_marathi(text: str) -> bool:
    """Detect if text contains typical Marathi words or markers."""
    marathi_markers = [
        "आहे", "आहेत", "नाही", "गोव्यात", "कुठे", "कोणते", "कोणती", "कोणता", "कधी",
        "कसे", "कशी", "कसा", "सांगा", "फिरण्यासाठी", "धबधबा", "समुद्रकिनारे", "खाद्यपदार्थ",
        "सण", "उत्सव", "पर्यटन", "किंवा", "झाले", "होते", "करायचे", "पाहिजे", "म्हणून"
    ]
    t = text.lower()
    return any(w in t for w in marathi_markers)


def is_devanagari(text: str) -> bool:
    """Detect if text contains Devanagari characters."""
    for char in text:
        if '\u0900' <= char <= '\u097F':
            return True
    return False


_tts_client: Optional[httpx.AsyncClient] = None


def _get_tts_client() -> httpx.AsyncClient:
    global _tts_client
    if _tts_client is None:
        _tts_client = httpx.AsyncClient(
            timeout=8.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _tts_client


async def synthesize_sarvam(text: str, target_lang: Optional[str] = None) -> tuple[bytes, str]:
    """Generate speech audio from text using Sarvam bulbul:v2.
    
    Supports Hindi (hi-IN), Marathi (mr-IN), and English (en-IN).
    Returns (audio_bytes, mime_type).
    """
    if not C.SARVAM_API_KEY:
        raise ValueError("Sarvam API key is not configured")

    clean_text = text.strip()[:480]
    if not clean_text:
        raise ValueError("Text is empty")

    if not target_lang:
        if is_marathi(clean_text):
            target_lang = "mr-IN"
        elif is_devanagari(clean_text):
            target_lang = "hi-IN"
        else:
            target_lang = "en-IN"

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": C.SARVAM_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "inputs": [clean_text],
        "target_language_code": target_lang,
        "speaker": "anushka",
        "pitch": 0,
        "pace": 1.05,
        "loudness": 1.5,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
        "model": "bulbul:v2"
    }

    client = _get_tts_client()
    resp = await client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    audios = data.get("audios", [])
    if not audios:
        raise RuntimeError("No audio returned by Sarvam TTS")

    audio_bytes = base64.b64decode(audios[0])
    return audio_bytes, "audio/wav"
