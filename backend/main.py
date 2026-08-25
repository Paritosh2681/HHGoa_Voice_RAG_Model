"""FastAPI application for HH GOA Voice RAG.

Routes:
  GET  /                  -> frontend (HH GOA design)
  GET  /api/health        -> liveness + provider status
  GET  /api/index-info    -> index & chunking inventory
  POST /api/ask           -> JSON answer
  POST /api/ask/stream    -> SSE answer (streaming tokens)
  POST /api/stt           -> audio -> transcript
  POST /api/voice/stream  -> audio -> STT -> RAG -> SSE (end-to-end voice)
  GET  /api/metrics       -> P50/P70/P100 latency summary
  POST /api/metrics/reset -> clear rolling window
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import config as C
from .harness import PipelineHarness, StreamEvent
from .ingest import build_index, describe_index
from .latency import store
from .models import AskRequest, AskResponse, now_iso
from .stt import provider_status, transcribe
from .tts import synthesize_sarvam

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


app = FastAPI(title="HH GOA Voice RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _on_startup():
    import logging, threading
    _log = logging.getLogger("uvicorn")
    
    def _bg_warmup():
        try:
            _log.info("[startup-bg] Preloading index + harness (FAISS warmup)...")
            _get_harness()
            _log.info("[startup-bg] Ready!")
        except Exception as e:
            _log.error(f"[startup-bg] Warmup error: {e}")
            
    threading.Thread(target=_bg_warmup, daemon=True).start()
    _log.info("[startup] Server listening immediately; background warmup started.")

_index = None
_harness: Optional[PipelineHarness] = None
_harness_lock = __import__('threading').Lock()


def _get_index():
    global _index
    if _index is None:
        _index = build_index(force=False)
    return _index


def _get_harness() -> PipelineHarness:
    global _harness
    if _harness is not None:
        return _harness
    with _harness_lock:
        if _harness is None:
            _harness = PipelineHarness(_get_index())
    return _harness


# ------------------------------------------------------------------ frontend
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/assets/{name}", include_in_schema=False)
async def asset(name: str):
    p = (ASSETS_DIR / name).resolve()
    if not p.is_relative_to(ASSETS_DIR.resolve()) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ------------------------------------------------------------------ api
@app.get("/api/health")
async def health():
    ready = _index is not None
    return {
        "status": "ok",
        "ready": ready,
        "service": "HH GOA Voice RAG",
        "stt": provider_status(),
        "index": describe_index(_index) if ready else {"status": "warming_up", "n_chunks": 0},
        "version": "1.0.0",
    }


@app.get("/api/index-info")
async def index_info():
    return describe_index(_get_index())


@app.get("/api/metrics")
async def metrics():
    return store.summary().model_dump()


@app.post("/api/metrics/reset")
async def metrics_reset():
    store.clear()
    return {"ok": True}


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    return await _get_harness().run(req)


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    harness = _get_harness()

    async def gen():
        async for e in harness.stream(req):
            yield e.to_sse()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/stt")
async def stt(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > C.MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio too large")
    if not data:
        raise HTTPException(400, "empty audio")
    try:
        result = await transcribe(data, file.filename or "audio.webm")
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"STT provider error: {exc}")
    return result.model_dump()


@app.post("/api/voice/stream")
async def voice_stream(file: UploadFile = File(...), strategy: Optional[str] = Query(default=None)):
    data = await file.read()
    if len(data) > C.MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio too large")
    if not data:
        raise HTTPException(400, "empty audio")
    try:
        stt_result = await transcribe(data, file.filename or "audio.webm")
    except Exception as exc:
        return StreamingResponse(
            StreamEvent("error", {"message": f"Speech-to-text failed: {exc}"}).to_sse(),
            media_type="text/event-stream")
    if not stt_result.transcript.strip():
        return StreamingResponse(
            StreamEvent("error", {"message": "No speech detected in the audio."}).to_sse(),
            media_type="text/event-stream")

    harness = _get_harness()

    async def gen():
        yield StreamEvent("stt", {"transcript": stt_result.transcript,
                                  "provider": stt_result.provider,
                                  "language": stt_result.language_code,
                                  "stt_ms": stt_result.latency_ms}).to_sse()
        async for e in harness.stream(AskRequest(text=stt_result.transcript, strategy=strategy)):
            yield e.to_sse()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class TTSRequest(BaseModel):
    text: str
    target_lang: Optional[str] = None


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Text is empty")
    try:
        audio_bytes, mime = await synthesize_sarvam(req.text, req.target_lang)
        return Response(content=audio_bytes, media_type=mime)
    except Exception as exc:
        raise HTTPException(500, f"TTS synthesis failed: {exc}")


@app.get("/{tail:path}", include_in_schema=False)
async def static_tail(tail: str):
    """Catch-all for frontend files — registered AFTER /api routes on purpose."""
    p = (FRONTEND_DIR / tail).resolve()
    if not p.is_relative_to(FRONTEND_DIR.resolve()) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


def main():
    import uvicorn
    uvicorn.run(app, host=C.HOST, port=C.PORT)


if __name__ == "__main__":
    main()
