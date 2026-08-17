"""The harness: structured orchestration around the model.

Requirements it satisfies:
- structured I/O (Pydantic contracts in/out, not raw strings)
- retries with exponential backoff + jitter for external calls
- per-stage timeouts and tracing
- error recovery (llm -> extractive -> refused, never a bare 500)
- latency capture per stage
- optional streaming event generator for the frontend

Pipeline:  guard -> embed -> retrieve -> rerank/gate -> generate -> verify
"""
from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import httpx

from . import config as C
from .embeddings import get_embedding_provider
from .guardrails import check_input, grounding_score, refusal
from .index_store import HybridIndex
from .latency import MetricPoint, store
from .models import AskRequest, AskResponse, GuardResult, RetrievedDoc, now_iso


@dataclass
class StreamEvent:
    type: str          # stage | guard | sources | answer_start | chunk | done | error
    payload: dict = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"data: {json.dumps({'type': self.type, **self.payload}, ensure_ascii=False)}\n\n"


class PipelineHarness:
    def __init__(self, index: HybridIndex) -> None:
        self.index = index
        self._emb = get_embedding_provider(C.EMBED_MODEL)
        self._client: Optional[httpx.AsyncClient] = None
        self._answer_cache: dict[str, str] = {}

    def _get_client(self) -> httpx.AsyncClient:
        """Persistent connection pool (keep-alive) — avoids TLS + TCP setup on
        every call, which alone costs hundreds of ms per request."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=C.LLM_TIMEOUT,
                                             limits=httpx.Limits(max_keepalive_connections=4))
        return self._client

    # ------------------------------------------------------------- stage runs
    async def _run_llm(self, messages: list[dict], retries: int = C.MAX_LLM_RETRIES,
                       on_tokens: Optional[Callable[[str], None]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        if C.LLM_API_KEY and "localhost" not in C.LLM_BASE_URL and "127.0.0.1" not in C.LLM_BASE_URL:
            headers["Authorization"] = f"Bearer {C.LLM_API_KEY}"
        body = {
            "model": C.LLM_MODEL,
            "messages": messages,
            "temperature": C.LLM_TEMPERATURE,
            "max_tokens": C.LLM_MAX_TOKENS,
            "stream": on_tokens is not None,
        }
        client = self._get_client()
        attempt = 0
        backoff = 0.4
        while True:
            try:
                if on_tokens is None:
                    resp = await client.post(f"{C.LLM_BASE_URL}/chat/completions",
                                             headers=headers, json=body)
                    resp.raise_for_status()
                    j = resp.json()
                    return (j["choices"][0]["message"]["content"] or "").strip()
                # streaming
                chunks: list[str] = []
                async with client.stream("POST", f"{C.LLM_BASE_URL}/chat/completions",
                                         headers=headers, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj["choices"][0]["delta"].get("content", "")
                        except Exception:
                            continue
                        if delta:
                            chunks.append(delta)
                            on_tokens(delta)
                return "".join(chunks).strip()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (429, 500, 502, 503, 504) and attempt < retries:
                    await asyncio.sleep(backoff + random.uniform(0, 0.15))
                    backoff *= 2
                    attempt += 1
                    continue
                raise RuntimeError(f"LLM provider error (HTTP {status})") from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt < retries:
                    await asyncio.sleep(backoff + random.uniform(0, 0.15))
                    backoff *= 2
                    attempt += 1
                    continue
                raise RuntimeError(f"LLM provider unreachable: {exc}") from exc

    # ------------------------------------------------------------- generation
    def _build_prompt(self, query: str, docs: list[RetrievedDoc]) -> list[dict]:
        context = "\n\n".join(
            f"[SOURCE {i + 1}] ({d.strategy}/{d.source})\n{d.text}" for i, d in enumerate(docs)
        )
        system = (
            "You are a grounded retrieval assistant. Answer the question using ONLY the "
            "provided [SOURCE] passages. Rules:\n"
            "- If the sources do not contain the answer, say you cannot answer from the given context.\n"
            "- Answer in the SAME LANGUAGE as the question (Hindi/English or as asked).\n"
            "- Be concise: 1-4 sentences. Never invent facts.\n"
            "- Do not mention that you are reading sources; just answer."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"QUESTION: {query}\n\nCONTEXT:\n{context}"},
        ]

    def _extractive_answer(self, query: str, docs: list[RetrievedDoc]) -> str:
        """Fast local fallback: cite the strongest passage directly."""
        if not docs:
            return "I couldn't find anything relevant in the knowledge base."
        return docs[0].text.strip()

    # ------------------------------------------------------------- main entry
    async def run(self, req: AskRequest,
                  emit: Optional[Callable[[StreamEvent], None]] = None,
                  record_metrics: bool = True) -> AskResponse:
        rid = req.request_id or f"hhg-{uuid.uuid4().hex[:10]}"
        times: dict[str, float] = {}
        pipeline: list[str] = []

        def _emit(e: StreamEvent):
            if emit:
                emit(e)

        # -- 1. guardrails -----------------------------------------------
        t = time.perf_counter()
        guard: GuardResult = check_input(req.text)
        times["guard"] = (time.perf_counter() - t) * 1000.0
        pipeline.append("guard")
        _emit(StreamEvent("stage", {"stage": "guard", "ms": times["guard"]}))
        if not guard.input_ok:
            pipeline.append("refuse")
            times["total"] = (time.perf_counter() - t) * 1000.0
            _emit(StreamEvent("guard_result", {"ok": False, "code": guard.reject_code,
                                               "reasons": guard.reasons}))
            resp = AskResponse(request_id=rid, query=req.text,
                               answer="I can't process that. " + "; ".join(guard.reasons),
                               mode="refused", grounded=False, guardrails=guard,
                               latency_ms=times, total_ms=times["total"],
                               pipeline=pipeline, created_at=now_iso())
            if record_metrics:
                store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms,
                                         stages=times, mode="refused", grounded=False,
                                         created_at=now_iso()))
            _emit(StreamEvent("done", resp.model_dump()))
            return resp
        _emit(StreamEvent("guard_result", {"ok": True}))

        # -- 2. embed ----------------------------------------------------
        t = time.perf_counter()
        query_vec = await asyncio.to_thread(self._emb.embed_one, guard.normalized)
        times["embed"] = (time.perf_counter() - t) * 1000.0
        pipeline.append("embed")
        _emit(StreamEvent("stage", {"stage": "embed", "ms": times["embed"]}))

        # -- 3. retrieve --------------------------------------------------
        t = time.perf_counter()
        docs = await asyncio.to_thread(
            self.index.search, guard.normalized, query_vec, req.strategy, C.TOP_K)
        times["retrieve"] = (time.perf_counter() - t) * 1000.0
        pipeline.append("retrieve")
        _emit(StreamEvent("stage", {"stage": "retrieve", "ms": times["retrieve"]}))
        _emit(StreamEvent("sources", {"docs": [d.model_dump() for d in docs[: C.RERANK_TOP] if d.score >= 0.01]}))

        # -- 4. gate (off-topic) -------------------------------------------
        t = time.perf_counter()
        should_refuse, reason = refusal(docs, query=guard.normalized)
        times["gate"] = (time.perf_counter() - t) * 1000.0
        pipeline.append("gate")
        _emit(StreamEvent("stage", {"stage": "gate", "ms": times["gate"]}))

        if should_refuse:
            pipeline.append("refuse")
            times["total"] = (time.perf_counter() - t) * 1000.0
            _emit(StreamEvent("refuse", {"reason": reason}))
            resp = AskResponse(request_id=rid, query=req.text, answer=reason,
                               mode="refused", grounded=False, guardrails=guard,
                               sources=docs[:2], latency_ms=times, total_ms=times["total"],
                               pipeline=pipeline, created_at=now_iso())
            if record_metrics:
                store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms,
                                         stages=times, mode="refused", grounded=False,
                                         created_at=now_iso()))
            _emit(StreamEvent("done", resp.model_dump()))
            return resp

        # -- 5. generate ---------------------------------------------------
        # Fast-path: confident retrieval -> extractive answer in ~0 ms, keeping
        # the whole pipeline inside the 200 ms budget. LLM is reserved for
        # weak retrievals (or when fast-path is disabled).
        mode = "llm"
        answer = ""
        _emit(StreamEvent("answer_start", {}))
        t = time.perf_counter()
        try:
            top_score = docs[0].score if docs else 0.0
            fast_ok = C.FAST_PATH and top_score >= C.FAST_PATH_MIN_SCORE
            if C.LLM_API_KEY and not fast_ok:
                cache_key = guard.normalized
                cached = self._answer_cache.get(cache_key)
                if cached is not None:
                    answer = cached
                else:
                    def _on_token(delta: str):
                        _emit(StreamEvent("chunk", {"delta": delta}))

                    # feed the LLM fewer, best-scored chunks (less prefill = faster)
                    prompt_docs = docs[: C.RERANK_TOP]
                    answer = await self._run_llm(self._build_prompt(guard.normalized, prompt_docs),
                                                 on_tokens=_on_token)
                    if not answer:
                        raise RuntimeError("empty LLM output")
                    self._answer_cache[cache_key] = answer
            else:
                answer = self._extractive_answer(guard.normalized, docs)
                mode = "extractive"
                _emit(StreamEvent("chunk", {"delta": answer}))
        except Exception as exc:
            # error recovery: fall back to extractive before ever failing hard
            mode = "extractive"
            answer = self._extractive_answer(guard.normalized, docs)
            _emit(StreamEvent("fallback", {"reason": str(exc)}))
            _emit(StreamEvent("chunk", {"delta": answer}))
        times["generate"] = (time.perf_counter() - t) * 1000.0
        pipeline.append("generate")
        _emit(StreamEvent("stage", {"stage": "generate", "ms": times["generate"]}))

        # -- 6. verify (hallucination check) --------------------------------
        t = time.perf_counter()
        top_docs = [d for d in docs[: C.RERANK_TOP] if d.score >= 0.01]
        answer_vec = await asyncio.to_thread(self._emb.embed_one, answer)
        g = grounding_score(
            answer,
            [d.text for d in top_docs],
            self._emb,
            answer_vec=answer_vec,
            doc_vecs=self.index.vectors_for([d.id for d in top_docs]),
        )
        times["verify"] = (time.perf_counter() - t) * 1000.0
        pipeline.append("verify")
        _emit(StreamEvent("stage", {"stage": "verify", "ms": times["verify"]}))

        final_answer = answer
        if not g["grounded"] and mode == "llm":
            final_answer = answer + "\n\n(⚠ grounded confidence low — verify against sources above)"
        grounded = g["grounded"]

        times["total"] = round(sum(times.values()), 2)
        resp = AskResponse(
            request_id=rid, query=req.text, answer=final_answer, mode=mode,
            grounded=grounded, guardrails=guard,
            sources=top_docs,
            latency_ms=times, total_ms=times["total"], pipeline=pipeline,
            created_at=now_iso(),
        )
        if record_metrics:
            store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms,
                                     stages=times, mode=mode, grounded=grounded,
                                     created_at=now_iso()))
        _emit(StreamEvent("done", resp.model_dump()))
        return resp

    # ------------------------------------------------------------- streaming
    async def stream(self, req: AskRequest) -> AsyncIterator[StreamEvent]:
        events: list[StreamEvent] = []

        def _emit(e: StreamEvent):
            events.append(e)

        await self.run(req, emit=_emit)
        for e in events:
            yield e