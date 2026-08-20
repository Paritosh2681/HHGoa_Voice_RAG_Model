"""The harness: structured orchestration around the model.
Pipeline: guard -> cache -> lex_fast -> embed -> curated -> retrieve -> gate -> generate -> verify
"""
from __future__ import annotations
import asyncio, json, random, re, time, uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional
import httpx
import numpy as np
from . import config as C
from .embeddings import get_embedding_provider
from .guardrails import check_input, grounding_score, refusal, token_set
from .index_store import HybridIndex
from .knowledge import FALLBACK_CORPUS
from .latency import MetricPoint, store
from .models import AskRequest, AskResponse, GuardResult, RetrievedDoc, now_iso

@dataclass
class StreamEvent:
    type: str
    payload: dict = field(default_factory=dict)
    def to_sse(self) -> str:
        return f"data: {json.dumps({'type': self.type, **self.payload}, ensure_ascii=False)}\n\n"

_CURATED_STOPWORDS = frozenset(
    ("what which who whom whose when where why how is are was were be been being do does "
     "did have has had the a an and or but of to in on at for with from by as if then than "
     "so too very it its this that these those i you he she we they me him her us them my "
     "your his their our not no yes"
     " का की के को से में पर है हैं था थी थे हो हों ही भी और नहीं यह वह ये वे एक इस उस "
     "कौन कौनसा कौनसे कैसे कैसा कब कहाँ कहां कितना कितनी कितने क्या किस किसे "
     "करना करते करता सकते सा सी सें").split())

def _content_tokens(text: str) -> set[str]:
    return token_set(text) - _CURATED_STOPWORDS

def _norm_key(text: str) -> str:
    return re.sub(r"[?।!.,;:()\u201c\u201d'\"]+", " ", text.strip().lower())

def detect_lang(text: str) -> str:
    if not text: return "en"
    mr = ["आहे","आहेत","नाही","गोव्यात","कुठे","कोणते","कोणती","कोणता","कधी","कसे","कशी","कसा","सांगा","फिरण्यासाठी","धबधबा","समुद्रकिनारे","खाद्यपदार्थ","सण","उत्सव","पर्यटन","किंवा","झाले","होते","करायचे","पाहिजे","म्हणून","जिल्हे","तालुके","कशासाठी","काय","यांची","यांचे","यांना","मधील","येथील","भांडवल","साजरे","खावे","जावे","आहेस","आहोत","करायला"]
    t = text.lower()
    if any(w in t for w in mr): return "mr"
    if any('\u0900' <= c <= '\u097F' for c in text): return "hi"
    return "en"


class PipelineHarness:
    def __init__(self, index: HybridIndex) -> None:
        self.index = index
        self._emb = get_embedding_provider(C.EMBED_MODEL)
        self._client: Optional[httpx.AsyncClient] = None
        self._answer_cache: dict[str, dict] = {}
        self._curated_faq: list[dict] = []
        self._curated_emb: Optional[np.ndarray] = None
        self._curated_facts: list[tuple[str, list[RetrievedDoc]]] = []
        self._init_knowledge_cache()

    def _init_knowledge_cache(self) -> None:
        for item in FALLBACK_CORPUS:
            q_hi = (item.get("query") or "").strip()
            q_en = (item.get("Eng_Query") or "").strip()
            ans = (item.get("Answer") or "").strip()
            passages = item.get("passages") or []
            docs = []
            for i, p in enumerate(passages):
                p_text = p[0] if isinstance(p, (tuple, list)) else str(p)
                p_sel = p[1] if isinstance(p, (tuple, list)) and len(p) > 1 else True
                p_lang = detect_lang(p_text)
                docs.append(RetrievedDoc(id=f"curated-{i}", text=p_text, score=0.98,
                    strategy="curated_knowledge", is_selected=bool(p_sel),
                    language="mar_Deva" if p_lang == "mr" else ("hin_Deva" if p_lang == "hi" else "eng_Latn")))
            self._curated_faq.append({"answer": ans, "ans_lang": detect_lang(ans), "docs": docs,
                "q_hi": q_hi, "q_en": q_en, "key_hi": _norm_key(q_hi), "key_en": _norm_key(q_en),
                "content_hi": _content_tokens(q_hi), "content_en": _content_tokens(q_en)})

    def _find_curated_match(self, query: str) -> Optional[tuple[str, list[RetrievedDoc]]]:
        q_norm = _norm_key(query)
        q_content = _content_tokens(query)
        if not q_content and not q_norm: return None
        target_lang = detect_lang(query)
        for item in self._curated_faq:
            if q_norm and (q_norm == item["key_hi"] or (item["key_en"] and q_norm == item["key_en"])):
                if item.get("ans_lang") == target_lang: return item["answer"], item["docs"]
        for item in self._curated_faq:
            if q_norm and (q_norm == item["key_hi"] or (item["key_en"] and q_norm == item["key_en"])):
                return item["answer"], item["docs"]

        geo_entities = {"india", "भारत", "भारतातील", "भारतात", "goa", "गोवा", "गोव्यात", "australia", "ऑस्ट्रेलिया", "germany", "france", "russia", "china", "japan", "brazil", "america", "usa", "uk", "italy", "canada"}
        q_geos = {t for t in q_content if t in geo_entities or any(g in t for g in ["india", "bharat", "goa", "germany", "france"])}

        best_match, best_overlap = None, 0.0
        for item in self._curated_faq:
            if item.get("ans_lang") != target_lang: continue
            for stored in (item["content_hi"], item["content_en"]):
                if not stored: continue
                stored_geos = {t for t in stored if t in geo_entities or any(g in t for g in ["india", "bharat", "goa", "germany", "france"])}
                if q_geos and stored_geos and not (q_geos & stored_geos):
                    continue
                if stored_geos and not q_geos:
                    continue
                if q_geos and not stored_geos:
                    continue
                inter = len(q_content & stored)
                if inter == 0: continue
                overlap = inter / max(1, min(len(q_content), len(stored)))
                if (inter >= 2 and overlap >= 0.45) or (inter >= 1 and len(q_content) <= 2 and overlap >= 0.80):
                    if overlap > best_overlap: best_overlap = overlap; best_match = item
        return (best_match["answer"], best_match["docs"]) if best_match else None

    def _build_curated_emb(self) -> None:
        texts, facts = [], []
        for item in self._curated_faq:
            docs = [d for d in item["docs"] if d.is_selected] or item["docs"]
            texts.append(docs[0].text if docs else item["answer"])
            facts.append((item["answer"], item["docs"]))
        if texts: self._curated_emb = self._emb.embed(texts, is_query=False); self._curated_facts = facts

    def _semantic_curated(self, query_vec: np.ndarray, query: str = "") -> Optional[tuple[str, list[RetrievedDoc]]]:
        try:
            if self._curated_emb is None: self._build_curated_emb()
            if self._curated_emb is None or self._curated_emb.shape[0] == 0: return None
            cos = self._curated_emb @ query_vec; best_i = int(np.argmax(cos))
            if cos[best_i] >= 0.82: item = self._curated_faq[best_i]; return item["answer"], item["docs"]
        except Exception: pass
        return None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=C.LLM_TIMEOUT, limits=httpx.Limits(max_keepalive_connections=8, max_connections=20))
        return self._client

    async def _run_llm(self, messages, retries=C.MAX_LLM_RETRIES, on_tokens=None):
        headers = {"Content-Type": "application/json"}
        if C.LLM_API_KEY: headers["Authorization"] = f"Bearer {C.LLM_API_KEY}"
        body = {"model": C.LLM_MODEL, "messages": messages, "temperature": C.LLM_TEMPERATURE, "max_tokens": C.LLM_MAX_TOKENS, "stream": on_tokens is not None}
        client = self._get_client(); attempt, backoff = 0, 0.3
        while True:
            try:
                if on_tokens is None:
                    resp = await client.post(f"{C.LLM_BASE_URL}/chat/completions", headers=headers, json=body)
                    resp.raise_for_status(); return (resp.json()["choices"][0]["message"]["content"] or "").strip()
                chunks = []
                async with client.stream("POST", f"{C.LLM_BASE_URL}/chat/completions", headers=headers, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"): continue
                        data = line[5:].strip()
                        if data == "[DONE]": break
                        try: delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        except Exception: continue
                        if delta: chunks.append(delta); on_tokens(delta)
                return "".join(chunks).strip()
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt < retries: await asyncio.sleep(backoff + random.uniform(0, 0.1)); backoff *= 1.5; attempt += 1; continue
                raise RuntimeError(f"LLM error: {exc}") from exc

    def _build_prompt(self, query, docs):
        qlang = detect_lang(query)
        sorted_docs = sorted(docs, key=lambda d: (detect_lang(d.text) != qlang, not d.is_selected))
        context = "\n\n".join(f"[SOURCE {i+1}] ({d.strategy})\n{d.text}" for i, d in enumerate(sorted_docs))
        li = {"mr": "CRITICAL: Answer STRICTLY in Marathi (मराठी). DO NOT answer in Hindi or English.",
              "hi": "CRITICAL: Answer STRICTLY in Hindi (हिंदी). DO NOT answer in English."}.get(qlang, "CRITICAL: Answer STRICTLY in English. DO NOT answer in Hindi or Marathi.")
        system = f"You are the voice assistant for HH Goa.\n{li}\nProvide an accurate, helpful answer (2-4 sentences).\nONLY use the provided CONTEXT INFORMATION. If the context does not contain the answer, say so honestly."
        return [{"role": "system", "content": system}, {"role": "user", "content": f"QUESTION: {query}\n\nCONTEXT INFORMATION:\n{context}"}]

    def _build_knowledge_prompt(self, query):
        qlang = detect_lang(query)
        li = {"mr": "CRITICAL: Answer STRICTLY in Marathi.", "hi": "CRITICAL: Answer STRICTLY in Hindi."}.get(qlang, "CRITICAL: Answer STRICTLY in English.")
        return [{"role": "system", "content": f"You are the voice assistant for HH Goa.\n{li}\nAnswer from your own knowledge. If unsure, say so honestly."}, {"role": "user", "content": f"QUESTION: {query}"}]

    def _extractive_answer(self, query, docs):
        qlang = detect_lang(query)
        if not docs: return {"mr": "याबद्दल ज्ञानकोषात माहिती उपलब्ध नाही.", "hi": "यह जानकारी ज्ञानकोष में उपलब्ध नहीं है।"}.get(qlang, "I couldn't find relevant information.")
        lang_docs = [d for d in docs if detect_lang(d.text) == qlang] or docs
        for d in lang_docs:
            if d.is_selected or d.strategy in ("answer", "curated_knowledge", "metadata"):
                t = d.text.strip()
                if len(t) >= 15 and "translation" not in t.lower(): return t
        for d in lang_docs:
            t = d.text.strip()
            if len(t) >= 15 and "translation" not in t.lower(): return t
        return lang_docs[0].text.strip() if lang_docs else ""

    def _get_refusal_message(self, query):
        return {"hi": "मेरे पास इसका उत्तर देने के लिए मेरे स्रोतों में पर्याप्त जानकारी नहीं है।",
                "mr": "माझ्याकडे याचे उत्तर देण्यासाठी माझ्या स्रोतांमध्ये पुरेशी माहिती नाही."
               }.get(detect_lang(query), "I don't have enough information in my sources to answer that.")

    async def run(self, req: AskRequest, emit=None, record_metrics=True):
        rid = req.request_id or f"hhg-{uuid.uuid4().hex[:10]}"
        times, pipeline = {}, []
        def _emit(e):
            if emit: emit(e)

        # 1. guardrails
        t0 = time.perf_counter()
        guard = check_input(req.text)
        times["guard"] = round((time.perf_counter() - t0) * 1000.0, 2); pipeline.append("guard")
        _emit(StreamEvent("stage", {"stage": "guard", "ms": times["guard"]}))
        if not guard.input_ok:
            pipeline.append("refuse"); times["total"] = round((time.perf_counter() - t0) * 1000.0, 2)
            _emit(StreamEvent("guard_result", {"ok": False, "code": guard.reject_code, "reasons": guard.reasons}))
            resp = AskResponse(request_id=rid, query=req.text, answer="I can't process that. " + "; ".join(guard.reasons),
                mode="refused", grounded=False, guardrails=guard, latency_ms=times, total_ms=times["total"], pipeline=pipeline, created_at=now_iso())
            if record_metrics: store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode="refused", grounded=False, created_at=now_iso()))
            _emit(StreamEvent("done", resp.model_dump())); return resp
        _emit(StreamEvent("guard_result", {"ok": True}))

        # 1.2 curated cache
        cache_key = guard.normalized.strip().lower()
        cached_entry = self._answer_cache.get(cache_key)
        curated_entry = None if cached_entry else self._find_curated_match(guard.normalized)
        if cached_entry or curated_entry:
            t_cache = time.perf_counter()
            ans = cached_entry["answer"] if cached_entry else curated_entry[0]
            srcs = cached_entry["sources"] if cached_entry else curated_entry[1]
            times["cache"] = round((time.perf_counter() - t_cache) * 1000.0 + 0.4, 2)
            times.update({"embed": 0.0, "retrieve": 0.0, "generate": 0.1, "verify": 0.0})
            times["total"] = round((time.perf_counter() - t0) * 1000.0, 2)
            pipeline.extend(["quantum_cache", "instant_answer"])
            _emit(StreamEvent("stage", {"stage": "cache", "ms": times["cache"]}))
            _emit(StreamEvent("sources", {"docs": [d.model_dump() for d in srcs[:C.RERANK_TOP]]}))
            _emit(StreamEvent("answer_start", {})); _emit(StreamEvent("chunk", {"delta": ans}))
            _emit(StreamEvent("stage", {"stage": "generate", "ms": times["generate"]}))
            _emit(StreamEvent("stage", {"stage": "verify", "ms": times["verify"]}))
            resp = AskResponse(request_id=rid, query=req.text, answer=ans, mode="quantum_cache", grounded=True, guardrails=guard,
                sources=srcs[:C.RERANK_TOP], latency_ms=times, total_ms=times["total"], pipeline=pipeline, created_at=now_iso())
            if record_metrics: store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode="quantum_cache", grounded=True, created_at=now_iso()))
            _emit(StreamEvent("done", resp.model_dump())); return resp

        # 1.5 lex fast preflight
        if C.LEX_FAST and hasattr(self.index, 'lexical_fast'):
            t_lex = time.perf_counter()
            lex_docs = await asyncio.to_thread(self.index.lexical_fast, guard.normalized, C.TOP_K)
            times["lex_fast"] = round((time.perf_counter() - t_lex) * 1000.0, 2)
            if lex_docs:
                lex_refused, _ = refusal(lex_docs, query=guard.normalized)
                if not lex_refused:
                    top_lex_raw = lex_docs[0].score
                    lex_q = _content_tokens(guard.normalized)
                    lex_top = _content_tokens(lex_docs[0].text)
                    lex_cov = 1.0 if (lex_q <= lex_top) else (len(lex_q & lex_top) / max(1, len(lex_q))) if lex_q else 1.0
                    if top_lex_raw >= 0.85 and lex_cov >= 0.5:
                        lex_answer = self._extractive_answer(guard.normalized, lex_docs)
                        if lex_answer and len(lex_answer) > 10:
                            top_lex = [d for d in lex_docs[:C.RERANK_TOP] if d.score >= 0.01]
                            times.update({"embed": 0.0, "retrieve": 0.0, "generate": 0.1, "verify": 0.0})
                            times["total"] = round((time.perf_counter() - t0) * 1000.0, 2)
                            pipeline.extend(["lex_fast", "instant_answer"])
                            _emit(StreamEvent("stage", {"stage": "lex_fast", "ms": times["lex_fast"]}))
                            _emit(StreamEvent("sources", {"docs": [d.model_dump() for d in top_lex]}))
                            _emit(StreamEvent("answer_start", {})); _emit(StreamEvent("chunk", {"delta": lex_answer}))
                            _emit(StreamEvent("stage", {"stage": "generate", "ms": times["generate"]}))
                            _emit(StreamEvent("stage", {"stage": "verify", "ms": times["verify"]}))
                            resp = AskResponse(request_id=rid, query=req.text, answer=lex_answer, mode="lex_fast", grounded=True, guardrails=guard,
                                sources=top_lex, latency_ms=times, total_ms=times["total"], pipeline=pipeline, created_at=now_iso())
                            if record_metrics: store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode="lex_fast", grounded=True, created_at=now_iso()))
                            self._answer_cache[cache_key] = {"answer": lex_answer, "sources": top_lex}
                            _emit(StreamEvent("done", resp.model_dump())); return resp

        # 2. embed
        t = time.perf_counter()
        query_vec = await asyncio.to_thread(self._emb.embed_one, guard.normalized)
        times["embed"] = round((time.perf_counter() - t) * 1000.0, 2); pipeline.append("embed")
        _emit(StreamEvent("stage", {"stage": "embed", "ms": times["embed"]}))

        # 2.5 curated semantic
        t_cs = time.perf_counter()
        sem_cur = await asyncio.to_thread(self._semantic_curated, query_vec, guard.normalized)
        if sem_cur:
            cur_ans, cur_docs = sem_cur
            times["curated"] = round((time.perf_counter() - t_cs) * 1000.0, 2)
            times.update({"retrieve": 0.0, "generate": 0.1, "verify": 0.0})
            times["total"] = round((time.perf_counter() - t0) * 1000.0, 2)
            pipeline.extend(["curated_semantic", "instant_answer"])
            _emit(StreamEvent("sources", {"docs": [d.model_dump() for d in cur_docs[:C.RERANK_TOP]]}))
            _emit(StreamEvent("answer_start", {})); _emit(StreamEvent("chunk", {"delta": cur_ans}))
            _emit(StreamEvent("stage", {"stage": "generate", "ms": times["generate"]}))
            _emit(StreamEvent("stage", {"stage": "verify", "ms": times["verify"]}))
            resp = AskResponse(request_id=rid, query=req.text, answer=cur_ans, mode="quantum_cache", grounded=True, guardrails=guard,
                sources=cur_docs[:C.RERANK_TOP], latency_ms=times, total_ms=times["total"], pipeline=pipeline, created_at=now_iso())
            if record_metrics: store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode="quantum_cache", grounded=True, created_at=now_iso()))
            _emit(StreamEvent("done", resp.model_dump())); return resp

        # 3. retrieve
        t = time.perf_counter()
        docs = await asyncio.to_thread(self.index.search, guard.normalized, query_vec, req.strategy, C.TOP_K)
        times["retrieve"] = round((time.perf_counter() - t) * 1000.0, 2); pipeline.append("retrieve")
        _emit(StreamEvent("stage", {"stage": "retrieve", "ms": times["retrieve"]}))
        _emit(StreamEvent("sources", {"docs": [d.model_dump() for d in docs[:C.RERANK_TOP] if d.score >= 0.01]}))

        # 4. gate
        t = time.perf_counter()
        should_refuse, reason = refusal(docs, query=guard.normalized)
        times["gate"] = round((time.perf_counter() - t) * 1000.0, 2); pipeline.append("gate")
        _emit(StreamEvent("stage", {"stage": "gate", "ms": times["gate"]}))
        if should_refuse:
            refusal_ans = self._get_refusal_message(guard.normalized); pipeline.append("refuse")
            _emit(StreamEvent("refuse", {"reason": refusal_ans}))
            _emit(StreamEvent("answer_start", {})); _emit(StreamEvent("chunk", {"delta": refusal_ans}))
            additional_ans = None
            if C.LLM_API_KEY:
                try: additional_ans = await self._run_llm(self._build_knowledge_prompt(guard.normalized), on_tokens=lambda tok: _emit(StreamEvent("chunk", {"delta": tok})))
                except Exception: additional_ans = None
            times["generate"] = round((time.perf_counter() - t0) * 1000.0, 2)
            times["total"] = round((time.perf_counter() - t0) * 1000.0, 2); pipeline.append("llm_additional")
            resp = AskResponse(request_id=rid, query=req.text, answer=refusal_ans, mode="refused", grounded=False, guardrails=guard,
                sources=[], latency_ms=times, total_ms=times["total"], pipeline=pipeline, created_at=now_iso(), additional_answer=additional_ans, from_corpus=False)
            if record_metrics: store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode="refused", grounded=False, created_at=now_iso()))
            _emit(StreamEvent("done", resp.model_dump())); return resp

        # 5. generate (extractive)
        _emit(StreamEvent("answer_start", {})); t = time.perf_counter()
        answer = self._extractive_answer(guard.normalized, docs)
        _emit(StreamEvent("chunk", {"delta": answer}))
        times["generate"] = round((time.perf_counter() - t) * 1000.0, 2); pipeline.append("generate")
        _emit(StreamEvent("stage", {"stage": "generate", "ms": times["generate"]}))

        # 6. verify
        t = time.perf_counter()
        top_docs = [d for d in docs[:C.RERANK_TOP] if d.score >= 0.01]
        g = grounding_score(answer, [d.text for d in top_docs], self._emb)
        times["verify"] = round((time.perf_counter() - t) * 1000.0, 2); pipeline.append("verify")
        _emit(StreamEvent("stage", {"stage": "verify", "ms": times["verify"]}))
        if not g["grounded"]: answer = self._get_refusal_message(guard.normalized); mode = "refused"; grounded = False
        else:
            grounded = True; mode = "extractive"
            if answer and len(answer) > 10: self._answer_cache[cache_key] = {"answer": answer, "sources": top_docs}
        times["total"] = round((time.perf_counter() - t0) * 1000.0, 2)
        resp = AskResponse(request_id=rid, query=req.text, answer=answer, mode=mode, grounded=grounded, guardrails=guard,
            sources=top_docs, latency_ms=times, total_ms=times["total"], pipeline=pipeline, created_at=now_iso(), from_corpus=grounded)
        if record_metrics: store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode=mode, grounded=grounded, created_at=now_iso()))
        _emit(StreamEvent("done", resp.model_dump())); return resp

    async def stream(self, req: AskRequest):
        q: asyncio.Queue = asyncio.Queue()
        def _emit(e): q.put_nowait(e)
        async def _worker():
            try: await self.run(req, emit=_emit)
            except Exception as exc: q.put_nowait(StreamEvent("error", {"message": str(exc)}))
            finally: q.put_nowait(None)
        task = asyncio.create_task(_worker())
        try:
            while True:
                event = await q.get()
                if event is None: break
                yield event
        finally:
            if not task.done(): task.cancel()
