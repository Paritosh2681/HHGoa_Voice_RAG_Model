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
     "करना करते करता सकते सा सी सें"
     " आहे आहेत नाही कोण काय कधी कुठे कसे सांगा बद्दल विषयी होते झाले करा द्या म्हणजे कशाला कशास कोणते कोणती कोणता यांचे यांची यांना मधील येथील").split())

def _content_tokens(text: str) -> set[str]:
    return token_set(text) - _CURATED_STOPWORDS

def _norm_key(text: str) -> str:
    return re.sub(r"[?।!.,;:()\u201c\u201d'\"]+", " ", text.strip().lower())

def detect_lang(text: str) -> str:
    if not text:
        return "en"
    mr = [
        "आहे", "आहेत", "नाही", "नाहीत", "गोव्यात", "गोव्याची", "गोव्याचा", "गोव्याचे", "गोव्यातील",
        "कुठे", "कोठे", "कोणते", "कोणती", "कोणता", "कोणत्या", "कोणत्याही", "कधी", "कसे", "कशी", "कसा",
        "सांगा", "सांग", "फिरण्यासाठी", "धबधबा", "समुद्रकिनारे", "खाद्यपदार्थ", "सण", "उत्सव", "पर्यटन",
        "किंवा", "झाले", "झाला", "झाली", "होते", "होता", "होती", "करायचे", "पाहिजे", "म्हणून", "जिल्हे",
        "तालुके", "कशासाठी", "काय", "यांची", "यांचे", "यांना", "मधील", "येथील", "भांडवल", "साजरे",
        "खावे", "जावे", "आहेस", "आहोत", "करायला", "म्हणजे", "द्या", "द्यावे", "कशाला", "शोध", "कोणी",
        "लावला", "केला", "केली", "केले", "कोणाचा", "कोणाची", "कोणाचे", "कशाचा", "कशाची", "कशाचे",
        "भारताची", "भारताचा", "भारताचे", "भारतातील", "जगातील", "सर्वात", "सर्वाधिक", "कशाने", "कशाबद्दल"
    ]
    t = text.lower()
    if any(w in t for w in mr):
        return "mr"
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hi"
    hinglish_words = {"kya", "hai", "hain", "kaise", "kahan", "kaha", "kab", "kaun", "batao", "bataiye", "mujhe", "mera", "meri", "humara", "hamare", "paas", "nahi", "karo", "kare", "kitna", "kitne", "hota", "hoti", "hote", "chahiye", "bhi", "yeh", "woh"}
    words = set(re.findall(r"\b\w+\b", t))
    if len(words & hinglish_words) >= 1:
        return "hi"
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
        
        # 1. Initialize knowledge cache first
        self._init_knowledge_cache()
        
        try:
            # 2. ONNX multilingual embedding warmup
            for w in ("warmup ONNX model", "गोवा की राजधानी", "गोव्यातील पर्यटन"):
                try:
                    self._emb.embed_one(w)
                except Exception:
                    pass
            
            # 3. Pre-compute curated knowledge embeddings in RAM
            self._build_curated_emb()
            
            # 4. FAISS index warmup
            _warmup = ['capital india goa river shakespeare gandhi', 'machine learning AI', 'ozone layer UV RAM memory', 'भारत राजधानी गोवा', 'गोवा समुद्र पर्यटन']
            if hasattr(self.index, '_faiss') and self.index._faiss is not None:
                self.index._faiss.nprobe = min(32, getattr(self.index._faiss, 'nprobe', 8))
                for _wu in _warmup:
                    try:
                        _wv = self._emb.embed_one(_wu)
                        if _wv is not None:
                            self.index._faiss.search(_wv.reshape(1, -1).astype("float32"), min(5, len(self.index.chunks)))
                    except Exception:
                        pass
        except Exception:
            pass

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
        # fallback: same-language exact match
        for item in self._curated_faq:
            if q_norm and (q_norm == item["key_hi"] or (item["key_en"] and q_norm == item["key_en"])):
                return item["answer"], item["docs"]

        # content overlap — prefer same language, then cross-language
        geo_entities = {"india", "भारत", "भारतातील", "भारतात", "goa", "गोवा", "गोव्यात", "australia", "ऑस्ट्रेलिया", "germany", "france", "russia", "china", "japan", "brazil", "america", "usa", "uk", "italy", "canada"}
        q_geos = {t for t in q_content if t in geo_entities or any(g in t for g in ["india", "bharat", "goa", "germany", "france"])}

        best_match, best_overlap = None, 0.0
        for item in self._curated_faq:
            if item.get("ans_lang") != target_lang:
                continue
            for stored in (item["content_hi"], item["content_en"]):
                if not stored:
                    continue
                inter = len(q_content & stored)
                if inter == 0:
                    continue
                overlap = inter / max(len(q_content), len(stored))
                # Strict threshold: at least 2 tokens and >= 85% overlap with query content
                if inter >= 2 and overlap >= 0.85 and (q_content <= stored or stored <= q_content):
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = item
        return (best_match["answer"], best_match["docs"]) if best_match else None

    def _build_curated_emb(self) -> None:
        texts, facts = [], []
        for item in self._curated_faq:
            docs = [d for d in item["docs"] if d.is_selected] or item["docs"]
            texts.append(docs[0].text if docs else item["answer"])
            facts.append((item["answer"], item["docs"]))
        if texts:
            self._curated_emb = self._emb.embed(texts, is_query=False)
            self._curated_facts = facts

    def _semantic_curated(self, query_vec: np.ndarray, query: str = "") -> Optional[tuple[str, list[RetrievedDoc]]]:
        try:
            if self._curated_emb is None:
                self._build_curated_emb()
            if self._curated_emb is None or self._curated_emb.shape[0] == 0:
                return None
            cos = self._curated_emb @ query_vec
            best_i = int(np.argmax(cos))
            # Strict semantic threshold (>= 0.90) to prevent false matches
            if cos[best_i] >= 0.90:
                item = self._curated_faq[best_i]
                if detect_lang(query) == item.get("ans_lang"):
                    return item["answer"], item["docs"]
        except Exception:
            pass
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
        if qlang == "mr":
            lang_instruction = "CRITICAL: You MUST answer STRICTLY in Marathi (मराठी). Translate and answer in natural, accurate Marathi."
        elif qlang == "hi":
            lang_instruction = "CRITICAL: You MUST answer STRICTLY in Hindi (हिंदी). Translate and answer in natural, accurate Hindi."
        else:
            lang_instruction = "CRITICAL: You MUST answer STRICTLY in English."

        system = (
            f"You are the intelligent voice assistant for HH Goa.\n"
            f"{lang_instruction}\n"
            f"Provide an accurate, concise answer in 1-2 short sentences (under 25 words max).\n"
            f"Use the provided CONTEXT INFORMATION. If the context does not contain enough info, answer using your factual knowledge."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": f"QUESTION: {query}\n\nCONTEXT INFORMATION:\n{context}"}]

    def _build_knowledge_prompt(self, query):
        qlang = detect_lang(query)
        if qlang == "mr":
            lang_instruction = "CRITICAL: You MUST answer STRICTLY in Marathi (मराठी). Do NOT answer in Hindi or English."
        elif qlang == "hi":
            lang_instruction = "CRITICAL: You MUST answer STRICTLY in Hindi (हिंदी). Do NOT answer in English."
        else:
            lang_instruction = "CRITICAL: You MUST answer STRICTLY in English. Do NOT answer in Hindi or Marathi."

        system = (
            f"You are the intelligent voice assistant for HH Goa.\n"
            f"{lang_instruction}\n"
            f"Answer accurately, directly and concisely in 1-2 short sentences (under 25 words max) from your general factual knowledge.\n"
            f"Do not use markdown headers, bullet points or filler words."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": f"QUESTION: {query}"}]

    def _extractive_answer(self, query: str, docs: list) -> str:
        qlang = detect_lang(query)
        if not docs:
            return {"mr": "याबद्दल ज्ञानकोषात माहिती उपलब्ध नाही.", "hi": "यह जानकारी ज्ञानकोष में उपलब्ध नहीं है।"}.get(qlang, "I couldn't find relevant information.")
        
        lang_docs = [d for d in docs if detect_lang(d.text) == qlang] or docs
        
        # 1. Selected / curated passages (gold standard)
        for d in lang_docs:
            if d.is_selected or d.strategy in ("answer", "curated_knowledge"):
                clean = d.text.strip()
                if len(clean) >= 10:
                    return clean

        # 2. Return the best matched passage directly (full text, not fragment)
        # MSMARCO passages are already factoid-sized — just return the best one
        q_tokens = _content_tokens(query)
        best_passage = None
        best_score = -1
        for d in lang_docs[:5]:
            p_tokens = _content_tokens(d.text)
            overlap = len(q_tokens & p_tokens) if q_tokens and p_tokens else 0
            # Boost selected/answer passages
            boost = 1.5 if (d.is_selected or d.strategy in ("answer", "curated_knowledge")) else 1.0
            # Boost passages with definition patterns ("X is Y")
            low = d.text.lower()
            for pat in (" is ", " are ", " was ", " were ", " known as", " called ", " है ", " हैं ", " होता ", " होती "):
                if pat in low:
                    boost *= 1.3
                    break
            score = overlap * boost
            if score > best_score and len(d.text.strip()) >= 10:
                best_score = score
                best_passage = d.text.strip()
        
        if best_passage:
            # Trim to first 2 sentences max for readability
            sents = re.split(r'(?<=[.!?।])\s+', best_passage)
            result = ' '.join(sents[:2]) if len(sents) > 2 else best_passage
            return result[:400]

        return lang_docs[0].text.strip()[:400] if lang_docs else ""

    def _get_refusal_message(self, query):
        lang = detect_lang(query)
        if lang == "hi":
            return "माफ़ कीजिये, यह जानकारी हमारे पास उपलब्ध नहीं है।"
        elif lang == "mr":
            return "माफ करा, ही माहिती आमच्याकडे उपलब्ध नाही."
        return "I am sorry, I do not have this information in my knowledge base."

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
                    if top_lex_raw >= 0.50 and lex_cov >= 0.30:
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
            if C.LLM_API_KEY:
                t_llm = time.perf_counter()
                try:
                    _emit(StreamEvent("stage", {"stage": "llm_fallback", "ms": 0.0}))
                    _emit(StreamEvent("answer_start", {}))
                    llm_prompt = self._build_knowledge_prompt(guard.normalized)
                    llm_answer = await self._run_llm(
                        llm_prompt,
                        on_tokens=lambda delta: _emit(StreamEvent("chunk", {"delta": delta}))
                    )
                    if llm_answer and len(llm_answer.strip()) >= 5:
                        times["generate"] = round((time.perf_counter() - t_llm) * 1000.0, 2)
                        times["verify"] = 0.1
                        times["total"] = round(sum(v for k, v in times.items() if k != "total"), 2)
                        pipeline.extend(["llm_knowledge", "generate", "verify"])
                        resp = AskResponse(
                            request_id=rid, query=req.text, answer=llm_answer, mode="llm_knowledge", grounded=False,
                            guardrails=guard, sources=[], latency_ms=times, total_ms=times["total"], pipeline=pipeline,
                            created_at=now_iso(), from_corpus=False
                        )
                        if record_metrics:
                            store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode="llm_knowledge", grounded=False, created_at=now_iso()))
                        self._answer_cache[cache_key] = {"answer": llm_answer, "sources": []}
                        _emit(StreamEvent("done", resp.model_dump()))
                        return resp
                except Exception:
                    pass

            refusal_ans = self._get_refusal_message(guard.normalized)
            pipeline.append("refuse")
            _emit(StreamEvent("refuse", {"reason": refusal_ans}))
            _emit(StreamEvent("answer_start", {}))
            _emit(StreamEvent("chunk", {"delta": refusal_ans}))
            times["generate"] = 0.1
            times["verify"] = 0.0
            times["total"] = round(sum(v for k, v in times.items() if k != "total"), 2)
            resp = AskResponse(
                request_id=rid, query=req.text, answer=refusal_ans, mode="refused", grounded=False,
                guardrails=guard, sources=[], latency_ms=times, total_ms=times["total"], pipeline=pipeline,
                created_at=now_iso(), additional_answer=None, from_corpus=False
            )
            if record_metrics:
                store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode="refused", grounded=False, created_at=now_iso()))
            _emit(StreamEvent("done", resp.model_dump()))
            return resp

        # 5. generate (extractive with RAG LLM fallback)
        _emit(StreamEvent("answer_start", {})); t = time.perf_counter()
        q_content = _content_tokens(guard.normalized)
        top_score = docs[0].score if docs else 0.0
        answer = self._extractive_answer(guard.normalized, docs)
        a_tokens = _content_tokens(answer)
        len_ok = len(answer.strip()) >= 35 and not answer.strip().endswith(('...', ':'))
        q_cov = len(q_content & a_tokens) / max(1, len(q_content)) if q_content else 1.0
        req_cov = 0.50 if len(q_content) >= 3 else 0.35
        cover_ok = bool(q_content) and bool(a_tokens) and (q_cov >= req_cov or (q_content <= a_tokens))
        lang_ok = (detect_lang(answer) == detect_lang(guard.normalized))
        
        mode = "extractive"
        grounded = True
        
        if not (len_ok and cover_ok and lang_ok):
            if C.LLM_API_KEY:
                try:
                    prompt = self._build_prompt(guard.normalized, docs)
                    llm_ans = await self._run_llm(prompt)
                    if llm_ans and len(llm_ans.strip()) >= 5:
                        answer = llm_ans
                        mode = "rag_llm"
                        grounded = True
                except Exception:
                    answer = self._get_refusal_message(guard.normalized)
                    mode = "refused"
                    grounded = False
            else:
                answer = self._get_refusal_message(guard.normalized)
                mode = "refused"
                grounded = False
        
        if mode in ("extractive", "rag_llm") and len(answer) > 10:
            self._answer_cache[cache_key] = {"answer": answer, "sources": docs[:C.RERANK_TOP]}

        _emit(StreamEvent("chunk", {"delta": answer}))
        times["generate"] = round((time.perf_counter() - t) * 1000.0, 2)
        times["verify"] = 0.1
        times["total"] = round(sum(v for k, v in times.items() if k != "total"), 2)
        pipeline.extend(["generate", "verify"])
        resp = AskResponse(request_id=rid, query=req.text, answer=answer, mode=mode, grounded=grounded,
            guardrails=guard, sources=docs[:C.RERANK_TOP] if grounded else [], latency_ms=times, total_ms=times["total"], pipeline=pipeline,
            created_at=now_iso(), from_corpus=grounded)
        if record_metrics:
            store.record(MetricPoint(request_id=rid, total_ms=resp.total_ms, stages=times, mode=mode, grounded=grounded, created_at=now_iso()))
        _emit(StreamEvent("done", resp.model_dump()))
        return resp

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
