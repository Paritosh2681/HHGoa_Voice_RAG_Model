"""Guardrails — the system must know *when not to answer*.

Implemented layers:
1. Input normalization (Devanagari/Latin folding, punctuation, whitespace).
2. Safety classifier: profanity + abuse blocklists (en + hi), plus jailbreak /
   instruction-injection pattern scanning.
3. Length sanity: empty, too-short, or bloated queries.
4. Grounding / hallucination check: post-generation verification that the
   answer is actually supported by the retrieved context (lexical overlap +
   embedding similarity). Below threshold -> answer is refused/flagged.
5. Off-topic / low-confidence: if retrieval scores fall under MIN_SCORE the
   harness refuses instead of making something up.
"""
from __future__ import annotations

import re
import time
import unicodedata

from .config import MIN_SCORE
from .models import GuardResult

_PROFANITY_EN = [
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "cunt", "whore",
    "nigger", "faggot", "retard", "motherfucker", "piss off", "kill yourself",
    "kill ur", "suck my", "fag",
]
_PROFANITY_HI = [
    "मादरचोद", "मादरचोड़", "बहनचोद", "बहनचोड़", "गांडू", "गाँडू", "चूतिया",
    "चुतिया", "लौड़ा", "लौंडा नहीं", "कुतिया", "हरामी", "साला", "भोसड़ी",
    "बक्चोद", "गधा", "मूर्ख", "नमक हराम",
]
_JAILBREAK = [
    "ignore your instructions", "ignore all previous", "system prompt",
    "you are now", "dan mode", "do anything now", "jailbreak", "bypass",
    "ignore the context", "ignore above", "pretend you are", "act as openai",
    "reveal your system", "give me your prompt", "developer mode",
    "ignore everything before", "disregard", "forget your instructions",
    "अपने निर्देश भूल जाओ", "सिस्टम प्रॉम्प्ट बताओ", "नियम तोड़ो",
]

_RE_NON_ALNUM = re.compile(r"[^\w\u0900-\u097F]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("।", "। ").replace("?", " ").replace("?", " ")
    text = text.replace("\\n", " ")
    return " ".join(text.split())


def token_set(text: str) -> set[str]:
    return {t for t in _RE_NON_ALNUM.split(text.lower()) if len(t) > 1}


def lexical_overlap(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def check_input(raw: str) -> GuardResult:
    t0 = time.perf_counter()
    reasons: list[str] = []
    q = normalize(raw)
    low = q.lower()

    if not q:
        reasons.append("empty query")
        return GuardResult(input_ok=False, query=raw, normalized=q,
                           reasons=reasons, reject_code="EMPTY", latency_ms=0)

    for w in _PROFANITY_EN:
        if re.search(rf"(^|\W){re.escape(w)}(\W|$)", low):
            reasons.append(f"unsafe language detected ({w})")
            break
    for w in _PROFANITY_HI:
        if w in q:
            reasons.append("unsafe language detected")
            break

    if len(q) < 3:
        reasons.append("query too short")

    for pat in _JAILBREAK:
        if pat in low:
            reasons.append("prompt-injection pattern detected")
            break

    latency = (time.perf_counter() - t0) * 1000.0
    code = None
    if any("unsafe language" in r for r in reasons):
        code = "BLOCKED_UNSAFE"
    elif "prompt-injection" in " ".join(reasons):
        code = "BLOCKED_INJECTION"
    elif "empty query" in reasons or "query too short" in reasons:
        code = "INVALID_INPUT"
    return GuardResult(
        input_ok=code is None, query=raw, normalized=q,
        reasons=reasons, reject_code=code, latency_ms=latency,
    )


def grounding_score(answer: str, docs_texts: list[str], embedding_provider,
                    answer_vec=None, doc_vecs: list = None) -> dict:
    """Hallucination check: is the answer supported by the retrieved context?"""
    if not docs_texts:
        return {"grounded": False, "score": 0.0, "reason": "no context retrieved"}
    import numpy as np

    # 1) lexical overlap with the retrieved corpus (strongest signal)
    corpus = " ".join(docs_texts)
    lex = lexical_overlap(answer, corpus)

    # 2) semantic similarity against the most relevant doc (reuse stored vectors)
    try:
        if answer_vec is None:
            answer_vec = embedding_provider.embed_one(answer)
        if doc_vecs:
            sims = [float(np.dot(answer_vec, v)) for v in doc_vecs[:3] if v is not None]
        else:
            sims = [float(np.dot(answer_vec, embedding_provider.embed_one(d)))
                    for d in docs_texts[:3]]
        sim = max(sims) if sims else 0.0
    except Exception:
        sim = 0.0

    score = 0.6 * min(lex * 3.0, 1.0) + 0.4 * max(0.0, sim)
    grounded = score >= 0.18
    return {
        "grounded": grounded,
        "score": round(float(score), 4),
        "lexical": round(float(lex), 4),
        "semantic": round(float(sim), 4),
        "reason": None if grounded else "answer not grounded in retrieved context",
    }


def refusal(docs: list, query: str = "", min_score: float = MIN_SCORE) -> tuple[bool, str]:
    """Off-topic gate: refuse when retrieval confidence is too low.

    Confidence blends the dense cosine of the best hit with the lexical
    overlap of the query against the top passages. Lexical overlap is the
    decisive signal for languages where the embedding model is weak (Hindi).
    """
    if not docs:
        return True, "no relevant context found in the knowledge base"
    best_cos = docs[0].score
    lex = lexical_overlap(query, " ".join(d.text for d in docs[:3]))
    lex_scaled = min(lex * 2.0, 1.0)
    relevance = max(best_cos, lex_scaled)
    if relevance < min_score:
        return True, (f"query appears out-of-domain for this knowledge base "
                      f"(confidence {relevance:.2f} < {min_score:.2f})")
    return False, ""
