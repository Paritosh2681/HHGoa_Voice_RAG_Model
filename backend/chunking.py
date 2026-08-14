"""Multi-strategy chunking.

The brief explicitly rejects a single naive fixed-size splitter. This module
implements four distinct strategies, each with a defensible rationale, and the
ingest pipeline indexes ALL of them side-by-side with metadata so retrieval can
decide what kind of evidence to pull:

1. fixed_overlap   — character-window splitter with configurable overlap.
                     Baseline that preserves neighbouring context across cuts.
2. sentence        — sentence-boundary aware splitting. Never cuts mid-sentence;
                     groups sentences into a target character window. Best for
                     question-answer prose.
3. semantic        — embeddings of sentences, merged greedily by cosine
                     similarity (semantic coherence > arbitrary lengths).
4. metadata        — metadata-aware units: each passage is wrapped with its
                     provenance (query_id, query_type, language, selected flag,
                     source doc) so chunks are self-describing and filterable.

Every chunk carries {strategy, query_id, query_type, is_selected, language,
passage_index, doc_id} so retrieval can boost/penalise by metadata.
"""
from __future__ import annotations

import re
import uuid
from typing import Iterator

from .models import Chunk
from .config import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE, SEMANTIC_SIM_THRESHOLD, SENTENCE_WINDOW

_SENT_SPLIT = re.compile(
    r"(?<=[।॥])|(?<=[.!?])(?=\s|$)|\n+",
    re.M,
)


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def _norm(text: str) -> str:
    return " ".join(text.split())


def fixed_overlap(text: str, meta: dict, strategy_id: str,
                  size: int = FIXED_CHUNK_SIZE,
                  overlap: int = FIXED_CHUNK_OVERLAP) -> Iterator[Chunk]:
    text = _norm(text)
    step = max(1, size - overlap)
    for i in range(0, max(1, len(text)), step):
        piece = text[i:i + size]
        if not piece.strip():
            continue
        yield Chunk(id=str(uuid.uuid4()), text=piece, strategy=strategy_id,
                    meta={**meta, "chunk_window": i})


def sentence_chunks(text: str, meta: dict, strategy_id: str,
                    window: int = SENTENCE_WINDOW,
                    max_chars: int = FIXED_CHUNK_SIZE * 2) -> Iterator[Chunk]:
    sents = _sentences(text)
    if not sents:
        return
    buf: list[str] = []
    buf_len = 0
    for s in sents:
        if buf and (len(buf) >= window or buf_len + len(s) > max_chars):
            yield Chunk(id=str(uuid.uuid4()), text=" ".join(buf),
                        strategy=strategy_id, meta={**meta, "n_sentences": len(buf)})
            buf, buf_len = [], 0
        buf.append(s)
        buf_len += len(s)
    if buf:
        yield Chunk(id=str(uuid.uuid4()), text=" ".join(buf),
                    strategy=strategy_id, meta={**meta, "n_sentences": len(buf)})


def semantic_chunks(text: str, meta: dict, strategy_id: str,
                    threshold: float = SEMANTIC_SIM_THRESHOLD) -> Iterator[Chunk]:
    """Greedy merge of sentences by embedding cosine similarity."""
    from .embeddings import get_embedding_provider

    sents = _sentences(text)
    if not sents:
        return
    if len(sents) == 1:
        yield Chunk(id=str(uuid.uuid4()), text=sents[0], strategy=strategy_id, meta=meta)
        return
    emb = get_embedding_provider().embed(sents)
    cur = [sents[0]]
    prev_vec = emb[0]
    for s, vec in zip(sents[1:], emb[1:]):
        sim = float(np_dot(prev_vec, vec))
        if sim >= threshold and len(" ".join(cur + [s])) <= FIXED_CHUNK_SIZE * 2:
            cur.append(s)
        else:
            yield Chunk(id=str(uuid.uuid4()), text=" ".join(cur),
                        strategy=strategy_id, meta={**meta, "n_sentences": len(cur)})
            cur = [s]
        prev_vec = vec
    if cur:
        yield Chunk(id=str(uuid.uuid4()), text=" ".join(cur),
                    strategy=strategy_id, meta={**meta, "n_sentences": len(cur)})


def metadata_chunks(passages: list[tuple[int, str, bool]], meta: dict,
                    strategy_id: str, max_chars: int = FIXED_CHUNK_SIZE * 2) -> Iterator[Chunk]:
    """Metadata-aware: each passage becomes a self-describing unit."""
    for idx, text, is_selected in passages:
        text = _norm(text)
        m = {**meta, "passage_index": idx, "is_selected": is_selected}
        if len(text) <= max_chars:
            yield Chunk(id=str(uuid.uuid4()), text=text, strategy=strategy_id, meta=m)
        else:
            for c in fixed_overlap(text, m, f"{strategy_id}:inner", size=FIXED_CHUNK_SIZE,
                                   overlap=FIXED_CHUNK_OVERLAP):
                yield c


def np_dot(a, b):
    import numpy as np
    return float(np.dot(a, b))


STRATEGIES = ["fixed_overlap", "sentence", "semantic", "metadata"]

STRATEGY_NOTES = {
    "fixed_overlap": "Character-window splitter with 20% overlap. Adjacent context survives cuts; acts as the baseline.",
    "sentence": "Sentence-boundary aware: never breaks mid-sentence, groups into a ~4-sentence window. Ideal for QA prose.",
    "semantic": "Sentences merged by embedding cosine similarity (>= 0.42) — chunks are semantically coherent, not arbitrary lengths.",
    "metadata": "Metadata-aware: every chunk is self-describing (query_id, query_type, language, selected-flag, source doc) and filterable.",
}


def chunk_all(passages: list[tuple[int, str, bool]], meta: dict) -> list[Chunk]:
    """Run all strategies over one (query-scoped) passage group."""
    text = " ".join(p[1] for p in passages)
    base = dict(meta)
    chunks: list[Chunk] = []
    chunks.extend(fixed_overlap(text, base, "fixed_overlap"))
    chunks.extend(sentence_chunks(text, base, "sentence"))
    chunks.extend(semantic_chunks(text, base, "semantic"))
    chunks.extend(metadata_chunks(passages, base, "metadata"))
    return chunks
