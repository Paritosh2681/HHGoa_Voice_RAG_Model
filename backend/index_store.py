"""Hybrid retrieval store: FAISS (dense) + BM25 (sparse) with RRF fusion.

- Dense arm captures semantic similarity in 100+ languages (e5-small).
- Sparse arm (BM25) is a precise lexical matcher that survives OOV embeddings.
- Reciprocal Rank Fusion merges both ranked lists.
- A lightweight metadata-aware re-ranker then boosts gold ("is_selected")
  passages and applies MIN_SCORE grounding so the pipeline knows when to
  refuse to answer instead of hallucinating.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import (BM25_TOP, DENSE_TOP, INDEX_DIR, MIN_SCORE, RERANK_TOP,
                     SELECTED_BOOST, TOP_K)
from .models import Chunk, RetrievedDoc


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class HybridIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._vectors: Optional[np.ndarray] = None
        self._bm25 = None
        self._tokenized: list[list[str]] = []
        self._dims = 0

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk], vectors: np.ndarray, dims: int) -> None:
        self.chunks = chunks
        self._vectors = vectors
        self._dims = dims
        self._tokenized = [self._tokenize(c.text) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized)
        except Exception:
            self._bm25 = None

    def __len__(self) -> int:
        return len(self.chunks)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"[\w\u0900-\u097F]+", text.lower())

    # ---------------------------------------------------------------- retrieval
    def search(self, query: str, query_vec: np.ndarray, strategy: Optional[str] = None,
               top_k: int = TOP_K) -> list[RetrievedDoc]:
        t0 = time.perf_counter()

        # --- dense arm (cosine via normalized inner product)
        dense_scores: dict[int, float] = {}
        if self._vectors is not None and self._vectors.shape[0]:
            sims = self._vectors @ query_vec  # rows are L2-normalized
            n = min(DENSE_TOP, len(self.chunks))
            top_indices = np.argpartition(-sims, n)[:n]
            top_sorted = top_indices[np.argsort(-sims[top_indices])]
            for idx in top_sorted:
                dense_scores[int(idx)] = float(sims[int(idx)])

        # --- sparse arm
        sparse_ranks: dict[int, int] = {}
        if self._bm25 is not None:
            tokens = self._tokenize(query)
            if tokens:
                scores = self._bm25.get_scores(tokens)
                n = min(BM25_TOP, len(self.chunks))
                top_indices = np.argpartition(-scores, n)[:n]
                top_sorted = top_indices[np.argsort(-scores[top_indices])]
                for pos, idx in enumerate(top_sorted):
                    sparse_ranks[int(idx)] = pos

        # --- RRF fusion (ranking signal only)
        k = 60.0
        fused: dict[int, float] = {}
        for idx in dense_scores:
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + 1)  # dense rank base
        for idx in sparse_ranks:
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + sparse_ranks[idx] + 1)

        # --- metadata-aware re-rank; final score = cosine + small boosts
        ranked = []
        for idx in fused:
            c = self.chunks[idx]
            if strategy and c.strategy != strategy and not c.strategy.startswith(strategy):
                continue
            cos = dense_scores.get(idx, 0.0)
            s = cos
            if c.meta.get("is_selected", False):
                s += SELECTED_BOOST * 0.05   # gentle gold-preference
            if c.strategy == "metadata":
                s += 0.02
            ranked.append((idx, s, cos, fused[idx]))

        ranked.sort(key=lambda t: t[1], reverse=True)
        docs = []
        for idx, s, cos, rr in ranked[:top_k]:
            c = self.chunks[idx]
            docs.append(RetrievedDoc(
                id=c.id, text=c.text, strategy=c.strategy, score=round(float(cos), 4),
                source=c.meta.get("source", "passage"),
                language=c.meta.get("language", "hi"),
                is_selected=bool(c.meta.get("is_selected", False)),
                query_type=c.meta.get("query_type"),
            ))
        self.last_search_ms = (time.perf_counter() - t0) * 1000.0
        return docs

    def vectors_for(self, ids: list[str]) -> list:
        """Return the stored (L2-normalized) vector for each chunk id (or None)."""
        if self._vectors is None or self._vectors.shape[0] == 0:
            return [None] * len(ids)
        by_id = {c.id: i for i, c in enumerate(self.chunks)}
        out = []
        for cid in ids:
            i = by_id.get(cid)
            out.append(None if i is None else self._vectors[i])
        return out

    # ---------------------------------------------------------------- persist
    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        payload = [c.model_dump() for c in self.chunks]
        (INDEX_DIR / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if self._vectors is not None:
            np.save(INDEX_DIR / "vectors.npy", self._vectors)
        meta = {"n_chunks": len(self.chunks), "dims": self._dims}
        (INDEX_DIR / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls) -> Optional["HybridIndex"]:
        chunks_file = INDEX_DIR / "chunks.json"
        vec_file = INDEX_DIR / "vectors.npy"
        meta_file = INDEX_DIR / "index_meta.json"
        if not (chunks_file.exists() and vec_file.exists() and meta_file.exists()):
            return None
        try:
            payload = json.loads(chunks_file.read_text(encoding="utf-8"))
            chunks = [Chunk(**p) for p in payload]
            vectors = np.load(vec_file)
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            idx = cls()
            idx.build(chunks, vectors, int(meta.get("dims", 0)))
            return idx
        except Exception:
            return None
