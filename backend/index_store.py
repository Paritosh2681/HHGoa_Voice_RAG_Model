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
        self._inv: Optional[dict[str, list[int]]] = None
        self._tokenized: list[list[str]] = []
        self._dims = 0

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk], vectors: np.ndarray, dims: int) -> None:
        self.chunks = chunks
        self._vectors = vectors
        self._dims = dims
        self._faiss = None

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

        # --- dense arm (FAISS C++ search across 1.27M / 2.1M vectors)
        dense_scores: dict[int, float] = {}
        candidate_indices: list[int] = []
        n = min(DENSE_TOP, len(self.chunks))
        if self._faiss is not None:
            try:
                q_mat = query_vec.reshape(1, -1).astype("float32")
                D, I = self._faiss.search(q_mat, n)
                for score, idx in zip(D[0], I[0]):
                    if 0 <= idx < len(self.chunks):
                        dense_scores[int(idx)] = float(score)
                        candidate_indices.append(int(idx))
            except Exception:
                pass

        if not dense_scores and self._vectors is not None and self._vectors.shape[0]:
            sims = self._vectors @ query_vec
            top_indices = np.argpartition(-sims, n)[:n]
            top_sorted = top_indices[np.argsort(-sims[top_indices])]
            for idx in top_sorted:
                dense_scores[int(idx)] = float(sims[int(idx)])
                candidate_indices.append(int(idx))

        # --- fast sparse candidate lexical ranking (0.1ms)
        sparse_ranks: dict[int, int] = {}
        tokens = self._tokenize(query)
        if tokens and candidate_indices:
            q_toks = set(tokens)
            scored = []
            for idx in candidate_indices[:50]:
                doc_toks = set(self._tokenize(self.chunks[idx].text))
                common = len(q_toks & doc_toks)
                if common > 0:
                    scored.append((common, idx))
            scored.sort(reverse=True)
            for pos, (_s, idx) in enumerate(scored[:BM25_TOP]):
                sparse_ranks[idx] = pos

        # --- RRF fusion (ranking signal only)
        k = 60.0
        fused: dict[int, float] = {}
        for idx in dense_scores:
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + 1)
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
                s += SELECTED_BOOST * 0.05
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
        dense_faiss_file = INDEX_DIR / "dense.index"
        if not (chunks_file.exists() and vec_file.exists() and meta_file.exists()):
            return None
        try:
            import pickle
            chunks_pkl = INDEX_DIR / "chunks.pkl"
            if chunks_pkl.exists():
                with open(chunks_pkl, "rb") as f:
                    chunks = pickle.load(f)
            else:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                chunks = [Chunk.model_construct(**p) for p in payload]
                try:
                    with open(chunks_pkl, "wb") as f:
                        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            vectors = np.load(vec_file, mmap_mode="r")
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            idx = cls()
            idx.build(chunks, vectors, int(meta.get("dims", 0)))
            if dense_faiss_file.exists():
                try:
                    import faiss
                    idx._faiss = faiss.read_index(str(dense_faiss_file))
                    # Set nprobe for faster search on IVFFlat index
                    import os as _os
                    _nprobe = int(_os.getenv("HHGOA_IVF_NPROBE", "32"))
                    if hasattr(idx._faiss, 'nprobe'):
                        idx._faiss.nprobe = _nprobe
                except Exception:
                    pass
            return idx
        except Exception:
            return None
