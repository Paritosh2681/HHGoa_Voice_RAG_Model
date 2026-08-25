"""Embedding provider — local ONNX via fastembed.

Why local embeddings? The latency budget is 200ms for the whole RAG core. A
round-trip to an embedding API alone would blow that budget; a local
multilingual model embeds a short query in single-digit milliseconds and it
supports Hindi + 99 languages, which matters for the MSMARCO-XI dataset.
"""
from __future__ import annotations

import numpy as np


class EmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model_name = model_name
        self._model = None
        # e5-family models expect "query: " / "passage: " instruction prefixes.
        self._uses_prefix = "e5" in model_name.lower()

    def _ensure(self):
        if self._model is None:
            import os
            from fastembed import TextEmbedding
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            try:
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as e:
                print(f"[embeddings] Warning: failed to load {self.model_name} ({e}), falling back to dummy provider")
                self._model = DummyEmbeddingProvider()
        return self._model

    def _maybe_prefix(self, texts: list[str], is_query: bool) -> list[str]:
        if not self._uses_prefix:
            return texts
        inst = "query: " if is_query else "passage: "
        return [inst + t for t in texts]

    def embed(self, texts: list[str], batch_size: int = 32,
              is_query: bool = False) -> np.ndarray:
        """Return a (n, dim) float32 array (L2-normalized rows)."""
        model = self._ensure()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = list(model.embed(self._maybe_prefix(texts, is_query), batch_size=batch_size))
        arr = np.asarray([t.tolist() for t in out], dtype=np.float32)
        return _l2_normalize(arr)

    def embed_one(self, text: str, is_query: bool = False) -> np.ndarray:
        arr = self.embed([text], is_query=is_query)
        return arr[0]


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class DummyEmbeddingProvider:
    """Deterministic hashing fallback if the model cannot be downloaded.

    Not semantically useful, but keeps the pipeline alive and testable in
    air-gapped environments.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t[:self.dim]):
                out[i, j] = (ord(ch) % 251) / 251.0
        return _l2_normalize(out)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


_provider = None


def get_embedding_provider(model_name: str | None = None) -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = EmbeddingProvider(model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _provider
