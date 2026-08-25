"""Embedder adapter for evaluation suite (app.embedder)."""
from backend.embeddings import get_embedding_provider

_provider = get_embedding_provider()


def embed(texts: list[str]):
    """Embed a list of texts. Returns numpy array (n, dim)."""
    return _provider.embed(texts)


def embed_one(text: str):
    """Embed a single text. Returns numpy array (dim,)."""
    return _provider.embed_one(text)


def get_model():
    """Trigger model loading (side effect only)."""
    return _provider
