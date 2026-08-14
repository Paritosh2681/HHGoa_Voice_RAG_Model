"""Ingestion: stream MSMARCO-XI -> build hybrid index.

Streaming (not a full 55 GB download): we iterate the HuggingFace dataset with
`streaming=True` and keep `SAMPLE_ROWS` rows, `MAX_PASSAGES_PER_ROW` passages
each. Every passage is chunked with ALL four strategies (see `chunking.py`),
embedded with the local multilingual model, and indexed by `HybridIndex`.

If the network is unavailable or the dataset cannot be reached, we fall back to
the curated `knowledge.FALLBACK_CORPUS` so the pipeline is always usable.
"""
from __future__ import annotations

import time

from . import config as C
from .chunking import chunk_all
from .embeddings import get_embedding_provider
from .index_store import HybridIndex
from .knowledge import FALLBACK_CORPUS

DATASET_ROW_MARKER = "MSMARCO-XI"
FALLBACK_ROW_MARKER = "FALLBACK"


def _row_to_passages(row: dict) -> tuple[str, str, str, list[tuple[int, str, bool]], dict]:
    """Flatten a dataset row into (query, answer, qtype, passages, meta)."""
    q = (row.get("query") or "").strip()
    ans = (row.get("Answer") or "").strip()
    qtype = (row.get("query_type") or "DESCRIPTION")
    lang = str(row.get("target_lang") or row.get("source_lang") or "hin_Deva")
    passages = row.get("passages") or {}
    if isinstance(passages, list):
        # list-of-dicts: [{"translated_passage":..., "english_passage":..., "is_selected":...}]
        out = []
        for i, p in enumerate(passages[: C.MAX_PASSAGES_PER_ROW]):
            if isinstance(p, dict):
                text = p.get("translated_passage") or p.get("Translated_passages") or p.get("text") or p.get("passage") or ""
                if not text:
                    text = p.get("english_passage") or p.get("English_passages") or ""
                sel = bool(p.get("is_selected", 0) == 1)
            else:
                text, sel = str(p), False
            text = str(text).strip()
            if text:
                out.append((i, text, sel))
        return q, ans, qtype, out, {"language": lang}
    sel = passages.get("is_selected") or []
    translated = passages.get("Translated_passages") or []
    english = passages.get("English_passages") or []
    out = []
    for i, p in enumerate(translated[: C.MAX_PASSAGES_PER_ROW]):
        if not p or not str(p).strip():
            continue
        is_sel = bool(i < len(sel) and sel[i] == 1)
        out.append((i, str(p).strip(), is_sel))
    if not out:
        for i, p in enumerate(english[: C.MAX_PASSAGES_PER_ROW]):
            if not p or not str(p).strip():
                continue
            is_sel = bool(i < len(sel) and sel[i] == 1)
            out.append((i, str(p).strip(), is_sel))
    return q, ans, qtype, out, {"language": lang}


def _stream_dataset() -> list[dict]:
    """Prefer bounded HTTP-range parquet reads (small download), fall back to
    streaming the auto-converted dataset, then to the curated corpus."""
    rows = _range_parquet_rows()
    if rows:
        return rows
    from datasets import load_dataset
    try:
        ds = load_dataset(
            C.DATASET_ID, C.DATASET_CONFIG, split=C.DATASET_SPLIT,
            streaming=True,
        )
    except ValueError:
        # some revisions only expose the auto-parquet "default" config
        ds = load_dataset(
            C.DATASET_ID, "default", split=C.DATASET_SPLIT, streaming=True,
        )
    rows = []
    for row in ds:
        if C.DATASET_LANGUAGE:
            lang = str(row.get("target_lang") or row.get("source_lang") or "")
            if not lang.lower().startswith(C.DATASET_LANGUAGE.lower()):
                continue
        q, ans, qtype, passages, meta = _row_to_passages(row)
        if not q or not passages:
            continue
        rows.append({
            "query": q, "Answer": ans, "query_type": qtype,
            "passages": passages, "meta": meta, "marker": DATASET_ROW_MARKER,
        })
        if len(rows) >= C.SAMPLE_ROWS:
            break
    return rows


def _range_parquet_rows() -> list[dict]:
    """Read only the first few row groups of a per-language parquet via HTTP
    Range requests, so a 3.5 GB file costs only tens of MB of download."""
    import io

    import httpx
    import pyarrow.parquet as pq

    if not C.DATASET_LANGUAGE:
        return []
    lang = C.DATASET_LANGUAGE.lower()
    url = f"https://huggingface.co/datasets/{C.DATASET_ID}/resolve/main/train/{lang}train.parquet"

    class RangeReader(io.RawIOBase):
        def __init__(self, url_: str):
            self._url = url_
            self._client = httpx.Client(timeout=60, follow_redirects=True)
            self._pos = 0
            self._size = self._head_size()

        def _head_size(self) -> int:
            r = self._client.head(self._url)
            r.raise_for_status()
            return int(r.headers.get("content-length", "0"))

        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return True

        def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
            if whence == io.SEEK_SET:
                self._pos = offset
            elif whence == io.SEEK_CUR:
                self._pos += offset
            elif whence == io.SEEK_END:
                self._pos = self._size + offset
            return self._pos

        def tell(self) -> int:
            return self._pos

        def read(self, n: int = -1) -> bytes:
            if n is None or n < 0:
                n = self._size - self._pos
            end = self._pos + n - 1
            headers = {"Range": f"bytes={self._pos}-{end}"}
            r = self._client.get(self._url, headers=headers)
            if r.status_code == 416:
                return b""
            r.raise_for_status()
            self._pos += len(r.content)
            return r.content

    try:
        wrapper = RangeReader(url)
        pf = pq.ParquetFile(wrapper)
        n_groups = pf.num_row_groups
        rows: list[dict] = []
        for gi in range(min(3, n_groups)):
            table = pf.read_row_group(gi, columns=["query", "Answer", "query_type",
                                                    "target_lang", "source_lang", "passages"])
            for row in table.to_pylist():
                if len(rows) >= C.SAMPLE_ROWS:
                    break
                lang_field = str(row.get("target_lang") or row.get("source_lang") or "")
                if not lang_field.lower().startswith(lang):
                    continue
                q, ans, qtype, passages, meta = _row_to_passages(row)
                if not q or not passages:
                    continue
                rows.append({
                    "query": q, "Answer": ans, "query_type": qtype,
                    "passages": passages, "meta": meta, "marker": DATASET_ROW_MARKER,
                })
            if len(rows) >= C.SAMPLE_ROWS:
                break
        wrapper._client.close()
        return rows
    except Exception as exc:
        print(f"[ingest] range parquet read failed ({exc!r})")
        return []


def _fallback_rows() -> list[dict]:
    rows = []
    for ex in FALLBACK_CORPUS:
        passages = [(i, p, sel) for i, (p, sel) in enumerate(ex["passages"])]
        rows.append({
            "query": ex["query"], "Answer": ex["Answer"],
            "query_type": ex["query_type"], "passages": passages,
            "meta": {"language": "hin_Deva"}, "marker": FALLBACK_ROW_MARKER,
        })
    return rows


def build_index(force: bool = False) -> HybridIndex:
    """Build (or reload cached) hybrid index."""
    if not force:
        cached = HybridIndex.load()
        if cached is not None and len(cached) > 0:
            return cached

    t0 = time.perf_counter()
    rows = None
    source = None
    if C.USE_REAL_DATASET:
        try:
            rows = _stream_dataset()
            source = DATASET_ROW_MARKER
        except Exception as exc:  # network / HF down
            print(f"[ingest] real dataset stream failed ({exc!r}) — using curated corpus")
            rows = _fallback_rows()
            source = FALLBACK_ROW_MARKER
    else:
        rows = _fallback_rows()
        source = FALLBACK_ROW_MARKER
        print("[ingest] using curated corpus (set HHGOA_USE_REAL=1 for the live MSMARCO-XI stream)")

    # answer pass-through: keep gold answers as "answer" chunks too
    chunks = []
    for r in rows:
        meta = {
            "query": r["query"], "query_type": r["query_type"],
            "language": r["meta"]["language"], "source": "passage",
            "query_id": r.get("query_id", ""), "marker": source,
        }
        chunks.extend(chunk_all(r["passages"], meta))
        if r.get("Answer"):
            chunks.append(_answer_chunk(r, source))

    # dedupe
    seen = set()
    uniq = []
    for c in chunks:
        key = c.text[:120]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    chunks = uniq

    # embed (batch)
    provider = get_embedding_provider(C.EMBED_MODEL)
    texts = [c.text for c in chunks]
    vectors = provider.embed(texts, batch_size=48)

    idx = HybridIndex()
    idx.build(chunks, vectors, int(vectors.shape[1]) if vectors.ndim else 0)
    idx.save()
    dt = time.perf_counter() - t0
    print(f"[ingest] source={source} chunks={len(chunks)} dims={idx._dims} took={dt:.1f}s")
    return idx


def _answer_chunk(r: dict, source: str):
    from .models import Chunk
    import uuid
    return Chunk(
        id=str(uuid.uuid4()),
        text=r["Answer"],
        strategy="answer",
        meta={
            "query": r["query"], "query_type": r["query_type"],
            "language": r["meta"]["language"], "source": "answer",
            "marker": source,
        },
    )


def describe_index(idx: HybridIndex) -> dict:
    from collections import Counter
    strat = Counter(c.strategy for c in idx.chunks)
    markers = Counter(c.meta.get("marker", "?") for c in idx.chunks)
    return {
        "n_chunks": len(idx.chunks),
        "strategies": dict(strat),
        "markers": dict(markers),
        "dims": idx._dims,
        "bm25": idx._bm25 is not None,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build/rebuild the HH GOA Voice RAG index")
    ap.add_argument("--force", action="store_true",
                    help="rebuild from scratch even if a cached index exists")
    args = ap.parse_args()
    idx = build_index(force=args.force)
    print(describe_index(idx))


if __name__ == "__main__":
    main()
