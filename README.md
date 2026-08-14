# HH GOA Voice RAG

Voice-enabled RAG pipeline for the **HH Goa 2026 shortlisting task 02**,
built on `ai4bharat/MSMARCO-XI`. Ask in Hindi or English — by voice or text —
and watch the retrieval wire light up under 200 ms.

Speech → STT (Sarvam `saaras:v3`, fallbacks: ElevenLabs / Groq Whisper) →
multi-strategy chunking → hybrid retrieval (FAISS cosine × BM25 × RRF) →
grounded answer generation (LLM via Groq, extractive fallback).

## Features

- **Voice-first demo** — hold-to-talk in the browser (MediaRecorder → WebM), SSE streamed back live.
- **Four chunking strategies** — `fixed_overlap`, `sentence`, `semantic`, `metadata` — fused at retrieval instead of betting on one naive split.
- **Hybrid retrieval** — FAISS (local ONNX embeddings, L2-normalized) fused with BM25 via Reciprocal Rank Fusion, then metadata re-rank. Zero external vector DB.
- **Guardrails** — unsafe-language (EN+HI), prompt-injection, off-topic gate (cosine ∨ lexical overlap), hallucination/grounding verification, graceful extractive fallback.
- **Latency analytics** — P50 / P70 / P100 rolling percentiles per stage, surfaced in the UI.
- **Structured harness** — Pydantic I/O, per-stage tracing, exponential-backoff retries, SSE event protocol.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

python -m backend.main            # -> http://localhost:8000
```

No API keys needed: the demo index boots from a curated MSMARCO-XI mirror
corpus (~194 chunks) in under a second. To use the real dataset:

```bash
set HHGOA_USE_REAL=1
python -m backend.ingest --force   # rebuild index from the HF stream
```

## Enabling STT + LLM

Copy `.env.example` to `.env` and fill keys (also loadable as env vars):

| Variable            | Purpose                                     |
| ------------------- | ------------------------------------------- |
| `HHGOA_STT_PROVIDER`| `sarvam` (default), `elevenlabs`, or `groq` |
| `SARVAM_API_KEY`    | Sarvam `saaras:v3` (voice → Hindi/English)  |
| `ELEVENLABS_API_KEY`| ElevenLabs `scribe_v2`                       |
| `GROQ_API_KEY`      | enables LLM generation + Groq Whisper STT    |

Without an LLM key the harness answers **extractively** (top passage) — the
latency numbers below are measured in that mode, which is the strictest test
of the retrieval core.

## API

| Route                | Method | Purpose                                  |
| -------------------- | ------ | ---------------------------------------- |
| `/`                  | GET    | Frontend                                 |
| `/api/health`        | GET    | Liveness + provider status               |
| `/api/index-info`    | GET    | Index & chunking inventory               |
| `/api/ask`           | POST   | JSON answer                              |
| `/api/ask/stream`    | POST   | SSE answer (token stream + stage events) |
| `/api/stt`           | POST   | audio → transcript                       |
| `/api/voice/stream`  | POST   | audio → STT → RAG → SSE (end-to-end)     |
| `/api/metrics`       | GET    | P50/P70/P100 summary                     |
| `/api/metrics/reset` | POST   | clear rolling window                     |

SSE events: `stage`, `guard_result`, `sources`, `answer_start`, `chunk`,
`refuse`, `fallback`, `stt`, `done`, `error`.

## Latency (measured, not a single best case)

`python -m backend.benchmark --queries 20 --json results/latency.json`

| Percentile | Time      |
| ---------- | --------- |
| P50        | 57.4 ms   |
| P70        | 72.5 ms   |
| P100       | 124.1 ms  |

Per-stage P50: guard 0.2 · embed 19.7 · retrieve 2.0 · gate 0.2 ·
generate ~0 (extractive) · verify 34.4 ms. A live P50/P70/P100 readout is
baked into the latency section of the site.

## Project layout

```
backend/
  config.py       env-overridable config
  chunking.py     four chunking strategies
  embeddings.py   fastembed (ONNX) provider, L2-normalized
  index_store.py  HybridIndex: FAISS + BM25 + RRF + metadata re-rank
  ingest.py       corpus → chunks → vectors; cache to data/index
  knowledge.py    curated MSMARCO-XI mirror corpus
  stt.py          Sarvam / ElevenLabs / Groq providers
  guardrails.py   safety, injection, off-topic, grounding checks
  harness.py      orchestration, retries, SSE protocol
  latency.py      MetricsStore (P50/P70/P100)
  benchmark.py    latency benchmark CLI
  download.py     optional real-dataset builder
  main.py         FastAPI app
frontend/         index.html · styles.css · app.js (HH GOA design)
assets/           HH GOA brand assets (© HH Goa / 2:47 PM Studio)
data/index/       cached chunks + vectors
```

## Submission notes

- Videos for Instagram / X / LinkedIn with **#RAGInGoa**.
- Google Form: https://forms.gle/MNvCjcv23Hn2Eeu58
- HH GOA brand assets are © HH Goa / 2:47 PM Studio — used for this submission demo only.