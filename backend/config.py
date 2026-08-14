"""Global configuration — everything overridable via environment variables.

Design notes
------------
- Every external integration (STT, LLM) is optional. The pipeline degrades
  gracefully: no STT key -> text input still works; no LLM key -> the harness
  falls back to a fast extractive answer from retrieved passages.
- Paths are resolved relative to the project root.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("HHGOA_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.getenv("HHGOA_INDEX_DIR", ROOT / "data" / "index"))

# --- Dataset ------------------------------------------------------------------
DATASET_ID = os.getenv("HHGOA_DATASET_ID", "ai4bharat/MSMARCO-XI")
DATASET_CONFIG = os.getenv("HHGOA_DATASET_CONFIG", "default")  # HF auto-parquet "default"
DATASET_SPLIT = os.getenv("HHGOA_DATASET_SPLIT", "train")
DATASET_LANGUAGE = os.getenv("HHGOA_DATASET_LANGUAGE", "hin")   # filter by target_lang prefix, e.g. "hin"
# Set HHGOA_USE_REAL=1 to build the index from the real MSMARCO-XI stream
# (network + time heavy). Default 0 builds from the curated mirror corpus so
# the demo boots instantly and works offline.
USE_REAL_DATASET = os.getenv("HHGOA_USE_REAL", "0") == "1"
SAMPLE_ROWS = int(os.getenv("HHGOA_SAMPLE_ROWS", "700"))   # rows sampled from HF
MAX_PASSAGES_PER_ROW = int(os.getenv("HHGOA_MAX_PASSAGES", "8"))
EMBED_MODEL = os.getenv("HHGOA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- Chunking ---------------------------------------------------------------
FIXED_CHUNK_SIZE = int(os.getenv("HHGOA_CHUNK_SIZE", "320"))
FIXED_CHUNK_OVERLAP = int(os.getenv("HHGOA_CHUNK_OVERLAP", "64"))
SENTENCE_WINDOW = int(os.getenv("HHGOA_SENTENCE_WINDOW", "4"))
SEMANTIC_SIM_THRESHOLD = float(os.getenv("HHGOA_SEMANTIC_THRESHOLD", "0.42"))

# --- Retrieval ---------------------------------------------------------------
TOP_K = int(os.getenv("HHGOA_TOP_K", "6"))
RERANK_TOP = int(os.getenv("HHGOA_RERANK_TOP", "4"))
BM25_TOP = int(os.getenv("HHGOA_BM25_TOP", "20"))
DENSE_TOP = int(os.getenv("HHGOA_DENSE_TOP", "20"))
MIN_SCORE = float(os.getenv("HHGOA_MIN_SCORE", "0.5"))       # blended gate (cosine ∨ lexical): below → refuse
SELECTED_BOOST = float(os.getenv("HHGOA_SELECTED_BOOST", "0.55"))

# --- STT ---------------------------------------------------------------------
# Provider: sarvam | elevenlabs | groq | local
STT_PROVIDER = os.getenv("HHGOA_STT_PROVIDER", "sarvam").lower()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL = os.getenv("HHGOA_SARVAM_MODEL", "saaras:v3")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL = os.getenv("HHGOA_ELEVENLABS_MODEL", "scribe_v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAX_AUDIO_BYTES = int(os.getenv("HHGOA_MAX_AUDIO_BYTES", str(12 * 1024 * 1024)))

# --- LLM / generation ---------------------------------------------------------
LLM_BASE_URL = os.getenv("HHGOA_LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("HHGOA_LLM_MODEL", "openai/gpt-oss-20b")
LLM_MAX_TOKENS = int(os.getenv("HHGOA_LLM_MAX_TOKENS", "240"))
LLM_TEMPERATURE = float(os.getenv("HHGOA_LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = float(os.getenv("HHGOA_LLM_TIMEOUT", "8.0"))
MAX_LLM_RETRIES = int(os.getenv("HHGOA_MAX_LLM_RETRIES", "2"))

# --- Harness ------------------------------------------------------------------
STAGE_TIMEOUT = float(os.getenv("HHGOA_STAGE_TIMEOUT", "10.0"))
LATENCY_WINDOW = int(os.getenv("HHGOA_LATENCY_WINDOW", "500"))  # rolling samples kept

# --- App ----------------------------------------------------------------------
HOST = os.getenv("HHGOA_HOST", "0.0.0.0")
PORT = int(os.getenv("HHGOA_PORT", "8000"))

for _d in (DATA_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)
