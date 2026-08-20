"""Structured I/O contracts for the pipeline.

The harness accepts and returns validated models — the whole point of
"structured input/output handling" — instead of raw string soup.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SttResult(BaseModel):
    transcript: str
    provider: str
    language_code: Optional[str] = None
    language_probability: Optional[float] = None
    audio_duration_secs: Optional[float] = None
    latency_ms: float = 0.0


class Chunk:
    __slots__ = ("id", "text", "strategy", "meta")

    def __init__(self, id: str = "", text: str = "", strategy: str = "unknown", meta: Optional[dict[str, Any]] = None) -> None:
        self.id = id
        self.text = text
        self.strategy = strategy
        self.meta = meta if meta is not None else {}

    def model_dump(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "strategy": self.strategy, "meta": self.meta}

    @classmethod
    def model_construct(cls, id: str = "", text: str = "", strategy: str = "unknown", meta: Optional[dict[str, Any]] = None, **kwargs) -> "Chunk":
        return cls(id=id, text=text, strategy=strategy, meta=meta)


class RetrievedDoc(BaseModel):
    id: str
    text: str
    strategy: str
    score: float
    source: str = "passage"
    language: str = "hi"
    is_selected: bool = False
    query_type: Optional[str] = None


class GuardResult(BaseModel):
    input_ok: bool = True
    query: str = ""
    normalized: str = ""
    reasons: list[str] = Field(default_factory=list)
    reject_code: Optional[str] = None
    latency_ms: float = 0.0


class AskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    language: Optional[str] = None
    strategy: Optional[str] = None
    request_id: Optional[str] = None


class AskResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    mode: Literal["llm", "extractive", "refused", "quantum_cache", "lex_fast"]
    grounded: bool
    guardrails: GuardResult
    sources: list[RetrievedDoc] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    total_ms: float = 0.0
    pipeline: list[str] = Field(default_factory=list)
    created_at: str = ""
    additional_answer: Optional[str] = None   # LLM's own knowledge when RAG refuses
    from_corpus: bool = True                  # False when answer is from LLM, not corpus


class MetricPoint(BaseModel):
    request_id: str
    total_ms: float
    stages: dict[str, float] = Field(default_factory=dict)
    mode: str = ""
    grounded: bool = True
    created_at: str = ""


class LatencySummary(BaseModel):
    total_requests: int
    p50_ms: float
    p70_ms: float
    p100_ms: float
    mean_ms: float
    by_stage: dict[str, dict[str, float]] = Field(default_factory=dict)
    mode_counts: dict[str, int] = Field(default_factory=dict)
    recent: list[MetricPoint] = Field(default_factory=list)
    target_ms: float = 200.0
    under_target: Optional[float] = None


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
