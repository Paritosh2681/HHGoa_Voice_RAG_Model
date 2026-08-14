"""Latency benchmark CLI.

Measures the pipeline across a batch of real queries and prints the
P50 / P70 / P100 table the brief demands — "measured across a reasonable
number of test queries, not a single best-case run."

Usage:
    python -m backend.benchmark --queries 30 --json results/latency.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from . import config as C
from .ingest import build_index
from .harness import PipelineHarness
from .latency import MetricPoint, MetricsStore
from .models import AskRequest

# Deterministic, language-diverse sample queries spanning MSMARCO query types.
SAMPLE_QUERIES = [
    "मेनहट्टन प्रोजेक्ट की सफलता का तात्कालिक प्रभाव क्या था?",
    "एक्वेरियम में मछलियों की देखभाल कैसे करें?",
    "What is the capital of Australia?",
    "प्रकाश संश्लेषण क्या है?",
    "What is the boiling point of water?",
    "हिंदी भाषा किस लिपि में लिखी जाती है?",
    "माउंट एवरेस्ट की ऊँचाई कितनी है?",
    "सौरमंडल का सबसे बड़ा ग्रह कौन सा है?",
    "हीमोग्लोबिन का कार्य क्या है?",
    "ओजोन परत क्यों महत्वपूर्ण है?",
    "What is RAM in a computer?",
    "गांधीजी का जन्म कब हुआ था?",
    "पानी का रासायनिक सूत्र क्या है?",
    "भारत का राष्ट्रीय पक्षी कौन सा है?",
    "What is the largest ocean in the world?",
    "टाइप 2 मधुमेह में कौन सा भोजन नहीं खाना चाहिए?",
    "What is the currency of Japan?",
    "कंप्यूटर का आविष्कार किसने किया?",
    "सूर्य का तापमान कितना होता है?",
    "What is an API?",
]


async def run_benchmark(n: int, json_out: str | None, quiet: bool = False):
    print("[bench] building/loading index ...")
    index = build_index(force=False)
    harness = PipelineHarness(index)
    mem = MetricsStore(window=n)

    # warm up the ONNX session so the reported numbers exclude cold-start
    await asyncio.to_thread(harness._emb.embed_one, "warmup")

    queries = (SAMPLE_QUERIES * ((n // len(SAMPLE_QUERIES)) + 1))[:n]

    results = []
    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        resp = await harness.run(AskRequest(text=q), record_metrics=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mem.record(MetricPoint(
            request_id=resp.request_id,
            total_ms=elapsed_ms,
            stages=resp.latency_ms,
            mode=resp.mode,
            grounded=resp.grounded,
            created_at=resp.created_at,
        ))
        results.append({"query": q, "total_ms": round(elapsed_ms, 2),
                        "mode": resp.mode, "grounded": resp.grounded,
                        "stages": resp.latency_ms})
        if not quiet:
            print(f"  [{i:2}/{n}] {elapsed_ms:7.1f} ms  {resp.mode:11} {q[:46]}")

    summary = mem.summary()

    print("\n=== HH GOA VOICE RAG — LATENCY BENCHMARK ===")
    print(f"total requests   : {summary.total_requests}")
    print(f"P50              : {summary.p50_ms} ms")
    print(f"P70              : {summary.p70_ms} ms")
    print(f"P100 (worst)     : {summary.p100_ms} ms")
    print(f"mean             : {summary.mean_ms} ms")
    print("\n--- per stage (P50 / P70 / P100) ---")
    for stage, s in summary.by_stage.items():
        print(f"  {stage:10} {s['p50_ms']:7.1f} / {s['p70_ms']:7.1f} / {s['p100_ms']:7.1f} ms  (n={s['n']})")

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary.model_dump(),
            "results": results,
            "config": {
                "embed_model": C.EMBED_MODEL,
                "llm_model": C.LLM_MODEL if C.LLM_API_KEY else None,
                "top_k": C.TOP_K,
            },
        }
        Path(json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[saved] {json_out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="HH GOA Voice RAG latency benchmark")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    asyncio.run(run_benchmark(args.queries, args.json, args.quiet))


if __name__ == "__main__":
    main()
