"""Optional: build the index from the REAL MSMARCO-XI dataset stream.

The default demo corpus mirrors the dataset structure so the app works
instantly and offline. When you have reliable bandwidth (and optionally a
HF_TOKEN), run this to pull live MSMARCO-XI rows and rebuild the index:

    python -m backend.download

Set HHGOA_SAMPLE_ROWS (default 700) to control how many rows are streamed,
and HHGOA_DATASET_LANGUAGE (default hin) to pick the language.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HHGOA_USE_REAL", "1")

from .ingest import build_index, describe_index  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    print("[download] building index from live MSMARCO-XI stream ...")
    idx = build_index(force=True)
    print(describe_index(idx))
    print(f"[download] done in {time.perf_counter() - t0:.1f}s")
    print("[download] restart the server; the new index is now live.")


if __name__ == "__main__":
    main()