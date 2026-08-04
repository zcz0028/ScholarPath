from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.retrieval.base import SearchRequest
from scholarpath.retrieval.openalex import OpenAlexConfig, OpenAlexError, OpenAlexRetriever


def main() -> int:
    if not os.getenv("OPENALEX_API_KEY"):
        print("[ERROR] OPENALEX_API_KEY is not set in this terminal.")
        return 2

    retriever = OpenAlexRetriever(
        OpenAlexConfig(cache_dir="data/cache/openalex_smoke")
    )
    try:
        result = retriever.search(
            SearchRequest(
                query="scientific paper retrieval",
                per_page=3,
                to_publication_date="2024-09-24",
            ),
            refresh_cache=True,
        )
    except OpenAlexError as exc:
        print(f"[ERROR] OpenAlex check failed: {exc}")
        return 2

    print(f"[OK] HTTP status: {result.stats.http_status}")
    print(f"[OK] API calls: {result.stats.actual_api_calls}")
    print(f"[OK] Papers returned: {len(result.papers)}")
    print(
        "[OK] Request latency: "
        f"{result.stats.request_latency_ms:.2f} ms"
    )
    for index, paper in enumerate(result.papers, start=1):
        print(f"  {index}. {paper.title} ({paper.year})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
